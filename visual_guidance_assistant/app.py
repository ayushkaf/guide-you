"""Main entry point for the vision-first Visual Guidance Assistant."""
from __future__ import annotations

import os
import time
import queue
import signal
import sys
import cv2
import threading

# Windows consoles default to cp1252, which can't encode emoji or many
# Unicode characters â€” Claude's replies can include them, and an unguarded
# print() would raise UnicodeEncodeError before the reply reaches speech_queue.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config.config_loader import load_config
from utils.env_file import load_env_file
from utils.logger import setup_logger
from utils.fps_counter import FPSCounter
from utils.latest import LatestValue
from utils.paths import CONFIG_FILE, KNOWN_FACES_DIR
from camera.capture import CameraCapture
from speech.synthesizer import SpeechSynthesizer
from speech.azure_tts_client import AzureTTSClient
from voice_input.microphone import MicrophoneListener
from dialogue.dialogue_processor import DialogueProcessor
from sound.sound_detector import SoundDetector
from memory.memory_store import MemoryStore
from care.routines import RoutineChecker, RoutineStore
from memory.alarm_checker import AlarmChecker
from memory.memory_processor import MemoryProcessor
from memory.morning_routine import MorningRoutine
from memory.layered_memory import LayeredMemory
from profiles.profile_store import ProfileStore
from recognition.azure_face_client import AzureFaceClient, DEFAULT_PERSON_GROUP_ID
from recognition.face_recognizer import FaceRecognizer, safe_filename
from recognition.identity_thread import IdentityProcessor
from detection.detection_processor import DetectionProcessor

LOGGER = setup_logger("vga.app")


def _migrate_profiles_to_azure(azure_face_client: AzureFaceClient, profile_store: ProfileStore) -> None:
    """One-time-per-startup sweep: push any already-enrolled local profile
    that Azure doesn't know about yet into its PersonGroup, using the same
    reference photo every local enrolment already saves. Never forces
    anyone to re-enrol — a profile with no reference photo on disk (should
    not happen for anything enrolled through this app, but is possible for
    very old migrated records) is skipped, not treated as an error.

    Idempotent and safe to run every startup: a profile that already carries
    an azure_person_id (from a previous run, or from being enrolled fresh
    after Azure was already configured — see FaceTrainer._mirror_to_azure)
    is left alone.
    """
    azure_face_client.load_person_id_map(profile_store.active_profiles())
    migrated = 0
    skipped_no_photo = 0
    for profile in profile_store.active_profiles():
        if profile.get("azure_person_id"):
            continue
        photo_path = os.path.join(KNOWN_FACES_DIR, f"{safe_filename(profile['name'])}.jpg")
        if not os.path.exists(photo_path):
            skipped_no_photo += 1
            continue
        person_id = azure_face_client.sync_profile(profile["name"], photo_path)
        if person_id:
            profile_store.set_azure_person_id(profile["name"], person_id)
            migrated += 1
    if migrated or skipped_no_photo:
        print(f"[azure-face] startup migration: {migrated} profile(s) synced to Azure, "
              f"{skipped_no_photo} skipped (no reference photo on disk)")


def main() -> int:
    # Read .env before anything below touches os.environ for a key — normally
    # already done as a side effect of importing dialogue_processor (which
    # imports llm_connector, which loads it at module level), but calling it
    # explicitly here means this does not silently depend on that import
    # order. Safe to call more than once — see utils/env_file.py. verbose is
    # off here specifically because the module-level call above already
    # printed exactly what was loaded; a second identical print would be
    # confusing rather than informative.
    load_env_file(verbose=False)

    # Absolute path: the app must behave the same regardless of the directory
    # it was launched from.
    cfg = load_config(CONFIG_FILE)

    frame_queue = queue.Queue(maxsize=2)
    speech_queue = queue.Queue(maxsize=20)
    sound_queue = queue.Queue(maxsize=5)
    command_queue = queue.Queue(maxsize=10)

    # Not a queue: the render loop and the dialogue thread both need the newest
    # detections, and a queue would let each steal items from the other.
    detection_store = LatestValue([])

    conversation_history = []
    conversation_lock = threading.Lock()

    layered_memory = LayeredMemory()
    # ONE MemoryStore for the whole process, injected everywhere. Separate
    # instances each hold their own copy of the list and overwrite the file on
    # save, which is what used to make alarms never fire and reminders vanish.
    memory_store = MemoryStore()
    memory_processor = MemoryProcessor(memory_store)
    # One ProfileStore for the process, same injection rule as MemoryStore:
    # it rewrites the whole file on save, so two instances would clobber
    # each other's enrolments.
    profile_store = ProfileStore()
    print(f"[app] enrolled people: {profile_store.names() or 'none yet'}")
    face_cfg = cfg.get("face_recognition") or {}
    identity_interval = float(face_cfg.get("interval_seconds", 2.5))
    identity_threaded = bool(face_cfg.get("threaded", True))

    # Azure Face is an optional ALTERNATIVE recognition backend, never a
    # requirement — AzureFaceClient itself never raises, and reports
    # "not usable" the same way whether AZURE_FACE_KEY is blank, invalid, or
    # unreachable. See recognition/azure_face_client.py's module docstring.
    azure_face_client = AzureFaceClient(
        key=os.environ.get("AZURE_FACE_KEY"),
        endpoint=os.environ.get("AZURE_FACE_ENDPOINT"),
        person_group_id=face_cfg.get("azure_person_group_id", DEFAULT_PERSON_GROUP_ID),
    )
    if azure_face_client.ensure_ready():
        _migrate_profiles_to_azure(azure_face_client, profile_store)
    else:
        print("[face] Azure not configured, using local YuNet+dlib")

    face_recognizer = FaceRecognizer(
        profile_store,
        layered_memory,
        sticky_seconds=float(face_cfg.get("sticky_seconds", 60.0)),
        detector=face_cfg.get("detector", "yunet"),
        min_interval=identity_interval,
        azure_face_client=azure_face_client,
    )

    cam = CameraCapture(cfg, frame_queue)
    cam.start()

    # dlib never runs on the render thread. With identity_threaded it does not
    # run on the detection thread either â€” DetectionProcessor only reads the
    # cache IdentityProcessor keeps warm, which costs microseconds.
    detection_processor = DetectionProcessor(
        frame_queue,
        detection_store,
        cfg,
        face_recognizer=face_recognizer,
        identity_is_threaded=identity_threaded,
    )
    detection_processor.start()

    identity_processor = None
    if identity_threaded:
        identity_processor = IdentityProcessor(
            cam,
            detection_store,
            face_recognizer,
            interval=identity_interval,
            low_priority=True,
        )
        identity_processor.start()

    # Both audio consumers are constructed (not yet started) before
    # SpeechSynthesizer so they can be passed in â€” it pauses and resumes both
    # around TTS playback so the assistant doesn't hear, answer, or
    # misclassify its own voice.
    microphone = MicrophoneListener(command_queue, cfg)
    sound_detector = SoundDetector(sound_queue, cfg)

    # Azure Neural TTS is an optional ALTERNATIVE playback backend for
    # SpeechSynthesizer._speak_one, never a requirement — AzureTTSClient
    # never raises, and reports "not usable" the same way whether
    # AZURE_SPEECH_KEY is blank, invalid, or unreachable. Voice identification
    # stays local MFCC unconditionally (see voice_id/voice_embedding.py) —
    # Azure's Speaker Recognition API was retired by Microsoft in 2025, so
    # there is no cloud alternative to fall forward to for that one.
    speech_cfg = cfg.get("speech") or {}
    azure_tts_client = AzureTTSClient(
        key=os.environ.get("AZURE_SPEECH_KEY"),
        region=os.environ.get("AZURE_SPEECH_REGION"),
        voice=speech_cfg.get("azure_voice", "en-US-JennyNeural"),
    )
    if not azure_tts_client.ensure_ready():
        print("[tts] Azure not configured, using local pyttsx3")
    print("[voice_id] Azure Speaker Recognition was retired by Microsoft "
          "(2025) — using local MFCC voice fingerprint (unconditionally)")

    print(f"[services] face={azure_face_client.health.mode()}, "
          f"voice_id=local, tts={azure_tts_client.health.mode()}")

    speech_synth = SpeechSynthesizer(
        speech_queue,
        cfg,
        mic_listener=microphone,
        sound_detector=sound_detector,
        azure_tts=azure_tts_client,
    )
    speech_synth.start()

    sound_detector.start()

    routine_store = RoutineStore()

    alarm_checker = AlarmChecker(memory_store, speech_queue)
    alarm_checker.start()
    morning_routine = MorningRoutine(memory_store, speech_queue)
    morning_routine.start()

    microphone.start()

    dialogue = DialogueProcessor(
        command_queue,
        speech_queue,
        cfg,
        conversation_history,
        conversation_lock,
        memory_processor,
        layered_memory,
        face_recognizer,
        cam,
        memory_store,
        profile_store,
        detection_store,
        sound_queue,
        mic_listener=microphone,  # lets DialogueProcessor pause/resume the mic during enrolment
        routine_store=routine_store,
    )
    dialogue.start()

    # Started after the dialogue processor because it asks it who is present
    # and whether a conversation is going on before it will speak.
    routine_checker = RoutineChecker(
        routine_store,
        speech_queue,
        patient_present=dialogue.patient_present,
        is_busy=dialogue.is_busy,
        caregiver_names=dialogue.caregiver_names,
    )
    routine_checker.start()

    LOGGER.info("Warming up camera...")
    time.sleep(1.0)

    fps = FPSCounter()
    window_name = "Visual Guidance Assistant"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    running = True

    def _signal_handler(signum, frame):
        nonlocal running
        LOGGER.info("Received signal %s, stopping...", signum)
        running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    latest_detections = []

    def _zone_label(center_x: int, width: int) -> str:
        if width <= 0:
            return "center"
        if center_x < width * 0.33:
            return "left"
        if center_x > width * 0.66:
            return "right"
        return "center"

    def _draw_detections(frame, detections):
        for detection in detections or []:
            # Show the recognized person's name when we have one.
            name = detection.get("name")
            class_name = detection.get("class_name", "object")
            if name and name != "unknown":
                class_name = name
            confidence = detection.get("confidence", 0.0)
            bbox = detection.get("bbox", (0, 0, 0, 0))
            center = detection.get("center", (0, 0))
            label = f"{class_name} {confidence:.2f}"
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(frame, label, (x1, max(y1 - 12, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            zone = _zone_label(center[0], frame.shape[1])
            cv2.putText(frame, f"{zone}", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    try:
        while running:
            frame = cam.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Non-consuming read â€” the dialogue thread sees the same detections.
            # They already carry face identities (DetectionProcessor does that
            # on its own thread), so no dlib work happens on this loop.
            latest_detections = detection_store.get(default=[])

            fps.tick()
            if cfg.get("ui", {}).get("show_fps", True):
                cv2.putText(frame, f"FPS: {fps.fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            _draw_detections(frame, latest_detections)

            with conversation_lock:
                exchanges = conversation_history[-3:]

            fh = frame.shape[0]
            for i, exch in enumerate(reversed(exchanges)):
                y = fh - 20 - (i * 32)
                if 'user' in exch:
                    cv2.putText(frame, f"You: {exch['user']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                if 'ai' in exch:
                    cv2.putText(frame, f"AI: {exch['ai']}", (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                LOGGER.info("'q' pressed, exiting...")
                break

    finally:
        LOGGER.info("Shutting down...")
        for obj in [microphone, dialogue, routine_checker, identity_processor, detection_processor, alarm_checker, morning_routine, sound_detector, speech_synth, cam]:
            if obj is None:
                continue
            try:
                obj.stop()
            except Exception:
                pass
        for obj in [microphone, dialogue, routine_checker, identity_processor, detection_processor, alarm_checker, morning_routine, sound_detector, speech_synth, cam]:
            if obj is None:
                continue
            try:
                obj.join(timeout=2.0)
            except Exception:
                pass
        cv2.destroyAllWindows()
        LOGGER.info("Shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())