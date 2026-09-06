"""Run the exact real "remember my face" enrolment, without the spoken trigger.

    .venv\\Scripts\\python.exe visual_guidance_assistant\\self_enrol_test.py

Why this exists: DialogueProcessor._handle_remember_face() does exactly one
thing once the trigger phrase is heard - it calls
face_trainer.capture_and_train(frame, camera=cam). This script builds the same
real camera, the same real live profile store, and the same real FaceTrainer,
and calls that identical method directly. Everything downstream is completely
real: the tkinter popup you type your name into, the real spoken voice prompt,
your real microphone recording your voice, and the real encrypted save to
known_faces/profiles.json.

The only thing skipped is Google speech-to-text recognising the words
"remember my face" out loud, which is a separate, already-evidenced problem
(garbled transcriptions in real testing), not what this script exists to
verify. If you want to test the spoken trigger itself, use the running app.
"""
import sys
import time

sys.path.insert(0, r"C:\Users\Acer\Downloads\Guide YOU\Guide YOU\visual_guidance_assistant")

from camera.capture import CameraCapture
from config.config_loader import load_config
from profiles.profile_store import ProfileStore
from recognition.face_recognizer import FaceRecognizer
from recognition.face_trainer import FaceTrainer
from utils.paths import CONFIG_FILE

import queue
from speech.synthesizer import SpeechSynthesizer
from voice_input.microphone import MicrophoneListener


def main():
    cfg = load_config(CONFIG_FILE)

    print("=" * 70)
    print("  REAL self-enrolment, real live profile store, no voice trigger")
    print("=" * 70)

    store = ProfileStore()          # the REAL live store — this IS persistence
    print(f"  profiles before: {store.names()}")

    cam = CameraCapture(cfg, None)
    cam.start()
    time.sleep(2.0)

    speech_queue = queue.Queue(maxsize=10)
    command_queue = queue.Queue(maxsize=10)
    mic = MicrophoneListener(command_queue, cfg)
    mic.start()
    speech = SpeechSynthesizer(speech_queue, cfg, mic_listener=mic)
    speech.start()

    recog = FaceRecognizer(store, detector=(cfg.get("face_recognition") or {}).get("detector", "yunet"))
    trainer = FaceTrainer(recog, store, speech_queue=speech_queue, mic_listener=mic)

    frame = cam.get_latest_frame()
    print(f"  captured frame: {frame.shape[1]}x{frame.shape[0]}" if frame is not None else "  NO FRAME")

    print("\n  A popup will appear for your name. Then a voice prompt will play")
    print("  and record you. This is the identical real flow 'remember my")
    print("  face' triggers — just without needing to say the trigger.\n")

    ok = trainer.capture_and_train(frame, camera=cam)

    time.sleep(1.0)   # let any final TTS finish
    mic.stop()
    speech.stop()
    cam.stop()

    print("\n" + "=" * 70)
    print(f"  RESULT: {ok}")
    print("=" * 70)
    fresh = ProfileStore()   # separate instance — proves it round-trips through disk
    for p in fresh.all_profiles():
        print(f"  - {p['name']!r}  role={p['role']!r}  "
              f"face_samples={len(p.get('face_encodings') or [])}  "
              f"voice={'yes' if p.get('voice_embedding') else 'no'}")


if __name__ == "__main__":
    main()
