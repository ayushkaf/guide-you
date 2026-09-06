"""Exercises the conversational auto-enrolment of an unrecognised face as
known_person, end to end, through the real DialogueProcessor.handle_command()
and the real FaceTrainer/ProfileStore -- everything except the camera, the
LLM call, and the tkinter popups used by the (untouched) photo/self enrolment
paths is real.

    .venv\\Scripts\\python.exe visual_guidance_assistant\\conversational_enrol_test.py

Uses a throwaway ProfileStore/MemoryStore/LayeredMemory under a temp
directory -- this NEVER writes to the real known_faces/profiles.json. It also
redirects FaceTrainer's reference-photo save path to a scratch folder for the
duration of the run (see main()): that path is a module-level constant, not
derived from the ProfileStore passed in, so without this an earlier version
of this exact script left Jordan.jpg/Riley.jpg/PhotoFriend.jpg/
SelfEnrolled.jpg behind in the real known_faces/ directory. The "camera" is
faked to return an existing photo from known_faces/ as raw pixels, purely so
dlib has an actual detectable face to encode; the resulting biometric samples
are stored under a fresh test name in the throwaway store, never associated
with the real person in that photo anywhere.

Three scenarios, matching the feature spec:
  A. Unrecognised stranger chats casually, gets asked their name naturally,
     is auto-enrolled as known_person with zero access. Rate limiting is
     also demonstrated (the follow-up turn does not get asked again).
  B. Unrecognised stranger CLAIMS to be a caregiver mid-conversation --
     confirmed they get known_person only, never caregiver, and are
     redirected to an admin.
  C. The existing "remember this person" (photo) and "remember me" (self,
     via the real capture_and_train / DEV_DEFAULT_ROLE dev path) flows are
     exercised unmodified, to prove this feature did not touch them.
"""
import os
import queue
import shutil
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from dialogue.dialogue_processor import DialogueProcessor
from memory.layered_memory import LayeredMemory
from memory.memory_processor import MemoryProcessor
from memory.memory_store import MemoryStore
from profiles.profile_store import (
    ROLE_ADMIN,
    ROLE_CAREGIVER,
    ROLE_KNOWN_PERSON,
    ROLE_PATIENT,
    ProfileStore,
)
import recognition.face_trainer as face_trainer_module
from recognition.face_recognizer import FaceRecognizer
from utils.latest import LatestValue
from utils.paths import KNOWN_FACES_DIR


class FakeCamera:
    """Returns a real detectable face (an existing repo photo, read fresh
    off disk) every time it's asked -- stands in for a live webcam so
    FaceTrainer's dlib pipeline has real pixels to encode."""

    def __init__(self, source_jpg):
        self._frame = cv2.imread(source_jpg)
        if self._frame is None:
            raise RuntimeError(f"could not read {source_jpg}")

    def get_latest_frame(self):
        return self._frame

    def get_latest_frame_bytes(self):
        ok, buf = cv2.imencode(".jpg", self._frame)
        return buf.tobytes() if ok else None

    def use(self, source_jpg):
        """Swap in a different face -- e.g. so two separate enrolments in
        the same scenario are not accidentally deduplicated as the same
        underlying face (see scenario_c)."""
        frame = cv2.imread(source_jpg)
        if frame is None:
            raise RuntimeError(f"could not read {source_jpg}")
        self._frame = frame


def make_processor(tmpdir, label, source_jpg, seed_patient=True):
    """One fully-wired DialogueProcessor against a throwaway store."""
    profile_store = ProfileStore(filepath=os.path.join(tmpdir, f"{label}_profiles.json"))
    if seed_patient:
        profile_store.upsert(name="TestPatient", role=ROLE_PATIENT)

    memory_store = MemoryStore(filepath=os.path.join(tmpdir, f"{label}_memory.json"))
    memory_processor = MemoryProcessor(memory_store)
    layered_memory = LayeredMemory(filepath=os.path.join(tmpdir, f"{label}_layered.json"))
    face_recognizer = FaceRecognizer(profile_store, layered_memory=layered_memory)
    camera = FakeCamera(source_jpg)
    detection_store = LatestValue([])

    dp = DialogueProcessor(
        command_queue=queue.Queue(),
        speech_queue=queue.Queue(),
        config={},
        conversation_history=[],
        conversation_lock=threading.Lock(),
        memory_processor=memory_processor,
        layered_memory=layered_memory,
        face_recognizer=face_recognizer,
        camera=camera,
        memory_store=memory_store,
        profile_store=profile_store,
        detection_store=detection_store,
        sound_queue=None,
        mic_listener=None,
        routine_store=None,
    )

    # Real LLM calls need network + a valid key; this test is about the
    # enrolment logic, not the model, so the model turn is stubbed to a
    # fixed, clearly-marked line. Every deterministic guard/handler in
    # handle_command() -- including everything this feature added -- still
    # runs for real.
    dp._ask_llm = lambda command: "[stubbed LLM reply]"
    return dp, profile_store


def set_unrecognized_face_present(dp, present=True):
    """What _update_scene_description() would have set, for a person
    detection in frame that did not match any enrolled profile."""
    dp.current_speaker = None
    dp.current_stored_role = None
    dp.current_identity_source = "default"
    dp._unrecognized_face_in_view = present


def audit(profile_store, name):
    p = profile_store.get(name)
    assert p is not None, f"{name} was not enrolled at all"
    print(f"    profile audit: name={p['name']!r} role={p['role']!r} "
          f"relationship={p.get('relationship')!r} "
          f"face_samples={len(p.get('face_encodings') or [])} "
          f"has_care_access={ProfileStore.has_care_access(p['role'])} "
          f"interaction_mode={ProfileStore.interaction_role(p['role'])}")
    return p


def scenario_a(tmpdir, source_jpg):
    print("\n" + "=" * 78)
    print("SCENARIO A: unrecognised stranger chats casually -> asked name -> "
          "auto-enrolled as known_person")
    print("=" * 78)
    dp, store = make_processor(tmpdir, "scenario_a", source_jpg)
    set_unrecognized_face_present(dp, True)

    print("\n[turn 1] stranger: \"Hi there, nice weather today isn't it\"")
    reply = dp.handle_command("Hi there, nice weather today isn't it")
    print(f"AI -> {reply!r}")
    assert "what's your name" in reply.lower(), "expected the natural name-ask to be appended"
    assert dp._pending_conversational_enrol is not None
    assert dp._pending_conversational_enrol["stage"] == "name"
    print("    OK: asked naturally, no trigger phrase used")

    print("\n[turn 2] stranger: \"My name is Jordan\"")
    reply = dp.handle_command("My name is Jordan")
    print(f"AI -> {reply!r}")
    assert "jordan" in reply.lower()
    profile = audit(store, "Jordan")
    assert profile["role"] == ROLE_KNOWN_PERSON
    assert not ProfileStore.has_care_access(profile["role"])
    print("    OK: enrolled as known_person, zero care access confirmed")

    print("\n[turn 3] stranger: \"He's my neighbour\" (optional relationship follow-up)")
    reply = dp.handle_command("He's my neighbour")
    print(f"AI -> {reply!r}")
    profile = audit(store, "Jordan")
    assert profile["relationship"] == "neighbour"
    print("    OK: relationship captured on the SAME known_person record (no role change)")

    print("\n[turn 4] stranger keeps talking: \"What time is it\" "
          "(rate-limit check -- must NOT ask again)")
    set_unrecognized_face_present(dp, True)  # still shows as unrecognised in this fake state
    reply = dp.handle_command("What time is it")
    print(f"AI -> {reply!r}")
    assert "what's your name" not in reply.lower(), \
        "should not re-ask so soon after already asking -- rate limit broken"
    print("    OK: cooldown suppressed a repeat ask")
    return True


def scenario_b(tmpdir, source_jpg):
    print("\n" + "=" * 78)
    print("SCENARIO B: unrecognised stranger CLAIMS to be a caregiver "
          "(privilege-escalation guard)")
    print("=" * 78)
    dp, store = make_processor(tmpdir, "scenario_b", source_jpg)
    set_unrecognized_face_present(dp, True)

    print("\n[turn 1] stranger: \"Hey, I'm here to help out\"")
    reply = dp.handle_command("Hey, I'm here to help out")
    print(f"AI -> {reply!r}")
    assert dp._pending_conversational_enrol is not None

    print("\n[turn 2] stranger: \"I'm Riley, I'm his caregiver\"")
    reply = dp.handle_command("I'm Riley, I'm his caregiver")
    print(f"AI -> {reply!r}")

    profile = audit(store, "Riley")
    assert profile["role"] == ROLE_KNOWN_PERSON, \
        f"CRITICAL FAILURE: Riley was enrolled as {profile['role']!r}, not known_person!"
    assert profile["role"] != ROLE_CAREGIVER
    assert profile["role"] != ROLE_ADMIN
    assert not ProfileStore.has_care_access(profile["role"])
    assert profile["name"] not in [c["name"] for c in store.caregivers()]
    assert profile["name"] not in [a["name"] for a in store.admins()]
    assert "ask an admin" in reply.lower()
    assert "caregiver" in reply.lower()
    print("    OK: enrolled as known_person ONLY -- caregiver claim was NOT granted")
    print("    OK: response redirects to admin setup, exactly as specified")
    return True


def scenario_c(tmpdir, source_jpg, second_jpg):
    print("\n" + "=" * 78)
    print("SCENARIO C: existing enrolment flows unaffected by this change")
    print("=" * 78)
    dp, store = make_processor(tmpdir, "scenario_c", source_jpg, seed_patient=False)

    # -- 1. "remember this person" (photo path) -- still hardcodes known_person,
    #    still uses the popup flow (monkeypatched here only to avoid blocking
    #    on a real tkinter dialog during an automated run; the method itself
    #    is untouched).
    answers = iter(["PhotoFriend", "cousin"])
    dp.face_trainer._ask_popup = lambda prompt, default="": next(answers, default)
    print("\n[command] \"remember this person\" (photo-based, popup-driven)")
    reply = dp.handle_command("remember this person")
    print(f"AI -> {reply!r} (trainer speaks its own outcome; empty string is expected)")
    profile = store.get("PhotoFriend")
    assert profile is not None, "existing photo-enrolment path is broken"
    assert profile["role"] == ROLE_KNOWN_PERSON
    print(f"    OK: photo-enrolled '{profile['name']}' still hardcoded to "
          f"known_person (relationship={profile['relationship']!r})")

    # -- 2. "remember my face" (self-enrolment, live path) -- untouched,
    #    including the DEV_DEFAULT_ROLE dev-mode shortcut that skips the role
    #    popup and defaults to admin (see recognition/face_trainer.py).
    #    A DIFFERENT face than step 1's, so this is a genuinely new person
    #    rather than being (correctly) deduplicated against PhotoFriend by
    #    the existing "already know you" check in _handle_remember_face.
    dp.camera.use(second_jpg)
    name_answers = iter(["SelfEnrolled"])
    dp.face_trainer._ask_popup = lambda prompt, default="": next(name_answers, default)
    print("\n[command] \"remember my face\" (self-enrolment, live path)")
    reply = dp.handle_command("remember my face")
    print(f"AI -> {reply!r}")
    if dp._pending_remember_me is not None:
        # bardan.jpg landed inside the existing ambiguous-match band against
        # PhotoFriend's encoding (an untouched, pre-existing behaviour of
        # _handle_remember_face / REMEMBER_ME_AMBIGUOUS_BAND) -- answer its
        # follow-up exactly as a real person would.
        print("[command] \"new\" (resolving the pre-existing ambiguous-match question)")
        reply = dp.handle_command("new")
        print(f"AI -> {reply!r} (trainer speaks its own outcome; empty string is expected)")
    profile = store.get("SelfEnrolled")
    assert profile is not None, "existing self-enrolment path is broken"
    from recognition.face_trainer import DEV_DEFAULT_ROLE
    print(f"    profile role={profile['role']!r} (DEV_DEFAULT_ROLE={DEV_DEFAULT_ROLE!r}, "
          f"a pre-existing dev-mode convenience this change did not touch)")
    assert profile["role"] == DEV_DEFAULT_ROLE

    # -- 3. Admin/caregiver access gating is unchanged: a known_person from
    #    EITHER path above still cannot pass has_care_access / caregivers().
    for n in ("PhotoFriend",):
        p = store.get(n)
        assert not ProfileStore.has_care_access(p["role"])
        assert p["name"] not in [c["name"] for c in store.caregivers()]
    print("    OK: known_person profiles from the pre-existing photo flow "
          "still have zero care access")
    return True


def main():
    tmpdir = tempfile.mkdtemp(prefix="guideyou_conv_enrol_test_")
    print(f"Throwaway state directory (never touches the real known_faces/): {tmpdir}")

    # Every FaceTrainer enrolment method saves its reference photo to
    # KNOWN_FACES_DIR -- a module-level constant in recognition.face_trainer,
    # NOT derived from the ProfileStore instance passed to it. A throwaway
    # ProfileStore alone is therefore NOT enough to keep this test from
    # writing real files into the project's real known_faces/ folder (this
    # bit a previous run of this exact script, leaving Jordan.jpg,
    # Riley.jpg, PhotoFriend.jpg and SelfEnrolled.jpg behind there).
    # Redirecting the module global for the duration of this run, and
    # restoring it after, fixes that at the source.
    real_known_faces_dir = face_trainer_module.KNOWN_FACES_DIR
    scratch_known_faces_dir = os.path.join(tmpdir, "known_faces")
    os.makedirs(scratch_known_faces_dir, exist_ok=True)
    face_trainer_module.KNOWN_FACES_DIR = scratch_known_faces_dir
    # A real, already-enrolled face photo from the project, used purely as
    # pixel data for the fake camera -- see FakeCamera's docstring.
    all_jpgs = sorted(f for f in os.listdir(KNOWN_FACES_DIR) if f.lower().endswith(".jpg"))
    if len(all_jpgs) < 2:
        print("Need at least two sample photos in known_faces/ to use as stand-in faces.")
        return 1
    # NOTE: Tom.jpg and Sarah.jpg used to live here but were byte-identical
    # (confirmed duplicate placeholder data, unrelated to this feature) and
    # have since been deleted -- see known_faces/README.txt. Priya.jpg's face
    # also isn't detectable by the live-path locator this test exercises
    # (capture_and_train's compute_encoding, a stricter single-pass check
    # than the photo path's multi-variant one). Aayush.jpg and bardan.jpg are
    # confirmed distinct (different hashes, different detected face boxes)
    # and both detectable by the live-path locator.
    preferred = [f for f in ("Aayush.jpg", "bardan.jpg") if f in all_jpgs]
    chosen = (preferred + all_jpgs)[:2]
    source_jpg = os.path.join(KNOWN_FACES_DIR, chosen[0])
    second_jpg = os.path.join(KNOWN_FACES_DIR, chosen[1])
    print(f"Using {source_jpg} as the stand-in 'stranger' face for dlib encoding.")
    print(f"Using {second_jpg} as a second, distinct stand-in face (scenario C only).\n")

    results = {}
    try:
        results["A"] = scenario_a(tmpdir, source_jpg)
        results["B"] = scenario_b(tmpdir, source_jpg)
        results["C"] = scenario_c(tmpdir, source_jpg, second_jpg)
    finally:
        face_trainer_module.KNOWN_FACES_DIR = real_known_faces_dir
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    for name, ok in results.items():
        print(f"  Scenario {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
