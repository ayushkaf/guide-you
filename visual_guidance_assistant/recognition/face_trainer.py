"""Enrolment: capture a person's face, role, and voice into one profile record.

Popups (tkinter) rather than console input() — a background thread cannot read
stdin while the OpenCV window holds keyboard focus, which is what silently
enrolled a face under an empty name in the original build.

The voice step is spoken rather than shown, because being asked out loud to say
a phrase is a natural thing to respond to, whereas a dialog box demanding speech
is not.
"""
import os
import time
import tkinter as tk
from tkinter import simpledialog

import cv2

from profiles.profile_store import (
    ROLE_ADMIN,
    ROLE_CAREGIVER,
    ROLE_KNOWN_PERSON,
    ROLE_PATIENT,
    ProfileStore,
)
from recognition import photo_prep
from recognition.face_recognizer import FaceRecognizer, safe_filename
from utils.paths import DEBUG_DIR, KNOWN_FACES_DIR

# Written on every photo-enrolment attempt, before detection runs.
PHOTO_ATTEMPT_FILE = os.path.join(DEBUG_DIR, "last_photo_attempt.jpg")
from voice_id import voice_embedding

# Role every enrolment gets right now, with NO role popup. This device is in
# active development by its own user: forcing a patient/caregiver/admin choice
# on every re-enrolment was pure friction, and admin already has full access.
# The role SYSTEM is untouched — guards, alert-log gating and routine gating
# all still key off roles — this only skips the question. Set to None to bring
# the popup back for real deployment (and routines only fire for a "patient",
# so a real deployment must enrol the supported person with the popup on).
DEV_DEFAULT_ROLE = ROLE_ADMIN

# Spoken exactly once per patient, ever — right when enrolment completes, not
# on later recognition (a lengthy self-introduction every time they are seen
# would read as the assistant repeatedly failing to recognise them, which is
# the opposite of reassuring). Tracked by ProfileStore.mark_introduced().
PATIENT_INTRO_TEXT = (
    "Hi, I'm Guide YOU. I'm here to keep you company, help you remember "
    "things, and let you know who's around you. I'm not going anywhere — "
    "just say my name if you need me."
)

# Everyone else on a successful enrolment, and a patient who is simply
# re-enrolling (fresh face or voice samples) rather than meeting Guide YOU for
# the first time — has_been_introduced already being true is exactly how that
# case is told apart from a genuinely new patient.
ENROL_CONFIRM_TEXT = "Thank you, I'll remember you."

VOICE_PROMPT = (
    "Now please say a short phrase, so I can learn the sound of your voice. "
    "Anything you like."
)
VOICE_SECONDS = 4.0

# How many face samples to gather, and over how long. One sample is not enough:
# measured on this camera, the SAME face across four seconds gave encoding
# distances of 0.43, 0.75 and 0.64, and anything past 0.6 counts as a different
# person. Storing several samples lets the matcher take the best of them, which
# absorbs pose and lighting drift without loosening the tolerance — loosening it
# would start matching different people, and here that hands out caregiver mode.
ENROL_SAMPLES = 6
ENROL_SAMPLE_GAP = 0.45

# A photograph does not move, blink or turn its head, so extra samples of it are
# near-duplicates. Three is enough to survive a moment of glare or a shaky hand
# on one of them, without making someone hold a phone up for three seconds.
# Photographing a screen is noisier than a live face, so take more attempts and
# spread them wider — a glare band or a shaky moment passes within a second.
PHOTO_SAMPLES = 7
PHOTO_SAMPLE_GAP = 0.55

# Pause after asking, before sampling starts, so there is time to lift the
# phone. The spoken prompt plays during this. Sampling then runs for roughly
# another 3.5s, giving about a 6 second window in total rather than 1.5.
PHOTO_LEAD_IN = 2.5

# Accept a weaker detection than a live face needs. Anything above this is
# enrolled; between this and PHOTO_SCORE_GOOD it is enrolled with a warning,
# because for this feature a decent match that greets someone by name beats a
# perfect match that never happens.
PHOTO_SCORE_MIN = 0.45
PHOTO_SCORE_GOOD = 0.70


class FaceTrainer:
    def __init__(
        self,
        face_recognizer: FaceRecognizer,
        profile_store: ProfileStore,
        speech_queue=None,
        mic_listener=None,
    ):
        self.face_recognizer = face_recognizer
        self.profiles = profile_store
        self.speech_queue = speech_queue
        self.mic_listener = mic_listener

    # ---------- popups ----------

    def _ask_popup(self, prompt: str, default: str = "") -> str:
        """Modal text prompt. Returns `default` if cancelled or left blank."""
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        answer = simpledialog.askstring("Guide YOU — Set up", prompt, parent=root)
        root.destroy()
        if answer is None:
            return default
        return answer.strip() or default

    def _ask_role(self) -> str:
        """Role decides how the assistant speaks and what it will disclose, so
        it must be explicit.

        Anything not clearly caregiver or admin resolves to patient — the
        gentler mode is the safe default when the answer is unclear or the box
        is dismissed. Matching is prefix/keyword based because this is typed by
        hand and "Carer", "admin ", "ADMIN" all mean what they look like.
        """
        raw = self._ask_popup(
            "Who is this person?\n\n"
            "  patient    - the person being supported\n"
            "  caregiver  - someone who helps them\n"
            "  admin      - whoever set this device up (full access)\n\n"
            "Type one of: patient, caregiver, admin",
            default=ROLE_PATIENT,
        ).strip().lower()

        # Admin checked first: "admin" contains no 'c', but check order matters
        # if someone types something like "care admin".
        if raw.startswith("a") or "admin" in raw:
            return ROLE_ADMIN
        if raw.startswith("c") or "care" in raw or "help" in raw:
            return ROLE_CAREGIVER
        return ROLE_PATIENT

    # ---------- Azure Face mirror ----------

    def _mirror_to_azure(self, name: str, photo_path: str) -> None:
        """Best-effort: push a freshly (re-)enrolled person's reference photo
        into Azure's PersonGroup, if Azure Face is configured and healthy.

        Called after EVERY successful local enrolment — capture_known_person,
        capture_and_train, and capture_conversational_person all call this
        right after invalidate_cache(). Local enrolment is already complete
        and saved by the time this runs; this only keeps Azure's PersonGroup
        in sync going forward so recognition stays reliable if Azure is (or
        later becomes) the active backend. Never raises, never blocks on
        Azure's training — see AzureFaceClient.sync_profile().
        """
        azure_client = getattr(self.face_recognizer.encoder, "azure_client", None)
        if azure_client is None:
            return
        existing_id = (self.profiles.get(name) or {}).get("azure_person_id")
        person_id = azure_client.sync_profile(name, photo_path, existing_person_id=existing_id)
        if person_id and person_id != existing_id:
            self.profiles.set_azure_person_id(name, person_id)

    # ---------- voice ----------

    def _speak(self, text: str) -> None:
        if self.speech_queue is None:
            print(f"[enrol] (would say) {text}")
            return
        try:
            self.speech_queue.put_nowait(text)
        except Exception:
            pass

    def _capture_voice(self, name: str):
        """Speak a prompt, record a few seconds, return an embedding or None.

        Voice is optional. If it fails, enrolment still succeeds with the face —
        losing a supplementary signal is not a reason to make someone repeat the
        whole setup.
        """
        if self.mic_listener is None:
            print("[enrol] no microphone available; skipping voice capture")
            return None

        self._speak(VOICE_PROMPT)
        # Wait for the prompt to finish playing, otherwise the recording is of
        # the assistant's own voice. SpeechSynthesizer pauses the mic while it
        # speaks and resumes when done, so waiting for resume is the signal.
        deadline = time.time() + 15.0
        time.sleep(0.4)
        while self.mic_listener.is_paused() and time.time() < deadline:
            time.sleep(0.1)
        time.sleep(0.3)

        # Hold the mic loop off so it doesn't fight us for the device.
        self.mic_listener.pause()
        try:
            print(f"[enrol] recording {VOICE_SECONDS:.0f}s of voice for {name}...")
            audio = self.mic_listener.record_phrase(VOICE_SECONDS)
        finally:
            self.mic_listener.resume()

        if audio is None:
            print("[enrol] voice capture failed")
            return None

        embedding = voice_embedding.embed(audio)
        if embedding is None:
            print("[enrol] not enough speech in the recording to build a voice print")
            return None
        print(f"[enrol] voice print captured ({len(embedding)} values)")
        return embedding.tolist()

    # ---------- main flow ----------

    def _gather_samples(self, frame, camera):
        """Collect several usable face encodings, a moment apart.

        Sampling over a couple of seconds catches small changes in pose and
        expression, which is exactly the variation that made a single-sample
        enrolment fail to match later.
        """
        samples = []
        first = frame

        for index in range(ENROL_SAMPLES):
            if index == 0:
                current = first
            elif camera is not None:
                time.sleep(ENROL_SAMPLE_GAP)
                current = camera.get_latest_frame()
            else:
                break
            if current is None:
                continue
            # BGR straight through — the locator wants OpenCV's native order.
            # verbose only on the first, so a normal run isn't buried in noise.
            encoding = self.face_recognizer.encoder.compute_encoding(
                current, verbose=(index == 0)
            )
            if encoding is not None:
                samples.append([float(v) for v in encoding])
            print(f"[enrol] sample {index + 1}/{ENROL_SAMPLES}: "
                  f"{'captured' if encoding is not None else 'no clear single face'}")

        return samples

    # ---------- enrolling someone from a photograph ----------

    @staticmethod
    def _save_attempt_frame(frame) -> None:
        """Keep the exact frame enrolment was attempted on, success or not."""
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            cv2.imwrite(PHOTO_ATTEMPT_FILE, frame)
            print(f"[enrol-photo] saved the attempted frame to {PHOTO_ATTEMPT_FILE}")
        except Exception as exc:
            print(f"[enrol-photo] could not save the attempt frame: {exc}")

    def _encode_photo_face(self, bgr):
        """Find and encode the face in a photo held up to the camera.

        Returns (encoding, score, variant_name, faces_seen).

        Differs from the live path in three ways, all because a screen is a
        harder subject than a face:
          - several cheap preprocessing variants are tried, best score wins
          - a weaker detection is accepted (PHOTO_SCORE_MIN)
          - more than one face is reported rather than silently refused, so the
            person can be told to keep themselves out of shot
        """
        import face_recognition

        locator = self.face_recognizer.encoder.locator
        best = (None, 0.0, None, 0)

        for name, transform in photo_prep.VARIANTS:
            try:
                variant = transform(bgr)
            except Exception:
                continue

            scored = locator.locate_scored(variant, min_score=PHOTO_SCORE_MIN)
            if not scored:
                continue
            if len(scored) > 1:
                # Record the ambiguity but keep looking: another variant may
                # resolve it, and we want the count for the spoken guidance.
                best = (best[0], best[1], best[2], max(best[3], len(scored)))
                continue

            box, score = scored[0]
            if score <= best[1]:
                continue

            scale = photo_prep.SCALE_OF.get(name)
            try:
                rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(
                    rgb, known_face_locations=[box]
                )
            except Exception as exc:
                print(f"[enrol-photo] encoding failed on '{name}': "
                      f"{type(exc).__name__}: {exc}")
                continue
            if encodings:
                best = (encodings[0], score, name, max(best[3], 1))

        return best

    def capture_known_person(self, frame, camera=None):
        """Enrol someone by holding a photo of them up to the camera.

        Different from capture_and_train in two ways that matter:

        1. Fewer, faster samples. A photo does not move, so six samples of it
           are six near-identical encodings — they add storage, not robustness.
           The variation that multi-sampling exists to capture simply is not
           there.
        2. It fails loudly. Photographing a screen introduces glare, moire
           interference and softness that a live face does not have, so "no face
           found" is a likely and RECOVERABLE outcome here. The person is told
           what to change rather than just that it did not work.

        The result is a known_person: greeted by name, granted nothing.
        """
        if frame is None:
            print("[enrol-photo] ERROR: no frame available from the camera")
            self._speak("I couldn't see anything just then. Let's try again.")
            return False

        print(f"[enrol-photo] using frame {frame.shape[1]}x{frame.shape[0]}")

        # Saved BEFORE any detection runs, so there is always a record of what
        # the camera actually saw — especially when enrolment fails, which is
        # exactly when the frame was previously thrown away and the failure
        # became impossible to diagnose.
        self._save_attempt_frame(frame)

        stats = photo_prep.describe_frame(frame)
        print(f"[enrol-photo] frame: brightness {stats['brightness']:.0f} "
              f"contrast {stats['contrast']:.0f} sharpness {stats['sharpness']:.0f} "
              f"blown-out {stats['blown_out_pct']:.1f}% dark {stats['very_dark_pct']:.1f}%")

        self._speak("Show me the photo now, and hold it steady.")

        samples = []
        best_score = 0.0
        multi_face_seen = 0
        last_stats = stats

        # Give the person time to actually raise the phone.
        #
        # This is the likeliest reason enrolment failed in real use while every
        # simulated glare/moire/blur condition passed: the frame was grabbed the
        # instant the command was recognised, and the whole sampling window was
        # over in about a second and a half. Somebody who says "remember this
        # person" and THEN lifts their phone was never in shot for any of it.
        #
        # The first sample still uses the frame from the moment of the command,
        # in case the photo was already up.
        if camera is not None:
            time.sleep(PHOTO_LEAD_IN)

        for index in range(PHOTO_SAMPLES):
            if index == 0:
                current = frame
            elif camera is not None:
                time.sleep(PHOTO_SAMPLE_GAP)
                current = camera.get_latest_frame()
            else:
                break
            if current is None:
                continue

            encoding, score, variant, n_faces = self._encode_photo_face(current)
            last_stats = photo_prep.describe_frame(current)
            if encoding is not None:
                samples.append([float(v) for v in encoding])
                best_score = max(best_score, score)
                note = f"score {score:.3f} via {variant}"
                if score < PHOTO_SCORE_GOOD:
                    note += " (weak)"
            elif n_faces > 1:
                multi_face_seen += 1
                note = f"{n_faces} faces in view — ambiguous, skipped"
            else:
                note = "no face found"
            print(f"[enrol-photo] sample {index + 1}/{PHOTO_SAMPLES}: {note}")

        if not samples:
            if multi_face_seen:
                # Almost certainly the person holding the phone is in shot too.
                # This is the one failure the person cannot fix by holding
                # steadier, so it gets its own instruction.
                print(f"[enrol-photo] FAILED: more than one face in "
                      f"{multi_face_seen}/{PHOTO_SAMPLES} samples")
                self._speak(
                    "I can see more than one face, so I'm not sure which one to "
                    "learn. Hold the phone closer so the photo fills my view, "
                    "and keep your own face out of the picture. Then say "
                    "remember this person again."
                )
            else:
                print("[enrol-photo] FAILED: no usable face in any sample")
                self._speak(photo_prep.advice_for(last_stats) +
                            " Then say remember this person again.")
            print(f"[enrol-photo] the frame is saved at {PHOTO_ATTEMPT_FILE}")
            return False

        quality = "good" if best_score >= PHOTO_SCORE_GOOD else "reduced"
        print(f"[enrol-photo] {len(samples)} usable sample(s) of {PHOTO_SAMPLES}, "
              f"best score {best_score:.3f} ({quality} confidence)")
        if quality == "reduced":
            print("[enrol-photo] WARNING: enrolled with reduced confidence — "
                  "recognition may be unreliable and this person may need "
                  "re-enrolling from a clearer photo")

        name = self._ask_popup(
            "Who is this?\n(the name you'd like me to call them)", default=""
        ).strip()
        if not name:
            print("[enrol-photo] no name given — aborting")
            self._speak("No name given, so I haven't saved anyone.")
            return False

        relationship = self._ask_popup(
            f"How is {name} related to the person you support?\n"
            "e.g. daughter, son, friend, neighbour  (optional, press OK to skip)",
            default="",
        ).strip() or None

        profile = self.profiles.upsert(
            name=name,
            role=ROLE_KNOWN_PERSON,
            relationship=relationship,
            face_encodings=samples,
        )

        photo_path = os.path.join(KNOWN_FACES_DIR, f"{safe_filename(name)}.jpg")
        try:
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            cv2.imwrite(photo_path, frame)
        except Exception as exc:
            print(f"[enrol-photo] could not save reference photo: {exc}")

        self.face_recognizer.invalidate_cache()
        self._mirror_to_azure(name, photo_path)

        print(f"[enrol-photo] saved known person: name={profile['name']} "
              f"role={profile['role']} relationship={profile['relationship']} "
              f"face_samples={len(samples)}")

        if relationship:
            self._speak(f"I'll remember {name}, your {relationship}.")
        else:
            self._speak(f"I'll remember {name}.")
        return True

    # ---------- enrolling someone auto-detected in conversation ----------

    def capture_conversational_person(self, frame, name: str, camera=None, relationship=None):
        """Auto-enrol someone from a live conversation, with no popup at all.

        Reached only from DialogueProcessor's "I don't think we've met, what's
        your name?" flow (see _resolve_pending_conversational_enrol there) for
        a face the camera does not otherwise recognise. Deliberately has NO
        role parameter, unlike capture_and_train below — this path enrols
        known_person and nothing else, structurally, so no caller of this
        method can ever hand it a role that would grant real system access,
        no matter what the person being enrolled claims about themselves in
        conversation (see the caller's privileged-role-claim guard).

        Samples are gathered live over a couple of seconds, the same way
        capture_and_train does it, rather than the photo path's single-frame
        variants — this is a live face in front of the camera, not a photo
        held up to it.
        """
        clean_name = (name or "").strip()
        if not clean_name:
            print("[enrol-conversational] no name given — aborting (never enrol a nameless face)")
            return None
        if frame is None:
            print("[enrol-conversational] ERROR: no frame available from the camera")
            return None

        print(f"[enrol-conversational] using frame {frame.shape[1]}x{frame.shape[0]} "
              f"for {clean_name!r}")
        samples = self._gather_samples(frame, camera)
        if not samples:
            print("[enrol-conversational] no usable face in any sample — enrolment aborted")
            return None
        print(f"[enrol-conversational] {len(samples)} usable face sample(s) of {ENROL_SAMPLES}")

        profile = self.profiles.upsert(
            name=clean_name,
            role=ROLE_KNOWN_PERSON,
            relationship=relationship,
            face_encodings=samples,
        )

        photo_path = os.path.join(KNOWN_FACES_DIR, f"{safe_filename(clean_name)}.jpg")
        try:
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            cv2.imwrite(photo_path, frame)
        except Exception as exc:
            print(f"[enrol-conversational] could not save reference photo: {exc}")

        self.face_recognizer.invalidate_cache()
        self._mirror_to_azure(clean_name, photo_path)

        print(f"[enrol-conversational] saved known person: name={profile['name']} "
              f"role={profile['role']} relationship={profile['relationship']} "
              f"face_samples={len(samples)}")
        return profile["name"]

    # ---------- enrolling a user of the system ----------

    def capture_and_train(self, frame, camera=None):
        """Full enrolment. `camera` lets several samples be taken over a moment.

        Returns the enrolled name on success, or None on failure — the caller
        needs the name to stop the recognition-triggered greeting immediately
        repeating whatever this method already said.
        """
        if frame is None:
            print("[enrol] ERROR: no frame available from the camera")
            return None

        print(f"[enrol] using frame {frame.shape[1]}x{frame.shape[0]}")

        # Gather the face samples BEFORE asking anything. If there is no usable
        # face, there is no point walking someone through four dialogs.
        self._speak("Hold still for a moment while I look at you.")
        samples = self._gather_samples(frame, camera)
        if not samples:
            print("[enrol] no usable face in any sample — enrolment aborted")
            self._speak("I couldn't see a face clearly. Let's try again in a moment.")
            return None
        print(f"[enrol] {len(samples)} usable face sample(s) of {ENROL_SAMPLES}")

        name = self._ask_popup("What is this person's name?", default="").strip()
        if not name:
            print("[enrol] no name given — aborting (never enrol a nameless face)")
            return None

        relationship = None
        contact_note = None
        if DEV_DEFAULT_ROLE is not None:
            # Development mode: name is the ONLY popup. See DEV_DEFAULT_ROLE.
            role = DEV_DEFAULT_ROLE
            print(f"[enrol] role defaulted to {role!r} (dev mode, no popup)")
        else:
            role = self._ask_role()
            # Admins get the same contact questions as caregivers: whoever set
            # the device up is usually also worth being able to name and reach.
            #
            # One combined popup rather than two sequential ones — relationship
            # and a contact note are both optional metadata, neither gates any
            # permission (role, asked above, is what does that), so there is
            # nothing safety-relevant lost by asking them together. Splits on
            # the first comma; a reply with no comma is taken as relationship
            # only, which is the common case ("daughter") and needs no comma
            # to be usable.
            if role in (ROLE_CAREGIVER, ROLE_ADMIN):
                combined = self._ask_popup(
                    f"Relationship and contact note for {name}?\n"
                    "(e.g. \"daughter, mobile 555-0100\" — both optional, "
                    "press OK to skip. Relationship first, then a comma, "
                    "then any contact note.)",
                    default="",
                ).strip()
                if combined:
                    if "," in combined:
                        rel_part, note_part = combined.split(",", 1)
                        relationship = rel_part.strip() or None
                        contact_note = note_part.strip() or None
                    else:
                        relationship = combined
                        contact_note = None

        voice = self._capture_voice(name)

        profile = self.profiles.upsert(
            name=name,
            role=role,
            relationship=relationship,
            contact_note=contact_note,
            face_encodings=samples,
            voice_embedding=voice,
        )

        # Reference photo, for a caregiver auditing who has been enrolled.
        photo_path = os.path.join(KNOWN_FACES_DIR, f"{safe_filename(name)}.jpg")
        try:
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            cv2.imwrite(photo_path, frame)
        except Exception as exc:
            print(f"[enrol] could not save reference photo: {exc}")

        self.face_recognizer.invalidate_cache()
        self._mirror_to_azure(name, photo_path)

        print(f"[enrol] saved profile: name={profile['name']} role={profile['role']} "
              f"relationship={profile['relationship']} "
              f"face_samples={len(samples)} voice={'yes' if voice else 'no'}")

        # already_introduced reflects the record as it was BEFORE this upsert —
        # upsert() never touches has_been_introduced, so for a genuinely new
        # patient this is False (set by _new_record) and for anyone re-running
        # "remember my face" to refresh their samples it is whatever it already
        # was. That is exactly the distinction between "meeting for the first
        # time" and "updating an existing profile".
        already_introduced = bool(profile.get("has_been_introduced"))
        if role == ROLE_PATIENT and not already_introduced:
            self.profiles.mark_introduced(name)
            print(f"[enrol] first-time patient introduction spoken for {name!r}")
            self._speak(PATIENT_INTRO_TEXT)
        else:
            self._speak(ENROL_CONFIRM_TEXT)

        return name
