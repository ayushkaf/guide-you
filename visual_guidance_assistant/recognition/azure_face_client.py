"""Azure Face API backend — an optional, best-effort ALTERNATIVE to the
local YuNet+dlib pipeline in face_encoder.py, never a replacement for it.

Every public method here returns None (or, for identify(), a dict shaped
exactly like FaceEncoder.match_details()'s return value) and NEVER raises —
FaceEncoder consults this first and falls through to its own unchanged local
logic on anything but a clean, confident answer. See face_encoder.py's
recognize()/match_details() for the branch point.

CONFIDENCE TRANSLATION, NOT A FRESH DESIGN
Azure's identify() reports a confidence score, 0.0-1.0, HIGHER meaning more
confident — the opposite scale from dlib's distance (LOWER = closer/better).
To let every caller downstream of match_details() (the "remember me"
ambiguity check in dialogue_processor.py, the margin/solo-tolerance logic
here) go on using the exact same "smaller number = better" arithmetic
regardless of which backend actually answered, every confidence this module
reports is converted to a distance-shaped number via `1.0 - confidence`
before it leaves this file. The three safety RULES this preserves, from
face_encoder.py's own tolerances, are:
  1. an absolute floor below which a match is never trusted
     (AZURE_CONFIDENCE_FLOOR, analogous to FACE_MATCH_TOLERANCE)
  2. a STRICTER floor when nobody else is enrolled to cross-check against —
     less evidence means a tighter bar, not the same bar
     (AZURE_CONFIDENCE_FLOOR_SOLO, analogous to FACE_MATCH_SOLO_TOLERANCE)
  3. a required gap between the best candidate and the next DIFFERENT
     person, or the match is refused as ambiguous rather than guessed —
     reuses FACE_MATCH_MARGIN itself rather than inventing a second number,
     since both operate on the same post-translation distance-shaped scale

THESE NUMBERS ARE UNMEASURED STARTING DEFAULTS, not measured the way the
local tolerances were (face_encoder.py's constants each cite a specific
measured impostor/genuine score that justified them). There is no enrolled
Azure data on this deployment yet to measure against. They are chosen to
err toward "unknown" over a wrong match, matching this app's stated
priority (an unknown costs a greeting; a wrong match can hand out caregiver
mode) — but they should be revisited once real enrolled faces exist and a
real impostor test can be run, exactly like the local ones originally were.
"""
from __future__ import annotations

import io
import threading
import time
from typing import Optional

import cv2

from recognition.face_encoder import FACE_MATCH_MARGIN
from utils.service_health import ServiceHealth

# See the module docstring's "CONFIDENCE TRANSLATION" section for why these
# are expressed as a floor on Azure's own [0,1] confidence scale (higher is
# stricter) rather than copying face_encoder.py's numbers, which are on a
# different, non-comparable scale (dlib distance).
AZURE_CONFIDENCE_FLOOR = 0.5
AZURE_CONFIDENCE_FLOOR_SOLO = 0.6

DEFAULT_PERSON_GROUP_ID = "guideyou-people"

# How long to keep polling after kicking off a PersonGroup training pass
# before giving up and logging it as still-pending. Training is normally
# fast for a handful of people; this is generous headroom, not an expected
# wait.
TRAINING_POLL_TIMEOUT = 20.0
TRAINING_POLL_INTERVAL = 1.0


class AzureFaceClient:
    def __init__(
        self,
        key: Optional[str],
        endpoint: Optional[str],
        person_group_id: str = DEFAULT_PERSON_GROUP_ID,
        timeout: float = 8.0,
    ):
        self.person_group_id = person_group_id
        self.timeout = timeout
        self.client = None
        # Name a caller can look up once identify() names a person_id.
        # Populated by sync_profile() and by load_person_id_map() at
        # startup migration. Deliberately in-memory only — the durable
        # record is ProfileStore's own "azure_person_id" field per profile;
        # this is just a fast reverse index built from it.
        self._person_id_to_name: dict[str, str] = {}
        self._group_ready = False

        configured = bool(key) and bool(endpoint)
        self.health = ServiceHealth("azure-face", configured=configured, cooldown_seconds=30.0)

        if not configured:
            return
        try:
            from azure.cognitiveservices.vision.face import FaceClient
            from msrest.authentication import CognitiveServicesCredentials

            self.client = FaceClient(endpoint.rstrip("/"), CognitiveServicesCredentials(key))
        except Exception as exc:
            # A key and endpoint were given but the client couldn't even be
            # constructed (bad endpoint URL shape, SDK import problem, ...).
            # Treat exactly like "not configured" — never attempt again this
            # run, since retrying construction on a cooldown gains nothing.
            print(f"[azure-face] could not construct client, staying on local "
                  f"YuNet+dlib for this run: {type(exc).__name__}: {exc}")
            self.health.configured = False
            self.client = None

    # ---------- startup readiness ----------

    def ensure_ready(self) -> bool:
        """One-time startup check: authenticate, and make sure this
        deployment's PersonGroup exists (creating it on first run). Returns
        True only if Azure Face is genuinely usable right now.

        Safe to call even when not configured — just returns False.
        """
        if self.client is None or not self.health.configured:
            return False
        try:
            self.client.person_group.get(self.person_group_id)
            self._group_ready = True
            self.health.record_success()
            print(f"[azure-face] PersonGroup {self.person_group_id!r} found and reachable")
            return True
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                return self._create_person_group()
            self.health.record_failure(exc)
            return False

    def _create_person_group(self) -> bool:
        try:
            self.client.person_group.create(
                self.person_group_id, name="Guide YOU enrolled people"
            )
            self._group_ready = True
            self.health.record_success()
            print(f"[azure-face] created new PersonGroup {self.person_group_id!r}")
            return True
        except Exception as exc:
            self.health.record_failure(exc)
            return False

    # ---------- enrolment mirror ----------

    def sync_profile(self, name: str, photo_path: str, existing_person_id: Optional[str] = None):
        """Best-effort: create (or reuse) this person's Azure Person, add a
        face from their reference photo, and kick off retraining in the
        background. Returns the Azure person_id on success, or None on any
        failure — never raises, never blocks the caller on training.

        Called from two places: the one-time startup migration sweep over
        already-enrolled local profiles (app.py), and FaceTrainer right
        after every NEW local enrolment succeeds, so Azure's PersonGroup
        stays in sync going forward without anyone having to re-run a
        migration step. Local enrolment (the actual capture and the
        ProfileStore record) is complete and authoritative before this is
        ever called — this only ever mirrors it, and its failure changes
        nothing about what was already saved locally.
        """
        if not self.health.should_attempt() or not self._group_ready:
            return None
        try:
            if existing_person_id:
                person_id = existing_person_id
            else:
                person = self.client.person_group_person.create(self.person_group_id, name=name)
                person_id = person.person_id

            with open(photo_path, "rb") as handle:
                self.client.person_group_person.add_face_from_stream(
                    self.person_group_id, person_id, handle
                )

            self.client.person_group.train(self.person_group_id)
            self.health.record_success()
            self._person_id_to_name[person_id] = name
            print(f"[azure-face] synced {name!r} to Azure (person_id={person_id}), "
                  "training started")

            threading.Thread(
                target=self._await_training, args=(name,), daemon=True
            ).start()
            return person_id
        except Exception as exc:
            self.health.record_failure(exc)
            return None

    def _await_training(self, name: str) -> None:
        """Poll training status off the calling thread, purely for a clear
        log line — identify() does not wait on this; Azure simply keeps
        answering from whatever the last SUCCEEDED training snapshot was
        until this one completes."""
        deadline = time.time() + TRAINING_POLL_TIMEOUT
        while time.time() < deadline:
            try:
                status = self.client.person_group.get_training_status(self.person_group_id)
                if status.status == "succeeded":
                    print(f"[azure-face] training succeeded after enrolling {name!r}")
                    return
                if status.status == "failed":
                    print(f"[azure-face] training FAILED after enrolling {name!r}: "
                          f"{getattr(status, 'message', '(no detail)')}")
                    return
            except Exception as exc:
                print(f"[azure-face] could not check training status: "
                      f"{type(exc).__name__}: {exc}")
                return
            time.sleep(TRAINING_POLL_INTERVAL)
        print(f"[azure-face] training still running after enrolling {name!r} "
              f"({TRAINING_POLL_TIMEOUT:.0f}s) — will keep answering from the "
              "previous snapshot until it finishes")

    def load_person_id_map(self, profiles) -> None:
        """Rebuild the in-memory person_id->name index from ProfileStore
        records that already carry an azure_person_id — called once at
        startup after migration, so identify() can resolve names for people
        who were already migrated on a PREVIOUS run without re-syncing them."""
        for profile in profiles:
            person_id = profile.get("azure_person_id")
            if person_id:
                self._person_id_to_name[person_id] = profile["name"]

    # ---------- recognition ----------

    def identify(self, bgr_image, verbose: bool = False):
        """Azure-backed analogue of FaceEncoder.match_details() — same
        return shape, same meaning of None. See the module docstring for
        the confidence-to-distance translation this applies before
        returning anything, so callers never need to know which backend
        answered.
        """
        if not self.health.should_attempt() or not self._group_ready:
            return None
        try:
            ok, buf = cv2.imencode(".jpg", bgr_image)
            if not ok:
                return None
            detected = self.client.face.detect_with_stream(
                io.BytesIO(buf.tobytes()), detection_model="detection_01",
                recognition_model="recognition_01",
            )
            self.health.record_success()

            if not detected:
                if verbose:
                    print("[azure-face] no face located in this frame")
                return None  # "couldn't check" — let the caller fall through to local

            if len(detected) > 1:
                # Ambiguous crop — mirrors face_encoder.compute_encoding()'s
                # own refusal to guess between more than one face. Fall
                # through to local rather than arbitrarily picking one.
                if verbose:
                    print(f"[azure-face] {len(detected)} faces in this crop — "
                          "ambiguous, deferring to local")
                return None

            face_id = detected[0].face_id
            # confidence_threshold=0.0: deliberately asks Azure for candidates
            # EVEN BELOW its own opaque default cutoff (~0.5), the same way
            # face_encoder.py always computes the distance to every enrolled
            # encoding rather than trusting a library default to decide what
            # counts as "close enough" — the tolerance/margin decision below
            # is this codebase's to make, not the API's.
            results = self.client.face.identify(
                [face_id], person_group_id=self.person_group_id,
                max_num_of_candidates_returned=2, confidence_threshold=0.0,
            )
            self.health.record_success()

            candidates = results[0].candidates if results else []
            if not candidates:
                # Unlike face_encoder.py's local "nobody enrolled" case
                # (which IS the source of truth and can safely say so),
                # Azure's PersonGroup can be behind ProfileStore — mid
                # migration, or a sync that failed for one person. Treating
                # an empty Azure group as authoritative "nobody enrolled"
                # here could report someone who IS enrolled locally as
                # unknown just because Azure hasn't caught up yet. Fall
                # through to local instead, which always reflects the
                # current ProfileStore state.
                if verbose:
                    print("[azure-face] identify: PersonGroup has nobody enrolled "
                          "(or not yet synced) — deferring to local")
                return None

            ranked = sorted(candidates, key=lambda c: c.confidence, reverse=True)
            best_name = self._person_id_to_name.get(ranked[0].person_id)
            if best_name is None:
                # A person_id Azure knows but we have no local name for —
                # should not happen once load_person_id_map() has run, but an
                # unresolved id must never be reported as if it were a name.
                if verbose:
                    print(f"[azure-face] identify returned unmapped person_id "
                          f"{ranked[0].person_id!r} — deferring to local")
                return None
            best_distance = 1.0 - ranked[0].confidence

            rival_name, rival_distance = None, None
            for candidate in ranked[1:]:
                name = self._person_id_to_name.get(candidate.person_id)
                if name and name != best_name:
                    rival_name, rival_distance = name, 1.0 - candidate.confidence
                    break

            # No rival = nothing to corroborate the match against, so the bar
            # is tighter — exactly face_encoder.FACE_MATCH_SOLO_TOLERANCE's
            # reasoning, translated to Azure's scale (see module docstring).
            ceiling = (1.0 - AZURE_CONFIDENCE_FLOOR) if rival_name else (1.0 - AZURE_CONFIDENCE_FLOOR_SOLO)
            confident = best_distance < ceiling and not (
                rival_name is not None and (rival_distance - best_distance) < FACE_MATCH_MARGIN
            )

            if verbose:
                rival_note = (f", next different person {rival_name} at {rival_distance:.4f}"
                              if rival_name else ", nobody else enrolled — solo tolerance applies")
                print(f"[azure-face] identify: closest {best_name} at "
                      f"{best_distance:.4f} (ceiling {ceiling:.4f}){rival_note} "
                      f"-> confident={confident}")

            # ALWAYS the actual best-guess name, exactly like
            # face_encoder.match_details() — "confident" is the separate
            # flag callers check; recognize() is what collapses an
            # unconfident guess to "unknown", not this method.
            return {
                "name": best_name, "distance": best_distance, "ceiling": ceiling,
                "rival_name": rival_name, "rival_distance": rival_distance,
                "confident": confident,
            }
        except Exception as exc:
            self.health.record_failure(exc)
            return None
