"""Face encoding and matching, backed by the JSON profile store.

This no longer owns any storage. It computes encodings and compares them; the
profile store is the single place anything about a person is written. That split
is what lets face data, voice data and role live in one record instead of a
pickle plus a JSON file that can drift apart.
"""
import cv2
import numpy as np

from profiles.profile_store import ProfileStore
from recognition.face_detect import FaceLocator

# face_recognition's own guidance: below ~0.6 euclidean distance is a match.
# Kept at the library default rather than loosened. Measured on this camera, the
# SAME person across four seconds produced distances of 0.43, 0.75 and 0.64 —
# so raising the threshold far enough to absorb that variation would also start
# matching different people, and in this app a wrong match hands someone
# caregiver mode. The variation is handled by storing several encodings per
# person instead (see ENROL_SAMPLES), which is the safe way to solve it.
FACE_MATCH_TOLERANCE = 0.6

# face_recognition.face_encodings() with no location hint internally calls
# face_locations(..., number_of_times_to_upsample=1). Measured on this camera at
# 1280x720 that finds the face in 1 frame out of 4, while upsample=0 finds it in
# 4 out of 4 and is four times faster. Upsampling doubles an already large
# close-up face past what the HOG detector scans for. This is why enrolment
# appeared to work and recognition then silently never matched.
DETECT_UPSAMPLE = 0

# How much closer the best match must be than the nearest DIFFERENT person
# before the name is trusted.
#
# The absolute tolerance alone is not enough once more than one person is
# enrolled. Measured on two real enrolled faces here, two genuinely different
# people sat 0.4596 apart — inside the 0.60 tolerance — while the SAME person
# across a few seconds reached 0.75. Those ranges overlap, so on an unlucky
# frame the wrong name is closer than the right one, and an absolute threshold
# cannot tell the difference.
#
# Requiring a gap fixes what the threshold cannot: if two enrolled people are
# both plausible, say nobody rather than guess. In this app an "unknown" costs
# a greeting; a wrong name can hand out caregiver mode.
FACE_MATCH_MARGIN = 0.08

# The tolerance a match must clear when there is nobody else enrolled to
# cross-check against.
#
# BUG, evidenced live: with a single profile enrolled, a genuine SECOND real
# person was labelled with the first person's name. Measured directly against
# the live enrolled profile's actual 6 stored samples (not a synthetic test):
# a real, different, unenrolled person matched EVERY ONE of those 6 samples at
# 0.4421-0.4956 - comfortably under the 0.60 tolerance. Ruled out before
# accepting this as a genuine model limit rather than a bug elsewhere: the
# YuNet crop was visually confirmed accurate; every photo_prep variant
# (contrast, de-moire, upscale) left the distance in the same 0.43-0.46 band;
# jittered re-encoding (num_jitters=10) widened the gap but costs 9.3s per
# face on this CPU, not usable live.
#
# The margin rule added earlier only fires when a DIFFERENT enrolled person is
# the rival, so with one profile there is nothing to compare against and the
# code fell back to the plain 0.60 ceiling — the SAME bar used when a rival
# exists to corroborate the match. That is backwards: with nobody to cross-
# check against there is LESS evidence, not the same amount, so the bar here
# is tighter, not equal.
#
# 0.42 is set from the measured pair above: it sits just under this real
# stranger's floor (0.4421) and inside the enrolled profile's own tighter
# samples (as low as 0.2058). It will legitimately increase how often the
# real enrolled person is called "unknown" on a poor frame — sticky identity
# covers that gap for up to sticky_seconds, and an "unknown" costs a greeting
# while a wrong name can hand out caregiver mode, so that trade is intentional.
FACE_MATCH_SOLO_TOLERANCE = 0.42


class FaceEncoder:
    def __init__(self, profile_store: ProfileStore, locator=None, azure_client=None):
        self.profiles = profile_store
        # Injected so enrolment and recognition share one detector instance —
        # YuNet holds a loaded ONNX net that should not be rebuilt per call.
        self.locator = locator or FaceLocator()
        # Optional AzureFaceClient — an alternative recognition BACKEND, not
        # a different code path. When present and healthy, recognize() and
        # match_details() below try it first and use its answer whenever it
        # returns one; every line of local dlib matching in both methods is
        # completely unchanged and is exactly what still runs when Azure is
        # unconfigured, unreachable, or returns None (meaning "could not
        # check this frame, not that nobody matched"). See
        # recognition/azure_face_client.py's module docstring for how its
        # confidence score is translated onto this file's distance scale so
        # the SAME tolerance/margin arithmetic below applies either way.
        self.azure_client = azure_client

    @property
    def known_names(self):
        names, _ = self.profiles.face_index()
        return names

    def encode_all(self, bgr_image, verbose=False):
        """Every face encoding in a BGR frame.

        Detection and encoding are deliberately separate: the locator finds the
        boxes (YuNet, or HOG as fallback) and dlib encodes exactly those boxes.
        Letting face_encodings() run its own detector is what made recognition
        fail silently, because its default settings could not find the face.
        """
        try:
            import face_recognition
        except ImportError:
            print("[face-encoder] face_recognition not installed")
            return []
        try:
            locations = self.locator.locate(bgr_image)
            if not locations:
                if verbose:
                    print("[face-encoder] no face located in this frame")
                return []
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            return face_recognition.face_encodings(rgb, known_face_locations=locations)
        except Exception as exc:
            print(f"[face-encoder] encoding failed: {type(exc).__name__}: {exc}")
            return []

    def compute_encoding(self, bgr_image, verbose=True):
        """128-d encoding for the single face in a BGR frame, or None.

        Returns None when there is no face, or more than one — enrolling from an
        ambiguous frame would silently attach someone else's face to a name.
        """
        encodings = self.encode_all(bgr_image, verbose=verbose)
        if not encodings:
            if verbose:
                print("[face-encoder] no face found in the image")
            return None
        if len(encodings) > 1:
            if verbose:
                print(f"[face-encoder] {len(encodings)} faces in frame — refusing to "
                      "enrol an ambiguous face; only one person should be in view")
            return None
        return encodings[0]

    def recognize(self, face_image, verbose=False):
        """Name for a BGR crop, or 'unknown'.

        Matches against EVERY stored encoding for each person and keeps the best,
        so a person enrolled from several angles is recognised from any of them.

        Tries Azure Face first if configured and healthy — see azure_client's
        docstring on __init__. A None from Azure means "could not check",
        not "no match", so this falls through to the local dlib path exactly
        as if Azure were never configured.
        """
        if self.azure_client is not None:
            azure_result = self.azure_client.identify(face_image, verbose=verbose)
            if azure_result is not None:
                return azure_result["name"] if azure_result["confident"] else "unknown"

        names, encodings = self.profiles.face_index()
        if not encodings:
            if verbose:
                print("[face-encoder] nobody enrolled yet")
            return "unknown"
        try:
            import face_recognition

            found = self.encode_all(face_image, verbose=verbose)
            if not found:
                return "unknown"
            distances = face_recognition.face_distance(
                [np.array(e) for e in encodings], found[0]
            )
            best = int(np.argmin(distances))
            best_name, best_distance = names[best], float(distances[best])

            # Closest sample belonging to somebody ELSE. Several samples of the
            # same person are expected to be close and must not count as a
            # rival — only a different name does.
            rival_name, rival_distance = None, float("inf")
            for name, distance in zip(names, distances):
                if name != best_name and distance < rival_distance:
                    rival_name, rival_distance = name, float(distance)

            # No rival = nothing to corroborate the match against, so the bar
            # is tighter than when a different enrolled person is available to
            # cross-check the margin against. See FACE_MATCH_SOLO_TOLERANCE.
            ceiling = FACE_MATCH_TOLERANCE if rival_name else FACE_MATCH_SOLO_TOLERANCE

            if verbose:
                rival = (f", next different person {rival_name} at {rival_distance:.4f}"
                         if rival_name else ", nobody else enrolled — solo tolerance applies")
                print(f"[face-encoder] closest: {best_name} at {best_distance:.4f} "
                      f"(ceiling {ceiling}){rival}")

            if best_distance >= ceiling:
                return "unknown"

            # Two enrolled people both plausible: refuse rather than pick one.
            if rival_name and (rival_distance - best_distance) < FACE_MATCH_MARGIN:
                if verbose:
                    print(f"[face-encoder] AMBIGUOUS: {best_name} "
                          f"{best_distance:.4f} vs {rival_name} "
                          f"{rival_distance:.4f} — gap "
                          f"{rival_distance - best_distance:.4f} is under "
                          f"{FACE_MATCH_MARGIN}; returning unknown")
                return "unknown"
            return best_name
        except ImportError:
            pass
        except Exception as exc:
            print(f"[face-encoder] recognize failed: {type(exc).__name__}: {exc}")
        return "unknown"

    def get_known_names(self):
        return self.known_names

    def match_details(self, face_image, verbose=False):
        """Same matching signal recognize() uses — identical constants,
        identical logic — but returns the full picture instead of collapsing
        straight to a name or 'unknown'.

        Added for "remember me" (dialogue_processor._handle_remember_face),
        which needs to tell "clearly nobody enrolled" apart from "possibly an
        already-enrolled person read poorly" — recognize() alone cannot do
        that; both come back as "unknown". Does NOT replace or modify
        recognize(), which stays exactly as tested for the live recognition
        path — this is purely additive, read-only against the same profiles.

        Returns a dict:
          name            best-matching enrolled name, or None if nobody enrolled
          distance        that match's distance, or None
          ceiling         the tolerance that applied (solo or corroborated)
          rival_name      the next different enrolled person, or None
          rival_distance  that rival's distance, or None
          confident       True exactly when recognize() would return `name`
                          rather than "unknown"
        Returns None (not a dict) if no face was found in the image at all —
        a different failure from "no confident match": the caller should
        treat that as "couldn't check, proceed to the normal enrolment flow
        and let it try its own several samples", not as evidence of anything.

        Tries Azure Face first if configured and healthy, same as
        recognize() — a None from Azure (not configured, unhealthy, call
        failed, or Azure itself found no usable face) falls through to the
        local computation below exactly as if Azure did not exist.
        """
        if self.azure_client is not None:
            azure_result = self.azure_client.identify(face_image, verbose=verbose)
            if azure_result is not None:
                return azure_result

        names, encodings = self.profiles.face_index()
        if not encodings:
            return {"name": None, "distance": None, "ceiling": None,
                    "rival_name": None, "rival_distance": None, "confident": False}
        try:
            import face_recognition

            found = self.encode_all(face_image, verbose=verbose)
            if not found:
                return None

            distances = face_recognition.face_distance(
                [np.array(e) for e in encodings], found[0]
            )
            best = int(np.argmin(distances))
            best_name, best_distance = names[best], float(distances[best])

            rival_name, rival_distance = None, None
            best_rival = float("inf")
            for name, distance in zip(names, distances):
                if name != best_name and distance < best_rival:
                    rival_name, best_rival = name, float(distance)
            if rival_name is not None:
                rival_distance = best_rival

            ceiling = FACE_MATCH_TOLERANCE if rival_name else FACE_MATCH_SOLO_TOLERANCE
            confident = best_distance < ceiling and not (
                rival_name is not None
                and (rival_distance - best_distance) < FACE_MATCH_MARGIN
            )
            if verbose:
                rival_note = (f", next different person {rival_name} at {rival_distance:.4f}"
                              if rival_name else ", nobody else enrolled — solo tolerance applies")
                print(f"[face-encoder] match_details: closest {best_name} at "
                      f"{best_distance:.4f} (ceiling {ceiling}){rival_note} -> "
                      f"confident={confident}")
            return {
                "name": best_name, "distance": best_distance, "ceiling": ceiling,
                "rival_name": rival_name, "rival_distance": rival_distance,
                "confident": confident,
            }
        except ImportError:
            return {"name": None, "distance": None, "ceiling": None,
                    "rival_name": None, "rival_distance": None, "confident": False}
