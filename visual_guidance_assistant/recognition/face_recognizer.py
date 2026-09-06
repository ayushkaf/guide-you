"""Face recognizer that labels person detections with names and personas.

Recognition is throttled internally. A dlib face encoding costs roughly
50-200ms on CPU, per person, per call — running that inside a 30 FPS render
loop (which is what used to happen, and again from the dialogue thread) means
the app spends essentially all its time encoding faces. The throttle bounds
that cost no matter how often or from how many threads this is called.
"""
import os
import re
import time
import cv2

from recognition.face_detect import FaceLocator
from recognition.face_detector import FaceDetector
from recognition.face_encoder import FaceEncoder
from memory.layered_memory import LayeredMemory
from profiles.profile_store import ProfileStore
from utils.paths import KNOWN_FACES_DIR

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def safe_filename(name: str, fallback: str = "user") -> str:
    """Names come from a free-text popup, so they reach the filesystem
    untrusted — strip anything that could escape known_faces/."""
    cleaned = _UNSAFE_NAME_CHARS.sub("_", (name or "").strip()).strip("._")
    return cleaned or fallback


class FaceRecognizer:
    def __init__(
        self,
        profile_store: ProfileStore,
        layered_memory: LayeredMemory = None,
        min_interval: float = 1.0,
        cache_ttl: float = 3.0,
        sticky_seconds: float = 60.0,
        detector: str = None,
        azure_face_client=None,
    ):
        self.detector = FaceDetector()
        self.profiles = profile_store
        self.encoder = FaceEncoder(
            profile_store, locator=FaceLocator(detector or "yunet"),
            azure_client=azure_face_client,
        )
        self.layered_memory = layered_memory

        # At most one real recognition pass per `min_interval` seconds; results
        # stay valid for `cache_ttl` seconds so names don't flicker in between.
        self.min_interval = float(min_interval)
        self.cache_ttl = float(cache_ttl)
        self._last_run = 0.0
        self._cached_names: list[str] = []
        self._cached_at = 0.0

        # STICKY IDENTITY.
        # No face detector finds a face in every frame — people look away, tilt,
        # move. Without this, one missed frame drops the person back to
        # "unknown", which in this app means being asked who they are by
        # something that knew them a second ago, and losing caregiver mode
        # mid-sentence.
        #
        # People do not swap places in front of a camera every few seconds, so
        # the last confident identification is held for a while. Deliberately a
        # separate layer from the detector: it makes ANY detector steadier, and
        # it is what keeps the assistant coherent if YuNet is unavailable and
        # the far weaker HOG fallback is doing the work.
        self.sticky_seconds = float(sticky_seconds)
        self._sticky_name: str | None = None
        self._sticky_at = 0.0

    @staticmethod
    def _crop(frame, bbox):
        """Crop a detection box, clamped to the frame. YOLO boxes can sit
        partly outside the image; negative indices would wrap around and hand
        dlib a nonsense region."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def recognize_faces(self, frame, tracked_persons):
        detections = tracked_persons or []
        if frame is None:
            return detections

        # Left-to-right ordering gives a stable slot per person between frames,
        # so cached names line up with the right detection on skipped passes.
        persons = sorted(
            (d for d in detections if d.get("class_name") == "person"),
            key=lambda d: d.get("bbox", (0, 0, 0, 0))[0],
        )
        if not persons:
            return detections

        now = time.monotonic()
        due = (now - self._last_run) >= self.min_interval
        cache_fresh = (now - self._cached_at) <= self.cache_ttl
        usable_cache = cache_fresh and len(self._cached_names) == len(persons)

        if due or not usable_cache:
            self._last_run = now
            names = []
            for person in persons:
                region = self._crop(frame, person.get("bbox", (0, 0, 0, 0)))
                if region is None or region.size == 0:
                    names.append("unknown")
                    continue
                names.append(self.encoder.recognize(region))
            names = self._apply_sticky(names, now)
            self._cached_names = names
            self._cached_at = now
        else:
            names = self._cached_names

        for person, name in zip(persons, names):
            self._apply_identity(person, name)

        # Any non-person detection keeps whatever it had; persons are updated
        # in place above, so the original list is what callers want back.
        return detections

    def apply_known(self, detections):
        """Label people from what is ALREADY known. Never runs dlib.

        The cheap half of recognition. recognize_faces() does the expensive
        encoding and updates the cache; this just reads it, so the thread
        publishing detections never has to block on a face encode.

        Costs microseconds, so it is safe to call on every published frame.
        """
        detections = detections or []
        persons = sorted(
            (d for d in detections if d.get("class_name") == "person"),
            key=lambda d: d.get("bbox", (0, 0, 0, 0))[0],
        )
        if not persons:
            return detections

        now = time.monotonic()
        names = None
        # BUG fixed here: this used to accept the cache by sticky_seconds
        # (60s) rather than cache_ttl (3s) whenever the person COUNT matched.
        # Positional names are assigned left-to-right in recognize_faces() —
        # if the actual people at those positions change while the count
        # stays the same (one leaves, a different enrolled person arrives; or
        # two people swap places), this branch reused the OLD name for
        # whoever is now standing in that slot, for up to 60 seconds instead
        # of at most one recognition interval. That silently mislabels a
        # second person as the first, which is what fed current_speaker and
        # then attributed their conversation to the wrong profile's memory.
        # cache_ttl is the correct freshness window: it is how long a real
        # recognize_faces() pass is trusted before being considered stale,
        # which is exactly the question being asked here. sticky_seconds
        # stays reserved for the single-person "no assignment happened at
        # all this pass" fallback below, its actual documented purpose.
        if len(self._cached_names) == len(persons) and \
                (now - self._cached_at) <= self.cache_ttl:
            names = list(self._cached_names)
        elif (len(persons) == 1 and self._sticky_name
                and (now - self._sticky_at) <= self.sticky_seconds):
            # Cache shape does not match (someone came or went) but we still
            # remember who was here, and there is exactly one candidate.
            names = [self._sticky_name]

        if names is None:
            names = ["unknown"] * len(persons)

        for person, name in zip(persons, names):
            self._apply_identity(person, name)
        return detections

    def _apply_identity(self, detection, name):
        detection["name"] = name
        if name and name != "unknown":
            profile = self.profiles.get(name)
            # `role` is what drives how the assistant speaks, so it travels with
            # the detection rather than being looked up again later.
            detection["role"] = (profile or {}).get("role")
            detection["relationship"] = (profile or {}).get("relationship")
        else:
            detection["role"] = None
            detection["relationship"] = None

    def _apply_sticky(self, names, now):
        """Hold the last confident identification through detection gaps.

        A real match always refreshes the memory. A miss inherits the remembered
        name until it expires. Only ever fills in an "unknown" — a frame that
        positively identifies somebody else is believed immediately, so this can
        never keep asserting the wrong person once the camera can see who is
        really there.
        """
        real = [n for n in names if n and n != "unknown"]

        if real:
            self._sticky_name = real[0]
            self._sticky_at = now
            return names

        if not self._sticky_name:
            return names
        if (now - self._sticky_at) > self.sticky_seconds:
            self._sticky_name = None
            return names

        # Exactly one person in view and we cannot see who: assume it is still
        # the person who was there. With several people it is ambiguous, so
        # nothing is asserted.
        if len(names) == 1:
            held = now - self._sticky_at
            print(f"[face-recog] holding identity {self._sticky_name!r} "
                  f"through a detection gap ({held:.0f}s of {self.sticky_seconds:.0f}s)")
            return [self._sticky_name]
        return names

    def invalidate_cache(self):
        """Force the next call to do a real recognition pass — used right after
        enrolling a face so the new identity shows up immediately."""
        self._last_run = 0.0
        self._cached_names = []
        self._cached_at = 0.0

    def forget_identity(self):
        """Drop the remembered identity as well as the cache. For tests, and for
        anywhere that genuinely needs a clean slate."""
        self.invalidate_cache()
        self._sticky_name = None
        self._sticky_at = 0.0

    def get_known_names(self):
        return list(self.encoder.known_names)
