"""Locating faces, separated from encoding them.

These are two different jobs and they were failing for different reasons. dlib's
128-d encoding is accurate and is what the enrolled samples are built from, so
that stays. But dlib's HOG *detector* could not find the face in 9 frames out of
10 on this camera — sharp, well-lit frames, simply not frontal enough for it.
Measured, per frame:

    HOG          located a face in  1/10   ~0.2-0.6s
    dlib CNN     located a face in  5/5    72-78s   <- accurate, unusable
    YuNet (here) see the benchmark in the README

So detection is pluggable. YuNet is a small ONNX model run through OpenCV's DNN
module: fast, and far more tolerant of head angle than HOG.

Falling back is a supported path, not an error case. A device deployed offline
without the model file still works — it just locates faces less reliably, and
says so once at startup rather than failing mysteriously later.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.paths import YUNET_MODEL_FILE

DETECTOR_YUNET = "yunet"
DETECTOR_HOG = "hog"

# YuNet keeps boxes tight to the facial features. dlib's encoder was trained on
# its own detector's slightly looser crops, so the box is padded outward before
# encoding — without this the encodings drift and stop matching.
YUNET_BOX_PADDING = 0.18

# Below this score YuNet's proposals are mostly noise. Used for live faces.
YUNET_SCORE_THRESHOLD = 0.75

# The floor the detector itself is set to when a caller wants to filter by its
# own, lower bar — photo-of-a-screen enrolment does. Set low so the raw scores
# are visible; the caller decides what to accept.
YUNET_FLOOR_THRESHOLD = 0.20

YUNET_NMS_THRESHOLD = 0.3

Box = Tuple[int, int, int, int]  # (top, right, bottom, left), face_recognition's order


class FaceLocator:
    """Finds faces. Returns boxes in face_recognition's (top, right, bottom, left)."""

    def __init__(self, preferred: str = DETECTOR_YUNET, model_path: Optional[str] = None):
        self.requested = (preferred or DETECTOR_YUNET).strip().lower()
        self.model_path = model_path or YUNET_MODEL_FILE
        self._yunet = None
        self._yunet_size = None
        self._yunet_score_floor = None
        self.active = DETECTOR_HOG

        if self.requested == DETECTOR_YUNET:
            self.active = DETECTOR_YUNET if self._init_yunet() else DETECTOR_HOG
        elif self.requested != DETECTOR_HOG:
            print(f"[face-detect] unknown detector {self.requested!r}; using hog")

        print(f"[face-detect] detector: {self.active}"
              + (f" (requested {self.requested})" if self.active != self.requested else ""))

    # ---------- setup ----------

    def _init_yunet(self) -> bool:
        if not hasattr(cv2, "FaceDetectorYN"):
            print("[face-detect] WARNING: this OpenCV build has no FaceDetectorYN; "
                  "falling back to the hog detector")
            return False
        if not os.path.exists(self.model_path):
            # One clear line. This is the offline-deployment case and it is fine.
            print(f"[face-detect] WARNING: YuNet model not found at {self.model_path} — "
                  "falling back to the hog detector. Run fetch_models.py to install it "
                  "(one-time, ~230KB); see README.")
            return False
        try:
            self._yunet = cv2.FaceDetectorYN.create(
                self.model_path, "", (320, 320),
                YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD, 5000,
            )
            self._yunet_size = (320, 320)
            self._yunet_score_floor = YUNET_SCORE_THRESHOLD
            return True
        except Exception as exc:
            print(f"[face-detect] WARNING: could not start YuNet "
                  f"({type(exc).__name__}: {exc}); falling back to the hog detector")
            return False

    # ---------- detection ----------

    @staticmethod
    def _pad(box: Box, width: int, height: int, factor: float) -> Box:
        top, right, bottom, left = box
        pad_y = int((bottom - top) * factor)
        pad_x = int((right - left) * factor)
        return (
            max(0, top - pad_y),
            min(width, right + pad_x),
            min(height, bottom + pad_y),
            max(0, left - pad_x),
        )

    def _locate_yunet(self, bgr: np.ndarray, min_score: float = None) -> List[Box]:
        return [box for box, _ in self._locate_yunet_scored(bgr, min_score)]

    def _locate_yunet_scored(self, bgr: np.ndarray, min_score: float = None):
        """Boxes WITH their confidence scores.

        YuNet returns a score per detection in column 14. It was being thrown
        away, which made "no face found" impossible to tell apart from "a face
        at 0.62 that just missed the threshold" — exactly the distinction that
        matters when a photo of a screen is the input.
        """
        height, width = bgr.shape[:2]
        if self._yunet_size != (width, height):
            self._yunet.setInputSize((width, height))
            self._yunet_size = (width, height)

        # Ask the detector for everything, then filter here, so a caller can
        # use a lower bar without rebuilding the net.
        floor = YUNET_FLOOR_THRESHOLD if min_score is not None else YUNET_SCORE_THRESHOLD
        if self._yunet_score_floor != floor:
            self._yunet.setScoreThreshold(floor)
            self._yunet_score_floor = floor

        _, faces = self._yunet.detect(bgr)
        if faces is None:
            return []

        cutoff = YUNET_SCORE_THRESHOLD if min_score is None else min_score
        results = []
        for face in faces:
            x, y, w, h = (int(round(v)) for v in face[:4])
            score = float(face[14]) if len(face) > 14 else 1.0
            if w <= 0 or h <= 0 or score < cutoff:
                continue
            box = (y, x + w, y + h, x)
            results.append((self._pad(box, width, height, YUNET_BOX_PADDING), score))
        return results

    def locate_scored(self, bgr: np.ndarray, min_score: float = None):
        """Face boxes with confidence. YuNet only — HOG has no score to give."""
        if bgr is None or bgr.size == 0 or self.active != DETECTOR_YUNET:
            return []
        try:
            return self._locate_yunet_scored(bgr, min_score)
        except Exception as exc:
            print(f"[face-detect] scored detection failed: {type(exc).__name__}: {exc}")
            return []

    @staticmethod
    def _locate_hog(bgr: np.ndarray) -> List[Box]:
        import face_recognition

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # upsample=0, not the library default of 1. Measured on this camera,
        # upsample=1 at 1280x720 found the face in 1 frame of 4 where upsample=0
        # found it in 4 of 4, and was four times slower: a large close-up face
        # gets doubled past what the HOG cascade scans for.
        return face_recognition.face_locations(rgb, number_of_times_to_upsample=0)

    def locate(self, bgr: np.ndarray) -> List[Box]:
        """Face boxes in a BGR frame. Never raises — an empty list means none."""
        if bgr is None or bgr.size == 0:
            return []
        try:
            if self.active == DETECTOR_YUNET:
                boxes = self._locate_yunet(bgr)
                if boxes:
                    return boxes
                # YuNet found nothing. HOG occasionally catches a pose YuNet
                # misses, and at this point we have already paid the cheap
                # detector, so the fallback is worth the extra few hundred ms.
                return self._locate_hog(bgr)
            return self._locate_hog(bgr)
        except Exception as exc:
            print(f"[face-detect] detection failed: {type(exc).__name__}: {exc}")
            return []
