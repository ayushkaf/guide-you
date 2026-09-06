"""Preprocessing for photographing a photo — usually one shown on a phone.

A screen is not a face. Pointing a webcam at one adds failure modes a live face
never has:

  glare     a bright panel, or a reflection off it, blows out the highlights and
            flattens exactly the contrast a face detector keys on
  moire     the camera's sensor grid beating against the screen's pixel grid,
            producing fine interference bands over the face
  scale     the face inside the photo is a fraction of the phone, which is
            itself a fraction of the frame, so the face lands far smaller than
            a live one at the same distance
  blur      a handheld phone is never quite still

So rather than one detection attempt, the photo path tries several cheap
variants of the frame and keeps whichever scores best. Each targets one of the
above. Order matters only for speed — the plain frame is tried first because
when it works, it is the truest version of the image.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import cv2
import numpy as np

Variant = Tuple[str, Callable[[np.ndarray], np.ndarray]]


def _identity(bgr: np.ndarray) -> np.ndarray:
    return bgr


def _clahe(bgr: np.ndarray) -> np.ndarray:
    """Local contrast equalisation on the lightness channel.

    The single most useful one for screen glare: it lifts detail back out of
    washed-out regions without touching colour, and unlike a global stretch it
    does not darken the rest of the frame to compensate.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _denoise_moire(bgr: np.ndarray) -> np.ndarray:
    """Soften the fine interference banding from photographing a screen.

    A small median blur removes the high-frequency grid while leaving facial
    structure — which is much lower frequency — essentially intact.
    """
    return cv2.medianBlur(bgr, 3)


def _clahe_denoise(bgr: np.ndarray) -> np.ndarray:
    return _clahe(_denoise_moire(bgr))


def _upscale(bgr: np.ndarray) -> np.ndarray:
    """Double the image.

    A face inside a photo inside a frame is small, and YuNet has a minimum size
    it can resolve. Upscaling adds no information but does put the face into the
    range the detector actually scans.
    """
    return cv2.resize(bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def _upscale_clahe(bgr: np.ndarray) -> np.ndarray:
    return _clahe(_upscale(bgr))


# Tried in order; the best-scoring result wins, so this is not a first-match.
VARIANTS: List[Variant] = [
    ("plain", _identity),
    ("contrast", _clahe),
    ("de-moire", _denoise_moire),
    ("contrast+de-moire", _clahe_denoise),
    ("upscaled", _upscale),
    ("upscaled+contrast", _upscale_clahe),
]

# Variants that change geometry, so a box found in them must be mapped back.
SCALE_OF = {"upscaled": 2.0, "upscaled+contrast": 2.0}


def describe_frame(bgr: np.ndarray) -> dict:
    """Cheap measurements that explain WHY a frame failed, so the spoken advice
    can be specific rather than 'try again'."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    blown = float((gray >= 250).sum()) / gray.size * 100.0
    dark = float((gray <= 12).sum()) / gray.size * 100.0
    return {
        "width": width,
        "height": height,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "blown_out_pct": blown,
        "very_dark_pct": dark,
    }


def advice_for(stats: dict) -> str:
    """One specific, actionable sentence about what to change.

    Ordered by how strongly each signal indicates the cause, so the most likely
    problem is the one mentioned.
    """
    if stats["brightness"] < 55:
        return ("It looked very dark. Try turning your phone's brightness up, "
                "and keep the photo out of any shadow.")
    if stats["blown_out_pct"] > 12:
        return ("The screen was glaring. Try turning your phone's brightness "
                "down a little, or tilting it so the light doesn't reflect.")
    if stats["sharpness"] < 60:
        return ("It looked blurry. Try resting your hand on something steady, "
                "and hold the photo still for a moment.")
    if stats["contrast"] < 30:
        return ("The picture looked washed out. Try tilting the phone slightly "
                "away from the light.")
    return ("I couldn't find a face in it. Try holding the phone a little "
            "closer so the face fills more of the picture, and keep it "
            "straight on rather than at an angle.")
