"""Photo enrolment, run directly — no voice command needed.

    .venv\\Scripts\\python.exe visual_guidance_assistant\\photo_enrol_test.py

Why this exists: diagnosing photo enrolment through the voice command means a
mis-heard phrase looks identical to a detection failure. This runs the exact
same enrolment code with the speech step removed, so what you see is the
detector's behaviour and nothing else.

A live preview shows what the camera sees with a box drawn round any face it
finds, and the confidence score printed on it. Position the phone until you see
a green box, THEN let it sample. That single piece of feedback is usually enough
to fix the problem on the spot.

    green box   found, comfortably    -> will enrol cleanly
    amber box   found, but weak       -> will enrol with a warning
    no box      not found             -> move the phone closer, or fix the glare

Press SPACE to start sampling, or Q to quit.
"""
from __future__ import annotations

import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera.capture import CameraCapture
from config.config_loader import load_config
from profiles.profile_store import ProfileStore
from recognition import photo_prep
from recognition.face_recognizer import FaceRecognizer
from recognition.face_trainer import (
    PHOTO_ATTEMPT_FILE, PHOTO_SCORE_GOOD, PHOTO_SCORE_MIN, FaceTrainer,
)
from utils.paths import CONFIG_FILE

WINDOW = "Guide YOU - photo enrolment test  [SPACE] start   [Q] quit"


def best_box(locator, frame):
    """Best detection across the preprocessing variants, mapped back to the
    original frame so it can be drawn."""
    best = (None, 0.0, None)
    for name, transform in photo_prep.VARIANTS:
        try:
            variant = transform(frame)
        except Exception:
            continue
        scored = locator.locate_scored(variant, min_score=PHOTO_SCORE_MIN)
        for box, score in scored:
            if score <= best[1]:
                continue
            scale = photo_prep.SCALE_OF.get(name, 1.0)
            if scale != 1.0:
                box = tuple(int(round(v / scale)) for v in box)
            best = (box, score, name)
    return best


def main() -> int:
    cfg = load_config(CONFIG_FILE)
    store = ProfileStore()
    recog = FaceRecognizer(store, detector=(cfg.get("face_recognition") or {}).get("detector", "yunet"))
    trainer = FaceTrainer(recog, store, speech_queue=None, mic_listener=None)
    locator = recog.encoder.locator

    cam = CameraCapture(cfg, None)
    if cam._cap is None:
        print("Camera did not open. Check camera.index in config/settings.yaml.")
        return 1
    cam.start()
    time.sleep(1.5)

    print("=" * 74)
    print("  Hold the photo up to the camera.")
    print("  A GREEN box means it is found clearly. AMBER means weak but usable.")
    print("  Press SPACE when you see a box, or Q to quit.")
    print("=" * 74)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    start = None

    while True:
        frame = cam.get_latest_frame()
        if frame is None:
            time.sleep(0.02)
            continue

        shown = frame.copy()
        box, score, variant = best_box(locator, frame)

        if box is not None:
            top, right, bottom, left = box
            good = score >= PHOTO_SCORE_GOOD
            colour = (0, 200, 0) if good else (0, 190, 255)
            cv2.rectangle(shown, (left, top), (right, bottom), colour, 2)
            label = f"{score:.2f} ({variant})" + ("" if good else "  WEAK")
            cv2.putText(shown, label, (left, max(top - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        else:
            stats = photo_prep.describe_frame(frame)
            cv2.putText(shown, "no face found", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(shown, photo_prep.advice_for(stats)[:64], (12, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\nquit without enrolling")
            cam.stop()
            cv2.destroyAllWindows()
            return 0
        if key == ord(" "):
            start = cam.get_latest_frame()
            break

    cv2.destroyAllWindows()
    print("\n" + "=" * 74)
    print("  SAMPLING - keep holding the photo steady")
    print("=" * 74)

    ok = trainer.capture_known_person(start, camera=cam)
    cam.stop()

    print("\n" + "=" * 74)
    print("  RESULT")
    print("=" * 74)
    print(f"  enrolled: {ok}")
    print(f"  frame saved at: {PHOTO_ATTEMPT_FILE}")

    if ok:
        for p in store.all_profiles():
            print(f"\n  name         : {p['name']!r}")
            print(f"  relationship : {p['relationship']!r}")
            print(f"  role         : {p['role']!r}")
            print(f"  face samples : {len(p.get('face_encodings') or [])}")
        print("\n  Now hold the same photo up and run the app, then say "
              "\"who is that\".")
    else:
        print("\n  Open the saved frame above and look at it — if the face is not")
        print("  clearly visible there, the camera never saw it, and no amount of")
        print("  detector tuning will help.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
