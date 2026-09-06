"""One-time download of the YuNet face detector model.

    .venv\\Scripts\\python.exe visual_guidance_assistant\\fetch_models.py

Run this ONCE during setup. The app never downloads anything at runtime — a care
device should not depend on the internet being up when someone walks into the
room, and it should not quietly reach out to a server on every launch.

If the file is missing the app logs one line and falls back to dlib's HOG
detector, which needs no model file. Everything still works, just less reliably.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

from utils.paths import MODELS_DIR, YUNET_MODEL_FILE

# opencv_zoo is the OpenCV project's own model repository.
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def fetch(url: str, destination: str) -> bool:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    print(f"downloading {os.path.basename(destination)}")
    print(f"  from {url}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "guide-you-setup"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return False

    if len(data) < 10_000:
        print(f"  FAILED: got only {len(data)} bytes, that is not the model")
        return False

    with open(destination, "wb") as handle:
        handle.write(data)

    digest = hashlib.sha256(data).hexdigest()
    print(f"  saved {len(data):,} bytes to {destination}")
    print(f"  sha256 {digest}")
    return True


def main() -> int:
    print("=" * 72)
    print("Guide YOU — one-time model download")
    print("=" * 72)

    if os.path.exists(YUNET_MODEL_FILE):
        size = os.path.getsize(YUNET_MODEL_FILE)
        print(f"already present: {YUNET_MODEL_FILE} ({size:,} bytes)")
        print("delete it and re-run to fetch again.")
        return 0

    ok = fetch(YUNET_URL, YUNET_MODEL_FILE)
    if not ok:
        print("\nThe app will still run — it falls back to the HOG detector, which")
        print("needs no model file but locates faces far less reliably.")
        return 1

    print(f"\nmodels directory: {MODELS_DIR}")
    print("done.")
    return 0


if __name__ == "__main__":
    # Allow running this file directly from the project root.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
