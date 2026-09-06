"""Camera diagnostic: map DirectShow device NAMES to the indices OpenCV uses.

Run it from the project root:

    .venv\\Scripts\\python.exe visual_guidance_assistant\\list_cameras.py

cv2.VideoCapture(N) addresses cameras by bare integer index and never tells you
which physical device N is. pygrabber enumerates the same DirectShow device
list in the same order, so position in that list is the index to put in
config/settings.yaml under camera.index.

For each index that opens, this also reports the resolution the driver actually
delivers under MJPG vs uncompressed, which is what determines whether 1080p is
reachable over the USB link.
"""
from __future__ import annotations

import cv2

# Candidate modes to probe, widest first.
PROBE_MODES = [(1920, 1080), (1280, 720), (640, 480)]


def dshow_device_names() -> list[str]:
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        print("pygrabber is not installed — run: python -m pip install pygrabber")
        print("Falling back to index-only probing (no device names).\n")
        return []
    try:
        return FilterGraph().get_input_devices()
    except Exception as exc:
        print(f"pygrabber failed to enumerate devices: {type(exc).__name__}: {exc}\n")
        return []


def fourcc_to_str(value: float) -> str:
    code = int(value)
    if not code:
        return "none"
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def probe(index: int) -> dict | None:
    """Open one camera index and report what it can actually do."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None

    info = {
        "index": index,
        "default": (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
        "default_fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
        "modes": [],
    }

    for want_w, want_h in PROBE_MODES:
        for codec in ("MJPG", "default"):
            if codec == "MJPG":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
            got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok, frame = cap.read()
            actual = f"{frame.shape[1]}x{frame.shape[0]}" if (ok and frame is not None) else "read failed"
            info["modes"].append(
                {
                    "requested": f"{want_w}x{want_h}",
                    "codec": codec,
                    "reported": f"{got_w}x{got_h}",
                    "delivered": actual,
                    "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
                }
            )

    cap.release()
    return info


def main(max_index: int = 10) -> int:
    print("=" * 72)
    print("CAMERA DIAGNOSTIC")
    print("=" * 72)

    names = dshow_device_names()
    if names:
        print("\nDirectShow devices (list position == the index for camera.index):\n")
        for i, name in enumerate(names):
            print(f"  camera.index: {i}   ->   {name!r}")
    else:
        print("\nNo device names available.")

    # Silence the harmless per-probe warnings for indices that don't exist.
    previous_log_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    try:
        print("\n" + "-" * 72)
        print("Probing which indices actually open, and what they deliver")
        print("-" * 72)
        opened = []
        for index in range(max_index):
            info = probe(index)
            if info is None:
                continue
            opened.append(info)
            label = names[index] if index < len(names) else "<name unknown>"
            print(f"\n  INDEX {index}: {label}")
            print(f"    default mode: {info['default'][0]}x{info['default'][1]} "
                  f"({info['default_fourcc']})")
            for mode in info["modes"]:
                print(f"    request {mode['requested']:>9} via {mode['codec']:<7} "
                      f"-> reported {mode['reported']:>9}  delivered {mode['delivered']:>12}  "
                      f"[{mode['fourcc']}]")
    finally:
        cv2.utils.logging.setLogLevel(previous_log_level)

    print("\n" + "=" * 72)
    if not opened:
        print("NO CAMERAS OPENED. Nothing for the app to use.")
        print("=" * 72)
        return 1

    print("SUMMARY")
    print("=" * 72)
    for info in opened:
        index = info["index"]
        label = names[index] if index < len(names) else "<name unknown>"
        best = max(
            (m for m in info["modes"] if "x" in m["delivered"]),
            key=lambda m: int(m["delivered"].split("x")[0]) if "x" in m["delivered"] else 0,
            default=None,
        )
        best_text = best["delivered"] if best else "unknown"
        print(f"  index {index}: {label}  — best delivered resolution: {best_text}")

    print("\nSet the index you want in config/settings.yaml:\n\n  camera:\n    index: <N>\n")
    if names and len(names) < 2:
        print("NOTE: only ONE camera is visible to Windows. If you expected an")
        print("      external USB webcam here, Windows is not enumerating it —")
        print("      that is a driver/connection issue, not something the app")
        print("      can work around. Check Device Manager > Cameras.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
