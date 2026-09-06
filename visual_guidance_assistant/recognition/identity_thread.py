"""Face recognition on its own low-priority thread.

WHY
dlib's face encoding is the single most expensive thing this app does, and it
was running inside DetectionProcessor — the same thread that publishes
detections. Every encode therefore stalled the whole detection pipeline, and
because dlib saturates all cores it starved the camera thread too. Measured
worst frame stalls were 1.0-1.8s even after the resolution and YOLO fixes.

Moving it here changes what a slow encode blocks. DetectionProcessor now calls
FaceRecognizer.apply_known(), which reads the cache in microseconds, and this
thread does the encoding off to one side. A slow encode delays a NAME, not the
video.

THREAD PRIORITY
Set to lowest on Windows, so the camera and detection threads win any contest
for a core. A name arriving a moment later is invisible; a stuttering picture is
not.

STACKS WITH, DOES NOT REPLACE
The throttle (min_interval), the cache (cache_ttl) and the 60s sticky hold all
still live in FaceRecognizer and all still apply. This only changes WHERE the
expensive call happens, so identity stays just as stable across detection gaps.
"""
from __future__ import annotations

import copy
import sys
import threading
import time


def _set_lowest_priority() -> str:
    """Best-effort. Never fatal: a wrong priority is a performance detail."""
    if not sys.platform.startswith("win"):
        return "not applied (not Windows)"
    try:
        import ctypes
        from ctypes import wintypes

        THREAD_PRIORITY_LOWEST = -2
        kernel32 = ctypes.windll.kernel32

        # restype MUST be declared. GetCurrentThread returns a HANDLE, which is
        # 64-bit here; ctypes defaults to a 32-bit int return and silently
        # truncates it, so SetThreadPriority is handed a garbage handle and
        # fails. That is exactly what happened the first time this was written.
        kernel32.GetCurrentThread.restype = wintypes.HANDLE
        kernel32.GetCurrentThread.argtypes = []
        kernel32.SetThreadPriority.restype = wintypes.BOOL
        kernel32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]

        handle = kernel32.GetCurrentThread()
        if not kernel32.SetThreadPriority(handle, THREAD_PRIORITY_LOWEST):
            return f"failed (win error {ctypes.get_last_error()})"
        return "lowest"
    except Exception as exc:
        return f"failed ({type(exc).__name__}: {exc})"


class IdentityProcessor(threading.Thread):
    def __init__(self, camera, detection_store, face_recognizer,
                 interval: float = 2.0, low_priority: bool = True):
        super().__init__(name="IdentityProcessor", daemon=True)
        self.camera = camera
        self.detection_store = detection_store
        self.face_recognizer = face_recognizer
        self.interval = float(interval)
        self.low_priority = low_priority
        self._stop_event = threading.Event()
        self.passes = 0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        priority = _set_lowest_priority() if self.low_priority else "normal"
        print(f"[identity] face recognition thread started "
              f"(every {self.interval:.1f}s, priority {priority})")

        while not self._stop_event.is_set():
            # Wait first: at startup the camera and model are still warming up.
            if self._stop_event.wait(self.interval):
                break

            frame = self.camera.get_latest_frame() if self.camera else None
            detections = self.detection_store.get([]) if self.detection_store else []
            if frame is None or not detections:
                continue
            if not any(d.get("class_name") == "person" for d in detections):
                continue

            try:
                # Deep-copied so the expensive call cannot mutate the list other
                # threads are reading. What we actually want out of this is the
                # updated cache and sticky state inside FaceRecognizer, which
                # apply_known() then reads on the publishing thread.
                self.face_recognizer.recognize_faces(frame, copy.deepcopy(detections))
                self.passes += 1
            except Exception as exc:
                print(f"[identity] recognition failed: {type(exc).__name__}: {exc}")
