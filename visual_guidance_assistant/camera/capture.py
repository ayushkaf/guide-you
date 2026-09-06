"""Threaded camera capture that stores the latest frame and exposes JPEG bytes."""
from __future__ import annotations

import threading
import time
import logging
import queue
from typing import Optional

import cv2
import numpy as np

_cv2_log_level_lock = threading.Lock()


class CameraCapture(threading.Thread):
    """Threaded webcam capture.

    Args:
        config: Dictionary containing camera settings (index, width, height, fps_limit,
            detection_skip).
        frame_queue: Optional queue.Queue to which frames may be enqueued.
    """

    def __init__(self, config: dict, frame_queue: "queue.Queue[np.ndarray]" = None) -> None:
        super().__init__(name="CameraCapture", daemon=True)
        self.log = logging.getLogger("camera.capture")
        self.config = config
        self.frame_queue = frame_queue

        camera_cfg = config.get("camera", {})
        self.index = int(camera_cfg.get("index", 0))
        self.requested_width = int(camera_cfg.get("width", 640))
        self.requested_height = int(camera_cfg.get("height", 480))
        # Filled in from the driver after negotiation — the camera decides what
        # it will actually deliver, and forcing frames to the *requested* size
        # would just upscale (wasting CPU and inventing detail that isn't there).
        self.width = self.requested_width
        self.height = self.requested_height
        self.fps_limit = float(camera_cfg.get("fps_limit", 30))
        self.detection_skip = int(camera_cfg.get("detection_skip", 2))
        # MJPG is what makes 1080p reachable: uncompressed YUY2 needs far more
        # USB bandwidth than USB 2.0 has, so the driver silently caps the
        # resolution instead. Set to "" / null in config to skip forcing it.
        self.fourcc = camera_cfg.get("fourcc", "MJPG")

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame_count = 0

        self._open_camera()

    @staticmethod
    def list_available_cameras(max_index: int = 10) -> list:
        """Probe indices 0..max_index-1 and return the ones that actually open."""
        # Non-existent indices log a harmless "can't be used to capture by
        # index" warning per probe; silence it for the duration of the scan.
        # Guarded by a lock since cv2's log level is process-global state.
        with _cv2_log_level_lock:
            previous_log_level = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
            try:
                available = []
                for index in range(max_index):
                    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                    if cap is not None and cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        available.append((index, width, height))
                    if cap is not None:
                        cap.release()
                return available
            finally:
                cv2.utils.logging.setLogLevel(previous_log_level)

    def _open_camera(self) -> None:
        scan_limit = 10
        available = self.list_available_cameras(max_index=scan_limit)
        print(f"[camera-debug] available camera indices: {available}")

        available_indices = [index for index, _, _ in available]
        chosen_index = self.index
        if chosen_index not in available_indices:
            if available_indices:
                chosen_index = available_indices[-1]
                print(
                    f"[camera-debug] WARNING: configured camera.index={self.index} not available; "
                    f"falling back to index {chosen_index} (last one detected, usually the most "
                    "recently attached USB camera)"
                )
            else:
                self.log.warning("No camera detected at all (tried indices 0-%d)", scan_limit - 1)
                self._cap = None
                return
        self.index = chosen_index

        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._cap or not self._cap.isOpened():
            self.log.warning("Failed to open camera index %s", self.index)
            self._cap = None
            return

        print(f"[camera-debug] VideoCapture opened: backend=CAP_DSHOW index={self.index}")
        print(f"[camera-debug] before negotiation: "
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
              f"({self._fourcc_str()}) @ {self._cap.get(cv2.CAP_PROP_FPS)}fps")

        # ORDER MATTERS: the pixel format has to be set BEFORE the resolution.
        # Ask for 1920x1080 while the driver is still in uncompressed YUY2 and
        # it will quietly hand back 640x480 — USB 2.0 doesn't have the
        # bandwidth for uncompressed 1080p. Switching to MJPG first makes the
        # high-resolution modes actually available to request.
        if self.fourcc:
            try:
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(self.fourcc)))
                print(f"[camera-debug] requested pixel format: {self.fourcc}")
            except Exception as exc:
                print(f"[camera-debug] could not set FOURCC {self.fourcc!r}: {exc}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        try:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps_limit)
        except Exception:
            pass

        # Read back what the driver actually agreed to, and trust a real frame
        # over the reported property values — some drivers report the request
        # rather than the truth.
        applied_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        applied_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, probe_frame = self._cap.read()
        if ret and probe_frame is not None:
            applied_height, applied_width = probe_frame.shape[0], probe_frame.shape[1]

        if applied_width > 0 and applied_height > 0:
            self.width, self.height = applied_width, applied_height

        print(f"[camera-debug] AFTER negotiation: {self.width}x{self.height} "
              f"({self._fourcc_str()}) @ {self._cap.get(cv2.CAP_PROP_FPS)}fps")
        if (self.width, self.height) != (self.requested_width, self.requested_height):
            print(f"[camera-debug] NOTE: requested "
                  f"{self.requested_width}x{self.requested_height} but the camera "
                  f"delivers {self.width}x{self.height}. Using the delivered size "
                  f"(upscaling to the request would add no real detail). Run "
                  f"list_cameras.py to see which modes this device supports.")

        self.log.info("Opened camera index %s (%dx%d)", self.index, self.width, self.height)

    def _fourcc_str(self) -> str:
        if self._cap is None:
            return "none"
        code = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        if not code:
            return "none"
        return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))

    def run(self) -> None:
        if self._cap is None:
            self.log.error("Camera not available; exiting capture thread.")
            return

        desired_period = 1.0 / max(1.0, self.fps_limit)

        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            with self._lock:
                self._latest_frame = frame.copy()

            if self.frame_queue is not None and self._frame_count % max(1, self.detection_skip) == 0:
                try:
                    self.frame_queue.put_nowait(self._latest_frame.copy())
                except queue.Full:
                    try:
                        _ = self.frame_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self.frame_queue.put_nowait(self._latest_frame.copy())
                    except Exception:
                        pass

            self._frame_count += 1
            elapsed = time.perf_counter() - t0
            to_sleep = desired_period - elapsed
            if to_sleep > 0:
                time.sleep(min(to_sleep, 0.1))

        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass

        self.log.info("Camera capture thread stopped")

    def stop(self) -> None:
        self._stop_event.set()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_latest_frame_bytes(self) -> Optional[bytes]:
        frame = self.get_latest_frame()
        if frame is None:
            return None

        success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success or encoded is None:
            return None
        return encoded.tobytes()
