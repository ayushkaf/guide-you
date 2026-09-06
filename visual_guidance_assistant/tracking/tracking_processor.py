"""Thread that converts detections into tracked objects and maps zones."""
from __future__ import annotations

import threading
import queue
import logging
from typing import List

from tracking.track_manager import TrackManager
from spatial.zone_mapper import get_zone


class TrackingProcessor(threading.Thread):
    """Consume detection lists, run tracking, map zones, and publish tracks."""

    def __init__(self, detection_queue: "queue.Queue", tracking_queue: "queue.Queue", config: dict) -> None:
        super().__init__(name="TrackingProcessor", daemon=True)
        self.log = logging.getLogger("tracking.processor")
        self.detection_queue = detection_queue
        self.tracking_queue = tracking_queue
        self.config = config
        self.track_manager = TrackManager(config)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        frame_width = int(self.config.get("camera", {}).get("width", 640))
        zones = self.config.get("zones", {})

        while not self._stop_event.is_set():
            try:
                detections = self.detection_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                tracked = self.track_manager.update(detections)

                # map zones and publish
                for t in tracked:
                    cx = t.get("center", (0, 0))[0]
                    t["zone"] = get_zone(int(cx), frame_width, zones)

                try:
                    self.tracking_queue.put_nowait(tracked)
                except queue.Full:
                    try:
                        _ = self.tracking_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.tracking_queue.put_nowait(tracked)
                    except queue.Full:
                        pass

            except Exception:
                self.log.exception("Error in tracking processor run loop")
