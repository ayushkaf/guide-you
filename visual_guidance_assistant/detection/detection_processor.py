"""Threaded YOLO detection processor for hybrid scene understanding.

This thread is the single producer of detections. It runs YOLO, enriches the
result with face identities, and publishes the finished list to a LatestValue
that any number of consumers can read without stealing it from each other.

Doing recognition here (rather than in the render loop and again in the
dialogue thread) keeps dlib off the UI thread entirely and means both consumers
see the same, already-named detections.
"""
from __future__ import annotations

import queue
import threading
import time

from detection.detector import ObjectDetector


class DetectionProcessor(threading.Thread):
    def __init__(
        self,
        frame_queue=None,
        detection_store=None,
        config=None,
        face_recognizer=None,
        identity_is_threaded=False,
    ) -> None:
        super().__init__(name="DetectionProcessor", daemon=True)
        self._stop_event = threading.Event()
        self.frame_queue = frame_queue
        self.detection_store = detection_store
        self.config = config or {}
        self.face_recognizer = face_recognizer
        # When an IdentityProcessor is running, this thread must never do a face
        # encode itself — it only reads the cache that thread keeps warm.
        self.identity_is_threaded = identity_is_threaded
        self.detector = None
        self.frame_count = 0

        if (self.config.get("model") or {}).get("enabled", True):
            try:
                self.detector = ObjectDetector(self.config)
            except Exception as exc:
                print(f"DetectionProcessor warning: YOLO model failed to load: {exc}")
                self.detector = None
        else:
            print("DetectionProcessor: YOLO detection disabled via config (model.enabled=false)")

        # CameraCapture already enqueues only every Nth frame, so applying
        # detection_skip a second time here would compound to 1/N^2 of the
        # camera's frame rate. Everything that arrives on the queue is meant to
        # be processed.
        self.frame_skip = 1

    def stop(self) -> None:
        self._stop_event.set()

    def _publish(self, detections) -> None:
        if self.detection_store is not None:
            self.detection_store.set(detections)

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self.frame_count += 1
            if self.detector is None:
                continue

            try:
                detections = self.detector.predict(frame)
                if detections is None:
                    detections = []
            except Exception as exc:
                print(f"DetectionProcessor error during YOLO inference: {exc}")
                continue

            if self.face_recognizer is not None and detections:
                try:
                    if self.identity_is_threaded:
                        # Cheap: read the cache IdentityProcessor keeps warm.
                        # Encoding on this thread stalled the whole pipeline,
                        # because dlib saturates every core and starves the
                        # camera thread with it.
                        detections = self.face_recognizer.apply_known(detections)
                    else:
                        # No identity thread: do it here, internally throttled.
                        detections = self.face_recognizer.recognize_faces(frame, detections)
                except Exception as exc:
                    # Log rather than swallow: a persistent dlib/model failure
                    # used to be completely invisible.
                    print(f"DetectionProcessor: face recognition failed on this frame: "
                          f"{type(exc).__name__}: {exc}")

            if detections:
                labels = [d.get("name") or d.get("class_name", "?") for d in detections]
                print(f"[detection-debug] {len(detections)} object(s): {labels}")

            self._publish(detections)
            time.sleep(0.005)
