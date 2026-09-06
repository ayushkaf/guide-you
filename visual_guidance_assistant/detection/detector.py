"""YOLO-based object detection interface."""
from __future__ import annotations

import os
from typing import List, Dict
import numpy as np
from ultralytics import YOLO

from detection.results_parser import parse_detections
from utils.paths import resource_path


class ObjectDetector:
    """Wrapper around YOLO model inference."""

    def __init__(self, config: dict) -> None:
        model_path = config["model"].get("path") or config["model"].get("model_path")
        if not model_path:
            raise ValueError("YOLO model path is not configured")
        # The weights sit next to the package. Resolving relative to the package
        # (not the launch directory) stops ultralytics from silently
        # re-downloading them into whatever folder the app was started from.
        if not os.path.isabs(model_path):
            packaged = resource_path(model_path)
            if os.path.exists(packaged):
                model_path = packaged
        self.confidence = float(config["model"].get("confidence", config["model"].get("confidence_threshold", 0.25)))
        self.allowed_classes = list(config["model"].get("classes", config["model"].get("allowed_classes", [])))
        # YOLO letterboxes every frame into a fixed square before inference, so
        # this — not the camera resolution — is what sets the compute cost.
        # Measured at 480p source with face recognition running: default (640)
        # gave 6.1-6.8 fps, imgsz=480 gave 11.7-16.6. Do not drop to 320: it
        # stops detecting people entirely (verified — a person in clear view
        # returned no detections at all).
        self.imgsz = int(config["model"].get("imgsz", 480))
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLO inference and return parsed detections."""
        results = self.model(frame, conf=self.confidence, imgsz=self.imgsz, verbose=False)
        return parse_detections(results, self.allowed_classes, self.confidence)

    def predict(self, frame: np.ndarray) -> List[Dict]:
        return self.detect(frame)
