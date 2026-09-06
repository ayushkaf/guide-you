"""Convert Ultralytics Results into application detection dictionaries."""
from __future__ import annotations

from typing import List


def parse_detections(results, allowed_classes: list[str], conf_threshold: float) -> List[dict]:
    """Parse Ultralytics detection results into a list of dictionaries."""
    detections: List[dict] = []

    for result in results:
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        xyxy = getattr(boxes, "xyxy", None)
        confidences = getattr(boxes, "conf", None)
        class_ids = getattr(boxes, "cls", None)

        if xyxy is None or confidences is None or class_ids is None:
            continue

        xyxy_list = xyxy.tolist()
        conf_list = confidences.tolist()
        cls_list = class_ids.tolist()

        for bbox, score, class_id in zip(xyxy_list, conf_list, cls_list):
            if score < conf_threshold:
                continue

            class_index = int(class_id)
            class_name = str(names.get(class_index, str(class_index)))
            if allowed_classes and class_name not in allowed_classes:
                continue

            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))

            detections.append(
                {
                    "class_name": class_name,
                    "confidence": float(score),
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                }
            )

    return detections
