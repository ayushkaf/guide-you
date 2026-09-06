from __future__ import annotations

from collections import Counter
from typing import Iterable


def _get_zone_label(center_x: int, frame_width: int) -> str:
    if frame_width <= 0:
        return "center"

    relative_x = center_x / frame_width
    if relative_x < 0.33:
        return "left"
    if relative_x > 0.66:
        return "right"
    return "center"


def _get_size_label(area_ratio: float) -> str:
    if area_ratio >= 0.30:
        return "large"
    if area_ratio >= 0.10:
        return "medium"
    return "small"


def build_scene_description(detections: Iterable[dict], frame_width: int, frame_height: int) -> str:
    """Turn YOLO detections into a concise scene description."""
    if not detections:
        return "I do not see any recognizable objects in the scene right now."

    items = []
    zone_counts: dict[str, Counter[str]] = {"left": Counter(), "center": Counter(), "right": Counter()}
    object_sizes: dict[str, str] = {}

    for detection in detections:
        class_name = detection.get("class_name", "object")
        if class_name == "person":
            recognized_name = detection.get("name")
            if recognized_name and recognized_name != "unknown":
                class_name = recognized_name
        bbox = detection.get("bbox", (0, 0, 0, 0))
        x1, y1, x2, y2 = bbox
        area = max(0, (x2 - x1) * (y2 - y1))
        frame_area = max(1, frame_width * frame_height)
        size_label = _get_size_label(area / frame_area)
        zone = _get_zone_label(detection.get("center", (0, 0))[0], frame_width)

        items.append((class_name, zone, size_label))
        zone_counts[zone][class_name] += 1
        existing_size = object_sizes.get(class_name)
        if existing_size is None or size_label == "large":
            object_sizes[class_name] = size_label

    descriptions = []
    for zone, counter in zone_counts.items():
        if not counter:
            continue

        parts = []
        for class_name, count in counter.items():
            if class_name[0].isupper():
                # A recognized person's name, not a generic object class —
                # no article or pluralization.
                parts.append(class_name)
            elif count == 1:
                parts.append(f"a {class_name}")
            else:
                parts.append(f"{count} {class_name}s")

        descriptions.append(f"{', '.join(parts)} on the {zone}")

    if descriptions:
        scene_text = "There is " + ", and there is ".join(descriptions) + "."
    else:
        scene_text = "There are objects in the scene, but I cannot describe their arrangement precisely."

    size_notes = []
    for class_name, size_label in object_sizes.items():
        label = class_name if class_name[0].isupper() else f"A {class_name}"
        if size_label == "large":
            size_notes.append(f"{label} appears very prominent in view")
        elif size_label == "small":
            size_notes.append(f"{label} appears relatively small")

    if size_notes:
        scene_text += " " + ". ".join(size_notes) + "."

    return scene_text
