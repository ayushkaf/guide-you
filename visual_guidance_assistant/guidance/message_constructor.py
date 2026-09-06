"""Natural language message constructors for guidance assistant."""
from __future__ import annotations

from typing import Dict, List


def build_object_message(obj: Dict) -> str:
    class_name = obj.get("class_name", "object")
    zone = obj.get("zone", "")
    if zone == "center":
        return f"There is a {class_name} in front of you."
    return f"There is a {class_name} on your {zone}."


def build_movement_message(obj: Dict) -> str:
    class_name = obj.get("class_name", "something")
    zone = obj.get("zone", "")
    event = obj.get("event", "moving")
    if class_name == "person":
        return f"A person is {event} on your {zone}."
    return f"Movement detected on your {zone}."


def build_enter_message(obj: Dict) -> str:
    class_name = obj.get("class_name", "object")
    zone = obj.get("zone", "")
    if zone:
        return f"A {class_name} has entered from the {zone}."
    return f"A {class_name} has entered the room."


def build_leave_message(obj: Dict) -> str:
    class_name = obj.get("class_name", "object")
    return f"The {class_name} has left the room."


def _format_list(items: List[str]) -> str:
    items = [str(i) for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_scene_summary(tracks_by_zone: Dict[str, List]) -> str:
    parts = []
    zone_phrases = {
        "left": "on the left",
        "center": "in front of you",
        "right": "on the right",
    }
    for zone in ("left", "center", "right"):
        items = tracks_by_zone.get(zone, [])
        if not items:
            continue
        # Extract class names, remove duplicates
        names = []
        seen = set()
        for item in items:
            if isinstance(item, dict):
                name = item.get("class_name", "object")
            else:
                name = str(item)
            if name not in seen:
                names.append(name)
                seen.add(name)
        if not names:
            continue
        formatted = _format_list(names)
        parts.append(f"{formatted} {zone_phrases.get(zone, '')}".strip())

    if not parts:
        return "No objects detected."
    if len(parts) == 1:
        return f"There is {parts[0]}."
    if len(parts) == 2:
        return f"There is {parts[0]} and {parts[1]}."
    return f"There is {parts[0]}, {parts[1]}, and {parts[2]}."


def build_no_activity_message() -> str:
    return "No activity detected."


def build_obstacle_message(zone: str) -> str:
    opposite = {"left": "right", "right": "left", "center": "right"}
    opp = opposite.get(zone, "right")
    return f"Obstacle ahead. Move slightly {opp}."


def build_new_object_message(obj: Dict) -> str:
    """Announce a newly appeared object."""
    class_name = obj.get("class_name", "object")
    zone = obj.get("zone", "center")
    zone_text = "in front of you" if zone == "center" else f"on your {zone}"
    return f"There is now a {class_name} {zone_text}."