"""Cooldown manager to prevent message spam."""
from __future__ import annotations

import time


class CooldownManager:
    def __init__(self, config: dict) -> None:
        speech_config = config.get("speech", {})
        self.cooldowns = {
            "objects": speech_config.get("cooldown_objects", 15),
            "scene": speech_config.get("cooldown_scene", 30),
            "movement": speech_config.get("cooldown_movement", 5),
            "enter_leave": speech_config.get("cooldown_enter_leave", 3),
        }
        self.last_spoken = {}

    def can_speak(self, key: str, priority: str = "LOW") -> bool:
        """Return True if message identified by key may be spoken now."""
        now = time.time()

        if key.startswith("obstacle"):
            kind = "movement"
        elif key.startswith("enter_person"):
            kind = "enter_leave"
        elif key.startswith("move_person"):
            kind = "movement"
        elif key.startswith("new_obj"):
            kind = "objects"
        elif key.startswith("leave"):
            kind = "enter_leave"
        elif key.startswith("scene_summary"):
            kind = "scene"
        elif key.startswith("no_activity"):
            kind = "scene"
        else:
            kind = "objects"

        duration = self.cooldowns.get(kind, 10.0)

        if priority == "CRITICAL":
            duration = 3.0

        last = self.last_spoken.get(key)
        if last is None:
            return True

        elapsed = now - last
        return elapsed >= duration

    def record(self, key: str) -> None:
        """Record that a message was spoken."""
        self.last_spoken[key] = time.time()