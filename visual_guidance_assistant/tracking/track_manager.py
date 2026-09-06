"""High-level track manager that detects enter/leave and movement events."""
from __future__ import annotations

from typing import Dict, List
from tracking.tracker import CentroidTracker
from spatial.movement_analyzer import analyze_movement


class TrackManager:
    """Wraps CentroidTracker and adds event detection and zone placeholder.

    Returned track dicts will include `zone` and `event` fields.
    """

    def __init__(self, config: Dict) -> None:
        tcfg = config.get("tracking", {})
        self.tracker = CentroidTracker(max_age=tcfg.get("max_age", 30), min_hits=tcfg.get("min_hits", 2))
        self._prev_ids = set()
        self._prev_centers: Dict[int, tuple] = {}

    def update(self, detections: List[dict]) -> List[dict]:
        tracked = self.tracker.update(detections)

        current_ids = set(t["track_id"] for t in tracked)
        entered = current_ids - self._prev_ids
        left = self._prev_ids - current_ids

        results = []
        for t in tracked:
            tid = t["track_id"]
            prev_center = self._prev_centers.get(tid)
            curr_center = t["center"]
            movement = analyze_movement(prev_center, curr_center)

            event = "stationary"
            if tid in entered:
                event = "entered"
            elif tid in left:
                event = "left"
            else:
                event = movement

            results.append(
                {
                    **t,
                    "zone": "unknown",
                    "event": event,
                }
            )

        # update history
        self._prev_ids = current_ids
        self._prev_centers = {t["track_id"]: t["center"] for t in tracked}

        return results
