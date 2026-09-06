"""Simple centroid-based tracker implementation."""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


class CentroidTracker:
    """A lightweight centroid tracker using nearest-neighbor matching."""

    def __init__(self, max_age: int = 30, min_hits: int = 2) -> None:
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self._tracks: Dict[int, Dict] = {}
        self._next_id = 1
        self._frame_idx = 0

    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def update(self, detections: List[Dict]) -> List[Dict]:
        self._frame_idx += 1

        # If no detections, just age tracks and return what's left
        if len(detections) == 0:
            remove_ids = [tid for tid, t in self._tracks.items() 
                         if (self._frame_idx - t["last_seen"]) > self.max_age]
            for tid in remove_ids:
                del self._tracks[tid]
            return self._get_active_tracks()

        det_centers = [tuple(d["center"]) for d in detections]

        # First frame: create tracks but DON'T return them yet (min_hits check)
        if len(self._tracks) == 0:
            for det, center in zip(detections, det_centers):
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "bbox": det["bbox"],
                    "center": center,
                    "class_name": det.get("class_name", ""),
                    "confidence": float(det.get("confidence", 0.0)),
                    "last_seen": self._frame_idx,
                    "frames_seen": 1,
                    "prev_center": None,
                }
            return []  # Don't return tracks until they hit min_hits

        track_ids = list(self._tracks.keys())
        track_centers = [self._tracks[tid]["center"] for tid in track_ids]

        matched_tracks = set()
        matched_detections = set()

        # Greedy matching with distance threshold 80 pixels (more forgiving)
        for d_idx, det in enumerate(detections):
            center = det_centers[d_idx]
            best_tid = None
            best_dist = float("inf")
            for tid, t_center in zip(track_ids, track_centers):
                if tid in matched_tracks:
                    continue
                dist = self._distance(center, t_center)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None and best_dist <= 80.0:
                t = self._tracks[best_tid]
                t["prev_center"] = t["center"]
                t["center"] = center
                t["bbox"] = det["bbox"]
                t["class_name"] = det.get("class_name", t.get("class_name"))
                t["confidence"] = float(det.get("confidence", t.get("confidence", 0.0)))
                t["last_seen"] = self._frame_idx
                t["frames_seen"] += 1
                matched_tracks.add(best_tid)
                matched_detections.add(d_idx)

        # Create tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx in matched_detections:
                continue
            center = det_centers[idx]
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                "bbox": det["bbox"],
                "center": center,
                "class_name": det.get("class_name", ""),
                "confidence": float(det.get("confidence", 0.0)),
                "last_seen": self._frame_idx,
                "frames_seen": 1,
                "prev_center": None,
            }

        # Remove stale tracks
        remove_ids = [tid for tid, t in self._tracks.items() 
                     if (self._frame_idx - t["last_seen"]) > self.max_age]
        for tid in remove_ids:
            del self._tracks[tid]

        return self._get_active_tracks()

    def _get_active_tracks(self) -> List[Dict]:
        """Return only tracks that have met min_hits requirement."""
        tracked = []
        for tid, t in self._tracks.items():
            if t["frames_seen"] >= self.min_hits:
                tracked.append({
                    "track_id": int(tid),
                    "class_name": t["class_name"],
                    "bbox": t["bbox"],
                    "center": t["center"],
                    "confidence": float(t.get("confidence", 0.0)),
                    "frames_seen": int(t.get("frames_seen", 0)),
                })
        return tracked