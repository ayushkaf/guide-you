"""Zone mapping utilities."""
from __future__ import annotations

from typing import Dict


def _parse_range(value: str) -> tuple[float, float]:
    """Parse a range string into (start, end) floats in 0..1.

    Accepts forms like "0-33%", "33-66%", or "0-0.33".
    """
    if isinstance(value, (int, float)):
        v = float(value)
        return v, v

    s = str(value).strip()
    if "%" in s:
        s = s.replace("%", "")
        a, b = s.split("-")
        return float(a) / 100.0, float(b) / 100.0

    a, b = s.split("-")
    return float(a), float(b)


def get_zone(center_x: int, frame_width: int, zones: Dict) -> str:
    """Return 'left', 'center', or 'right' based on center_x and zones config.

    zones is expected to be a dict with keys 'left','center','right' and
    values describing ranges (e.g., "0-33%" or "0-0.33").
    """
    if frame_width <= 0:
        return "unknown"

    # compute normalized x
    nx = float(center_x) / float(frame_width)

    try:
        left_range = _parse_range(zones.get("left"))
        center_range = _parse_range(zones.get("center"))
        right_range = _parse_range(zones.get("right"))
    except Exception:
        # fallback to thirds
        left_range = (0.0, 1.0 / 3.0)
        center_range = (1.0 / 3.0, 2.0 / 3.0)
        right_range = (2.0 / 3.0, 1.0)

    if left_range[0] <= nx <= left_range[1]:
        return "left"
    if center_range[0] <= nx <= center_range[1]:
        return "center"
    if right_range[0] <= nx <= right_range[1]:
        return "right"

    # if none matched, choose nearest
    mid = nx
    if mid < 1.0 / 3.0:
        return "left"
    if mid < 2.0 / 3.0:
        return "center"
    return "right"
