"""Analyze movement between previous and current centers."""
from __future__ import annotations

import math
from typing import Optional, Tuple


def analyze_movement(prev_center: Optional[Tuple[int, int]], curr_center: Optional[Tuple[int, int]], threshold: int = 30) -> str:
    """Return 'moving' if distance > threshold else 'stationary'.

    If prev_center is None, consider the object as 'moving' (it just appeared).
    """
    if prev_center is None or curr_center is None:
        return "moving"

    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    dist = math.hypot(dx, dy)
    return "moving" if dist > threshold else "stationary"
