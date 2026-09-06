"""Simple FPS counter using a sliding window of deltas."""
from collections import deque
import time
from typing import Deque


class FPSCounter:
    """Track instantaneous FPS using recent frame timestamps.

    Example:
        fps = FPSCounter()
        while running:
            # at end of each frame
            fps.tick()
            print(fps.fps)
    """

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = int(window_size)
        self._deltas: Deque[float] = deque(maxlen=self._window_size)
        self._last_time = None

    def tick(self) -> None:
        """Call once per frame to update internal timing window."""
        now = time.perf_counter()
        if self._last_time is None:
            # Seed the timer on first call
            self._last_time = now
            return

        delta = now - self._last_time
        # protect against zero or negative deltas
        if delta <= 0:
            delta = 1e-6

        self._deltas.append(delta)
        self._last_time = now

    @property
    def fps(self) -> float:
        """Estimated frames-per-second over the window. Returns 0.0 if
        not enough samples yet.
        """
        if not self._deltas:
            return 0.0

        total = sum(self._deltas)
        if total <= 0:
            return 0.0

        return len(self._deltas) / total
