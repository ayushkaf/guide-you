"""A single-slot, thread-safe holder for the most recent value.

Detections used to be published to one queue.Queue that *both* the render loop
and the dialogue thread drained with get_nowait(). Every item went to whichever
thread grabbed it first, so each consumer only ever saw a random subset — stale
bounding boxes on screen, and a scene description lagging reality.

A queue is the wrong primitive here: neither consumer wants a backlog, they both
want "whatever is current". Reading does not consume, so any number of readers
can each see the freshest value.
"""
from __future__ import annotations

import threading
from typing import Any, Optional


class LatestValue:
    def __init__(self, initial: Any = None) -> None:
        self._lock = threading.Lock()
        self._value = initial
        self._version = 0

    def set(self, value: Any) -> None:
        with self._lock:
            self._value = value
            self._version += 1

    def get(self, default: Any = None) -> Any:
        with self._lock:
            return self._value if self._value is not None else default

    def get_versioned(self) -> tuple[int, Any]:
        """Value plus a monotonic version, for callers that want to know whether
        anything actually changed since they last looked."""
        with self._lock:
            return self._version, self._value
