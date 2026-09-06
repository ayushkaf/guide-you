"""Priority message queue for guidance messages."""
from __future__ import annotations

import queue
import time
from typing import Optional


class PriorityMessageQueue:
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

    def __init__(self) -> None:
        self._pq = queue.PriorityQueue()
        self._messages = set()  # track message texts to avoid duplicates

    def push(self, priority: int, message: str) -> None:
        if message in self._messages:
            return
        timestamp = time.time()
        self._messages.add(message)
        # priority queue sorts by smallest tuple first
        self._pq.put((priority, timestamp, message))

    def pop(self) -> Optional[str]:
        try:
            priority, ts, message = self._pq.get_nowait()
        except queue.Empty:
            return None
        self._messages.discard(message)
        return message

    def empty(self) -> bool:
        return self._pq.empty()
