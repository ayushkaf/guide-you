"""Shared availability tracking for an optional cloud service that always
has a local fallback (Azure Face, Azure TTS, ...).

Not configured -> never attempted, full stop. Configured but failing ->
back off for a cooldown rather than retrying on every single call. Without
this, a down or misconfigured cloud service would add a full network
timeout's worth of latency to every recognition frame or every spoken
reply, which in this app means every single turn of conversation. After the
cooldown it is tried again once, so a transient outage heals itself without
anyone having to restart anything.

Each integration owns one of these; nothing here talks to any network
service itself.
"""
from __future__ import annotations

import threading
import time


class ServiceHealth:
    def __init__(self, name: str, configured: bool, cooldown_seconds: float = 30.0):
        self.name = name
        # Set once at construction from whether a key/endpoint is present at
        # all. A missing key is not a "failure" to back off from — it is a
        # permanent, deliberate "never attempt this" until the .env changes
        # and the process restarts (env vars are read once at startup).
        self.configured = configured
        self.cooldown_seconds = float(cooldown_seconds)
        self._lock = threading.Lock()
        self._last_failure = 0.0
        # True once at least one call has actually succeeded. Used only for
        # the startup summary line — "configured" alone does not mean the
        # key is valid or the service is reachable.
        self.ever_succeeded = False

    def should_attempt(self) -> bool:
        if not self.configured:
            return False
        with self._lock:
            if self._last_failure == 0.0:
                return True
            return (time.time() - self._last_failure) >= self.cooldown_seconds

    def record_success(self) -> None:
        with self._lock:
            self._last_failure = 0.0
        self.ever_succeeded = True

    def record_failure(self, exc: Exception) -> None:
        with self._lock:
            already_cooling_down = self._last_failure != 0.0
            self._last_failure = time.time()
        if not already_cooling_down:
            # Only the transition into a cooldown is worth a print; repeating
            # this on every failed call for the whole cooldown window would
            # just be noise on top of whatever already logged the failure.
            print(f"[{self.name}] falling back to local for now — retrying "
                  f"the cloud service again in {self.cooldown_seconds:.0f}s "
                  f"({type(exc).__name__}: {exc})")

    def mode(self) -> str:
        """For the startup/status summary line — see app.py.

        "azure" only once a real call has actually succeeded (ever_succeeded)
        AND nothing has failed recently enough to still be in cooldown.
        Being configured is necessary but never sufficient — a present but
        invalid key must still report "local".
        """
        return "azure" if (self.ever_succeeded and self.should_attempt()) else "local"
