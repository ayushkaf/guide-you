"""Approximate local weather, for casual conversational colour only.

Deliberately its own module, separate from safety/orientation.py: orientation
answers "where am I" only from a caregiver-set home_location, deterministically,
and must never be reachable from an IP-based guess. IP geolocation can be off
by a whole city (further on VPNs or mobile networks), which is fine for "looks
like rain today" and not acceptable for an orientation question a vulnerable
person may depend on — see PART 3 of the memory/tracking review this shipped
with. This module's output is always framed as approximate and is never passed
to orientation.answer() or anywhere identity/safety-related.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import requests

# Refetched at most this often. Weather worth mentioning casually does not
# change minute to minute, and re-running two network calls on every single
# conversation turn would only add latency for no benefit.
REFRESH_SECONDS = 3 * 3600

# ipapi.co (the more commonly recommended free option) turned out to be
# persistently rate-limited (HTTP 429) from this environment during testing —
# observed on repeated attempts, not a one-off. geojs.io was verified working
# over HTTPS, with no key, from the same environment, so that's what's used.
_IP_LOCATION_URL = "https://get.geojs.io/v1/ip/geo.json"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_FETCH_TIMEOUT = 4.0

# https://open-meteo.com/en/docs — WMO weather interpretation codes, the
# subset actually likely to come up in casual conversation.
_WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy",
    45: "foggy", 48: "foggy",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
    95: "thunderstorms",
}


class WeatherContext:
    """Fetches in a background thread so a slow or unreachable network never
    adds latency to a conversation turn. A turn that starts before the first
    fetch finishes simply gets no weather context for that turn — never a
    blocked reply, and never a guess."""

    def __init__(self):
        self._lock = threading.Lock()
        self._line: Optional[str] = None
        self._fetched_at = 0.0
        self._fetching = False

    def _fetch(self) -> None:
        try:
            loc_resp = requests.get(_IP_LOCATION_URL, timeout=_FETCH_TIMEOUT)
            loc_resp.raise_for_status()
            location = loc_resp.json() or {}
            # geojs.io returns latitude/longitude as strings, not numbers.
            raw_lat, raw_lon = location.get("latitude"), location.get("longitude")
            city = location.get("city")
            if raw_lat is None or raw_lon is None:
                raise ValueError("no coordinates in IP geolocation response")
            lat, lon = float(raw_lat), float(raw_lon)

            weather_resp = requests.get(
                _WEATHER_URL,
                params={"latitude": lat, "longitude": lon, "current_weather": "true"},
                timeout=_FETCH_TIMEOUT,
            )
            weather_resp.raise_for_status()
            current = (weather_resp.json() or {}).get("current_weather") or {}
            temp_c = current.get("temperature")
            code = current.get("weathercode")
            description = _WEATHER_CODES.get(code)

            if temp_c is None and description is None:
                raise ValueError("weather API returned nothing usable")
            if description and temp_c is not None:
                detail = f"it's {description}, about {temp_c:.0f}°C"
            elif temp_c is not None:
                detail = f"it's about {temp_c:.0f}°C"
            else:
                detail = f"it's {description}"

            near = f" near {city}" if city else ""
            line = (
                f"Casual local weather context (approximate, from IP-based "
                f"location{near} — this is NOT reliable enough to state as "
                f"where the person physically is; never use it to answer "
                f"'where am I'): {detail}."
            )
            with self._lock:
                self._line = line
                self._fetched_at = time.time()
            print(f"[weather-context] fetched: {detail}{near}")
        except Exception as exc:
            print(f"[weather-context] fetch failed (non-fatal, casual "
                  f"context only, never blocks a reply): "
                  f"{type(exc).__name__}: {exc}")
            with self._lock:
                # Still stamp the attempt time, so a persistently unreachable
                # API is retried on the normal refresh cadence, not hammered
                # every single turn.
                self._fetched_at = time.time()
        finally:
            with self._lock:
                self._fetching = False

    def get_context_line(self) -> Optional[str]:
        """Best-effort, never blocks. Starts a background refresh when the
        cache is stale or empty; returns whatever is cached right now, which
        is None before the first successful fetch completes."""
        with self._lock:
            stale = (time.time() - self._fetched_at) > REFRESH_SECONDS
            should_start = stale and not self._fetching
            if should_start:
                self._fetching = True
            line = self._line
        if should_start:
            threading.Thread(target=self._fetch, daemon=True).start()
        return line
