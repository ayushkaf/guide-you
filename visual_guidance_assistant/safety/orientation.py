"""Orientation answers (day, date, time, place) — computed, never generated.

Found during testing: asked "where am I right now?", the model answered
"You're at home, Margaret." It has no location information whatsoever. It
guessed, fluently and warmly, and would have kept guessing differently every
time. Asked the date with no date in context, it does the same.

For someone relying on this to stay oriented, a confidently wrong answer is
worse than no answer — it is actively disorienting, and it is inconsistent
between askings, which is precisely what dementia care guidance says to avoid.

So orientation is answered from the system clock and from a location a caregiver
has explicitly configured. If the location is not configured, the assistant says
so warmly rather than inventing somewhere. Same input, same answer, every time.
"""
from __future__ import annotations

import re
import time
from typing import Optional

_DAY_PATTERNS = [
    r"\bwhat day is it\b", r"\bwhat.?s the day\b", r"\bwhich day is it\b",
    r"\bwhat day of the week\b", r"\bis it (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
]
_DATE_PATTERNS = [
    r"\bwhat.?s (today.?s )?date\b", r"\bwhat is (today.?s )?date\b",
    r"\bwhat.?s the date\b", r"\bwhat date is it\b", r"\bwhat month is it\b",
    r"\bwhat year is it\b",
]
_TIME_PATTERNS = [
    r"\bwhat time is it\b", r"\bwhat.?s the time\b", r"\bis it (morning|afternoon|evening|night)\b",
]
_PLACE_PATTERNS = [
    r"\bwhere am i\b", r"\bwhere are we\b", r"\bwhat is this place\b",
    r"\bwhere is this\b", r"\bwhose house is this\b",
]

_DAY_RE = [re.compile(p, re.I) for p in _DAY_PATTERNS]
_DATE_RE = [re.compile(p, re.I) for p in _DATE_PATTERNS]
_TIME_RE = [re.compile(p, re.I) for p in _TIME_PATTERNS]
_PLACE_RE = [re.compile(p, re.I) for p in _PLACE_PATTERNS]


def _part_of_day(hour: int) -> str:
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    # 9pm still reads as evening to most people; night starts later.
    if hour < 22:
        return "evening"
    return "night"


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def clock_phrase(now=None) -> str:
    """'8:52 in the evening'. Shared so every spoken time in the app reads the
    same way — a caregiver hearing the alert log should not get a different
    time format from the one the person being supported hears."""
    now = now or time.localtime()
    clock = time.strftime("%I:%M", now).lstrip("0")
    part = _part_of_day(now.tm_hour)
    # "in the morning/afternoon/evening" but "at night" — the article is wrong
    # for night and it reads as stilted when spoken aloud.
    return f"{clock} at night" if part == "night" else f"{clock} in the {part}"


def answer(text: str, home_location: Optional[str] = None, now=None) -> Optional[str]:
    """Deterministic orientation answer, or None if not an orientation question."""
    body = text or ""
    now = now or time.localtime()

    if any(p.search(body) for p in _PLACE_RE):
        if home_location:
            return f"You're at {home_location}. You're safe here."
        # Honest, warm, and does NOT invent a location.
        return (
            "I'm not able to tell where you are, but you're safe, and I'm here with you. "
            "Your carer can tell you for certain."
        )

    if any(p.search(body) for p in _TIME_RE):
        return f"It's {clock_phrase(now)}."

    if any(p.search(body) for p in _DAY_RE):
        return (f"It's {time.strftime('%A', now)} today, "
                f"the {_ordinal(now.tm_mday)} of {time.strftime('%B', now)}.")

    if any(p.search(body) for p in _DATE_RE):
        return (f"It's {time.strftime('%A', now)} the {_ordinal(now.tm_mday)} "
                f"of {time.strftime('%B %Y', now)}.")

    return None


def context_line(home_location: Optional[str] = None, now=None) -> str:
    """Ground truth handed to the model, so ordinary conversation doesn't drift
    onto an invented date either."""
    now = now or time.localtime()
    line = ("Right now it is "
            f"{time.strftime('%A %d %B %Y, %H:%M', now)} "
            f"({_part_of_day(now.tm_hour)}).")
    if home_location:
        line += f" The device is at {home_location}."
    else:
        line += (" You do NOT know where the person is located. Never guess or state "
                 "a location; if asked, say you cannot tell and their carer will know.")
    return line
