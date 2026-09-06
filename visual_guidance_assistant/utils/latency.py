"""End-to-end latency tracking across the voice pipeline.

The individual stages were already timed, but nothing measured the number that
actually matters: how long the person waits between finishing speaking and
hearing a reply. That total spans three threads (microphone, dialogue, speech),
so no single one of them could report it.

Marks are process-global and thread-safe because the stages genuinely run on
different threads and there is only ever one conversation in flight.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional

_lock = threading.Lock()
_marks: Dict[str, float] = {}

# Stage names
SPEECH_END = "speech_end"          # person stopped talking (capture finished)
TRANSCRIBED = "transcribed"        # speech-to-text returned
RESPONSE_READY = "response_ready"  # reply text decided (model or guard)
TTS_START = "tts_start"            # audio actually begins playing


def mark(name: str) -> float:
    now = time.monotonic()
    with _lock:
        _marks[name] = now
    return now


def get(name: str) -> Optional[float]:
    with _lock:
        return _marks.get(name)


def since(name: str) -> Optional[float]:
    start = get(name)
    return None if start is None else time.monotonic() - start


def between(start_name: str, end_name: str) -> Optional[float]:
    with _lock:
        start, end = _marks.get(start_name), _marks.get(end_name)
    if start is None or end is None:
        return None
    return end - start


def report() -> str:
    """Human-readable breakdown of the last complete interaction."""
    capture = between(SPEECH_END, TRANSCRIBED)
    think = between(TRANSCRIBED, RESPONSE_READY)
    speak = between(RESPONSE_READY, TTS_START)
    total = between(SPEECH_END, TTS_START)

    def fmt(value):
        return "   n/a" if value is None else f"{value:6.2f}s"

    return (
        "\n"
        "+---------------------------------------------------------------+\n"
        "|  LATENCY: person stops speaking  ->  assistant starts speaking |\n"
        "+---------------------------------------------------------------+\n"
        f"|  speech-to-text (Google, network)          {fmt(capture)}      |\n"
        f"|  decide the reply (guards or Claude)       {fmt(think)}      |\n"
        f"|  hand off to the voice + start playback    {fmt(speak)}      |\n"
        "|                                            ---------          |\n"
        f"|  TOTAL WAIT THE PERSON EXPERIENCES         {fmt(total)}      |\n"
        "+---------------------------------------------------------------+"
    )
