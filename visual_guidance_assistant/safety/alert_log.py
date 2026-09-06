"""Append-only local log of moments a caregiver should be able to review.

Deliberately local-only. Sending an SMS, email or phone call needs real
credentials, a real consent conversation with the person being monitored, and a
decision about who is allowed to receive it — none of which can be inferred
here. So nothing is transmitted anywhere. What this guarantees instead is that
nothing is silently lost: every help request, distress phrase, and deflected
medical question lands in a timestamped file a caregiver can read.

FUTURE INTEGRATION POINT: notify() is the single place an outbound notification
would hook in. It is intentionally left as local logging only.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from utils import secure_store
from utils.paths import ALERT_LOG_FILE, ensure_parent_dir

# Alert kinds. Kept as plain strings so the log stays readable without this code.
# Renamed from HELP_REQUEST: this fires only for genuine emergencies now, and
# the old name invited exactly the over-triggering it was renamed for. The
# string value is kept so alerts logged before the change still read back.
EMERGENCY = "help_request"
HELP_REQUEST = EMERGENCY  # backwards-compatible alias
DISTRESS = "distress"
MEDICAL_QUESTION = "medical_question"

_lock = threading.Lock()


def _read_all(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        data = secure_store.read_json(path, default={})
        return data.get("alerts", []) if isinstance(data, dict) else []
    except secure_store.DecryptionError as exc:
        # Loud, but not fatal: a caregiver must never be shown an empty alert
        # list that silently means "unreadable" rather than "nothing happened".
        print(f"[alerts] CANNOT DECRYPT ALERT LOG: {exc}")
        return []
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[alerts] WARNING: could not read {path}: {exc}")
        return []


def record(
    kind: str,
    speaker: Optional[str],
    role: Optional[str],
    utterance: str,
    response: str = "",
    detail: Optional[Dict[str, Any]] = None,
    filepath: Optional[str] = None,
) -> Dict[str, Any]:
    """Append one alert. Returns the entry written."""
    path = filepath or ALERT_LOG_FILE
    ensure_parent_dir(path)

    entry = {
        "kind": kind,
        "timestamp": time.time(),
        "time_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "speaker": speaker or "unidentified",
        "role": role or "unknown",
        "utterance": utterance,
        "response_spoken": response,
        "detail": detail or {},
        # Explicit, so a caregiver reading the file is never left assuming
        # someone was contacted automatically.
        "notified": False,
        "notify_note": "logged locally only - no message was sent to anyone",
    }

    with _lock:
        alerts = _read_all(path)
        alerts.append(entry)
        secure_store.write_json(
            path, {"alerts": alerts}, description="Guide YOU care alert log"
        )

    print(f"[alert] {kind} | speaker={entry['speaker']} | {entry['time_local']} | {utterance!r}")
    return entry


def recent(limit: int = 20, filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    return _read_all(filepath or ALERT_LOG_FILE)[-limit:]


def today(filepath: Optional[str] = None, now=None) -> List[Dict[str, Any]]:
    """Alerts from the current calendar day, oldest first.

    Uses the same system clock as safety/orientation.py, so "today" means the
    same thing everywhere in the app.
    """
    now = now or time.localtime()
    start = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1))
    end = start + 86400
    entries = [
        entry for entry in _read_all(filepath or ALERT_LOG_FILE)
        if start <= float(entry.get("timestamp", 0)) < end
    ]
    return sorted(entries, key=lambda e: float(e.get("timestamp", 0)))


# What each alert kind sounds like when read back. Plain sentences, because a
# caregiver is listening to this, not reading a table.
_SPOKEN_KIND = {
    EMERGENCY: "needed urgent help",
    DISTRESS: "sounded distressed",
    MEDICAL_QUESTION: "asked about medication or health",
}

MAX_SPOKEN = 8


def summarise_for_speech(entries: List[Dict[str, Any]]) -> str:
    """Read today's alerts back concisely, oldest first."""
    # Imported here rather than at module scope to keep the dependency one-way:
    # orientation does not import alert_log.
    from safety.orientation import clock_phrase

    if not entries:
        return "No alerts logged today."

    lines = []
    for entry in entries[:MAX_SPOKEN]:
        when = clock_phrase(time.localtime(float(entry.get("timestamp", 0))))
        who = entry.get("speaker") or "someone"
        if who == "unidentified":
            who = "someone I didn't recognise"
        what = _SPOKEN_KIND.get(entry.get("kind"), "logged something")
        line = f"At {when}, {who} {what}"
        # The exact words matter most for distress — that is the detail a
        # caregiver actually wants to hear.
        if entry.get("kind") == DISTRESS and entry.get("utterance"):
            line += f", and said: {entry['utterance']}"
        lines.append(line + ".")

    count = len(entries)
    header = ("There is 1 alert logged today." if count == 1
              else f"There are {count} alerts logged today.")
    if count > MAX_SPOKEN:
        lines.append(f"That's the first {MAX_SPOKEN}. The rest are in the log.")
    return " ".join([header] + lines)


def notify(entry: Dict[str, Any]) -> bool:
    """FUTURE INTEGRATION POINT — intentionally does nothing.

    Outbound notification (SMS/email/call) is out of scope until there are real
    credentials and the person's explicit, informed consent about who gets told
    what. Returning False keeps callers honest: nothing was sent.
    """
    return False
