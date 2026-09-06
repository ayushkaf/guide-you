"""Gentle spoken prompts at set times of day.

A predictable daily rhythm is one of the more genuinely useful things this
device can offer: the same few prompts, at the same times, in the same words.
That last part matters. Elsewhere in this codebase the safety-critical replies
are written out rather than generated precisely so they never vary, and the same
reasoning applies harder here — a reminder whose wording drifts day to day is
worse at doing the one job a routine has.

So routine text comes from config/routines.json and is spoken as written. The
config is the thing a carer edits; nothing is hardcoded in Python. An entry can
opt into having the model phrase it instead (`"natural": true`), which trades
predictability for variety, and medication entries may never do so.

WHAT THIS DELIBERATELY DOES NOT DO
  - It never tells anyone to take medication. A medication entry is a check-in
    that offers to involve a carer, and its wording is generated here rather
    than read from config so it cannot be edited into an instruction.
  - It never records that medication was taken. The device cannot see someone
    swallow a tablet, and a log saying "confirmed" when nobody confirmed
    anything is worse than no log at all.
  - It never speaks into an empty room, and never aims a prompt meant for the
    person being supported at a visiting carer.
  - It never talks over a conversation in progress.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Callable, Dict, List, Optional

from utils.paths import ROUTINE_STATE_FILE, ROUTINES_FILE, ensure_parent_dir

KIND_MEAL = "meal"
KIND_ACTIVITY = "activity"
KIND_MEDICATION = "medication_checkin"
KIND_CUSTOM = "custom"

# How often to check the clock. Seconds, not minutes: a prompt deferred because
# someone was mid-conversation should resume promptly once they stop.
POLL_SECONDS = 20

# A prompt more than this far past its time is dropped rather than delivered
# late. Being told it is about lunchtime at half past three is disorienting,
# which is the opposite of the point.
LATE_GRACE_MINUTES = 45

# Wording that would turn a check-in into medical instruction. Checked against
# any custom message on a medication entry; if it matches, the safe generated
# wording is used instead and a warning is logged.
_INSTRUCTION_WORDING = re.compile(
    r"\b(take|swallow|have)\b.{0,20}\b(your |the |it|them|dose|tablet|pill|"
    r"medication|medicine|meds)\b|\btime to take\b|\bdon'?t forget to take\b",
    re.IGNORECASE,
)


def medication_checkin_message(caregiver_names: List[str]) -> str:
    """The only wording a medication routine ever uses.

    Generated here, never read from config, so it cannot be edited into an
    instruction. It orients ("around the time you usually...") and offers to
    involve a human. It does not tell anyone to take anything, and it does not
    ask a question the device would then be tempted to treat as a record.
    """
    if caregiver_names:
        return (
            "It's about the time you usually check about your medication. "
            f"Would you like me to let {caregiver_names[0]} know, or do you "
            "have it sorted?"
        )
    return (
        "It's about the time you usually check about your medication. "
        "Would you like me to make a note for whoever helps you, or do you "
        "have it sorted?"
    )


class RoutineStore:
    """Reads config/routines.json, and remembers what has already fired today."""

    def __init__(self, filepath: Optional[str] = None,
                 state_path: Optional[str] = None):
        self.filepath = filepath or ROUTINES_FILE
        self.state_path = state_path or ROUTINE_STATE_FILE
        self._lock = threading.RLock()
        self._routines: List[Dict] = []
        self._fired: Dict[str, str] = {}     # routine id -> "YYYY-MM-DD"
        ensure_parent_dir(self.filepath)
        self.load()
        self._load_state()

    # ---------- config ----------

    def load(self) -> None:
        with self._lock:
            if not os.path.exists(self.filepath):
                print(f"[routines] no {self.filepath}; no routines will run")
                self._routines = []
                return
            try:
                with open(self.filepath, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (json.JSONDecodeError, OSError) as exc:
                # A broken routines file must not stop the assistant running.
                print(f"[routines] WARNING: could not read {self.filepath}: {exc}")
                print("[routines] continuing with no routines this session")
                self._routines = []
                return
            self._routines = [r for r in data.get("routines", []) if isinstance(r, dict)]
            valid = [r for r in self._routines if self._parse_time(r.get("time")) is not None]
            dropped = len(self._routines) - len(valid)
            if dropped:
                print(f"[routines] WARNING: {dropped} entr(ies) have an unreadable "
                      "time and were ignored; use 24-hour HH:MM")
            self._routines = valid
            enabled = sum(1 for r in self._routines if r.get("enabled", True))
            print(f"[routines] loaded {len(self._routines)} routine(s), {enabled} enabled")

    @staticmethod
    def _parse_time(value) -> Optional[tuple]:
        if not isinstance(value, str):
            return None
        match = re.fullmatch(r"\s*(\d{1,2})\s*:\s*(\d{2})\s*", value)
        if not match:
            return None
        hour, minute = int(match.group(1)), int(match.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return hour, minute

    def all_routines(self) -> List[Dict]:
        with self._lock:
            return [dict(r) for r in self._routines]

    def enabled_routines(self) -> List[Dict]:
        return [r for r in self.all_routines() if r.get("enabled", True)]

    def todays_schedule(self) -> List[Dict]:
        """Enabled routines in clock order, for reading back to a carer."""
        rows = self.enabled_routines()
        rows.sort(key=lambda r: self._parse_time(r.get("time")) or (0, 0))
        return rows

    # ---------- fired state ----------

    def _load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                self._fired = json.load(handle).get("fired", {})
        except (json.JSONDecodeError, OSError):
            self._fired = {}

    def _save_state(self) -> None:
        try:
            ensure_parent_dir(self.state_path)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump({"fired": self._fired}, handle, indent=2)
            os.replace(tmp, self.state_path)
        except OSError as exc:
            print(f"[routines] could not save fired state: {exc}")

    def has_fired_today(self, routine_id: str, today: str) -> bool:
        with self._lock:
            return self._fired.get(routine_id) == today

    def mark_fired(self, routine_id: str, today: str) -> None:
        with self._lock:
            self._fired[routine_id] = today
            self._save_state()

    def reset_today(self) -> None:
        """For tests, and for a carer who wants a prompt to run again."""
        with self._lock:
            self._fired = {}
            self._save_state()


class RoutineChecker(threading.Thread):
    """Polls the clock and speaks due routines. Mirrors AlarmChecker.

    Everything about WHEN it is allowed to speak is injected, so the rules stay
    testable without a camera or a microphone:

      patient_present()  the person the prompts are for is recognised, right now
      is_busy()          a conversation is in progress, or speech is playing
      caregiver_names()  who to offer to involve in a medication check-in
    """

    def __init__(
        self,
        store: RoutineStore,
        speech_queue,
        patient_present: Callable[[], bool],
        is_busy: Callable[[], bool],
        caregiver_names: Callable[[], List[str]] = None,
        phrase_naturally: Callable[[str], str] = None,
        poll_seconds: float = POLL_SECONDS,
    ):
        super().__init__(name="RoutineChecker", daemon=True)
        self.store = store
        self.speech_queue = speech_queue
        self.patient_present = patient_present
        self.is_busy = is_busy
        self.caregiver_names = caregiver_names or (lambda: [])
        self.phrase_naturally = phrase_naturally
        self.poll_seconds = float(poll_seconds)
        self.running = False
        self._deferred: Dict[str, float] = {}

    def stop(self) -> None:
        self.running = False

    # ---------- message ----------

    def message_for(self, routine: Dict) -> Optional[str]:
        kind = routine.get("kind", KIND_CUSTOM)

        if kind == KIND_MEDICATION:
            custom = routine.get("message")
            if custom and _INSTRUCTION_WORDING.search(custom):
                print(f"[routines] WARNING: medication entry "
                      f"{routine.get('id')!r} has wording that reads as an "
                      "instruction to take medication. Ignoring it and using "
                      "the safe check-in wording instead.")
            elif custom:
                print(f"[routines] note: medication entry {routine.get('id')!r} "
                      "has custom wording; the standard check-in is used anyway "
                      "so this path cannot drift into instruction.")
            return medication_checkin_message(self.caregiver_names())

        text = (routine.get("message") or "").strip()
        if not text:
            return None
        # Opt-in only, and never for medication. Fixed wording is the default
        # because a routine's value is in being the same every day.
        if routine.get("natural") and self.phrase_naturally:
            try:
                rephrased = (self.phrase_naturally(text) or "").strip()
                if rephrased:
                    return rephrased
            except Exception as exc:
                print(f"[routines] natural phrasing failed ({exc}); "
                      "using the written wording")
        return text

    # ---------- scheduling ----------

    def due_routines(self, now=None) -> List[Dict]:
        now = now or time.localtime()
        today = time.strftime("%Y-%m-%d", now)
        minutes_now = now.tm_hour * 60 + now.tm_min

        due = []
        for routine in self.store.enabled_routines():
            parsed = RoutineStore._parse_time(routine.get("time"))
            if parsed is None:
                continue
            scheduled = parsed[0] * 60 + parsed[1]
            if minutes_now < scheduled:
                continue
            if minutes_now - scheduled > LATE_GRACE_MINUTES:
                continue
            if self.store.has_fired_today(routine.get("id", routine["time"]), today):
                continue
            due.append(routine)
        return due

    def run(self) -> None:
        self.running = True
        print(f"[routines] checker started, polling every {self.poll_seconds:.0f}s")
        while self.running:
            try:
                self.tick()
            except Exception as exc:
                print(f"[routines] error in checker loop: {exc}")
            time.sleep(self.poll_seconds)

    def tick(self, now=None) -> List[str]:
        """One pass. Returns the ids actually spoken, for tests."""
        now = now or time.localtime()
        today = time.strftime("%Y-%m-%d", now)
        spoken = []

        for routine in self.due_routines(now):
            routine_id = routine.get("id", routine["time"])

            # Nobody the prompt is for. Do not talk to an empty room, and do
            # not aim a prompt meant for the person being supported at a
            # visiting carer. Left unfired so it can still land if they come
            # back within the grace window.
            if not self.patient_present():
                if routine_id not in self._deferred:
                    print(f"[routines] {routine_id}: due, but the person it's for "
                          "isn't here — holding")
                    self._deferred[routine_id] = time.time()
                continue

            # Mid-conversation. Wait rather than talk over them.
            if self.is_busy():
                if routine_id not in self._deferred:
                    print(f"[routines] {routine_id}: due, but a conversation is "
                          "in progress — holding until it's quiet")
                    self._deferred[routine_id] = time.time()
                continue

            message = self.message_for(routine)
            if not message:
                print(f"[routines] {routine_id}: nothing to say, skipping")
                self.store.mark_fired(routine_id, today)
                continue

            waited = ""
            if routine_id in self._deferred:
                waited = f" (held {time.time() - self._deferred.pop(routine_id):.0f}s)"
            print(f"[routines] {routine_id} at {routine.get('time')}{waited}: {message}")

            try:
                self.speech_queue.put_nowait(message)
            except Exception as exc:
                print(f"[routines] could not queue speech: {exc}")
                continue

            # Marked only after it has actually been handed to the voice, so a
            # failure to speak does not silently consume the day's prompt.
            self.store.mark_fired(routine_id, today)
            spoken.append(routine_id)

        return spoken


def describe_schedule(routines: List[Dict]) -> str:
    """Read the day's routines back to a carer."""
    if not routines:
        return "There are no reminders set for today."

    parts = []
    for routine in routines:
        when = (routine.get("time") or "").strip()
        kind = routine.get("kind", KIND_CUSTOM)
        if kind == KIND_MEDICATION:
            what = "a check about medication"
        else:
            what = (routine.get("message") or routine.get("id") or "a reminder").strip()
            what = what.rstrip(".")
        parts.append(f"at {when}, {what}")

    count = len(parts)
    lead = ("There's one reminder set for today: " if count == 1
            else f"There are {count} reminders set for today. ")
    return lead + "; ".join(parts) + "."
