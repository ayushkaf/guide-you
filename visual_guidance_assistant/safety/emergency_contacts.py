"""Emergency contact numbers — entered and confirmed by a caregiver/admin
ONLY, exactly like safety/orientation.py's home_location. The assistant never
auto-fetches, geolocation-looks-up, or guesses one of these.

Same reasoning as orientation: for someone who may depend on this device to
reach help, a confidently wrong number is far worse than an honest "I don't
have that yet". Guessing here is the bug, not the feature.

This module does not touch the existing emergency-DETECTION path at all
(safety/guards.py: "I've fallen", "call an ambulance", etc.) — that keeps
notifying the caregiver and logging, unchanged, and never reads a number
aloud. This only answers a calm, direct question like "what's the hospital's
number", or lets an authorised caregiver/admin set one.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from utils.paths import CONFIG_FILE

FIELDS = (
    "local_emergency_number",
    "nearest_hospital_name",
    "nearest_hospital_phone",
    "police_non_emergency_number",
)

_SPOKEN_NAME = {
    "local_emergency_number": "the emergency number",
    "nearest_hospital_name": "the nearest hospital's name",
    "nearest_hospital_phone": "the hospital's phone number",
    "police_non_emergency_number": "the police non-emergency number",
}

# Checked BEFORE _ASK_PATTERNS in both this module's ordering and the
# caller's — "the hospital number is 555-1234" also contains the bare phrase
# "hospital number", which _ASK_PATTERNS matches on its own. Whichever list is
# checked first wins, so the caller (dialogue_processor) must try SET first.
_SET_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bset\b.{0,15}\bemergency number\b.{0,6}\b(?:to|as|is)\b\s*(?P<value>.+)", re.I),
     "local_emergency_number"),
    (re.compile(r"\bemergency number is\b\s*(?P<value>.+)", re.I),
     "local_emergency_number"),
    (re.compile(r"\bset\b.{0,15}\bhospital.?s? (?:phone )?number\b.{0,6}\b(?:to|as|is)\b\s*(?P<value>.+)", re.I),
     "nearest_hospital_phone"),
    (re.compile(r"\bhospital.?s? (?:phone )?number is\b\s*(?P<value>.+)", re.I),
     "nearest_hospital_phone"),
    (re.compile(r"\bset\b.{0,15}\bhospital.?s? name\b.{0,6}\b(?:to|as|is)\b\s*(?P<value>.+)", re.I),
     "nearest_hospital_name"),
    (re.compile(r"\bhospital.?s? name is\b\s*(?P<value>.+)", re.I),
     "nearest_hospital_name"),
    (re.compile(r"\bset\b.{0,15}\bpolice (?:non.?emergency )?number\b.{0,6}\b(?:to|as|is)\b\s*(?P<value>.+)", re.I),
     "police_non_emergency_number"),
    (re.compile(r"\bpolice (?:non.?emergency )?number is\b\s*(?P<value>.+)", re.I),
     "police_non_emergency_number"),
)

_ASK_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bemergency number\b", re.I), "local_emergency_number"),
    (re.compile(r"\bhospital.?s? (?:phone )?number\b", re.I), "nearest_hospital_phone"),
    (re.compile(r"\b(?:nearest|which) hospital\b", re.I), "nearest_hospital_name"),
    (re.compile(r"\bwhat hospital\b", re.I), "nearest_hospital_name"),
    (re.compile(r"\bpolice (?:non.?emergency )?number\b", re.I), "police_non_emergency_number"),
    (re.compile(r"\bnumber for the police\b", re.I), "police_non_emergency_number"),
)


def default_contacts() -> Dict[str, str]:
    return {field: "" for field in FIELDS}


def answer(text: str, contacts: Dict[str, str]) -> Optional[str]:
    """A calm question about a saved contact, answered honestly — or None if
    this utterance isn't one. Never guesses: a blank field is reported as
    blank, exactly like orientation.answer() with no home_location set."""
    body = text or ""
    for pattern, field in _ASK_PATTERNS:
        if pattern.search(body):
            value = (contacts or {}).get(field)
            if value:
                return f"{_SPOKEN_NAME[field]} is {value}."
            return (f"I don't have {_SPOKEN_NAME[field]} saved yet — "
                    "your carer can add it.")
    return None


def parse_set_command(text: str) -> Optional[Tuple[str, str]]:
    """(field, value) if this looks like a caregiver setting a contact value,
    else None. Parsing only — the caller must confirm admin/caregiver access
    BY FACE before acting on it; this function has no way to check that."""
    body = text or ""
    for pattern, field in _SET_PATTERNS:
        match = pattern.search(body)
        if not match:
            continue
        value = match.group("value").strip(" .,!?'\"")
        if not value:
            continue
        return field, value
    return None


def confirmation_phrase(field: str, value: str) -> str:
    """Read back exactly what was captured, so a caregiver can catch a
    mis-hearing immediately rather than discovering it during an emergency."""
    return f"I've saved {_SPOKEN_NAME[field]} as {value} — is that right?"


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def save_contact(field: str, value: str, config_path: str = None) -> bool:
    """Persist one field's value into settings.yaml's existing
    care.emergency block, in place.

    Deliberately NOT a yaml.safe_load / yaml.safe_dump round-trip: this file
    carries extensive hand-written comments recording measured tuning
    rationale (camera resolution, detector thresholds, and so on), and a
    generic YAML dump strips every comment on the way back out. This instead
    does a plain text substitution of just the one line
    `    field_name: "..."` inside the emergency: block, so every other byte
    of the file — comments included — is untouched. It can only update a
    field that already has a line in the file; the initial four-field
    skeleton is checked into settings.yaml itself, not generated here.
    """
    if field not in FIELDS:
        raise ValueError(f"unknown emergency contact field: {field!r}")
    path = config_path or CONFIG_FILE
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        print(f"[emergency-contacts] could not read {path}: {exc}")
        return False

    pattern = re.compile(
        rf'^(?P<indent>[ \t]*){re.escape(field)}:[ \t]*(?:"[^"]*"|\'[^\']*\'|\S.*)?[ \t]*$',
        re.MULTILINE,
    )
    new_line = lambda m: f"{m.group('indent')}{field}: {_quote(value)}"  # noqa: E731
    updated, count = pattern.subn(new_line, text, count=1)
    if count == 0:
        print(f"[emergency-contacts] WARNING: no '{field}:' line found in "
              f"{path} — settings.yaml's care.emergency skeleton may be "
              "missing or malformed. Value NOT saved.")
        return False

    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(updated)
    except OSError as exc:
        print(f"[emergency-contacts] could not write {path}: {exc}")
        return False
    return True
