"""Universal layered memory store for the AI companion.

HISTORY OF A REAL BUG: for weeks this module was read on every single LLM turn
(get_context_for_llm) and written by NOTHING. The original writers — identity
learned during face training — were removed when identity moved to the profile
store, and no writer replaced them, so the context card said "goals: none,
mood: stable" forever and the 8-layer design was effectively dead code wearing
a seatbelt. learn_from_utterance() below is the writer, called by the dialogue
thread on every recognised command.
"""
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from utils.paths import LAYERED_MEMORY_FILE, ensure_parent_dir

PERSONA_MAP = {
    "elderly": "elderly",
    "student": "student",
    "professional": "professional",
    "fitness": "fitness",
    "creative": "creative",
    "child": "child",
    "unknown": "unknown",
}

RETENTION_SECONDS = {
    "1": 7 * 24 * 3600,
    "2": 30 * 24 * 3600,
    "3": 7 * 24 * 3600,
    "4": 30 * 24 * 3600,
    "5": 7 * 24 * 3600,
    "6": None,
    "7": 30 * 24 * 3600,
}

class LayeredMemory:
    def __init__(self, filepath: Optional[str] = None):
        # Absolute by default: a relative path here meant the identity learned
        # during face training was written next to whatever directory the app
        # happened to be launched from, and silently not found next run.
        self.filepath = filepath or LAYERED_MEMORY_FILE
        self.data: Dict[str, Any] = {"layers": {str(i): {} for i in range(8)}, "created": time.time()}
        ensure_parent_dir(self.filepath)
        self.load()

    def save(self) -> None:
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def load(self) -> None:
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.data = json.load(f)
                if "layers" not in self.data:
                    self.data = {"layers": {str(i): {} for i in range(8)}, "created": time.time()}

    def add_entry(self, layer: int, category: str, entry: Any) -> None:
        layer_key = str(layer)
        now = time.time()
        if layer_key not in self.data["layers"]:
            self.data["layers"][layer_key] = {}

        if layer_key in ("0", "6"):
            if not isinstance(entry, dict):
                raise ValueError("Layer 0 and 6 entries must be dicts")
            layer_store = self.data["layers"][layer_key]
            category_store = layer_store.get(category, {})
            if not isinstance(category_store, dict):
                category_store = {}
            category_store.update(entry)
            category_store["updated"] = now
            layer_store[category] = category_store
        else:
            layer_store = self.data["layers"][layer_key]
            layer_store.setdefault(category, []).append({
                "data": entry,
                "timestamp": now,
            })
        self.save()

    # ---------- learning from what people actually say ----------
    #
    # Deterministic patterns, not the model. The same reasoning as the safety
    # guards: a memory that sometimes writes and sometimes doesn't is impossible
    # to trust or to test, and this must keep working when the API is down.
    # Precision over recall — a missed goal costs nothing, a hallucinated
    # "health note" in a care context is actively harmful.
    _LEARN_PATTERNS: List[Tuple[int, str, Any]] = [
        # (layer, category, compiled pattern) — first match wins per rule set;
        # learning is checked before goals because "I want to learn X" is both.
        (4, "learning", re.compile(
            r"\b(?:i (?:want|would like|'?d like) to learn|teach me)"
            r"(?: about| to)? (?P<gist>[a-z0-9][\w\s'-]{2,50})", re.I)),
        (2, "goals", re.compile(
            r"\b(?:my goal is(?: to)?|i(?:'m| am) trying to|i really want to"
            r"|i want to(?! learn)) (?P<gist>[a-z0-9][\w\s'-]{2,50})", re.I)),
        (3, "tasks", re.compile(
            r"\b(?:i (?:need|have) to|i must) (?P<gist>[a-z0-9][\w\s'-]{2,50})"
            r"(?: today| this (?:morning|afternoon|evening))\b", re.I)),
        # "have to VERB ... today" above never matched an appointment stated
        # the way people actually say them — "I have a dentist appointment
        # Thursday" is "have A", not "have TO", and names a day rather than
        # "today"/"this afternoon". Real gap, found live: this exact sentence
        # produced learned=[] and no entry anywhere in layered_memory.json.
        # Requires an appointment-flavoured word rather than matching any
        # bare "I have a/an X", so "I have a headache" or "I have a cat"
        # don't get filed as tasks.
        (3, "tasks", re.compile(
            r"\bi have (?:an?|my) (?P<gist>[a-z0-9][\w\s'-]{2,60}"
            r"(?:appointment|meeting|check-?up|call|visit|reminder|session"
            r"|exam|interview)[\w\s'-]{0,30})", re.I)),
        (1, "health", re.compile(
            r"\bmy (?P<gist>(?:back|knee|knees|head|hip|leg|arm|shoulder|neck"
            r"|stomach|chest|feet|foot|hand|eyes?) (?:hurts?|aches?|is sore"
            r"|are sore))", re.I)),
        (1, "health", re.compile(
            r"\bi (?P<gist>(?:didn'?t|couldn'?t|can'?t) sleep(?: well| last night)?)",
            re.I)),
        # No IGNORECASE here, deliberately: the second alternative relies on a
        # capital letter to tell a person's name from an ordinary word. The
        # pronoun is [Ii] so the rule still fires mid-sentence or lowercased.
        # "my <relation>" captures the relation word plus an optional Name and
        # nothing further, so "I saw my sister yesterday" stores "my sister".
        (5, "social", re.compile(
            r"\b[Ii] (?:saw|met|visited|spoke to|talked to|called)"
            r" (?P<gist>my [a-z]+(?: [A-Z][\w'-]{1,20})?|[A-Z][\w'-]{1,20})")),
        # Layer 6 was reserved with permanent retention (RETENTION_SECONDS)
        # but had no writer at all — "I prefer X" matched nothing, anywhere,
        # confirmed live (learned=[]). Preferences fit permanent retention
        # far better than goals/tasks' 30/7-day windows: a stated preference
        # like "I prefer short answers" isn't something that should expire.
        (6, "preferences", re.compile(
            r"\bi(?:'d| would)? prefer(?: to)? (?P<gist>[a-z0-9][\w\s'-]{2,50})",
            re.I)),
    ]

    def learn_from_utterance(self, text: str, profile: Optional[str]) -> List[str]:
        """Write anything memorable in one utterance to its layer.

        Returns human-readable notes of what was learned, for the console —
        an invisible memory write is indistinguishable from no write, which is
        exactly how this system stayed broken for weeks.
        """
        learned: List[str] = []
        if not text:
            return learned
        for layer, category, pattern in self._LEARN_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            gist = match.group("gist").strip(" .,!?'\"-")
            if len(gist) < 3:
                continue
            if self._already_known(layer, category, gist, profile):
                continue
            entry = {"data": gist, "timestamp": time.time()}
            if profile:
                entry["profile"] = profile
            layer_store = self.data["layers"].setdefault(str(layer), {})
            layer_store.setdefault(category, []).append(entry)
            learned.append(f"{category}: {gist!r}"
                           + (f" (for {profile})" if profile else ""))
        if learned:
            self.save()
            for note in learned:
                print(f"[memory] learned {note}")
        return learned

    def _already_known(self, layer: int, category: str, gist: str,
                       profile: Optional[str]) -> bool:
        """Only a match against the SAME profile (None counts as its own
        profile: "still nobody recognised") blocks a duplicate write.

        BUG this replaces: `entry.get("profile") in (None, profile)` treated
        an unattributed entry (profile=None — recorded before anyone was
        recognised) as matching every possible profile, not just a future
        None. So if the first thing ever said to this device — before any
        face was recognised — happened to share wording with a fact a second
        enrolled person later stated, that second person's entry was silently
        dropped as "already known", even though it had never been recorded
        under their name. That is real cross-contamination between profiles,
        just narrower than losing every fact: it only bites on matching text.
        """
        for entry in self.data["layers"].get(str(layer), {}).get(category, []):
            if not isinstance(entry, dict):
                continue
            if str(entry.get("data", "")).lower() != gist.lower():
                continue
            if entry.get("profile") == profile:
                return True
        return False

    def _items_for(self, layer: str, category: str,
                   profile: Optional[str]) -> List[Dict[str, Any]]:
        """Entries visible to one profile: their own, plus any recorded before
        a speaker was recognised. With no profile given, everything — the solo
        developer testing without being recognised yet still sees their notes."""
        items = self.data["layers"].get(layer, {}).get(category, [])
        if not profile:
            return items
        wanted = profile.lower()
        return [e for e in items
                if not isinstance(e, dict)
                or e.get("profile") is None
                or str(e.get("profile", "")).lower() == wanted]

    def load_persona(self, name: str) -> Dict[str, Any]:
        identity = self.data["layers"]["0"].get("identity", {})
        persona = {
            "name": name,
            "persona_type": identity.get("persona_type", "unknown"),
            "preferences": identity.get("preferences", {}),
            "voice_profile": identity.get("voice_profile", {}),
            "age": identity.get("age"),
            "emergency_contacts": identity.get("emergency_contacts", []),
        }
        if identity.get("name") and identity.get("name").lower() == name.lower():
            persona["name"] = identity.get("name")
        return persona

    def detect_persona_type(self, user_data: Any) -> str:
        text = str(user_data).lower()
        if "exam" in text or "study" in text or "class" in text:
            return "student"
        if "meeting" in text or "project" in text or "report" in text:
            return "professional"
        if "workout" in text or "gym" in text or "run" in text or "yoga" in text:
            return "fitness"
        if "paint" in text or "write" in text or "create" in text:
            return "creative"
        if "grand" in text or "medicine" in text or "age" in text:
            return "elderly"
        if "school" in text or "toys" in text or "homework" in text:
            return "child"
        return "unknown"

    def get_context_for_llm(self, persona: Dict[str, Any]) -> str:
        """Notes about the PERSON BEING HELPED, for the system prompt.

        This used to be drawn as a box headed "USER: <name> | PERSONA: Patient".
        "Persona" is a strong convention for the ASSISTANT's own persona in a
        system prompt, so that line read as "your persona is: Patient" and
        directly contradicted the surrounding prose. The heading now says whose
        details these are, and says explicitly that they are not the assistant's.

        Plain labelled lines rather than box-drawing: the box added nothing a
        model uses, cost tokens on every single turn, and its fixed-width
        padding did not line up anyway once a real name went in.
        """
        # Filtered to THIS person's entries (plus any noted before a speaker
        # was recognised) — the layers are shared storage, the card is not.
        who = persona.get("name")
        rows = [
            ("name", who or "not known yet"),
            ("goals", self._format_items(self._items_for("2", "goals", who), "goal")),
            ("today", self._format_items(self._items_for("3", "tasks", who), "task")),
            ("learning", self._format_items(self._items_for("4", "learning", who), "topic")),
            ("health notes", self._format_items(self._items_for("1", "health", who), "health item")),
            ("social", self._format_items(self._items_for("5", "social", who), "social note")),
            ("preferences", self._format_items(self._items_for("6", "preferences", who), "preference")),
            ("mood", persona.get("preferences", {}).get("mood", "stable")),
        ]
        lines = "\n".join(f"  {label}: {value}" for label, value in rows)
        return (
            "ABOUT THE PERSON YOU ARE HELPING — these are THEIR details, not "
            "yours:\n" + lines
        )

    def search_layers(self, query: str) -> List[Dict[str, Any]]:
        results = []
        lookup = query.lower()
        for layer_key, layer_data in self.data["layers"].items():
            if layer_key in ("0", "6"):
                for category, item in layer_data.items():
                    if lookup in str(category).lower() or lookup in str(item).lower():
                        results.append({"layer": layer_key, "category": category, "data": item})
            else:
                for category, items in layer_data.items():
                    for entry in items:
                        if lookup in str(category).lower() or lookup in str(entry.get("data", "")).lower():
                            results.append({"layer": layer_key, "category": category, "data": entry})
        return results

    def purge_expired(self) -> None:
        now = time.time()
        for layer_key, retention in RETENTION_SECONDS.items():
            if retention is None:
                continue
            layer_data = self.data["layers"].get(layer_key, {})
            for category, entries in list(layer_data.items()):
                if not isinstance(entries, list):
                    continue
                filtered = [entry for entry in entries if now - entry.get("timestamp", now) <= retention]
                if filtered:
                    layer_data[category] = filtered
                else:
                    layer_data.pop(category, None)
        self.save()

    def export_for_caregiver(self) -> Dict[str, Any]:
        layers = self.data["layers"]
        return {
            "identity": layers["0"].get("identity", {}),
            "health": layers["1"].get("health", []),
            "goals": layers["2"].get("goals", []),
            "tasks": layers["3"].get("tasks", []),
            "social": layers["5"].get("social", []),
            "recent_history": layers["7"].get("conversations", []),
        }

    def get_persona(self, name: str) -> Dict[str, Any]:
        identity = self.data["layers"]["0"].get("identity", {})
        if identity.get("name", "").lower() == name.lower():
            return identity
        return {"name": name, "persona_type": "unknown", "preferences": {}}

    def _format_items(self, items, label):
        if not items:
            return "none"
        if isinstance(items, list):
            texts = [str(item.get("data", item)) if isinstance(item, dict) else str(item) for item in items]
            return ", ".join(texts[:3])
        return str(items)
