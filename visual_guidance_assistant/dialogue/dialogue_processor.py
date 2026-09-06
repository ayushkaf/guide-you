"""Dialogue processor: resolve who is speaking, then answer in the right mode.

Order of handling is deliberate and safety-first. Deterministic guards run
BEFORE the model is ever called, so a request for help or a question about
medication produces the same calm, predictable response every time rather than
whatever the model decides in that moment.
"""
import os
import random
import re
import threading
import queue
import time
import traceback
from typing import Optional

import numpy as np

from llm.llm_connector import (
    CAREGIVER_MAX_TOKENS,
    PATIENT_MAX_TOKENS,
    ROLE_CAREGIVER,
    ROLE_PATIENT,
    LLMConnector,
)
from memory.memory_store import MemoryStore
from memory.layered_memory import LayeredMemory
from profiles.profile_store import ROLE_ADMIN, ROLE_KNOWN_PERSON, ProfileStore
from recognition.face_encoder import FACE_MATCH_MARGIN
from recognition.face_recognizer import FaceRecognizer, safe_filename
from recognition.face_trainer import PATIENT_INTRO_TEXT, FaceTrainer
from care import routines
from care.weather_context import WeatherContext
from safety import alert_log, emergency_contacts, guards, orientation
from scene.scene_builder import build_scene_description
from utils import latency
from utils.paths import KNOWN_FACES_DIR
from voice_id import voice_embedding

# The API key is never passed through source — LLMConnector constructs the
# Anthropic client with no key argument and the SDK reads ANTHROPIC_API_KEY from
# the environment itself. This check exists only to fail loudly and early
# instead of deep in a background thread on the first LLM call.
if not os.environ.get("ANTHROPIC_API_KEY"):
    # A banner, not a one-liner. The old single line scrolled past in the
    # startup noise, so the first symptom anyone actually noticed was the
    # assistant apologising to every single thing it was told.
    print("\n" + "!" * 78)
    print("!!  ANTHROPIC_API_KEY IS NOT SET IN THIS PROCESS")
    print("!!")
    print("!!  Every reply will be:  'I can't reach my thinking service at the moment.'")
    print("!!")
    print("!!  The key is read from the environment when the app STARTS. If it was set")
    print("!!  with setx AFTER this terminal was opened, this process did not inherit")
    print("!!  it — setx only affects terminals opened afterwards.")
    print("!!")
    print("!!    1. close this terminal (and VS Code, if the app runs from its terminal)")
    print("!!    2. open a new one")
    print("!!    3. verify:  echo %ANTHROPIC_API_KEY%")
    print("!!    4. start the app again")
    print("!!")
    print("!!  If it is genuinely unset:  setx ANTHROPIC_API_KEY \"sk-ant-...\"")
    print("!!  Get a key at https://console.anthropic.com/settings/keys")
    print("!" * 78 + "\n")

_LIST_PEOPLE_PATTERNS = [
    re.compile(r"\bwho do you know\b", re.I),
    re.compile(r"\bwho have you met\b", re.I),
    re.compile(r"\bwho do you recognis[ez]e\b", re.I),
    re.compile(r"\blist (the )?(known )?people\b", re.I),
]

# Admin/caregiver setup audit — distinct from _LIST_PEOPLE_PATTERNS above,
# which is the warm, patient-facing "who do you know" summary. This one names
# roles, enrolment dates and what data is on file, so the wording requires
# "profile(s)" or "enrolled" to avoid ever firing for the patient-facing ask.
_LIST_PROFILES_ADMIN_PATTERNS = [
    re.compile(r"\blist (all )?(the )?(enrolled )?profiles\b", re.I),
    re.compile(r"\bshow (me )?(all )?(the )?enrolled profiles\b", re.I),
    re.compile(r"\bprofile (list|details|audit)\b", re.I),
    re.compile(r"\bwho.?s enrolled\b", re.I),
]

# Admin/caregiver only, same gate as the audit above. Restores a profile
# mark_archived() previously excluded from live matching — see that method's
# docstring for why archiving (not deleting) is how a duplicate-face profile
# gets fixed. Requires the name in the same utterance, e.g. "unarchive Sarah".
_UNARCHIVE_PATTERNS = [
    re.compile(r"\b(?:un-?archive|restore|reactivate)\s+(?P<name>[A-Za-z][\w-]*)"
               r"(?:'s)?(?:\s+profile)?\b", re.I),
]

_READ_ROUTINES_PATTERNS = [
    re.compile(r"\bwhat.{0,6}\b(are|is)\b.{0,12}\breminders?\b", re.I),
    re.compile(r"\breminders? (for )?today\b", re.I),
    re.compile(r"\btoday.?s (reminders?|routines?|schedule)\b", re.I),
    re.compile(r"\bwhat.{0,12}\b(routines?|schedule)\b", re.I),
    re.compile(r"\bread (me )?(the )?(reminders?|routines?|schedule)\b", re.I),
]

_READ_ALERTS_PATTERNS = [
    re.compile(r"\bread (me )?(today.?s )?alerts?\b", re.I),
    re.compile(r"\bany alerts (today)?\b", re.I),
    re.compile(r"\bwhat happened today\b", re.I),
]

# Asking who you are is one of the questions this app most needs to answer well,
# and it must be answered from stored data rather than by the model, which has
# no reliable way to know and will improvise a name.
# Enrolling someone else, usually by holding a photo up to the camera.
# Matched on keywords rather than whole phrases, because speech recognition
# routinely drops small words — "remember this person" comes back as "remember
# person" often enough to matter.
_REMEMBER_PERSON_PATTERNS = [
    re.compile(r"\bremember\b.{0,12}\bthis (person|one|face)\b", re.I),
    re.compile(r"\bremember\b.{0,12}\bperson\b", re.I),
    re.compile(r"\b(add|remember|learn|save)\b.{0,16}\bfamily member\b", re.I),
    re.compile(r"\b(add|remember|learn|save)\b.{0,12}\b(someone|somebody)\b", re.I),
    re.compile(r"\bthis is (my|a)\b.{0,20}\b(photo|picture)\b", re.I),
    re.compile(r"\b(learn|remember)\b.{0,12}\b(their|her|his) face\b", re.I),
    re.compile(r"\bnew person\b", re.I),
]

# Self-enrolment trigger — "remember me" and "remember my face" are the same
# command. Keyword-based, like _REMEMBER_PERSON_PATTERNS above: speech
# recognition drops small words unpredictably, so "remember my face" can come
# back as "remember face" or missing "face" entirely. Checked AFTER
# _REMEMBER_PERSON_PATTERNS (see the ordering note above _handle_remember_
# person's call site) so "remember her face" / "remember this person" are
# never mistaken for a self-enrolment. Deliberately does NOT match bare
# "remember" + "my" alone — that would also fire on "remember my daughter",
# which means someone else entirely.
_REMEMBER_ME_PATTERNS = [
    re.compile(r"\bremember\b.{0,10}\bme\b", re.I),
    re.compile(r"\bremember\b.{0,10}\bmy face\b", re.I),
    re.compile(r"\bremember\b.{0,10}\bface\b", re.I),
]

# The two answers to the ambiguous "remember me" follow-up question — see
# DialogueProcessor._resolve_pending_remember_me. Only ever checked against
# the single utterance right after that question was asked, so a bare "new"
# or "already" elsewhere in ordinary conversation is never at risk of being
# misread as answering a question that was never asked.
_REMEMBER_ME_FOLLOWUP_NEW = re.compile(r"\bnew\b", re.I)
_REMEMBER_ME_FOLLOWUP_KNOWN = re.compile(r"\b(?:already|know me|known)\b", re.I)

# "Who is that" — about somebody else in view, as opposed to _WHO_AM_I_PATTERNS
# which is about the speaker. Both must be answered from stored data rather than
# by the model: asked who someone is, a model with no reliable knowledge will
# produce a confident, friendly, invented name, and a wrong name is worse than
# an honest "I don't recognise them" for someone relying on this for orientation.
# Admin-only testing switch, so the gentle patient behaviour can be checked
# without re-enrolling as a patient.
_MODE_PATIENT_PATTERNS = [
    re.compile(r"\b(switch|change|go|set)\w*\s+to\s+patient mode\b", re.I),
    re.compile(r"\b(test|try)\s+patient mode\b", re.I),
    re.compile(r"\bpatient mode on\b", re.I),
]
_MODE_NORMAL_PATTERNS = [
    # Any mention of normal/admin/caregiver mode, with or without a leading
    # verb — "back to normal mode" has no verb at all, which the first version
    # of this required and therefore missed.
    re.compile(r"\b(normal|admin|caregiver) mode\b", re.I),
    re.compile(r"\bback to normal\b", re.I),
    re.compile(r"\bpatient mode off\b", re.I),
    re.compile(r"\bstop testing\b", re.I),
]

_WHO_IS_THAT_PATTERNS = [
    re.compile(r"\bwho.?s (that|this|he|she|they)\b", re.I),
    re.compile(r"\bwho is (that|this|he|she|they|the)\b", re.I),
    re.compile(r"\bwho are (they|these)\b", re.I),
    re.compile(r"\bdo you (know|recognis|recogniz)\w*\s+(them|him|her|this|that)\b", re.I),
    re.compile(r"\bwho.{0,10}\bin (the )?(picture|photo|frame|room)\b", re.I),
    re.compile(r"\bwho.?s (here|with me|in front of me)\b", re.I),
]

# What the assistant calls itself. Kept here so the deterministic identity
# answers and the system prompt cannot drift apart.
ASSISTANT_NAME = "Guide YOU"
_ASSISTANT_ALIASES = {"guide you", "guide", "guideyou", "guide u"}

# Words that follow "are you" or "am I" without being a name. Without this,
# "are you okay?" would be treated as an identity question about someone called
# Okay, and answered with a flat denial.
_NOT_A_NAME = {
    "okay", "ok", "alright", "right", "there", "here", "sure", "awake", "real",
    "human", "a", "an", "the", "still", "listening", "busy", "ready", "well",
    "fine", "sorry", "serious", "joking", "kidding", "talking", "speaking",
    "hearing", "watching", "recording", "on", "off", "up", "good", "bad",
    "my", "your", "that", "this", "it", "someone", "somebody", "anyone",
    "hungry", "tired", "safe", "lost", "going", "doing", "coming", "working",
    "married", "old", "new", "free", "happy", "sad", "angry", "mad", "asleep",
    # his/her/their/him — found live: "I am his caregiver" (a real answer
    # given with no name in it at all) matched the "i am <word>" pattern in
    # _extract_conversational_name and took "his" AS the name, enrolling a
    # phantom profile called "His". Excluding these here means that phrase
    # now correctly extracts NO name, which is what actually happened.
    "his", "her", "hers", "him", "their", "theirs", "our", "ours",
}

# "Are YOU <name>?" — about the ASSISTANT's identity.
_IS_ASSISTANT_NAME = [
    re.compile(r"\bare\s+you\s+(?P<name>[\w'-]+)", re.I),
    re.compile(r"\byou(?:'?re|\s+are)\s+(?P<name>[\w'-]+)\s*,?\s*"
               r"(right|aren'?t\s+you|yeah|yes|no|correct)\b", re.I),
    re.compile(r"\bis\s+(?:this|that)\s+(?P<name>[\w'-]+)", re.I),
    re.compile(r"\bam\s+i\s+(?:talking|speaking|chatting)\s+(?:to|with)\s+(?P<name>[\w'-]+)", re.I),
]

# "Am I <name>?" — about the SPEAKER's identity. A different question with a
# different answer, and easy to conflate with the one above.
_IS_SPEAKER_NAME = [
    re.compile(r"\bam\s+i\s+(?P<name>[\w'-]+)", re.I),
    re.compile(r"\bi'?m\s+(?P<name>[\w'-]+)\s*,?\s*(right|aren'?t\s+i|yeah|yes|no)\b", re.I),
    re.compile(r"\bi\s+am\s+(?P<name>[\w'-]+)\s*,?\s*(right|aren'?t\s+i)\b", re.I),
]

# "Who are you" — about the ASSISTANT. Answered deterministically for the same
# reason as the rest: identity should never vary between turns.
_WHO_ARE_YOU_PATTERNS = [
    re.compile(r"\bwho\s+are\s+you\b", re.I),
    re.compile(r"\bwhat\s+are\s+you\b", re.I),
    re.compile(r"\bwhat'?s\s+your\s+name\b", re.I),
    re.compile(r"\bwho\s+am\s+i\s+(?:talking|speaking|chatting)\s+(?:to|with)\b", re.I),
]

_WHO_AM_I_PATTERNS = [
    re.compile(r"\bwho am i\b", re.I),
    re.compile(r"\bwhat.?s my name\b", re.I),
    re.compile(r"\bwhat is my name\b", re.I),
    re.compile(r"\bdo you know who i am\b", re.I),
    re.compile(r"\bdo you remember me\b", re.I),
    re.compile(r"\bdo you know my name\b", re.I),
]

# ---- Conversational auto-enrolment of an unrecognised face (known_person
# only) — see DialogueProcessor._maybe_ask_unrecognized_name and
# _resolve_pending_conversational_enrol. No trigger phrase gates this: it
# fires on ordinary conversation from someone the camera cannot match to any
# profile. Separate and deliberately looser than _extract_claimed_name's
# patterns above, because these are only ever checked against the single
# utterance right after "what's your name?" was asked — there is no
# ordinary conversation this could be confused with, unlike a bare identity
# claim floating in normal speech.
#
# "name is" (not just "my name is") is its own alternative, found the hard
# way in a real live-microphone run: Google's free STT quietly dropped "my"
# and transcribed "My name is Ayush" as "name is ayush". Without this, that
# fell through every structured pattern to the bare-reply fallback below,
# which then took "name" — the first word of "name is ayush" — AS the name,
# enrolling someone literally named "Name". "name is" alone is unambiguous
# enough in this narrow, already-gated context to enrol directly on.
_CONVERSATIONAL_NAME_PATTERNS = [
    re.compile(r"\b(?:my name is|name is|i am|i'?m)\s+(?P<name>[A-Za-z][\w'-]*)", re.I),
    re.compile(r"\b(?:it'?s|this is|call me)\s+(?P<name>[A-Za-z][\w'-]*)", re.I),
]

_DECLINE_NAME_PATTERNS = [
    re.compile(r"\b(?:no thanks|no thank you|rather not|never\s?mind|not telling|"
               r"not saying|prefer not|don'?t want to say|not (?:now|today))\b", re.I),
    re.compile(r"^\s*no\.?\s*$", re.I),
]

# Words that, alone, could be mistaken for a bare one-word name reply
# ("John") if not excluded. Combined with _NOT_A_NAME (declared above) so a
# stray pronoun or filler word in a reply to "what's your name?" is never
# captured as if it were the name itself — see _extract_conversational_name.
_BARE_NAME_STOPWORDS = _NOT_A_NAME | {
    "i", "im", "my", "he", "she", "it", "we", "you", "yes", "maybe",
    "please", "thanks", "thank", "hi", "hello", "hey", "name", "is",
}

# A claim to a role that grants REAL system access, made mid-conversation by
# someone the camera does not recognise. This is checked against every
# answer given during conversational enrolment (name stage and relationship
# stage alike) and must NEVER be trusted on its own: see
# DialogueProcessor._resolve_pending_conversational_enrol and
# FaceTrainer.capture_conversational_person, which structurally has no role
# parameter at all. There is no wording of this claim that this path is
# allowed to act on — the only response is to enrol as known_person (or not
# at all) and redirect to an admin.
_PRIVILEGED_ROLE_CLAIM = re.compile(
    r"\b(caregivers?|care\s?givers?|carers?|admins?|administrators?|patients?)\b", re.I
)

# Whitelist, not a blacklist — deliberately. Extracting a relationship word
# by matching AGAINST a fixed list of harmless family/friend terms means
# "caregiver"/"carer"/"admin" cannot end up stored as a "relationship" no
# matter how the sentence is phrased, even if the privileged-claim check
# above were somehow bypassed. See _extract_relationship.
_RELATIONSHIP_WORDS = (
    "daughter", "son", "wife", "husband", "friend", "neighbour", "neighbor",
    "sister", "brother", "niece", "nephew", "granddaughter", "grandson",
    "cousin", "aunt", "uncle", "partner",
)
_RELATIONSHIP_PATTERN = re.compile(
    r"\b(?P<rel>" + "|".join(_RELATIONSHIP_WORDS) + r")\b", re.I
)


class DialogueProcessor(threading.Thread):
    def __init__(
        self,
        command_queue,
        speech_queue,
        config,
        conversation_history,
        conversation_lock,
        memory_processor,
        layered_memory: LayeredMemory,
        face_recognizer: FaceRecognizer,
        camera,
        memory_store: MemoryStore,
        profile_store: ProfileStore,
        detection_store=None,
        sound_queue=None,
        mic_listener=None,
        routine_store=None,
    ):
        super().__init__(daemon=True)
        # Read-only here: the checker thread owns firing, this only reads the
        # schedule back to a carer who asks.
        self.routine_store = routine_store
        self.command_queue = command_queue
        self.speech_queue = speech_queue
        self.llm = LLMConnector()
        self.memory = memory_store
        self.memory_processor = memory_processor
        self.layered_memory = layered_memory
        self.profiles = profile_store
        self.face_recognizer = face_recognizer
        self.sound_queue = sound_queue
        self.camera = camera
        self.detection_store = detection_store
        self.conversation_history = conversation_history
        self.conversation_lock = conversation_lock
        self.mic_listener = mic_listener
        self.face_trainer = FaceTrainer(
            face_recognizer,
            profile_store,
            speech_queue=speech_queue,
            mic_listener=mic_listener,
        )
        self.running = False
        self.last_command = ""
        self.latest_scene_description = "No scene detected yet."
        # Blank unless a caregiver sets it. Blank means "say you don't know",
        # never "guess" — see safety/orientation.py.
        self.home_location = ((config or {}).get("care", {}) or {}).get("home_location") or None
        # Same blank-means-honest-not-guessed rule as home_location, for the
        # same reason — see safety/emergency_contacts.py. Loaded once here;
        # _handle_set_emergency_contact keeps this dict and settings.yaml
        # in sync at runtime so a caregiver's voice-set value takes effect
        # immediately, not just after a restart.
        configured_contacts = ((config or {}).get("care", {}) or {}).get("emergency") or {}
        self.emergency_contacts = emergency_contacts.default_contacts()
        self.emergency_contacts.update(
            {k: v for k, v in configured_contacts.items() if k in self.emergency_contacts}
        )
        # Fetched in the background (see WeatherContext) so a slow or
        # unreachable network never adds latency to a conversation turn.
        # Casual colour only — never used for "where am I", which stays
        # exclusively on home_location above.
        self.weather_context = WeatherContext()
        self.weather_context.get_context_line()  # kicks off the first fetch now
        # Identity of whoever is currently in view, refreshed from detections.
        self.current_speaker = None
        self.current_role = ROLE_PATIENT
        # How that identity was established: "face", "voice", or "default".
        # Some commands require face specifically — see handle_command.
        self.current_identity_source = "default"
        # The role as ENROLLED (patient/caregiver/admin/known_person), kept
        # separate from current_role, which is the interaction mode it maps to.
        # Permission checks use this one: admin and caregiver both map to
        # caregiver mode, but only this distinguishes them.
        self.current_stored_role = None
        # Admin-only: forces patient mode on demand so the gentle behaviour can
        # be checked without re-enrolling. None means "use the enrolled role".
        self.mode_override = None
        # When the last thing was said or answered, so scheduled prompts can
        # wait rather than talk over a conversation in progress.
        self._last_interaction = 0.0
        # Who was last greeted on arrival, so a greeting fires once per
        # appearance rather than on every recognition tick they remain in
        # view. Reset to None whenever nobody is recognised, so leaving and
        # coming back — including a full app restart — counts as a fresh
        # arrival. See _update_scene_description / _greet_on_arrival.
        self._last_greeted_speaker = None
        # Set by _handle_remember_face() when a "remember me" attempt comes
        # back ambiguous, so the very next utterance can resolve it without
        # making the person repeat the whole command. One-shot: consumed by
        # the next command regardless of whether it actually answers the
        # question, and also expires after REMEMBER_ME_FOLLOWUP_SECONDS so a
        # stale question from minutes ago can never hijack an unrelated later
        # command. See handle_command's early check and _handle_remember_face.
        self._pending_remember_me = None
        # Conversational auto-enrol of an unrecognised face as known_person —
        # the "I don't think we've met, what's your name?" flow. Same
        # one-shot pending + rate-limit shape as _pending_remember_me above:
        #   _pending_conversational_enrol: the question currently awaiting an
        #       answer (name, then optionally relationship) — one-shot,
        #       expiring, consumed regardless of outcome.
        #   _last_unknown_ask_at: when a stranger was last asked their name,
        #       so declining once, or the conversation simply moving on, does
        #       not make the assistant repeat the question every turn they
        #       remain unrecognised.
        #   _unrecognized_face_in_view: refreshed once per scene tick in
        #       _update_scene_description, so the ask-check reads the same
        #       detections resolve_speaker just used rather than re-fetching.
        self._pending_conversational_enrol = None
        self._last_unknown_ask_at = 0.0
        self._unrecognized_face_in_view = False

    def stop(self):
        self.running = False

    # ---------- identity ----------

    def caregiver_names(self):
        """Who the assistant names when it says it is letting someone know.

        Includes admins: whoever set the device up is a real person who can be
        called, and if the only enrolled helper is an admin then an emergency
        would otherwise name nobody at all.
        """
        names = [c["name"] for c in self.profiles.caregivers()]
        names += [a["name"] for a in self.profiles.admins() if a["name"] not in names]
        return names

    @staticmethod
    def _detection_area(detection) -> int:
        """Pixel area of a detection box — a stand-in for how close they are."""
        bbox = detection.get("bbox") or (0, 0, 0, 0)
        try:
            x1, y1, x2, y2 = bbox
            return max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
        except (TypeError, ValueError):
            return 0

    def resolve_speaker(self, detections):
        """Decide who is speaking and which mode to use.

        THE SAFETY RULE: caregiver mode requires a face match. A voice match can
        supply a name but can never on its own promote someone to caregiver
        mode, because the MFCC voice signal is weak (see voice_embedding) and
        the two failure directions are not symmetric:

          caregiver wrongly in patient mode -> a warmer, shorter answer. Fine.
          patient wrongly in caregiver mode -> medical questions answered
                                               instead of redirected. Not fine.

        When nothing is confident, patient mode is the default, for the same
        reason: it is the gentler behaviour and it never improvises about health.
        """
        recognised = [d for d in (detections or [])
                      if d.get("name") and d.get("name") != "unknown"]

        if len(recognised) > 1:
            # More than one person in view. Taking the first in the list made
            # the speaker depend on detection order — the same two people in
            # the other order produced a different answer, so it could name the
            # wrong one at random.
            #
            # The person addressing a device is normally the one nearest it, so
            # the largest face wins. If a voice match names somebody who is
            # actually in view, that beats size: it is direct evidence about who
            # is SPEAKING rather than who is closest.
            recognised.sort(key=self._detection_area, reverse=True)
            heard = getattr(self, "last_voice_match", None)
            if heard:
                for detection in recognised:
                    if detection.get("name", "").lower() == heard.lower():
                        recognised.insert(0, recognised.pop(recognised.index(detection)))
                        print(f"[dialogue-debug] {len(recognised)} people in view; "
                              f"voice picks {detection.get('name')!r}")
                        break
                else:
                    print(f"[dialogue-debug] {len(recognised)} people in view; "
                          f"voice heard {heard!r} who is not among them — "
                          f"using the closest, {recognised[0].get('name')!r}")
            else:
                print(f"[dialogue-debug] {len(recognised)} people in view; "
                      f"using the closest, {recognised[0].get('name')!r}")

        for detection in recognised:
            name = detection.get("name")
            if name and name != "unknown":
                stored = detection.get("role") or self.profiles.role_of(name)
                self.current_stored_role = stored
                # Map the STORED role to an interaction mode. A known_person —
                # enrolled from a photo, granted nothing — resolves to patient
                # mode. Passing the raw stored role through would have been a
                # real hole: "known_person" is not ROLE_PATIENT, so the distress
                # and orientation guards would have been skipped for them, and
                # llm_connector would have handed them caregiver-mode rules.
                return name, ProfileStore.interaction_role(stored), "face"
        self.current_stored_role = None
        return None, ROLE_PATIENT, "default"

    def social_context(self, detections):
        """Who else is in view that the person knows, for the model to mention.

        This is the payoff of photo enrolment: the assistant can say "Sarah's
        here" rather than "there's a person". Only people explicitly enrolled as
        known_person appear here.
        """
        seen = []
        for detection in detections or []:
            name = detection.get("name")
            if not name or name == "unknown":
                continue
            profile = self.profiles.get(name)
            if not profile or profile.get("role") != ROLE_KNOWN_PERSON:
                continue
            rel = profile.get("relationship")
            entry = f"{name} ({rel})" if rel else name
            if entry not in seen:
                seen.append(entry)
        if not seen:
            return None
        who = ", ".join(seen)
        return (f"People you know who are visible right now: {who}. "
                "Greet them naturally by name if it fits the conversation.")

    def identify_by_voice(self, audio):
        """Supplementary only — returns a name or None, never a role."""
        if audio is None:
            return None, 0.0
        names, embeddings = self.profiles.voice_index()
        if not names:
            return None, 0.0
        embedding = voice_embedding.embed(audio)
        if embedding is None:
            return None, 0.0
        return voice_embedding.best_match(embedding, names, embeddings)

    # ---------- scene ----------

    def _update_scene_description(self):
        if self.detection_store is None:
            return
        latest_detections = self.detection_store.get(default=[])

        frame = self.camera.get_latest_frame()
        if frame is None:
            return

        # A person detection present but not matched to any profile — the
        # trigger condition for _maybe_ask_unrecognized_name. Computed here,
        # off the same detections resolve_speaker is about to use, rather
        # than re-fetching detection_store later from inside handle_command.
        self._unrecognized_face_in_view = any(
            d.get("class_name") == "person" and d.get("name") in (None, "unknown")
            for d in latest_detections
        )

        name, role, source = self.resolve_speaker(latest_detections)
        if name:
            self.current_speaker, self.current_role = name, role
            self.current_identity_source = source
            # An admin can force patient mode for testing. Only ever downgrades
            # to the gentler mode, so it can never grant anything.
            if self.mode_override and ProfileStore.has_care_access(self.current_stored_role):
                self.current_role = self.mode_override
        else:
            # Nobody recognised in view: fall back to the safe default rather
            # than keeping a stale caregiver identity from minutes ago.
            self.current_speaker, self.current_role = None, ROLE_PATIENT
            self.current_identity_source = "default"
            self.current_stored_role = None

        self.latest_scene_description = build_scene_description(
            latest_detections, frame.shape[1], frame.shape[0]
        )
        self.latest_social_context = self.social_context(latest_detections)
        self._greet_on_arrival(name)

    # ---------- greeting on arrival ----------

    # Short, warm, and deliberately NOT identical every time — unlike the
    # identity/orientation answers, nothing here is safety-critical, so a
    # little natural variation is more comforting than a script. None open
    # with the person's name, matching the existing patient-mode style rule
    # ("don't open every reply with their name; it starts to sound like a
    # form letter" — see PATIENT_PERSONA in llm_connector.py).
    _SHORT_GREETINGS = (
        "Hi, it's Guide YOU — good to see you.",
        "Hello again, it's good to see you.",
        "Hi there — I'm here if you need me.",
    )

    def _speak_unprompted(self, text: str) -> None:
        """Say something that was not a reply to anything asked.

        Deliberately bypasses _send_response: that method logs a user/AI PAIR
        to memory and conversation_history, which is right for an answer but
        wrong for a greeting nobody prompted — pairing it with self.last_command
        would misrecord an unrelated earlier command as what this greeting was
        "answering".
        """
        print(f"AI (greeting) -> {text}")
        try:
            self.speech_queue.put_nowait(text)
        except Exception:
            pass

    def _greet_on_arrival(self, name):
        """Greet a patient when they are freshly recognised — not on every
        tick they remain in view, and not while a conversation with someone
        else is already in progress.

        Patient-mode ONLY, and gated on the STORED role rather than the
        interaction-mode mapping: a known_person (a photo-enrolled relative)
        also maps to patient-mode tone, but they are not "the patient" and
        have no relationship of their own with Guide YOU to greet them into —
        checking current_stored_role directly (rather than current_role)
        excludes them correctly.
        """
        if name != self._last_greeted_speaker:
            self._last_greeted_speaker = name

            if not name or self.current_stored_role != ROLE_PATIENT:
                return
            if self.is_busy():
                # Arriving mid-conversation with someone else. Skipped rather
                # than queued: by the time any later conversation ends, "just
                # arrived" is no longer true, and an unprompted greeting
                # arriving out of nowhere partway through an unrelated reply
                # would be more startling than useful.
                print(f"[dialogue-debug] {name} arrived mid-conversation; "
                      "skipping the greeting")
                return

            profile = self.profiles.get(name)
            if profile and not profile.get("has_been_introduced"):
                # Safety net for a profile that reached patient status without
                # ever going through capture_and_train's own introduction (for
                # example one created before this feature existed) — this is
                # genuinely their first time hearing from Guide YOU, so it gets
                # the full introduction, not the short greeting.
                self.profiles.mark_introduced(name)
                print(f"[dialogue-debug] {name} recognised for the first time "
                      "with no prior introduction on record — giving the full "
                      "introduction now")
                self._speak_unprompted(PATIENT_INTRO_TEXT)
                return

            self._speak_unprompted(random.choice(self._SHORT_GREETINGS))

    # ---------- speaking ----------

    def _send_response(self, response: str):
        if not response:
            return
        print(f"AI [{self.current_role}] -> {response}")
        try:
            self.speech_queue.put_nowait(response)
        except Exception:
            pass
        self.memory.add_conversation(self.last_command, response, profile=self.current_speaker)
        with self.conversation_lock:
            self.conversation_history.append({"user": self.last_command, "ai": response})
            if len(self.conversation_history) > 6:
                self.conversation_history.pop(0)

    def _ask_llm(self, command: str) -> str:
        frame_bytes = None
        try:
            frame_bytes = self.camera.get_latest_frame_bytes()
        except Exception:
            frame_bytes = None

        return self.llm.ask_with_context(
            command,
            self.latest_scene_description,
            frame_bytes,
            self.layered_memory.get_context_for_llm(
                {"name": self.current_speaker or "Unknown", "persona_type": self.current_role}
            ),
            role=self.current_role,
            speaker_name=self.current_speaker,
            # Real clock and location, so casual mentions of the date in
            # ordinary conversation are grounded rather than invented.
            now_context=orientation.context_line(self.home_location),
            social_context=getattr(self, "latest_social_context", None),
            casual_context=self.weather_context.get_context_line(),
        ) or ""

    # ---------- commands ----------

    # ---------- signals for scheduled prompts ----------

    # How long after the last exchange a conversation still counts as active.
    # Long enough to cover someone gathering their thoughts between sentences.
    CONVERSATION_QUIET_SECONDS = 40.0

    def patient_present(self) -> bool:
        """Is the person the routine prompts are FOR actually here?

        Requires a face. A voice match is too weak to start speaking into a
        room on, and a prompt about lunch aimed at a visiting carer — or at
        nobody — is worse than no prompt.
        """
        return (
            self.current_speaker is not None
            and self.current_identity_source == "face"
            and self.current_stored_role == ROLE_PATIENT
        )

    def is_busy(self) -> bool:
        """Is a conversation in progress, or is the assistant already talking?"""
        if time.time() - self._last_interaction < self.CONVERSATION_QUIET_SECONDS:
            return True
        try:
            if self.speech_queue is not None and not self.speech_queue.empty():
                return True
        except Exception:
            pass
        try:
            if self.command_queue is not None and not self.command_queue.empty():
                return True
        except Exception:
            pass
        return False

    def can_read_alerts(self) -> bool:
        """Alert read-back requires a caregiver confirmed BY FACE.

        Voice alone is not enough, for the same reason it can never grant
        caregiver mode: the MFCC signal is weak, and the alert log contains the
        person's distress moments and help requests. Reading those out to the
        wrong listener — or to the person themselves — is a real harm, so this
        needs the stronger signal.
        """
        return (
            # Checked against the ENROLLED role, not the interaction mode, so an
            # admin who has switched to patient mode for testing does not lose
            # access — and so a known_person, who maps to patient mode, can
            # never gain it.
            ProfileStore.has_care_access(self.current_stored_role)
            and self.current_identity_source == "face"
        )

    def _handle_read_routines(self) -> str:
        if self.routine_store is None:
            return "I don't have any reminders set up."
        schedule = self.routine_store.todays_schedule()
        print(f"[dialogue-debug] read back {len(schedule)} routine(s) to "
              f"{self.current_speaker} ({self.current_stored_role}, by face)")
        return routines.describe_schedule(schedule)

    def _handle_read_alerts(self) -> str:
        entries = alert_log.today()
        summary = alert_log.summarise_for_speech(entries)
        print(f"[dialogue-debug] read back {len(entries)} alert(s) to "
              f"{self.current_speaker} (caregiver, confirmed by face)")
        return summary

    def _handle_who_am_i(self) -> str:
        """Answer from the profile store, never from the model.

        Restored after being lost in the Phase A-C rewrite. Answering this from
        the model would be worse than not answering: with no reliable knowledge
        of who is present it will produce a confident, friendly, invented name,
        and being told the wrong name is genuinely distressing for someone
        already unsure of it.

        Reads the SAME ProfileStore instance that enrolment writes to, so a name
        saved by "remember my face" is available immediately.
        """
        described = self._speaker_identity_sentence()
        if described:
            return described

        known = self.profiles.names()
        if not known:
            # Nobody enrolled at all — say what to do about it, warmly.
            return ("I haven't learned your name yet. If you say "
                    "remember my face, I'll learn who you are.")

        # Someone is enrolled, but the camera cannot confirm this is them.
        # Guessing would be worse than admitting it.
        if len(known) == 1:
            return (f"I think you might be {known[0]}, but I can't see you clearly "
                    "enough to be sure. Come a little closer and I'll know you.")
        return ("I can't see you clearly enough to be sure who you are just now. "
                "Come a little closer and I'll know you.")

    def _speaker_identity_sentence(self):
        """'You're Sarah, Margaret's daughter.' — or None if nobody is known.

        Shared by every question that turns on who is speaking, so they can
        never drift into giving different answers to the same fact.
        """
        if self.current_speaker:
            profile = self.profiles.get(self.current_speaker)
            name = (profile or {}).get("name") or self.current_speaker
            relationship = (profile or {}).get("relationship")
            if relationship:
                # A relationship is always TO THE PERSON BEING SUPPORTED, never
                # to the assistant. This used to say "You're my daughter", which
                # had the assistant claiming a family tie of its own — the same
                # identity confusion the prompt had, hardcoded and wrong every
                # time. Name the patient if we know them, so it is unambiguous.
                #
                # active_profiles(), NOT all_profiles(): an archived duplicate
                # profile (see mark_archived()) can still hold role=patient —
                # that's exactly how "You're bardan, GuideYouLiveTest's son"
                # got spoken once, a name that was never a real relationship,
                # just a leftover test profile that happened to still be
                # findable through this specific lookup after being excluded
                # everywhere else. If no ACTIVE patient exists, the fallback
                # below ("You're {name}, the {relationship}.") is correct —
                # there is no real patient to name, so none should be invented.
                patient = next((p["name"] for p in self.profiles.active_profiles()
                                if p.get("role") == ROLE_PATIENT), None)
                if patient and patient.lower() != name.lower():
                    return f"You're {name}, {patient}'s {relationship}."
                return f"You're {name}, the {relationship}."
            return f"You're {name}. It's good to see you."

        known = self.profiles.names()
        if not known:
            # Nobody enrolled at all — say what to do about it, warmly.
            return ("I haven't learned your name yet. If you say "
                    "remember my face, I'll learn who you are.")

        # Someone is enrolled, but the camera cannot confirm this is them.
        # Guessing would be worse than admitting it.
        if len(known) == 1:
            return (f"I think you might be {known[0]}, but I can't see you clearly "
                    "enough to be sure. Come a little closer and I'll know you.")
        return ("I can't see you clearly enough to be sure who you are just now. "
                "Come a little closer and I'll know you.")

    @staticmethod
    def _spell_count(n: int) -> str:
        """Small numbers as words — this is read aloud, and "2" spoken by a
        synthesiser lands worse than "two"."""
        words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        return words.get(n, str(n))

    def _describe_person(self, name: str) -> str:
        """How to refer to one recognised person out loud.

        A known_person gets their relationship, because that is the whole reason
        they were enrolled — "your niece" is what makes the answer useful rather
        than just a name. A caregiver gets theirs too when it is recorded. The
        person being supported is named plainly; announcing their own role back
        at them would be strange.
        """
        profile = self.profiles.get(name)
        if not profile:
            return name
        rel = profile.get("relationship")
        role = profile.get("role")
        if role == ROLE_KNOWN_PERSON and rel:
            return f"{name}, your {rel}"
        if role == ROLE_CAREGIVER and rel:
            return f"{name}, your {rel}"
        return name

    def _extract_claimed_name(self, command, patterns):
        """The name embedded in an identity question, or None.

        Deciding whether the captured word is a NAME is the whole difficulty.
        "am I Margaret?" and "am I late?" are the same shape, and a stopword
        list cannot enumerate every adjective anyone might use — "late" and
        "funny" both slipped through one.

        So a word counts as a name only if it is either already enrolled, or
        capitalised. Speech-to-text capitalises proper nouns and not ordinary
        adjectives, and the captured word is never at the start of a sentence,
        so sentence-case does not produce false positives. Anything else stays
        ordinary conversation.
        """
        known = {n.lower() for n in self.profiles.names()}
        for pattern in patterns:
            match = pattern.search(command or "")
            if not match:
                continue
            name = (match.group("name") or "").strip(" ,.?!'\"")
            if not name or name.lower() in _NOT_A_NAME:
                continue
            # The assistant's own aliases count however they were transcribed —
            # speech-to-text has no reason to capitalise "guide".
            if (name.lower() in known or name.lower() in _ASSISTANT_ALIASES
                    or name[0].isupper()):
                return name
        return None

    def _handle_identity_claim(self, command):
        """Answer 'are you X' / 'am I X' from recognition, never from the model.

        Identity is already known deterministically from the face, so letting
        the model field these leaves it able to be led into agreeing with a
        false premise — it did exactly that when asked "are you Margaret?" by
        someone who was not Margaret.

        Two distinct questions live in these phrasings and they need different
        answers:
          "are you Margaret?"      -> about the ASSISTANT
          "am I Margaret?"         -> about the SPEAKER
        Assistant patterns are checked first because "am I talking to Margaret"
        would otherwise capture "talking" as a name.
        """
        asked = self._extract_claimed_name(command, _IS_ASSISTANT_NAME)
        if asked is not None:
            return self._answer_about_assistant(asked)

        asked = self._extract_claimed_name(command, _IS_SPEAKER_NAME)
        if asked is not None:
            return self._answer_about_speaker(asked)
        return None

    def _answer_about_assistant(self, asked: str) -> str:
        who = self._speaker_identity_sentence()
        if asked.lower() in _ASSISTANT_ALIASES:
            reply = f"Yes, I'm {ASSISTANT_NAME}, your assistant."
        else:
            # Never hedge. The assistant is not a person and never was.
            reply = f"No, I'm {ASSISTANT_NAME}, your assistant."
        # Correcting the premise is only half of it — say who they actually are,
        # so the answer leaves them oriented rather than just contradicted.
        return f"{reply} {who}" if who else reply

    def _answer_about_speaker(self, asked: str) -> str:
        if not self.current_speaker:
            known = self.profiles.names()
            if not known:
                return ("I haven't learned your name yet. If you say remember "
                        "my face, I'll learn who you are.")
            return ("I can't see you clearly enough to be sure just now. Come a "
                    "little closer and I'll know you.")

        actual = self.profiles.get(self.current_speaker) or {}
        actual_name = actual.get("name") or self.current_speaker
        if asked.lower() == actual_name.lower():
            return f"Yes, you're {actual_name}."

        # They are somebody else. Say so warmly and factually — this is
        # orientation, which the care guidance says to answer directly, and it
        # is not the same as contradicting someone's memories or beliefs.
        return self._speaker_identity_sentence()

    def _handle_mode_override(self, target) -> str:
        """Admin-only. Force patient mode for testing, or clear it.

        Gated on the ENROLLED role, so a patient cannot talk themselves into
        caregiver mode with it. It can only ever move TO the gentler mode, so
        even if the gate were wrong it could not grant anything.
        """
        if not ProfileStore.has_care_access(self.current_stored_role):
            # Not an admin. Say nothing about modes existing — fall through to
            # ordinary conversation.
            return None

        if target == ROLE_PATIENT:
            self.mode_override = ROLE_PATIENT
            self.current_role = ROLE_PATIENT
            print(f"[dialogue-debug] mode override ON: {self.current_speaker} "
                  f"({self.current_stored_role}) is testing patient mode")
            return ("Alright, I'll talk as if I were with the person you support. "
                    "Say normal mode when you want me back.")

        self.mode_override = None
        self.current_role = ProfileStore.interaction_role(self.current_stored_role)
        print(f"[dialogue-debug] mode override OFF: back to {self.current_role}")
        return "Back to normal. What do you need?"

    def _handle_who_is_that(self) -> str:
        """Identify whoever is in view. Answered from recognition, never guessed.

        Deliberately available in every interaction mode: knowing who is in the
        room is orientation support, which is one of the most valuable things
        this app does, not a privileged feature.

        Works identically for a live face and for a photo held up to the camera,
        because photo-enrolled people go through the same recognition path.
        """
        detections = []
        if self.detection_store is not None:
            detections = self.detection_store.get(default=[]) or []

        people = [d for d in detections if d.get("class_name") == "person"]
        if not people:
            return "I don't see anyone right now."

        # Named, de-duplicated, preserving left-to-right order.
        named = []
        for detection in people:
            name = detection.get("name")
            if name and name != "unknown" and name not in named:
                named.append(name)

        if not named:
            # Somebody is there but not recognised. Say so honestly, say how
            # many, and offer the thing that fixes it.
            if len(people) == 1:
                seen = "I can see someone, but I don't recognise them."
            else:
                seen = (f"I can see {self._spell_count(len(people))} people, "
                        "but I don't recognise any of them.")
            return (f"{seen} Would you like me to remember them? "
                    "Just say remember this person.")

        described = [self._describe_person(n) for n in named]
        if len(described) == 1:
            lead = f"That's {described[0]}."
        elif len(described) == 2:
            lead = f"That's {described[0]}, and {described[1]}."
        else:
            lead = "That's " + ", ".join(described[:-1]) + f", and {described[-1]}."

        # If there are more people in frame than we could put a name to, say so
        # rather than implying everyone present has been identified.
        unnamed = len(people) - len(named)
        if unnamed > 0:
            lead += (" There's also someone else I don't recognise."
                     if unnamed == 1 else
                     f" There are also {self._spell_count(unnamed)} people "
                     "I don't recognise.")
        return lead

    def _handle_list_people(self) -> str:
        """Read back who is known, separating users of the system from people
        it simply recognises. A carer checking the setup needs to see which is
        which — 'Sarah' being a caregiver and 'Sarah' being a face in a photo
        mean very different things."""
        users = self.profiles.system_users()
        known = self.profiles.known_people()

        if not users and not known:
            return ("I haven't met anyone yet. Say remember my face and I'll "
                    "learn who you are.")

        sentences = []
        if users:
            parts = []
            for profile in users:
                name = profile["name"]
                role = profile.get("role")
                rel = profile.get("relationship")
                if role == ROLE_ADMIN:
                    parts.append(f"{name}, who looks after this device"
                                 + (f" — your {rel}" if rel else ""))
                elif role == ROLE_CAREGIVER:
                    parts.append(f"{name}, who helps out" + (f" as your {rel}" if rel else ""))
                else:
                    parts.append(name)
            sentences.append("I know " + ", and ".join(parts) + ".")

        if known:
            parts = []
            for profile in known:
                name = profile["name"]
                rel = profile.get("relationship")
                parts.append(f"{name}, your {rel}" if rel else name)
            lead = "I can also recognise " if users else "I can recognise "
            sentences.append(lead + ", and ".join(parts) + ".")

        return " ".join(sentences)

    def _handle_unarchive_profile(self, name: str) -> str:
        """Admin/caregiver only, confirmed by face — caller has already
        checked can_read_alerts(). Restores a profile to live matching
        without ever having touched its stored data."""
        profile = self.profiles.get(name)
        if not profile:
            return f"I don't have anyone enrolled called {name}."
        if not profile.get("archived"):
            return f"{profile['name']} isn't archived."
        self.profiles.mark_archived(profile["name"], False)
        self.face_recognizer.invalidate_cache()
        print(f"[dialogue-debug] {profile['name']} restored from archive by "
              f"{self.current_speaker} ({self.current_stored_role}, confirmed by face)")
        return f"I've restored {profile['name']}'s profile."

    def _handle_set_emergency_contact(self, field: str, value: str) -> str:
        """Admin/caregiver only, confirmed by face — caller (handle_command)
        has already checked can_read_alerts() before calling this.

        Updates the in-memory copy immediately (so a later question in the
        same session sees it right away) and persists to settings.yaml
        (see emergency_contacts.save_contact — a targeted line edit, not a
        full YAML re-dump, so the file's existing comments survive).
        """
        self.emergency_contacts[field] = value
        persisted = emergency_contacts.save_contact(field, value)
        print(f"[dialogue-debug] emergency contact set: {field}={value!r} "
              f"(persisted to settings.yaml: {persisted}) by "
              f"{self.current_speaker} ({self.current_stored_role}, "
              "confirmed by face)")
        return emergency_contacts.confirmation_phrase(field, value)

    def _handle_list_profiles_admin(self) -> str:
        """Full enrolment audit for whoever set the device up: every
        profile's role, relationship, enrolment date, and whether voice and a
        reference photo were actually captured — read-only, gated exactly
        like the alert log (can_read_alerts). Distinct from
        _handle_list_people, which is a short warm summary for the person
        being supported; this is a setup/audit tool for a carer checking the
        device is set up correctly, so it says everything, not just names.

        The photo check reads known_faces/ directly rather than trusting a
        stored flag, so it reflects what is actually on disk right now — the
        one path that can enrol someone with no photo is migration from the
        old pickle store, which never held a source image to save, so a
        missing photo here is real and this is how a carer would notice it.
        """
        profiles = self.profiles.all_profiles()
        if not profiles:
            return "Nobody is enrolled yet."

        lines = []
        for profile in sorted(profiles, key=lambda p: p.get("created") or 0):
            name = profile["name"]
            role = profile.get("role")
            rel = profile.get("relationship")
            created = profile.get("created")
            when = (time.strftime("%d %B %Y", time.localtime(created))
                    if created else "an unknown date")
            has_voice = "yes" if profile.get("voice_embedding") else "no"
            photo_path = os.path.join(KNOWN_FACES_DIR, f"{safe_filename(name)}.jpg")
            has_photo = "yes" if os.path.exists(photo_path) else "no"
            detail = f"{name} (archived), {role}" if profile.get("archived") else f"{name}, {role}"
            if rel:
                detail += f" ({rel})"
            detail += f", enrolled {when}, voice captured: {has_voice}, photo on file: {has_photo}"
            lines.append(detail)

        print(f"[dialogue-debug] profile audit ({len(profiles)} profile(s)) read back "
              f"to {self.current_speaker} (caregiver, confirmed by face)")
        count = self._spell_count(len(profiles)) if len(profiles) <= 6 else str(len(profiles))
        return (f"There {'is' if len(profiles) == 1 else 'are'} {count} enrolled "
                f"profile{'s' if len(profiles) != 1 else ''}. " + ". ".join(lines) + ".")

    # How far above the applicable tolerance a distance can sit and still be
    # "possibly this person, read poorly" rather than "clearly someone else".
    # Not arbitrary: FaceEncoder's own solo-tolerance comment documents a
    # real, measured stranger scoring 0.4421 against a 0.42 ceiling — a gap of
    # only ~0.02. A band this size (0.05) means a genuinely new person will
    # sometimes land inside it and get asked rather than silently enrolled —
    # that is the explicit, stated trade-off for this feature: erring toward
    # asking is correct here, not a shortcut. See _handle_remember_face.
    REMEMBER_ME_AMBIGUOUS_BAND = 0.05

    def _handle_remember_face(self):
        """"remember me" / "remember my face": check whether this face is
        ALREADY confidently recognised before enrolling, so repeating the
        command does not create a duplicate profile for someone the device
        already knows.

        Reuses the exact matching signal recognize() uses for live
        recognition (FaceEncoder.match_details — same YuNet crop, same
        encoder, same solo/margin tolerance already measured and fixed), not
        a separate or looser check. Three outcomes:

          CONFIDENT match   -> say so, do NOT enrol. No popups, no samples.
          Clearly no match  -> proceed to the existing enrolment flow exactly
                                as before (including "nobody enrolled yet",
                                which is its own clearly-no-match case).
          AMBIGUOUS         -> a near-miss that could be a poor-quality read
                                of someone already enrolled, or two enrolled
                                people plausibly close. Ask rather than
                                silently guessing either way — proceeding to
                                enrol here is exactly how a duplicate profile
                                for an already-known person would get created.

        A frame with no face found at all (details is None) is a different
        failure from "no confident match" — it means this one pre-check frame
        happened to catch nobody clearly, not that nobody is enrolled. That
        falls through to the normal enrolment flow too, which takes several
        samples over a couple of seconds and is far more robust than betting
        everything on this single check frame.
        """
        frame = self.camera.get_latest_frame()
        details = None
        if frame is not None:
            details = self.face_recognizer.encoder.match_details(frame, verbose=True)

        if details is not None and details.get("name"):
            name = details["name"]
            distance = details["distance"]
            ceiling = details["ceiling"]
            rival_name = details.get("rival_name")
            rival_distance = details.get("rival_distance")

            if details["confident"]:
                print(f"[remember-me] confident match: {name} at {distance:.4f} "
                      f"(ceiling {ceiling}) — not re-enrolling")
                # Whatever is said next already counts as this arrival's
                # greeting, for the same reason a fresh enrolment sets this —
                # without it the very next recognition tick would speak a
                # second greeting on top of this reply.
                self._last_greeted_speaker = name
                profile = self.profiles.get(name)
                relationship = (profile or {}).get("relationship")
                if relationship:
                    return f"I already know you, you're {name}, the {relationship}!"
                return f"I already know you, you're {name}!"

            near_miss = distance < (ceiling + self.REMEMBER_ME_AMBIGUOUS_BAND)
            rival_ambiguous = (
                rival_name is not None and rival_distance is not None
                and (rival_distance - distance) < FACE_MATCH_MARGIN
            )
            if near_miss or rival_ambiguous:
                print(f"[remember-me] AMBIGUOUS: closest {name} at {distance:.4f} "
                      f"(ceiling {ceiling}, band {self.REMEMBER_ME_AMBIGUOUS_BAND}) "
                      f"— asking rather than guessing")
                # The very next utterance gets one chance to resolve this —
                # see REMEMBER_ME_FOLLOWUP_SECONDS and handle_command's early
                # check. Recorded now, while `name` is the actual closest
                # candidate from THIS read, not re-derived later from a stale
                # or absent frame.
                self._pending_remember_me = {
                    "expires_at": time.time() + self.REMEMBER_ME_FOLLOWUP_SECONDS,
                    "candidate": name,
                }
                return ("I'm not sure if we've met — should I remember you as "
                        "someone new, or do you already have a profile?")

            print(f"[remember-me] closest {name} at {distance:.4f} (ceiling "
                  f"{ceiling}) — clearly not a match, proceeding to enrol")

        # No confident or ambiguous match — proceed exactly as enrolment
        # always has.
        return self._enrol_new_person()

    # How long the answer to an ambiguous "remember me" question stays live
    # for the NEXT utterance to resolve. Long enough to actually answer a
    # spoken question, short enough that an unrelated command minutes later
    # can never be misread as answering it.
    REMEMBER_ME_FOLLOWUP_SECONDS = 30.0

    def _resolve_pending_remember_me(self, command: str):
        """Consumes self._pending_remember_me (one-shot, regardless of
        outcome) and returns a response if `command` answers the ambiguous
        question just asked, or None if it doesn't — in which case the
        caller continues routing `command` normally, exactly as if no
        question were pending.
        """
        pending = self._pending_remember_me
        self._pending_remember_me = None
        if pending is None or time.time() >= pending["expires_at"]:
            return None

        if _REMEMBER_ME_FOLLOWUP_NEW.search(command):
            print(f"[remember-me] follow-up {command!r} resolves the "
                  "ambiguity as NEW -> enrolling")
            return self._enrol_new_person()

        if _REMEMBER_ME_FOLLOWUP_KNOWN.search(command):
            name = pending["candidate"]
            print(f"[remember-me] follow-up {command!r} resolves the "
                  f"ambiguity as ALREADY KNOWN -> best available match {name!r}")
            profile = self.profiles.get(name)
            if not profile:
                # The candidate's profile vanished between the question and
                # the answer (e.g. archived in the meantime) — say so rather
                # than confidently naming someone that no longer resolves.
                return "I'm still not sure — your carer may need to check my profiles."
            relationship = profile.get("relationship")
            self._last_greeted_speaker = name
            if relationship:
                return f"Of course — good to see you again, {name}, the {relationship}."
            return f"Of course — good to see you again, {name}."

        return None

    def _enrol_new_person(self):
        """The actual capture-and-save flow, shared by the two paths that can
        reach it: a "remember me" that was clearly not a match, and an
        ambiguous one resolved as "someone new" by _resolve_pending_remember_me.
        Always captures a FRESH frame — this may run a command or more after
        the frame that triggered the original request.

        The trainer speaks its own outcome (the first-time introduction or
        the plain confirmation), so returning empty avoids talking over it,
        matching _handle_remember_person's pattern.
        """
        frame = self.camera.get_latest_frame()
        if self.mic_listener is not None:
            self.mic_listener.pause()
        try:
            # The camera is handed in so enrolment can take several samples a
            # moment apart rather than betting everything on one frame.
            enrolled_name = self.face_trainer.capture_and_train(frame, camera=self.camera)
        finally:
            if self.mic_listener is not None:
                self.mic_listener.resume()
        self._drain_command_queue()
        if enrolled_name:
            # Whatever the trainer just said (the full introduction, or the
            # plain confirmation) already counts as this person's greeting for
            # this arrival — without this, the very next recognition tick sees
            # them as a "new" arrival and immediately speaks a second greeting
            # on top of the one that just finished.
            self._last_greeted_speaker = enrolled_name
            return ""
        return "I couldn't save that. We can try again whenever you like."

    def _handle_remember_person(self):
        """Enrol somebody else, usually from a photo held up to the camera.

        The trainer speaks its own outcome — including the specific "hold it
        steadier, reduce the glare" retry advice, which is the likely failure
        and one the person can act on. Returning empty avoids talking over it.
        """
        if self.mic_listener is not None:
            self.mic_listener.pause()
        frame = self.camera.get_latest_frame()
        try:
            self.face_trainer.capture_known_person(frame, camera=self.camera)
        finally:
            if self.mic_listener is not None:
                self.mic_listener.resume()
        self._drain_command_queue()
        return ""

    # ---------- conversational auto-enrolment (known_person only) ----------

    # How long a stranger's "what's your name?" (and the follow-up "how do
    # you know X?") stays live for the very next utterance to answer. Same
    # reasoning as REMEMBER_ME_FOLLOWUP_SECONDS: long enough to actually
    # answer a spoken question, short enough that an unrelated command
    # minutes later is never mistaken for answering it.
    CONVERSATIONAL_ENROL_FOLLOWUP_SECONDS = 45.0

    # How long after asking (or after being declined, or the question simply
    # expiring unanswered) before an unrecognised face gets asked again.
    # Several minutes, deliberately: the point is to feel natural once, not
    # to interrogate someone who didn't want to answer or was mid-thought
    # about something else entirely.
    UNKNOWN_NAME_ASK_COOLDOWN_SECONDS = 300.0

    def _maybe_ask_unrecognized_name(self):
        """Naturally ask an unrecognised face's name, mid-conversation.

        No trigger phrase: this fires on ANY utterance from someone the
        camera cannot match to a profile, not "remember this person" or any
        other specific phrase. Rate-limited by _last_unknown_ask_at so
        declining once, or the conversation just moving on to something
        else, does not make the assistant repeat itself every turn.

        Returns a sentence to APPEND to this turn's ordinary answer, or
        None — this never replaces the answer to whatever was actually
        asked; see _with_name_prompt, the only caller.
        """
        if self._pending_conversational_enrol is not None:
            # Already mid-question; the resolver handles the next utterance,
            # not this ask-again path.
            return None
        if self.current_stored_role is not None:
            return None  # somebody IS recognised as the speaker
        if not self._unrecognized_face_in_view:
            return None  # nobody's unrecognised face is even in view
        if (time.time() - self._last_unknown_ask_at) < self.UNKNOWN_NAME_ASK_COOLDOWN_SECONDS:
            return None

        self._last_unknown_ask_at = time.time()
        self._pending_conversational_enrol = {
            "expires_at": time.time() + self.CONVERSATIONAL_ENROL_FOLLOWUP_SECONDS,
            "stage": "name",
            "name": None,
        }
        print("[conversational-enrol] unrecognised face engaging in "
              "conversation — asking their name")
        return "I don't think we've met — what's your name?"

    def _with_name_prompt(self, response: str) -> str:
        """Append the "what's your name?" ask to an ordinary conversational
        reply, if this is a good moment (see _maybe_ask_unrecognized_name).
        Never replaces the reply itself — only ever used on the two paths
        that count as "just talking": the memory processor and the model.
        """
        name_prompt = self._maybe_ask_unrecognized_name()
        if not name_prompt:
            return response
        return f"{response} {name_prompt}".strip() if response else name_prompt

    @staticmethod
    def _title_name(word: str) -> str:
        return word[:1].upper() + word[1:] if word else word

    def _extract_conversational_name(self, command):
        """The name out of a reply to "what's your name?".

        Deliberately looser than _extract_claimed_name (used for identity
        claims elsewhere): that one guards against misreading ordinary
        conversation as an identity claim, but this is only ever checked
        against the single utterance immediately after that specific
        question was asked, so there is no ordinary conversation it could be
        confused with.
        """
        command = (command or "").strip()
        for pattern in _CONVERSATIONAL_NAME_PATTERNS:
            match = pattern.search(command)
            if match:
                name = (match.group("name") or "").strip(" ,.?!'\"")
                if name and name.lower() not in _NOT_A_NAME:
                    return self._title_name(name)

        # A bare reply — "John", "John Smith" — with no framing phrase at
        # all. Accepted only when short and free of anything that reads as a
        # real sentence (checked against every word, not just the first), so
        # "I don't want to say" is never mistaken for someone whose name is
        # "I".
        words = command.strip(" .!?").split()
        if 1 <= len(words) <= 3 and all(w.replace("'", "").isalpha() for w in words):
            normalized = [w.replace("'", "").lower() for w in words]
            if not any(w in _BARE_NAME_STOPWORDS for w in normalized):
                return self._title_name(words[0])
        return None

    def _extract_relationship(self, command):
        """A relationship word from a whitelist — see _RELATIONSHIP_PATTERN's
        comment for why a whitelist, not a blacklist, is the guard here."""
        match = _RELATIONSHIP_PATTERN.search(command or "")
        return match.group("rel").lower() if match else None

    def _current_patient_name(self):
        return next(
            (p["name"] for p in self.profiles.active_profiles()
             if p.get("role") == ROLE_PATIENT),
            None,
        )

    def _privileged_claim_redirect(self, name):
        """The response to ANY claim of caregiver/admin/patient status made
        by someone the camera does not recognise, at any stage of
        conversational enrolment. Warm, but structurally incapable of
        granting the claim: by the time this can be called the profile (if
        any) is already saved as known_person — see
        FaceTrainer.capture_conversational_person, which has no role
        parameter at all — so there is nothing left here that COULD
        escalate it.
        """
        patient = self._current_patient_name()
        who = patient or "the person I support"
        print(f"[conversational-enrol] privileged-role claim from an "
              f"unrecognised face (name={name!r}) — enrolled as known_person "
              f"only (or not at all), redirected to admin")
        greeting = f"I'll remember you as {name}. " if name else ""
        return (f"{greeting}If you're {who}'s caregiver, ask an admin to set "
                "that up so I can help you properly.")

    def _enrol_conversational_person(self, name: str, relationship=None) -> Optional[str]:
        """Auto-enrol an unrecognised face as known_person from the current
        conversation — no popup, no trigger phrase. See
        FaceTrainer.capture_conversational_person for the hard role guard
        that makes this structurally unable to grant anything else.
        """
        frame = self.camera.get_latest_frame()
        if self.mic_listener is not None:
            self.mic_listener.pause()
        try:
            enrolled_name = self.face_trainer.capture_conversational_person(
                frame, name, camera=self.camera, relationship=relationship
            )
        finally:
            if self.mic_listener is not None:
                self.mic_listener.resume()
        self._drain_command_queue()
        if enrolled_name:
            # Same reasoning as every other enrolment path: whatever is said
            # next already counts as this arrival's greeting, so the very
            # next recognition tick does not speak a second one on top of it.
            self._last_greeted_speaker = enrolled_name
        return enrolled_name

    def _resolve_pending_conversational_enrol(self, command: str):
        """Consumes self._pending_conversational_enrol (one-shot, regardless
        of outcome) and returns a response if `command` answers the question
        just asked, or None if it doesn't — in which case the caller
        continues routing `command` normally, exactly as if no question were
        pending. Same shape as _resolve_pending_remember_me.

        CRITICAL GUARD: no matter what is said here, the role handed to
        FaceTrainer.capture_conversational_person is ALWAYS known_person —
        that method does not even accept a role argument. A claim to be a
        caregiver, admin, or the patient is acknowledged warmly (see
        _privileged_claim_redirect) but never once checked against anything
        that could grant it.
        """
        pending = self._pending_conversational_enrol
        self._pending_conversational_enrol = None
        if pending is None or time.time() >= pending["expires_at"]:
            return None

        claims_privilege = bool(_PRIVILEGED_ROLE_CLAIM.search(command or ""))

        if pending["stage"] == "name":
            if any(p.search(command or "") for p in _DECLINE_NAME_PATTERNS):
                print("[conversational-enrol] declined to give a name — "
                      "not asking again for a while")
                return "That's alright, no need."

            name = self._extract_conversational_name(command)
            if not name:
                if claims_privilege:
                    # No name given, but a privileged claim was made anyway —
                    # still must never grant it, and nothing was enrolled
                    # since a nameless face is never saved (see
                    # capture_conversational_person).
                    return self._privileged_claim_redirect(None)
                # Didn't sound like a name at all — don't hijack an unrelated
                # reply as if it answered the question; let handle_command
                # continue routing it normally instead.
                return None

            enrolled = self._enrol_conversational_person(name)
            if not enrolled:
                return "Sorry, I couldn't see you clearly enough just then — never mind for now."

            if claims_privilege:
                return self._privileged_claim_redirect(enrolled)

            # One more natural beat, entirely optional — see stage
            # "relationship" below. Not required: if this next question goes
            # unanswered or gets no relationship word, the known_person
            # enrolment above already stands complete on its own.
            patient = self._current_patient_name()
            self._pending_conversational_enrol = {
                "expires_at": time.time() + self.CONVERSATIONAL_ENROL_FOLLOWUP_SECONDS,
                "stage": "relationship",
                "name": enrolled,
            }
            if patient:
                return f"Nice to meet you, {enrolled}. How do you know {patient}?"
            return f"Nice to meet you, {enrolled}."

        if pending["stage"] == "relationship":
            name = pending["name"]
            if claims_privilege:
                return self._privileged_claim_redirect(name)
            relationship = self._extract_relationship(command)
            if relationship:
                self.profiles.upsert(name=name, role=ROLE_KNOWN_PERSON, relationship=relationship)
                print(f"[conversational-enrol] {name} relationship captured: "
                      f"{relationship!r}")
            # No relationship offered — that's fine, proceed exactly as told;
            # the name-only enrolment from the previous turn is already
            # complete and valid. Falls through to ordinary handling of
            # whatever they actually said.
            return None

        return None

    def _drain_command_queue(self):
        drained = 0
        try:
            while True:
                self.command_queue.get_nowait()
                drained += 1
        except queue.Empty:
            pass
        if drained:
            print(f"[dialogue-debug] drained {drained} stale queued command(s)")

    def _handle_sound_event(self, event: str) -> str:
        if event == "doorbell":
            return "I think someone's at the door."
        if event == "phone_ringing":
            return "I think the phone is ringing."
        if event == "glass_breaking":
            return "I heard something break. Please be careful where you step."
        if event == "alarm_clock":
            return "That's the alarm going off."
        return ""

    # ---------- main loop ----------

    def run(self):
        self.running = True
        print("DEBUG: Universal Dialogue Processor ready!")
        while self.running:
            try:
                self._run_once()
            except Exception as exc:
                print(f"[dialogue-debug] unhandled error in dialogue loop: {exc}")
                traceback.print_exc()

    def _run_once(self):
        if self.sound_queue is not None:
            try:
                event = self.sound_queue.get_nowait()
                if event:
                    response = self._handle_sound_event(event)
                    if response:
                        self.last_command = f"sound event: {event}"
                        self._send_response(response)
            except queue.Empty:
                pass

        self._update_scene_description()

        try:
            command = self.command_queue.get(timeout=1.0)
        except queue.Empty:
            return

        if not command:
            return
        self.last_command = command
        self._last_interaction = time.time()

        # Who does this utterance SOUND like? Only consulted when several
        # recognised people are in view, and never enough on its own to grant
        # anything — see resolve_speaker.
        self.last_voice_match = None
        if self.mic_listener is not None:
            audio = getattr(self.mic_listener, "last_utterance_audio", None)
            if audio is not None:
                heard, score = self.identify_by_voice(audio)
                if heard:
                    self.last_voice_match = heard
                    print(f"[dialogue-debug] voice sounds like {heard!r} ({score:.3f})")

        # Anything memorable in what was just said gets written to layered
        # memory BEFORE answering, so even this turn's LLM call sees it. This
        # is the write path the 8-layer system was missing — it was read on
        # every turn and written by nothing.
        try:
            self.layered_memory.learn_from_utterance(command, self.current_speaker)
        except Exception as exc:
            print(f"[memory] learn failed (non-fatal): {exc}")

        response = self.handle_command(command)
        latency.mark(latency.RESPONSE_READY)
        self._send_response(response)

    def handle_command(self, command: str) -> str:
        """Route one utterance. Deterministic safety guards come first.

        Separated from the queue loop so it can be exercised directly in tests
        without a microphone, a camera, or a running thread.
        """
        speaker = self.current_speaker
        role = self.current_role
        carers = self.caregiver_names()

        # 1. Emergency only — physical danger, medical crisis, urgent summons.
        #    Deliberately narrow. Everyday requests that merely contain the word
        #    "help" fall straight through to conversation, because an alert that
        #    fires on "help me find my glasses" is one that gets ignored by the
        #    time somebody has actually fallen.
        matched = guards.is_emergency(command, carers)
        if matched:
            # Don't tell Sarah we're letting Sarah know.
            others = [c for c in carers if c.lower() != (speaker or "").lower()]
            response = guards.emergency_acknowledgement(others)
            alert_log.record(alert_log.EMERGENCY, speaker, role, command, response,
                             detail={"matched": matched})
            return response

        # 2. Distress or disorientation, for the person being supported.
        #    Logged so a caregiver can see these moments later; nothing more
        #    automated than that, by design.
        #
        #    "Where am I" is both at once: a genuine sign of disorientation the
        #    caregiver should see, AND a question that deserves a real answer.
        #    Doing only the first — which is what happened before this — left
        #    someone asking where they are and getting sympathy instead of an
        #    answer. So: log it as distress, but lead with the factual answer.
        if role == ROLE_PATIENT:
            matched = guards.is_distress(command, carers)
            if matched:
                # The orientation answers already end warmly, so they are used
                # as-is rather than having reassurance appended onto them.
                oriented = orientation.answer(command, self.home_location)
                response = oriented or guards.distress_reassurance(speaker, carers)
                alert_log.record(alert_log.DISTRESS, speaker, role, command, response,
                                 detail={"matched": matched, "answered_orientation": bool(oriented)})
                return response

        # 3. Anything medical goes to a human, for every speaker, and is logged
        #    rather than silently deflected.
        matched = guards.is_medical_question(command)
        if matched:
            response = guards.MEDICAL_REDIRECT
            alert_log.record(alert_log.MEDICAL_QUESTION, speaker, role, command, response,
                             detail={"matched": matched})
            return response

        # 3b. Resolving an ambiguous "remember me" question, if one is still
        #     pending from the previous turn. Checked AFTER emergency/
        #     distress/medical (safety always wins over this — a real
        #     emergency must never be swallowed as an answer to an unrelated
        #     earlier question, even in the contrived case where it happens to
        #     contain the word "already"), but before everything else, so the
        #     answer is spent whether or not it actually resolves anything.
        resolved = self._resolve_pending_remember_me(command)
        if resolved is not None:
            return resolved

        # 3c. Resolving the "what's your name?" question, if one is still
        #     pending from the previous turn — see
        #     _maybe_ask_unrecognized_name. Same shape and same priority
        #     reasoning as 3b: checked after emergency/distress/medical, one-
        #     shot, consumed whether or not it actually resolves anything.
        resolved = self._resolve_pending_conversational_enrol(command)
        if resolved is not None:
            return resolved

        # 4. Orientation — day, date, time, place. Computed from the clock and
        #    from configured location, never generated. The model will happily
        #    invent "you're at home" and a plausible date; for someone using
        #    this to stay oriented, a fluent wrong answer is worse than none,
        #    and it must be identical every time it is asked.
        oriented = orientation.answer(command, self.home_location)
        if oriented:
            return oriented

        # 4b. SETTING an emergency contact — admin/caregiver only, confirmed
        #     by face, same gate as the alert log and the profile audit.
        #     Checked BEFORE the plain ask/answer below: "the hospital number
        #     is 555-1234" also contains the bare phrase "hospital number",
        #     which the ask patterns match on their own, so SET must win first
        #     or a caregiver setting the number would just be told it's blank.
        set_attempt = emergency_contacts.parse_set_command(command)
        if set_attempt:
            field, value = set_attempt
            if self.can_read_alerts():
                return self._handle_set_emergency_contact(field, value)
            # Same reasoning as every other declined caregiver-only command
            # here: falls through to ordinary conversation rather than
            # announcing a refusal at the person being supported.
            print(f"[dialogue-debug] emergency contact set declined "
                  f"(role={self.current_stored_role} source={self.current_identity_source})")

        # 4c. ASKING what a contact number is — honest from whatever is
        #     configured, never guessed (see safety/emergency_contacts.py).
        #     Available to every speaker: knowing a caregiver-verified number
        #     is safe for anyone to hear, unlike setting one.
        contact_reply = emergency_contacts.answer(command, self.emergency_contacts)
        if contact_reply:
            return contact_reply

        # 5. Ordinary commands.
        if any(p.search(command) for p in _READ_ROUTINES_PATTERNS):
            if self.can_read_alerts():
                return self._handle_read_routines()
            # Same reasoning as the alert log: not a command for the person
            # being supported, so it falls through to ordinary conversation
            # rather than announcing a refusal at them.
            print(f"[dialogue-debug] routine read-back declined "
                  f"(role={self.current_stored_role} "
                  f"source={self.current_identity_source})")

        if any(p.search(command) for p in _READ_ALERTS_PATTERNS):
            if self.can_read_alerts():
                return self._handle_read_alerts()
            # Not a confirmed caregiver. Deliberately fall through to normal
            # conversation rather than refusing: telling the person being
            # supported "you're not allowed to hear that" would be confusing
            # and slightly alarming, and reading them a list of their own
            # distress moments would be worse. It simply isn't a command for
            # them, so it is treated as ordinary talk.
            print(f"[dialogue-debug] alert read-back declined "
                  f"(role={self.current_role} source={self.current_identity_source})")

        # "Who is that" is checked BEFORE "who am I": "who is this" and "who am
        # I" are distinct questions, and a mis-heard "who's this" must not be
        # answered with the speaker's own name.
        if any(p.search(command) for p in _MODE_PATIENT_PATTERNS):
            reply = self._handle_mode_override(ROLE_PATIENT)
            if reply:
                return reply
        if any(p.search(command) for p in _MODE_NORMAL_PATTERNS):
            reply = self._handle_mode_override(None)
            if reply:
                return reply

        # Before both who-is-that and who-am-I: "are you Margaret?" contains
        # neither, and a leading question about identity must never reach the
        # model, which can be talked into agreeing with it.
        identity_reply = self._handle_identity_claim(command)
        if identity_reply:
            return identity_reply

        if any(p.search(command) for p in _WHO_ARE_YOU_PATTERNS):
            return (f"I'm {ASSISTANT_NAME}, your assistant. I'm here to keep "
                    "you company and help out.")

        if any(p.search(command) for p in _WHO_IS_THAT_PATTERNS):
            return self._handle_who_is_that()

        if any(p.search(command) for p in _WHO_AM_I_PATTERNS):
            return self._handle_who_am_i()

        if any(p.search(command) for p in _LIST_PROFILES_ADMIN_PATTERNS):
            if self.can_read_alerts():
                return self._handle_list_profiles_admin()
            # Same reasoning as the alert log and routine read-back: not a
            # command for the person being supported, so it falls through to
            # ordinary conversation rather than announcing a refusal at them.
            print(f"[dialogue-debug] profile audit declined "
                  f"(role={self.current_stored_role} source={self.current_identity_source})")

        for pattern in _UNARCHIVE_PATTERNS:
            match = pattern.search(command)
            if not match:
                continue
            if self.can_read_alerts():
                return self._handle_unarchive_profile(match.group("name"))
            print(f"[dialogue-debug] unarchive declined "
                  f"(role={self.current_stored_role} source={self.current_identity_source})")
            break

        if any(p.search(command) for p in _LIST_PEOPLE_PATTERNS):
            return self._handle_list_people()

        # Checked BEFORE "remember my face": "remember this person" contains
        # neither "my" nor "face", but a mis-heard variant could overlap, and
        # enrolling someone else must not be mistaken for self-enrolment.
        if any(p.search(command) for p in _REMEMBER_PERSON_PATTERNS):
            return self._handle_remember_person()

        if any(p.search(command) for p in _REMEMBER_ME_PATTERNS):
            return self._handle_remember_face()

        memory_response = self.memory_processor.process_command(command, None)
        if memory_response:
            return self._with_name_prompt(memory_response)

        # 6. Everything else goes to the model, in the resolved mode. This is
        #    also where an unrecognised face who is just talking naturally —
        #    no trigger phrase — gets asked their name; see
        #    _maybe_ask_unrecognized_name / _with_name_prompt.
        return self._with_name_prompt(self._ask_llm(command))
