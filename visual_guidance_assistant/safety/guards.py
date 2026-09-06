"""Deterministic phrase guards, checked BEFORE the model is ever consulted.

Why not just instruct the model? Because for the paths that matter — someone
asking whether to take their tablets, someone saying they are frightened — a
prompt instruction is a strong tendency, not a guarantee. Unusual phrasing, a
long conversation, or a bad sampling draw can all erode it. These checks are
plain string matching: they cannot be talked out of firing, they behave
identically every single time, and they are readable by a clinician or family
member who wants to know exactly what the device will do.

The model still gets the same instructions as defence in depth (see
llm_connector). This layer is the floor, not the ceiling.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

# --- medical / medication -------------------------------------------------
# Anything here is redirected to a human and logged. The assistant must never
# improvise an answer about medication, symptoms, or treatment.
_MEDICAL_PATTERNS = [
    r"\b(medication|medicine|medicines|meds|tablet|tablets|pill|pills|capsule|capsules)\b",
    r"\b(dose|dosage|prescription|prescribed|insulin|injection)\b",
    r"\bshould i (take|stop|skip|double)\b",
    r"\b(have i|did i) (taken|had) my\b",
    r"\bhow many .*(take|tablet|pill)\b",
    r"\b(diagnos|symptom|side effect)\w*\b",
    r"\bwhat.s wrong with me\b",
    r"\b(do|have) i (have|got) (dementia|alzheimer|cancer|diabetes)\w*\b",
    r"\b(blood pressure|blood sugar|heart rate|temperature)\b.*\b(normal|okay|ok|high|low|should)\b",
    r"\b(am i|is it) (sick|ill|dying|unwell)\b",
]

# --- three tiers ----------------------------------------------------------
# Asking for help is an ordinary thing to do many times a day. Treating the
# word "help" as an alarm meant "I need help going to the shop" summoned a
# carer, which is both useless and actively harmful: an alert that fires on
# routine requests is an alert people learn to ignore, and it is then not there
# when someone has actually fallen.
#
# So the bar for EMERGENCY is deliberately high and specific, and anything
# ambiguous falls to EVERYDAY, where the assistant simply helps.
TIER_EVERYDAY = "everyday"
TIER_DISTRESS = "distress"
TIER_EMERGENCY = "emergency"

# Only genuine physical danger, medical crisis, or an unmistakable call for
# urgent human help. Every one of these describes something that has already
# gone wrong — not something someone wants doing.
_EMERGENCY_PATTERNS = [
    # falls and being unable to move
    r"\bi.?(ve| have) fallen\b",
    r"\bi fell\b",
    r"\bi.?(ve| have) had a fall\b",
    r"\b(can.?t|cannot) get up\b",
    r"\b(can.?t|cannot) move\b",
    # medical crisis
    r"\bchest pain\b",
    r"\bpain in my chest\b",
    r"\b(can.?t|cannot) breathe\b",
    r"\bi.?m bleeding\b",
    r"\bi am bleeding\b",
    r"\bheart attack\b",
    r"\bi think i.?m having a stroke\b",
    r"\bi.?ve been hurt\b",
    r"\bi.?m hurt\b",
    r"\bi.?ve hurt myself\b",
    # explicit declarations and summons
    r"\bthis is an emergency\b",
    r"\bit.?s an emergency\b",
    r"\bcall an ambulance\b",
    r"\bcall 999\b",
    r"\bcall 911\b",
    r"\bcall the (doctor|hospital|paramedics)\b",
    # an unmistakable cry, as opposed to a request for a task
    r"\bsomebody help\b",
    r"\bsomeone help me\b",
    r"\bhelp me[!]+",
]

# A request for help that has an object — something the person wants DONE — is
# an everyday request no matter how it is phrased. "I need help getting to the
# shop", "help me with the telly", "I need help finding my glasses" are all
# ordinary life, and the assistant should just help.
#
# This is checked BEFORE the emergency list, so a task complement always wins.
_EVERYDAY_TASK = re.compile(
    r"\b(help|helping)\b[^.?!]{0,40}?\b("
    r"find|finding|get|getting|go|going|with|open|opening|put|putting|carry|carrying|"
    r"reach|reaching|look|looking|choose|choosing|pick|picking|read|reading|remember|"
    r"remembering|make|making|cook|cooking|clean|cleaning|write|writing|call|calling|"
    r"work|working|use|using|understand|understanding|decide|deciding|plan|planning|"
    r"dress|dressing|wash|washing|eat|eating|order|ordering|buy|buying"
    r")\b",
    re.IGNORECASE,
)

# A bare "I need help" with nothing else is genuinely ambiguous. Per the design
# rule it resolves to EVERYDAY: the assistant asks what they need, warmly. If it
# really is an emergency, the answer to that question will say so and will match
# the list above. Guessing "emergency" here is what produced the false alarms.
_BARE_HELP = re.compile(r"^\W*(i need help|help me|i want help|i need some help)\W*$", re.IGNORECASE)

# --- distress / disorientation -------------------------------------------
_DISTRESS_PATTERNS = [
    r"\bi don.?t know where i am\b",
    r"\bwhere am i\b",
    r"\bi.?m (scared|frightened|afraid|terrified)\b",
    r"\bi am (scared|frightened|afraid|terrified)\b",
    r"\bi.?m lost\b",
    r"\bi want to go home\b",
    r"\bi don.?t know what.?s happening\b",
    r"\bi don.?t know who you are\b",
    r"\bi.?m (all )?alone\b",
    r"\bnobody.?s here\b",
    r"\bno one is here\b",
    r"\bsomething.?s wrong\b",
    r"\bi.?m confused\b",
]

_MEDICAL_RE = [re.compile(p, re.IGNORECASE) for p in _MEDICAL_PATTERNS]
_EMERGENCY_RE = [re.compile(p, re.IGNORECASE) for p in _EMERGENCY_PATTERNS]
_DISTRESS_RE = [re.compile(p, re.IGNORECASE) for p in _DISTRESS_PATTERNS]


def is_medical_question(text: str) -> Optional[str]:
    """Return the matched pattern, or None. Applies to every speaker: the
    assistant should not be improvising medical answers for anyone."""
    for pattern in _MEDICAL_RE:
        if pattern.search(text or ""):
            return pattern.pattern
    return None


def is_everyday_task(text: str) -> bool:
    """Is this a request to help DO something? Ordinary life, handled in chat."""
    return bool(_EVERYDAY_TASK.search(text or "") or _BARE_HELP.match(text or ""))


def is_emergency(text: str, caregiver_names: Iterable[str] = ()) -> Optional[str]:
    """Physical danger, medical crisis, or an urgent summons — nothing less.

    A task complement always wins: "I need help getting to the shop" is a trip
    to the shop, not a fall.
    """
    body = text or ""
    if is_everyday_task(body):
        return None
    for pattern in _EMERGENCY_RE:
        if pattern.search(body):
            return pattern.pattern
    # "call Sarah now" / "get Sarah quickly" reads as a summons. Without the
    # urgency word it is just a request to ring someone, which is everyday.
    for name in caregiver_names:
        if not name:
            continue
        if re.search(
            rf"\b(call|phone|get|ring|fetch)\s+{re.escape(name)}\b[^.?!]{{0,20}}"
            rf"\b(now|quick|quickly|urgent|urgently|right away|immediately|please come)\b",
            body, re.IGNORECASE,
        ):
            return f"urgent summons for {name}"
    return None


def classify(text: str, caregiver_names: Iterable[str] = ()) -> Tuple[str, Optional[str]]:
    """Sort one utterance into exactly one tier.

    Precedence is emergency, then distress, then everyday. Medical questions are
    handled separately by is_medical_question(); they are orthogonal to this and
    can occur at any tier.
    """
    names = list(caregiver_names)
    reason = is_emergency(text, names)
    if reason:
        return TIER_EMERGENCY, reason
    reason = is_distress(text, names)
    if reason:
        return TIER_DISTRESS, reason
    return TIER_EVERYDAY, None


def is_distress(text: str, caregiver_names: Iterable[str] = ()) -> Optional[str]:
    """Signs of confusion or fear. Logged and answered warmly — nothing more
    automated than that, by design."""
    body = text or ""
    for pattern in _DISTRESS_RE:
        if pattern.search(body):
            return pattern.pattern
    # "where is Sarah" reads very differently from "where is the remote".
    for name in caregiver_names:
        if not name:
            continue
        if re.search(rf"\bwhere(.s| is| are)?\s+{re.escape(name)}\b", body, re.IGNORECASE):
            return f"asking for {name}"
    return None


# --- fixed spoken responses ----------------------------------------------
# Written out rather than generated, so they never vary. Someone hearing the
# same reassuring sentence every time is the point, not a limitation.

MEDICAL_REDIRECT = (
    "That's a good question, but it's one for your doctor or your carer. "
    "I've made a note so someone can help you with it."
)


def emergency_acknowledgement(caregiver_names: List[str]) -> str:
    """Only ever said for a real emergency, so it can stay strong and calm."""
    if not caregiver_names:
        return "Okay. I've made a note that you need help now, and someone will see it."
    if len(caregiver_names) == 1:
        return f"Okay, I'm letting {caregiver_names[0]} know right now. Stay where you are, you're alright."
    return (f"Okay, I'm letting {caregiver_names[0]} and {caregiver_names[1]} know right now. "
            "Stay where you are, you're alright.")


def distress_reassurance(speaker_name: Optional[str], caregiver_names: List[str]) -> str:
    who = f" {speaker_name}" if speaker_name and speaker_name.lower() != "unknown" else ""
    if caregiver_names:
        return (
            f"You're safe{who}, and I'm right here with you. "
            f"{caregiver_names[0]} knows where you are. Let's take a moment together."
        )
    return f"You're safe{who}, and I'm right here with you. Let's take a moment together."
