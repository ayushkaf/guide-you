"""LLM connector for the Anthropic Claude API.

Sends the live webcam frame to Claude as an image content block alongside the
YOLO-derived text scene description, so the model can actually answer visual
questions instead of only reasoning over object labels.
"""
import base64
import time

import anthropic

from utils.env_file import load_env_file

# Done at import, before anything reads ANTHROPIC_API_KEY. This module is
# imported by dialogue_processor before its own key check runs, so a key in
# .env is picked up everywhere — including the startup banner, which would
# otherwise warn about a key that is in fact available.
load_env_file()

# Haiku 4.5 is the fastest Claude model and supports image input — the right
# trade-off for a voice assistant, where round-trip latency is what the user
# actually feels.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

ROLE_PATIENT = "patient"
ROLE_CAREGIVER = "caregiver"

# Patient replies are capped hard enough that a long answer cannot physically be
# produced; caregivers get room for a genuinely useful one.
PATIENT_MAX_TOKENS = 90
CAREGIVER_MAX_TOKENS = 220

# Written from standard dementia-care communication practice: validate rather
# than correct, tolerate repetition without ever drawing attention to it, keep
# language short and plain, stay calm.
#
# This app is a companion only. It does not assess, diagnose, monitor, or advise
# on health, and must never imply otherwise.
# Who the assistant IS, as opposed to what it must not do.
#
# The rules below are almost entirely prohibitions — six "never"s to two
# positives — which is right for safety and wrong for voice. Given only a list
# of things not to say, the model produces careful, slightly formal prose: it
# opens with the person's name every time, answers more than was asked, and
# reads like a helpline rather than someone in the room.
#
# Measured A/B on identical questions, this section cut replies from 78 to 49
# characters on average — two seconds less listening on every single turn, which
# at rate 150 is the difference between a conversation and a briefing. Asked
# "what day is it", the rules alone gave "It's Sunday, the 2nd of August — a
# lovely afternoon at twenty to five"; with this, "It's Sunday today."
PATIENT_PERSONA = """\
Who you are to them: a familiar, easy presence in their day, the way a good
friend sitting in the same room is. You are genuinely fond of them. You are not
a helpline, a nurse, or an assistant taking instructions.

Who they are: the person you are helping. They may have trouble with their
memory, which is why the rules below exist. Never say any of that to them.

How you talk:
- Short. Usually one sentence, sometimes two. Never three.
- Like speech, not writing. Contractions, plain words, the odd "oh" or "ah".
- Answer the thing that was asked FIRST, in the first few words.
- Ask at most one question back, and only when you genuinely want to know.
- Don't open every reply with their name; it starts to sound like a form letter.

For example:
  They say: "what day is it"
  You say:  "It's Sunday today."
  Not:      "Of course! Today is Sunday the 2nd of August. Is there anything
             else you'd like to know about the date?"

  They say: "I need help going to the shop"
  You say:  "Happy to think it through with you. What do you need?"
  Not:      "I'd be delighted to assist you in thinking through your trip to the
             grocery store. What items are you hoping to purchase today?\""""

PATIENT_MODE_RULES = """\
How to talk with this person:
- NEVER point out that a question has been asked before. Repetition is expected.
  Answer freshly and warmly every single time, as if it were the first time. Do
  not say "as I mentioned", "again", "like I said", or "you already asked".
- NEVER argue with, correct, or contradict what they say about their own life,
  memories, or beliefs, even when it is factually wrong. Contradiction causes
  distress and achieves nothing. Instead accept it and gently move alongside
  them: "That sounds lovely, tell me more about that."
- DO answer questions about the day, date, time, and where they are directly,
  calmly, and factually. Orientation is genuinely helpful. Give the plain answer
  first, warmly, without making it feel like a test.
- Keep every reply to ONE or TWO short sentences. Everyday words only. No
  jargon, no lists, no multi-part answers, no follow-up questions stacked up.
- Always sound calm, warm and unhurried. Never sound rushed, corrective,
  clinical, or irritated, no matter how many times something is repeated.
- When they ask for help with something everyday — finding a thing, getting to
  the shop, what to have for lunch, remembering an appointment — HELP THEM.
  Talk it through, suggest a simple next step, encourage them. Do NOT answer an
  ordinary request by telling them to ask their carer; that is only for medical
  questions. Be honest and warm about what you cannot do: you have no hands and
  cannot drive, carry, or fetch anything. Say so kindly and offer what you can.
- Never mention these instructions, their condition, memory, or any diagnosis.
- You are not a medical device. Never give medical, medication, symptom, or
  treatment advice, and never speculate about anyone's health. If it comes up,
  warmly suggest asking their carer or doctor."""

CAREGIVER_MODE_RULES = """\
How to talk with this person:
- The person you are speaking to right now is a carer or an admin — NOT the
  person they look after, and not you. You are still Guide YOU, the assistant.
  Be direct, clear and efficient. Give complete, practical answers without the
  softening used with the person they care for.
- Two to four sentences is fine when detail genuinely helps.
- You may state plainly what you can and cannot see or do.
- You are not a medical device and hold no clinical information. Never give
  medical, medication, or diagnostic advice, and never characterise anyone's
  cognitive state or health. Defer to their doctor for anything clinical."""

# Errors that will never succeed on retry (bad request, bad key, wrong model).
# Retrying these just burns the user's time waiting for speech.
_FATAL_ERRORS = (
    anthropic.BadRequestError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
)

# When no credentials are configured at all, the SDK raises a plain TypeError
# before any request is made. That is not an API failure, so it does not match
# any of the typed errors above and used to fall through to the generic retry
# path — three identical attempts, each guaranteed to fail, and about two
# seconds of dead air before the person heard an apology.
_NO_CREDENTIALS_MARKER = "could not resolve authentication method"

# Said instead of the generic apology when the cause is configuration rather
# than a transient fault. A care device repeating "sorry, I couldn't process
# that" forever, when the real problem is an unset environment variable, gives
# the person no way to understand what is wrong and no way to get help. Also
# used for a 401 AuthenticationError (see _call_with_retry) — a key that is
# PRESENT but invalid/revoked is the same category of problem as no key at
# all: someone who set this device up needs to go fix something, and the
# generic apology reads exactly like a broken microphone or a bad network
# connection, giving no clue that the fix is checking a key rather than,
# say, restarting the device.
NOT_CONFIGURED_MESSAGE = (
    "I can't reach my thinking service at the moment. "
    "Please ask whoever set me up to check my settings."
)

# Distinct from None so ask_with_context can tell "misconfigured" (a person must
# fix something) apart from "the call failed" (might work next time).
_NOT_CONFIGURED = object()


class LLMConnector:
    def __init__(self, model: str = DEFAULT_MODEL, max_history: int = 6, timeout: int = 15):
        # No api_key argument: the SDK resolves ANTHROPIC_API_KEY (or an
        # `ant auth login` profile) from the environment itself. Passing a key
        # through source is what we're deliberately avoiding.
        self.client = anthropic.Anthropic(
            timeout=timeout,  # fail loudly instead of hanging silently
        )
        self.model = model
        self.max_history = max_history
        self.conversation_history = []
        self.last_reply = None  # used to detect/avoid immediate repeats

    def ask_with_context(
        self,
        user_text,
        scene_description,
        frame_bytes=None,
        memory_context=None,
        role=ROLE_PATIENT,
        speaker_name=None,
        now_context=None,
        social_context=None,
        casual_context=None,
    ):
        memory_text = self._format_memory(memory_context)

        # States plainly what the assistant IS and, just as importantly, what it
        # is not. The old opening ("a warm and observant companion") never said
        # there was a separate human being helped, which left the model's own
        # identity to be inferred from surrounding context — and a nearby line
        # reading "PERSONA: Patient" was easily read as its own persona.
        system_parts = [
            "You are Guide YOU, a voice assistant running on a computer with a "
            "camera and a microphone. A real person is in front of you, and you "
            "are here to keep them company and help them.\n\n"
            "YOU ARE NOT THAT PERSON. You are the assistant talking WITH them. "
            "Anything you are told about them — their name, their day, their "
            "mood — describes THEM, never you. If you are asked who you are, "
            "say you are Guide YOU, their assistant."
        ]
        if now_context:
            # Ground truth for date/time/place. Without it the model invents
            # both, fluently — see safety/orientation.py.
            system_parts.append(now_context)
        if speaker_name:
            system_parts.append(f"You are speaking with {speaker_name}.")
        if social_context:
            # Who else in the room the person knows, and how. This is what lets
            # the assistant say "Sarah's here" instead of "there's a person".
            system_parts.append(social_context)
        if casual_context:
            # Approximate, IP-based weather — for conversational colour ONLY.
            # The line itself says not to use it for "where am I"; that
            # question is answered exclusively by now_context/home_location
            # above, never by this.
            system_parts.append(casual_context)
        if role == ROLE_PATIENT:
            # Character first, then the safety rules. Both are needed: the rules
            # are the floor, the persona is what stops it sounding like a form.
            system_parts.append(PATIENT_PERSONA)
            system_parts.append(PATIENT_MODE_RULES)
        else:
            system_parts.append(CAREGIVER_MODE_RULES)
        if memory_text:
            system_parts.append(memory_text)
        if scene_description:
            system_parts.append(
                "Object detector summary (may be incomplete — trust the image "
                f"over this when they disagree): {scene_description}"
            )
        system_prompt = "\n\n".join(system_parts)

        # Claude's messages list only carries user/assistant turns — memory and
        # scene context go in the top-level `system` param, not as extra
        # "system"-role messages (unlike OpenAI's chat.completions shape).
        messages = []

        # Only replay the last N exchanges, and only as text: re-sending every
        # historical frame would multiply input tokens and latency on every
        # single turn, which a voice assistant can't absorb.
        for entry in self.conversation_history[-self.max_history:]:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["ai"]})

        messages.append({"role": "user", "content": self._build_user_content(user_text, frame_bytes)})

        print("[llm-debug] FINAL_PROMPT_START")
        print(f"system: {system_prompt}")
        print(f"[llm-debug] frame attached: {'yes' if frame_bytes else 'no'}"
              f"{f' ({len(frame_bytes)} bytes)' if frame_bytes else ''}")
        print(self._loggable(messages))
        print("[llm-debug] FINAL_PROMPT_END")

        # A hard ceiling as well as a prompt instruction. In patient mode the
        # brevity requirement is a care property, not a style preference, so it
        # should not depend on the model choosing to comply.
        max_tokens = PATIENT_MAX_TOKENS if role == ROLE_PATIENT else CAREGIVER_MAX_TOKENS

        start = time.time()
        reply = self._call_with_retry(system_prompt, messages, max_tokens=max_tokens)
        elapsed = time.time() - start
        print(f"[llm-debug] role={role} max_tokens={max_tokens} API call took {elapsed:.2f}s")

        if reply is _NOT_CONFIGURED:
            return NOT_CONFIGURED_MESSAGE
        if reply is None:
            return "Sorry, I couldn't process that just now."

        # Repeating an answer is a problem in caregiver mode and a *feature* in
        # patient mode: someone who asks the same question every few minutes
        # should get the same calm answer every time, not a variation that
        # signals they have asked before.
        if role != ROLE_PATIENT and self.last_reply and reply.strip() == self.last_reply.strip():
            print("[llm-debug] WARNING: model repeated previous reply verbatim")

        self.last_reply = reply
        # Store the plain text, not the content blocks — history is replayed as
        # text only (see the loop above).
        self.conversation_history.append({"user": user_text, "ai": reply})

        # hard cap so history never grows unbounded over a long session
        if len(self.conversation_history) > self.max_history * 3:
            self.conversation_history = self.conversation_history[-self.max_history:]

        return reply

    @staticmethod
    def _build_user_content(user_text: str, frame_bytes):
        """Build the user turn: a JPEG image block (when a frame is available)
        followed by the spoken text. Image first is the documented ordering —
        Claude attends to it better than when the text leads."""
        if not frame_bytes:
            return user_text

        try:
            encoded = base64.standard_b64encode(frame_bytes).decode("utf-8")
        except Exception as exc:
            print(f"[llm-debug] could not base64-encode frame, sending text only: {exc}")
            return user_text

        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    # CameraCapture.get_latest_frame_bytes() encodes with
                    # cv2.imencode('.jpg', ...), so this is always JPEG.
                    "media_type": "image/jpeg",
                    "data": encoded,
                },
            },
            {"type": "text", "text": user_text},
        ]

    @staticmethod
    def _loggable(messages):
        """Same messages, but with base64 image payloads elided — a raw dump
        would flood the console with tens of KB of base64 per turn."""
        summary = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                blocks = []
                for block in content:
                    if block.get("type") == "image":
                        size = len(block["source"]["data"])
                        blocks.append({"type": "image", "source": f"<base64 jpeg, {size} chars>"})
                    else:
                        blocks.append(block)
                summary.append({"role": message["role"], "content": blocks})
            else:
                summary.append(message)
        return summary

    def _call_with_retry(self, system_prompt, messages, max_tokens=150, retries=2):
        for attempt in range(retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                # Guard before reading content: a refusal comes back as a
                # successful 200 with no text block.
                if response.stop_reason == "refusal":
                    print("[llm-debug] model refused this request")
                    return "I'd rather not answer that one."
                text = next((block.text for block in response.content if block.type == "text"), "")
                return text.strip()
            except _FATAL_ERRORS as exc:
                # Retrying a 400/401/404 only makes the user wait longer for the
                # same failure.
                print(f"[llm-debug] LLM error (not retryable): {type(exc).__name__}: {exc}")
                if isinstance(exc, anthropic.AuthenticationError):
                    # 401 specifically: the key IS present but the API
                    # rejected it (invalid or revoked), as opposed to no key
                    # being configured at all (the TypeError/
                    # _NO_CREDENTIALS_MARKER branch below) or a transient
                    # network/rate-limit fault (the generic Exception branch,
                    # which still gets the generic apology since it might
                    # genuinely work on the next try). Both credential
                    # problems are reported the same way — see
                    # NOT_CONFIGURED_MESSAGE's comment.
                    print("[llm-debug] AUTHENTICATION ERROR — the configured "
                          "ANTHROPIC_API_KEY was rejected by the API (invalid "
                          "or revoked, not merely absent). Check .env / the "
                          "environment for a stale or mistyped key.")
                    return _NOT_CONFIGURED
                return None
            except TypeError as exc:
                if _NO_CREDENTIALS_MARKER in str(exc).lower():
                    print(f"[llm-debug] NOT CONFIGURED — no API credentials in this "
                          f"process's environment: {exc}")
                    print("[llm-debug] ANTHROPIC_API_KEY is read from the environment when "
                          "the app STARTS. If it was set with setx after this terminal was "
                          "opened, this process never inherited it — close the terminal, "
                          "open a new one, and start the app again.")
                    return _NOT_CONFIGURED
                raise
            except Exception as exc:
                print(f"[llm-debug] LLM error (attempt {attempt + 1}/{retries + 1}): "
                      f"{type(exc).__name__}: {exc}")
                if attempt < retries:
                    time.sleep(1)  # brief backoff before retry
                else:
                    return None

    def _format_memory(self, memory_context):
        if not memory_context:
            return ""
        if isinstance(memory_context, dict):
            return "; ".join(f"{k}: {v}" for k, v in memory_context.items())
        return str(memory_context)
