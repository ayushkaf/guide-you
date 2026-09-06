"""Azure Neural TTS backend — an optional, best-effort ALTERNATIVE to
pyttsx3, never a replacement for it. synthesize() never raises; it returns
audio bytes on success or None on anything else, and SpeechSynthesizer
falls straight through to the existing pyttsx3 code on None.

Plain REST calls rather than the azure-cognitiveservices-speech SDK: that
SDK carries a large native (C++) dependency for a single, simple HTTP call,
and every part of this integration's error handling is already written
around "catch anything, fall back" — a REST call gives that same behaviour
with a dependency this project already has (requests) rather than a new,
heavier one.
"""
from __future__ import annotations

import time
from typing import Optional
from xml.sax.saxutils import escape as _xml_escape

import requests

from utils.service_health import ServiceHealth

TOKEN_URL_FMT = "https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
SYNTHESIZE_URL_FMT = "https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

# Issued tokens are valid for 10 minutes; refreshed a little early so a
# request never races an expiry.
TOKEN_LIFETIME_SECONDS = 9 * 60

# RIFF/WAV PCM — winsound.PlaySound(..., SND_MEMORY) plays this directly
# with no decoding step and no extra dependency (winsound ships with
# CPython on Windows, and this app is Windows-only already — DPAPI,
# DirectShow, pywin32 for TTS's own pyttsx3 backend).
OUTPUT_FORMAT = "riff-24khz-16bit-mono-pcm"


class AzureTTSClient:
    def __init__(self, key: Optional[str], region: Optional[str],
                 voice: str = "en-US-JennyNeural", timeout: float = 10.0):
        self.key = key
        self.region = region
        self.voice = voice
        self.timeout = timeout
        configured = bool(key) and bool(region)
        self.health = ServiceHealth("azure-tts", configured=configured, cooldown_seconds=30.0)
        self._token: Optional[str] = None
        self._token_expires_at = 0.0

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires_at:
            return self._token
        response = requests.post(
            TOKEN_URL_FMT.format(region=self.region),
            headers={"Ocp-Apim-Subscription-Key": self.key},
            timeout=self.timeout,
        )
        response.raise_for_status()
        self._token = response.text
        self._token_expires_at = time.time() + TOKEN_LIFETIME_SECONDS
        return self._token

    def ensure_ready(self) -> bool:
        """One-time startup check: a real (tiny) synthesis call, so the
        startup summary line reflects the voice/region/key actually being
        valid together, not just that a token could be fetched. Safe to
        call even when not configured — just returns False."""
        if not self.health.configured:
            return False
        audio = self.synthesize("Ready.")
        if audio:
            print(f"[azure-tts] verified — voice={self.voice!r} region={self.region!r}")
            return True
        return False

    def synthesize(self, text: str) -> Optional[bytes]:
        """RIFF/WAV PCM bytes for `text` in self.voice, or None on any
        failure — not configured, unhealthy (cooling down from a recent
        failure), or the call itself failed for any reason."""
        if not self.health.should_attempt():
            return None
        try:
            token = self._get_token()
            ssml = (
                "<speak version='1.0' xml:lang='en-US'>"
                f"<voice xml:lang='en-US' name='{_xml_escape(self.voice)}'>"
                f"{_xml_escape(text)}</voice></speak>"
            )
            response = requests.post(
                SYNTHESIZE_URL_FMT.format(region=self.region),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": OUTPUT_FORMAT,
                    "User-Agent": "GuideYOU",
                },
                data=ssml.encode("utf-8"),
                timeout=self.timeout,
            )
            response.raise_for_status()
            self.health.record_success()
            return response.content
        except Exception as exc:
            self.health.record_failure(exc)
            return None
