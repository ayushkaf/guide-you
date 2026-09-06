"""Speech synthesizer using pyttsx3."""
from __future__ import annotations

import threading
import queue
import logging
import time

import pyttsx3

from utils import latency


class SpeechSynthesizer(threading.Thread):
    def __init__(
        self,
        speech_queue: queue.Queue,
        config: dict,
        mic_listener=None,
        sound_detector=None,
        azure_tts=None,
    ) -> None:
        super().__init__(name="SpeechSynthesizer", daemon=True)
        self.log = logging.getLogger("speech.synthesizer")
        self.speech_queue = speech_queue
        self.config = config
        self.running = False
        # Optional AzureTTSClient — an alternative playback BACKEND for
        # _speak_one below, not a different code path. Everything around it
        # (the pause/resume of mic_listener and sound_detector, and the
        # post-speech cooldown in run()) is unchanged and applies exactly
        # the same regardless of which backend actually spoke.
        self.azure_tts = azure_tts
        speech_cfg = config.get("speech", {}) or {}
        self.rate = speech_cfg.get("rate", 150)
        # How long to wait after speaking before the mic starts listening
        # again — covers audio-driver tail so the assistant doesn't hear
        # (and respond to) its own voice.
        self.post_speech_cooldown = float(speech_cfg.get("post_speech_cooldown", 0.6))
        self.mic_listener = mic_listener
        # The sound detector runs its own audio stream and needs the same
        # treatment: unpaused, it hears the TTS voice and misclassifies it as a
        # doorbell / phone / baby, which triggers another spoken response.
        self.sound_detector = sound_detector

    def set_mic_listener(self, mic_listener) -> None:
        self.mic_listener = mic_listener

    def set_sound_detector(self, sound_detector) -> None:
        self.sound_detector = sound_detector

    def _listeners(self):
        return [obj for obj in (self.mic_listener, self.sound_detector) if obj is not None]

    def stop(self) -> None:
        self.running = False

    def _speak_one(self, msg: str) -> None:
        """Speak one message. Tries Azure Neural TTS first if configured and
        healthy; falls back to the local pyttsx3 engine (unchanged below) on
        anything else — not configured, no key, network failure, or the
        audio it returned failing to play. Both paths block until the audio
        has finished, which is what the pause/resume/cooldown wrapper in
        run() depends on.
        """
        if self.azure_tts is not None:
            audio = self.azure_tts.synthesize(msg)
            if audio is not None:
                try:
                    import winsound
                    winsound.PlaySound(audio, winsound.SND_MEMORY)
                    return
                except Exception as exc:
                    print(f"DEBUG: Azure TTS audio failed to play, falling back "
                          f"to pyttsx3: {exc}")

        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.say(msg)
            engine.runAndWait()
        except Exception as e:
            print(f"DEBUG: Speech error: {e}")
        finally:
            if engine:
                try:
                    engine.stop()
                except:
                    pass

    def run(self) -> None:
        self.running = True
        print("DEBUG: SpeechSynthesizer started")

        while self.running:
            try:
                msg = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Audio is about to start: this is the end of the person's wait.
            latency.mark(latency.TTS_START)
            print(latency.report())
            speak_started = time.time()
            print(f"DEBUG: Speaking: {msg}")
            listeners = self._listeners()
            for listener in listeners:
                try:
                    listener.pause()
                except Exception as exc:
                    print(f"DEBUG: could not pause {type(listener).__name__}: {exc}")
            try:
                self._speak_one(msg)
            finally:
                if listeners:
                    # Let the audio tail decay before either audio consumer
                    # starts listening again, so the assistant doesn't hear
                    # (or classify) its own voice.
                    time.sleep(self.post_speech_cooldown)
                for listener in listeners:
                    try:
                        listener.resume()
                    except Exception as exc:
                        print(f"DEBUG: could not resume {type(listener).__name__}: {exc}")
            print(f"DEBUG: Done speaking (playback took {time.time() - speak_started:.2f}s, "
                  f"{len(msg)} chars)")