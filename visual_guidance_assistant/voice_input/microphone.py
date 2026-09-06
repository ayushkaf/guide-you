"""Microphone listener using sounddevice + Google speech recognition."""
import os
import threading
import queue
import time
import traceback
import wave
import numpy as np
import sounddevice as sd
import speech_recognition as sr

from utils import latency

# Device names that mean "virtual/routing endpoint", not a physical
# microphone. This is the exact, documented signature of a real incident: the
# built-in mic array silently slid to a different index, and the fallback
# picked 'Microsoft Sound Mapper - Input' — which has real input channels and
# a non-WDM-KS host API, so it passed every usability check below, while
# capturing nothing useful. Every recording that session was genuine,
# clearly-spoken audio fed to the wrong device; STT failed because of THAT,
# not because speech recognition itself was broken. See settings.yaml's
# voice_input.device_index comment for the full account.
_SUSPECT_VIRTUAL_DEVICE_MARKERS = ("mapper", "virtual", "stereo mix", "what u hear", "loopback")

# How many recognize_google() misses in a row before the self-check warns.
# Occasional misses are normal (background noise, an unclear word); a run
# this long, with nothing successfully recognized in between, is the pattern
# that took hours to diagnose last time because it looked identical to "the
# STT engine is failing" right up until the device itself was checked.
UNKNOWN_VALUE_ERROR_WARNING_THRESHOLD = 3


class MicrophoneListener(threading.Thread):
    def __init__(self, command_queue, config):
        super().__init__(daemon=True)
        self.command_queue = command_queue
        self.config = config
        self.running = False
        self.sample_rate = 16000

        voice_cfg = config.get("voice_input") or {}
        configured_index = voice_cfg.get("device_index", 1)
        self.device_name = None
        try:
            self.device_index = self._resolve_device_index(configured_index)
        except Exception as exc:
            print(f"[audio-debug] WARNING: device enumeration failed ({exc}); using configured device_index as-is")
            self.device_index = int(configured_index) if configured_index is not None else None
        # Consecutive recognize_google() misses with nothing successful in
        # between — see UNKNOWN_VALUE_ERROR_WARNING_THRESHOLD.
        self._consecutive_unknown_value_errors = 0
        self.min_peak_amplitude = float(voice_cfg.get("min_peak_amplitude", 0.02))
        self.debug_save_wav = bool(voice_cfg.get("debug_save_wav", True))

        self.chunk_duration = 0.25
        self.max_record_seconds = float(voice_cfg.get("max_record_seconds", 8))
        self.silence_stop_seconds = float(voice_cfg.get("silence_stop_seconds", 1.0))
        self.speech_start_rms = float(voice_cfg.get("speech_start_rms", 0.015))
        self.silence_rms = float(voice_cfg.get("silence_rms", 0.008))

        self.recognizer = sr.Recognizer()
        self.target_rms = float(voice_cfg.get("target_rms", 0.09))
        self.max_gain = float(voice_cfg.get("max_gain", 20.0))

        # Pause control — set when something else (e.g. console input()) needs
        # exclusive attention and the mic should stop listening temporarily
        self._paused = threading.Event()

        # The audio behind the most recent recognised command, so the dialogue
        # thread can work out WHO said it when several people are in view.
        self.last_utterance_audio = None

    @staticmethod
    def _usable_input_devices():
        """Input-capable devices whose host API supports the blocking read
        API this listener uses (WDM-KS only supports callback mode and
        raises 'Blocking API not supported yet')."""
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        usable = []
        for index, dev in enumerate(devices):
            if dev["max_input_channels"] <= 0:
                continue
            hostapi_name = hostapis[dev["hostapi"]]["name"]
            if "WDM-KS" in hostapi_name:
                continue
            usable.append((index, dev["name"], hostapi_name))
        return usable

    def _resolve_device_index(self, configured_index):
        usable = self._usable_input_devices()
        usable_indices = {i for i, _, _ in usable}
        names_by_index = {index: name for index, name, _ in usable}

        print("[audio-debug] usable input devices:")
        for index, name, hostapi_name in usable:
            print(f"[audio-debug]   {index}: {name!r} ({hostapi_name})")

        if configured_index is not None and int(configured_index) in usable_indices:
            chosen = int(configured_index)
        elif not usable:
            print("[audio-debug] WARNING: no usable input device found on this system; voice input will be disabled")
            return None
        else:
            chosen = usable[0][0]
            print(
                f"[audio-debug] WARNING: configured device_index={configured_index} is not a usable "
                f"input device; falling back to {chosen} ({usable[0][1]!r})"
            )

        self.device_name = names_by_index.get(chosen)
        self._startup_self_check(chosen, self.device_name)
        return chosen

    def _startup_self_check(self, index, name):
        """Warn BEFORE a single word is spoken if the chosen device looks
        virtual — see _SUSPECT_VIRTUAL_DEVICE_MARKERS. A device can pass
        every check above (real input channels, non-WDM-KS host API) and
        still be the wrong one; this is the one thing those checks cannot
        catch, because they were never able to catch it before either."""
        lowered = (name or "").lower()
        matched = next((m for m in _SUSPECT_VIRTUAL_DEVICE_MARKERS if m in lowered), None)
        if not matched:
            print(f"[audio-debug] self-check: device {index} ({name!r}) does not "
                  "match any known virtual/routing device pattern")
            return
        print("\n" + "!" * 78)
        print(f"!!  MICROPHONE SELF-CHECK: device {index} ({name!r}) LOOKS VIRTUAL")
        print("!!")
        print(f"!!  Its name contains {matched!r}, which matches the exact signature of")
        print("!!  a past incident: the real microphone silently moved to a different")
        print("!!  index, and a virtual/routing device — which still has input channels")
        print("!!  and passes every usability check above — was picked up instead.")
        print("!!  Every recording that session was genuine, clearly-spoken audio fed")
        print("!!  to the wrong place; nothing was wrong with speech recognition itself.")
        print("!!")
        print("!!  Run  .venv\\Scripts\\python.exe visual_guidance_assistant\\list_audio_devices.py")
        print("!!  and check voice_input.device_index in config/settings.yaml points at")
        print("!!  a real physical microphone before assuming anything else is broken.")
        print("!" * 78 + "\n")

    def stop(self):
        self.running = False

    def pause(self):
        self._paused.set()
        print("[audio-debug] microphone paused")

    def resume(self):
        self._paused.clear()
        print("[audio-debug] microphone resumed")

    def is_paused(self) -> bool:
        """True while TTS is playing. Voice enrolment waits on this so it does
        not record the assistant's own prompt instead of the person."""
        return self._paused.is_set()

    def record_phrase(self, seconds: float = 4.0):
        """Record a fixed window of raw audio, for voice enrolment.

        Only safe to call while the listener is paused — the recognition loop
        opens its own stream, and two streams on one device fail. The caller
        (FaceTrainer, via DialogueProcessor) pauses first.

        Returns float32 samples in [-1, 1] at self.sample_rate, or None.
        """
        if self.device_index is None:
            print("[audio-debug] record_phrase: no usable input device")
            return None

        chunk_samples = int(self.chunk_duration * self.sample_rate)
        chunks = []
        elapsed = 0.0
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                device=self.device_index,
                blocksize=chunk_samples,
            )
            with stream:
                while elapsed < seconds:
                    chunk, _ = stream.read(chunk_samples)
                    chunks.append(chunk.copy())
                    elapsed += self.chunk_duration
        except Exception as exc:
            print(f"[audio-debug] record_phrase failed: {type(exc).__name__}: {exc}")
            return None

        if not chunks:
            return None
        audio = np.concatenate(chunks, axis=0).astype(np.float32) / 32768.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        print(f"[audio-debug] record_phrase captured {elapsed:.1f}s, peak={peak:.4f}")
        return audio

    def _save_debug_wav(self, audio_bytes):
        if not self.debug_save_wav:
            return None
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        output_path = os.path.join(debug_dir, "last_capture.wav")
        if os.path.exists(output_path):
            os.remove(output_path)
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_bytes)
        return output_path

    def _recognize_google_with_timeout(self, audio, timeout=10):
        result_holder = {}
        error_holder = {}

        def run_recognition():
            try:
                result_holder["result"] = self.recognizer.recognize_google(audio)
            except Exception as exc:
                error_holder["error"] = exc

        worker = threading.Thread(target=run_recognition, daemon=True)
        worker.start()
        worker.join(timeout)

        if worker.is_alive():
            raise TimeoutError(f"recognize_google timed out after {timeout} seconds")
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("result")

    def _normalize_audio(self, audio_float):
        rms = float(np.sqrt(np.mean(np.square(audio_float)))) if audio_float.size else 0.0
        if rms <= 1e-6:
            return audio_float, rms, rms
        gain = min(self.target_rms / rms, self.max_gain)
        normalized = np.clip(audio_float * gain, -1.0, 1.0)
        rms_after = float(np.sqrt(np.mean(np.square(normalized))))
        return normalized, rms, rms_after

    def _warn_repeated_stt_failures(self):
        """Fires every UNKNOWN_VALUE_ERROR_WARNING_THRESHOLD misses in a row
        (3, 6, 9, ...) with nothing successfully recognised in between —
        surfaces the exact failure mode that took hours to diagnose last
        time, instead of it looking like silent, unexplained STT failure."""
        print("\n" + "!" * 78)
        print(f"!!  {self._consecutive_unknown_value_errors} SPEECH RECOGNITION FAILURES IN A ROW")
        print("!!")
        print(f"!!  Currently listening on device {self.device_index} ({self.device_name!r}).")
        print("!!")
        print("!!  This can mean the microphone itself is misconfigured, not that speech")
        print("!!  recognition is broken — a past incident produced this EXACT pattern:")
        print("!!  audio captured with a normal-looking volume, transcription failing")
        print("!!  every time, because the array mic had silently moved to a different")
        print("!!  device index and a virtual routing device was picked up instead.")
        print("!!")
        print("!!  Before assuming anything else is wrong, run:")
        print("!!    .venv\\Scripts\\python.exe visual_guidance_assistant\\list_audio_devices.py")
        print("!!  and check voice_input.device_index in config/settings.yaml still points")
        print("!!  at a real microphone.")
        print("!" * 78 + "\n")

    def _record_with_auto_stop(self):
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        chunks = []
        speech_detected = False
        silence_run = 0.0
        elapsed = 0.0

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            device=self.device_index,
            blocksize=chunk_samples,
        )
        with stream:
            while elapsed < self.max_record_seconds:
                if self._paused.is_set():
                    break  # abort recording immediately if paused mid-capture

                chunk, _ = stream.read(chunk_samples)
                chunks.append(chunk.copy())
                elapsed += self.chunk_duration

                chunk_float = chunk.astype(np.float32) / 32768.0
                chunk_rms = float(np.sqrt(np.mean(np.square(chunk_float))))

                if not speech_detected:
                    if chunk_rms >= self.speech_start_rms:
                        speech_detected = True
                        silence_run = 0.0
                else:
                    if chunk_rms < self.silence_rms:
                        silence_run += self.chunk_duration
                        if silence_run >= self.silence_stop_seconds:
                            break
                    else:
                        silence_run = 0.0

        audio_data = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype=np.int16)
        return audio_data, elapsed, speech_detected

    def run(self):
        self.running = True
        print("=" * 50)
        print(f"AUDIO MODE - Using microphone device {self.device_index}")
        print("=" * 50)

        if self.device_index is None:
            print("[audio-debug] No usable input device — voice input is disabled for this session.")

        while self.running:
            try:
                if self.device_index is None:
                    time.sleep(5.0)
                    continue

                if self._paused.is_set():
                    time.sleep(0.2)
                    continue

                print("Listening... (speak now)")
                cycle_start = time.time()

                try:
                    audio_data, recorded_seconds, speech_detected = self._record_with_auto_stop()
                except Exception as exc:
                    print(f"[audio-debug] recording raised: {exc}")
                    time.sleep(0.5)
                    continue

                if self._paused.is_set():
                    continue  # discard whatever was captured right before a pause

                if not speech_detected:
                    continue

                # The person has stopped talking. Everything after this point is
                # wait time from their point of view.
                latency.mark(latency.SPEECH_END)
                print(f"[audio-debug] captured {recorded_seconds:.2f}s of audio")

                audio_float = audio_data.astype(np.float32) / 32768.0
                peak = float(np.max(np.abs(audio_float))) if audio_float.size else 0.0

                if peak < self.min_peak_amplitude:
                    print(f"Volume: {peak:.4f} - TOO QUIET")
                    continue

                print(f"Volume: {peak:.4f} - OK")
                normalized, rms_before, rms_after = self._normalize_audio(audio_float)
                print(f"[audio-debug] peak={peak:.4f} rms_before={rms_before:.4f} rms_after={rms_after:.4f}")

                normalized_int16 = (normalized * 32767.0).astype(np.int16)
                audio_bytes = normalized_int16.tobytes()
                self._save_debug_wav(audio_bytes)
                audio = sr.AudioData(audio_bytes, self.sample_rate, 2)

                recog_start = time.time()
                try:
                    text = self._recognize_google_with_timeout(audio)
                    latency.mark(latency.TRANSCRIBED)
                    # A successful transcription — whatever streak of misses
                    # came before it is over; this device is working now.
                    self._consecutive_unknown_value_errors = 0
                    # Keep the audio that produced this command so the dialogue
                    # thread can ask who spoke it. Only useful when more than
                    # one recognised person is in view — see resolve_speaker.
                    self.last_utterance_audio = audio_float
                    recog_elapsed = time.time() - recog_start
                    total_elapsed = time.time() - cycle_start
                    print(f"[audio-debug] recognized in {recog_elapsed:.2f}s (total: {total_elapsed:.2f}s)")
                    print(f"You said: {text}")
                    try:
                        self.command_queue.put_nowait(text)
                    except queue.Full:
                        pass
                except sr.UnknownValueError:
                    self._consecutive_unknown_value_errors += 1
                    print(f"[audio-debug] UnknownValueError after {time.time() - recog_start:.2f}s "
                          f"(streak: {self._consecutive_unknown_value_errors})")
                    if self._consecutive_unknown_value_errors % UNKNOWN_VALUE_ERROR_WARNING_THRESHOLD == 0:
                        self._warn_repeated_stt_failures()
                except sr.RequestError as exc:
                    print(f"[audio-debug] RequestError: {exc}")
                    time.sleep(2)
                except TimeoutError as exc:
                    print(f"[audio-debug] Timeout: {exc}")
                except Exception as exc:
                    print(f"[audio-debug] Unexpected error: {type(exc).__name__}: {exc}")
                    traceback.print_exc()

            except Exception as e:
                print(f"[audio-debug] Loop error: {e}")
                time.sleep(0.5)