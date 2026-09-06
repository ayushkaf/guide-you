"""Sound detection thread for environmental event awareness.

Pause/resume mirrors MicrophoneListener's contract. SpeechSynthesizer pauses
both while it speaks: this detector's thresholds (rms > 0.05 in the 300-3000 Hz
band) sit squarely on top of a synthesized voice, so without pausing the
assistant classifies its own speech as a doorbell or phone ringing, answers the
"event", and re-triggers itself — a feedback loop that reads as the AI talking
to itself.
"""
import threading
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


class SoundDetector(threading.Thread):
    def __init__(self, sound_queue, config, sample_rate: int = 16000, block_duration: float = 1.0):
        super().__init__(daemon=True)
        self.sound_queue = sound_queue
        self.config = config
        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.running = False
        self.last_sound_time = time.time()
        self.last_event = None
        self._paused = threading.Event()

    def stop(self):
        self.running = False

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def _detect_sound_type(self, audio_data: np.ndarray) -> str:
        if audio_data.size == 0:
            return ""
        rms = np.sqrt(np.mean(audio_data ** 2))
        freq = np.abs(np.fft.rfft(audio_data))
        peak_freq = np.argmax(freq) * (self.sample_rate / len(audio_data)) if freq.size else 0
        if rms > 0.15 and peak_freq > 3000:
            return "glass_breaking"
        if rms > 0.08 and 500 < peak_freq <= 3000:
            return "doorbell"
        if rms > 0.05 and peak_freq <= 500:
            return "knock"
        if rms > 0.07 and 300 < peak_freq <= 1200:
            return "phone_ringing"
        if rms > 0.06 and 1200 < peak_freq <= 3000:
            return "baby_crying"
        if rms > 0.05 and 3000 < peak_freq <= 5000:
            return "applause"
        return ""

    def run(self):
        if sd is None:
            print("SoundDetector disabled: sounddevice not installed.")
            return

        self.running = True
        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * self.block_duration),
                dtype="float32",
            ) as stream:
                while self.running:
                    block, overflow = stream.read(int(self.sample_rate * self.block_duration))

                    # Keep draining the stream while paused so the driver buffer
                    # doesn't overflow — just don't classify what we hear, since
                    # during a pause the loudest thing in the room is us.
                    if self._paused.is_set():
                        self.last_sound_time = time.time()
                        continue

                    if overflow:
                        continue
                    audio = block.flatten()
                    event = self._detect_sound_type(audio)
                    if event and event != self.last_event:
                        self.last_event = event
                        self.last_sound_time = time.time()
                        try:
                            self.sound_queue.put_nowait(event)
                        except Exception:
                            pass
                    if time.time() - self.last_sound_time > 4 * 3600:
                        try:
                            self.sound_queue.put_nowait("long_silence")
                        except Exception:
                            pass
                        self.last_sound_time = time.time()
        except Exception as e:
            print(f"SoundDetector error: {e}")
