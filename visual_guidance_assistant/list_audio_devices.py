"""Audio device diagnostic — the microphone counterpart to list_cameras.py.

    .venv\\Scripts\\python.exe visual_guidance_assistant\\list_audio_devices.py
    .venv\\Scripts\\python.exe visual_guidance_assistant\\list_audio_devices.py --test

Run this whenever voice input stops working, and ALWAYS after plugging in or
removing a USB device that carries a microphone (a webcam, a headset, a dock).
Windows renumbers audio devices when hardware changes, so an index that was
correct yesterday can point at something else entirely today — that is exactly
how voice_input.device_index came to be wrong on this machine.

Devices are ranked by whether they actually work with this app's capture path,
which is not the same as whether they look correct in a list:

  - WASAPI devices only offer 48 kHz and REJECT the app's 16 kHz capture
    outright with "Invalid sample rate".
  - Some DirectSound endpoints open happily and then return pure silence.
  - WDM-KS supports callback mode only, not the blocking reads used here.

So every candidate is opened for real, at the app's exact settings, before
anything is recommended.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import sounddevice as sd

# Must match MicrophoneListener
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK = int(0.25 * SAMPLE_RATE)

# Thresholds from config/settings.yaml
MIN_PEAK_AMPLITUDE = 0.008

VIRTUAL_HINTS = (
    "stereo mix", "wave out", "what u hear", "loopback", "virtual", "vb-audio",
    "voicemeeter", "cable output", "obs", "sound mapper", "primary sound",
)


def enumerate_inputs():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    rows = []
    for index, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        host = hostapis[dev["hostapi"]]["name"]
        low = dev["name"].lower()
        reasons = []
        if any(h in low for h in VIRTUAL_HINTS):
            reasons.append("virtual/loopback")
        if "WDM-KS" in host:
            reasons.append("WDM-KS: blocking reads unsupported")
        rows.append({
            "index": index, "name": dev["name"], "host": host,
            "channels": dev["max_input_channels"], "skip_reasons": reasons,
        })
    return rows


def probe(index, seconds=1.2):
    """Open at the app's exact settings and measure. Returns (ok, peak, error)."""
    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                dtype=DTYPE, device=index, blocksize=CHUNK)
        chunks = []
        with stream:
            for _ in range(max(1, int(seconds / 0.25))):
                chunk, _ = stream.read(CHUNK)
                chunks.append(chunk.copy())
        audio = np.concatenate(chunks, axis=0).astype(np.float32) / 32768.0
        return True, float(np.max(np.abs(audio))), None
    except Exception as exc:
        return False, 0.0, f"{type(exc).__name__}: {exc}"


def live_test(index, seconds=12.0):
    """Record while the person speaks, then transcribe exactly as the app does."""
    import speech_recognition as sr

    print(f"\n  Listening on device {index} for {seconds:.0f} seconds.")
    for n in (3, 2, 1):
        print(f"    starting in {n}...")
        time.sleep(1)
    print("\n    >>> SPEAK NOW <<<\n")

    chunks = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype=DTYPE, device=index, blocksize=CHUNK)
    with stream:
        t0 = time.time()
        while time.time() - t0 < seconds:
            chunk, _ = stream.read(CHUNK)
            chunks.append(chunk.copy())
            level = float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2)))
            print(f"\r    level |{'#' * min(40, int(level * 400)):<40}| rms={level:.4f}",
                  end="", flush=True)

    audio = np.concatenate(chunks, axis=0).astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"\n\n    peak = {peak:.5f}  (below {MIN_PEAK_AMPLITUDE} is discarded as TOO QUIET)")
    print(f"    rms  = {rms:.5f}")
    if peak < MIN_PEAK_AMPLITUDE:
        print("    -> TOO QUIET. Either nothing was said, or this mic is too "
              "insensitive at this distance.")

    gain = min(0.09 / rms, 20.0) if rms > 1e-6 else 1.0
    pcm = (np.clip(audio * gain, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    try:
        text = sr.Recognizer().recognize_google(sr.AudioData(pcm, SAMPLE_RATE, 2))
        print(f'    TRANSCRIBED: "{text}"')
        return text
    except sr.UnknownValueError:
        print("    TRANSCRIBED: (nothing recognisable)")
    except Exception as exc:
        print(f"    transcription failed: {type(exc).__name__}: {exc}")
    return None


def main() -> int:
    do_live = "--test" in sys.argv

    print("=" * 86)
    print("AUDIO INPUT DEVICES ON THIS MACHINE")
    print("=" * 86)
    rows = enumerate_inputs()
    if not rows:
        print("  No input devices at all. Voice input cannot work.")
        return 1

    print(f"{'idx':>3}  {'ch':>2}  {'host API':<22}  name")
    print("-" * 86)
    for row in rows:
        note = f"   [{'; '.join(row['skip_reasons'])}]" if row["skip_reasons"] else ""
        print(f"{row['index']:>3}  {row['channels']:>2}  {row['host']:<22}  {row['name']}{note}")

    candidates = [r for r in rows if not r["skip_reasons"]]
    print("\n" + "=" * 86)
    print("OPENING EACH PLAUSIBLE DEVICE AT THE APP'S REAL SETTINGS "
          f"({SAMPLE_RATE} Hz, mono, blocking)")
    print("=" * 86)

    working = []
    for row in candidates:
        ok, peak, err = probe(row["index"])
        if ok and peak > 0.0:
            print(f"  index {row['index']:>2}  WORKS    room-tone peak={peak:.5f}  {row['name']}")
            working.append((row, peak))
        elif ok:
            print(f"  index {row['index']:>2}  SILENT   opens but returns pure silence  {row['name']}")
        else:
            print(f"  index {row['index']:>2}  UNUSABLE {err.split(':')[0]}  {row['name']}")

    if not working:
        print("\n  No device both opens AND passes audio. Voice input cannot work.")
        return 1

    working.sort(key=lambda w: -w[1])
    best = working[0][0]
    print("\n" + "=" * 86)
    print(f"RECOMMENDED  ->  voice_input.device_index: {best['index']}")
    print(f"             {best['name']!r} ({best['host']})")
    print("=" * 86)
    print("Ranked by measured sensitivity. A mic whose level sits below")
    print(f"min_peak_amplitude ({MIN_PEAK_AMPLITUDE}) will have real speech discarded.")

    if do_live:
        print("\n" + "=" * 86)
        print("LIVE SPEECH TEST")
        print("=" * 86)
        live_test(best["index"])
    else:
        print("\nRe-run with --test to speak into the recommended device and see")
        print("the transcription.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
