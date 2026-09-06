"""Calibrate the voice-match threshold against real voices.

    .venv\\Scripts\\python.exe visual_guidance_assistant\\calibrate_voice.py

WHY THIS EXISTS
voice_id.voice_embedding.MATCH_THRESHOLD has never been set from real data. It was picked
as a placeholder and only ever exercised against synthesised audio, where two
"different speakers" are far more different than two real people are. A
threshold guessed that way is worthless: too strict and it never recognises
anyone, too loose and it confuses two members of the same family.

It cannot be calibrated without two real voices, so this asks for them.

WHAT IT DOES
Records several phrases from person A, then several from person B, and reports:
  - same-person scores  (A vs A) -- these must sit ABOVE the threshold
  - different-person    (A vs B) -- these must sit BELOW it
A usable threshold is one with clear air between those two groups. If they
overlap, voice matching is not reliable enough to use for identification at all,
and the honest answer is to leave face recognition as the only identifier.

Voice is only ever a supplementary hint in this app. It can add a name when the
camera cannot see clearly; it can NEVER on its own grant caregiver mode. So a
poor result here costs a convenience, not a safety property.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from config.config_loader import load_config
from utils.paths import CONFIG_FILE
from voice_id import voice_embedding
from voice_input.microphone import MicrophoneListener

BAR = "=" * 78
PHRASES = [
    "the kettle is on the kitchen table",
    "I think it might rain again this afternoon",
    "my favourite colour has always been blue",
    "we could go for a walk down by the river",
    "there is a photograph on the shelf by the window",
]
SECONDS = 4.0


def record_set(listener, who: str):
    embeddings = []
    print("\n" + BAR)
    print(f"  RECORDING: {who}")
    print(BAR)
    for i, phrase in enumerate(PHRASES, 1):
        print(f"\n  [{i}/{len(PHRASES)}] {who}, please say:")
        print(f'      "{phrase}"')
        for n in (3, 2, 1):
            print(f"      starting in {n}...")
            time.sleep(1)
        print(f"      >>> SPEAK NOW ({SECONDS:.0f} seconds) <<<")
        audio = listener.record_phrase(SECONDS)
        if audio is None:
            print("      recording failed, skipping")
            continue
        emb = voice_embedding.embed(audio)
        if emb is None:
            print("      not enough speech in that recording, skipping")
            continue
        embeddings.append(emb)
        print(f"      captured ({len(emb)} values)")
    return embeddings


def pairwise(a_list, b_list=None):
    """Similarity scores. Within one list, or across two."""
    scores = []
    if b_list is None:
        for i in range(len(a_list)):
            for j in range(i + 1, len(a_list)):
                scores.append(voice_embedding.similarity(a_list[i], a_list[j]))
    else:
        for a in a_list:
            for b in b_list:
                scores.append(voice_embedding.similarity(a, b))
    return [s for s in scores if s is not None]


def describe(label, scores):
    if not scores:
        print(f"  {label}: no scores")
        return None
    arr = np.array(scores)
    print(f"  {label}")
    print(f"    n={len(arr)}  min={arr.min():.4f}  mean={arr.mean():.4f}  max={arr.max():.4f}")
    print(f"    all: {', '.join(f'{s:.4f}' for s in sorted(arr))}")
    return arr


def main() -> int:
    cfg = load_config(CONFIG_FILE)
    listener = MicrophoneListener(None, cfg)
    if listener.device_index is None:
        print("No usable microphone. Run list_audio_devices.py first.")
        return 1

    print(BAR)
    print("  VOICE THRESHOLD CALIBRATION")
    print(BAR)
    print(f"  Microphone device {listener.device_index}.")
    print(f"  Two people, {len(PHRASES)} phrases each, {SECONDS:.0f} seconds per phrase.")
    print("\n  If a second person is not available, the same person can do the")
    print("  second set in a deliberately different voice — but note in the")
    print("  results that it was not a genuinely different speaker, because")
    print("  that flatters the numbers.")
    input("\n  Press Enter when ready...")

    name_a = input("  Name of the FIRST person: ").strip() or "Person A"
    a = record_set(listener, name_a)
    if len(a) < 2:
        print("\nNot enough usable recordings from the first person.")
        return 1

    input(f"\n  Now the second person. Press Enter when {name_a} has swapped out...")
    name_b = input("  Name of the SECOND person: ").strip() or "Person B"
    b = record_set(listener, name_b)
    if len(b) < 1:
        print("\nNot enough usable recordings from the second person.")
        return 1

    print("\n" + BAR)
    print("  RESULTS")
    print(BAR)
    same_a = describe(f"SAME person ({name_a} vs {name_a})", pairwise(a))
    same_b = describe(f"SAME person ({name_b} vs {name_b})", pairwise(b)) if len(b) > 1 else None
    diff = describe(f"DIFFERENT people ({name_a} vs {name_b})", pairwise(a, b))

    same_all = np.concatenate([s for s in (same_a, same_b) if s is not None])
    if diff is None or same_all.size == 0:
        return 1

    print("\n" + BAR)
    print("  RECOMMENDATION")
    print(BAR)
    same_min, diff_max = same_all.min(), diff.max()
    print(f"  lowest  same-person score : {same_min:.4f}")
    print(f"  highest different-person  : {diff_max:.4f}")
    gap = same_min - diff_max
    print(f"  separation                : {gap:+.4f}")

    current = voice_embedding.MATCH_THRESHOLD
    print(f"\n  current threshold         : {current}")

    if gap <= 0:
        print("\n  THE TWO GROUPS OVERLAP. No threshold separates these voices:")
        print("  any value that accepts the same person also accepts the other one.")
        print("  Recommendation: do NOT use voice for identification. Leave the")
        print("  threshold high so it effectively never fires, and rely on the face.")
        return 0

    suggested = round(diff_max + gap / 2, 3)
    print(f"\n  suggested threshold       : {suggested}")
    print(f"    (midway between the groups, so both have {gap/2:.4f} of margin)")
    below = int((same_all < current).sum())
    if below:
        print(f"\n  At the current {current}, {below} of {same_all.size} same-person")
        print("  comparisons would be REJECTED — too strict.")
    above = int((diff > current).sum())
    if above:
        print(f"\n  At the current {current}, {above} of {diff.size} different-person")
        print("  comparisons would be ACCEPTED — too loose.")
    if not below and not above:
        print(f"\n  The current {current} already separates these two voices correctly.")

    print("\n  Set the chosen value in voice_id/voice_embedding.py")
    print("  (MATCH_THRESHOLD), and paste this output into the commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
