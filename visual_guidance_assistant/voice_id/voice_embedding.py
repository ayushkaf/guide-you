"""Speaker fingerprint from MFCC statistics — numpy only, no new dependencies.

Why this and not resemblyzer / speechbrain (both checked, both rejected):
  - resemblyzer pulls webrtcvad, which ships only as a source tarball and needs
    Visual C++ this machine does not have, and it would downgrade numpy from
    2.5.1 to 2.4.6 underneath the verified torch/opencv install.
  - speechbrain pulls torchaudio 2.11 against torch 2.13 (a mismatch that
    DLL-fails on import here) and downloads pretrained weights at runtime, which
    makes a care device depend on the network to boot.

HONEST LIMITS — read before trusting this for anything.
MFCC means and variances capture voice timbre and channel characteristics. They
are NOT a reliable speaker identifier: they drift with microphone position,
room acoustics, illness, tiredness, and background noise, and similar-sounding
family members can score close together. This is a *supplementary hint* only.

The consequence is designed for, not hoped away: voice never decides identity on
its own. Face recognition is primary; voice is consulted only when the face is
unavailable or ambiguous, and only above a deliberately high similarity
threshold. When nothing is confident, the caller falls back to the safest
interaction mode rather than guessing. See dialogue_processor.resolve_speaker().
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

SAMPLE_RATE = 16000
N_MFCC = 20
N_FILTERS = 26
FFT_SIZE = 512
FRAME_LENGTH = 400   # 25ms at 16kHz
FRAME_STEP = 160     # 10ms at 16kHz
PRE_EMPHASIS = 0.97

# Deliberately high, and NEEDS CALIBRATION against real enrolled voices.
#
# Measured on synthetic signals, this metric's useful range is roughly
# 0.95-1.00, not 0-1: two different synthetic speakers still scored 0.97. A
# threshold picked from intuition (0.86 was the first guess) would therefore
# have matched literally everyone. Until there are two real enrolled voices to
# measure against, this stays high and the ambiguity margin below does the real
# work.
#
# The structural safety rule matters more than this number, and lives in
# dialogue_processor.resolve_speaker(): a voice match can NEVER by itself grant
# caregiver mode. Caregiver mode requires face confirmation. The asymmetry is
# deliberate — a caregiver wrongly given patient mode just gets a warmer,
# shorter answer, which is harmless; a patient wrongly given caregiver mode
# would get medical questions answered instead of redirected, which is not.
MATCH_THRESHOLD = 0.98

# Minimum gap between the best and second-best candidate. With scores this
# tightly packed, "clearly ahead of everyone else" is a more meaningful
# criterion than "above an absolute bar".
AMBIGUITY_MARGIN = 0.01

# Below this much speech we simply refuse to produce an embedding rather than
# storing a fingerprint built from silence.
MIN_SECONDS = 1.0


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_filters: int, fft_size: int) -> np.ndarray:
    low_mel = _hz_to_mel(0.0)
    high_mel = _hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_filters, fft_size // 2 + 1))
    for i in range(1, n_filters + 1):
        left, centre, right = bins[i - 1], bins[i], bins[i + 1]
        # Guard against degenerate (zero-width) bands at low frequencies.
        if centre == left:
            centre = left + 1
        if right == centre:
            right = centre + 1
        if right >= filters.shape[1]:
            right = filters.shape[1] - 1
        if centre >= right or left >= centre:
            continue
        filters[i - 1, left:centre] = (np.arange(left, centre) - left) / (centre - left)
        filters[i - 1, centre:right] = (right - np.arange(centre, right)) / (right - centre)
    return filters


# Built once — it depends only on constants.
_FILTERBANK = _mel_filterbank(SAMPLE_RATE, N_FILTERS, FFT_SIZE)

# Orthonormal DCT-II matrix, so we don't need scipy.
_DCT = np.zeros((N_MFCC, N_FILTERS))
for _k in range(N_MFCC):
    _DCT[_k] = np.cos(np.pi * _k * (2 * np.arange(N_FILTERS) + 1) / (2 * N_FILTERS))
_DCT[0] *= 1.0 / np.sqrt(2.0)
_DCT *= np.sqrt(2.0 / N_FILTERS)


def _to_mono_float(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float64)
    # int16 input -> normalise to [-1, 1]
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak > 1.5:
        samples = samples / 32768.0
    return samples


def compute_mfcc(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """Return an (n_frames, N_MFCC) matrix, or None if there isn't enough audio."""
    samples = _to_mono_float(audio)
    if samples.size < FRAME_LENGTH * 2:
        return None

    emphasised = np.append(samples[0], samples[1:] - PRE_EMPHASIS * samples[:-1])

    n_frames = 1 + (len(emphasised) - FRAME_LENGTH) // FRAME_STEP
    if n_frames < 3:
        return None

    indices = (
        np.tile(np.arange(FRAME_LENGTH), (n_frames, 1))
        + np.tile(np.arange(0, n_frames * FRAME_STEP, FRAME_STEP), (FRAME_LENGTH, 1)).T
    )
    frames = emphasised[indices] * np.hamming(FRAME_LENGTH)

    power = (np.abs(np.fft.rfft(frames, FFT_SIZE)) ** 2) / FFT_SIZE
    energy = power @ _FILTERBANK.T
    energy = np.where(energy <= 1e-12, 1e-12, energy)
    log_energy = np.log(energy)

    mfcc = log_energy @ _DCT.T

    # Cepstral mean normalisation removes a lot of the microphone/room signature,
    # leaving relatively more of the speaker.
    mfcc -= mfcc.mean(axis=0, keepdims=True)
    return mfcc


def _voiced_frames(mfcc: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Drop near-silent frames so pauses don't dominate the fingerprint."""
    samples = _to_mono_float(audio)
    n_frames = mfcc.shape[0]
    rms = np.array([
        np.sqrt(np.mean(samples[i * FRAME_STEP: i * FRAME_STEP + FRAME_LENGTH] ** 2) + 1e-12)
        for i in range(n_frames)
    ])
    if not np.any(np.isfinite(rms)):
        return mfcc
    keep = rms > max(rms.max() * 0.15, 1e-4)
    return mfcc[keep] if keep.sum() >= 3 else mfcc


def embed(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """Fixed-length speaker fingerprint, L2-normalised. None if unusable."""
    samples = _to_mono_float(audio)
    if samples.size < MIN_SECONDS * sample_rate:
        return None

    mfcc = compute_mfcc(samples, sample_rate)
    if mfcc is None:
        return None
    mfcc = _voiced_frames(mfcc, samples)
    if mfcc.shape[0] < 3:
        return None

    vector = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])

    # Z-score across dimensions before normalising. Raw MFCC statistic vectors
    # all point in a very similar direction — the shared "this is human speech
    # through a laptop mic" component dominates — so plain cosine similarity
    # compressed every pair into 0.97-1.00 and could not separate speakers at
    # all. Removing each vector's own mean and scale strips that common
    # component and leaves the speaker-specific shape, which is what we want to
    # compare. Measured separation improves by more than an order of magnitude.
    vector = vector - vector.mean()
    spread = vector.std()
    if spread < 1e-8:
        return None
    vector = vector / spread

    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return None
    return vector / norm


def similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity mapped to 0..1 (both inputs are already L2-normalised)."""
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if va.size != vb.size or va.size == 0:
        return 0.0
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip((va @ vb) / (na * nb), -1.0, 1.0) * 0.5 + 0.5)


def best_match(
    embedding: Sequence[float],
    names: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    threshold: float = MATCH_THRESHOLD,
) -> Tuple[Optional[str], float]:
    """Closest enrolled voice, or (None, score) when nothing clears the bar.

    Also returns (None, ...) when the top two candidates are within
    AMBIGUITY_MARGIN of each other: an ambiguous match is not a match, and
    saying so is better than picking the marginally higher number.
    """
    if embedding is None or not names:
        return None, 0.0
    scores = sorted(
        ((similarity(embedding, ref), name) for name, ref in zip(names, embeddings)),
        reverse=True,
    )
    top_score, top_name = scores[0]
    if top_score < threshold:
        return None, top_score
    if len(scores) > 1 and (top_score - scores[1][0]) < AMBIGUITY_MARGIN:
        return None, top_score
    return top_name, top_score
