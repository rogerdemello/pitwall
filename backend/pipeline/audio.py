"""Conditioning the audio before the affect model reads it.

The problem this addresses is a confound, not a quality issue. Team radio gain
varies systematically by team and by era - different radios, different broadcast
encoding, different years of compression. An arousal model reads loudness. So
some unknown share of what v1 calls "driver arousal" may be a measurement of
*which team's radio it is*, and that would be invisible in every number the
project publishes.

Normalising the model's input removes the confound. But a driver genuinely
shouting *is* louder, and that is real signal, so removing loudness entirely
would throw away something true. The resolution is to do both: normalise what
the model sees, and emit loudness separately as an explicit feature, then let
the fitting decide whether it earns any weight. Neither choice is made by
assertion.

What is deliberately NOT here: neural speech enhancement. DeepFilterNet, demucs
and friends alter exactly the cues an affect model reads - spectral tilt,
breathiness, vocal effort - and the model was trained on un-enhanced speech, so
enhancement is a distribution shift dressed up as cleaning. A high-pass and a
gain change are safe because they are the two operations that do not touch the
shape of the spectrum above the rumble.

Loudness is ITU-R BS.1770 (K-weighted, gated), implemented here with scipy
rather than adding pyloudnorm for two biquads and a mean.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
from scipy import signal

SAMPLE_RATE = 16000

#: EBU R128 programme loudness target. The absolute value matters less than
#: every clip sharing one, which is the point.
TARGET_LUFS = -23.0

#: Radio rumble, handling noise and DC live below here and carry no speech.
HIGHPASS_HZ = 80.0

#: Never amplify by more than this. A clip that is nearly silent would otherwise
#: have its noise floor lifted to speech level and be scored as a real message.
MAX_GAIN_DB = 20.0


@dataclass
class Loudness:
    lufs: float | None      # integrated, K-weighted, gated
    rms_db: float           # simple RMS, always computable
    peak_db: float
    gain_applied_db: float
    clipped: bool


@functools.lru_cache(maxsize=8)
def _k_weighting(sr: int):
    """The two BS.1770 pre-filters: a shelving boost, then a high-pass.

    Coefficients are the standard's, specified at 48 kHz and bilinear-warped to
    the working rate so this stays correct at 16 kHz.
    """
    # Stage 1: high-frequency shelving filter (head-related, +4 dB above ~2 kHz).
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = np.tan(np.pi * f0 / sr)
    Vh = np.power(10.0, G / 20.0)
    Vb = np.power(Vh, 0.4996667741545416)
    a0 = 1.0 + K / Q + K * K
    b_shelf = np.array([
        (Vh + Vb * K / Q + K * K) / a0,
        2.0 * (K * K - Vh) / a0,
        (Vh - Vb * K / Q + K * K) / a0,
    ])
    a_shelf = np.array([1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0])

    # Stage 2: RLB high-pass.
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = np.tan(np.pi * f0 / sr)
    b_hp = np.array([1.0, -2.0, 1.0])
    a_hp = np.array([
        1.0,
        2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K),
        (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K),
    ])
    return (b_shelf, a_shelf), (b_hp, a_hp)


def integrated_lufs(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float | None:
    """Gated K-weighted loudness, per ITU-R BS.1770.

    None when the clip is shorter than one 400 ms block, or when gating removes
    every block - both of which mean "too quiet or too short to have a loudness",
    not "zero".
    """
    audio = np.asarray(audio, dtype=np.float64).flatten()
    block = int(0.400 * sr)
    if len(audio) < block:
        return None

    (b1, a1), (b2, a2) = _k_weighting(sr)
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, audio))

    step = block // 4  # 75% overlap, as the standard specifies
    powers = [np.mean(y[i:i + block] ** 2)
              for i in range(0, len(y) - block + 1, step)]
    powers = np.array([p for p in powers if p > 0])
    if not len(powers):
        return None

    loud = -0.691 + 10 * np.log10(powers)

    # Two-stage gate: drop silence, then drop anything 10 LU below the mean of
    # what is left. Without it, a clip that is mostly dead air reads far quieter
    # than the speech in it actually is.
    keep = powers[loud > -70.0]
    if not len(keep):
        return None
    relative = -0.691 + 10 * np.log10(np.mean(keep)) - 10.0
    keep = powers[(loud > -70.0) & (loud > relative)]
    if not len(keep):
        return None
    return float(-0.691 + 10 * np.log10(np.mean(keep)))


def measure(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Loudness:
    """Loudness of a clip, without changing it."""
    a = np.asarray(audio, dtype=np.float32)
    if not len(a):
        return Loudness(None, -np.inf, -np.inf, 0.0, False)
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(a)))
    return Loudness(
        lufs=integrated_lufs(a, sr),
        rms_db=round(20 * np.log10(rms), 2) if rms > 0 else -np.inf,
        peak_db=round(20 * np.log10(peak), 2) if peak > 0 else -np.inf,
        gain_applied_db=0.0,
        clipped=bool(peak >= 0.999),
    )


def highpass(audio: np.ndarray, sr: int = SAMPLE_RATE,
             cutoff: float = HIGHPASS_HZ) -> np.ndarray:
    """Remove rumble and DC below `cutoff`. Nothing above it is touched."""
    if len(audio) < 32 or cutoff <= 0 or cutoff >= sr / 2:
        return audio
    sos = signal.butter(2, cutoff / (sr / 2), btype="highpass", output="sos")
    return signal.sosfilt(sos, audio).astype(np.float32)


def normalise(audio: np.ndarray, sr: int = SAMPLE_RATE,
              target_lufs: float = TARGET_LUFS,
              reference: np.ndarray | None = None) -> tuple[np.ndarray, Loudness]:
    """High-pass, then bring the clip to `target_lufs`.

    `reference` is the audio the gain is *measured* on while the gain is
    *applied* to the whole clip. That matters: loudness should be measured over
    speech only, or a clip that is 80% silence gets over-amplified until its
    noise floor sits where the speech should be.

    Returns the conditioned audio and what was measured, so the caller can emit
    the loudness as a feature instead of losing it.
    """
    a = highpass(np.asarray(audio, dtype=np.float32), sr)
    if not len(a):
        return a, Loudness(None, -np.inf, -np.inf, 0.0, False)

    measured_on = a if reference is None else highpass(
        np.asarray(reference, dtype=np.float32), sr)
    before = measure(measured_on, sr)

    if before.lufs is None:
        # Too short or too quiet to measure. Leave the audio alone rather than
        # inventing a gain for it.
        return a, before

    gain_db = float(np.clip(target_lufs - before.lufs, -MAX_GAIN_DB, MAX_GAIN_DB))
    out = a * (10.0 ** (gain_db / 20.0))

    # Scale back rather than clip: distortion would change the spectrum, which
    # is the one thing this module is careful not to do.
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.999:
        out = out * (0.999 / peak)
        gain_db += 20 * np.log10(0.999 / peak)

    return out.astype(np.float32), Loudness(
        lufs=round(before.lufs, 2),
        rms_db=before.rms_db,
        peak_db=before.peak_db,
        gain_applied_db=round(gain_db, 2),
        clipped=before.clipped,
    )


def prepare(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, dict]:
    """Full conditioning path: detect speech, normalise against it, report both.

    This is what the pipeline calls. The returned dict is emitted into the raw
    record so every downstream decision can see what the audio looked like.
    """
    from pipeline import vad

    regions = vad.detect(audio, sr)
    speech = vad.speech_only(audio, regions, sr)
    conditioned, loud = normalise(
        audio, sr, reference=speech if len(speech) else None)

    return conditioned, {
        "vad": regions.to_dict(),
        "loudness": {
            "lufs": loud.lufs,
            "rms_db": loud.rms_db if np.isfinite(loud.rms_db) else None,
            "peak_db": loud.peak_db if np.isfinite(loud.peak_db) else None,
            "gain_applied_db": loud.gain_applied_db,
            "clipped": loud.clipped,
            "target_lufs": target_lufs_of(loud),
        },
    }


def target_lufs_of(_: Loudness) -> float:
    return TARGET_LUFS
