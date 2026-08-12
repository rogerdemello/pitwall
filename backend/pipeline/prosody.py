"""Dimensional affect from voice: how it was said, not what was said.

Uses audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim, which predicts
continuous arousal / dominance / valence in roughly 0..1 rather than emitting a
handful of discrete emotion classes.

Continuous dimensions matter here. "Stressed" and "tired" are not two unrelated
buckets: both are low-valence, and what separates them is arousal. A dimensional
model lets us place a driver on that plane and track drift across a stint, which
a 4-way classifier cannot express.

The model has no classification head we can use directly, so we attach the
regression head described on the model card.

Three things changed from v1, all of which were measurement problems rather than
model problems:

  **Windows are 4s over speech only, not 15s over everything.** The model was
  trained on MSP-Podcast utterances averaging ~5s, so a 15s window is far
  outside its training regime. Worse, v1 windowed the *whole clip* including
  silence and channel noise - and 304 clips exceed 15s while the longest is
  105s, much of which is dead air between transmissions.

  **Every window is emitted; aggregation happens in stage 2.** v1 mean-pooled
  inside this function, so the choice of aggregation was baked into an hour of
  GPU inference and could not be revisited without re-running it. Emitting
  windows makes mean-vs-median-vs-p90 a cheap stage-2 decision that can be
  picked by held-out performance instead of by assertion.

  **peak/min become p90/p10.** Raw max over N windows is upward-biased in N: a
  105s clip has ~50 windows and will always "peak" higher than a 5s clip with
  two, so v1's peak_arousal partly measured clip length. p90/p10 are stable
  under that, and the bias is reported rather than assumed away.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from pipeline import device

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
SAMPLE_RATE = 16000

#: Matched to the model's training regime (MSP-Podcast utterances average ~5s).
#: v1 used 15s, which is far outside it.
WINDOW_S = 4.0
HOP_S = 2.0
#: Below this a window carries no reliable prosody; it is padded and flagged
#: rather than dropped, so short transmissions still get a score.
MIN_WINDOW_S = 1.2
#: Batch of fixed-length windows through the GPU at once.
BATCH = 32


@dataclass
class Window:
    """One scored window, with where it came from."""
    index: int
    start: float
    end: float
    arousal: float
    dominance: float
    valence: float
    short: bool = False

    def to_dict(self) -> dict:
        return {
            "i": self.index,
            "t0": round(self.start, 3), "t1": round(self.end, 3),
            "arousal": self.arousal, "dominance": self.dominance,
            "valence": self.valence,
            **({"short": True} if self.short else {}),
        }


@dataclass
class Affect:
    arousal: float   # 0 = calm/still, 1 = highly activated
    dominance: float  # 0 = diffident, 1 = assertive/in-control
    valence: float   # 0 = negative, 1 = positive
    peak_arousal: float = 0.0   # p90 across windows, not raw max - see module docstring
    min_valence: float = 0.0    # p10 across windows
    windows: int = 1
    # New in v2, defaulted so v1 construction keeps working.
    window_scores: list[Window] = field(default_factory=list)
    arousal_range: float = 0.0   # p90 - p10, how much the voice moved
    arousal_slope: float = 0.0   # OLS slope over window time; did it rise?
    voiced_fraction: float = 1.0
    speech_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "arousal": self.arousal, "dominance": self.dominance,
            "valence": self.valence,
            "peak_arousal": self.peak_arousal, "min_valence": self.min_valence,
            "windows": self.windows,
            "arousal_range": self.arousal_range,
            "arousal_slope": self.arousal_slope,
            "voiced_fraction": self.voiced_fraction,
            "speech_s": round(self.speech_s, 3),
            "window_scores": [w.to_dict() for w in self.window_scores],
        }


class _RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(torch.tanh(self.dense(features)))
        return self.out_proj(x)


class _AffectModel(Wav2Vec2PreTrainedModel):
    """wav2vec2 trunk + mean pooling + regression head (per the model card)."""

    # Recent transformers versions consult this during weight loading. Nothing is
    # tied in this architecture, so an empty mapping is correct.
    all_tied_weights_keys: dict = {}

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0]
        pooled = torch.mean(hidden, dim=1)
        return self.classifier(pooled)


@functools.lru_cache(maxsize=1)
def _load():
    config = AutoConfig.from_pretrained(MODEL_ID)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = _AffectModel.from_pretrained(MODEL_ID, config=config)
    # fp32 deliberately: wav2vec2-large's feature-encoder group norm can overflow
    # in fp16 and emit NaNs, which mean-pooling then turns into a plausible-
    # looking score rather than an obvious failure. See pipeline/device.py.
    return processor, device.place(model, prefer_fp16=False)


def _predict_batch(chunks: list[np.ndarray], sampling_rate: int) -> list[tuple]:
    """Score a batch of equal-length windows in one forward pass."""
    processor, model = _load()
    if not chunks:
        return []
    width = max(len(c) for c in chunks)
    padded = np.stack([np.pad(c, (0, width - len(c))) for c in chunks])
    inputs = processor(list(padded), sampling_rate=sampling_rate,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(device.inputs_to(inputs).input_values).cpu().numpy()
    out = np.atleast_2d(out)
    if not np.all(np.isfinite(out)):
        raise ValueError(
            f"{MODEL_ID} returned non-finite affect - refusing to score it")
    return [tuple(float(v) for v in row) for row in out]  # arousal, dominance, valence


def _predict(audio: np.ndarray, sampling_rate: int) -> tuple[float, float, float]:
    """Single-window convenience, kept for callers that score one span."""
    return _predict_batch([audio], sampling_rate)[0]


def _spans_to_windows(audio: np.ndarray, spans, sampling_rate: int
                      ) -> list[tuple[float, float, np.ndarray]]:
    """Sliding windows inside each speech span, never crossing a boundary.

    Not crossing matters: a window that straddles the gap between two
    transmissions averages two speakers, which is exactly the confusion the
    whole speaker-attribution effort is trying to remove.
    """
    win = int(WINDOW_S * sampling_rate)
    hop = int(HOP_S * sampling_rate)
    out = []
    for span in spans:
        a, b = span.samples(sampling_rate)
        a, b = max(0, a), min(len(audio), b)
        if b - a < int(MIN_WINDOW_S * sampling_rate):
            if b > a:
                out.append((a / sampling_rate, b / sampling_rate, audio[a:b]))
            continue
        start = a
        while start < b:
            end = min(start + win, b)
            if end - start >= int(MIN_WINDOW_S * sampling_rate):
                out.append((start / sampling_rate, end / sampling_rate,
                            audio[start:end]))
            if end >= b:
                break
            start += hop
    return out


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return float(s[min(len(s) - 1, int(len(s) * q))])


def _slope(times: list[float], values: list[float]) -> float:
    """OLS slope of value against time, in units per second.

    Enables a claim the clip-level mean cannot make: the driver's arousal *rose
    over the course of the message*.
    """
    n = len(values)
    if n < 3:
        return 0.0
    mt, mv = sum(times) / n, sum(values) / n
    denom = sum((t - mt) ** 2 for t in times)
    if denom <= 0:
        return 0.0
    return round(sum((t - mt) * (v - mv) for t, v in zip(times, values)) / denom, 5)


def analyse(audio: np.ndarray, sampling_rate: int = SAMPLE_RATE,
            use_vad: bool = True) -> Affect:
    """Predict arousal / dominance / valence for one clip.

    With `use_vad`, windows are drawn from detected speech only and the audio is
    loudness-normalised first, so the model is not scoring silence, channel
    noise, or one team's radio gain. `use_vad=False` reproduces v1's behaviour
    for comparison.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if not len(audio):
        return Affect(0.5, 0.5, 0.5, windows=0, voiced_fraction=0.0)

    clip_s = len(audio) / sampling_rate
    windows: list[tuple[float, float, np.ndarray]] = []
    voiced_fraction, speech_s = 1.0, clip_s

    if use_vad:
        from pipeline import audio as audio_mod
        from pipeline import vad

        regions = vad.detect(audio, sampling_rate)
        voiced_fraction, speech_s = regions.voiced_fraction, regions.speech_s
        if regions.is_silent:
            # No speech: return a neutral score flagged as such rather than
            # scoring channel noise and calling it a driver state.
            return Affect(0.5, 0.5, 0.5, windows=0, voiced_fraction=0.0,
                          speech_s=0.0)
        speech = vad.speech_only(audio, regions, sampling_rate)
        conditioned, _ = audio_mod.normalise(audio, sampling_rate, reference=speech)
        windows = _spans_to_windows(conditioned, regions.spans, sampling_rate)

    if not windows:
        # v1 path, or VAD found spans too short to window: fall back to fixed
        # windows over the whole clip.
        win = int(WINDOW_S * sampling_rate)
        windows = [(i / sampling_rate, min(i + win, len(audio)) / sampling_rate,
                    audio[i:i + win])
                   for i in range(0, len(audio), win)] or [(0.0, clip_s, audio)]

    scored: list[Window] = []
    for i in range(0, len(windows), BATCH):
        batch = windows[i:i + BATCH]
        preds = _predict_batch([w[2] for w in batch], sampling_rate)
        for j, ((t0, t1, chunk), (a, d, v)) in enumerate(zip(batch, preds)):
            scored.append(Window(
                index=i + j, start=t0, end=t1,
                arousal=round(a, 4), dominance=round(d, 4), valence=round(v, 4),
                short=(t1 - t0) < MIN_WINDOW_S,
            ))

    arousals = [w.arousal for w in scored]
    valences = [w.valence for w in scored]
    mids = [(w.start + w.end) / 2 for w in scored]

    return Affect(
        # The clip-level mean stays the headline so stage 2 keeps working
        # unchanged; every alternative aggregation is recoverable from
        # window_scores without re-running the model.
        arousal=round(float(np.mean(arousals)), 4),
        dominance=round(float(np.mean([w.dominance for w in scored])), 4),
        valence=round(float(np.mean(valences)), 4),
        peak_arousal=round(_pct(arousals, 0.90), 4),
        min_valence=round(_pct(valences, 0.10), 4),
        windows=len(scored),
        window_scores=scored,
        arousal_range=round(_pct(arousals, 0.90) - _pct(arousals, 0.10), 4),
        arousal_slope=_slope(mids, arousals),
        voiced_fraction=voiced_fraction,
        speech_s=speech_s,
    )
