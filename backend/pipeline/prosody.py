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
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
SAMPLE_RATE = 16000


MAX_WINDOW_S = 15  # wav2vec2-large cost grows with length; cap it


@dataclass
class Affect:
    arousal: float   # 0 = calm/still, 1 = highly activated
    dominance: float  # 0 = diffident, 1 = assertive/in-control
    valence: float   # 0 = negative, 1 = positive
    peak_arousal: float = 0.0   # highest arousal in any window
    min_valence: float = 0.0    # lowest valence in any window
    windows: int = 1


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
    model.eval()
    return processor, model


def _predict(audio: np.ndarray, sampling_rate: int) -> tuple[float, float, float]:
    processor, model = _load()
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs.input_values).squeeze().cpu().numpy()
    return tuple(float(v) for v in out)  # arousal, dominance, valence


def analyse(audio: np.ndarray, sampling_rate: int = SAMPLE_RATE) -> Affect:
    """Predict arousal / dominance / valence for one clip.

    Long clips are split into windows: inference cost on wav2vec2-large scales
    with duration, and a 50-second engineer briefing is not one emotional state
    anyway. We report the mean plus the extremes, because a single spike inside
    an otherwise calm message is exactly the signal we care about.
    """
    win = MAX_WINDOW_S * sampling_rate
    chunks = [audio[i:i + win] for i in range(0, len(audio), win)] or [audio]
    # A trailing sliver carries no reliable prosody; fold it into the previous
    # window. Pop first, then append - doing it in one expression re-indexes the
    # assignment target against the already-shortened list.
    if len(chunks) > 1 and len(chunks[-1]) < sampling_rate:
        tail = chunks.pop()
        chunks[-1] = np.concatenate([chunks[-1], tail])

    preds = [_predict(c, sampling_rate) for c in chunks]
    arousals = [p[0] for p in preds]
    valences = [p[2] for p in preds]

    return Affect(
        arousal=round(float(np.mean(arousals)), 4),
        dominance=round(float(np.mean([p[1] for p in preds])), 4),
        valence=round(float(np.mean(valences)), 4),
        peak_arousal=round(max(arousals), 4),
        min_valence=round(min(valences), 4),
        windows=len(chunks),
    )
