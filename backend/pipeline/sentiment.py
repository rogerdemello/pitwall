"""Sentiment of *what was said*, as the counterpart to prosody's *how*.

Deliberately a separate signal from prosody. The whole point of the fusion layer
is that these two can disagree, and the disagreement is the interesting part.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline import device

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@dataclass
class TextSentiment:
    label: str          # negative | neutral | positive
    negative: float
    neutral: float
    positive: float

    @property
    def polarity(self) -> float:
        """-1 (fully negative) .. +1 (fully positive)."""
        return round(self.positive - self.negative, 4)


@functools.lru_cache(maxsize=1)
def _load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    return tok, device.place(model, prefer_fp16=False)


def analyse(text: str) -> TextSentiment:
    if not text or not text.strip():
        return TextSentiment("neutral", 0.0, 1.0, 0.0)

    tok, model = _load()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**device.inputs_to(inputs)).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

    # id2label from the model config rather than positional assumption.
    by_label = {model.config.id2label[i].lower(): float(p) for i, p in enumerate(probs)}
    return TextSentiment(
        label=max(by_label, key=by_label.get),
        negative=round(by_label.get("negative", 0.0), 4),
        neutral=round(by_label.get("neutral", 0.0), 4),
        positive=round(by_label.get("positive", 0.0), 4),
    )
