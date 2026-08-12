"""Calibrate affect scores to the F1 radio domain.

The prosody model was trained on MSP-Podcast: studio-quality conversational
speech. Team radio is 3-second bursts through a compressed, clipped, noisy
channel. Measured on real clips, the model's outputs collapse into a narrow band
(arousal ~0.55-0.80, valence ~0.50-0.78) and essentially never go near the ends
of its nominal 0..1 range.

Applying a textbook 0.5 threshold to that band labels almost every message
"high arousal, positive valence", which is how our first pass ended up calling
a whole Grand Prix "Energised" with a stress index pinned between 44 and 54.

So we stop asking "is this arousal high in absolute terms?" and start asking
"is this arousal high *for F1 radio*?" - a percentile rank against the corpus.
That restores full dynamic range and makes the scale mean something concrete:
DSI 90 is the top decile of stress for this race, not an abstract 0.9.

The calibration is fitted from precomputed raw outputs, so it costs nothing at
request time.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field


@dataclass
class Calibrator:
    """Empirical percentile mapping, fitted on a corpus of raw affect scores."""

    arousal: list[float] = field(default_factory=list)
    valence: list[float] = field(default_factory=list)
    dominance: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, records: list[dict]) -> "Calibrator":
        def col(name):
            return sorted(r[name] for r in records if r.get(name) is not None)
        return cls(arousal=col("arousal"), valence=col("valence"), dominance=col("dominance"))

    def _rank(self, sorted_vals: list[float], value: float) -> float:
        if not sorted_vals:
            return 0.5
        lo = bisect.bisect_left(sorted_vals, value)
        hi = bisect.bisect_right(sorted_vals, value)
        # Midpoint of the tied range, so identical values map consistently.
        return ((lo + hi) / 2) / len(sorted_vals)

    def pct_arousal(self, v: float) -> float:
        return round(self._rank(self.arousal, v), 4)

    def pct_valence(self, v: float) -> float:
        return round(self._rank(self.valence, v), 4)

    def pct_dominance(self, v: float) -> float:
        return round(self._rank(self.dominance, v), 4)

    def summary(self) -> dict:
        """Reported on the Evidence screen to justify that calibration is needed."""
        def stats(vals):
            if not vals:
                return {}
            n = len(vals)
            return {
                "n": n,
                "min": round(vals[0], 3),
                "p10": round(vals[int(n * 0.10)], 3),
                "median": round(vals[n // 2], 3),
                "p90": round(vals[min(n - 1, int(n * 0.90))], 3),
                "max": round(vals[-1], 3),
            }
        return {
            "arousal": stats(self.arousal),
            "valence": stats(self.valence),
            "dominance": stats(self.dominance),
        }

    def to_json(self, path: str) -> None:
        json.dump(
            {"arousal": self.arousal, "valence": self.valence, "dominance": self.dominance},
            open(path, "w", encoding="utf-8"),
        )

    @classmethod
    def from_json(cls, path: str) -> "Calibrator":
        d = json.load(open(path, encoding="utf-8"))
        return cls(arousal=d["arousal"], valence=d["valence"], dominance=d["dominance"])
