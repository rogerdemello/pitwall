"""Turning per-window affect scores into one number per clip.

v1 mean-pooled inside `prosody.analyse`, which meant the choice of aggregation
was baked into an hour of GPU inference: changing it required re-running the
whole corpus. Now every window is emitted and the choice lives here, where it
costs milliseconds and can be picked by held-out performance rather than by
assertion.

Which is the point. Mean is not obviously right. A 40-second transmission where
the driver is calm for 35 seconds and shouts for 5 has a mean that says "calm"
and a p90 that says "something happened", and only one of those is the thing a
race engineer wants to be told. The honest answer is to measure which one
predicts the labels and use that - which requires the alternatives to exist as
named, comparable options.

Nothing here is the default yet. `prosody.Affect.arousal` remains the mean so
stage 2 keeps working unchanged; these are the candidates an ablation ranks.
"""

from __future__ import annotations

import statistics as st
from typing import Callable, Sequence

#: A strategy takes the per-window values (and their durations, for the weighted
#: ones) and returns one number.
Strategy = Callable[[Sequence[float], Sequence[float]], float]


def _clean(values: Sequence[float], weights: Sequence[float] | None
           ) -> tuple[list[float], list[float]]:
    vals = [float(v) for v in values]
    if weights is None or len(weights) != len(vals):
        weights = [1.0] * len(vals)
    return vals, [max(0.0, float(w)) for w in weights]


def mean(values, weights=None) -> float:
    v, _ = _clean(values, weights)
    return st.fmean(v) if v else 0.0


def median(values, weights=None) -> float:
    v, _ = _clean(values, weights)
    return st.median(v) if v else 0.0


def duration_weighted(values, weights=None) -> float:
    """Long windows count more. Corrects for a clip whose windows differ in size."""
    v, w = _clean(values, weights)
    total = sum(w)
    return sum(x * y for x, y in zip(v, w)) / total if total else mean(v)


def trimmed(values, weights=None, trim: float = 0.1) -> float:
    """Mean with the extreme tails dropped - robust to a single bad window."""
    v, _ = _clean(values, weights)
    if len(v) < 5:
        return mean(v)
    v = sorted(v)
    k = max(1, int(len(v) * trim))
    return st.fmean(v[k:-k])


def p90(values, weights=None) -> float:
    """The high tail. 'Did anything happen in this message?' rather than
    'what was the average of this message?'"""
    return _pct(values, 0.90)


def p10(values, weights=None) -> float:
    return _pct(values, 0.10)


def last_window(values, weights=None) -> float:
    """How the message ended, which is often how the driver was left."""
    v, _ = _clean(values, weights)
    return v[-1] if v else 0.0


def max_run(values, weights=None, n: int = 2) -> float:
    """Highest mean over `n` consecutive windows.

    A sustained spike is a different thing from one loud window, and this is the
    cheapest way to tell them apart.
    """
    v, _ = _clean(values, weights)
    if len(v) < n:
        return mean(v)
    return max(st.fmean(v[i:i + n]) for i in range(len(v) - n + 1))


def _pct(values: Sequence[float], q: float) -> float:
    v = sorted(float(x) for x in values)
    if not v:
        return 0.0
    return v[min(len(v) - 1, int(len(v) * q))]


#: Every candidate, by name, so an ablation can iterate them and a result can
#: say which one it used.
STRATEGIES: dict[str, Strategy] = {
    "mean": mean,                       # v1's behaviour, the incumbent
    "median": median,
    "duration_weighted": duration_weighted,
    "trimmed": trimmed,
    "p90": p90,
    "p10": p10,
    "last_window": last_window,
    "max_run2": max_run,
}

DEFAULT = "mean"


def apply(name: str, windows: Sequence[dict], field: str) -> float:
    """Aggregate `field` across window dicts using the named strategy.

    Window dicts are the shape `prosody.Window.to_dict()` emits, so this reads
    straight from the raw records without rehydrating anything.
    """
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; have {sorted(STRATEGIES)}")
    values = [w[field] for w in windows if field in w]
    durations = [max(0.0, w.get("t1", 0) - w.get("t0", 0)) for w in windows
                 if field in w]
    return round(STRATEGIES[name](values, durations), 4)


def all_strategies(windows: Sequence[dict], field: str) -> dict[str, float]:
    """Every aggregation of one field, for the ablation table."""
    return {name: apply(name, windows, field) for name in STRATEGIES}
