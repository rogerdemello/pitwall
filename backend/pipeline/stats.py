"""Significance and uncertainty, in one place.

These were scattered across `analysis.py`, `eval_convergent.py` and
`eval_affect_gold.py`, which is how two of them ended up subtly wrong:

  `_pearson_p` computed a t-statistic, then `del t`-ed it and returned a
  Fisher z-transform approximation instead. The approximation is good - within
  1.5% of the exact test even at n=12, and indistinguishable above n=100 - so
  no published number was wrong. But it was gating a *published* significance
  claim by a route its own comment admitted was a substitute, and scipy was
  already installed transitively via scikit-learn.

  `cohens_kappa` lived inside an eval script, so nothing else could use it.

Every published figure should carry an interval. `bootstrap_ci` is here so that
adding one costs a line rather than a decision.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Callable, Sequence

from scipy import stats


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation. None when undefined rather than 0.0."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    num = sum(a * b for a, b in zip(dx, dy))
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    return round(num / den, 4) if den else None


def pearson_p(r: float | None, n: int) -> float | None:
    """Two-sided p-value for a correlation, exact under the t distribution."""
    if r is None or n < 4 or abs(r) >= 1.0:
        return None
    t = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    return round(float(2 * stats.t.sf(t, n - 2)), 4)


def fisher_z_ci(r: float | None, n: int, conf: float = 0.95
                ) -> tuple[float, float] | None:
    """Confidence interval for a correlation, via the Fisher z-transform.

    A correlation reported without one invites reading r=0.043 on n=1155 and
    r=0.62 on n=10 as comparable claims. They are not, and the interval is what
    makes that visible.
    """
    if r is None or n < 4 or abs(r) >= 1.0:
        return None
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1 / math.sqrt(n - 3)
    crit = float(stats.norm.ppf(1 - (1 - conf) / 2))
    lo, hi = z - crit * se, z + crit * se
    return round(math.tanh(lo), 4), round(math.tanh(hi), 4)


def sign_test_p(successes: int, n: int) -> float | None:
    """Two-sided exact binomial test against p=0.5.

    Guards the "most drivers were slower when stressed" style of claim, which
    sounds compelling and is often just a coin flip: 23 of 37 gives p=0.19.
    """
    if n == 0:
        return None
    k = max(successes, n - successes)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return round(min(1.0, 2 * tail), 4)


def bootstrap_ci(values: Sequence, statistic: Callable[[Sequence], float],
                 conf: float = 0.95, resamples: int = 1000,
                 seed: int = 11) -> tuple[float, float] | None:
    """Percentile bootstrap interval for any statistic.

    Seeded, because an interval that moves between runs of the same script is
    not evidence. Resample at the unit the claim is about - clips for WER, not
    words, or the interval comes out far too narrow.
    """
    values = list(values)
    n = len(values)
    if n < 2:
        return None
    rng = random.Random(seed)
    stats_out = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        try:
            stats_out.append(statistic(sample))
        except (ZeroDivisionError, ValueError):
            continue
    if len(stats_out) < resamples // 2:
        return None
    stats_out.sort()
    lo = stats_out[int(len(stats_out) * (1 - conf) / 2)]
    hi = stats_out[min(len(stats_out) - 1, int(len(stats_out) * (1 - (1 - conf) / 2)))]
    return round(lo, 4), round(hi, 4)


def proportion_ci(successes: int, n: int, conf: float = 0.95
                  ) -> tuple[float, float] | None:
    """Wilson interval for a proportion.

    Wilson rather than normal-approximation because the proportions here are
    often near 0 or 1 on small n - jargon recall of 9/10, say - where the normal
    interval runs outside [0, 1] and stops meaning anything.
    """
    if n == 0:
        return None
    z = float(stats.norm.ppf(1 - (1 - conf) / 2))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def cohens_kappa(pairs: Sequence[tuple[str, str]],
                 classes: Sequence[str]) -> float | None:
    """Agreement corrected for what chance alone would produce.

    Raw agreement is misleading whenever both raters over-produce the same
    class, which is exactly the convergent-validity case: two models that each
    put 80% of clips in one bucket agree often and inform nothing.
    """
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum((ca[c] / n) * (cb[c] / n) for c in classes)
    if expected >= 1.0:
        return None
    return round((observed - expected) / (1 - expected), 4)


def kappa_band(k: float | None) -> str:
    """Landis & Koch labels, so a number is never reported without its reading."""
    if k is None:
        return "not computable"
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def bonferroni_alpha(n_tests: int, alpha: float = 0.05) -> float:
    """The corrected threshold, reported alongside every multiple comparison."""
    return alpha / max(1, n_tests)
