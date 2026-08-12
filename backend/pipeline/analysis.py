"""Test the central claim: does driver stress track lap-time loss?

A single Pearson r over every message pooled together is the weakest possible
version of this question, and also the hardest to explain to anyone. Three
problems with it here:

  - Drivers have different baselines. Pooling a calm midfielder with a title
    contender mixes between-driver variation into a within-driver question.
  - Lap-time delta is dominated by traffic, safety cars, fuel load and pit
    stops - effects far larger than mood.
  - Radio is emitted irregularly, so messages cluster around incidents.

So we report three views and let them disagree if they disagree:

  1. pooled r          - the naive number, for completeness
  2. per-driver r      - the same question asked within each driver
  3. tercile contrast  - the interpretable one: when a driver sounded most
                         stressed, how much slower were those laps than when he
                         sounded calmest?

The tercile contrast is the figure worth putting on a slide, because it is
stated in seconds rather than in correlation units.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from pipeline import stats


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    try:
        return round(statistics.correlation(xs, ys), 4)
    except statistics.StatisticsError:
        return None


def analyse(messages: list[dict], min_per_driver: int = 6) -> dict:
    """Messages must carry `dsi`, `driver_code` and a joined `lap` dict.

    Only *representative* laps are used - green-flag laps with no pit entry or
    exit. Including them made the first run of this analysis meaningless: one
    driver's "stressed" tercile averaged +18.8s, which was a pit stop, not a
    mood. A 20-second stop and a 30-second safety-car lap are an order of
    magnitude larger than any plausible effect of driver state.
    """
    excluded = sum(
        1 for m in messages
        if m["lap"].get("in_race") and not m["lap"].get("is_representative")
    )
    pairs = [
        (m["dsi"], m["lap"]["delta_to_median_s"], m["driver_code"])
        for m in messages
        if m["lap"].get("in_race")
        and m["lap"].get("is_representative")
        and m["lap"].get("delta_to_median_s") is not None
    ]

    if len(pairs) < 3:
        return {"n": len(pairs), "excluded_non_racing_laps": excluded,
                "pooled_r": None, "per_driver": [], "tercile": None}

    dsis = [p[0] for p in pairs]
    deltas = [p[1] for p in pairs]

    # 1. pooled
    pooled = _pearson(dsis, deltas)

    # 2. per driver
    by_driver: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for d, delta, code in pairs:
        by_driver[code].append((d, delta))

    per_driver = []
    for code, vals in sorted(by_driver.items()):
        if len(vals) < min_per_driver:
            continue
        r = _pearson([v[0] for v in vals], [v[1] for v in vals])
        if r is not None:
            per_driver.append({"driver": code, "n": len(vals), "r": r})
    per_driver.sort(key=lambda d: -abs(d["r"]))

    # 3. tercile contrast, computed within driver then averaged, so a driver who
    #    simply talks more cannot dominate the result.
    contrasts = []
    for code, vals in by_driver.items():
        if len(vals) < 6:
            continue
        ordered = sorted(vals, key=lambda v: v[0])
        k = max(1, len(ordered) // 3)
        calm = [v[1] for v in ordered[:k]]
        tense = [v[1] for v in ordered[-k:]]
        contrasts.append({
            "driver": code,
            "n": len(vals),
            "calm_mean_delta_s": round(statistics.fmean(calm), 3),
            "stressed_mean_delta_s": round(statistics.fmean(tense), 3),
            "gap_s": round(statistics.fmean(tense) - statistics.fmean(calm), 3),
        })
    contrasts.sort(key=lambda c: -c["gap_s"])

    tercile = None
    if contrasts:
        gaps = [c["gap_s"] for c in contrasts]
        positive = sum(1 for g in gaps if g > 0)
        tercile = {
            "drivers": contrasts,
            "mean_gap_s": round(statistics.fmean(gaps), 3),
            "drivers_slower_when_stressed": positive,
            "drivers_total": len(gaps),
            # "More drivers than not were slower when stressed" is only worth
            # saying if it beats a coin flip. Sign test against p=0.5.
            "sign_test_p": _sign_test_p(positive, len(gaps)),
        }

    return {
        "n": len(pairs),
        "excluded_non_racing_laps": excluded,
        "pooled_r": pooled,
        "per_driver": per_driver,
        "tercile": tercile,
        "lag": lag_analysis(messages),
    }


def lag_analysis(messages: list[dict], max_lag: int = 3) -> dict:
    """Does stress *precede* a drop in pace, or merely accompany it?

    This is the question that matters for the hackathon's actual theme. A
    correlation at lag 0 says "these co-occur", which is interesting but not
    actionable - by the time you see it, the lap is already lost. A relationship
    at lag +1 or +2 says the radio call carries information about laps that have
    not happened yet, which is the difference between a dashboard and a decision.

    For each radio message on lap L we compare that driver's pace on laps L+1..L+k
    against their own race median, using representative laps only.

    Reported whichever way it comes out. With one race the sample is small and a
    null result is the expected outcome; the corpus exists to make this question
    answerable at all.
    """
    # Lap time by (driver, lap), representative laps only.
    pace: dict[tuple[str, int], float] = {}
    for m in messages:
        lap = m.get("lap") or {}
        if lap.get("in_race") and lap.get("is_representative") and lap.get("delta_to_median_s") is not None:
            pace[(m["driver_code"], lap["lap_number"])] = lap["delta_to_median_s"]

    out = {}
    for k in range(0, max_lag + 1):
        xs, ys = [], []
        for m in messages:
            lap = m.get("lap") or {}
            if not lap.get("in_race") or lap.get("lap_number") is None:
                continue
            if m.get("speaker") == "engineer":
                continue  # not a claim about the driver
            future = pace.get((m["driver_code"], lap["lap_number"] + k))
            if future is None:
                continue
            xs.append(m["dsi"])
            ys.append(future)
        out[f"lag_{k}"] = {"n": len(xs), "r": _pearson(xs, ys)}

    # Guard against the obvious trap. Testing four lags and reporting the largest
    # |r| will "find" a relationship in noise. This gate has had to be tightened
    # twice, because each version still let a spurious result through:
    #
    #   v1  no minimum sample -> selected r=0.62 at lag 2 from n=10 on one race.
    #   v2  minimum n plus an effect-size margin -> on the pooled corpus that
    #       still passed r=-0.25 at lag 3 from n=56, which is p=0.06: not
    #       significant, and not significant by a wide margin once you account
    #       for having tested four lags.
    #
    # So the gate now requires actual significance, Bonferroni-corrected for the
    # number of lags tested, and the reported direction is explicit - a negative
    # r means higher stress preceded *faster* laps, which is the opposite of the
    # hypothesis and must never be reported as "predictive".
    MIN_N = 30
    n_tests = len(out)
    alpha = 0.05 / n_tests

    for v in out.values():
        v["p"] = _pearson_p(v["r"], v["n"])

    scored = [(k, v) for k, v in out.items() if v["r"] is not None and v["n"] >= MIN_N]
    best = max(scored, key=lambda kv: abs(kv[1]["r"]), default=None)

    underpowered = [k for k, v in out.items() if v["n"] < MIN_N]
    lag0 = out["lag_0"]["r"]

    predictive = bool(
        best
        and best[0] != "lag_0"
        and lag0 is not None
        and abs(best[1]["r"]) > abs(lag0) + 0.15
        and best[1]["p"] is not None
        and best[1]["p"] < alpha
        and best[1]["r"] > 0  # only a positive r means stress precedes pace LOSS
    )

    return {
        "by_lag": out,
        "min_n_for_consideration": MIN_N,
        "bonferroni_alpha": round(alpha, 4),
        "underpowered_lags": underpowered,
        "best_lag": best[0] if best else None,
        "best_r": best[1]["r"] if best else None,
        "best_p": best[1]["p"] if best else None,
        "predictive": predictive,
        "caveat": (
            f"{n_tests} lags are tested and the largest |r| is reported, which is a "
            f"multiple comparison, so significance is judged at p < {alpha:.4f} "
            f"(Bonferroni). Lags below n={MIN_N} are excluded from selection. A "
            "negative r means higher stress preceded faster laps - the opposite of "
            "the hypothesis - and never counts as predictive."
        ),
    }


# Both now live in pipeline/stats.py, so the eval scripts can use the same ones.
# `_pearson_p` previously computed a t-statistic, discarded it with `del t`, and
# returned a Fisher z-transform approximation instead. The approximation is good
# - within 1.5% of the exact test at n=12 and indistinguishable above n=100, so
# no published p-value was wrong - but it was gating a published significance
# claim by a route its own comment called a substitute. It is now the exact t.
_sign_test_p = stats.sign_test_p
_pearson_p = stats.pearson_p
