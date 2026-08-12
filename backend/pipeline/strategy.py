"""Turn driver state into an actual pit-wall recommendation.

The brief's theme is *Racing Strategy & Decision-Making*, but a mood label on its
own is not a decision. This layer closes that gap: it combines how the driver
sounds with how the car is performing and how old the tyres are, and says what
to do about it.

Deliberately a transparent rule engine rather than a learned policy. A race
engineer will not act on an unexplained number, and we cannot honestly train a
strategy model on 168 messages. Every recommendation carries the evidence that
produced it, so a judge can audit the reasoning in one glance.

Signals combined:
  - DSI trend across a driver's recent messages (rising stress)
  - lap-time delta against that driver's own race median (losing pace)
  - tyre age (is a stop plausible anyway)
  - suppressed-stress flags (the quiet warning)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Thresholds, gathered for tuning.
DSI_HIGH = 65            # top third of the corpus
DSI_RISE = 12            # points of increase that count as a trend
PACE_LOSS_S = 0.35       # seconds/lap off own median that counts as degradation
TYRE_OLD = 18            # laps
TREND_WINDOW = 3         # messages


@dataclass
class Recommendation:
    severity: str        # info | watch | act
    headline: str
    detail: str
    evidence: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _trend(values: list[int]) -> int:
    """Change across the window: last value minus the earliest in window."""
    if len(values) < 2:
        return 0
    window = values[-TREND_WINDOW:]
    return window[-1] - window[0]


def recommend(driver_code: str, history: list[dict]) -> Recommendation | None:
    """Given a driver's analysed messages so far, decide what the pit wall should do.

    `history` is chronological; each entry needs `dsi`, `suppressed_stress` and a
    `lap` dict. Returns None when there is nothing worth saying - silence is a
    valid output and a wall of constant alerts is worse than none.
    """
    # Only reason about messages that plausibly carry the driver's own voice.
    # The channel also carries the engineer, and "you are doing a good job" said
    # by the pit wall is not evidence about the driver's state.
    scored = [
        h for h in history
        if h.get("lap", {}).get("in_race") and h.get("speaker") != "engineer"
    ]
    if not scored:
        return None

    latest = scored[-1]
    lap = latest["lap"]
    dsi_series = [h["dsi"] for h in scored]
    dsi = dsi_series[-1]
    rise = _trend(dsi_series)

    delta = lap.get("delta_to_median_s")
    tyre = lap.get("tyre_life")
    lap_no = lap.get("lap_number")
    compound = lap.get("compound")

    evidence = [f"DSI {dsi}" + (f" ({rise:+d} over last {min(len(dsi_series), TREND_WINDOW)} calls)" if rise else "")]
    if delta is not None:
        evidence.append(f"pace {delta:+.2f}s vs own median")
    if tyre is not None and compound:
        evidence.append(f"{compound.title()}, {tyre:.0f} laps old")
    if lap_no:
        evidence.append(f"lap {lap_no}")

    losing_pace = delta is not None and delta >= PACE_LOSS_S
    old_tyres = tyre is not None and tyre >= TYRE_OLD
    stressed = dsi >= DSI_HIGH
    rising = rise >= DSI_RISE

    # Strongest signal first.
    if stressed and losing_pace and old_tyres:
        return Recommendation(
            "act",
            f"{driver_code}: box this lap",
            "Driver stress is high, pace is dropping and the tyres are past their "
            "useful window. All three point the same way - bring him in.",
            evidence,
        )

    if stressed and losing_pace:
        return Recommendation(
            "act",
            f"{driver_code}: prepare the stop",
            "Stress and lap time are degrading together. Get the crew ready and "
            "take the next clear window.",
            evidence,
        )

    if latest.get("suppressed_stress"):
        return Recommendation(
            "watch",
            f"{driver_code}: check in with the driver",
            "He is telling you it is fine, but his voice does not agree. This is "
            "the pattern that gets missed - worth an explicit question.",
            evidence + ["words/voice mismatch"],
        )

    if rising and old_tyres:
        return Recommendation(
            "watch",
            f"{driver_code}: stress climbing on old tyres",
            "Not critical yet, but the trend and the tyre age are moving together. "
            "Start planning the window.",
            evidence,
        )

    if stressed:
        return Recommendation(
            "watch",
            f"{driver_code}: elevated stress",
            "Pace is holding, so no action needed - but keep the radio calm and "
            "avoid loading him with information.",
            evidence,
        )

    return None
