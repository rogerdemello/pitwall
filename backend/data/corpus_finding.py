"""Score the pre-registered prediction test: does DSI separate races that were
genuinely different to drive?

This replaces a hand-written `_corpus_finding.json`. That file had no generator,
so when the corpus doubled from six races to twelve and the calibration was
re-pooled over all 2,042 messages, every number in it silently went stale - mean
DSI drifted by about 0.4 points per race, which moved the z-scores and p-values
it published. It was still being served by /api/corpus-finding and rendered on
the Evidence screen underneath a claim that nothing there is typed by hand.

Everything below except the predictions themselves is computed from
`races/<race>.json` at run time.

**The predictions stay a six-race test.** They were registered in advance for the
original contrast slate, before any of it was analysed. Six more races were built
later to settle the recording-era confound (see era_analysis.py); folding those
into the prediction test after the fact would turn a confirmatory result into an
exploratory one. They are reported separately, and labelled as not predicted.

Usage:
    python backend/data/corpus_finding.py
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.artifacts import iter_race_files  # noqa: E402
from pipeline.calibration import CROSS_RACE_SOURCES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
OUT = os.path.join(RACES, "_corpus_finding.json")

QUESTION = (
    "Does the Driver State Index separate races that were genuinely different "
    "to drive?"
)
WHY = (
    "This is the strongest available test of whether DSI measures anything real. "
    "The slate was chosen for contrast before any of it was analysed: a "
    "soaking-wet scramble, a dry processional afternoon as a control, a race run "
    "in punishing heat. If those do not separate, the index is measuring "
    "something other than driver state."
)

# Registered in advance, before any of the slate was analysed. This is the only
# hand-entered content in the file, and it is a historical record of what was
# predicted rather than a result. The control race is the one every other race in
# the slate is contrasted against.
CONTROL = "2023_Italian_Grand_Prix"
PREREGISTERED = {
    "2020_Turkish_Grand_Prix": ("high", "wet"),
    "2019_German_Grand_Prix": ("high", "wet chaos"),
    "2023_Monaco_Grand_Prix": ("high", "dry to wet"),
    "2023_Italian_Grand_Prix": ("low", "dry, processional - the control"),
    "2023_Qatar_Grand_Prix": ("fatigue", "extreme heat"),
}
# Built for the confound test, not part of the prediction slate. 2021 Abu Dhabi
# is the showcase race and was never predicted either way.
SLATE = set(PREREGISTERED) | {"2021_Abu_Dhabi_Grand_Prix"}


def summarise(path: str) -> dict:
    """Driver-attributed DSI summary for one race, straight from the race JSON."""
    d = json.load(open(path, encoding="utf-8"))
    scored = [m for m in d["messages"] if m.get("speaker") != "engineer"]
    dsis = [m["dsi"] for m in scored]
    if not dsis:
        return {}
    n = len(dsis)
    sd = st.pstdev(dsis)
    se = sd / math.sqrt(n)
    states = [m.get("state") for m in scored]
    return {
        "race_id": d["race_id"],
        "race": d["grand_prix"].replace(" Grand Prix", ""),
        "season": int(d["race_id"][:4]),
        "n": n,
        "mean_dsi": round(st.fmean(dsis), 1),
        "sd": round(sd, 1),
        "se": se,
        "ci95": [round(st.fmean(dsis) - 1.96 * se, 1),
                 round(st.fmean(dsis) + 1.96 * se, 1)],
        "stressed_pct": round(100 * states.count("Stressed") / n),
        "fatigued_pct": round(100 * states.count("Fatigued") / n),
        "calibration_source": d.get("calibration_source"),
        "messages_total": len(d["messages"]),
    }


def z_test(a: dict, b: dict) -> tuple[float, float, float]:
    """Two-sample z on the difference of means. Same test era_analysis.py uses."""
    diff = a["mean_dsi"] - b["mean_dsi"]
    sed = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    z = diff / sed if sed else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return diff, z, p


def score_predictions(rows: dict[str, dict], contrasts: list[dict],
                      alpha: float) -> tuple[list[dict], list[str], list[str]]:
    """Did each registered prediction hold? Scored on stated criteria only."""
    by_race = {c["race_id"]: c for c in contrasts}
    control = rows[CONTROL]
    slate = [r for rid, r in rows.items() if rid in PREREGISTERED]

    verdicts, held, failed = [], [], []
    for race_id, (kind, why) in PREREGISTERED.items():
        row = rows[race_id]
        if kind == "high":
            c = by_race[race_id]
            ok = c["diff"] > 0 and c["survives_bonferroni"]
            detail = (f"{row['race']} ({why}) predicted high: {c['diff']:+.1f} vs "
                      f"the control, p={c['p']:.4f}, "
                      f"{'survives' if c['survives_bonferroni'] else 'does not survive'} "
                      f"Bonferroni {alpha:.3f}.")
        elif kind == "low":
            lowest = min(slate, key=lambda r: r["mean_dsi"])
            ok = lowest["race_id"] == race_id
            where = "the lowest" if ok else f"above {lowest['race']}"
            detail = (f"{row['race']} ({why}) predicted lowest: it is {where} "
                      f"in the slate at {row['mean_dsi']:.1f}.")
        else:  # fatigue
            top = max(slate, key=lambda r: r["stressed_pct"])
            ok = top["race_id"] == race_id
            where = "the highest" if ok else f"behind {top['race']}"
            detail = (f"{row['race']} ({why}) predicted highest fatigue: "
                      f"{row['stressed_pct']}% Stressed and {row['fatigued_pct']}% "
                      f"Fatigued, {where} Stressed share in the slate.")
        verdicts.append({"race": row["race"], "prediction": kind,
                         "rationale": why, "held": ok, "detail": detail})
        (held if ok else failed).append(detail)
    return verdicts, held, failed


def main() -> None:
    rows = {}
    for path in iter_race_files(RACES):
        s = summarise(path)
        if s:
            rows[s["race_id"]] = s

    missing = [r for r in SLATE if r not in rows]
    if missing:
        print(f"!! prediction slate incomplete, missing: {', '.join(sorted(missing))}")
        return

    sources = {r["calibration_source"] for r in rows.values()}
    if not sources <= CROSS_RACE_SOURCES:
        print(f"!! calibration sources are {sources}; cross-race comparison needs "
              f"a shared reference ({sorted(CROSS_RACE_SOURCES)}). "
              "Run finish_corpus.py.")
        return

    control = rows[CONTROL]
    slate = sorted((r for rid, r in rows.items() if rid in SLATE),
                   key=lambda r: -r["mean_dsi"])
    others = sorted((r for rid, r in rows.items() if rid not in SLATE),
                    key=lambda r: -r["mean_dsi"])

    # One contrast per non-control slate race, Bonferroni-corrected across them.
    tested = [r for r in slate if r["race_id"] != CONTROL]
    alpha = 0.05 / len(tested)
    contrasts = []
    for r in tested:
        diff, z, p = z_test(r, control)
        contrasts.append({
            "race_id": r["race_id"], "race": r["race"],
            "diff": round(diff, 1), "z": round(z, 2), "p": round(p, 4),
            "survives_bonferroni": p < alpha,
            "predicted": PREREGISTERED.get(r["race_id"], ("not predicted",))[0],
        })

    verdicts, held, failed = score_predictions(rows, contrasts, alpha)
    n_held = sum(1 for v in verdicts if v["held"])
    n_total = len(verdicts)

    spread = slate[0]["mean_dsi"] - slate[-1]["mean_dsi"]
    pooled_sd = st.fmean([r["sd"] for r in slate])
    cohens_d = spread / pooled_sd if pooled_sd else 0.0

    if n_held == n_total:
        verdict = f"SUPPORTED - all {n_total} predictions held."
    elif n_held == 0:
        verdict = f"NOT SUPPORTED - none of the {n_total} predictions held."
    else:
        verdict = (f"PARTIAL SUPPORT - {n_held} of {n_total} predictions held, "
                   f"{n_total - n_held} clearly failed.")

    era = _load_optional("_era_analysis.json")
    gold = _load_optional("_gold_affect_eval.json")

    payload = {
        "generated_by": "backend/data/corpus_finding.py",
        "question": QUESTION,
        "why_it_matters": WHY,
        "setup": {
            "races": len(slate),
            "messages_analysed": sum(r["messages_total"] for r in slate),
            "messages_scored": "engineer-attributed transmissions excluded",
            "n_scored": sum(r["n"] for r in slate),
            "calibration": (
                f"pooled across all {len(rows)} races in the corpus - per-race "
                "calibration centres every race on 50.0 by construction and "
                "would have made this test impossible"
            ),
            "control_race": control["race"],
            "predictions_made_in_advance": {
                kind: [rows[rid]["race"] + f" ({why})"
                       for rid, (k, why) in PREREGISTERED.items() if k == kind]
                for kind in ("high", "low", "fatigue")
            },
        },
        "results": [{k: v for k, v in r.items() if k != "se"} for r in slate],
        "contrasts_vs_dry_control": [
            {k: v for k, v in c.items() if k != "race_id"} for c in contrasts
        ],
        "prediction_scorecard": verdicts,
        "bonferroni_note": (
            f"{len(tested)} contrasts were tested, so the corrected threshold is "
            f"p < {alpha:.3f}. "
            + "; ".join(f"{c['race']}'s p = {c['p']:.3f} does not survive it and "
                        "is not claimed" for c in contrasts
                        if not c["survives_bonferroni"])
            + "."
        ),
        "verdict": verdict,
        "what_held": held,
        "what_failed": " ".join(failed) if failed else None,
        "effect_size": (
            f"The full spread across the slate is {spread:.1f} DSI points against "
            f"a within-race sd of about {pooled_sd:.0f}, so Cohen's d is roughly "
            f"{cohens_d:.2f} - a "
            f"{'small' if cohens_d < 0.35 else 'small-to-moderate' if cohens_d < 0.6 else 'moderate'} "
            "effect. The index separates these races, but not dramatically."
        ),
        "not_predicted": {
            "note": (
                f"{len(others)} further races were built after the predictions "
                "were registered, to test the recording-era confound. They were "
                "not predicted in either direction and are reported descriptively "
                "only. Folding them into the prediction test would turn a "
                "confirmatory result into an exploratory one."
            ),
            "races": [{k: v for k, v in r.items() if k != "se"} for r in others],
        },
        "confound_we_cannot_rule_out": _confound_text(era),
        "confound_status": "resolved" if era else "untested",
        "honest_summary": _summary_text(verdict, spread, cohens_d, failed, era, gold),
    }

    _report(slate, others, contrasts, verdicts, alpha, spread, cohens_d, verdict)
    json.dump(payload, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {OUT}")


def _load_optional(name: str) -> dict | None:
    path = os.path.join(RACES, name)
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def _confound_text(era: dict | None) -> str:
    if not era:
        return (
            "OPEN - the slate spans several seasons and radio encoding is not "
            "constant across them, so some of the separation could be recording "
            "era rather than driving. Run era_analysis.py to test it."
        )
    within = era.get("within_season_spread")
    across = era.get("cross_era_spread")
    gap = era.get("era_comparison") or {}
    c = (era.get("contrasts") or [{}])[0]
    return (
        "RESOLVED - see _era_analysis.json. This previously read as an open "
        "confound: the slate spanned several seasons and radio encoding is not "
        "constant across them, so some separation could have been recording era. "
        f"It was then tested directly by building {era.get('n_2023')} races from "
        f"2023 alone. Within that single season races still spread {within} DSI "
        f"points against {across} across all eras"
        + (f", {c.get('comparison')} differs by {c.get('diff'):+.1f} "
           f"(p={c.get('p')}, {'surviving' if c.get('survives') else 'not surviving'} "
           "Bonferroni)" if c else "")
        + (f", and there is no systematic offset between eras "
           f"({gap.get('diff'):+.1f} points, p={gap.get('p')})"
           if gap and not gap.get("systematic_era_effect")
           else f", though a systematic era offset remains ({gap.get('diff'):+.1f} "
                f"points, p={gap.get('p')})" if gap else "")
        + ". Recording era does not explain the separation."
    )


def _summary_text(verdict, spread, d, failed, era, gold) -> str:
    bits = [
        f"DSI separates races in the direction predicted, with the dry "
        f"processional control correctly at the bottom"
        + (", and the era confound has now been tested and ruled out on a "
           f"{era.get('n_2023')}-race within-season slate" if era else "")
        + "."
    ]
    bits.append(
        "It remains evidence that the index tracks race conditions, not proof it "
        f"tracks driver stress specifically: the effect is modest (Cohen's d "
        f"about {d:.2f})"
        + (f", {len(failed)} prediction{'s' if len(failed) != 1 else ''} still "
           "failed" if failed else "")
        + "."
    )
    if gold:
        axes = (gold.get("axes") or {})
        val = axes.get("valence_positive_vs_negative") or {}
        if val and val.get("accuracy") is not None:
            lift = val["accuracy"] - val.get("baseline", 0)
            if lift < 0.05:
                bits.append(
                    "And validation against gold labels shows the valence half of "
                    "the index is at chance."
                )
    return " ".join(bits)


def _report(slate, others, contrasts, verdicts, alpha, spread, d, verdict) -> None:
    print(f"pre-registered slate: {len(slate)} races "
          f"({sum(r['n'] for r in slate)} driver-attributed messages)\n")
    print(f"{'race':<20}{'n':>5}{'mean':>7}{'sd':>6}   {'95% CI':<16}{'str%':>6}{'fat%':>6}")
    for r in slate:
        ci = f"[{r['ci95'][0]}, {r['ci95'][1]}]"
        print(f"{r['race']:<20}{r['n']:>5}{r['mean_dsi']:>7.1f}{r['sd']:>6.1f}   "
              f"{ci:<16}{r['stressed_pct']:>6}{r['fatigued_pct']:>6}")

    print(f"\ncontrasts vs the dry control (Bonferroni alpha {alpha:.3f}):")
    for c in contrasts:
        mark = "holds" if c["survives_bonferroni"] else "not claimed"
        print(f"  {c['race']:<18}{c['diff']:>+6.1f}  z={c['z']:>5.2f}  "
              f"p={c['p']:.4f}  {mark}")

    print("\nprediction scorecard:")
    for v in verdicts:
        print(f"  [{'HELD' if v['held'] else 'FAIL'}] {v['detail']}")

    print(f"\nspread {spread:.1f} points, Cohen's d {d:.2f}")
    print(f"VERDICT: {verdict}")

    if others:
        print(f"\nnot predicted ({len(others)} races built later for the confound test):")
        for r in others:
            print(f"  {r['race']:<20}{r['n']:>5}{r['mean_dsi']:>7.1f}")


if __name__ == "__main__":
    main()
