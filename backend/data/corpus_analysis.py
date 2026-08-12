"""Run the central analyses over the whole corpus rather than one race.

Everything on the Evidence screen so far is per-race, and on a single race the
numbers that matter are underpowered: the stress-vs-pace question had 88 paired
observations across 6 drivers, and the lag question had 8-14 per lag, which is
too few to say anything and was correctly refused.

Pooling six races changes that. The lag question in particular only becomes
answerable at this scale - and it is the one that matters for the hackathon's
theme, because a relationship at lag 0 says "these co-occur" while a relationship
at lag +1 says the radio call carries information about laps that have not
happened yet.

Writes races/_corpus_analysis.json. Reported whichever way it comes out.

Usage:
    python backend/data/corpus_analysis.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import analysis  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
OUT = os.path.join(RACES, "_corpus_analysis.json")


def load_all() -> tuple[list[dict], list[str]]:
    msgs, names = [], []
    for path in sorted(glob.glob(os.path.join(RACES, "*.json"))):
        base = os.path.basename(path)
        if base.startswith("_") or base.count(".") > 1:
            continue
        d = json.load(open(path, encoding="utf-8"))
        names.append(d["race_id"])
        for m in d["messages"]:
            # Namespace the driver so the same driver in two races is not treated
            # as one continuous history - laps restart at 1 every race.
            msgs.append({**m, "driver_code": f"{d['race_id']}|{m['driver_code']}"})
    return msgs, names


def main() -> None:
    msgs, races = load_all()
    if not msgs:
        print("no races built")
        return

    print(f"pooling {len(msgs)} messages from {len(races)} races")
    res = analysis.analyse(msgs, min_per_driver=8)
    lag = res["lag"]

    print(f"\nstress vs pace, representative laps only")
    print(f"  paired observations: {res['n']}  (excluded {res['excluded_non_racing_laps']} non-racing laps)")
    print(f"  pooled r: {res['pooled_r']}")
    t = res["tercile"]
    if t:
        print(f"  within-driver gap: {t['mean_gap_s']:+.3f}s "
              f"({t['drivers_slower_when_stressed']}/{t['drivers_total']} slower when stressed)")

    print(f"\nlag analysis (does stress PRECEDE a drop in pace?)")
    for k, v in lag["by_lag"].items():
        mark = "" if v["n"] >= lag["min_n_for_consideration"] else "  (underpowered)"
        print(f"  {k:<6} n={v['n']:<5} r={v['r']}{mark}")
    print(f"  best lag: {lag['best_lag']}  r={lag['best_r']}")
    print(f"  predictive: {lag['predictive']}")

    verdict = _verdict(res, lag)
    print(f"\nVERDICT: {verdict}")

    out = {
        "races": races,
        "messages_pooled": len(msgs),
        "stress_vs_pace": {k: v for k, v in res.items() if k != "lag"},
        "lag": lag,
        "verdict": verdict,
        "caveat": (
            "Driver histories are namespaced per race, so no comparison spans two "
            "races. Lap-time delta is measured against each driver's own median "
            "within that race. The era confound noted in _corpus_finding.json "
            "applies here too."
        ),
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"wrote {OUT}")


def _verdict(res: dict, lag: dict) -> str:
    r = res["pooled_r"]
    t = res["tercile"]
    powered = [k for k, v in lag["by_lag"].items()
               if v["n"] >= lag["min_n_for_consideration"] and k != "lag_0"]

    bits = []
    if r is None or abs(r) < 0.1:
        bits.append(f"No pooled linear relationship between stress and lap-time loss (r={r}).")
    else:
        bits.append(f"Pooled r={r} between stress and lap-time loss.")

    if t:
        share = f"{t['drivers_slower_when_stressed']}/{t['drivers_total']}"
        p = t.get("sign_test_p")
        sig = (
            f"no better than chance (sign test p={p})"
            if p is not None and p >= 0.05
            else f"more than chance would give (sign test p={p})"
        )
        bits.append(
            f"Within drivers, the most-stressed calls sit {t['mean_gap_s']:+.2f}s off "
            f"the calmest ones, with {share} drivers slower when stressed - {sig}."
        )

    if not powered:
        bits.append("Every lag beyond 0 is still underpowered, so no predictive claim is made.")
    elif lag["predictive"]:
        bits.append(
            f"Stress at {lag['best_lag'].replace('lag_', 'lap +')} precedes slower laps "
            f"(r={lag['best_r']}, p={lag['best_p']}), which is a genuine lead relationship."
        )
    else:
        best_r, best_p = lag["best_r"], lag["best_p"]
        detail = ""
        if best_r is not None and best_r < 0:
            # Worth naming: the largest lag correlation points the wrong way.
            detail = (
                f" The largest lag correlation ({lag['best_lag']}, r={best_r}) is negative, "
                "meaning higher stress preceded *faster* laps - the opposite of the "
                f"hypothesis - and at p={best_p} it is not significant once the four "
                "lags tested are accounted for."
            )
        bits.append(
            "Stress does not appear to lead pace loss in this corpus." + detail
        )
    return " ".join(bits)


if __name__ == "__main__":
    main()
