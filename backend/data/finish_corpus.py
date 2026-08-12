"""Post-build step: pool the calibration, then re-apply it to every race.

Must run after build_all.py, because pooling needs all the raw outputs to exist.
Splitting it out means the overnight job can complete unattended:

    build_all.py  ->  finish_corpus.py

Re-running calibrate.py per race is cheap (no model inference - stage 2 only
reads cached raw outputs and the FastF1 cache), so applying the pooled mapping
costs about a minute a race rather than an hour.

Ends by printing the contrast table, which is the point of the whole corpus: if
a soaking-wet race does not separate from a processional dry one, the index is
not measuring what we claim - and then regenerating the evidence artifacts that
depend on the calibration, so they cannot drift away from it. `_corpus_finding`
went stale exactly this way once: the pool grew from six races to twelve, every
mean DSI moved by about 0.4 points, and the published file still carried the old
numbers.

Usage:
    python backend/data/finish_corpus.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import calibrate, corpus_finding, era_analysis, pool_calibration  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(HERE, "..", "raw")
RACES_ROOT = os.path.join(HERE, "..", "races")


def built_races() -> list[str]:
    out = []
    for path in sorted(glob.glob(os.path.join(RAW_ROOT, "*.raw.json"))):
        base = os.path.basename(path)
        out.append(base[: -len(".raw.json")])
    return out


def main() -> None:
    races = built_races()
    if not races:
        print("no raw races found - run build_all.py first")
        return

    print(f"pooling calibration over {len(races)} races ...")
    if not pool_calibration.build():
        return

    print("\nre-applying the pooled mapping to each race ...")
    for race in races:
        try:
            calibrate.run(race)
        except Exception:
            print(f"!! {race} failed:\n{traceback.format_exc()}")

    # The contrast check.
    print(f"\n{'=' * 66}\nCROSS-RACE CONTRAST (pooled calibration)\n{'=' * 66}")
    rows = []
    for race in races:
        path = os.path.join(RACES_ROOT, f"{race}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path, encoding="utf-8"))
        msgs = [m for m in d["messages"] if m.get("speaker") != "engineer"]
        if not msgs:
            continue
        dsis = [m["dsi"] for m in msgs]
        states = {}
        for m in msgs:
            states[m["state"]] = states.get(m["state"], 0) + 1
        rows.append({
            "race": d["grand_prix"].replace(" Grand Prix", ""),
            "n": len(msgs),
            "mean": sum(dsis) / len(dsis),
            "peak": max(dsis),
            "stressed": states.get("Stressed", 0) / len(msgs),
            "fatigued": states.get("Fatigued", 0) / len(msgs),
            "source": d.get("calibration_source"),
        })

    rows.sort(key=lambda r: -r["mean"])
    print(f"{'race':<26}{'n':>5}{'mean DSI':>10}{'peak':>7}{'stressed':>10}{'fatigued':>10}")
    for r in rows:
        print(f"{r['race']:<26}{r['n']:>5}{r['mean']:>10.1f}{r['peak']:>7}"
              f"{r['stressed'] * 100:>9.0f}%{r['fatigued'] * 100:>9.0f}%")

    sources = {r["source"] for r in rows}
    if not sources <= {"pooled", "leave-one-race-out"}:
        print(f"\n!! calibration sources are {sources} - expected a shared cross-race calibration")
    elif len(rows) > 1:
        spread = rows[0]["mean"] - rows[-1]["mean"]
        print(f"\nspread between highest and lowest mean DSI: {spread:.1f} points")
        if spread < 3:
            print("That is a flat result: the index is NOT separating these races.")
            print("Report it as such rather than reaching for an explanation.")
        else:
            print("The index does separate the races. Check the ordering matches")
            print("expectation (wet/chaotic high, dry processional low) before claiming it.")

    # Every artifact below is a function of the calibration that was just
    # re-pooled, so regenerate them here rather than trusting anyone to remember.
    print(f"\n{'=' * 66}\nREGENERATING CALIBRATION-DEPENDENT EVIDENCE\n{'=' * 66}")
    for name, run in (("era_analysis", era_analysis.main),
                      ("corpus_finding", corpus_finding.main)):
        print(f"\n--- {name} ---")
        try:
            run()
        except Exception:
            print(f"!! {name} failed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
