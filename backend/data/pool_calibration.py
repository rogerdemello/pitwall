"""Fit one calibration across every race, so DSI means something between races.

Per-race percentile calibration has a flaw that only becomes visible once a
second race exists: it centres *every* race on exactly 50.0 by construction. A
soaking-wet scramble and a processional dry afternoon produce identical
distributions, and "DSI 80" silently means "top fifth of this race" rather than
anything absolute.

Fitting the percentile mapping once over the pooled corpus fixes that. DSI then
means "top fifth of F1 radio *in general*", races become comparable, and the
wet-versus-dry contrast has somewhere to show up.

Run after build_all.py, then re-run calibrate.py for each race so the stored
race files use the pooled mapping.

Usage:
    python backend/data/pool_calibration.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.calibration import Calibrator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(HERE, "..", "raw")
OUT = os.path.join(HERE, "..", "races", "_pooled.calibration.json")


def build() -> str:
    records, races = [], []
    for path in sorted(glob.glob(os.path.join(RAW_ROOT, "*.raw.json"))):
        if ".bak." in os.path.basename(path):
            continue
        data = json.load(open(path, encoding="utf-8"))
        good = [m for m in data["messages"] if "error" not in m and m.get("arousal") is not None]
        if not good:
            continue
        records.extend(good)
        races.append((data["race_id"], len(good)))

    if not records:
        print("no raw races found - run build_all.py first")
        return ""

    cal = Calibrator.fit(records)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cal.to_json(OUT)

    print(f"pooled calibration over {len(records)} messages from {len(races)} races:")
    for rid, n in races:
        print(f"  {rid}: {n}")
    s = cal.summary()
    for dim in ("arousal", "valence", "dominance"):
        d = s[dim]
        print(f"  {dim:<10} min {d['min']}  p10 {d['p10']}  median {d['median']}  "
              f"p90 {d['p90']}  max {d['max']}")
    print(f"wrote {OUT}")
    return OUT


if __name__ == "__main__":
    build()
