"""Fit one calibration across every race, so DSI means something between races.

Per-race percentile calibration has a flaw that only becomes visible once a
second race exists: it centres *every* race on exactly 50.0 by construction. A
soaking-wet scramble and a processional dry afternoon produce identical
distributions, and "DSI 80" silently means "top fifth of this race" rather than
anything absolute.

Fitting the percentile mapping once over the pooled corpus fixes that. DSI then
means "top fifth of F1 radio *in general*", races become comparable, and the
wet-versus-dry contrast has somewhere to show up.

That fix introduced a subtler problem, which this file also now addresses. The
pooled mapping is fitted on all 2,042 messages and then applied to those same
2,042 messages, so every DSI this project has published is an in-sample score. A
message helps set the percentile scale it is then measured against. With one race
that is a rounding error; across a corpus it means "top fifth of F1 radio" is a
claim about the fitting set rather than about F1 radio.

So a **leave-one-race-out** mapping is emitted alongside the pooled one: for each
race, a calibrator fitted on the other eleven. calibrate.py prefers it, and each
fold still has ~1,900 reference messages, which is ample. Every DSI in the race
files then becomes an out-of-sample score.

The pooled mapping is kept, because the live-upload path has no race to hold out
- an uploaded clip is genuinely unseen, so pooled is the correct reference there.

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
RACES_ROOT = os.path.join(HERE, "..", "races")
OUT = os.path.join(RACES_ROOT, "_pooled.calibration.json")
LORO_DIR = os.path.join(RACES_ROOT, "_loro")

#: Below this many reference messages a percentile map is too coarse to trust,
#: and the fold falls back to pooled rather than producing a confident-looking
#: mapping off a handful of values.
MIN_FOLD_RECORDS = 500


def _by_race() -> list[tuple[str, list[dict]]]:
    out = []
    for path in sorted(glob.glob(os.path.join(RAW_ROOT, "*.raw.json"))):
        if ".bak." in os.path.basename(path):
            continue
        data = json.load(open(path, encoding="utf-8"))
        good = [m for m in data["messages"]
                if "error" not in m and m.get("arousal") is not None]
        if good:
            out.append((data["race_id"], good))
    return out


def loro_path(race_id: str) -> str:
    """Where the calibrator fitted on every race *except* this one lives."""
    return os.path.join(LORO_DIR, f"{race_id}.calibration.json")


def build() -> str:
    races = _by_race()
    if not races:
        print("no raw races found - run build_all.py first")
        return ""

    records = [m for _, msgs in races for m in msgs]
    cal = Calibrator.fit(records)
    os.makedirs(RACES_ROOT, exist_ok=True)
    cal.to_json(OUT)

    print(f"pooled calibration over {len(records)} messages from {len(races)} races:")
    for rid, msgs in races:
        print(f"  {rid}: {len(msgs)}")
    s = cal.summary()
    for dim in ("arousal", "valence", "dominance"):
        d = s[dim]
        print(f"  {dim:<10} min {d['min']}  p10 {d['p10']}  median {d['median']}  "
              f"p90 {d['p90']}  max {d['max']}")
    print(f"wrote {OUT}")

    # Leave-one-race-out: each race scored against a mapping it did not help fit.
    if len(races) < 3:
        print("\nfewer than 3 races - skipping leave-one-race-out")
        return OUT

    os.makedirs(LORO_DIR, exist_ok=True)
    print(f"\nleave-one-race-out mappings ({len(races)} folds):")
    for rid, msgs in races:
        others = [m for other_id, other in races if other_id != rid for m in other]
        if len(others) < MIN_FOLD_RECORDS:
            print(f"  {rid:<34} only {len(others)} held-out records - using pooled")
            continue
        Calibrator.fit(others).to_json(loro_path(rid))
        print(f"  {rid:<34} fitted on {len(others)} messages from "
              f"{len(races) - 1} other races")
    print(f"wrote {LORO_DIR}")
    return OUT


if __name__ == "__main__":
    build()
