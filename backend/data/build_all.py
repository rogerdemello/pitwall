"""Build the whole race slate, resumably.

Wraps build_race.py -> calibrate.py for each race. Roughly an hour per race on
CPU, so this is an overnight job; every stage is resumable, so an interruption
costs at most the clip in flight.

Clips are expected to be on disk already (fetch_many.py pulls the whole slate in
one pass over the Hub, which is far cheaper than one scan per race).

Usage:
    python backend/data/build_all.py
    python backend/data/build_all.py 2020_Turkish_Grand_Prix
"""

from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import build_race, calibrate  # noqa: E402
from data.fetch_many import SLATE  # noqa: E402

CLIP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clips")


def main(races: list[str]) -> None:
    t0 = time.perf_counter()
    done, failed = [], []

    for i, race in enumerate(races, 1):
        if not os.path.exists(os.path.join(CLIP_ROOT, race, "manifest.json")):
            print(f"[{i}/{len(races)}] {race}: no clips on disk - run fetch_many.py first")
            failed.append(race)
            continue

        print(f"\n{'=' * 70}\n[{i}/{len(races)}] {race}\n{'=' * 70}")
        try:
            build_race.build(race)
            calibrate.run(race)
            done.append(race)
        except Exception:
            # One bad race must not cost the whole overnight run.
            print(f"!! {race} failed:\n{traceback.format_exc()}")
            failed.append(race)

        mins = (time.perf_counter() - t0) / 60
        print(f"-- elapsed {mins:.0f} min, {len(done)} built, {len(failed)} failed")

    print(f"\n{'=' * 70}")
    print(f"built:  {', '.join(done) or 'none'}")
    if failed:
        print(f"failed: {', '.join(failed)}")
    print(f"total {(time.perf_counter() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main(sys.argv[1:] or SLATE)
