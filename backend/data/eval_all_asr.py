"""Run the ASR ablation for every built race, so no Evidence card is blank.

About 15 minutes per race on CPU. Skips races already measured, so it is safe to
re-run.

Usage:
    python backend/data/eval_all_asr.py
"""

from __future__ import annotations

import glob
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import eval_asr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
CLIPS = os.path.join(HERE, "..", "clips")


def main(sample: int = 40) -> None:
    races = []
    for path in sorted(glob.glob(os.path.join(RACES, "*.json"))):
        base = os.path.basename(path)
        if base.startswith("_") or base.count(".") > 1:
            continue
        races.append(base[:-5])

    for i, race in enumerate(races, 1):
        out = os.path.join(RACES, f"{race}.asr_eval.json")
        if os.path.exists(out):
            print(f"[{i}/{len(races)}] {race}: already measured, skipping")
            continue
        if not os.path.exists(os.path.join(CLIPS, race, "manifest.json")):
            print(f"[{i}/{len(races)}] {race}: no clips, skipping")
            continue
        print(f"\n[{i}/{len(races)}] {race}")
        try:
            eval_asr.run(race, sample)
        except Exception:
            print(f"!! {race} failed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
