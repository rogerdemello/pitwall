"""Hand-labelling tool for driver state, plus the evaluation that consumes it.

The dataset ships no emotion labels. Percentile calibration makes our scale
internally consistent, but nothing so far makes it *externally* correct. This
closes that gap the only way it can be closed: a human listens and says what
they hear, and we score the model against that.

Two modes.

    label   plays clips in a balanced sample and records your judgement
    score   builds the confusion matrix and per-class precision/recall

Sampling is stratified across the model's own predicted states, so the sample
isn't dominated by whatever the model says most often - otherwise a model that
guesses "Energised" for everything would look good on a sample it chose itself.
Clips are presented without showing the prediction, so the label isn't anchored
by it.

Usage:
    python backend/data/label_affect.py label 2021_Abu_Dhabi_Grand_Prix 60
    python backend/data/label_affect.py score 2021_Abu_Dhabi_Grand_Prix
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
CLIPS = os.path.join(HERE, "..", "clips")
LABELS = os.path.join(HERE, "..", "labels")

STATES = ["Calm", "Energised", "Stressed", "Fatigued"]
KEYS = {"1": "Calm", "2": "Energised", "3": "Stressed", "4": "Fatigued", "s": "SKIP"}


def _labels_path(race_id: str) -> str:
    os.makedirs(LABELS, exist_ok=True)
    return os.path.join(LABELS, f"{race_id}.labels.json")


def _load_labels(race_id: str) -> dict[str, str]:
    p = _labels_path(race_id)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def label(race_id: str, n: int = 60) -> None:
    race = json.load(open(os.path.join(RACES, f"{race_id}.json"), encoding="utf-8"))
    done = _load_labels(race_id)

    # Stratify across predicted state so no single class dominates the sample.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for m in race["messages"]:
        if m["id"] not in done and len(m["transcript"].split()) >= 2:
            buckets[m["state"]].append(m)

    rng = random.Random(11)
    per_class = max(1, n // len(STATES))
    sample: list[dict] = []
    for st in STATES:
        pool = buckets.get(st, [])
        rng.shuffle(pool)
        sample.extend(pool[:per_class])
    rng.shuffle(sample)
    sample = sample[:n]

    if not sample:
        print("nothing left to label")
        return

    print(f"\nLabelling {len(sample)} clips from {race_id}")
    print("Play each clip, then press:")
    print("  1 Calm   2 Energised   3 Stressed   4 Fatigued   s Skip   q Save and quit")
    print("Judge the VOICE, not the words.\n")

    for i, m in enumerate(sample, 1):
        path = os.path.abspath(os.path.join(CLIPS, race_id, m["audio_file"]))
        print(f"[{i}/{len(sample)}] {m['driver_code']}  {m['duration_s']}s")
        print(f"    {path}")
        print(f'    "{m["transcript"][:100]}"')

        # Play it. Falls back to just printing the path if no player is available.
        try:
            if sys.platform == "win32":
                os.startfile(path)  # noqa: S606
            elif sys.platform == "darwin":
                os.system(f'afplay "{path}"')
            else:
                os.system(f'aplay -q "{path}" 2>/dev/null || ffplay -nodisp -autoexit -loglevel quiet "{path}"')
        except Exception:
            pass

        while True:
            k = input("    label> ").strip().lower()
            if k == "q":
                json.dump(done, open(_labels_path(race_id), "w", encoding="utf-8"), indent=1)
                print(f"\nsaved {len(done)} labels")
                return
            if k in KEYS:
                if KEYS[k] != "SKIP":
                    done[m["id"]] = KEYS[k]
                break
            print("    use 1/2/3/4/s/q")

        json.dump(done, open(_labels_path(race_id), "w", encoding="utf-8"), indent=1)

    print(f"\ndone - {len(done)} labels saved")


def score(race_id: str) -> None:
    race = json.load(open(os.path.join(RACES, f"{race_id}.json"), encoding="utf-8"))
    truth = _load_labels(race_id)
    if not truth:
        print("no labels yet - run `label` first")
        return

    pred = {m["id"]: m["state"] for m in race["messages"]}
    pairs = [(truth[i], pred[i]) for i in truth if i in pred]
    if not pairs:
        print("labels do not match any message ids")
        return

    matrix: dict[str, Counter] = {s: Counter() for s in STATES}
    for t, p in pairs:
        matrix[t][p] += 1

    correct = sum(1 for t, p in pairs if t == p)
    acc = correct / len(pairs)

    print(f"\n{race_id}: {len(pairs)} labelled clips")
    print(f"accuracy: {acc:.3f} ({correct}/{len(pairs)})\n")

    w = 11
    print(" " * w + "".join(f"{s[:9]:>{w}}" for s in STATES) + f"{'total':>{w}}")
    for t in STATES:
        row = matrix[t]
        print(f"{t:<{w}}" + "".join(f"{row.get(s, 0):>{w}}" for s in STATES)
              + f"{sum(row.values()):>{w}}")

    print("\nper-class:")
    report = {}
    for s in STATES:
        tp = matrix[s].get(s, 0)
        fp = sum(matrix[t].get(s, 0) for t in STATES if t != s)
        fn = sum(v for k, v in matrix[s].items() if k != s)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
        report[s] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        fmt = lambda v: f"{v:.3f}" if v is not None else "  n/a"  # noqa: E731
        print(f"  {s:<10} precision {fmt(prec)}   recall {fmt(rec)}   f1 {fmt(f1)}   n={tp + fn}")

    out = os.path.join(RACES, f"{race_id}.affect_eval.json")
    json.dump({
        "race_id": race_id,
        "n": len(pairs),
        "accuracy": round(acc, 4),
        "confusion": {t: dict(matrix[t]) for t in STATES},
        "per_class": report,
    }, open(out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "label"
    race = sys.argv[2] if len(sys.argv) > 2 else "2021_Abu_Dhabi_Grand_Prix"
    if mode == "score":
        score(race)
    else:
        label(race, int(sys.argv[3]) if len(sys.argv) > 3 else 60)
