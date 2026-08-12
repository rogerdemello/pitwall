"""Validate the affect scale against gold emotion labels.

Until now nothing in this project has checked the affect scoring against a label
of any kind. Percentile calibration makes the scale internally consistent; it
does not make it correct, and the Evidence page has been saying so.

The assumed blocker was the human listening pass. It wasn't: CREMA-D is on the
Hub, ungated, with 7,442 clips and six emotion labels. We can run the *exact*
production path - prosody.analyse -> Calibrator -> fusion._quadrant - against
those labels and get a real confusion matrix.

Two design decisions that this stands or falls on:

  1. **Calibration is fitted on CREMA-D's own distribution**, not on F1. Reusing
     the F1 mapping would test calibration transfer between two very different
     domains, which is a different question and a much easier one to fail for
     uninteresting reasons.

  2. **Accuracy is reported against the majority-class baseline**, not against
     25%. Six labels collapse into four unbalanced quadrants, so uniform chance
     is the wrong reference. Quoting "better than 25%" would be the fourth false
     positive this project has had to catch, and the guard belongs in the code
     rather than in a reviewer's head.

**Caveat that ships with the result:** CREMA-D is acted, studio-quality American
English. This validates the model and our quadrant logic. It does *not*
demonstrate in-domain performance on compressed team radio. The implication only
runs one way - if the model cannot recover emotion from clean acted speech, it
is certainly not doing so on radio.

Usage:
    python backend/data/eval_affect_gold.py [n_clips]
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from collections import Counter, defaultdict

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import fusion, labels  # noqa: E402
from pipeline.calibration import Calibrator  # noqa: E402
from pipeline.prosody import SAMPLE_RATE, analyse  # noqa: E402

DATASET_ID = "confit/cremad-parquet"
SPLIT = "test"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "races",
                   "_gold_affect_eval.json")

# Mapping onto the arousal/valence plane our product uses, from pipeline/labels.py.
# Disgust is excluded: it does not place cleanly on that plane, and forcing it
# would be us choosing the answer. eval_convergent.py makes the opposite choice,
# which used to be invisible - both files now record their treatment in the
# output so two results computed under different label sets cannot be read as
# comparable.
INCLUDE_DISGUST = False
LABEL_TO_QUADRANT = labels.quadrant_map(INCLUDE_DISGUST)
EXCLUDED = set(labels.DISPUTED)
STATES = labels.STATES


def load_clips(limit: int | None):
    from datasets import Audio, load_dataset

    ds = load_dataset(DATASET_ID, split=SPLIT)
    ds = ds.cast_column("audio", Audio(decode=False))
    rows = []
    for r in ds:
        label = str(r.get("emotion", "")).strip().lower()
        if label in EXCLUDED or label not in LABEL_TO_QUADRANT:
            continue
        rows.append(r)
        if limit and len(rows) >= limit:
            break
    return rows


HIGH_AROUSAL = labels.HIGH_AROUSAL
NEGATIVE_VALENCE = labels.NEGATIVE_VALENCE


def axis_breakdown(confusion: dict[str, dict[str, int]]) -> dict:
    """Score the two underlying dimensions separately.

    A single 4-way accuracy hides which half of the model works. The confusion
    matrix showed Stressed leaking heavily into Energised, and Calm into
    Fatigued - in both cases the pair differs *only* in valence, which points at
    one dimension carrying the errors. Collapsing the quadrants onto each axis
    turns that hunch into two numbers, and it matters because DSI weights
    valence at 0.45: if valence is noise, nearly half the index is noise.
    """
    def score(group: set[str]) -> dict:
        correct = total = in_group_truth = 0
        for truth, preds in confusion.items():
            t_in = truth in group
            in_group_truth += sum(preds.values()) if t_in else 0
            for pred, count in preds.items():
                total += count
                if t_in == (pred in group):
                    correct += count
        if not total:
            return {}
        # Majority baseline on a binary split is whichever side is larger.
        majority = max(in_group_truth, total - in_group_truth) / total
        acc = correct / total
        return {
            "accuracy": round(acc, 4),
            "majority_baseline": round(majority, 4),
            "lift": round(acc - majority, 4),
            "n": total,
        }

    arousal = score(HIGH_AROUSAL)
    valence = score(NEGATIVE_VALENCE)
    weak = [name for name, s in (("arousal", arousal), ("valence", valence))
            if s and s["lift"] < 0.05]
    return {
        "arousal_high_vs_low": arousal,
        "valence_negative_vs_positive": valence,
        "near_chance_axes": weak,
        "interpretation": (
            "Arousal is recovered well; valence is at or near chance. Since DSI "
            "weights valence at 0.45, close to half the index rests on the "
            "dimension the model does not resolve - which is the single most "
            "important limitation of the affect scale."
            if "valence" in weak and "arousal" not in weak else
            "Both dimensions carry signal above their baselines."
            if not weak else
            f"These axes sit near chance: {', '.join(weak)}."
        ),
    }


def main(limit: int | None = None) -> None:
    print(f"loading {DATASET_ID} [{SPLIT}] ...")
    rows = load_clips(limit)
    print(f"{len(rows)} clips after excluding {'/'.join(sorted(EXCLUDED))}")

    # Pass 1: raw affect for every clip.
    raw, t0 = [], time.perf_counter()
    for i, r in enumerate(rows, 1):
        data = r["audio"].get("bytes")
        if not data:
            continue
        try:
            audio, _ = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"  !! decode failed: {type(e).__name__}: {e}")
            continue
        af = analyse(np.asarray(audio, dtype=np.float32))
        raw.append({
            "label": str(r["emotion"]).strip().lower(),
            "arousal": af.arousal, "valence": af.valence, "dominance": af.dominance,
        })
        if i % 50 == 0:
            rate = (time.perf_counter() - t0) / i
            print(f"  {i}/{len(rows)}  ~{rate * (len(rows) - i) / 60:.1f} min remaining",
                  flush=True)

    if not raw:
        print("nothing decoded")
        return

    # Pass 2: calibrate on THIS corpus, then apply the production quadrant logic.
    cal = Calibrator.fit(raw)
    pairs = []
    for r in raw:
        a = cal.pct_arousal(r["arousal"])
        v = cal.pct_valence(r["valence"])
        pred, _ = fusion._quadrant(a, v)
        pairs.append((LABEL_TO_QUADRANT[r["label"]], pred))

    truth_counts = Counter(t for t, _ in pairs)
    correct = sum(1 for t, p in pairs if t == p)
    acc = correct / len(pairs)

    # The reference that matters: always guessing the most common class.
    majority = max(truth_counts.values()) / len(pairs)

    matrix = {t: Counter() for t in STATES}
    for t, p in pairs:
        matrix[t][p] += 1

    per_class = {}
    for s in STATES:
        tp = matrix[s].get(s, 0)
        fp = sum(matrix[t].get(s, 0) for t in STATES if t != s)
        fn = sum(v for k, v in matrix[s].items() if k != s)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        per_class[s] = {
            "precision": round(prec, 3) if prec is not None else None,
            "recall": round(rec, 3) if rec is not None else None,
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec and rec else None,
            "support": tp + fn,
        }

    beats = acc > majority
    result = {
        "dataset": DATASET_ID,
        "split": SPLIT,
        "n": len(pairs),
        "label_treatment": labels.treatment(INCLUDE_DISGUST),
        "excluded_labels": labels.excluded_labels(INCLUDE_DISGUST),
        "label_mapping": LABEL_TO_QUADRANT,
        "accuracy": round(acc, 4),
        "majority_class_baseline": round(majority, 4),
        "beats_baseline": beats,
        "confusion": {t: dict(matrix[t]) for t in STATES},
        "per_class": per_class,
        "axes": axis_breakdown({t: dict(matrix[t]) for t in STATES}),
        "verdict": (
            f"Accuracy {acc:.1%} vs a majority-class baseline of {majority:.1%} - "
            + ("the mapping carries real signal."
               if beats else
               "the mapping does NOT beat simply guessing the most common class.")
        ),
        "caveat": (
            "CREMA-D is acted, studio-quality American English. This validates the "
            "prosody model and our arousal/valence quadrant logic; it does not "
            "demonstrate in-domain performance on compressed F1 team radio. The "
            "implication runs one way only: a model that cannot recover emotion "
            "from clean acted speech is certainly not doing so on radio."
        ),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w", encoding="utf-8"), indent=1)

    print(f"\naccuracy {acc:.4f}  |  majority-class baseline {majority:.4f}  "
          f"|  {'BEATS' if beats else 'DOES NOT BEAT'} baseline")
    w = 11
    print("\n" + " " * w + "".join(f"{s[:9]:>{w}}" for s in STATES))
    for t in STATES:
        print(f"{t:<{w}}" + "".join(f"{matrix[t].get(s, 0):>{w}}" for s in STATES))
    print("\nper class:")
    for s, v in per_class.items():
        print(f"  {s:<10} precision {v['precision']}  recall {v['recall']}  n={v['support']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
