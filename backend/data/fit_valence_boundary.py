"""Where should the valence boundary sit, and does moving it help?

The diagnostic (`_valence_diagnostic.json`) established that the affect model
ranks valence at AUC 0.67 with arousal held constant, while the four-way
evaluation reports the valence axis at +0.0097 over baseline - chance. Both are
right. AUC scores *ranking*; the axis lift scores a *threshold*. `fusion._quadrant`
splits at exactly 0.5 of the calibrated percentile, and if the real class
boundary is not at the median then a model that ranks well still classifies at
chance.

So this asks two separate questions and keeps them separate:

  1. Is the 0.5 split misplaced? Fit the threshold and see how much accuracy it
     recovers on held-out speakers. This measures the *model's* headroom.

  2. Should production adopt the fitted number? That is a different question,
     because a percentile threshold encodes a base rate. CREMA-D is 62% negative
     valence by construction; F1 radio's true rate is unknown and there are no
     in-domain labels to measure it. A threshold fitted here would import
     CREMA-D's balance into a corpus that has no reason to share it.

**Split by actor, not by clip.** CREMA-D has 91 actors and ~12 clips each. A clip
level split puts the same speaker on both sides, so the threshold is tuned to
voices it is then tested on, and the number comes back flattering and useless.

Usage:
    python backend/data/fit_valence_boundary.py
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import labels, stats  # noqa: E402
from pipeline.calibration import Calibrator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
CACHE = os.path.join(RACES, "_gold_affect_raw.json")
OUT = os.path.join(RACES, "_valence_boundary.json")

DATASET_ID = "confit/cremad-parquet"
SPLIT = "test"
N_FOLDS = 5

#: Below this the fitted threshold is not worth importing another domain's base
#: rate for. Declared before the numbers are seen.
MATERIAL_LIFT = 0.05


def load_with_actors() -> list[dict]:
    """Cached raw scores, joined to the actor id encoded in each filename.

    CREMA-D filenames are <actor>_<sentence>_<emotion>_<intensity>.wav. The cache
    was written by iterating the split in order, so the join is positional - and
    verified by checking every label agrees before it is used.
    """
    if not os.path.exists(CACHE):
        print(f"no cache at {CACHE} - run eval_valence_diagnostic.py first")
        return []
    clips = json.load(open(CACHE, encoding="utf-8"))["clips"]

    from datasets import Audio, load_dataset
    ds = load_dataset(DATASET_ID, split=SPLIT).cast_column("audio", Audio(decode=False))
    files, emotions = ds["file"], [str(e).strip().lower() for e in ds["emotion"]]

    if len(files) != len(clips):
        print(f"!! cache has {len(clips)} clips, dataset has {len(files)}; "
              "rebuild the cache")
        return []
    mismatched = sum(1 for e, c in zip(emotions, clips) if e != c["label"])
    if mismatched:
        print(f"!! positional join is unsafe: {mismatched} labels disagree")
        return []

    out = []
    for f, c in zip(files, clips):
        c = dict(c)
        c["actor"] = os.path.basename(f).split("_")[0]
        out.append(c)
    return out


def best_threshold(scores: list[float], truth: list[bool]) -> tuple[float, float]:
    """The cut that maximises accuracy, and that accuracy.

    Candidates are midpoints between adjacent observed values, so the search is
    exact rather than a grid approximation.
    """
    pairs = sorted(zip(scores, truth))
    xs = [x for x, _ in pairs]
    cuts = [0.0] + [(xs[i] + xs[i + 1]) / 2 for i in range(len(xs) - 1)] + [1.0]
    best, best_acc = 0.5, -1.0
    for c in cuts:
        acc = sum(1 for x, t in pairs if (x < c) == t) / len(pairs)
        if acc > best_acc:
            best, best_acc = c, acc
    return round(best, 4), round(best_acc, 4)


def evaluate(scores: list[float], truth: list[bool], cut: float) -> float:
    """Accuracy of `score < cut` as a predictor of `truth`."""
    return sum(1 for x, t in zip(scores, truth) if (x < cut) == t) / len(scores)


def grouped_folds(actors: list[str], n: int) -> list[set[str]]:
    """Partition actors into n folds, largest-first for balance."""
    from collections import Counter
    counts = Counter(actors)
    folds: list[set[str]] = [set() for _ in range(n)]
    sizes = [0] * n
    for actor, c in counts.most_common():
        i = sizes.index(min(sizes))
        folds[i].add(actor)
        sizes[i] += c
    return folds


def axis_run(clips: list[dict], axis: str, negative_labels: set[str],
             include_disgust: bool, space: str = "percentile") -> dict:
    """Fit and score one axis under actor-grouped cross-validation.

    `space` decides what the threshold is a statement about, and it matters more
    than the accuracy figure:

      percentile  "the bottom X% of this corpus is negative". Encodes a base
                  rate, so a cut fitted on CREMA-D (62% negative by
                  construction) carries that balance into any corpus it is
                  applied to.

      raw         "model output below X is negative". A property of the model's
                  output scale rather than of the corpus, so it is the only one
                  of the two that can honestly transfer to F1 radio, where the
                  true base rate is unknown and unmeasurable without labels.
    """
    usable = [c for c in clips
              if labels.to_quadrant(c["label"], include_disgust) is not None]
    cal = Calibrator.fit(usable)
    pct = {"valence": cal.pct_valence, "arousal": cal.pct_arousal}[axis]

    scores = ([pct(c[axis]) for c in usable] if space == "percentile"
              else [c[axis] for c in usable])
    truth = [labels.to_quadrant(c["label"], include_disgust) in negative_labels
             for c in usable]
    actors = [c["actor"] for c in usable]

    # What production does today: split at the median of the calibrated
    # percentile. In raw space that is the median raw score, which is the same
    # rule expressed on a different scale.
    incumbent_cut = 0.5 if space == "percentile" else round(st.median(scores), 4)

    folds = grouped_folds(actors, N_FOLDS)
    rows = []
    for k, held in enumerate(folds):
        tr = [i for i, a in enumerate(actors) if a not in held]
        te = [i for i, a in enumerate(actors) if a in held]
        if not tr or not te:
            continue
        cut, _ = best_threshold([scores[i] for i in tr], [truth[i] for i in tr])
        te_scores = [scores[i] for i in te]
        te_truth = [truth[i] for i in te]
        base = max(sum(te_truth), len(te_truth) - sum(te_truth)) / len(te_truth)
        rows.append({
            "fold": k, "n_train": len(tr), "n_test": len(te),
            "n_actors_held_out": len(held),
            "fitted_cut": cut,
            "acc_at_fitted": round(evaluate(te_scores, te_truth, cut), 4),
            "acc_at_median": round(evaluate(te_scores, te_truth, incumbent_cut), 4),
            "majority_baseline": round(base, 4),
        })

    if not rows:
        return {"computable": False}

    def mean(key):
        return round(st.fmean(r[key] for r in rows), 4)

    fitted, median, base = mean("acc_at_fitted"), mean("acc_at_median"), mean("majority_baseline")
    cuts = [r["fitted_cut"] for r in rows]
    in_sample_cut, in_sample_acc = best_threshold(scores, truth)
    # Relative spread, so raw and percentile cuts are comparable: a spread of
    # 0.05 means very different things on a scale spanning 0.5 versus one
    # spanning 1.0.
    span = (max(scores) - min(scores)) or 1.0

    return {
        "computable": True,
        "space": space,
        "n_clips": len(usable), "n_actors": len(set(actors)), "n_folds": len(rows),
        "auc": _auc(scores, truth),
        "cv_acc_at_fitted_threshold": fitted,
        "cv_acc_at_median_split": median,
        "cv_majority_baseline": base,
        "lift_over_median": round(fitted - median, 4),
        "lift_over_baseline_fitted": round(fitted - base, 4),
        "lift_over_baseline_median": round(median - base, 4),
        "incumbent_cut": incumbent_cut,
        "fitted_cut_per_fold": cuts,
        "fitted_cut_mean": round(st.fmean(cuts), 4),
        "fitted_cut_spread": round(max(cuts) - min(cuts), 4),
        "fitted_cut_spread_relative": round((max(cuts) - min(cuts)) / span, 4),
        "score_span": [round(min(scores), 4), round(max(scores), 4)],
        "in_sample_cut": in_sample_cut,
        "in_sample_acc": in_sample_acc,
        "optimism": round(in_sample_acc - fitted, 4),
        "folds": rows,
    }


def transfer_check(cut: float, axis: str) -> dict:
    """Would the fitted cut survive the move from acted speech to team radio?

    A raw threshold transfers only if the model's output distribution is
    comparable across the two domains. If F1 radio sits somewhere else on the
    scale entirely, an absolute cut fitted on CREMA-D relabels the corpus for a
    reason that has nothing to do with how the drivers sound.
    """
    from pipeline.artifacts import iter_race_files
    races = os.path.join(RACES)
    f1 = []
    for path in iter_race_files(races):
        for m in json.load(open(path, encoding="utf-8"))["messages"]:
            v = m.get(f"{axis}_raw")
            if v is not None:
                f1.append(v)
    if not f1:
        return {"computable": False}

    cache = json.load(open(CACHE, encoding="utf-8"))["clips"]
    gold = [c[axis] for c in cache]

    def q(xs, p):
        return round(sorted(xs)[min(len(xs) - 1, int(len(xs) * p))], 4)

    shift = st.median(f1) - st.median(gold)
    below_f1 = sum(1 for v in f1 if v < cut) / len(f1)
    below_gold = sum(1 for v in gold if v < cut) / len(gold)
    # Expressed in gold standard deviations, which is the scale on which "the
    # same cut means the same thing" either holds or does not.
    sd = st.pstdev(gold) or 1.0
    return {
        "computable": True,
        "n_f1": len(f1), "n_gold": len(gold),
        "gold_median": round(st.median(gold), 4), "f1_median": round(st.median(f1), 4),
        "gold_iqr": [q(gold, 0.25), q(gold, 0.75)],
        "f1_iqr": [q(f1, 0.25), q(f1, 0.75)],
        "median_shift": round(shift, 4),
        "median_shift_in_gold_sds": round(shift / sd, 2),
        "share_below_cut_gold": round(below_gold, 4),
        "share_below_cut_f1": round(below_f1, 4),
        "distributions_comparable": abs(shift / sd) < 0.5,
    }


def _auc(scores: list[float], truth: list[bool]) -> float | None:
    """Rank-based separation, which no threshold can affect."""
    pos = [s for s, t in zip(scores, truth) if not t]   # positive valence
    neg = [s for s, t in zip(scores, truth) if t]
    if not pos or not neg:
        return None
    better = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return round((better + 0.5 * ties) / (len(pos) * len(neg)), 4)


def main() -> None:
    clips = load_with_actors()
    if not clips:
        return

    include_disgust = False
    low_arousal = set(labels.STATES) - labels.HIGH_AROUSAL
    valence = axis_run(clips, "valence", labels.NEGATIVE_VALENCE, include_disgust)
    arousal = axis_run(clips, "arousal", low_arousal, include_disgust)
    valence_raw = axis_run(clips, "valence", labels.NEGATIVE_VALENCE,
                           include_disgust, space="raw")
    arousal_raw = axis_run(clips, "arousal", low_arousal, include_disgust,
                           space="raw")

    # The raw-space fit decides. A percentile cut says "the bottom X% of this
    # corpus is negative", which cannot be carried to a corpus with a different
    # base rate; a raw cut says "model output below X is negative", which is a
    # statement about the model and transfers.
    v = valence_raw
    boundary_is_misplaced = v["computable"] and v["lift_over_median"] >= MATERIAL_LIFT
    stable = v["computable"] and v["fitted_cut_spread_relative"] <= 0.15
    transfer = transfer_check(v["fitted_cut_mean"], "valence") if v["computable"] else {}

    # Two questions, answered separately, because the first can be yes while the
    # second is no - and conflating them is how an out-of-domain constant ends up
    # silently relabelling a corpus.
    if not v["computable"]:
        finding, recommendation, why = "not computable", "no_change", \
            "Cross-validation did not run."
    elif not boundary_is_misplaced:
        finding = "boundary_is_fine"
        recommendation, why = "no_change", (
            f"Fitting the threshold on held-out speakers recovers only "
            f"{v['lift_over_median']:+.4f} accuracy over the incumbent split, "
            f"below the pre-declared {MATERIAL_LIFT} bar."
        )
    elif not stable:
        finding = "boundary_is_misplaced_but_unstable"
        recommendation, why = "no_change", (
            f"The fitted threshold beats the incumbent by {v['lift_over_median']:+.4f}, "
            f"but it moves {v['fitted_cut_spread']} between actor folds. A boundary "
            "that unstable across speakers will not transfer."
        )
    elif not transfer.get("distributions_comparable", False):
        finding = "boundary_is_misplaced_on_gold"
        recommendation, why = "do_not_transfer_to_f1", (
            f"On gold labels the boundary is definitively misplaced: moving it "
            f"lifts the valence axis from {v['lift_over_baseline_median']:+.4f} to "
            f"{v['lift_over_baseline_fitted']:+.4f} over baseline, stable to "
            f"{v['fitted_cut_spread']} across 91 actors. The valence axis is "
            "threshold-limited, not signal-limited, and the published claim that "
            "it sits 'at chance' understates the model.\n\n"
            f"    But the cut does NOT transfer. Raw valence on F1 radio has median "
            f"{transfer['f1_median']} against {transfer['gold_median']} on CREMA-D, "
            f"a shift of {transfer['median_shift_in_gold_sds']} gold standard "
            f"deviations. The same absolute cut catches "
            f"{transfer['share_below_cut_gold']:.1%} of gold clips but only "
            f"{transfer['share_below_cut_f1']:.1%} of radio messages. Applying it "
            "would relabel the corpus for a reason that has nothing to do with how "
            "the drivers sound, and there are no in-domain labels to say whether "
            "the shift is real or a channel artifact.\n\n"
            "    So the finding is published and the production boundary is left "
            "alone. This is now the single highest-value thing an in-domain "
            "labelling pass would settle: we know exactly which number it decides."
        )
    else:
        finding = "boundary_is_misplaced"
        recommendation, why = "move_the_boundary", (
            f"The fitted threshold beats the incumbent by {v['lift_over_median']:+.4f} "
            f"on held-out speakers, is stable across folds (spread "
            f"{v['fitted_cut_spread']}), and the two domains' distributions are "
            f"comparable ({transfer['median_shift_in_gold_sds']} gold sds apart), "
            "so it transfers."
        )

    payload = {
        "generated_by": "backend/data/fit_valence_boundary.py",
        "question": (
            "The model ranks valence at AUC 0.67 but the axis scores at chance. "
            "Is the 0.5 decision boundary what loses it?"
        ),
        "method": (
            f"Threshold fitted to maximise accuracy on training actors and scored "
            f"on held-out actors, {N_FOLDS}-fold grouped by actor. CREMA-D has 91 "
            "actors at ~12 clips each; a clip-level split would put the same voice "
            "on both sides and flatter the result."
        ),
        "dataset": DATASET_ID, "split": SPLIT,
        "label_treatment": labels.treatment(include_disgust),
        "material_lift_threshold": MATERIAL_LIFT,
        "valence_percentile_space": valence,
        "arousal_percentile_space": arousal,
        "valence_raw_space": valence_raw,
        "arousal_raw_space": arousal_raw,
        "decided_on": "valence_raw_space",
        "transfer_check": transfer,
        "finding": finding,
        "recommendation": recommendation,
        "why": why,
        "base_rate_caveat": (
            "A percentile threshold encodes a base rate. CREMA-D is 62% negative "
            "valence by construction; the true rate for F1 team radio is unknown "
            "and there are no in-domain labels to measure it. Any threshold "
            "fitted here would import CREMA-D's balance, so a fitted number is "
            "evidence about the model's headroom first and a production setting "
            "only second."
        ),
    }

    for name, r in (("VALENCE  [percentile space]", valence),
                    ("VALENCE  [raw space - decides]", valence_raw),
                    ("AROUSAL  [percentile space]", arousal),
                    ("AROUSAL  [raw space]", arousal_raw)):
        if not r["computable"]:
            print(f"{name}: not computable")
            continue
        print(f"{name}  ({r['n_clips']} clips, {r['n_actors']} actors, "
              f"{r['n_folds']} folds)")
        print(f"  AUC (threshold-free)          {r['auc']}")
        print(f"  majority baseline             {r['cv_majority_baseline']}")
        print(f"  accuracy at incumbent {r['incumbent_cut']:<7} {r['cv_acc_at_median_split']}  "
              f"(lift {r['lift_over_baseline_median']:+.4f})")
        print(f"  accuracy at fitted threshold  {r['cv_acc_at_fitted_threshold']}  "
              f"(lift {r['lift_over_baseline_fitted']:+.4f})")
        print(f"  gain from moving the boundary {r['lift_over_median']:+.4f}")
        print(f"  fitted cut  mean {r['fitted_cut_mean']}  spread "
              f"{r['fitted_cut_spread']} ({r['fitted_cut_spread_relative']:.1%} of "
              f"range {r['score_span']})")
        print(f"  in-sample optimism            {r['optimism']:+.4f}")
        print()

    print(f"RECOMMENDATION: {recommendation}\n  {why}")
    json.dump(payload, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
