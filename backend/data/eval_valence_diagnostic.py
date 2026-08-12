"""Is acoustic valence dead, or is our quadrant mapping at fault?

`_gold_affect_eval.json` says the valence axis scores 0.6285 against a 0.6188
majority baseline - a lift of +0.0097, which is chance. Since DSI weights valence
at 0.45, nearly half the headline index rests on it. Before removing it we need
to know *why* it fails, because the two possible causes have opposite fixes:

  the model cannot recover valence from speech  ->  source valence elsewhere
  our four-quadrant mapping is wrong            ->  fix the mapping, keep the model

The 4-way test cannot separate these. It is indirect: six acted emotions are
collapsed onto a plane of our own construction, the truth split is 61.9/38.1 so
the baseline is already high, and any mapping error shows up as a model error.

The decisive test is a **matched-arousal contrast**. Take two emotions that sit
at the same end of the arousal axis and differ almost purely in valence:

    happy vs anger     both high arousal, opposite valence
    neutral vs sad     both low arousal, opposite valence

If raw valence cannot separate those, no mapping can rescue it. And arousal
*should* fail to separate them - that is the control which proves the contrast
really is matched rather than just easy.

Thresholds are pre-registered in `races/_preregistration.json` (H4) and are
declared here before the numbers are seen:

    AUC <= 0.55   acoustic valence is dead; it leaves the index
    AUC >= 0.65   the model works; the quadrant mapping is what is broken
    otherwise     inconclusive, and reported as inconclusive rather than
                  resolved in whichever direction happens to suit

Per-clip raw scores are cached to `races/_gold_affect_raw.json`, because
re-running the model over CREMA-D costs ~15 minutes on CPU and every future
diagnostic should be free.

Usage:
    python backend/data/eval_valence_diagnostic.py [n_clips]
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import stats  # noqa: E402
from pipeline.prosody import SAMPLE_RATE, analyse  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
CACHE = os.path.join(RACES, "_gold_affect_raw.json")
OUT = os.path.join(RACES, "_valence_diagnostic.json")

DATASET_ID = "confit/cremad-parquet"
SPLIT = "test"

# Pre-registered decision thresholds. Declared before the numbers are seen.
DEAD_AT_OR_BELOW = 0.55
WORKS_AT_OR_ABOVE = 0.65

#: (name, positive-valence label, negative-valence label, shared arousal level)
CONTRASTS = [
    ("happy_vs_anger", {"happy", "happiness"}, {"anger", "angry"}, "high"),
    ("neutral_vs_sad", {"neutral"}, {"sad", "sadness"}, "low"),
]


def build_cache(limit: int | None = None) -> list[dict]:
    """Raw arousal/valence/dominance per CREMA-D clip, with its gold label."""
    from datasets import Audio, load_dataset

    print(f"loading {DATASET_ID} [{SPLIT}] ...", flush=True)
    ds = load_dataset(DATASET_ID, split=SPLIT).cast_column("audio", Audio(decode=False))
    total = min(len(ds), limit) if limit else len(ds)
    print(f"{total} clips to score", flush=True)

    # Iterate lazily. Materialising the split with list(ds) pulls every clip's
    # audio bytes into memory before a single one is scored, which on this
    # dataset means minutes of silence before any progress appears.
    out, t0 = [], time.perf_counter()
    for i, r in enumerate(ds, 1):
        if limit and i > limit:
            break
        data = (r.get("audio") or {}).get("bytes")
        if not data:
            continue
        try:
            audio, _ = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"  !! decode failed: {type(e).__name__}: {e}", flush=True)
            continue
        af = analyse(np.asarray(audio, dtype=np.float32))
        out.append({
            "label": str(r.get("emotion", "")).strip().lower(),
            "arousal": af.arousal, "valence": af.valence, "dominance": af.dominance,
            "duration_s": round(len(audio) / SAMPLE_RATE, 2),
        })
        # Frequent enough that a stalled run is obvious rather than ambiguous.
        if i % 25 == 0:
            rate = (time.perf_counter() - t0) / i
            print(f"  {i}/{total}  ~{rate * (total - i) / 60:.1f} min left", flush=True)

    json.dump({
        "generated_by": "backend/data/eval_valence_diagnostic.py",
        "dataset": DATASET_ID, "split": SPLIT,
        "note": "Raw model output per clip, cached so diagnostics are free to re-run.",
        "n": len(out), "clips": out,
    }, open(CACHE, "w", encoding="utf-8"), indent=1)
    print(f"cached {len(out)} clips -> {CACHE}")
    return out


def load_cache(limit: int | None = None) -> list[dict]:
    if os.path.exists(CACHE):
        d = json.load(open(CACHE, encoding="utf-8"))
        print(f"using cached raw scores ({d['n']} clips) from {CACHE}")
        return d["clips"]
    return build_cache(limit)


def auc(positive: list[float], negative: list[float]) -> float | None:
    """Probability a random positive scores above a random negative.

    Rank-based, so it needs no threshold and cannot be flattered by tuning one
    on the test set. 0.5 is chance; below 0.5 means the signal runs backwards.
    """
    if not positive or not negative:
        return None
    values = [(v, 1) for v in positive] + [(v, 0) for v in negative]
    values.sort(key=lambda x: x[0])
    # Average ranks so ties contribute 0.5 rather than an arbitrary direction.
    ranks, i = {}, 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1][0] == values[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_, lab) in enumerate(values) if lab == 1)
    n_pos, n_neg = len(positive), len(negative)
    return round((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg), 4)


def contrast(clips: list[dict], pos_labels: set, neg_labels: set,
             axis: str) -> dict:
    pos = [c[axis] for c in clips if c["label"] in pos_labels]
    neg = [c[axis] for c in clips if c["label"] in neg_labels]
    a = auc(pos, neg)
    return {
        "n_positive": len(pos), "n_negative": len(neg),
        "auc": a,
        "auc_ci95": stats.bootstrap_ci(
            [(v, 1) for v in pos] + [(v, 0) for v in neg],
            lambda s: auc([v for v, l in s if l == 1], [v for v, l in s if l == 0]) or 0.5,
        ),
        "mean_positive": round(sum(pos) / len(pos), 4) if pos else None,
        "mean_negative": round(sum(neg) / len(neg), 4) if neg else None,
    }


def stratified_valence_auc(clips: list[dict], pos_labels: set, neg_labels: set,
                           n_strata: int = 5, min_per_class: int = 8) -> dict:
    """Valence AUC computed *within* bands of equal arousal, then pooled.

    The naive contrast assumes happy and anger are matched on arousal. In this
    data they are not - the control shows arousal separating them at AUC 0.27,
    which is further from chance than the valence signal being claimed. So the
    naive number cannot distinguish "the model reads valence" from "the model
    reads arousal, and arousal happens to differ between these two labels".

    Slicing into arousal quintiles and scoring valence inside each one removes
    that leak: within a stratum the two classes have near-identical arousal, so
    whatever separates them there is not arousal. Strata are pooled by sample
    size.
    """
    tagged = ([(c["arousal"], c["valence"], 1) for c in clips if c["label"] in pos_labels]
              + [(c["arousal"], c["valence"], 0) for c in clips if c["label"] in neg_labels])
    if len(tagged) < n_strata * min_per_class * 2:
        return {"computable": False, "why": "too few clips to stratify"}

    tagged.sort(key=lambda t: t[0])
    size = len(tagged) // n_strata
    strata, weights = [], []
    for i in range(n_strata):
        chunk = tagged[i * size:(i + 1) * size] if i < n_strata - 1 else tagged[i * size:]
        p = [v for _, v, lab in chunk if lab == 1]
        n = [v for _, v, lab in chunk if lab == 0]
        if len(p) < min_per_class or len(n) < min_per_class:
            strata.append({"n_positive": len(p), "n_negative": len(n),
                           "auc": None, "skipped": "too few in one class"})
            continue
        a = auc(p, n)
        ar = [x for x, _, _ in chunk]
        strata.append({
            "arousal_range": [round(min(ar), 3), round(max(ar), 3)],
            "n_positive": len(p), "n_negative": len(n), "auc": a,
            "arousal_auc_within": auc([x for x, _, lab in chunk if lab == 1],
                                      [x for x, _, lab in chunk if lab == 0]),
        })
        weights.append((a, len(p) + len(n)))

    if not weights:
        return {"computable": False, "why": "no stratum had both classes"}
    total = sum(w for _, w in weights)
    pooled = sum(a * w for a, w in weights) / total
    return {
        "computable": True,
        "pooled_auc": round(pooled, 4),
        "n_strata_used": len(weights),
        "n_scored": total,
        "strata": strata,
    }


def verdict(naive_auc: float | None, matched: bool, strat: dict,
            arousal_control_auc: float | None) -> tuple[str, str]:
    """The stratified number decides, and only if the design held.

    The pre-registered test assumed happy and anger are matched on arousal. That
    assumption is checkable and it failed, so the naive AUC cannot be read as a
    valence result - it is contaminated by exactly the axis the design was meant
    to hold constant. Deferring to the arousal-stratified figure instead is the
    conservative reading, and if that is not computable the answer is that the
    test did not run, not that valence works.
    """
    if naive_auc is None:
        return "not computable", "Not enough clips in one of the classes."

    caveat = ""
    if not matched:
        caveat = (
            f" The pre-registered design assumed the pair was matched on arousal "
            f"and it is not: arousal separates these labels at AUC "
            f"{arousal_control_auc}, further from chance than the valence signal "
            f"being claimed ({naive_auc}). The naive figure is therefore "
            "contaminated by arousal leakage and is not used to decide."
        )

    if not strat.get("computable"):
        return "inconclusive", (
            "The matched-arousal assumption failed and the arousal-stratified "
            f"fallback could not be computed ({strat.get('why')}). The question "
            "is undecided, and reported as undecided rather than resolved from a "
            "contaminated number." + caveat
        )

    a = strat["pooled_auc"]
    basis = (f"Within bands of equal arousal - which removes the leak the control "
             f"exposed - valence separates the pair at AUC {a}, pooled over "
             f"{strat['n_strata_used']} strata and {strat['n_scored']} clips.")

    if a <= DEAD_AT_OR_BELOW:
        return "valence_is_dead", (
            f"{basis} That is at or below the pre-registered {DEAD_AT_OR_BELOW} "
            "threshold. This is the cleanest test available - clean studio "
            "speech, maximal valence contrast, arousal held constant - and the "
            "model cannot do it. No quadrant mapping can rescue it, so acoustic "
            "valence leaves the index." + caveat
        )
    if a >= WORKS_AT_OR_ABOVE:
        return "mapping_is_at_fault", (
            f"{basis} That is at or above the pre-registered {WORKS_AT_OR_ABOVE} "
            "threshold, and it survives holding arousal constant. The model does "
            "carry valence; the four-quadrant mapping is what loses it. Keep the "
            "model and fix the mapping." + caveat
        )
    return "inconclusive", (
        f"{basis} That falls between the pre-registered thresholds of "
        f"{DEAD_AT_OR_BELOW} and {WORKS_AT_OR_ABOVE}, so it does not decide the "
        "question. Reported as inconclusive rather than resolved in whichever "
        "direction happens to suit; the index is left unchanged." + caveat
    )


def main(limit: int | None = None) -> None:
    clips = load_cache(limit)
    if not clips:
        print("no clips")
        return

    results = {}
    for name, pos, neg, arousal_level in CONTRASTS:
        v = contrast(clips, pos, neg, "valence")
        a = contrast(clips, pos, neg, "arousal")
        results[name] = {
            "positive_labels": sorted(pos), "negative_labels": sorted(neg),
            "shared_arousal": arousal_level,
            "valence_naive": v,
            "arousal_control": a,
            # If arousal separates them strongly the pair is not matched, and the
            # naive valence number is measuring arousal leakage rather than valence.
            "contrast_is_matched": a["auc"] is not None and abs(a["auc"] - 0.5) < 0.15,
            "valence_arousal_stratified": stratified_valence_auc(clips, pos, neg),
        }

    primary = results["happy_vs_anger"]
    label, explanation = verdict(
        primary["valence_naive"]["auc"],
        primary["contrast_is_matched"],
        primary["valence_arousal_stratified"],
        primary["arousal_control"]["auc"],
    )

    payload = {
        "generated_by": "backend/data/eval_valence_diagnostic.py",
        "question": "Is acoustic valence dead, or is our quadrant mapping at fault?",
        "why": (
            "The 4-way gold eval reports valence at chance (+0.0097 lift), but it "
            "cannot say whether the model or our mapping is responsible. A "
            "matched-arousal contrast can: happy and anger sit at the same end of "
            "the arousal axis and differ almost purely in valence."
        ),
        "dataset": DATASET_ID, "split": SPLIT, "n_clips": len(clips),
        "preregistered_thresholds": {
            "dead_at_or_below": DEAD_AT_OR_BELOW,
            "works_at_or_above": WORKS_AT_OR_ABOVE,
            "registered_in": "races/_preregistration.json (H4)",
        },
        "contrasts": results,
        "verdict": label,
        "explanation": explanation,
        "caveat": (
            "CREMA-D is acted, studio-quality American English. The implication "
            "runs one way only: a model that cannot recover valence from clean "
            "acted speech certainly cannot do it on compressed team radio. A pass "
            "here would not have proven the reverse."
        ),
    }

    print(f"\n{len(clips)} clips\n")
    for name, r in results.items():
        v, a, s = r["valence_naive"], r["arousal_control"], r["valence_arousal_stratified"]
        print(f"{name}  ({r['positive_labels']} vs {r['negative_labels']}, "
              f"both {r['shared_arousal']} arousal)")
        print(f"  n = {v['n_positive']} vs {v['n_negative']}")
        print(f"  valence naive       auc {v['auc']}  ci95 {v['auc_ci95']}  "
              f"means {v['mean_positive']} vs {v['mean_negative']}")
        print(f"  arousal control     auc {a['auc']}  (should be near 0.5)  "
              f"{'matched' if r['contrast_is_matched'] else '!! NOT MATCHED'}")
        if s.get("computable"):
            print(f"  valence stratified  auc {s['pooled_auc']}  "
                  f"({s['n_strata_used']} arousal bands, {s['n_scored']} clips) "
                  "<- the one that counts")
        else:
            print(f"  valence stratified  not computable: {s.get('why')}")
        print()

    print(f"VERDICT: {label}\n  {explanation}")
    json.dump(payload, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
