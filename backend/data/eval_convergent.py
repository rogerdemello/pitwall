"""Convergent validity: does a second, independent model see the same thing?

Gold-label validation (eval_affect_gold.py) tells us whether the affect scale
works on clean acted speech. It cannot tell us whether it works on compressed
team radio, because no labelled radio exists.

This is the next best thing. If a completely separate model - different
architecture head, different training data, categorical rather than dimensional -
lands on the same quadrant as ours for the same clip more often than chance,
that is real evidence the signal is in the audio rather than in one model's
quirks. If it doesn't, that is worth knowing before anyone presents this.

Reported as **Cohen's kappa**, not raw agreement. Two models that both favour one
class will agree a lot purely by accident: on a corpus that is 40% one quadrant,
two independent coin-flippers weighted the same way agree ~30% of the time and it
means nothing. Kappa subtracts that expected agreement out.

Second model: Dpngtm/wav2vec2-emotion-recognition (ungated, transformers-native).
Its model card loads audio with torchaudio; we use librosa, so the broken
torchaudio build in this environment is never touched.

Usage:
    python backend/data/eval_convergent.py [n_clips]
"""

from __future__ import annotations

import functools
import glob
import json
import os
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import asr  # noqa: E402  (load_audio only)

MODEL_ID = "Dpngtm/wav2vec2-emotion-recognition"
SAMPLE_RATE = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
CLIPS = os.path.join(HERE, "..", "clips")
OUT = os.path.join(RACES, "_convergent_eval.json")

# The second model's 7 categorical labels onto our 4 quadrants. Surprise is
# excluded rather than forced - it is high-arousal but valence-ambiguous, and
# assigning it would be us choosing the answer.
OTHER_TO_QUADRANT = {
    "angry": "Stressed", "anger": "Stressed",
    "fear": "Stressed", "fearful": "Stressed",
    "disgust": "Stressed", "disgusted": "Stressed",
    "happy": "Energised", "happiness": "Energised",
    "sad": "Fatigued", "sadness": "Fatigued",
    "neutral": "Calm",
}
SKIP = {"surprise", "surprised"}
STATES = ["Calm", "Energised", "Stressed", "Fatigued"]


@functools.lru_cache(maxsize=1)
def _load():
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()
    return fe, model


def predict(audio) -> str:
    fe, model = _load()
    inputs = fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return str(model.config.id2label[int(torch.argmax(logits, dim=-1))]).strip().lower()


def cohens_kappa(pairs: list[tuple[str, str]], classes: list[str]) -> float | None:
    """Agreement corrected for what chance alone would produce."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum((ca[c] / n) * (cb[c] / n) for c in classes)
    if expected >= 1.0:
        return None
    return round((observed - expected) / (1 - expected), 4)


def band(k: float | None) -> str:
    if k is None:
        return "not computable"
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def sanity_check_on_gold(n: int = 120) -> dict:
    """Does the *second* model work at all, on data where it should?

    Without this the headline comparison is ambiguous. A kappa near zero has two
    very different explanations - our scoring is noise, or the reference model
    does not transfer to compressed radio - and they demand opposite conclusions.

    Running the same model on CREMA-D separates them. Clean acted speech with
    gold labels is the home turf for a model of this kind: if it scores well
    there and collapses to one class on radio, the failure is domain transfer in
    the *reference*, and the comparison simply cannot adjudicate our scoring. If
    it collapses on CREMA-D too, the model is unusable and the comparison was
    never going to mean anything.
    """
    import io

    import librosa
    from datasets import Audio, load_dataset

    print(f"\nsanity check: second model on CREMA-D ({n} clips)")
    ds = load_dataset("confit/cremad-parquet", split="test").cast_column(
        "audio", Audio(decode=False))

    truth, pred = [], []
    for i, r in enumerate(ds):
        if len(truth) >= n:
            break
        label = str(r.get("emotion", "")).strip().lower()
        if label in SKIP or label not in OTHER_TO_QUADRANT:
            continue
        data = r["audio"].get("bytes")
        if not data:
            continue
        try:
            audio, _ = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE, mono=True)
            out = predict(audio)
        except Exception:
            continue
        if out not in OTHER_TO_QUADRANT:
            continue
        truth.append(OTHER_TO_QUADRANT[label])
        pred.append(OTHER_TO_QUADRANT[out])

    if not truth:
        return {"ran": False}

    acc = sum(1 for a, b in zip(truth, pred) if a == b) / len(truth)
    counts = Counter(pred)
    top_share = max(counts.values()) / len(pred)
    majority = max(Counter(truth).values()) / len(truth)
    # "Degenerate" has to mean *more skewed than the data*, not skewed in
    # absolute terms. Three of CREMA-D's six labels map to Stressed, so the truth
    # itself is ~58% one class and a correct model must look skewed too. A fixed
    # 0.60 threshold flagged a model scoring 81.7% against a 58.3% baseline as
    # broken, which would have printed exactly the wrong conclusion.
    return {
        "ran": True,
        "n": len(truth),
        "accuracy_on_gold": round(acc, 4),
        "majority_baseline": round(majority, 4),
        "beats_baseline": acc > majority,
        "prediction_distribution": dict(counts),
        "largest_class_share": round(top_share, 4),
        "truth_largest_class_share": round(majority, 4),
        "degenerate_on_gold": top_share > majority + 0.15,
    }


def _verdict(observed: float, k: float | None, collapsed: bool,
             top_share: float, sanity: dict) -> str:
    head = f"Raw agreement {observed:.1%}, Cohen's kappa {k} ({band(k)}). "

    if k is not None and k >= 0.20:
        return head + (
            "Two independently trained models land on the same quadrant more often "
            "than chance, which is evidence the signal is in the audio rather than "
            "in one model's quirks."
        )

    if collapsed:
        detail = (
            f"But the reference model is degenerate on radio: it assigns one "
            f"quadrant to {top_share:.0%} of clips. "
        )
        if sanity.get("ran") and sanity.get("beats_baseline") and not sanity.get("degenerate_on_gold"):
            detail += (
                f"On CREMA-D it works normally ({sanity['accuracy_on_gold']:.1%} vs a "
                f"{sanity['majority_baseline']:.1%} baseline, largest class "
                f"{sanity['largest_class_share']:.0%}), so the failure is domain "
                "transfer in the *reference*, not evidence about our scoring. This "
                "comparison cannot adjudicate either way - it is reported because "
                "the attempt was made, not because it settles anything."
            )
        elif sanity.get("ran"):
            detail += (
                f"It is also weak on CREMA-D ({sanity['accuracy_on_gold']:.1%} vs a "
                f"{sanity['majority_baseline']:.1%} baseline), so it was never a "
                "usable reference and this test was uninformative from the start."
            )
        else:
            detail += "The sanity check did not run, so the cause is undetermined."
        return head + detail

    return head + (
        "Agreement is at chance once corrected, and the reference model is not "
        "obviously degenerate, so this does NOT provide independent support for "
        "the affect scoring on radio."
    )


def main(limit: int | None = None) -> None:
    msgs = []
    for path in sorted(glob.glob(os.path.join(RACES, "*.json"))):
        base = os.path.basename(path)
        if base.startswith("_") or base.count(".") > 1:
            continue
        d = json.load(open(path, encoding="utf-8"))
        for m in d["messages"]:
            msgs.append((d["race_id"], m))
    if limit:
        msgs = msgs[::max(1, len(msgs) // limit)][:limit]

    print(f"comparing on {len(msgs)} clips")
    pairs, skipped, t0 = [], 0, time.perf_counter()

    for i, (race_id, m) in enumerate(msgs, 1):
        path = os.path.join(CLIPS, race_id, m["audio_file"])
        if not os.path.exists(path):
            continue
        try:
            audio = asr.load_audio(path)
            other = predict(audio)
        except Exception as e:
            print(f"  !! {m['id']}: {type(e).__name__}: {e}")
            continue
        if other in SKIP or other not in OTHER_TO_QUADRANT:
            skipped += 1
            continue
        pairs.append((m["state"], OTHER_TO_QUADRANT[other]))
        if i % 100 == 0:
            rate = (time.perf_counter() - t0) / i
            print(f"  {i}/{len(msgs)}  ~{rate * (len(msgs) - i) / 60:.1f} min remaining",
                  flush=True)

    if not pairs:
        print("no comparable predictions")
        return

    k = cohens_kappa(pairs, STATES)
    observed = sum(1 for a, b in pairs if a == b) / len(pairs)

    matrix = {s: Counter() for s in STATES}
    for ours, theirs in pairs:
        matrix[ours][theirs] += 1

    # Is the reference model even functioning on this domain?
    theirs_counts = Counter(b for _, b in pairs)
    theirs_top_share = max(theirs_counts.values()) / len(pairs)
    reference_collapsed = theirs_top_share > 0.60
    sanity = sanity_check_on_gold()

    result = {
        "second_model": MODEL_ID,
        "n": len(pairs),
        "skipped_unmappable": skipped,
        "raw_agreement": round(observed, 4),
        "cohens_kappa": k,
        "kappa_band": band(k),
        "ours_distribution": dict(Counter(a for a, _ in pairs)),
        "theirs_distribution": dict(Counter(b for _, b in pairs)),
        "confusion_ours_vs_theirs": {s: dict(matrix[s]) for s in STATES},
        "label_mapping": OTHER_TO_QUADRANT,
        "reference_collapsed_on_radio": reference_collapsed,
        "reference_largest_class_share": round(theirs_top_share, 4),
        "reference_sanity_check": sanity,
        "verdict": _verdict(observed, k, reference_collapsed, theirs_top_share, sanity),
        "why_kappa": (
            "Raw agreement is misleading here: both models over-produce one or two "
            "quadrants, so a large share of agreement happens by accident. Kappa "
            "subtracts the agreement expected from the marginal distributions."
        ),
    }

    json.dump(result, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nraw agreement {observed:.4f} | kappa {k} ({band(k)})")
    print(f"ours:   {result['ours_distribution']}")
    print(f"theirs: {result['theirs_distribution']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
