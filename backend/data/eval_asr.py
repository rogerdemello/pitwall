"""Measure the ASR claim instead of asserting it.

Compares, on the same sample of real clips:
    unbiased  - whisper-small.en, no prompt
    biased    - whisper-small.en + the short F1 prompt

against the dataset's published transcripts.

Important caveat, and the reason two numbers are reported: those published
transcripts are themselves imperfect. They contain systematic F1-jargon errors
("supersoft" -> "SuperSalt", "Vandoorne" -> "Van der Waal"). So overall WER
against them measures agreement with a noisy baseline, not correctness. The
jargon-recovery count is the more meaningful figure: how often each variant
produces a domain term that the reference missed or mangled.

Usage:
    python backend/data/eval_asr.py 2021_Abu_Dhabi_Grand_Prix 40
"""

from __future__ import annotations

import json
import os
import re
import sys

import jiwer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import asr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP_ROOT = os.path.join(HERE, "..", "clips")
OUT_ROOT = os.path.join(HERE, "..", "races")

# Terms whose correct transcription is the whole point of domain biasing.
JARGON_TERMS = [
    "drs", "ers", "tyre", "tyres", "supersoft", "ultrasoft", "hypersoft",
    "inters", "intermediates", "undercut", "overcut", "deg", "degradation",
    "graining", "box", "safety car", "vsc", "delta", "stint", "pit",
    "understeer", "oversteer", "front wing", "brake bias",
]

NORMALISE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def jargon_hits(text: str) -> set[str]:
    t = " " + re.sub(r"[^\w\s]", " ", text.lower()) + " "
    return {j for j in JARGON_TERMS if f" {j} " in t}


def run(race_id: str, n: int = 40) -> None:
    clip_dir = os.path.join(CLIP_ROOT, race_id)
    manifest = json.load(open(os.path.join(clip_dir, "manifest.json"), encoding="utf-8"))

    # Skip clips whose reference is too short to score meaningfully.
    pool = [m for m in manifest if len(m["reference_transcription"].split()) >= 3]
    # Even spread across the session rather than the first N (which are all pre-race).
    step = max(1, len(pool) // n)
    sample = pool[::step][:n]

    print(f"evaluating {len(sample)} clips from {race_id}\n")

    refs, hyp_b, hyp_u = [], [], []
    recovered_b = recovered_u = 0
    rows = []

    for i, m in enumerate(sample, 1):
        audio = asr.load_audio(os.path.join(clip_dir, m["audio_file"]))
        tr = asr.transcribe(audio, bias=True, ab=True)
        ref = m["reference_transcription"]

        refs.append(ref)
        hyp_b.append(tr.text)
        hyp_u.append(tr.text_unbiased or "")

        ref_j = jargon_hits(ref)
        b_j, u_j = jargon_hits(tr.text), jargon_hits(tr.text_unbiased or "")
        recovered_b += len(ref_j & b_j)
        recovered_u += len(ref_j & u_j)

        rows.append({
            "id": m["id"], "reference": ref,
            "biased": tr.text, "unbiased": tr.text_unbiased,
            "ref_jargon": sorted(ref_j),
            "biased_jargon": sorted(b_j), "unbiased_jargon": sorted(u_j),
        })
        if i % 10 == 0:
            print(f"  {i}/{len(sample)}")

    def wer(hyps):
        # jiwer renamed truth_transform -> reference_transform in 3.x.
        try:
            return jiwer.wer(refs, hyps,
                             reference_transform=NORMALISE, hypothesis_transform=NORMALISE)
        except TypeError:
            return jiwer.wer(refs, hyps,
                             truth_transform=NORMALISE, hypothesis_transform=NORMALISE)

    wer_b, wer_u = wer(hyp_b), wer(hyp_u)
    total_ref_j = sum(len(jargon_hits(r)) for r in refs)

    result = {
        "race_id": race_id,
        "sample_size": len(sample),
        "wer_unbiased": round(wer_u, 4),
        "wer_biased": round(wer_b, 4),
        "wer_delta": round(wer_b - wer_u, 4),
        "jargon_terms_in_reference": total_ref_j,
        "jargon_recovered_unbiased": recovered_u,
        "jargon_recovered_biased": recovered_b,
        "jargon_recall_unbiased": round(recovered_u / total_ref_j, 4) if total_ref_j else None,
        "jargon_recall_biased": round(recovered_b / total_ref_j, 4) if total_ref_j else None,
        "note": (
            "WER is measured against the dataset's published transcripts, which "
            "themselves contain F1-jargon errors. Treat WER as agreement with a "
            "noisy baseline; jargon recall is the more meaningful signal."
        ),
        "examples": rows[:25],
    }

    os.makedirs(OUT_ROOT, exist_ok=True)
    out = os.path.join(OUT_ROOT, f"{race_id}.asr_eval.json")
    json.dump(result, open(out, "w", encoding="utf-8"), indent=1)

    print(f"\n  WER unbiased : {wer_u:.4f}")
    print(f"  WER biased   : {wer_b:.4f}   ({wer_b - wer_u:+.4f})")
    print(f"  jargon recall: {recovered_u}/{total_ref_j} unbiased -> "
          f"{recovered_b}/{total_ref_j} biased")
    print(f"  wrote {out}")


if __name__ == "__main__":
    race = sys.argv[1] if len(sys.argv) > 1 else "2021_Abu_Dhabi_Grand_Prix"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    run(race, n)
