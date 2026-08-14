"""Every number this project publishes, derived from the files that measured it.

The problem this solves: the corpus grew from six races to twelve, and the docs
did not all follow. `space/README.md` and the *published* dataset card still said
"556 paired observations, r = 0.047" against a measured 1,155 and 0.043. README
said 29 tests against 48. README contradicted itself - 1,155 on one line and 556
thirty lines later - because nobody re-reads 385 lines of prose.

Typed numbers drift. Derived numbers cannot. So no document states a figure in
its own right: templates carry placeholders, `render_docs.py` fills them from
here, and `test_docs_numbers.py` fails the build if a rendered file has been
hand-edited or a stale token reappears.

Read-only. Runs in milliseconds. Never imports a model.

Usage:
    python backend/data/facts.py            # print every fact
    python backend/data/facts.py pooled_r   # print one
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.artifacts import iter_race_files, race_ids, sidecar_path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, "..")
RACES = os.path.join(BACKEND, "races")
TESTS = os.path.join(BACKEND, "tests")

#: Tokens that were published, are now wrong, and must never reappear in a doc.
#: Each maps to the fact that supersedes it, so the failure message can say what
#: the number should be rather than only that it is wrong.
RETIRED = {
    "556": "paired_n",
    "0.047": "pooled_r",
    "1,042": "n_messages",
    "29 tests": "n_tests",
    "six races": "n_races",
    # Retired by the CREMA-D re-measurement - the artifacts behind these had been
    # generated before prosody.py gained VAD and per-window scoring.
    "78.1%": "arousal_acc",
    "62.9%": "valence_acc",
    "49.2%": "gold_accuracy",
    "+16.3": "arousal_lift",
    "+0.0605": "valence_lift_at_fitted_boundary",
    "+0.061": "valence_lift_at_fitted_boundary",
}


def _load(name: str) -> dict | None:
    path = os.path.join(RACES, name)
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path, encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def count_tests() -> int | None:
    """Ask pytest for the suite size.

    Deliberately NOT part of `facts()`. It shells out to a pytest collection,
    which makes every caller slow, and - more to the point - an exact test count
    does not belong in prose: it is stale the moment anyone adds a test, so a
    document stating one cannot stay true. The docs describe what the suite
    covers instead. This stays for anyone who wants the number interactively.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", TESTS, "--collect-only", "-q",
             "-p", "no:cacheprovider"],
            capture_output=True, text=True, timeout=120, cwd=BACKEND,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    for line in reversed(out.splitlines()):
        # pytest's summary line, e.g. "61 tests collected in 0.42s"
        if "test" in line and line.split()[:1] and line.split()[0].isdigit():
            return int(line.split()[0])
    return None


def _build_manifest() -> dict:
    """What produced the current stage-1 output, from the bundle that shipped it.

    `backend/raw/_manifest.json` is written by the GPU notebook and verified by
    import_raw_bundle against a checksum, so it records what actually ran rather
    than what a constant in the source currently says. Absent before any import,
    which is why every reader here uses .get().
    """
    path = os.path.join(RACES, "..", "raw", "_manifest.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _asr_ablation() -> dict:
    """The prompting ablation, aggregated across every race that measured it.

    README quoted the showcase race alone (0.2138 vs 0.2157) while DEMO and both
    Space cards quoted "worse in 4 of 6 races". Neither was wrong; they were
    never reconciled. Report both, each labelled with its scope.
    """
    rows = []
    for race_id in race_ids(RACES):
        d = _load(os.path.basename(sidecar_path(RACES, race_id, "asr_eval")))
        if d and d.get("wer_unbiased") is not None:
            rows.append(d)
    if not rows:
        return {"measured": False}
    hurt = [r for r in rows if r["wer_biased"] > r["wer_unbiased"]]
    return {
        "measured": True,
        "races_measured": len(rows),
        "races_hurt_by_prompting": len(hurt),
        "mean_wer_unbiased": round(sum(r["wer_unbiased"] for r in rows) / len(rows), 4),
        "mean_wer_biased": round(sum(r["wer_biased"] for r in rows) / len(rows), 4),
        "sample_per_race": rows[0].get("sample_size"),
        # Which pipeline produced these WERs. The corpus moved to large-v3 and
        # this ablation did not move with it: the prompting A/B only exists on
        # the transformers backend, and re-running it under faster-whisper would
        # score every unbiased hypothesis as an empty string - jiwer returns 1.0
        # for that rather than raising, so the published conclusion would have
        # inverted rather than errored. Exposed as a fact so a document quoting
        # the WER has to say which pipeline measured it.
        "measured_on_pipeline": rows[0].get("measured_on_pipeline", "v1"),
        "measured_on_asr_model": rows[0].get(
            "measured_on_asr_model", "openai/whisper-small.en"),
    }


def facts() -> dict:
    """Every published number, keyed by the placeholder documents use."""
    ca = _load("_corpus_analysis.json") or {}
    gold = _load("_gold_affect_eval.json") or {}
    era = _load("_era_analysis.json") or {}
    conv = _load("_convergent_eval.json") or {}
    finding = _load("_corpus_finding.json") or {}
    diar = _load("_diarization_experiment.json") or {}
    leak = _load("_calibration_leakage.json") or {}
    vdiag = _load("_valence_diagnostic.json") or {}
    build = _build_manifest()
    v1 = _load("_v1_baseline.json") or {}
    bound = _load("_valence_boundary.json") or {}

    svp = ca.get("stress_vs_pace") or {}
    tercile = svp.get("tercile") or {}
    lag = ca.get("lag") or {}
    axes = gold.get("axes") or {}
    arousal = axes.get("arousal_high_vs_low") or {}
    valence = axes.get("valence_negative_vs_positive") or {}
    asr = _asr_ablation()

    ids = race_ids(RACES)
    slate = []
    for path in iter_race_files(RACES):
        d = json.load(open(path, encoding="utf-8"))
        slate.append({
            "race_id": d["race_id"],
            "grand_prix": d["grand_prix"],
            "season": int(d["race_id"][:4]),
            "date": d.get("session_date"),
            "messages": d.get("message_count"),
            "in_race": d.get("in_race_count"),
            "drivers": len(d.get("drivers") or []),
        })

    return {
        # Corpus
        "n_races": len(ids),
        "n_messages": ca.get("messages_pooled") or sum(r["messages"] for r in slate),
        "race_slate": slate,
        "race_ids": ids,

        # The central question, and its null answer
        "paired_n": svp.get("n"),
        "excluded_non_racing": svp.get("excluded_non_racing_laps"),
        "pooled_r": round(svp["pooled_r"], 3) if svp.get("pooled_r") is not None else None,
        "tercile_gap_s": tercile.get("mean_gap_s"),
        "drivers_slower": tercile.get("drivers_slower_when_stressed"),
        "drivers_total": tercile.get("drivers_total"),
        "sign_test_p": tercile.get("sign_test_p"),
        "best_lag": lag.get("best_lag"),
        "best_lag_r": lag.get("best_r"),
        "lag_predictive": lag.get("predictive"),
        "stress_vs_pace_verdict": ca.get("verdict"),

        # Gold-label validation, including the axis that fails
        "gold_dataset": gold.get("dataset"),
        "gold_n": gold.get("n"),
        "gold_accuracy": gold.get("accuracy"),
        "gold_baseline": gold.get("majority_class_baseline"),
        "arousal_acc": arousal.get("accuracy"),
        "arousal_baseline": arousal.get("majority_baseline"),
        "arousal_lift": arousal.get("lift"),
        "valence_acc": valence.get("accuracy"),
        "valence_baseline": valence.get("majority_baseline"),
        "valence_lift": valence.get("lift"),
        # The axis scores at chance *at the 0.5 split*, which is a fact about the
        # threshold rather than about the model: ranking performance and the lift
        # at a fitted cut are both better. Every figure here is read from the
        # boundary artifact and none is restated in this comment, because the
        # last version of it quoted an AUC and a lift that a re-measurement then
        # moved - see _remeasurement.json. Both the pessimistic and the
        # optimistic facts are exposed so a doc cannot quote one alone, which is
        # what the old flat "valence is at chance" claim did.
        "valence_at_chance_at_median_split": (valence.get("lift") or 0) < 0.05,
        "valence_lift_at_fitted_boundary": (
            bound.get("valence_raw_space") or {}).get("lift_over_baseline_fitted"),
        "valence_is_threshold_limited": bound.get("finding") == "boundary_is_misplaced_on_gold",
        "valence_boundary_recommendation": bound.get("recommendation"),
        "valence_boundary_transfers": (
            bound.get("transfer_check") or {}).get("distributions_comparable"),
        "valence_domain_shift_sds": (
            bound.get("transfer_check") or {}).get("median_shift_in_gold_sds"),

        # Convergent validity
        "convergent_model": conv.get("second_model"),
        "convergent_n": conv.get("n"),
        "convergent_kappa": conv.get("cohens_kappa"),
        "convergent_band": conv.get("kappa_band"),

        # Cross-race separation
        "finding_verdict": finding.get("verdict"),
        "prereg_races": (finding.get("setup") or {}).get("races"),
        "prereg_held": sum(1 for v in finding.get("prediction_scorecard", []) if v["held"]),
        "prereg_total": len(finding.get("prediction_scorecard", [])),
        "within_season_spread": era.get("within_season_spread"),
        "cross_era_spread": era.get("cross_era_spread"),
        "era_races_2023": era.get("n_2023"),
        "era_confound_resolved": finding.get("confound_status") == "resolved",

        # ASR
        "asr_races_measured": asr.get("races_measured"),
        "asr_races_hurt": asr.get("races_hurt_by_prompting"),
        "asr_mean_wer": asr.get("mean_wer_unbiased"),
        "asr_mean_wer_prompted": asr.get("mean_wer_biased"),
        "asr_sample_per_race": asr.get("sample_per_race"),
        "asr_ablation_pipeline": asr.get("measured_on_pipeline"),
        "asr_ablation_model": asr.get("measured_on_asr_model"),
        # What actually built the shipped corpus, read from the build manifest
        # rather than from pipeline.asr - this module imports no model, and the
        # manifest is the better source anyway: it records what ran, not what
        # the constant currently says.
        "corpus_asr_model": build.get("asr_model"),
        "corpus_asr_backend": build.get("asr_backend"),
        "corpus_built_on_gpu": build.get("gpu"),

        # Calibration: is it held out, and what was the leak worth?
        "calibration_scheme": "leave-one-race-out",
        "leakage_ordering_preserved": leak.get("ordering_preserved"),
        "leakage_mean_abs_delta": (leak.get("per_message") or {}).get("mean_abs_delta"),
        "leakage_verdict": leak.get("verdict"),
        "leakage_spread_in_sample": leak.get("spread_in_sample"),
        "leakage_spread_held_out": leak.get("spread_held_out"),

        # Why acoustic valence was kept or removed. The stratified figure is the
        # one the verdict rests on: the naive contrast turned out not to be
        # matched on arousal, so it measures partly the axis it meant to hold
        # constant. Both are exposed so a doc can never quote only the flattering
        # one.
        "valence_diagnostic_verdict": vdiag.get("verdict"),
        "valence_diagnostic_auc": (
            ((vdiag.get("contrasts") or {}).get("happy_vs_anger") or {})
            .get("valence_arousal_stratified") or {}).get("pooled_auc"),
        "valence_diagnostic_auc_naive": (
            ((vdiag.get("contrasts") or {}).get("happy_vs_anger") or {})
            .get("valence_naive") or {}).get("auc"),
        "valence_contrast_was_matched": (
            (vdiag.get("contrasts") or {}).get("happy_vs_anger") or {}
        ).get("contrast_is_matched"),

        # v1 defects the rebuild has to beat
        "v1_clips_truncated": (v1.get("truncation") or {}).get("clips_truncated"),
        "v1_audio_lost_min": (v1.get("truncation") or {}).get("audio_never_transcribed_min"),
        "v1_hallucination_rate": (v1.get("hallucination") or {}).get("rate"),
        "v1_unknown_speaker_share": (
            v1.get("speaker_attribution") or {}).get("unknown_share"),

        # Rejected experiments
        "diarization_verdict": diar.get("verdict"),

    }


def main() -> None:
    f = facts()
    if len(sys.argv) > 1:
        for key in sys.argv[1:]:
            if key not in f:
                print(f"!! no such fact: {key}", file=sys.stderr)
                raise SystemExit(1)
            print(f[key])
        return

    missing = [k for k, v in f.items() if v is None]
    width = max(len(k) for k in f)
    for k, v in f.items():
        if k == "race_slate":
            v = f"{len(v)} races"
        elif isinstance(v, list):
            v = f"[{len(v)} items]"
        print(f"{k:<{width}}  {v}")
    if missing:
        print(f"\n!! {len(missing)} fact(s) unmeasured: {', '.join(missing)}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
