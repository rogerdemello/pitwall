"""How much was the in-sample calibration worth?

Every DSI this project published before now was an in-sample score. The pooled
percentile mapping was fitted on all 2,042 messages and then applied to those
same 2,042 messages, so each message helped set the scale it was measured
against. "Top fifth of F1 radio" was a claim about the fitting set.

Leave-one-race-out fixes it: each race is scored against a mapping fitted on the
other eleven. But fixing a leak silently is only half the work - the question a
reader should ask is *how much did the leak inflate things*, and that is
answerable by scoring the same messages both ways.

Two outcomes, both worth having:

  the numbers barely move  ->  a robustness result. The corpus is large enough
                               that one race out of twelve does not shift the
                               reference distribution, and every previously
                               published contrast stands.

  the ordering changes     ->  a correction, and it must be published as one.

The race ordering matters more than the absolute values here, because the
project's cross-race claim is that DSI separates races that were different to
drive. If that ordering survives an out-of-sample rescoring it is a real result;
if it does not, it was an artifact of the fitting.

Usage:
    python backend/data/eval_calibration_leakage.py
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import pool_calibration  # noqa: E402
from pipeline import fusion, stats  # noqa: E402
from pipeline.artifacts import iter_race_files  # noqa: E402
from pipeline.calibration import Calibrator  # noqa: E402
from pipeline.prosody import Affect  # noqa: E402
from pipeline.sentiment import TextSentiment  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
RAW = os.path.join(HERE, "..", "raw")
OUT = os.path.join(RACES, "_calibration_leakage.json")


def _score(msgs: list[dict], cal: Calibrator) -> list[int]:
    """DSI for every raw message under a given calibration."""
    out = []
    for m in msgs:
        affect = Affect(arousal=m["arousal"], dominance=m["dominance"],
                        valence=m["valence"])
        text = TextSentiment(
            label=m["text_label"], negative=m["text_negative"],
            neutral=max(0.0, 1 - m["text_negative"] - m["text_positive"]),
            positive=m["text_positive"])
        out.append(fusion.fuse(affect, text, calibrator=cal,
                               transcript=m["transcript"]).dsi)
    return out


def main() -> None:
    pooled_path = os.path.join(RACES, "_pooled.calibration.json")
    if not os.path.exists(pooled_path):
        print("no pooled calibration - run pool_calibration.py first")
        return
    pooled = Calibrator.from_json(pooled_path)

    # Which messages count as driver-attributed, taken from the built race files
    # so this matches exactly what the published race means are computed over.
    speakers: dict[str, str] = {}
    for path in iter_race_files(RACES):
        for m in json.load(open(path, encoding="utf-8"))["messages"]:
            speakers[m["id"]] = m.get("speaker", "unknown")

    rows, all_deltas = [], []
    for path in sorted(os.listdir(RAW)):
        if not path.endswith(".raw.json") or ".bak." in path:
            continue
        data = json.load(open(os.path.join(RAW, path), encoding="utf-8"))
        race_id = data["race_id"]
        msgs = [m for m in data["messages"]
                if "error" not in m and m.get("arousal") is not None]
        if not msgs:
            continue

        loro_path = pool_calibration.loro_path(race_id)
        if not os.path.exists(loro_path):
            print(f"  !! no held-out mapping for {race_id}; run pool_calibration.py")
            continue
        loro = Calibrator.from_json(loro_path)

        in_sample = _score(msgs, pooled)
        held_out = _score(msgs, loro)
        deltas = [h - i for h, i in zip(held_out, in_sample)]
        all_deltas.extend(deltas)

        # Race means are over driver-attributed messages only, as published.
        keep = [i for i, m in enumerate(msgs)
                if speakers.get(m["id"], "unknown") != "engineer"]
        rows.append({
            "race_id": race_id,
            "race": data.get("grand_prix", race_id).replace(" Grand Prix", ""),
            "n": len(keep),
            "mean_in_sample": round(st.fmean(in_sample[i] for i in keep), 2),
            "mean_held_out": round(st.fmean(held_out[i] for i in keep), 2),
            "mean_shift": round(st.fmean(held_out[i] for i in keep)
                                - st.fmean(in_sample[i] for i in keep), 2),
            "mean_abs_delta": round(st.fmean(abs(d) for d in deltas), 2),
            "max_abs_delta": max(abs(d) for d in deltas),
        })

    if not rows:
        print("nothing to compare")
        return

    by_in = [r["race_id"] for r in sorted(rows, key=lambda r: -r["mean_in_sample"])]
    by_out = [r["race_id"] for r in sorted(rows, key=lambda r: -r["mean_held_out"])]

    # Comparing the two orderings position-by-position counts tie-breaking as a
    # reordering, which is not a finding: two races whose means are equal to the
    # reported precision will swap on floating-point noise. So a swap only counts
    # when the pair is separated by more than the precision the means are
    # published at - otherwise this would manufacture exactly the kind of false
    # positive the rest of this project exists to catch.
    TIE = 0.15  # DSI points; means are published to one decimal
    in_mean = {r["race_id"]: r["mean_in_sample"] for r in rows}
    out_mean = {r["race_id"]: r["mean_held_out"] for r in rows}
    ids = [r["race_id"] for r in rows]

    real_swaps, tied_swaps = [], []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d_in = in_mean[a] - in_mean[b]
            d_out = out_mean[a] - out_mean[b]
            if (d_in > 0) == (d_out > 0):
                continue
            pair = f"{a} vs {b}"
            if abs(d_in) <= TIE and abs(d_out) <= TIE:
                tied_swaps.append(pair)
            else:
                real_swaps.append(
                    {"pair": pair, "in_sample_diff": round(d_in, 2),
                     "held_out_diff": round(d_out, 2)})

    ordering_preserved = not real_swaps
    spread_in = max(in_mean.values()) - min(in_mean.values())
    spread_out = max(out_mean.values()) - min(out_mean.values())

    verdict = (
        "The leak was not material. Scoring every message against a calibration "
        "fitted without its own race moves race means by at most "
        f"{max(abs(r['mean_shift']) for r in rows)} DSI points and no message by "
        f"more than {max(abs(d) for d in all_deltas)}. No pair of races separated "
        "by more than the reported precision changes order, so every previously "
        "published cross-race contrast stands - and now stands out of sample."
        + (f" {len(tied_swaps)} pair(s) swap within the tie threshold of {TIE} "
           f"points ({'; '.join(tied_swaps)}); those races were indistinguishable "
           "in both orderings and no claim rested on their order."
           if tied_swaps else "")
        if ordering_preserved else
        "The leak was material: "
        + "; ".join(f"{s['pair']} changes order ({s['in_sample_diff']:+.2f} in "
                    f"sample, {s['held_out_diff']:+.2f} held out)"
                    for s in real_swaps)
        + ". The in-sample ordering was partly an artifact of the fitting, and "
          "the affected contrasts are corrected rather than restated."
    )

    payload = {
        "generated_by": "backend/data/eval_calibration_leakage.py",
        "question": "How much did fitting the calibration in-sample inflate the results?",
        "method": (
            "Score the identical messages twice: once against the pooled mapping "
            "fitted on all 12 races (in-sample, as previously published), once "
            "against a mapping fitted on the other 11 (held out). Race means are "
            "over driver-attributed messages only, matching the published figures."
        ),
        "n_messages": len(all_deltas),
        "per_message": {
            "mean_abs_delta": round(st.fmean(abs(d) for d in all_deltas), 3),
            "max_abs_delta": max(abs(d) for d in all_deltas),
            "share_moving_more_than_2": round(
                sum(1 for d in all_deltas if abs(d) > 2) / len(all_deltas), 4),
        },
        "per_race": sorted(rows, key=lambda r: -r["mean_held_out"]),
        "ordering_preserved": ordering_preserved,
        "ordering_in_sample": by_in,
        "ordering_held_out": by_out,
        "material_swaps": real_swaps,
        "swaps_within_tie_threshold": tied_swaps,
        "tie_threshold_dsi": TIE,
        "spread_in_sample": round(spread_in, 2),
        "spread_held_out": round(spread_out, 2),
        "verdict": verdict,
        "note": (
            "The pooled mapping is retained for the live-upload path, where there "
            "is no race to hold out - an uploaded clip is genuinely unseen, so "
            "pooled is the correct reference there."
        ),
    }

    print(f"{'race':<20}{'in-sample':>11}{'held-out':>10}{'shift':>8}"
          f"{'mean|d|':>9}{'max|d|':>8}")
    for r in payload["per_race"]:
        print(f"{r['race']:<20}{r['mean_in_sample']:>11.1f}{r['mean_held_out']:>10.1f}"
              f"{r['mean_shift']:>+8.2f}{r['mean_abs_delta']:>9.2f}{r['max_abs_delta']:>8}")
    p = payload["per_message"]
    print(f"\nper message: mean |delta| {p['mean_abs_delta']}, max {p['max_abs_delta']}, "
          f"{p['share_moving_more_than_2'] * 100:.1f}% move more than 2 points")
    print(f"spread {spread_in:.1f} -> {spread_out:.1f}")
    print(f"ordering preserved: {ordering_preserved}")
    print(f"\nVERDICT: {verdict}")

    json.dump(payload, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
