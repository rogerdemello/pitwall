"""Stage 2: raw model outputs -> calibrated race JSON the frontend can serve.

Cheap and re-runnable. All the expensive inference happened in build_race.py, so
fusion thresholds can be re-tuned and this re-run in about a second.

Usage:
    python backend/data/calibrate.py 2021_Abu_Dhabi_Grand_Prix
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import fusion, roster, speaker, strategy  # noqa: E402
from pipeline.calibration import Calibrator  # noqa: E402
from pipeline.prosody import Affect  # noqa: E402
from pipeline.sentiment import TextSentiment  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(HERE, "..", "raw")
OUT_ROOT = os.path.join(HERE, "..", "races")

# Driver identity comes from pipeline/roster.json, generated from FastF1 by
# build_roster.py. It used to be a hand-typed map here that covered 22 of the 30
# drivers in the corpus; the rest fell through to a driver_id[:3] fallback, so
# 479 messages rendered a driver called "NICHUL01" with the code "NIC".
display = roster.display


def _lap_traces(session, numbers: dict[str, str]) -> dict[str, list[dict]]:
    """Every completed lap for each driver, for the pace chart."""
    import pandas as pd

    from data import race_data

    traces: dict[str, list[dict]] = {}
    for driver_id, number in numbers.items():
        dl = race_data.driver_laps(session, number)
        rows = []
        for _, lap in dl.iterrows():
            if pd.isna(lap["LapTime"]):
                continue
            rows.append({
                "lap": int(lap["LapNumber"]),
                "seconds": round(lap["LapTime"].total_seconds(), 3),
                "compound": lap["Compound"] if pd.notna(lap["Compound"]) else None,
                "tyre_life": float(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) else None,
                "position": float(lap["Position"]) if pd.notna(lap["Position"]) else None,
            })
        traces[driver_id] = sorted(rows, key=lambda r: r["lap"])
    return traces


def run(race_id: str) -> str:
    from data import race_data

    raw_path = os.path.join(RAW_ROOT, f"{race_id}.raw.json")
    raw = json.load(open(raw_path, encoding="utf-8"))
    msgs = [m for m in raw["messages"] if "error" not in m]

    # Recompute the lap join here rather than trusting whatever build_race.py
    # wrote. The join needs no model inference, so it costs nothing to redo, and
    # this lets the lap-context logic evolve without re-running an hour of GPU-
    # less transcription.
    session = race_data.load_session(race_id)
    lap_frames: dict[str, object] = {}
    for m in msgs:
        num = m["racing_number"]
        if num not in lap_frames:
            lap_frames[num] = race_data.driver_laps(session, num)
        m["lap"] = race_data.lap_for_timestamp(lap_frames[num], m["message_timestamp"]).__dict__

    # Prefer the pooled calibration when it exists. Fitting per-race centres every
    # race on 50.0 by construction, which makes races look identical and hides
    # exactly the wet-versus-dry contrast the corpus was built to expose.
    pooled_path = os.path.join(OUT_ROOT, "_pooled.calibration.json")
    if os.path.exists(pooled_path):
        cal = Calibrator.from_json(pooled_path)
        cal_source = "pooled"
    else:
        cal = Calibrator.fit(msgs)
        cal_source = "per-race"
    # Always keep the per-race view for comparison on the Evidence screen.
    own = Calibrator.fit(msgs)

    analysed = []
    for m in msgs:
        affect = Affect(
            arousal=m["arousal"], dominance=m["dominance"], valence=m["valence"],
            peak_arousal=m.get("peak_arousal", 0.0), min_valence=m.get("min_valence", 0.0),
        )
        text = TextSentiment(
            label=m["text_label"], negative=m["text_negative"],
            neutral=max(0.0, 1 - m["text_negative"] - m["text_positive"]),
            positive=m["text_positive"],
        )
        state = fusion.fuse(affect, text, calibrator=cal, transcript=m["transcript"])
        who, why = speaker.classify(m["transcript"], m.get("driver_id"))
        code, name = display(m["driver_id"])
        analysed.append({
            "id": m["id"],
            "driver_id": m["driver_id"],
            "driver_code": code,
            "driver_name": name,
            "racing_number": m["racing_number"],
            "timestamp": m["message_timestamp"],
            "audio_file": m["audio_file"],
            "transcript": m["transcript"],
            "reference_transcription": m["reference_transcription"],
            "duration_s": m["duration_s"],
            "lap": m["lap"],
            "speaker": who,
            "speaker_reason": why,
            **state.to_dict(),
        })

    analysed.sort(key=lambda r: r["timestamp"])

    # Per-driver strategy calls, replaying history as it would have arrived.
    by_driver: dict[str, list[dict]] = {}
    for row in analysed:
        hist = by_driver.setdefault(row["driver_id"], [])
        hist.append(row)
        # An engineer transmission is not a claim about the driver's state, so it
        # never triggers a recommendation of its own.
        if row["speaker"] == "engineer":
            row["recommendation"] = None
            row["suppressed_stress"] = False
            continue
        rec = strategy.recommend(row["driver_code"], hist)
        row["recommendation"] = rec.to_dict() if rec else None

    drivers = []
    for did, rows in sorted(by_driver.items(), key=lambda kv: -len(kv[1])):
        in_race = [r for r in rows if r["lap"].get("in_race")]
        code, name = display(did)
        drivers.append({
            "driver_id": did,
            "code": code,
            "name": name,
            "racing_number": rows[0]["racing_number"],
            "message_count": len(rows),
            "in_race_count": len(in_race),
            "mean_dsi": round(sum(r["dsi"] for r in in_race) / len(in_race), 1) if in_race else None,
            "peak_dsi": max((r["dsi"] for r in in_race), default=None),
            "suppressed_count": sum(1 for r in rows if r["suppressed_stress"]),
        })

    # Full lap trace per driver. Without this the pace chart could only connect
    # laps that happen to carry a radio message, which draws a jagged line that
    # misrepresents the race.
    lap_traces = _lap_traces(session, {d["driver_id"]: d["racing_number"] for d in drivers})

    out = {
        "race_id": race_id,
        "grand_prix": msgs[0]["grand_prix"] if msgs else race_id,
        "session_date": msgs[0]["session_date"] if msgs else None,
        "message_count": len(analysed),
        "in_race_count": sum(1 for r in analysed if r["lap"].get("in_race")),
        "drivers": drivers,
        "lap_traces": lap_traces,
        "calibration": cal.summary(),
        "calibration_source": cal_source,
        "calibration_own_race": own.summary(),
        "messages": analysed,
    }

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, f"{race_id}.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1)
    cal.to_json(os.path.join(OUT_ROOT, f"{race_id}.calibration.json"))

    flagged = sum(1 for r in analysed if r["suppressed_stress"])
    dsis = [r["dsi"] for r in analysed]
    print(f"{race_id}: {len(analysed)} messages, {out['in_race_count']} on-lap")
    print(f"  DSI range {min(dsis)}-{max(dsis)}  mean {sum(dsis)/len(dsis):.1f}")
    print(f"  suppressed-stress flags: {flagged} ({flagged/len(analysed)*100:.1f}%)")
    print(f"  wrote {out_path}")
    return out_path


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "2021_Abu_Dhabi_Grand_Prix")
