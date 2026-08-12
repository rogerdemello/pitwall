"""Fetch clips for several races in ONE pass over the Hub parquet.

fetch_race_clips.py scans the dataset per race. For a slate of five that is five
full scans of 2.57 GB. This filters once with an IN clause and writes each race
to its own directory, which is the difference between minutes and most of an
hour.

Usage:
    python backend/data/fetch_many.py                 # the default slate
    python backend/data/fetch_many.py RACE_A RACE_B
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import duckdb

DATASET_GLOB = "hf://datasets/MikCil/f1-team-radio/**/*.parquet"
CLIP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clips")

# Chosen for contrast, not volume: the index has to prove it discriminates
# between a soaking-wet scramble and a processional dry race.
SLATE = [
    "2020_Turkish_Grand_Prix",    # wet, near-continuous spins - peak stress
    "2023_Monaco_Grand_Prix",     # most messages in the dataset; dry -> wet
    "2019_German_Grand_Prix",     # wet chaos, many retirements
    "2023_Italian_Grand_Prix",    # dry and processional - the low-stress control
    "2023_Qatar_Grand_Prix",      # extreme heat - should light up Fatigued
]

# Held-era slate: nine races from 2023 alone (the six below plus Monaco, Italian
# and Qatar above). The cross-era corpus showed DSI separating races, but it
# spans 2019-2023 and radio encoding is not constant across seasons, so some of
# that separation may be recording era rather than driving conditions. Holding
# the season fixed is the only way to tell.
SEASON_2023 = [
    "2023_Mexico_City_Grand_Prix",
    "2023_Dutch_Grand_Prix",
    "2023_United_States_Grand_Prix",
    "2023_Spanish_Grand_Prix",
    "2023_Miami_Grand_Prix",
    "2023_Japanese_Grand_Prix",
]


def fetch(races: list[str]) -> None:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    placeholders = ", ".join("?" for _ in races)
    print(f"scanning the Hub once for {len(races)} races ...")
    rows = con.execute(
        f"""
        SELECT id, driver_id, racing_number, grand_prix, race_id,
               session_date, message_timestamp, transcription,
               audio.bytes AS audio_bytes, audio.path AS audio_path
        FROM '{DATASET_GLOB}'
        WHERE race_id IN ({placeholders})
        ORDER BY race_id, message_timestamp
        """,
        races,
    ).fetchall()

    by_race: dict[str, list[dict]] = defaultdict(list)
    for (mid, drv, num, gp, rid, sdate, ts, text, audio_bytes, audio_path) in rows:
        out_dir = os.path.join(CLIP_ROOT, rid)
        os.makedirs(out_dir, exist_ok=True)
        fname = audio_path or f"{mid}.mp3"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(audio_bytes)
        by_race[rid].append({
            "id": mid, "driver_id": drv, "racing_number": num,
            "grand_prix": gp, "race_id": rid, "session_date": sdate,
            "message_timestamp": ts, "reference_transcription": text,
            "audio_file": fname,
        })

    for rid, manifest in by_race.items():
        path = os.path.join(CLIP_ROOT, rid, "manifest.json")
        json.dump(manifest, open(path, "w", encoding="utf-8"), indent=2)
        drivers = len({m["driver_id"] for m in manifest})
        print(f"  {rid}: {len(manifest)} clips, {drivers} drivers")

    missing = set(races) - set(by_race)
    if missing:
        print(f"  !! no rows found for: {', '.join(sorted(missing))}")


if __name__ == "__main__":
    fetch(sys.argv[1:] or SLATE)
