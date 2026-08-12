"""Pull all radio clips for one race out of the HF dataset and cache them locally.

Uses duckdb against the Hub's parquet files so we only download the rows we
want, instead of streaming the full 2.57 GB dataset.

Usage:
    python backend/data/fetch_race_clips.py 2021_Abu_Dhabi_Grand_Prix
"""

import json
import os
import sys

import duckdb

DATASET_GLOB = "hf://datasets/MikCil/f1-team-radio/**/*.parquet"
CLIP_ROOT = os.path.join(os.path.dirname(__file__), "..", "clips")


def fetch(race_id: str) -> str:
    out_dir = os.path.join(CLIP_ROOT, race_id)
    os.makedirs(out_dir, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    rows = con.execute(
        f"""
        SELECT id, driver_id, racing_number, grand_prix, race_id,
               session_date, message_timestamp, transcription,
               audio.bytes AS audio_bytes, audio.path AS audio_path
        FROM '{DATASET_GLOB}'
        WHERE race_id = ?
        ORDER BY message_timestamp
        """,
        [race_id],
    ).fetchall()

    manifest = []
    for r in rows:
        (mid, drv, num, gp, rid, sdate, ts, text, audio_bytes, audio_path) = r
        fname = audio_path or f"{mid}.mp3"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            f.write(audio_bytes)
        manifest.append({
            "id": mid,
            "driver_id": drv,
            "racing_number": num,
            "grand_prix": gp,
            "race_id": rid,
            "session_date": sdate,
            "message_timestamp": ts,
            "reference_transcription": text,
            "audio_file": fname,
        })

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"{race_id}: {len(manifest)} clips -> {out_dir}")
    drivers = sorted({m["driver_id"] for m in manifest})
    print(f"  drivers ({len(drivers)}): {', '.join(drivers)}")
    return out_dir


if __name__ == "__main__":
    race = sys.argv[1] if len(sys.argv) > 1 else "2021_Abu_Dhabi_Grand_Prix"
    fetch(race)
