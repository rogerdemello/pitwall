"""Precompute a whole race: every radio message, analysed and placed on its lap.

Deliberately split into two stages.

  Stage 1 (this file, expensive): run the models over every clip and write the
  RAW outputs to disk. ~25s per clip on CPU, so a race is roughly an hour.

  Stage 2 (calibrate.py, cheap): turn raw outputs into DSI, states and pit
  calls. Because stage 1 is cached, the interpretation layer can be re-tuned in
  seconds instead of re-running an hour of inference.

That split is what makes iteration on the fusion thresholds practical at all.

Usage:
    python backend/data/build_race.py 2021_Abu_Dhabi_Grand_Prix
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import race_data  # noqa: E402
from pipeline import asr, prosody, sentiment  # noqa: E402

CLIP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clips")
RAW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")


def build(race_id: str, limit: int | None = None) -> str:
    clip_dir = os.path.join(CLIP_ROOT, race_id)
    manifest = json.load(open(os.path.join(clip_dir, "manifest.json"), encoding="utf-8"))
    if limit:
        manifest = manifest[:limit]

    os.makedirs(RAW_ROOT, exist_ok=True)
    out_path = os.path.join(RAW_ROOT, f"{race_id}.raw.json")

    # Resume support: an hour-long job should never be restarted from zero.
    done: dict[str, dict] = {}
    if os.path.exists(out_path):
        prev = json.load(open(out_path, encoding="utf-8"))
        # Failed clips are deliberately not treated as done, so a rerun retries them.
        done = {r["id"]: r for r in prev.get("messages", []) if "error" not in r}
        print(f"resuming: {len(done)} already processed")

    print(f"loading FastF1 session for {race_id}...")
    session = race_data.load_session(race_id)

    # Cache lap frames per car number rather than per message.
    lap_cache: dict[str, object] = {}

    def laps_for(num: str):
        if num not in lap_cache:
            lap_cache[num] = race_data.driver_laps(session, num)
        return lap_cache[num]

    results = []
    t_start = time.perf_counter()

    for i, m in enumerate(manifest, 1):
        if m["id"] in done:
            results.append(done[m["id"]])
            continue

        path = os.path.join(clip_dir, m["audio_file"])
        try:
            audio = asr.load_audio(path)
            tr = asr.transcribe(audio)
            af = prosody.analyse(audio)
            se = sentiment.analyse(tr.text)
            lap = race_data.lap_for_timestamp(laps_for(m["racing_number"]), m["message_timestamp"])

            results.append({
                **m,
                "transcript": tr.text,
                "duration_s": tr.duration_s,
                "asr_elapsed_s": tr.elapsed_s,
                "arousal": af.arousal,
                "valence": af.valence,
                "dominance": af.dominance,
                "peak_arousal": af.peak_arousal,
                "min_valence": af.min_valence,
                "text_label": se.label,
                "text_polarity": se.polarity,
                "text_negative": se.negative,
                "text_positive": se.positive,
                "lap": lap.__dict__,
            })
        except Exception as e:  # one bad clip must not lose the whole run
            print(f"  !! {m['id']}: {type(e).__name__}: {e}")
            results.append({**m, "error": f"{type(e).__name__}: {e}"})

        if i % 5 == 0 or i == len(manifest):
            rate = (time.perf_counter() - t_start) / max(1, i - len(done))
            left = rate * (len(manifest) - i)
            print(f"  {i}/{len(manifest)}  ~{left/60:.1f} min remaining")
            json.dump({"race_id": race_id, "messages": results},
                      open(out_path, "w", encoding="utf-8"), indent=1)

    json.dump({"race_id": race_id, "messages": results},
              open(out_path, "w", encoding="utf-8"), indent=1)

    ok = sum(1 for r in results if "error" not in r)
    in_race = sum(1 for r in results if r.get("lap", {}).get("in_race"))
    print(f"\ndone: {ok}/{len(results)} analysed, {in_race} landed on a race lap")
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    race = sys.argv[1] if len(sys.argv) > 1 else "2021_Abu_Dhabi_Grand_Prix"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    build(race, lim)
