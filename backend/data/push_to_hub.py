"""Package the analysed corpus as a Hugging Face dataset and push it.

The project consumes four things from the Hub. This gives something back: every
radio message we analysed, with its affect scores, driver state, speaker
attribution and the real lap it was joined to - which is the part nobody else
has, because the join is the work.

Audio is deliberately NOT redistributed. It belongs to the source dataset
(MikCil/f1-team-radio, CC BY 4.0); we ship the analysis and a message `id` that
points back at it.

Requires an authenticated account:
    hf auth login
    python backend/data/push_to_hub.py

Defaults to DEFAULT_REPO below. Pass a different id to override:
    python backend/data/push_to_hub.py someone-else/pitwall-f1-radio-analysis

Dry run (writes the files locally, pushes nothing):
    python backend/data/push_to_hub.py --dry-run
"""

from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
OUT_DIR = os.path.join(HERE, "..", "hub_dataset")

DEFAULT_REPO = "rogerdemello/pitwall-f1-radio-analysis"

CARD = """---
license: cc-by-4.0
task_categories:
  - audio-classification
language:
  - en
tags:
  - formula-1
  - motorsport
  - speech-emotion-recognition
  - affective-computing
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/messages.parquet
---

# PIT WALL - F1 team radio, analysed and joined to lap telemetry

{n_messages} radio messages from {n_races} Grand Prix, each scored for vocal
affect and **joined to the exact lap it was spoken on**.

Produced for the Grand Prix Hackathon. Source audio is
[`MikCil/f1-team-radio`](https://huggingface.co/datasets/MikCil/f1-team-radio)
(CC BY 4.0) and is **not** redistributed here - each row carries the source `id`
so you can join back to it. What is new is the analysis and the telemetry join.

## Why the join matters

The source dataset has a UTC `message_timestamp`; [FastF1](https://docs.fastf1.dev/)
gives every lap an absolute start time. Intersecting them puts each message on a
real lap with its lap time, tyre compound and stint age.

It validates itself. On the 2018 Australian GP a "Box, box, box" call lands on
lap 1 (SUPERSOFT, stint 1) and the next message on lap 2 (SOFT, stint 2) - the
telemetry independently confirms the stop the radio ordered.

## Fields

| Field | Meaning |
|---|---|
| `id`, `race_id`, `driver_code`, `timestamp` | keys back to the source dataset |
| `transcript` | our ASR output (`openai/whisper-large-v3`) |
| `arousal_raw`, `valence_raw` | raw dimensional affect (`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`) |
| `arousal_pct`, `valence_pct`, `dominance_pct` | percentile-calibrated against this corpus |
| `dsi` | Driver State Index, 0-100 |
| `state` | Calm / Energised / Stressed / Fatigued |
| `speaker` | driver / engineer / unknown (grammatical-direction heuristic) |
| `suppressed_stress` | words read positive, voice does not |
| `lap_number`, `lap_time_s`, `delta_to_median_s`, `compound`, `tyre_life`, `is_representative` | the telemetry join |

`is_representative` marks green-flag laps with no pit entry or exit. **Use it for
any pace analysis.** Without it a 20-second pit stop reads as a 20-second "stress
effect" - our first run reported exactly that.

## Read this before using `dsi`

Validated against CREMA-D gold labels, the underlying model recovers **arousal**
(79.4% vs a 61.8% baseline) but **valence is at chance** (60.9% vs 61.9%). DSI
weights valence at 0.45, so close to half the index rests on a dimension the
model does not resolve. Concretely: **Stressed vs Energised, and Calm vs
Fatigued, differ only in valence and should be treated as unreliable.** The
arousal axis is the part that holds up.

## The null result

We built this to test whether driver stress predicts lap-time loss. Across 1155
paired observations it does not: pooled r = 0.026, and no lag from +1 to +3 beats
the same-lap correlation once corrected for multiple comparisons. Published
anyway, because a null result with this much data is worth more than a
correlation manufactured by leaving pit stops in.

## Caveats

- Percentile calibration is pooled across these races; DSI is relative to this
  corpus, not absolute.
- Radio carries the engineer as well as the driver. `speaker` is a text
  heuristic that answers *unknown* rather than guessing; acoustic separation was
  attempted and rejected (cleared its pre-registered bar for 1 of 4 drivers).
- Transcripts are ASR output, not gold. Measured WER around 0.20 against the
  source dataset's own transcripts, which themselves contain F1-jargon errors.
- Races span 2019-2023, and radio encoding is not constant across seasons.
"""


def collect() -> tuple[list[dict], list[str]]:
    rows, races = [], []
    for path in sorted(glob.glob(os.path.join(RACES, "*.json"))):
        base = os.path.basename(path)
        if base.startswith("_") or base.count(".") > 1:
            continue
        d = json.load(open(path, encoding="utf-8"))
        races.append(d["race_id"])
        for m in d["messages"]:
            lap = m.get("lap") or {}
            rows.append({
                "id": m["id"],
                "race_id": d["race_id"],
                "grand_prix": d["grand_prix"],
                "driver_code": m["driver_code"],
                "driver_name": m["driver_name"],
                "timestamp": m["timestamp"],
                "duration_s": m["duration_s"],
                "transcript": m["transcript"],
                "arousal_raw": m["arousal_raw"],
                "valence_raw": m["valence_raw"],
                "arousal_pct": m["arousal_pct"],
                "valence_pct": m["valence_pct"],
                "dominance_pct": m["dominance_pct"],
                "text_polarity": m["text_polarity"],
                "dsi": m["dsi"],
                "state": m["state"],
                "speaker": m.get("speaker"),
                "suppressed_stress": m["suppressed_stress"],
                "in_race": lap.get("in_race", False),
                "lap_number": lap.get("lap_number"),
                "lap_time_s": lap.get("lap_time_s"),
                "delta_to_median_s": lap.get("delta_to_median_s"),
                "compound": lap.get("compound"),
                "tyre_life": lap.get("tyre_life"),
                "is_representative": lap.get("is_representative", False),
            })
    return rows, races


def main(repo_id: str | None, dry_run: bool) -> None:
    rows, races = collect()
    if not rows:
        print("no analysed races found - build the corpus first")
        return

    os.makedirs(os.path.join(OUT_DIR, "data"), exist_ok=True)

    # Parquet, with an explicit schema, is what the Hub's dataset viewer wants.
    # Shipping JSONL alone left the viewer erroring with "the information about
    # the size of the dataset is not coherent": it has to infer types, and
    # columns like `lap_number` are legitimately null for every pre-race message,
    # which makes that inference unreliable. Nullable dtypes state the intent
    # instead of leaving it to be guessed.
    import pandas as pd

    df = pd.DataFrame(rows)
    for col, dtype in {
        "lap_number": "Int64",
        "tyre_life": "Float64",
        "lap_time_s": "Float64",
        "delta_to_median_s": "Float64",
    }.items():
        if col in df:
            df[col] = df[col].astype(dtype)

    parquet = os.path.join(OUT_DIR, "data", "messages.parquet")
    df.to_parquet(parquet, index=False)

    # JSONL kept alongside it: parquet drives the viewer, but a plain-text copy
    # is far easier to eyeball or grep.
    jsonl = os.path.join(OUT_DIR, "messages.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    card = CARD.format(n_messages=len(rows), n_races=len(races))
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    print(f"prepared {len(rows)} rows from {len(races)} races in {OUT_DIR}")

    target = repo_id or DEFAULT_REPO

    if dry_run:
        print(f"\ndry run - nothing pushed. Target would be: {target}")
        print("To publish:")
        print("  hf auth login")
        print("  python backend/data/push_to_hub.py")
        return

    from huggingface_hub import HfApi
    api = HfApi()

    # Fail with something readable rather than a raw auth traceback.
    try:
        who = api.whoami()
    except Exception:
        print("Not logged in. Run `hf auth login` first (needs a write token from")
        print("https://huggingface.co/settings/tokens), then re-run this.")
        return

    owner = target.split("/")[0]
    if who.get("name") != owner and owner not in {
        o.get("name") for o in who.get("orgs", [])
    }:
        print(f"Logged in as '{who.get('name')}' but the target repo is owned by "
              f"'{owner}'. Pass a repo id you can write to.")
        return

    api.create_repo(target, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=OUT_DIR, repo_id=target, repo_type="dataset")
    print(f"pushed to https://huggingface.co/datasets/{target}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    main(args[0] if args else None, "--dry-run" in sys.argv)
