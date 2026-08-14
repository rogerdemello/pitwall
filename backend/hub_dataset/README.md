---
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

2042 radio messages from 12 Grand Prix, each scored for vocal
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
