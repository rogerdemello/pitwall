---
title: PIT WALL — The Silent Co-Driver
emoji: 🏎️
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Driver stress from F1 radio, synced to lap telemetry
tags:
  - audio
  - speech-emotion-recognition
  - formula-1
  - motorsport
  - speech-recognition
---

# PIT WALL — The Silent Co-Driver

Race engineers miss warning signs in driver radio because they are watching
numbers, not listening to tone. PIT WALL listens for them, places every message
on the real lap it was spoken on, and turns that into a pit-wall decision.

Built for the Grand Prix Hackathon (Problem Statement 1).

## What it does

- **Race Replay** — pick a race and driver; lap time and a Driver State Index sit
  in stacked panels sharing one x-axis. Radio calls are pins on the trace; click
  one to hear it, read it, and see the lap it landed on.
- **Live Analysis** — upload or record a clip and watch the same pipeline run.
- **Evidence** — every number the project claims, read from measured files,
  including the experiments that failed.

## The part worth reading

The central hypothesis — that driver stress predicts lap-time loss — **is not
supported**. Across 556 paired observations from six races, pooled r = 0.047,
and stress does not lead pace loss at any lag. That is on the Evidence screen,
stated plainly.

Three of our own false positives were caught and killed on the way there
(r = 0.62 from n = 10; r = −0.25 at p = 0.06; a negative correlation being
reported as "predictive"), and two planned features were measured and rejected:
ASR vocabulary prompting made word error rate worse in 4 of 6 races, and
acoustic speaker separation cleared its pre-registered bar for only 1 of 4
drivers.

Validated against CREMA-D gold labels, the affect scale recovers **arousal**
(+16 points over baseline) but **valence is at chance** (+1 point) — and DSI
weights valence at 0.45, so nearly half the index rests on a dimension the model
does not resolve. That is the single most important limitation and it is stated
in the product, not buried.

## Models and data

| Role | Asset |
|---|---|
| Data | [`MikCil/f1-team-radio`](https://huggingface.co/datasets/MikCil/f1-team-radio) |
| Prosody | [`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) |
| ASR | [`openai/whisper-small.en`](https://huggingface.co/openai/whisper-small.en) |
| Text sentiment | [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) |
| Validation | [`confit/cremad-parquet`](https://huggingface.co/datasets/confit/cremad-parquet) |

Telemetry via [FastF1](https://docs.fastf1.dev/).

Runs on CPU. Race data is precomputed, so the Replay screen is instant; only
Live Analysis runs inference, and it takes a few seconds per clip on the free
tier.
