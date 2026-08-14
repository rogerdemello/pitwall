---
title: PIT WALL — The Silent Co-Driver
emoji: 🏎️
colorFrom: gray
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Driver stress from F1 radio, synced to lap telemetry
tags:
  - formula-1
  - motorsport
  - speech-emotion-recognition
---

# PIT WALL — The Silent Co-Driver

Race engineers miss warning signs in driver radio because they are watching
numbers, not listening to tone. PIT WALL listens for them, places every message
on the real lap it was spoken on, and turns that into a pit-wall decision.

**2,042 radio messages across 12 Grands Prix**, each scored for vocal affect and
joined to real lap telemetry.

Built for the Grand Prix Hackathon (Problem Statement 1).

## Where to start

**Race Replay → Abu Dhabi 2021 → SAI, lap 35.** Play it:

> *"I mean, I have to stop. There's no way you beat these two guys in front of me."*

DSI 82, arousal in the 91st percentile. Nobody tagged that by hand — the model
heard it, and the telemetry says which lap he was on when he said it.

Then **Hamilton, lap 15**: the radio says "box box", the lap time is +20s and the
compound changes to Hard. Two independent sources — a Hugging Face audio dataset
and F1 telemetry — agreeing on the same event. That join is what the project is.

## The part worth reading

The central hypothesis — that driver stress predicts lap-time loss — **is not
supported**. Across 1,155 paired observations, pooled r = 0.026, and stress does
not lead pace loss at any lag. It is on the Evidence screen, stated plainly.

Three of our own false positives were caught and killed getting there
(r = 0.62 from n = 10; r = −0.25 at p = 0.06; a negative correlation being
reported as "predictive"), and two planned features were measured and rejected:
ASR vocabulary prompting made word error rate worse in 4 of 6 races, and
acoustic speaker separation cleared its pre-registered bar for only 1 of 4
drivers.

Validated against CREMA-D gold labels, the affect scale recovers **arousal**
(+16 points over baseline) but **valence is at chance** (+1 point) — and the
index weights valence at 0.45, so nearly half of it rests on a dimension the
model cannot resolve. That is marked in the product, not buried.

What *does* hold: the index separates races that were genuinely different to
drive, and the recording-era confound was tested on nine same-season races and
ruled out.

## Limitations of this deployment

**Live Analysis runs on a second Space.** This one is `sdk: static`, so it has
no process to run a model in — Hugging Face hosts static Spaces free but charges
for Docker ones. The models therefore live in
[pitwall-live](https://huggingface.co/spaces/rogerdemello/pitwall-live), a
ZeroGPU Gradio Space that imports the same `pipeline/` package rather than
reimplementing it, and this frontend calls it cross-origin. The first call of
the day has to wake it and load three models, so it is slow once and quick
after.

**Audio is bundled for the showcase race only** (Abu Dhabi 2021). All twelve
would be 327 MB, and the Evidence screens need none of it.

## Models and data

| Role | Asset |
|---|---|
| Data | [`MikCil/f1-team-radio`](https://huggingface.co/datasets/MikCil/f1-team-radio) |
| Prosody | [`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) |
| ASR | [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) |
| Text sentiment | [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) |
| Validation | [`confit/cremad-parquet`](https://huggingface.co/datasets/confit/cremad-parquet) |

Telemetry via [FastF1](https://docs.fastf1.dev/).

Our analysed corpus is published as a dataset:
[**rogerdemello/pitwall-f1-radio-analysis**](https://huggingface.co/datasets/rogerdemello/pitwall-f1-radio-analysis)
