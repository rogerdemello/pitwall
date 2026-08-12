---
title: PIT WALL — Model Backend
emoji: 📻
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Live radio analysis backend for the PIT WALL Space
preload_from_hub:
  - openai/whisper-small.en
  - audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
  - cardiffnlp/twitter-roberta-base-sentiment-latest
startup_duration_timeout: 1h
---

# PIT WALL — model backend

This Space is the **backend** for [PIT WALL](https://huggingface.co/spaces/rogerdemello/pitwall).
Go there first; this page is the raw endpoint.

## Why it is separate

PIT WALL is a Next.js frontend against a FastAPI backend, and the two run
together in the project's Docker image. Hugging Face charges for Docker and
Gradio Spaces on personal accounts, so the public deployment is a **static**
Space — which serves the precomputed 12-race corpus perfectly, but has nowhere
to run a model. Live Analysis was disabled in public as a result.

Free accounts in good standing may host two ZeroGPU Gradio Spaces, so the model
backend lives here and the static frontend calls it cross-origin. Gradio accepts
any origin when the host is not a localhost alias, which is the case on
`*.hf.space`, so no proxy is involved.

## What it runs

| Stage | Model |
|---|---|
| Speech to text | `openai/whisper-small.en` |
| Voice affect | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` |
| Text sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Calibration | percentile map fitted on 2,042 real F1 radio messages |

Every model call goes through the `pipeline/` package copied verbatim from the
main repository. This is not a second implementation — if it ever disagrees with
the app, that is a build bug rather than a fork.

## As an API

```bash
curl -X POST https://<this-space>.hf.space/gradio_api/call/analyze \
  -H "Content-Type: application/json" \
  -d '{"data": [{"path": "https://example.com/clip.mp3", "meta": {"_type": "gradio.FileData"}}]}'
# -> {"event_id": "..."}   then GET .../analyze/<event_id> for the SSE result
```

The response matches `POST /api/analyze` in the FastAPI backend field for field,
so the frontend renders live and precomputed results with the same component.

## Two honest caveats

**The valence axis is at chance.** Validated against CREMA-D gold labels, the
affect model's arousal axis scores 78.1% against a 61.8% baseline, but valence
scores 62.9% against 61.9% — no better than guessing. A state's high/low
*activation* is reliable; its calm/stressed *direction* is much less so. This is
measured and reported rather than smoothed over.

**Calibration is F1-specific.** Scores are percentiles against team radio, which
is shouted over engine noise through a compressed channel. Ordinary speech will
be placed as unusually calm because it is, relative to that reference.

## Quota

ZeroGPU time is charged to the caller, not to this Space. Anonymous visitors get
a couple of GPU-minutes a day, which is roughly a hundred clips. Signing in to
Hugging Face raises it.
