# Submission — Problem Statement 1

**PIT WALL — The Silent Co-Driver**

Live: [Space](https://huggingface.co/spaces/rogerdemello/pitwall) ·
[Dataset](https://huggingface.co/datasets/rogerdemello/pitwall-f1-radio-analysis)
Run locally: `.\run.ps1` → http://localhost:3000

---

## Every requirement, and where it is met

Each line was verified against the running application, not asserted.

| PS1 requirement | Where | Status |
|---|---|---|
| "play or upload a radio audio clip" | Race Replay plays any of 2,042 clips; Live Analysis accepts upload or mic | ✅ |
| "converts the speech to text" | `pipeline/asr.py` — `openai/whisper-small.en` | ✅ |
| "studies the tone of voice and shows if the driver seems calm, stressed, or tired" | `pipeline/prosody.py` + `fusion.py` → **Calm / Energised / Stressed / Fatigued** | ✅ |
| "shown alongside basic lap-time information" | Lap time and Driver State Index in stacked panels on one shared x-axis | ✅ |
| "see if stress is matching up with slower laps" | Evidence screen — measured, and reported as a null result | ✅ |
| **Frontend** — upload/play, text, mood, lap chart | Next.js, three screens | ✅ |
| **Backend** — audio processed, models think | FastAPI, 10 endpoints | ✅ |
| **Input** — audio clips + lap time data | HF dataset + FastF1 telemetry | ✅ |
| **Output** — transcript, mood label, visual | All three, per message | ✅ |

### General rules

| Rule | Status |
|---|---|
| Frontend **and** backend (no notebook-only) | ✅ Next.js + FastAPI |
| Uses something from the Hugging Face Hub | ✅ 4 models + 2 datasets consumed; 1 dataset + 1 Space **published** |
| Balanced difficulty | ✅ 4-model pipeline, telemetry join, calibration — not one API call, not trained from scratch |
| Every team member has their own HF account | ✅ **2 of 2** — [rogerdemello](https://huggingface.co/rogerdemello) · [vynride](https://huggingface.co/vynride) |

**All mandatory rules are met.**

---

## What makes it more than the brief

**Every radio message is placed on the real lap it was spoken on.** The dataset
carries a UTC timestamp; FastF1 gives every lap an absolute start time.
Intersecting them is the whole project. It validates itself: a "Box, box, box"
call lands on lap 1 (SUPERSOFT, stint 1) and the next message on lap 2 (SOFT,
stint 2) — telemetry independently confirming the stop the radio ordered.

**Incongruence detection.** Nobody misses a driver shouting. What gets missed is
"yeah, all good" said in a voice that isn't. Text sentiment vs prosody, flagged
when they disagree.

**A pit call, not just a label.** Stress trend + pace + tyre age → an actual
recommendation with its evidence attached.

---

## The evidence, including what failed

The Evidence screen reads from measured files; nothing on it is typed by hand.

- **The central hypothesis is not supported.** 1,155 paired observations,
  12 races: pooled r = 0.043. Stress does not predict lap-time loss, and does
  not lead it at any lag.
- **Three of our own false positives were caught and killed** — r = 0.62 from
  n = 10; r = −0.25 at p = 0.06; a *negative* correlation being reported as
  "predictive". Each guard is in the code and covered by tests.
- **Two planned features were measured and rejected** — ASR vocabulary prompting
  (worse in 4 of 6 races, and the prompt leaked into transcripts) and acoustic
  speaker separation (cleared its pre-registered bar for 1 of 4 drivers).
- **We know which half of our model works.** Against CREMA-D gold labels,
  arousal scores 78.1% vs a 61.8% baseline; valence 62.9% vs 61.9% — chance. The
  index weights valence 0.45, so nearly half of it rests on a dimension the model
  cannot resolve. Marked in the product, not buried.
- **What does hold:** the index separates races that were genuinely different to
  drive, and the recording-era confound was tested on nine same-season races and
  ruled out.

48 tests pass (`pytest backend/tests -q`).

---

## Known gaps, stated plainly

**Live Analysis is disabled on the public Space.** It needs a model backend;
Hugging Face hosts static Spaces free but charges for Docker ones. It works
locally, and the Docker image is built and verified — running that container is
how two real bugs were found. Everything on the Space's Race Replay is this
pipeline's actual output.

**Audio on the Space covers the showcase race only** (Abu Dhabi 2021). All twelve
would be 327 MB; the Evidence screens need none of it.

**No in-domain human labels.** The gold-label validation uses acted studio
speech. Nobody has listened to *this* audio and labelled it. The tool is built
(`label_affect.py`); the Evidence page says "still outstanding" rather than
implying otherwise.

This is the only remaining gap, and it is a limitation we chose to state rather
than a requirement we missed.

---

## Team

| Member | Hugging Face |
|---|---|
| Roger Demello | [rogerdemello](https://huggingface.co/rogerdemello) — owns the published Space and dataset |
| Vivian | [vynride](https://huggingface.co/vynride) |
