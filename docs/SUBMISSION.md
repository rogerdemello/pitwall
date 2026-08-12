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
  12 races: pooled r = 0.045. Stress does not predict lap-time loss, and does
  not lead it at any lag.
- **Three of our own false positives were caught and killed** — r = 0.62 from
  n = 10; r = −0.25 at p = 0.06; a *negative* correlation being reported as
  "predictive". Each guard is in the code and covered by tests.
- **Two planned features were measured and rejected** — ASR vocabulary prompting
  (worse in 4 of 6 races, and the prompt leaked into transcripts) and acoustic
  speaker separation (cleared its pre-registered bar for 1 of 4 drivers).
- **We know which half of our model works — and we were wrong about why.**
  Against CREMA-D gold labels, arousal scores 78.1% vs a 61.8% baseline; valence
  62.9% vs 61.9%, which is chance. We had planned to remove acoustic valence on
  that basis. A pre-registered matched-arousal test says the opposite: valence
  separates happy from anger at **AUC 0.67 with arousal held constant**, and
  fitting the boundary on held-out speakers lifts the axis to **+0.061** over
  baseline, six times the published figure. The axis is threshold-limited, not
  signal-limited. So the planned removal was abandoned.
- **And we did not ship the fix either.** The corrected boundary was measured on
  acted studio speech, where raw valence sits 2.3 standard deviations away from
  where it sits on radio. Applying it would relabel Stressed from 324 messages to
  72 for a reason unrelated to how the drivers sound, with no in-domain labels to
  say which is right. The finding is published; the production boundary is
  untouched.
- **Every DSI is now out of sample.** The calibration used to be fitted on the
  same 2,042 messages it scored. It is now leave-one-race-out, and the leak was
  quantified rather than quietly fixed: race means move by at most 0.41 points
  and the ordering survives, so every previously published contrast stands.
- **What does hold:** the index separates races that were genuinely different to
  drive — and more strongly out of sample than in, spread 5.5 → 7.7 points,
  Cohen's d 0.41 → 0.52. The recording-era confound was tested on nine
  same-season races and ruled out.

The suite passes (`pytest backend/tests -q`), and it includes tests that fail the
build if a published number drifts from the file that measured it.

---

## Known gaps, stated plainly

**Live Analysis needs a second Space.** Hugging Face hosts static Spaces free but
charges for Docker and Gradio ones, so the deployed frontend has no process to
run a model in. Free accounts in good standing may host two **ZeroGPU** Gradio
Spaces, so the model backend lives in `space_live/` — it imports `backend/pipeline`
rather than reimplementing it, and a test asserts the copy is byte-identical and
that its response matches `POST /api/analyze` field for field. Gradio accepts any
origin on `*.hf.space`, so the static frontend calls it directly with no proxy.

It works locally and in the verified Docker image either way — running that
container is how two real bugs were found. Everything on the Space's Race Replay
is this pipeline's actual output.

**Audio on the Space covers the showcase race only** (Abu Dhabi 2021). All twelve
would be 327 MB; the Evidence screens need none of it.

**No in-domain human labels.** The gold-label validation uses acted studio
speech. Nobody has listened to *this* audio and labelled it. The tool is built
(`label_affect.py`); the Evidence page says "still outstanding" rather than
implying otherwise.

That gap is now sharper than it was, which is progress of a kind: we know
precisely which number labels would settle. The valence boundary is misplaced by
a measured amount on gold data, and whether that correction transfers to radio is
exactly the question 300 labelled clips would answer. Before, "we need labels"
was a general wish; now it is a specific one.

These are limitations we chose to state rather than requirements we missed.

---

## Team

| Member | Hugging Face |
|---|---|
| Roger Demello | [rogerdemello](https://huggingface.co/rogerdemello) — owns the published Space and dataset |
| Vivian | [vynride](https://huggingface.co/vynride) |
