# PIT WALL — The Silent Co-Driver

**Grand Prix Hackathon · Problem Statement 1 · Powered by Hugging Face**

Race engineers miss warning signs in driver radio because they are watching
numbers, not listening to tone. PIT WALL listens for them — and turns what it
hears into a pit-wall decision.

We took **14,681 real F1 team-radio messages** from the Hugging Face Hub,
time-synced them to **real lap telemetry**, and measured how a driver actually
*sounded* against how he was actually *driving*.

---

## What makes this different

Most solutions to this brief are: upload a clip → Whisper → emotion label → bar
chart. Three things separate this one.

### 1. Every radio message is placed on a real lap

The dataset carries a UTC `message_timestamp`; FastF1 gives every lap an
absolute `LapStartDate`. Intersect them and each message inherits the real lap
number, lap time, tyre compound and stint age from the race it came from.

The join validates itself. On the 2018 Australian GP:

| Time | Message | Lap | Compound | Stint |
|---|---|---|---|---|
| 05:14:51 | "Box, Brendan, box, box." | 1 | SUPERSOFT | 1 |
| 05:15:43 | "we've got to make this work now" | 2 | **SOFT** | **2** |

The radio orders a pit stop; the telemetry independently confirms it happened on
the next lap, with the compound changing and the stint incrementing. Two
unrelated data sources agreeing is not a coincidence.

### 2. Incongruence detection — the warning that gets missed

Nobody misses a driver shouting. What gets missed is the driver saying *"yeah,
all good"* in a voice that is anything but.

We run text sentiment on **what was said** and prosody on **how it was said**,
and flag the gap. That is the exact failure mode the brief describes, and it is
the feature no other team will ship.

### 3. The numbers are measured, not asserted

There is a whole screen of evidence, including the result that did *not* come
out the way we hoped. See *Honest findings* below.

---

## The three screens

**Race Replay** — pick a race and a driver; lap time and Driver State Index sit
in stacked panels sharing one x-axis (never a dual-axis chart — that would invent
a correlation the data may not support, which is the exact claim we set out to
measure). Radio calls are pins on the DSI trace; clicking one plays the audio,
shows the transcript and the lap it landed on. Precomputed, so it is instant and
works offline.

**Live Analysis** — upload or record a clip and watch the same pipeline run. This
is the proof the replay is a real analysis and not a recording.

**Evidence** — every number the project claims, read from the eval files rather
than typed into the page, including the experiments that failed.

Each screen is scoped by a race picker, so the whole corpus is reachable.

## Architecture

```
frontend/            Next.js 16 + React 19 — Replay, Live, Evidence
backend/
  main.py            FastAPI: /race, /analyze, /evidence, /audio
  pipeline/
    asr.py           whisper-small.en + F1 vocabulary biasing
    prosody.py       audeering AVD model (continuous arousal/valence/dominance)
    sentiment.py     cardiffnlp RoBERTa sentiment
    calibration.py   percentile calibration to the F1 radio domain
    speaker.py       driver or engineer? grammatical-direction heuristic
    fusion.py        Driver State Index + incongruence detection
    strategy.py      transparent rule engine -> pit call
    analysis.py      does stress track pace? three ways
  data/
    fetch_many.py        pull the whole race slate in ONE pass over the Hub
    build_race.py        stage 1: run models, cache RAW outputs (~1 hr/race)
    calibrate.py         stage 2: raw -> calibrated race JSON (~1 sec)
    build_all.py         resumable overnight queue over the slate
    pool_calibration.py  one calibration across all races
    eval_asr.py          WER + jargon recall, biased vs unbiased
    label_affect.py      human labelling tool + confusion matrix
  research/              the spikes and ablations the Evidence screen cites
  tests/                 tests over the logic that is easy to break
```

**The two-stage split is deliberate.** Stage 1 is an hour of CPU inference per
race; stage 2 is a second. Because the raw model outputs are cached, the fusion
thresholds can be re-tuned and re-applied instantly — which is what made
iterating on them practical at all.

The Race Replay reads precomputed JSON, so the demo is instant and works with
the network unplugged. Only Live Analysis runs models at request time.

## Hugging Face assets

| Role | Asset |
|---|---|
| Data | [`MikCil/f1-team-radio`](https://huggingface.co/datasets/MikCil/f1-team-radio) — 14,681 clips, 149 GPs, 2018–2025 |
| Prosody | [`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) |
| ASR | [`openai/whisper-small.en`](https://huggingface.co/openai/whisper-small.en) |
| Text sentiment | [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) |

Telemetry via **FastF1**.

---

## The affect scale, validated — and its biggest flaw

The F1 dataset ships no emotion labels, so we validated the same pipeline
against **CREMA-D** (`confit/cremad-parquet`, 926 usable clips with gold labels).
Overall 4-way accuracy is **49.2% against a 41.3% majority-class baseline** —
real signal, but the useful part is the per-axis breakdown:

| Axis | Accuracy | Baseline | Lift |
|---|---|---|---|
| **Arousal** (high vs low) | 78.1% | 61.8% | **+16.3** |
| **Valence** (negative vs positive) | 62.9% | 61.9% | **+1.0** |

**Arousal works. Valence is at chance.** And DSI weights valence at 0.45, so
close to half the index rests on a dimension the model does not resolve. It also
explains the confusion matrix exactly: Stressed leaks into Energised, Calm into
Fatigued — pairs that differ *only* in valence.

The product says so. The state label carries a "valence unvalidated" marker, and
the Evidence screen leads with it. *Caveat: CREMA-D is acted studio speech, so
this validates the model and our quadrant logic, not in-domain radio. The
implication only runs one way — a model that cannot recover emotion from clean
speech is certainly not doing it on team radio.*

**A second model was no help, and that is itself a result.**
`Dpngtm/wav2vec2-emotion-recognition` agrees with us at chance (Cohen's kappa
−0.001). But it assigns one class to **82% of radio clips** while scoring 81.7%
against a 58.3% baseline on CREMA-D — so it works fine on studio speech and
collapses on compressed radio. The comparison cannot adjudicate our scoring, and
it independently justifies why this project uses a *dimensional* model with
domain calibration rather than off-the-shelf categorical emotion recognition.

## Honest findings

Five results that changed the build. All are shown in the app, including the two
that killed features we had planned to ship.

**We tried to separate the driver's voice from the engineer's, and failed.**
This was the most serious flaw in the project: 12% of clips provably contain both
speakers and 24% run over 15s, but the prosody model averages a whole clip into
one score — so "driver stress" was partly the engineer. We pooled each driver's
audio across 12 races, embedded 3-second windows with
`microsoft/wavlm-base-plus-sv`, and clustered into two speakers, naming the
clusters by grammatical direction as weak supervision.

Criteria were fixed **before** running. It cleared them for **1 of 4 drivers**:

| Driver | Anchors | Driver share | |
|---|---|---|---|
| VER | 30 | 0.71 vs 0.04 | pass |
| PER | 14 | 0.80 vs 0.11 | fail (anchors only) |
| HAM | 17 | 0.83 vs **0.55** | fail — cluster is mixed |
| NOR | 13 | 0.67 vs 0.43 | fail |

Acoustic separation is real (within-cluster cosine ~0.77 vs ~0.60 between), but
reliably *naming* which cluster is the driver is not achievable with the labels
available. Shipping it would have made attribution confidently wrong three times
in four — worse than the honest "unknown" the text heuristic already returns. It
is documented as a rejected experiment rather than quietly dropped. Three rounds
are recorded in `backend/races/_diarization_experiment.json`.

**A lag analysis that we refused to over-read.** Asking whether stress *precedes*
a drop in pace is the "Decision-Making" half of the theme, so we tested lags 0–3.
On one race it returned **r = 0.62 at lag 2 — from n = 10**. That is noise, and
picking the largest of four lags is a textbook multiple comparison. The code now
excludes any lag below n=30 from selection and ships the caveat alongside the
number, so the finding cannot be quoted without it.

**The prosody model needed domain calibration.** It was trained on studio
podcast speech. On compressed team radio its outputs collapse into a narrow band
(arousal p10–p90 ≈ 0.49–0.77), so a textbook 0.5 threshold labelled almost every
message identically — the first build called an entire Grand Prix "Energised"
with a stress index pinned between 44 and 54. Scoring each message as a
**percentile against the F1 radio corpus** restored the full range (31–91) and
made DSI mean something concrete: *DSI 80 = top fifth of stress for this race*.

**The F1 vocabulary biasing we planned as the differentiator does not work, so
we disabled it.** Measured over 40 clips: WER 0.2138 unbiased vs 0.2157
prompted — no improvement. Jargon recall moved 9/10 → 10/10, which is one term.
And the prompt leaks: *"Maybe I can just easily cut the corner"* came back as
*"**F1 radio**, the camera key just easily cut the cooler"*. The ablation is
kept and reported; the shipped transcripts don't use it.

**What actually mattered was the model.** `distil-whisper` collapses into
repetition loops when prompted at all (it was distilled without prompt
training), and mangles domain terms that `whisper-small.en` gets right:

| Truth | distil-whisper | whisper-small.en |
|---|---|---|
| does not have DRS | "the areas" | **DRS** |
| Hamilton's pitted | "I will turn to the pitted" | **Hamilton's pitted** |

**Does the index separate races that were genuinely different to drive?**
Partly — four of five predictions held, one clearly failed. The slate and the
predictions were fixed before any of it was analysed. Across **2,042 messages
from 12 races**, with calibration pooled so races are actually comparable:

| Race | n | Mean DSI | 95% CI | vs dry control |
|---|---|---|---|---|
| 2023 Monaco | 164 | 53.0 | 51.0–55.0 | +5.5, p<0.001 |
| 2020 Turkish (wet) | 143 | 53.0 | 51.0–54.9 | +5.5, p<0.001 |
| 2023 Qatar (heat) | 121 | 52.3 | 49.6–55.0 | +4.8, p=0.009 |
| 2021 Abu Dhabi | 108 | 51.4 | 49.0–53.8 | +3.9, p=0.025 (not claimed) |
| 2019 German (wet) | 114 | 48.8 | 46.4–51.2 | +1.3, p=0.46 |
| **2023 Italian (dry control)** | 132 | **47.5** | 45.2–49.9 | — |

The dry processional control came out lowest, as predicted. Wet Turkey and
Monaco sit significantly above it. Qatar — the extreme-heat race — has the
highest share of *Stressed* messages (21%) and 42% *Fatigued*.

**But 2019 German failed.** It was wet and chaotic and should have been near the
top; it came second-lowest and is statistically indistinguishable from the dry
control. We have no explanation we can support. Five contrasts were tested, so
the Bonferroni threshold is p<0.01 — Abu Dhabi's p=0.025 does not survive it and
is not claimed. The spread is 5.5 points against a within-race sd of ~13
(Cohen's d ≈ 0.42): real, but modest.

**The era confound: tested and ruled out.** This was an open weakness — the
slate spanned 2019–2023, radio encoding is not constant across seasons, and the
one race that failed its prediction was the oldest in the set, exactly what a
recording-era artefact would produce. So we built **nine races from 2023 alone**
and held the season fixed:

| | Spread |
|---|---|
| Within 2023 only (9 races) | **5.5 pts** |
| Across all eras (12 races) | 5.6 pts |

Essentially unchanged. Monaco vs Monza, both 2023, differ by **+5.5 (p = 0.0005)**,
surviving Bonferroni correction, and there is **no systematic offset between
eras** (2023 mean 50.9 vs pre-2023 51.5, p = 0.45). Recording era does not
explain the separation — the race-condition effect is real.

The within-season ordering is also coherent on its own terms: Monaco (street
circuit) top, Qatar (extreme heat) second, Monza (dry, processional) bottom.

**The central hypothesis is not supported, and that is the honest headline.**
Across **1,155 paired observations from twelve races**, there is no reliable
relationship between driver stress and lap-time loss:

- Pooled Pearson **r = 0.045**. Essentially nothing.
- Within drivers, the most-stressed calls sit **−0.07s** off the calmest, and
  only **38 of 80** drivers are slower when stressed — fewer than half, sign
  test **p = 0.74**.
- **Stress does not lead pace loss either.** Lags 0–3 were tested. The largest
  correlation (lap +3, r = −0.15) is *negative* — higher stress preceding
  *faster* laps, the opposite of the hypothesis — and not significant once four
  tests are accounted for.

**Doubling the corpus made the null cleaner, which is the point.** At 12 races
the within-driver gap read **+0.39s** with 23 of 37 drivers slower. At twelve it
is **−0.07s** with 38 of 80. That collapse toward zero as n grows is what a true
null looks like — and it means the earlier, more flattering number was itself
small-sample noise. We report the larger sample.

Reaching that took three rounds of killing our own false positives, and each
guard exists because one got through:

| Version | What slipped through | Fix |
|---|---|---|
| v1 | r = 0.62 at lag 2, from **n = 10** | minimum sample size |
| v2 | r = −0.25 at lag 3, from n = 56, **p = 0.06** | Bonferroni-corrected significance |
| v3 | a *negative* r reported as "predictive" | direction check |

Before that, the very first run of the same analysis included pit and
safety-car laps and reported one driver's "stressed" laps as **+18.8s** slower.
That was a pit stop, not a mood.

**What this means for the product.** The instrument works: it transcribes, scores
affect, places every message on a real lap, and separates races (below). What it
does *not* do is predict lap time from voice. Saying so is the finding — a null
result with 1155 observations and proper statistics is worth more than a
correlation manufactured by leaving pit stops in the data.

---

## Running it

```powershell
# dependencies
pip install fastf1 datasets transformers librosa soundfile jiwer duckdb `
            python-multipart scikit-learn fastapi uvicorn
cd frontend; npm install; cd ..

.\run.ps1            # start both servers against the precomputed race
.\run.ps1 -Build     # rebuild the race from scratch first (~1 hour of CPU)
```

Or step by step:

```bash
python backend/data/fetch_many.py                 # all races, one Hub scan
python backend/data/build_all.py                  # stage 1+2, ~1 hr/race, resumable
python backend/data/finish_corpus.py              # pool calibration, re-apply, print contrast
cd backend && uvicorn main:app --port 8000        # terminal 1
cd frontend && npm run dev                        # terminal 2
```

`wait_and_finish.py` chains the last two: it polls until every race's raw output
matches its manifest, then runs the pooling. Completion is detected from the data
rather than a process handle, so it survives the build being restarted.

`pytest backend/tests -q` runs the test suite (no models needed).

### The race slate

Chosen for **contrast**, not volume — the index has to prove it discriminates:

| Race | Why |
|---|---|
| 2021 Abu Dhabi | dry title decider (the showcase) |
| 2020 Turkish | soaking wet, near-continuous spins — peak stress |
| 2023 Monaco | most messages in the dataset; dry turning wet |
| 2019 German | wet chaos, many retirements |
| 2023 Italian | dry and processional — the **low-stress control** |
| 2023 Qatar | extreme heat — should light up *Fatigued* |

If wet Turkey does not show materially more stress than processional Monza, the
index is not measuring what we claim, and that is worth reporting loudly.

Runs CPU-only. A GPU is used automatically if present, but nothing requires one.

## Publishing

Both Hugging Face artifacts are built and verified; publishing needs an
authenticated account. See **[DEPLOY.md](DEPLOY.md)** for the exact commands.

- **Dataset** — 2,042 analysed messages with the telemetry join, packaged with a
  full card. Source audio is not redistributed; rows key back to
  `MikCil/f1-team-radio`.
- **Space** — Docker SDK, static frontend served by FastAPI, CPU only. Image
  builds, runs, serves every route and completes a live analysis in-container.

## Validating the affect scale

The dataset ships no emotion labels, so we built the tool to create them:

```bash
python backend/data/label_affect.py label 2021_Abu_Dhabi_Grand_Prix 60
python backend/data/label_affect.py score 2021_Abu_Dhabi_Grand_Prix
```

`label` plays clips and records what a human hears; `score` produces the
confusion matrix and per-class precision/recall. The sample is **stratified
across the model's own predicted states**, so a model that guesses one class for
everything can't look good on a sample it chose itself, and the prediction is
hidden while labelling so it can't anchor the judgement.

## Known limitations

- Affect is **not yet validated against human labels** — run the tool above to
  close this. Percentile calibration makes the scale internally consistent; it
  does not make it externally verified.
- Radio carries the engineer's voice as well as the driver's. `speaker.py`
  attributes each message by grammatical direction (a vocative like *"Okay,
  Lewis, box box"* is the pit wall; *"I have no grip"* is the driver) and
  suppresses driver-state claims on engineer transmissions. On Abu Dhabi 2021
  that splits **37 driver / 60 engineer / 71 unknown** — the unknown share is
  the largest because the heuristic answers "unknown" rather than guessing
  (a third-person mention like *"Checo is a legend"* is Verstappen talking
  about Pérez, not the pit wall addressing him).
- Because engineer messages can never carry the suppressed-stress flag, the flag
  rate is reported over **eligible** messages (9 of 108, 8.3%) rather than over
  all 168, which would have quietly understated it as 5.4%.
- The published reference transcripts contain systematic F1-jargon errors, so
  they are a comparison baseline, not gold truth.
- Lap-time delta is measured against each driver's own race median, which
  absorbs some but not all of traffic, fuel load and safety-car effects.
