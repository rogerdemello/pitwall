# Demo script

Five minutes. The brief says presentation counts as much as the build, and the
strongest thing this project has is that its numbers are honest — so lead with
the machine, then show the evidence, and do not oversell.

**Before you start:** run `.\run.ps1`, open `http://localhost:3000`, confirm the
Evidence page loads. Turn wifi off — nothing needs the network. Have the Race
Replay open on **2021 Abu Dhabi, Sainz**.

---

## 0:00 — The problem (20 seconds)

> During a race the engineers are watching numbers, not listening to tone. The
> warning signs in a driver's voice get missed because nobody has time to hear
> them.

Don't over-explain. The next thirty seconds does the work.

## 0:20 — Play one clip (40 seconds)

Race Replay → Key Moments → **SAI lap 35, DSI 82**.

Play it. Let them hear it before you say anything.

> *"I mean, I have to stop. There's no way you beat these two guys in front of
> me."*

Then point at the screen, in this order:

- **DSI 82, Stressed** — arousal in the 91st percentile, valence in the 21st
- **Lap 35** — not a label on a clip, the actual lap he was driving
- The lap-time trace above it

> We didn't tag this by hand. The model heard it, and the telemetry says exactly
> which lap he was on when he said it.

## 1:00 — Why the lap matters (45 seconds)

Switch to **Hamilton, lap 15**.

> The radio says "box box". The lap time is +20 seconds and the tyre compound
> changes to Hard. Two completely independent sources — a Hugging Face audio
> dataset and F1 telemetry — agreeing on the same event. That's the join, and
> it's what makes everything else possible.

This is the moment that separates you from a clip-labelling demo. Land it.

## 1:45 — The feature nobody else built (45 seconds)

Find a **suppressed stress** flag (white-ringed dot, or the Key Moments chip).

> Everyone notices a driver shouting. What gets missed is the driver saying
> "yeah, all good" in a voice that isn't. We run sentiment on the words and
> prosody on the voice, and flag it when they disagree.

Mention the guard, briefly — it shows judgement:

> Our first version fired on half of all messages, including a cheerful "Checo is
> a legend". It now fires on 8% of eligible messages.

## 2:30 — Live proof (40 seconds)

Live Analysis → upload any clip.

> The replay is precomputed so it can't fail on stage. This is the same pipeline
> running live, so you can see it's a real analysis and not a recording.

## 3:10 — Evidence (90 seconds) — *the part that wins*

Open the Evidence page. Do **not** skip the negatives; they are the strongest
material you have.

> We measured everything, and we're reporting it whichever way it came out.

1. **We pre-registered five predictions, rebuilt the pipeline, and four of them
   broke.** Under the old ASR, four of five held. On a better model — whisper
   large-v3, no 30-second truncation — only one survives. So we published the
   falsification clause we wrote in advance: *the v1 separation was largely an
   artifact of the v1 pipeline.*

   *This is the strongest thing you can say, and it is worth thirty seconds on
   its own. We improved our own instrument and it cost us our own result. Nobody
   made us look.*

   If asked why we didn't keep the flattering version: the spread between races
   actually **grew**, 7.7 to 8.5 points, so there is a reading where the clause
   doesn't fire. Spread on its own isn't a claim — any noisy index has spread.
   The directions were what we predicted, and the directions failed.

2. **The central claim failed, and a better model made it fail harder.**
   1155 observations, 12 races: r = 0.026, and every statistic moved *toward*
   zero from v1. Drivers slower when stressed: 41 of 80 — a coin flip, sign test
   p = 0.91. We predicted in advance that a better model would not reveal a
   relationship that isn't there. It didn't.
3. **We know which half of our model works.** Arousal 79% against a 62%
   baseline. Valence 61% against 62% — *below* chance. The index weights valence
   at 0.45, so nearly half of it rests on a dimension the model cannot resolve,
   and the product says so.
4. **We killed three of our own false positives getting there** — r = 0.62 from
   n = 10, then r = −0.25 at p = 0.06, then a negative correlation being called
   "predictive". Each guard is in the code and tested.
5. **Our hallucination fix mostly renamed the problem.** The 51 near-silent
   clips that used to transcribe as "you" now transcribe as "Thank you." Fifty
   of the fifty-one are still inventing text. The rate passed its threshold;
   the test we'd named did not.

> Every criterion was set before we saw the answer, not after — including the
> one that told us to retract.

## 4:40 — Close (20 seconds)

> The instrument works — transcription, affect, telemetry join, a pit call with
> its evidence attached. What we can't tell you is that voice predicts lap time,
> because we measured it and it doesn't. We'd rather show you a null result with
> 1155 observations than a correlation we manufactured.

---

## Questions you should expect

**"How do you know that's the driver and not the engineer?"**
Grammatical direction — a vocative like "Okay, Lewis, box box" is the pit wall,
"I have no grip" is the driver. It answers *unknown* 45% of the time rather than
guessing, and engineer messages never raise a driver-state flag. We tried to do
it acoustically and it didn't clear our bar; that's on the Evidence page.

**"Why is your correlation so weak?"**
Because it is. Lap time is dominated by traffic, fuel load and tyre state; mood
is a small term. We excluded pit and safety-car laps precisely so we wouldn't
report a fake +18.8s effect.

**"Isn't a null result bad for you?"**
The instrument is the deliverable. A tool that tells you honestly what it can't
support is more useful on a pit wall than one that always finds something.

**"How do you know your emotion model works at all?"**
We validated it against CREMA-D gold labels. Arousal is recovered well (+17.6
points over baseline); valence scores *below* the majority baseline. So we know
precisely what to trust: the high-arousal/low-arousal split holds, the
positive/negative split does not.
The caveat is that CREMA-D is acted studio speech — it validates the model, not
in-domain radio.

**"Why not use an off-the-shelf emotion classifier?"**
We tried one. It scores 82% on CREMA-D but assigns a single class to 74% of
radio clips — it completely fails to transfer to compressed audio. That's why we
use a dimensional model with domain calibration instead.

**"What's not finished?"**
No in-domain human labels. The gold validation uses acted speech; nobody has
listened to *this* audio and labelled it. The tool is built; the listening pass
isn't done, and the Evidence page says so.
