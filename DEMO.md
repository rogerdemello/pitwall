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

1. **We know exactly which half of our model works.** Validated against CREMA-D
   gold labels: arousal 78% against a 62% baseline. Valence 63% against 62% —
   chance. Our index weights valence at 0.45, so nearly half of it rests on a
   dimension the model can't resolve. It's labelled as such in the product.

   *This is the strongest thing you can say. Most teams cannot tell you which
   part of their model is broken. Lead with it.*

2. **The central claim failed.** 556 observations, six races: r = 0.047. Stress
   does not predict lap time. Say it plainly.
3. **We killed three of our own false positives getting there** — r = 0.62 from
   n = 10, then r = −0.25 at p = 0.06, then a negative correlation being called
   "predictive". Each guard is in the code and tested.
4. **What did work:** the index separates races. The dry processional control
   (Monza) is lowest; wet Turkey and Monaco are significantly above it. Four of
   five predictions held — 2019 German failed, which we can't explain.
5. **We rejected two planned features on the evidence**: ASR vocabulary
   prompting (worse in 4 of 6 races, and the prompt leaked into transcripts) and
   acoustic speaker separation (cleared its pre-set bar for 1 of 4 drivers).

> Every criterion was set before we saw the answer, not after.

## 4:40 — Close (20 seconds)

> The instrument works — transcription, affect, telemetry join, a pit call with
> its evidence attached. What we can't tell you is that voice predicts lap time,
> because we measured it and it doesn't. We'd rather show you a null result with
> 556 observations than a correlation we manufactured.

---

## Questions you should expect

**"How do you know that's the driver and not the engineer?"**
Grammatical direction — a vocative like "Okay, Lewis, box box" is the pit wall,
"I have no grip" is the driver. It answers *unknown* 42% of the time rather than
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
We validated it against CREMA-D gold labels. Arousal is recovered well (+16
points over baseline); valence is at chance. So we know precisely what to trust:
the high-arousal/low-arousal split holds, the positive/negative split does not.
The caveat is that CREMA-D is acted studio speech — it validates the model, not
in-domain radio.

**"Why not use an off-the-shelf emotion classifier?"**
We tried one. It scores 82% on CREMA-D but assigns a single class to 82% of
radio clips — it completely fails to transfer to compressed audio. That's why we
use a dimensional model with domain calibration instead.

**"What's not finished?"**
No in-domain human labels. The gold validation uses acted speech; nobody has
listened to *this* audio and labelled it. The tool is built; the listening pass
isn't done, and the Evidence page says so.
