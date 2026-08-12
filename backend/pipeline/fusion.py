"""Fusion: turn voice + words into a single driver state.

Two ideas do the work here.

**Driver State Index (DSI, 0-100).** Not a class label. Stress in speech is well
described by high arousal combined with low valence, so we compose the two
continuous dimensions rather than forcing a bucket. Dominance contributes a
smaller term: a driver who sounds activated *and* in control is coping, one who
sounds activated and diffident is not.

Scores are read as percentile ranks against the F1 radio corpus, not as raw
model outputs - see calibration.py for why that is necessary rather than merely
tidy. DSI 80 therefore means "in the most stressed fifth of radio messages in
this race", which is a claim a race engineer can act on.

**Incongruence.** The brief's actual pain point is warning signs being missed.
The signs that get missed are rarely the driver shouting - everyone notices
shouting. They are the driver saying "yeah, all good" in a voice that is
anything but. We score the gap between what the words claim and what the voice
shows, and surface it explicitly.

The first version of this detector fired on roughly half of all messages,
including a cheerful "Checo is a legend". Three constraints fixed it, and they
are the difference between a gimmick and a usable alert:
  1. compare like with like - both sides on a calibrated -1..+1 scale;
  2. require the voice to be genuinely in the negative tail of the corpus, not
     merely lower than the words;
  3. ignore transcripts too short or too garbled to carry real sentiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .calibration import Calibrator
from .prosody import Affect
from .sentiment import TextSentiment

# Thresholds live here so they can be tuned against the labelled set in one place.
VOICE_NEGATIVE_TAIL = 0.35   # calibrated valence below this = genuinely negative for F1 radio
TEXT_REASSURING = 0.25       # text polarity above this = words are claiming things are fine
INCONGRUENCE_FLAG = 0.55     # minimum words-vs-voice gap to raise the flag
MIN_WORDS_FOR_SENTIMENT = 3  # below this, text sentiment is noise


@dataclass
class DriverState:
    dsi: int
    state: str                # Calm | Energised | Stressed | Fatigued
    descriptor: str
    arousal_pct: float        # calibrated 0..1
    valence_pct: float
    dominance_pct: float
    arousal_raw: float
    valence_raw: float
    text_polarity: float
    incongruence: float
    suppressed_stress: bool
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def _quadrant(arousal_pct: float, valence_pct: float) -> tuple[str, str]:
    """Place the speaker on the calibrated arousal/valence plane."""
    hi_a = arousal_pct >= 0.5
    neg_v = valence_pct < 0.5
    if hi_a and neg_v:
        return "Stressed", "activated and negative - frustration or pressure"
    if hi_a and not neg_v:
        return "Energised", "activated and positive - focused, on it"
    if not hi_a and neg_v:
        return "Fatigued", "low energy and negative - resigned or worn down"
    return "Calm", "settled and composed"


def fuse(
    affect: Affect,
    text: TextSentiment,
    calibrator: Calibrator | None = None,
    transcript: str = "",
) -> DriverState:
    """Combine acoustic affect and text sentiment into one driver state."""
    if calibrator is not None:
        a = calibrator.pct_arousal(affect.arousal)
        v = calibrator.pct_valence(affect.valence)
        d = calibrator.pct_dominance(affect.dominance)
    else:
        # Uncalibrated fallback: usable, but compressed. See calibration.py.
        a, v, d = affect.arousal, affect.valence, affect.dominance

    stress = (0.55 * a) + (0.45 * (1.0 - v)) - (0.10 * (d - 0.5))
    dsi = int(round(max(0.0, min(1.0, stress)) * 100))

    state, descriptor = _quadrant(a, v)

    # Words vs voice, both on a comparable -1..+1 scale.
    voice_polarity = (v - 0.5) * 2
    gap = abs(text.polarity - voice_polarity)

    # Only flag the asymmetric case that matters, and only when the text is
    # substantial enough for its sentiment to mean anything.
    enough_text = len(transcript.split()) >= MIN_WORDS_FOR_SENTIMENT
    suppressed = (
        enough_text
        and gap >= INCONGRUENCE_FLAG
        and text.polarity >= TEXT_REASSURING
        and v <= VOICE_NEGATIVE_TAIL
    )

    if suppressed:
        note = (
            f"Words read {text.label}, but the voice sits in the bottom "
            f"{int(v * 100)}% of the race for valence. Possible suppressed stress."
        )
    else:
        note = f"Voice and words agree - {descriptor}."

    return DriverState(
        dsi=dsi,
        state=state,
        descriptor=descriptor,
        arousal_pct=round(a, 4),
        valence_pct=round(v, 4),
        dominance_pct=round(d, 4),
        arousal_raw=affect.arousal,
        valence_raw=affect.valence,
        text_polarity=text.polarity,
        incongruence=round(gap, 4),
        suppressed_stress=suppressed,
        note=note,
    )
