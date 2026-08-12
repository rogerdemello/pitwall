"""Who is talking - the driver, or the race engineer?

A team radio clip is a two-way channel. The dataset does not say which side of
it is speaking, and treating every clip as "the driver" produces claims that are
plainly wrong: "You are doing a good job" scored as driver stress is the
engineer's voice, not the driver's.

There is no speaker label to learn from and diarisation on 3-second compressed
radio is unreliable, so we use the one signal that is actually dependable here:
grammatical direction. An engineer addresses the driver by name and says "you".
A driver reports their own state and says "I".

Deliberately conservative. When the evidence is thin the answer is "unknown",
and downstream code treats unknown as "may be the driver" rather than guessing.
This does not need to be perfect - it needs to stop the product from asserting
that the engineer's tone is the driver's mood.
"""

from __future__ import annotations

import re

from pipeline import roster

# Names heard on the pit-to-car channel come from pipeline/roster.json, which is
# generated from FastF1. This used to be a hand-typed set of first names with the
# same gap as calibrate.py's display map: nyck, liam, guanyu, romain, daniil and
# robert were all absent, so for those six drivers the strongest single
# attribution cue never fired at all. The roster also supplies surnames and radio
# nicknames ("checo", "danny ric", "hulk"), which the old set had only partially.

# Phrases only the pit wall says.
#
# Note the absence of a bare `\byou\b`, which looks like an omission next to
# `\bi\b` in DRIVER_CUES and is not. It was tried: it moves unknown from 52.5%
# to 45.7%, but it reclassifies 76 confidently-attributed driver messages to
# unknown, because the channel is two-way and drivers say "you" to their
# engineer as readily as the reverse - "You decide on the tyre based on the
# weather forecast" and "I hope you remember what I was saying" are both the
# driver. Possessives and contractions ("your", "you're") stay, because those
# really are directional.
#
# Trading one error class for another with no labels to say which is smaller is
# guessing. The fix is a fitted model over segment-level features, scored against
# ground truth - not a hand-tuned lexicon.
ENGINEER_CUES = [
    r"\bbox box\b", r"\bbox this lap\b", r"\byou're\b", r"\byou are\b",
    r"\byour\b", r"\bwe need you\b", r"\bcopy that\b", r"\bconfirm\b",
    r"\bgap (is|to)\b", r"\bposition \w+\b", r"\bpush now\b", r"\bmode\b",
    r"\btarget\b", r"\bstay out\b", r"\bkeep it up\b", r"\bunderstood\b",
]

# Phrases that indicate the driver reporting on themselves or the car.
DRIVER_CUES = [
    r"\bi\b", r"\bi'm\b", r"\bmy\b", r"\bme\b", r"\bi've\b", r"\bi can\b",
    r"\bno grip\b", r"\blosing\b", r"\bthe car is\b", r"\bit's undriveable\b",
]


# A name followed by one of these is being talked *about*, not talked *to*.
THIRD_PERSON = r"\s*(?:'s|is|was|has|had|will|can|does|went|got|s\b)"


def _count(patterns: list[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text))


def classify(transcript: str, driver_id: str | None = None) -> tuple[str, str]:
    """Return (speaker, why). speaker is 'driver' | 'engineer' | 'unknown'.

    `driver_id` is the car the clip came from, and it is free metadata that was
    previously unused. It sharpens the vocative cue considerably, because an
    engineer addresses *this* driver: hearing "Lewis" on Hamilton's channel is
    the pit wall talking to him, while hearing "Checo" on Verstappen's channel is
    Verstappen talking about someone else. Without it, both look identical.
    """
    if not transcript or len(transcript.split()) < 2:
        return "unknown", "too short to attribute"

    t = " " + transcript.lower() + " "
    eng = _count(ENGINEER_CUES, t)
    drv = _count(DRIVER_CUES, t)

    # A name only signals "engineer" when used as a vocative - addressing someone.
    # A name followed by a verb is a third-person mention, which is just as likely
    # to be the driver talking about a rival: "Checo is a legend" is Verstappen
    # praising Perez, not the pit wall addressing him.
    addressed = [
        (name, did) for name, did in roster.find_vocatives(t)
        if not re.search(rf"\b{re.escape(name)}\b{THIRD_PERSON}\b", t)
    ]

    why_eng = why_drv = ""
    if addressed:
        own = [n for n, did in addressed if driver_id and did == driver_id]
        other = [n for n, did in addressed if did != driver_id]
        if own:
            # Addressed by their own name on their own channel: the strongest
            # single cue available.
            eng += 3
            why_eng = f" by name ({own[0]})"
        elif driver_id:
            # Someone else's name is a third-person mention, and drivers discuss
            # rivals far more than engineers do. Weak evidence only - it must not
            # override an explicit "you", which is why this adjusts the score
            # rather than deciding: "We are locking both axles, Danny. You can
            # afford more front lock" mentions another car and is still plainly
            # the engineer.
            drv += 1
            why_drv = f", mentions another driver ({other[0]})"
        else:
            # No driver_id supplied, so own-vs-other cannot be told apart. This
            # is the pre-roster behaviour, kept for callers that have only text.
            eng += 2
            why_eng = f" by name ({other[0]})"

    if eng > drv:
        return "engineer", f"addresses the driver{why_eng}"
    if drv > eng:
        return "driver", f"first-person report{why_drv}"
    return "unknown", "no clear direction"
