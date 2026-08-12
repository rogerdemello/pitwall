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

# First names heard on the pit-to-car channel. An engineer uses them as a
# vocative; a driver almost never says their own name.
DRIVER_NAMES = {
    "max", "lewis", "checo", "sergio", "valtteri", "charles", "carlos", "lando",
    "daniel", "pierre", "esteban", "fernando", "lance", "sebastian", "yuki",
    "george", "nicholas", "nikita", "mick", "kimi", "antonio", "seb", "danny",
    "brendon", "marcus", "alex", "nico", "oscar", "logan", "zhou", "kevin",
}

# Phrases only the pit wall says.
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


def _count(patterns: list[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text))


def classify(transcript: str) -> tuple[str, str]:
    """Return (speaker, why). speaker is 'driver' | 'engineer' | 'unknown'."""
    if not transcript or len(transcript.split()) < 2:
        return "unknown", "too short to attribute"

    t = " " + transcript.lower() + " "
    eng = _count(ENGINEER_CUES, t)
    drv = _count(DRIVER_CUES, t)

    # A name only signals "engineer" when used as a vocative - addressing someone.
    # A name followed by a verb is a third-person mention, which is just as likely
    # to be the driver talking about a rival: "Checo is a legend" is Verstappen
    # praising Perez, not the pit wall addressing him.
    named = [
        n for n in sorted(DRIVER_NAMES)
        if re.search(rf"\b{n}\b", t) and not re.search(rf"\b{n}\b\s*(?:'s|is|was|has|had|will|can|does|went|got)\b", t)
    ]
    if named:
        eng += 2  # a vocative is the strongest single cue

    if eng > drv:
        why = f"addresses the driver{' by name (' + named[0] + ')' if named else ''}"
        return "engineer", why
    if drv > eng:
        return "driver", "first-person report"
    return "unknown", "no clear direction"
