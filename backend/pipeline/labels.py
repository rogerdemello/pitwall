"""Mapping categorical emotion labels onto the arousal/valence plane.

Two published evidence files disagreed on this, silently:

  eval_affect_gold.py    excluded disgust
  eval_convergent.py     mapped disgust -> Stressed

Both are defensible and they are not the same experiment. Disgust is genuinely
awkward on a two-dimensional plane - it is negative, but its arousal is not
reliably high or low - so excluding it avoids asserting a placement we cannot
justify, while including it keeps more data. What is not defensible is the two
choices being invisible: `_gold_affect_eval.json` and `_convergent_eval.json`
report numbers computed under different label sets with nothing saying so.

So the treatment is now a named parameter with a default, and whichever was used
is recorded in the output.
"""

from __future__ import annotations

STATES = ["Calm", "Energised", "Stressed", "Fatigued"]

#: Labels that place cleanly on the plane. Both spellings of each appear across
#: the datasets used (CREMA-D and the convergent reference model).
BASE = {
    "anger": "Stressed", "angry": "Stressed",
    "fear": "Stressed", "fearful": "Stressed",
    "happy": "Energised", "happiness": "Energised",
    "sad": "Fatigued", "sadness": "Fatigued",
    "neutral": "Calm",
}

#: Negative, but with no dependable arousal direction, so its quadrant is a
#: choice rather than a reading.
DISPUTED = {"disgust": "Stressed", "disgusted": "Stressed"}

#: Not in BASE and not disputed - simply absent from our plane. Surprise is
#: high-arousal with no valence sign at all.
UNMAPPABLE = {"surprise", "surprised", "excited", "excitement", "calm"}

#: High-arousal and negative-valence halves, for collapsing the four quadrants
#: onto each axis separately. This is what reveals that arousal carries the
#: model's signal and valence does not.
HIGH_AROUSAL = {"Energised", "Stressed"}
NEGATIVE_VALENCE = {"Stressed", "Fatigued"}


def quadrant_map(include_disgust: bool = False) -> dict[str, str]:
    """Label -> quadrant.

    `include_disgust=False` is the default because it is the more conservative
    choice: it declines to place a label whose arousal we cannot read, rather
    than assigning one and inheriting the error.
    """
    m = dict(BASE)
    if include_disgust:
        m.update(DISPUTED)
    return m


def excluded_labels(include_disgust: bool = False) -> list[str]:
    """Exactly what was left out, for the record that ships with the result."""
    out = set(UNMAPPABLE)
    if not include_disgust:
        out |= set(DISPUTED)
    return sorted(out)


def treatment(include_disgust: bool = False) -> dict:
    """The label-set decision, as a block to embed in an evidence file.

    Present in every output so two results computed under different treatments
    can never again look comparable.
    """
    return {
        "disgust": "mapped to Stressed" if include_disgust else "excluded",
        "why": (
            "Disgust is negative but has no dependable arousal direction, so its "
            "quadrant is a choice rather than a reading. "
            + ("Included here to retain data; note the other evaluation excludes it."
               if include_disgust else
               "Excluded here rather than asserting a placement we cannot justify.")
        ),
        "excluded_labels": excluded_labels(include_disgust),
        "n_mapped_labels": len(quadrant_map(include_disgust)),
    }


def to_quadrant(label: str, include_disgust: bool = False) -> str | None:
    """None for anything unmappable, so callers must decide what to do."""
    return quadrant_map(include_disgust).get(str(label).strip().lower())
