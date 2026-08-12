"""The two evaluations must not disagree about their label set in silence.

`eval_affect_gold.py` excluded disgust; `eval_convergent.py` mapped it to
Stressed. Both choices are defensible and the two are different experiments -
but `_gold_affect_eval.json` and `_convergent_eval.json` published numbers
computed under different label sets with nothing in either file saying so, which
invites reading them as comparable.

These tests do not force the two to agree. They force the disagreement to be
declared.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import labels  # noqa: E402


class TestQuadrantMap:
    def test_disgust_excluded_by_default(self):
        """The conservative choice: decline to place what we cannot read."""
        assert "disgust" not in labels.quadrant_map()
        assert labels.to_quadrant("disgust") is None

    def test_disgust_included_on_request(self):
        assert labels.quadrant_map(include_disgust=True)["disgust"] == "Stressed"
        assert labels.to_quadrant("disgust", include_disgust=True) == "Stressed"

    @pytest.mark.parametrize("label,quadrant", [
        ("anger", "Stressed"), ("angry", "Stressed"),
        ("fear", "Stressed"), ("fearful", "Stressed"),
        ("happy", "Energised"), ("happiness", "Energised"),
        ("sad", "Fatigued"), ("sadness", "Fatigued"),
        ("neutral", "Calm"),
    ])
    def test_undisputed_labels_are_stable(self, label, quadrant):
        """These must not move: two published evaluations rest on them."""
        assert labels.to_quadrant(label) == quadrant
        assert labels.to_quadrant(label, include_disgust=True) == quadrant

    def test_every_mapped_label_lands_in_a_real_state(self):
        for include in (False, True):
            assert set(labels.quadrant_map(include).values()) <= set(labels.STATES)

    def test_surprise_is_never_mapped(self):
        """High arousal with no valence sign - it has no place on the plane."""
        assert labels.to_quadrant("surprise") is None
        assert labels.to_quadrant("surprise", include_disgust=True) is None

    def test_case_and_whitespace_tolerant(self):
        assert labels.to_quadrant("  ANGER ") == "Stressed"

    def test_unknown_label_is_none_not_a_guess(self):
        assert labels.to_quadrant("bored") is None


class TestTreatmentIsDeclared:
    """Whichever choice a script makes, its output must say which."""

    @pytest.mark.parametrize("include", [False, True])
    def test_treatment_names_the_choice(self, include):
        t = labels.treatment(include)
        assert t["disgust"] in ("excluded", "mapped to Stressed")
        assert (t["disgust"] == "mapped to Stressed") == include

    @pytest.mark.parametrize("include", [False, True])
    def test_treatment_lists_what_was_left_out(self, include):
        t = labels.treatment(include)
        assert ("disgust" in t["excluded_labels"]) != include

    def test_the_two_treatments_are_distinguishable(self):
        assert labels.treatment(False) != labels.treatment(True)

    def test_excluded_always_covers_the_unmappable(self):
        for include in (False, True):
            assert set(labels.UNMAPPABLE) <= set(labels.excluded_labels(include))


class TestAxes:
    def test_axes_partition_the_four_states(self):
        assert labels.HIGH_AROUSAL | (set(labels.STATES) - labels.HIGH_AROUSAL) \
            == set(labels.STATES)
        assert labels.NEGATIVE_VALENCE <= set(labels.STATES)

    def test_axes_are_orthogonal(self):
        """Each state is one unique (arousal, valence) combination.

        If two states shared a corner the quadrant scheme would be degenerate,
        and the per-axis breakdown that shows valence is at chance would be
        measuring something else.
        """
        corners = {(s in labels.HIGH_AROUSAL, s in labels.NEGATIVE_VALENCE)
                   for s in labels.STATES}
        assert len(corners) == 4

    def test_stressed_is_high_arousal_negative_valence(self):
        assert "Stressed" in labels.HIGH_AROUSAL
        assert "Stressed" in labels.NEGATIVE_VALENCE

    def test_calm_is_low_arousal_positive_valence(self):
        assert "Calm" not in labels.HIGH_AROUSAL
        assert "Calm" not in labels.NEGATIVE_VALENCE


class TestEvalScriptsDeclareTheirTreatment:
    """The scripts themselves, not just the helper."""

    def test_gold_excludes_disgust(self):
        from data import eval_affect_gold
        assert eval_affect_gold.INCLUDE_DISGUST is False

    def test_convergent_includes_disgust(self):
        from data import eval_convergent
        assert eval_convergent.INCLUDE_DISGUST is True

    def test_they_disagree_on_purpose(self):
        """Pinning the disagreement so it cannot be 'fixed' without a decision."""
        from data import eval_affect_gold, eval_convergent
        assert eval_affect_gold.INCLUDE_DISGUST != eval_convergent.INCLUDE_DISGUST
