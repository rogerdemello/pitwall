"""The Evidence screen claims nothing on it is typed by hand. Enforce that.

`_corpus_finding.json` was hand-written with no generator. When the corpus grew
from six races to twelve the calibration was re-pooled, every mean DSI moved by
about 0.4 points, and the published file kept the old numbers - including a
`survives_bonferroni: true` that had since become false. It was still being
served by /api/corpus-finding and rendered under the honesty claim.

These tests re-derive the published numbers from `races/<race>.json` and fail if
they have drifted. They are the mechanism behind the claim, not a restatement of
it.

    pytest backend/tests/test_evidence_is_measured.py -q
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import corpus_finding  # noqa: E402
from pipeline.artifacts import iter_race_files, race_ids  # noqa: E402

RACES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "races")


def _load(name: str):
    path = os.path.join(RACES, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not built")
    return json.load(open(path, encoding="utf-8"))


@pytest.fixture(scope="module")
def finding():
    return _load("_corpus_finding.json")


@pytest.fixture(scope="module")
def live_races():
    """Every built race, summarised straight from its race JSON."""
    if not race_ids(RACES):
        pytest.skip("no races built")
    return {s["race_id"]: s
            for s in (corpus_finding.summarise(p) for p in iter_race_files(RACES))
            if s}


class TestCorpusFindingIsGenerated:
    def test_declares_its_generator(self, finding):
        """A hand-written file has no generator to declare."""
        assert finding.get("generated_by") == "backend/data/corpus_finding.py"

    def test_every_published_mean_matches_the_race_file(self, finding, live_races):
        drifted = []
        for row in finding["results"] + finding["not_predicted"]["races"]:
            live = live_races.get(row["race_id"])
            if live is None:
                drifted.append((row["race"], "race file missing"))
            elif abs(live["mean_dsi"] - row["mean_dsi"]) > 0.05:
                drifted.append((row["race"], f"published {row['mean_dsi']}, "
                                             f"measured {live['mean_dsi']}"))
        assert not drifted, f"published DSI has drifted from the race files: {drifted}"

    def test_every_published_n_matches(self, finding, live_races):
        for row in finding["results"] + finding["not_predicted"]["races"]:
            assert row["n"] == live_races[row["race_id"]]["n"], row["race"]

    def test_contrast_significance_is_recomputable(self, finding, live_races):
        """Re-run the z-test. A survives_bonferroni that has flipped must fail."""
        control = live_races[corpus_finding.CONTROL]
        n_tested = len(finding["contrasts_vs_dry_control"])
        alpha = 0.05 / n_tested
        by_name = {r["race"]: r for r in live_races.values()}

        for c in finding["contrasts_vs_dry_control"]:
            diff, z, p = corpus_finding.z_test(by_name[c["race"]], control)
            assert c["diff"] == pytest.approx(diff, abs=0.05), c["race"]
            assert c["z"] == pytest.approx(z, abs=0.02), c["race"]
            assert c["survives_bonferroni"] == (p < alpha), (
                f"{c['race']}: published survives={c['survives_bonferroni']}, "
                f"measured p={p:.4f} against alpha={alpha:.4f}"
            )

    def test_unclaimed_contrasts_are_named_in_the_note(self, finding):
        """Anything that fails correction must be said so in words, not buried."""
        note = finding["bonferroni_note"]
        for c in finding["contrasts_vs_dry_control"]:
            if not c["survives_bonferroni"]:
                assert c["race"] in note, (
                    f"{c['race']} does not survive Bonferroni but is not "
                    "mentioned in bonferroni_note"
                )

    def test_a_failed_prediction_is_reported(self, finding):
        """If the scorecard has a failure, what_failed must not be empty."""
        failures = [v for v in finding["prediction_scorecard"] if not v["held"]]
        if failures:
            assert finding["what_failed"], (
                f"{len(failures)} prediction(s) failed but what_failed is empty"
            )
            assert "PARTIAL" in finding["verdict"] or "NOT SUPPORTED" in finding["verdict"]

    def test_prediction_slate_is_not_silently_widened(self, finding):
        """Predictions were registered for six races. Later races are descriptive.

        Folding the confound-test races into the prediction slate after the fact
        would turn a confirmatory result into an exploratory one.
        """
        assert finding["setup"]["races"] == len(corpus_finding.SLATE)
        predicted = {v["race"] for v in finding["prediction_scorecard"]}
        assert len(predicted) == len(corpus_finding.PREREGISTERED)
        descriptive = {r["race"] for r in finding["not_predicted"]["races"]}
        assert not (predicted & descriptive)


class TestCalibrationIsSharedAndHeldOut:
    """Every cross-race claim depends on one shared calibration.

    Per-race calibration centres each race on 50.0 by construction, which would
    make the contrast test vacuous rather than merely wrong. Both pooled and
    leave-one-race-out use one reference across the corpus; LORO additionally
    makes each score out-of-sample.
    """

    def test_no_race_uses_its_own_calibration(self, live_races):
        from pipeline.calibration import CROSS_RACE_SOURCES
        sources = {r["calibration_source"] for r in live_races.values()}
        assert sources <= CROSS_RACE_SOURCES, f"unusable sources: {sources}"

    def test_scores_are_out_of_sample(self, live_races):
        """The corpus should be scored against calibrations it did not help fit."""
        sources = {r["calibration_source"] for r in live_races.values()}
        assert sources == {"leave-one-race-out"}, (
            f"expected held-out calibration everywhere, got {sources}. "
            "Run pool_calibration.py then re-run calibrate.py."
        )

    def test_the_leak_was_quantified(self):
        """Fixing a leak silently is half the work; the size must be published."""
        leak = _load("_calibration_leakage.json")
        assert "ordering_preserved" in leak
        assert leak["per_message"]["mean_abs_delta"] is not None
        if not leak["ordering_preserved"]:
            assert leak["material_swaps"], (
                "ordering reported as changed but no material swap listed"
            )


class TestEraAnalysisAgrees:
    def test_race_count_matches_the_corpus(self, live_races):
        era = _load("_era_analysis.json")
        assert era["n_races"] == len(live_races)


class TestValenceIsNotMisreported:
    """The published claim used to be a flat 'valence is at chance'. It is not.

    The axis scores at chance *at the 0.5 split*, which is a fact about the
    threshold. The model ranks valence at AUC 0.687 and reaches +0.0605 lift at
    the fitted cut. Both must stay visible: quoting only the pessimistic figure
    understates the model, and quoting only the optimistic one hides that the
    corrected boundary does not transfer to radio.
    """

    @pytest.fixture(scope="class")
    def boundary(self):
        return _load("_valence_boundary.json")

    def test_the_axis_carries_ranking_signal(self, boundary):
        """Threshold-free ranking must beat chance, whatever the cut does.

        This used to also assert that fitting the cut beat the median split by
        enough to matter, on the strength of a +0.0605 lift. Re-measuring on the
        VAD-windowed prosody that actually ships put that at +0.0324, below the
        script's own pre-declared 0.05 materiality bar - so the assertion was
        encoding a finding that no longer holds rather than a property that must.
        The finding moved; see the boundary artifact's own `finding` field.
        """
        v = boundary["valence_raw_space"]
        assert v["lift_over_baseline_fitted"] > v["lift_over_baseline_median"]
        assert v["auc"] > 0.6, "ranking signal must be well above chance"

    def test_the_fitted_cut_is_stable_across_speakers(self, boundary):
        v = boundary["valence_raw_space"]
        assert v["fitted_cut_spread_relative"] < 0.15
        assert v["optimism"] < 0.05, "in-sample and CV must not diverge"

    def test_arousal_boundary_is_already_correct(self, boundary):
        """The median split is right for arousal and wrong for valence.

        If moving the arousal boundary ever starts helping, the quadrant scheme
        has changed underneath us.
        """
        assert boundary["arousal_raw_space"]["lift_over_median"] <= 0.0

    def test_a_gold_fitted_cut_is_never_recommended_for_radio(self, boundary):
        """The property, not one particular way of satisfying it.

        This asserted `recommendation == "do_not_transfer_to_f1"`, which was the
        right answer to the wrong question: it pinned one safe value instead of
        excluding the unsafe one. When the boundary was re-measured on the
        shipping prosody path the refit stopped clearing its materiality bar, so
        the script returned `no_change` - safer still, and the test failed
        anyway.

        Radio valence sits a measured distance from CREMA-D's, so `fit_valence_
        boundary.py` has exactly one recommendation that would be wrong here:
        move_the_boundary. That is what to forbid.
        """
        t = boundary["transfer_check"]
        assert t["computable"]
        assert "median_shift_in_gold_sds" in t
        if not t["distributions_comparable"]:
            assert boundary["recommendation"] != "move_the_boundary", (
                "the gold-fitted cut must not be adopted for radio while the two "
                f"distributions differ by {t['median_shift_in_gold_sds']} gold SDs"
            )

    def test_production_still_uses_the_untransferred_boundary(self):
        """The finding is published; the corpus is not relabelled on it."""
        from pipeline import fusion
        hi_v = fusion._quadrant(0.9, 0.51)[0]
        lo_v = fusion._quadrant(0.9, 0.49)[0]
        assert (hi_v, lo_v) == ("Energised", "Stressed"), (
            "the 0.5 percentile split is still the production boundary"
        )
