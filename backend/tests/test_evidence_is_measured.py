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


class TestCalibrationIsPooled:
    """Every cross-race claim depends on one shared calibration.

    Per-race calibration centres each race on 50.0 by construction, which would
    make the contrast test vacuous rather than merely wrong.
    """

    def test_all_races_share_the_pooled_calibration(self, live_races):
        sources = {r["calibration_source"] for r in live_races.values()}
        assert sources == {"pooled"}, f"mixed calibration sources: {sources}"


class TestEraAnalysisAgrees:
    def test_race_count_matches_the_corpus(self, live_races):
        era = _load("_era_analysis.json")
        assert era["n_races"] == len(live_races)
