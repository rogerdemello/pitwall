"""The scorer must not be able to flatter the rebuild.

`_preregistration.json` fixed four hypotheses before the run. This suite holds
the three places where scoring them could go quietly wrong:

  * H1's registered metric is a property of the audio file, so it returns the
    same number under any pipeline. If the scorer used it, H1 would fail
    spuriously - and if it substituted a better metric without saying so, the
    pre-registration would have been edited after the fact.
  * H3's intervention was never built and its precision clause has no
    measurement apparatus. Reporting its coverage number as a partial result
    would be this project's fourth false positive and the first self-inflicted
    one.
  * Every hypothesis compares two trees clip for clip. If the corpora differ,
    none of the comparisons mean what they say.

The tests build synthetic race trees so they assert the scorer's logic rather
than whatever happens to be in backend/races today.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

from data import v1_baseline as v1b  # noqa: E402
from data import v2_scoring  # noqa: E402

RACE = "2021_Abu_Dhabi_Grand_Prix"


def msg(i: int, **over) -> dict:
    m = {
        "id": f"clip_{i}", "audio_file": f"c{i}.mp3",
        "transcript": f"box box box {i}", "reference_transcription": "box box",
        "duration_s": 8.0, "dsi": 50, "state": "Calm", "speaker": "driver",
        "suppressed_stress": False, "recommendation": None,
    }
    m.update(over)
    return m


def tree(root: str, messages: list[dict], extra: dict | None = None) -> str:
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f"{RACE}.json"), "w", encoding="utf-8") as f:
        json.dump({"race_id": RACE, "messages": messages}, f)
    for name, payload in (extra or {}).items():
        with open(os.path.join(root, name), "w", encoding="utf-8") as f:
            json.dump(payload, f)
    return root


PREREG = {
    "registered_against_commit": "8eee02d",
    "confirmatory_hypotheses": [
        {"id": "H1", "claim": "truncation removed", "metric": "clips_truncated",
         "baseline": 92, "threshold": "exactly 0", "if_it_fails": "report it"},
        {"id": "H2", "claim": "hallucination no worse", "metric": "rate",
         "baseline": 0.0269, "threshold": "<= 0.0269", "if_it_fails": "report it"},
        {"id": "H3", "claim": "attribution improves", "metric": "unknown_share",
         "baseline": 0.525, "threshold": "< 0.35 at >= 90% precision",
         "if_it_fails": "report the curve"},
        {"id": "H4", "outcome": "NOT SUPPORTED - abandoned"},
    ],
    "the_single_confirmatory_test": {
        "baseline": {"pooled_r": 0.0428, "n": 1155,
                     "tercile_gap_s": -0.071, "sign_test_p": 0.7376},
    },
}


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Point every path in both modules at a scratch tree."""
    races = str(tmp_path / "races")
    races_v1 = str(tmp_path / "races_v1")
    os.makedirs(races, exist_ok=True)
    with open(os.path.join(races, "_preregistration.json"), "w", encoding="utf-8") as f:
        json.dump(PREREG, f)
    with open(os.path.join(races, "_v1_baseline.json"), "w", encoding="utf-8") as f:
        json.dump({"truncation": {"clips_truncated": 92},
                   "hallucination": {"rate": 0.0269},
                   "speaker_attribution": {"unknown_share": 0.525}}, f)
    for mod in (v2_scoring,):
        monkeypatch.setattr(mod, "RACES", races)
        monkeypatch.setattr(mod, "RACES_V1", races_v1)
        monkeypatch.setattr(mod, "RAW", str(tmp_path / "raw"))
        monkeypatch.setattr(mod, "RAW_V1", str(tmp_path / "raw_v1"))
        monkeypatch.setattr(mod, "OUT", os.path.join(races, "_v2_scoring.json"))
    monkeypatch.setattr(v1b, "RACES", races)
    return {"races": races, "races_v1": races_v1, "tmp": tmp_path}


class TestH1DoesNotUseTheRegisteredMetric:
    def test_it_declares_the_registered_metric_invalid_for_v2(self, wired):
        # Every clip is 40s long in BOTH trees, so `duration_s > 30` returns the
        # same count either way. That is precisely why it cannot score v2.
        long_ = [msg(i, duration_s=40.0) for i in range(4)]
        tree(wired["races_v1"], long_)
        tree(wired["races"], long_)

        out = v2_scoring.build()
        h1 = out["hypotheses"][0]
        assert h1["registered_metric_valid_for_v2"] is False
        assert "duration_s" in h1["why_the_registered_metric_cannot_score_v2"]
        assert h1["v2"]["clips_over_30s"] == 4, "the audio property is still reported"

    def test_the_registered_baseline_is_carried_unchanged(self, wired):
        tree(wired["races_v1"], [msg(0)])
        tree(wired["races"], [msg(0)])
        h1 = v2_scoring.build()["hypotheses"][0]
        assert h1["baseline"]["clips_truncated"] == 92
        assert h1["threshold"] == "exactly 0"

    def test_paired_word_counts_are_reported_for_the_at_risk_clips(self, wired):
        tree(wired["races_v1"], [msg(0, duration_s=40.0, transcript="one two")])
        tree(wired["races"], [msg(0, duration_s=40.0, transcript="one two three four")])
        h1 = v2_scoring.build()["hypotheses"][0]
        # No raw trees here, so pairing is empty - but the section must exist and
        # be honest about it rather than implying a measurement.
        assert "paired_evidence" in h1
        assert h1["paired_evidence"]["n"] == 0


class TestH3IsNotTested:
    def test_status_is_not_tested(self, wired):
        tree(wired["races_v1"], [msg(0, speaker="unknown"), msg(1)])
        tree(wired["races"], [msg(0), msg(1)])          # coverage "improved"
        h3 = v2_scoring.build()["hypotheses"][2]
        assert h3["status"] == "NOT TESTED"
        assert h3["registered_intervention_built"] is False
        assert h3["held_out_judgements_available"] is False

    def test_the_coverage_number_is_quarantined(self, wired):
        """It is reported, but nowhere a reader could mistake it for a verdict."""
        tree(wired["races_v1"], [msg(0, speaker="unknown"), msg(1)])
        tree(wired["races"], [msg(0), msg(1)])
        h3 = v2_scoring.build()["hypotheses"][2]

        d = h3["descriptive_only"]
        assert d["these_do_not_constitute_a_verdict_on_H3"] is True
        assert d["unknown_share_v2"] < d["unknown_share_v1_recomputed_at_rebuild"]
        assert d["precision"] is None
        # The improvement must not appear as a status anywhere.
        assert "MET" not in h3["status"]

    def test_it_says_why_coverage_alone_would_mislead(self, wired):
        tree(wired["races_v1"], [msg(0)])
        tree(wired["races"], [msg(0)])
        h3 = v2_scoring.build()["hypotheses"][2]
        assert "longer transcript" in h3["why_coverage_alone_is_not_reported_as_a_result"]
        assert h3["not_tested_is_not_failed"]

    def test_it_is_still_listed_as_an_unmet_registration(self, wired):
        tree(wired["races_v1"], [msg(0)])
        tree(wired["races"], [msg(0)])
        out = v2_scoring.build()
        hyp = [r for r in out["registered_regressions"]
               if "hypotheses failing" in r["registered"]][0]
        assert hyp["regressed"] is True, "NOT TESTED still counts as unmet"


class TestUnpairedCorporaInvalidateEverything:
    def test_a_differing_id_set_is_detected(self, wired):
        tree(wired["races_v1"], [msg(0), msg(1)])
        tree(wired["races"], [msg(0), msg(2)])
        out = v2_scoring.build()
        assert out["corpus_pairing"]["paired"] is False
        assert out["corpus_pairing"]["n_only_in_v1"] == 1
        assert out["corpus_pairing"]["n_only_in_v2"] == 1

    def test_no_hypothesis_is_reported_as_a_result(self, wired):
        tree(wired["races_v1"], [msg(0), msg(1)])
        tree(wired["races"], [msg(0), msg(2)])
        out = v2_scoring.build()
        for h in out["hypotheses"][:3]:
            assert h["status"] == "NOT TESTED", h["id"]
        assert "not the same clips" in out["summary"]

    def test_main_exits_nonzero(self, wired, monkeypatch):
        tree(wired["races_v1"], [msg(0), msg(1)])
        tree(wired["races"], [msg(0), msg(2)])
        assert v2_scoring.main([]) == 1


class TestTheSharedMetricsComeFromOneDefinition:
    def test_hallucination_uses_the_baseline_module_families(self, wired):
        # "you" alone is v1_baseline's bare_you family - the squelch artifact.
        tree(wired["races_v1"], [msg(0, transcript="you"), msg(1)])
        tree(wired["races"], [msg(0, transcript="clear track"), msg(1)])
        h2 = v2_scoring.build()["hypotheses"][1]
        assert h2["the_named_test"]["cohort_size"] == 1
        assert h2["the_named_test"]["genuinely_resolved"] == 1
        assert h2["the_named_test"]["passed"] is True
        assert h2["paired"]["n_fixed"] == 1
        assert h2["v2_rate"] < h2["baseline_rate"] or h2["v2_rate"] == 0.0

    def test_the_named_cohort_is_followed_by_id_not_by_family(self, wired):
        """Per-family counts hide a hallucination that merely changed wording.

        The pre-registration named the near-silent clips decoding to "you" as
        the specific test. Counted per family they read 51 -> 0 and look
        eliminated; followed by clip id they are almost all still inventing
        text, just saying "Thank you." instead. The rate barely moves for
        exactly that reason, so the rate alone cannot be the whole verdict.
        """
        tree(wired["races_v1"], [msg(0, transcript="you"), msg(1, transcript="you"),
                                 msg(2)])
        tree(wired["races"], [msg(0, transcript="Thank you."),   # renamed, not fixed
                              msg(1, transcript="box box now"),  # genuinely fixed
                              msg(2)])
        h2 = v2_scoring.build()["hypotheses"][1]
        named = h2["the_named_test"]
        assert named["cohort_size"] == 2
        assert named["still_hallucinating_in_v2"] == 1
        assert named["genuinely_resolved"] == 1
        assert named["what_they_became"] == {"thanks_only": 1}
        assert named["passed"] is False

    def test_meeting_the_threshold_does_not_hide_failing_the_named_test(self, wired):
        tree(wired["races_v1"], [msg(i, transcript="you") for i in range(2)]
                                + [msg(i) for i in range(2, 6)])
        tree(wired["races"], [msg(i, transcript="Thank you.") for i in range(2)]
                             + [msg(i) for i in range(2, 6)])
        h2 = v2_scoring.build()["hypotheses"][1]
        assert h2["status"] in ("MET", "MET AT TARGET"), "rate did not worsen"
        assert h2["threshold_met_but_named_test_failed"] is True

    def test_a_worsening_rate_is_a_registered_regression(self, wired):
        tree(wired["races_v1"], [msg(i) for i in range(4)])
        tree(wired["races"], [msg(0, transcript="you"), msg(1, transcript="you"),
                              msg(2), msg(3)])
        out = v2_scoring.build()
        h2 = out["hypotheses"][1]
        assert h2["status"] == "NOT MET"
        reg = [r for r in out["registered_regressions"]
               if "hallucination" in r["registered"]][0]
        assert reg["regressed"] is True


class TestReportingDiscipline:
    def test_speaker_precision_is_unperformed_not_passed(self, wired):
        tree(wired["races_v1"], [msg(0)])
        tree(wired["races"], [msg(0)])
        out = v2_scoring.build()
        reg = [r for r in out["registered_regressions"]
               if "precision" in r["registered"]][0]
        assert reg["measured"] is False
        assert reg["regressed"] is None, "must not read as a pass"

    def test_the_confirmatory_result_is_null_until_it_is_run(self, wired):
        tree(wired["races_v1"], [msg(0)])
        tree(wired["races"], [msg(0)])
        out = v2_scoring.build()
        assert out["confirmatory_result"] is None
        assert "exactly once" in out["confirmatory_result_note"]

    def test_h4_is_not_re_adjudicated(self, wired):
        tree(wired["races_v1"], [msg(0)])
        tree(wired["races"], [msg(0)])
        h4 = v2_scoring.build()["hypotheses"][3]
        assert h4["status"] == "ADJUDICATED BEFORE THE REBUILD"
        assert h4["outcome"].startswith("NOT SUPPORTED")

    def test_the_registered_baseline_is_never_edited(self, wired):
        tree(wired["races_v1"], [msg(0)],
             extra={"_corpus_analysis.json": {"pooled_r": 0.0446, "paired_n": 1155,
                                              "tercile_gap_s": -0.086,
                                              "sign_test_p": 0.5764}})
        tree(wired["races"], [msg(0)])
        rec = v2_scoring.build()["confirmatory_baseline_reconciliation"]
        assert rec["registered_in_prereg"]["pooled_r"] == 0.0428, "prereg untouched"
        assert rec["on_disk_before_the_rebuild"]["pooled_r"] == 0.0446
        assert rec["they_disagree"] is True
        assert rec["the_registered_value_is_not_edited"] is True
        assert rec["comparison_point_for_the_confirmatory_run"] == \
            "on_disk_before_the_rebuild"

    def test_no_disagreement_is_reported_when_there_is_none(self, wired):
        tree(wired["races_v1"], [msg(0)],
             extra={"_corpus_analysis.json": {"pooled_r": 0.0428, "paired_n": 1155,
                                              "tercile_gap_s": -0.071,
                                              "sign_test_p": 0.7376}})
        tree(wired["races"], [msg(0)])
        rec = v2_scoring.build()["confirmatory_baseline_reconciliation"]
        assert rec["they_disagree"] is False
        assert rec["moved_at_commit"] is None


class TestItRefusesRatherThanGuessing:
    def test_it_refuses_without_a_frozen_v1(self, wired):
        tree(wired["races"], [msg(0)])
        with pytest.raises(SystemExit, match="races_v1"):
            v2_scoring.build()

    def test_it_refuses_without_a_preregistration(self, wired):
        os.remove(os.path.join(wired["races"], "_preregistration.json"))
        tree(wired["races_v1"], [msg(0)])
        tree(wired["races"], [msg(0)])
        with pytest.raises(SystemExit, match="preregistration"):
            v2_scoring.build()


class TestTheFrozenBaselineCannotBeOverwritten:
    def test_v1_baseline_refuses_once_v1_is_frozen(self, tmp_path, monkeypatch):
        """Running v1_baseline.py after the import would score v2 against itself."""
        races = tmp_path / "races"
        races.mkdir()
        (races / "_v1_baseline.json").write_text("{}", encoding="utf-8")
        (tmp_path / "races_v1").mkdir()
        monkeypatch.setattr(v1b, "RACES", str(races))
        monkeypatch.setattr(v1b, "RACES_V1", str(tmp_path / "races_v1"))
        monkeypatch.setattr(v1b, "OUT", str(races / "_v1_baseline.json"))
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            v1b.main()
