"""Tests for the logic that is easy to break and expensive to get wrong.

Deliberately not testing the models - they are third-party and slow. These cover
the project's own reasoning: which laps count, how percentiles are computed, when
a suppressed-stress flag is allowed to fire, who is speaking, and the guards that
stop us reporting noise as a finding.

Several of these encode bugs that actually happened during the build.

    pytest backend/tests -q
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.race_data import _is_representative  # noqa: E402
from pipeline import analysis, fusion, speaker, strategy  # noqa: E402
from pipeline.artifacts import is_race_file, race_ids  # noqa: E402
from pipeline.calibration import Calibrator  # noqa: E402
from pipeline.prosody import Affect  # noqa: E402
from pipeline.sentiment import TextSentiment  # noqa: E402


def lap(laptime=90.0, pit_in=None, pit_out=None, status="1"):
    return pd.Series({
        "LapTime": pd.Timedelta(seconds=laptime) if laptime is not None else pd.NaT,
        "PitInTime": pd.Timedelta(seconds=pit_in) if pit_in else pd.NaT,
        "PitOutTime": pd.Timedelta(seconds=pit_out) if pit_out else pd.NaT,
        "TrackStatus": status,
    })


class TestRepresentativeLap:
    """A 20s pit stop is not a mood. This filter is why the stress-vs-pace
    number stopped reporting +18.8s."""

    def test_green_flag_racing_lap_counts(self):
        assert _is_representative(lap()) is True

    def test_pit_in_lap_excluded(self):
        assert _is_representative(lap(pit_in=30)) is False

    def test_pit_out_lap_excluded(self):
        assert _is_representative(lap(pit_out=30)) is False

    def test_safety_car_excluded(self):
        assert _is_representative(lap(status="4")) is False

    def test_mixed_status_excluded(self):
        # '14' means the lap saw both green and safety car.
        assert _is_representative(lap(status="14")) is False

    def test_missing_laptime_excluded(self):
        assert _is_representative(lap(laptime=None)) is False

    def test_empty_status_excluded(self):
        assert _is_representative(lap(status="")) is False


class TestCalibrator:
    def test_percentile_is_monotonic(self):
        c = Calibrator.fit([{"arousal": v, "valence": v, "dominance": v}
                            for v in [0.1, 0.2, 0.3, 0.4, 0.5]])
        assert c.pct_arousal(0.1) < c.pct_arousal(0.3) < c.pct_arousal(0.5)

    def test_ties_map_consistently(self):
        c = Calibrator.fit([{"arousal": 0.5, "valence": 0.5, "dominance": 0.5}] * 4)
        assert c.pct_arousal(0.5) == c.pct_arousal(0.5)

    def test_empty_corpus_is_neutral_not_a_crash(self):
        assert Calibrator().pct_arousal(0.7) == 0.5

    def test_spreads_a_narrow_band(self):
        """The whole point: raw scores clustered in 0.55-0.80 must still use the
        full 0-1 range after calibration."""
        vals = [0.55 + i * 0.005 for i in range(50)]
        c = Calibrator.fit([{"arousal": v, "valence": v, "dominance": v} for v in vals])
        assert c.pct_arousal(vals[0]) < 0.1
        assert c.pct_arousal(vals[-1]) > 0.9


def affect(a, v, d=0.5):
    return Affect(arousal=a, valence=v, dominance=d)


def text(pol):
    pos = max(0.0, pol)
    neg = max(0.0, -pol)
    label = "positive" if pol > 0.2 else "negative" if pol < -0.2 else "neutral"
    return TextSentiment(label=label, negative=neg, neutral=1 - pos - neg, positive=pos)


class TestFusion:
    def test_high_arousal_low_valence_is_stressed(self):
        s = fusion.fuse(affect(0.9, 0.2), text(0.0), transcript="I have no grip at all")
        assert s.state == "Stressed"
        assert s.dsi > 60

    def test_low_arousal_low_valence_is_fatigued(self):
        assert fusion.fuse(affect(0.2, 0.3), text(0.0),
                           transcript="yeah ok understood").state == "Fatigued"

    def test_calm_voice_calm_words_not_flagged(self):
        s = fusion.fuse(affect(0.3, 0.7), text(0.5), transcript="yeah all good mate")
        assert s.suppressed_stress is False

    def test_reassuring_words_over_negative_voice_is_flagged(self):
        """The signature feature: words say fine, voice does not."""
        s = fusion.fuse(affect(0.8, 0.1), text(0.9), transcript="yeah I am absolutely fine")
        assert s.suppressed_stress is True

    def test_short_transcript_never_flags(self):
        """Two words carry no reliable sentiment; this guard stopped the detector
        firing on radio squawks."""
        s = fusion.fuse(affect(0.8, 0.1), text(0.9), transcript="yeah ok")
        assert s.suppressed_stress is False

    def test_negative_words_and_negative_voice_agree(self):
        s = fusion.fuse(affect(0.9, 0.1), text(-0.8),
                        transcript="the car is completely undriveable now")
        assert s.suppressed_stress is False

    def test_dsi_stays_in_range(self):
        for a in (0.0, 0.5, 1.0):
            for v in (0.0, 0.5, 1.0):
                assert 0 <= fusion.fuse(affect(a, v), text(0.0), transcript="a b c").dsi <= 100

    def test_dominance_does_not_affect_the_index(self):
        """It was removed after measuring corr(arousal, dominance) = 0.97.

        At that collinearity it is not a third dimension, it is a rescaled
        arousal discount carrying its own coefficient. It is still reported.
        """
        low = fusion.fuse(affect(0.7, 0.3, d=0.0), text(0.0),
                          transcript="a b c")
        high = fusion.fuse(affect(0.7, 0.3, d=1.0), text(0.0),
                           transcript="a b c")
        assert low.dsi == high.dsi
        assert low.dominance_pct != high.dominance_pct, "still reported, just unused"

    def test_index_is_the_two_stated_terms(self):
        """Guards the published formula against silent drift."""
        for a, v in ((0.0, 1.0), (1.0, 0.0), (0.6, 0.4), (0.5, 0.5)):
            expected = round(max(0.0, min(1.0, 0.55 * a + 0.45 * (1 - v))) * 100)
            assert fusion.fuse(affect(a, v), text(0.0), transcript="a b c").dsi == expected


class TestSpeaker:
    def test_vocative_is_engineer(self):
        assert speaker.classify("Okay, Lewis, so box box")[0] == "engineer"

    def test_first_person_is_driver(self):
        assert speaker.classify("I have no grip, my rears are gone")[0] == "driver"

    def test_second_person_is_engineer(self):
        assert speaker.classify("You are doing a good job, keep it up")[0] == "engineer"

    def test_third_person_mention_is_not_attributed(self):
        """'Checo is a legend' is Verstappen talking about Perez, not the pit wall
        addressing him. Guessing here was a real bug."""
        assert speaker.classify("Checo is a legend. Absolute animal.")[0] != "engineer"

    def test_too_short_is_unknown(self):
        assert speaker.classify("copy")[0] == "unknown"


class TestStrategy:
    def _msg(self, dsi, delta, tyre, speaker_="driver", suppressed=False):
        return {
            "dsi": dsi, "suppressed_stress": suppressed, "speaker": speaker_,
            "lap": {"in_race": True, "delta_to_median_s": delta, "tyre_life": tyre,
                    "compound": "SOFT", "lap_number": 20, "is_representative": True},
        }

    def test_engineer_messages_never_trigger_a_call(self):
        hist = [self._msg(90, 1.0, 25, speaker_="engineer")]
        assert strategy.recommend("VER", hist) is None

    def test_stress_plus_pace_loss_plus_old_tyres_says_box(self):
        rec = strategy.recommend("VER", [self._msg(80, 0.9, 25)])
        assert rec is not None and rec.severity == "act"

    def test_calm_and_quick_says_nothing(self):
        assert strategy.recommend("VER", [self._msg(30, -0.3, 5)]) is None

    def test_recommendation_carries_its_evidence(self):
        rec = strategy.recommend("VER", [self._msg(80, 0.9, 25)])
        assert rec.evidence and any("DSI" in e for e in rec.evidence)


class TestRaceFileFilter:
    """races/ holds three kinds of file and only one of them is a race.

    This has caused a 500 twice: first when eval sidecars were added, then again
    when the diarization experiment landed with an underscore prefix.

    The definition now lives in pipeline/artifacts.py. It used to be duplicated
    in six places in two incompatible forms - main.py's required a .json suffix,
    the five inlined copies did not - so `notes.txt` was a race to five of them.
    """

    def test_real_race_file_accepted(self):
        assert is_race_file("2021_Abu_Dhabi_Grand_Prix.json") is True

    @pytest.mark.parametrize("name", [
        "2021_Abu_Dhabi_Grand_Prix.calibration.json",
        "2021_Abu_Dhabi_Grand_Prix.asr_eval.json",
        "2021_Abu_Dhabi_Grand_Prix.affect_eval.json",
    ])
    def test_sidecars_rejected(self, name):
        assert is_race_file(name) is False

    @pytest.mark.parametrize("name", [
        "_pooled.calibration.json",
        "_diarization_experiment.json",
        "_corpus_finding.json",
    ])
    def test_corpus_artifacts_rejected(self, name):
        assert is_race_file(name) is False

    @pytest.mark.parametrize("name", ["notes.txt", "README.md", "manifest"])
    def test_non_json_rejected(self, name):
        assert is_race_file(name) is False

    def test_main_uses_the_shared_definition(self):
        """main.py must not carry its own copy again."""
        import main
        assert main._is_race_file is is_race_file


class TestLagGuard:
    def test_underpowered_lag_cannot_win(self):
        """A big r on ten observations is noise. Selecting the max over four lags
        is a multiple comparison, and this guard is what stops us reporting
        r=0.62 from n=10 as a predictive finding."""
        msgs = []
        for i in range(60):
            msgs.append({
                "dsi": 50 + (i % 20), "driver_code": "VER", "speaker": "driver",
                "lap": {"in_race": True, "is_representative": True,
                        "lap_number": i + 1, "delta_to_median_s": (i % 7) * 0.1},
            })
        out = analysis.lag_analysis(msgs)
        assert out["best_lag"] is not None
        for k in out["underpowered_lags"]:
            assert out["by_lag"][k]["n"] < out["min_n_for_consideration"]
            assert out["best_lag"] != k

    def test_no_data_does_not_crash(self):
        out = analysis.lag_analysis([])
        assert out["best_lag"] is None


class TestSignificanceGuards:
    """These guards each exist because a spurious finding got through.

    v1 let r=0.62 from n=10 through. v2 added a minimum n and an effect-size
    margin, and still let r=-0.25 from n=56 (p=0.06) through. v3 requires
    Bonferroni-corrected significance and a positive direction.
    """

    def test_negative_correlation_is_never_predictive(self):
        """A negative r means stress preceded FASTER laps - the opposite of the
        hypothesis - and must never be reported as predictive."""
        msgs = []
        for i in range(200):
            # Higher DSI -> systematically quicker three laps later.
            dsi = 20 + (i % 60)
            msgs.append({
                "dsi": dsi, "driver_code": "VER", "speaker": "driver",
                "lap": {"in_race": True, "is_representative": True,
                        "lap_number": i + 1, "delta_to_median_s": -dsi / 40},
            })
        out = analysis.lag_analysis(msgs)
        if out["best_r"] is not None and out["best_r"] < 0:
            assert out["predictive"] is False

    def test_bonferroni_alpha_is_reported(self):
        out = analysis.lag_analysis([])
        assert out["bonferroni_alpha"] <= 0.05 / 2

    def test_sign_test_rejects_a_coin_flip(self):
        from pipeline.analysis import _sign_test_p
        # 23 of 37 sounds like a majority; it is not distinguishable from chance.
        assert _sign_test_p(23, 37) > 0.05
        # A real lopsided split is.
        assert _sign_test_p(34, 37) < 0.01

    def test_pearson_p_matches_known_value(self):
        from pipeline.analysis import _pearson_p
        # r=0.25, n=56 sits just above the conventional 0.05 threshold.
        p = _pearson_p(0.2512, 56)
        assert p is not None and 0.03 < p < 0.10

    def test_pearson_p_handles_degenerate_input(self):
        from pipeline.analysis import _pearson_p
        assert _pearson_p(None, 50) is None
        assert _pearson_p(0.5, 2) is None


class TestAxisBreakdown:
    """Scoring each dimension separately is what turned a vague '49% accuracy'
    into the project's sharpest finding: arousal works, valence is chance."""

    def _confusion(self, arousal_good: bool, valence_good: bool) -> dict:
        # 100 clips per true quadrant. Errors are directed at the neighbour that
        # shares the *other* axis, which is how real confusion actually looked.
        out = {s: {t: 0 for t in ["Calm", "Energised", "Stressed", "Fatigued"]}
               for s in ["Calm", "Energised", "Stressed", "Fatigued"]}
        valence_swap = {"Calm": "Fatigued", "Fatigued": "Calm",
                        "Energised": "Stressed", "Stressed": "Energised"}
        arousal_swap = {"Calm": "Energised", "Energised": "Calm",
                        "Stressed": "Fatigued", "Fatigued": "Stressed"}
        for s in out:
            out[s][s] += 60 if (arousal_good and valence_good) else 40
            out[s][valence_swap[s]] += 10 if valence_good else 50
            out[s][arousal_swap[s]] += 10 if arousal_good else 50
        return out

    def test_detects_a_weak_valence_axis(self):
        from data.eval_affect_gold import axis_breakdown
        res = axis_breakdown(self._confusion(arousal_good=True, valence_good=False))
        assert "valence" in res["near_chance_axes"]
        assert "arousal" not in res["near_chance_axes"]
        assert res["arousal_high_vs_low"]["lift"] > res["valence_negative_vs_positive"]["lift"]

    def test_detects_a_weak_arousal_axis(self):
        from data.eval_affect_gold import axis_breakdown
        res = axis_breakdown(self._confusion(arousal_good=False, valence_good=True))
        assert "arousal" in res["near_chance_axes"]

    def test_reports_baseline_alongside_accuracy(self):
        from data.eval_affect_gold import axis_breakdown
        res = axis_breakdown(self._confusion(True, True))
        for axis in ("arousal_high_vs_low", "valence_negative_vs_positive"):
            a = res[axis]
            assert a["majority_baseline"] >= 0.5  # binary split, never below half
            assert a["lift"] == pytest.approx(a["accuracy"] - a["majority_baseline"], abs=1e-9)


class TestCohensKappa:
    """Raw agreement is misleading when both raters favour one class. Kappa is
    what stopped a 'they agree 24% of the time' number being read as support."""

    def test_perfect_agreement_is_one(self):
        from data.eval_convergent import cohens_kappa
        pairs = [("Calm", "Calm")] * 10 + [("Stressed", "Stressed")] * 10
        assert cohens_kappa(pairs, ["Calm", "Stressed"]) == pytest.approx(1.0)

    def test_chance_agreement_is_about_zero(self):
        from data.eval_convergent import cohens_kappa
        # One rater spreads evenly, the other always says Stressed: they agree a
        # quarter of the time purely by accident.
        states = ["Calm", "Energised", "Stressed", "Fatigued"]
        pairs = [(s, "Stressed") for s in states for _ in range(25)]
        k = cohens_kappa(pairs, states)
        assert k is not None and abs(k) < 0.05

    def test_empty_input_is_none(self):
        from data.eval_convergent import cohens_kappa
        assert cohens_kappa([], ["Calm"]) is None

    def test_band_labels_are_ordered(self):
        from data.eval_convergent import band
        assert band(-0.1) == "worse than chance"
        assert band(0.1) == "slight"
        assert band(0.5) == "moderate"
        assert band(0.9) == "almost perfect"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
