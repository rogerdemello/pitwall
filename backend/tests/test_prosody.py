"""Per-window prosody, and the aggregation choice it unlocks.

v1 mean-pooled inside `prosody.analyse` over 15-second windows spanning the
whole clip, silence included. Three consequences, all measurable:

  the model was run far outside its training regime (MSP-Podcast utterances
  average ~5s), on 304 clips exceeding 15s and one of 105s;

  channel noise and dead air were averaged into the driver's affect score;

  the aggregation choice was baked into an hour of GPU inference, so revisiting
  it meant re-running the corpus.

These tests pin the window construction and the aggregation contract. The ones
needing model weights are marked slow.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

from pipeline import aggregate, prosody, vad  # noqa: E402

SR = prosody.SAMPLE_RATE


def tone(seconds, freq=200.0, amp=0.2):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestWindowing:
    def _spans(self, *pairs):
        return [vad.Span(a, b) for a, b in pairs]

    def test_windows_stay_inside_their_span(self):
        audio = tone(30.0)
        w = prosody._spans_to_windows(audio, self._spans((5.0, 15.0)), SR)
        assert w, "no windows produced"
        assert all(t0 >= 4.99 and t1 <= 15.01 for t0, t1, _ in w)

    def test_windows_never_cross_a_span_boundary(self):
        """A window straddling two transmissions averages two speakers."""
        audio = tone(30.0)
        spans = self._spans((0.0, 6.0), (20.0, 26.0))
        for t0, t1, _ in prosody._spans_to_windows(audio, spans, SR):
            assert (t1 <= 6.01) or (t0 >= 19.99), f"window {t0}-{t1} crosses the gap"

    def test_a_long_span_produces_several_windows(self):
        audio = tone(40.0)
        w = prosody._spans_to_windows(audio, self._spans((0.0, 40.0)), SR)
        assert len(w) > 5, f"only {len(w)} windows over 40s at {prosody.HOP_S}s hop"

    def test_windows_overlap_by_the_hop(self):
        audio = tone(20.0)
        w = prosody._spans_to_windows(audio, self._spans((0.0, 20.0)), SR)
        starts = [t0 for t0, _, _ in w]
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        assert all(abs(g - prosody.HOP_S) < 0.05 for g in gaps), gaps

    def test_a_very_short_span_still_yields_one_window(self):
        """A two-word transmission must still be scored, not dropped."""
        audio = tone(5.0)
        w = prosody._spans_to_windows(audio, self._spans((1.0, 1.6)), SR)
        assert len(w) == 1

    def test_no_spans_means_no_windows(self):
        assert prosody._spans_to_windows(tone(5.0), [], SR) == []


class TestSlopeAndPercentiles:
    def test_slope_is_positive_when_arousal_rises(self):
        assert prosody._slope([0, 1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4, 0.5]) > 0

    def test_slope_is_negative_when_it_falls(self):
        assert prosody._slope([0, 1, 2, 3, 4], [0.5, 0.4, 0.3, 0.2, 0.1]) < 0

    def test_slope_is_zero_when_flat(self):
        assert prosody._slope([0, 1, 2, 3], [0.3] * 4) == 0.0

    def test_slope_needs_enough_points(self):
        assert prosody._slope([0, 1], [0.1, 0.9]) == 0.0

    def test_percentiles_are_ordered(self):
        v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        assert prosody._pct(v, 0.10) < prosody._pct(v, 0.90)

    def test_percentile_of_empty_is_zero_not_a_crash(self):
        assert prosody._pct([], 0.9) == 0.0

    def test_p90_is_less_biased_by_count_than_max(self):
        """Raw max over N is upward-biased in N; that is why peak became p90.

        This is a claim about expectation, so it is tested over many draws.
        A single pair of samples is noise: max is a high-variance statistic and
        one draw can go either way.
        """
        rng = np.random.default_rng(11)
        max_gaps, p90_gaps = [], []
        for _ in range(400):
            small = list(rng.uniform(0, 1, 10))
            large = list(rng.uniform(0, 1, 50))
            max_gaps.append(max(large) - max(small))
            p90_gaps.append(prosody._pct(large, 0.9) - prosody._pct(small, 0.9))
        # A 5x window count lifts the max by a clear margin, and p90 by far less.
        assert float(np.mean(max_gaps)) > 0.05, "max should be biased upward by N"
        assert abs(float(np.mean(p90_gaps))) < float(np.mean(max_gaps)) / 2


class TestAffectContract:
    def test_v1_fields_survive(self):
        a = prosody.Affect(0.5, 0.5, 0.5)
        assert a.arousal == 0.5 and a.peak_arousal == 0.0 and a.windows == 1

    def test_new_fields_default(self):
        a = prosody.Affect(0.5, 0.5, 0.5)
        assert a.window_scores == [] and a.arousal_slope == 0.0

    def test_serialises_with_windows(self):
        a = prosody.Affect(0.5, 0.5, 0.5, window_scores=[
            prosody.Window(0, 0.0, 4.0, 0.6, 0.5, 0.4)])
        d = a.to_dict()
        assert len(d["window_scores"]) == 1
        assert d["window_scores"][0]["t0"] == 0.0

    def test_window_marks_short_ones(self):
        w = prosody.Window(0, 0.0, 0.8, 0.5, 0.5, 0.5, short=True)
        assert w.to_dict()["short"] is True

    def test_window_omits_short_flag_when_false(self):
        assert "short" not in prosody.Window(0, 0.0, 4.0, 0.5, 0.5, 0.5).to_dict()


class TestAggregation:
    WINDOWS = [
        {"i": 0, "t0": 0.0, "t1": 4.0, "arousal": 0.20},
        {"i": 1, "t0": 2.0, "t1": 6.0, "arousal": 0.25},
        {"i": 2, "t0": 4.0, "t1": 8.0, "arousal": 0.30},
        {"i": 3, "t0": 6.0, "t1": 10.0, "arousal": 0.90},
    ]

    def test_mean_is_the_incumbent(self):
        assert aggregate.DEFAULT == "mean"

    def test_mean_matches_arithmetic(self):
        assert aggregate.apply("mean", self.WINDOWS, "arousal") == pytest.approx(0.4125)

    def test_p90_catches_the_spike_the_mean_dilutes(self):
        """The case the whole module exists for: calm for most of a message,
        then a shout. The mean says calm; p90 says something happened."""
        assert aggregate.apply("p90", self.WINDOWS, "arousal") > \
            aggregate.apply("mean", self.WINDOWS, "arousal")

    def test_median_ignores_the_spike(self):
        assert aggregate.apply("median", self.WINDOWS, "arousal") == pytest.approx(0.275)

    def test_last_window(self):
        assert aggregate.apply("last_window", self.WINDOWS, "arousal") == 0.9

    def test_max_run_needs_a_sustained_spike(self):
        """One loud window should not score as high as two consecutive ones."""
        one = aggregate.apply("max_run2", self.WINDOWS, "arousal")
        sustained = aggregate.apply("max_run2", self.WINDOWS[:3] + [
            {"i": 3, "t0": 6.0, "t1": 10.0, "arousal": 0.90},
            {"i": 4, "t0": 8.0, "t1": 12.0, "arousal": 0.90}], "arousal")
        assert sustained > one

    def test_trimmed_falls_back_on_short_input(self):
        assert aggregate.apply("trimmed", self.WINDOWS, "arousal") == \
            aggregate.apply("mean", self.WINDOWS, "arousal")

    def test_unknown_strategy_raises(self):
        with pytest.raises(KeyError):
            aggregate.apply("vibes", self.WINDOWS, "arousal")

    def test_empty_windows_do_not_crash(self):
        for name in aggregate.STRATEGIES:
            assert aggregate.apply(name, [], "arousal") == 0.0

    def test_all_strategies_returns_every_candidate(self):
        out = aggregate.all_strategies(self.WINDOWS, "arousal")
        assert set(out) == set(aggregate.STRATEGIES)


@pytest.mark.slow
class TestAgainstAudio:
    def test_silence_returns_neutral_and_says_so(self):
        """Scoring channel noise and calling it a driver state is the bug."""
        a = prosody.analyse(np.zeros(2 * SR, dtype=np.float32))
        assert a.windows == 0 and a.voiced_fraction == 0.0
        assert a.arousal == 0.5

    def test_real_audio_produces_multiple_windows(self):
        a = prosody.analyse(tone(20.0), use_vad=False)
        assert a.windows > 1
        assert len(a.window_scores) == a.windows

    def test_scores_stay_in_range(self):
        a = prosody.analyse(tone(10.0), use_vad=False)
        for w in a.window_scores:
            assert 0.0 <= w.arousal <= 1.0 and 0.0 <= w.valence <= 1.0

    def test_headline_is_still_the_mean_of_the_windows(self):
        """Stage 2 keeps working unchanged; alternatives stay recoverable."""
        a = prosody.analyse(tone(15.0), use_vad=False)
        expected = float(np.mean([w.arousal for w in a.window_scores]))
        assert a.arousal == pytest.approx(expected, abs=1e-3)
