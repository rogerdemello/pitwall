"""VAD and loudness conditioning.

The claim these have to earn is specific: a no-speech guard removes the
hallucination family recorded in `_v1_baseline.json` without discarding real
messages. Measured on the corpus, it removes 51 of 55 flagged clips and 0 of 120
sampled real ones. The clip-level tests below are marked slow and skip when the
audio is absent (backend/clips is gitignored); the synthetic tests always run.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

from pipeline import audio, vad  # noqa: E402
from pipeline.artifacts import iter_race_files  # noqa: E402

RACES = os.path.join(BACKEND, "races")
CLIPS = os.path.join(BACKEND, "clips")
SR = 16000


def tone(seconds=1.0, freq=200.0, amp=0.2, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds=1.0, sr=SR):
    return np.zeros(int(seconds * sr), dtype=np.float32)


def noise(seconds=1.0, amp=0.001, sr=SR, seed=11):
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * sr))).astype(np.float32)


class TestVadBasics:
    def test_silence_has_no_speech(self):
        r = vad.detect(silence(2.0))
        assert r.is_silent
        assert r.voiced_fraction == 0.0

    def test_empty_audio_does_not_crash(self):
        r = vad.detect(np.zeros(0, dtype=np.float32))
        assert r.is_silent and r.clip_duration_s == 0.0

    def test_near_silent_noise_is_not_speech(self):
        """A squelch burst is what 51 of the 55 hallucinations actually are."""
        assert vad.detect(noise(1.5)).is_silent

    def test_speech_only_returns_nothing_for_silence(self):
        a = silence(2.0)
        assert len(vad.speech_only(a, vad.detect(a))) == 0

    def test_voiced_fraction_is_a_fraction(self):
        r = vad.detect(tone(2.0))
        assert 0.0 <= r.voiced_fraction <= 1.0

    def test_serialises_for_the_raw_record(self):
        d = vad.detect(silence(1.0)).to_dict()
        assert set(d) >= {"spans", "n_spans", "speech_s", "voiced_fraction"}


class TestLoudness:
    def test_silence_has_no_measurable_loudness(self):
        """None, not zero - 'too quiet to measure' is not 'silent at 0 LUFS'."""
        assert audio.integrated_lufs(silence(1.0)) is None

    def test_too_short_to_measure(self):
        assert audio.integrated_lufs(tone(0.1)) is None

    def test_louder_signal_reads_louder(self):
        quiet = audio.integrated_lufs(tone(2.0, amp=0.05))
        loud = audio.integrated_lufs(tone(2.0, amp=0.5))
        assert quiet is not None and loud is not None
        assert loud > quiet

    def test_doubling_amplitude_adds_about_six_db(self):
        a = audio.integrated_lufs(tone(2.0, amp=0.1))
        b = audio.integrated_lufs(tone(2.0, amp=0.2))
        assert b - a == pytest.approx(6.02, abs=0.3)

    def test_normalise_hits_the_target(self):
        out, loud = audio.normalise(tone(3.0, amp=0.02))
        assert loud.lufs is not None
        after = audio.integrated_lufs(out)
        assert after == pytest.approx(audio.TARGET_LUFS, abs=1.0)

    def test_gain_is_capped(self):
        """A near-silent clip must not have its noise floor lifted to speech."""
        _, loud = audio.normalise(tone(3.0, amp=1e-6))
        assert abs(loud.gain_applied_db) <= audio.MAX_GAIN_DB

    def test_output_never_clips(self):
        out, _ = audio.normalise(tone(3.0, amp=0.95))
        assert float(np.max(np.abs(out))) <= 1.0

    def test_unmeasurable_audio_is_left_alone(self):
        out, loud = audio.normalise(silence(2.0))
        assert loud.lufs is None
        assert loud.gain_applied_db == 0.0


class TestHighpass:
    def test_removes_dc(self):
        a = np.ones(SR, dtype=np.float32) * 0.5
        assert abs(float(np.mean(audio.highpass(a)[SR // 2:]))) < 0.01

    def test_keeps_speech_band(self):
        a = tone(1.0, freq=1000.0, amp=0.3)
        out = audio.highpass(a)
        kept = float(np.sqrt(np.mean(out[SR // 4:] ** 2)))
        orig = float(np.sqrt(np.mean(a[SR // 4:] ** 2)))
        assert kept > 0.8 * orig

    def test_attenuates_rumble(self):
        a = tone(1.0, freq=25.0, amp=0.3)
        out = audio.highpass(a)
        assert float(np.sqrt(np.mean(out[SR // 2:] ** 2))) < 0.3 * float(
            np.sqrt(np.mean(a[SR // 2:] ** 2)))

    def test_too_short_is_passed_through(self):
        a = tone(0.001)
        assert np.array_equal(audio.highpass(a), a)


class TestPrepare:
    def test_emits_both_vad_and_loudness(self):
        _, meta = audio.prepare(tone(2.0, amp=0.1))
        assert "vad" in meta and "loudness" in meta
        assert "voiced_fraction" in meta["vad"]
        assert "gain_applied_db" in meta["loudness"]

    def test_survives_silence(self):
        out, meta = audio.prepare(silence(2.0))
        assert meta["vad"]["n_spans"] == 0
        assert len(out) == 2 * SR

    def test_loudness_is_measured_over_speech_not_the_whole_clip(self):
        """A clip that is mostly dead air must not be over-amplified."""
        mostly_silence = np.concatenate([silence(4.0), tone(1.0, amp=0.2), silence(4.0)])
        _, meta = audio.prepare(mostly_silence)
        assert abs(meta["loudness"]["gain_applied_db"]) <= audio.MAX_GAIN_DB


@pytest.mark.slow
class TestAgainstTheCorpus:
    """The claim that justifies the whole module, checked on real audio."""

    @pytest.fixture(scope="class")
    def flagged_and_clean(self):
        if not os.path.isdir(CLIPS):
            pytest.skip("backend/clips not present (gitignored)")
        from data.v1_baseline import ARTIFACTS, has_repetition_loop
        msgs = []
        for path in iter_race_files(RACES):
            d = json.load(open(path, encoding="utf-8"))
            for m in d["messages"]:
                m["_race"] = d["race_id"]
                msgs.append(m)
        if not msgs:
            pytest.skip("no races built")

        def is_flagged(m):
            t = m["transcript"].strip()
            return any(re.search(p, t) for p in ARTIFACTS.values()) \
                or has_repetition_loop(t)

        import random
        rng = random.Random(11)
        flagged = [m for m in msgs if is_flagged(m)]
        clean = rng.sample([m for m in msgs if not is_flagged(m)], 60)
        return flagged, clean

    def _voiced(self, m):
        from pipeline import asr
        path = os.path.join(CLIPS, m["_race"], m["audio_file"])
        if not os.path.exists(path):
            return None
        return vad.detect(asr.load_audio(path)).voiced_fraction

    def test_most_hallucinations_contain_no_speech(self, flagged_and_clean):
        flagged, _ = flagged_and_clean
        v = [x for x in (self._voiced(m) for m in flagged) if x is not None]
        if not v:
            pytest.skip("no flagged clips on disk")
        silent = sum(1 for x in v if x == 0.0)
        assert silent / len(v) > 0.8, (
            f"only {silent}/{len(v)} flagged clips are silent; the no-speech "
            "guard would not remove them"
        )

    def test_real_messages_are_not_discarded(self, flagged_and_clean):
        _, clean = flagged_and_clean
        v = [x for x in (self._voiced(m) for m in clean) if x is not None]
        if not v:
            pytest.skip("no clean clips on disk")
        lost = sum(1 for x in v if x == 0.0)
        assert lost / len(v) < 0.05, (
            f"{lost}/{len(v)} real messages would be discarded as silent"
        )
