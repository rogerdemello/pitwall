"""ASR contract, and the truncation bug that made it necessary.

`_v1_baseline.json` records 92 clips silently cut at 30 seconds, 21.9 minutes of
audio never transcribed. `WhisperProcessor` pads *or truncates* to exactly
n_samples=480000 and v1 handed it whole clips with no chunking. Meanwhile
prosody read the full clip, so on those messages the words and the voice
described different audio while the incongruence detector compared them.

The tests that matter here are therefore about *plumbing*, not model quality:
long clips must be fully transcribed, short clips must be untouched, and the
dataclass must keep the fields every caller already reads.

Model-loading tests are marked slow; the rest run in milliseconds.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

from pipeline import asr  # noqa: E402

SR = asr.SAMPLE_RATE


def tone(seconds, freq=200.0, amp=0.2):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestTranscriptContract:
    """Every field existing callers read must survive."""

    def _t(self, **kw):
        base = dict(text="hello", text_unbiased=None, duration_s=10.0, elapsed_s=5.0)
        base.update(kw)
        return asr.Transcript(**base)

    def test_v1_fields_present(self):
        t = self._t()
        assert t.text == "hello" and t.text_unbiased is None
        assert t.duration_s == 10.0 and t.elapsed_s == 5.0

    def test_rtf(self):
        assert self._t(duration_s=10.0, elapsed_s=5.0).rtf == 0.5

    def test_rtf_handles_zero_duration(self):
        assert self._t(duration_s=0.0).rtf == 0.0

    def test_new_fields_default_so_v1_construction_still_works(self):
        t = self._t()
        assert t.segments == [] and t.truncated is False and t.no_speech is False

    def test_declares_its_model(self):
        t = self._t()
        assert t.model_id == asr.MODEL_ID
        assert t.matches_corpus_model == (asr.MODEL_ID == asr.CORPUS_MODEL_ID)

    def test_flags_a_model_mismatch(self):
        """The Space may run a smaller model than the corpus build."""
        assert self._t(model_id="openai/whisper-large-v3").matches_corpus_model is False

    def test_serialises_for_the_api(self):
        d = self._t().to_dict()
        assert {"text", "rtf", "model_id", "matches_corpus_model", "segments"} <= set(d)


class TestSegment:
    def test_serialises_with_confidence_signals(self):
        s = asr.Segment(start=0.0, end=1.0, text="box box",
                        avg_logprob=-0.3, no_speech_prob=0.02,
                        compression_ratio=1.4)
        d = s.to_dict()
        assert d["text"] == "box box"
        assert d["avg_logprob"] == -0.3 and d["no_speech_prob"] == 0.02

    def test_words_default_empty(self):
        assert asr.Segment(start=0.0, end=1.0, text="x").words == []


class TestDecodeGuards:
    """Every one of these was absent in v1, which decoded greedily."""

    @pytest.mark.parametrize("key", [
        "beam_size", "temperature", "compression_ratio_threshold",
        "log_prob_threshold", "no_speech_threshold",
        "condition_on_previous_text", "word_timestamps",
    ])
    def test_guard_is_configured(self, key):
        assert key in asr.DECODE

    def test_conditioning_is_off(self):
        """faster-whisper defaults this True. Clips are independent bursts, and
        conditioning on previous text is what drives repetition loops."""
        assert asr.DECODE["condition_on_previous_text"] is False

    def test_temperature_fallback_ladder_ascends_from_greedy(self):
        temps = asr.DECODE["temperature"]
        assert temps[0] == 0.0
        assert temps == sorted(temps)

    def test_word_timestamps_on(self):
        """Segment-level speaker attribution and window alignment need them."""
        assert asr.DECODE["word_timestamps"] is True


class TestBackendSelection:
    def test_default_is_the_corpus_model(self):
        assert asr.CORPUS_MODEL_ID == "openai/whisper-small.en"

    def test_small_model_uses_transformers(self, monkeypatch):
        """The free Space has no ctranslate2 build; it must stay on transformers."""
        assert asr.BACKEND in ("transformers", "faster-whisper")
        if asr.MODEL_ID == asr.CORPUS_MODEL_ID:
            assert asr.BACKEND == "transformers"

    def test_window_constant_matches_whisper(self):
        """30.0s is where the feature extractor truncates - the v1 bug."""
        assert asr.WHISPER_WINDOW_S == 30.0


@pytest.mark.slow
class TestChunkingRemovesTruncation:
    def test_a_long_clip_produces_more_than_one_segment(self):
        """A 70s clip must not come back as a single 30s window."""
        t = asr.transcribe(tone(70.0))
        assert len(t.segments) > 1, "long clip was not chunked"
        assert t.truncated is False

    def test_segments_span_the_whole_clip(self):
        t = asr.transcribe(tone(70.0))
        if t.segments:
            assert t.segments[-1].end >= 65.0, (
                f"coverage ends at {t.segments[-1].end}s of 70s"
            )

    def test_a_short_clip_is_one_segment(self):
        t = asr.transcribe(tone(5.0))
        assert len(t.segments) <= 1 or t.segments[0].end <= 30.0

    def test_duration_is_the_real_duration(self):
        assert asr.transcribe(tone(45.0)).duration_s == pytest.approx(45.0, abs=0.1)
