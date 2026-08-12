"""Where the speech actually is, so the models stop scoring silence.

Two defects in v1 that this addresses, both measured in `_v1_baseline.json`:

  55 clips (2.7%) are hallucinations on non-speech. 51 of them are near-silent
  bursts that Whisper decodes as the single word "you", median duration 1.2s.
  They still receive a transcript, a sentiment, a DSI and a state - 49 of the 55
  labelled Fatigued - and still sit in the calibration reference distribution.

  prosody.analyse mean-pools the whole clip, silence and channel noise included.
  On a 105-second clip that is mostly dead air between two transmissions, the
  affect score is largely a measurement of the radio channel.

There is a third thing, and it is the one that matters most: v1's transcript
comes from the first 30 seconds (Whisper truncates there) while prosody reads the
entire clip. On 92 clips the words and the voice describe *different audio*, and
the incongruence detector compares one against the other. Running both over the
same VAD-selected speech regions is what makes that comparison mean anything.

Silero v6 ships inside faster-whisper as ONNX, so this costs no new dependency
and no torchaudio - which requirements.txt documents as deliberately absent after
it broke the torch 2.9.1 ABI.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000

#: Silero's own defaults are tuned for clean speech. Team radio is clipped and
#: compressed, and transmissions are short, so the padding is generous and the
#: silence gap is short enough not to weld two turns together.
THRESHOLD = 0.5
MIN_SPEECH_MS = 120
MIN_SILENCE_MS = 300
SPEECH_PAD_MS = 200


@dataclass
class Span:
    """One region of speech, in seconds from the start of the clip."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def samples(self, sr: int = SAMPLE_RATE) -> tuple[int, int]:
        return int(self.start * sr), int(self.end * sr)


@dataclass
class SpeechRegions:
    spans: list[Span]
    total_duration_s: float
    clip_duration_s: float

    @property
    def speech_s(self) -> float:
        return sum(s.duration for s in self.spans)

    @property
    def voiced_fraction(self) -> float:
        """How much of the clip is speech. Near zero means squelch, not a message."""
        return round(self.speech_s / self.clip_duration_s, 4) if self.clip_duration_s else 0.0

    @property
    def is_silent(self) -> bool:
        return not self.spans

    def to_dict(self) -> dict:
        return {
            "spans": [{"start": round(s.start, 3), "end": round(s.end, 3)}
                      for s in self.spans],
            "n_spans": len(self.spans),
            "speech_s": round(self.speech_s, 3),
            "clip_duration_s": round(self.clip_duration_s, 3),
            "voiced_fraction": self.voiced_fraction,
        }


@functools.lru_cache(maxsize=1)
def _model():
    from faster_whisper.vad import get_vad_model
    return get_vad_model()


def detect(audio: np.ndarray, sampling_rate: int = SAMPLE_RATE,
           threshold: float = THRESHOLD) -> SpeechRegions:
    """Speech regions in `audio`.

    Returns an empty region list for a clip with no speech, which callers must
    handle rather than scoring anyway - that is the entire point.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = np.asarray(audio, dtype=np.float32)
    clip_s = len(audio) / sampling_rate
    if clip_s <= 0:
        return SpeechRegions([], 0.0, 0.0)

    opts = VadOptions(
        threshold=threshold,
        min_speech_duration_ms=MIN_SPEECH_MS,
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )
    try:
        chunks = get_speech_timestamps(audio, vad_options=opts)
    except Exception:
        # A VAD failure must not silently become "the whole clip is speech":
        # that is v1's behaviour and the thing being fixed. Report no speech and
        # let the caller decide.
        return SpeechRegions([], 0.0, clip_s)

    spans = [Span(c["start"] / sampling_rate, c["end"] / sampling_rate)
             for c in chunks]
    return SpeechRegions(spans, sum(s.duration for s in spans), clip_s)


def speech_only(audio: np.ndarray, regions: SpeechRegions,
                sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    """The speech regions concatenated, silence dropped."""
    if regions.is_silent:
        return np.zeros(0, dtype=np.float32)
    parts = []
    for span in regions.spans:
        a, b = span.samples(sampling_rate)
        parts.append(audio[max(0, a):min(len(audio), b)])
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
