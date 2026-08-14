"""Speech-to-text for F1 team radio.

Team radio is short, compressed, heavily distorted and dense with jargon that
general ASR models have never seen. The published transcripts in the source
dataset show exactly this failure: "supersoft" comes out as "SuperSalt",
"Vandoorne" as "Van der Waal".

Four findings shaped this module. All were measured, and two of them overturned
an approach we had already built.

  1. **Model choice is what actually mattered.** Against distil-whisper,
     openai/whisper-small.en recovers domain terms the distilled model mangles:
     "does not have DRS" vs "does not have the areas", "Hamilton's pitted" vs
     "I will turn to the pitted", "front tyres" vs "front tires". That gap is
     large and consistent, and it is why the upgrade path here is large-v3
     rather than distil-large-v3 or turbo. (see research/exp_prompting.py)

  2. **distil-whisper degenerates under prompt conditioning.** It was distilled
     without prompt training, so `prompt_ids` sends it into repetition loops.

  3. **Domain prompting does not help, so it is off by default.** Measured over
     40 clips per race across six races: mean WER 0.2005 unbiased vs 0.2348
     prompted, worse in four of the six. The prompt also leaks - "Maybe I can
     just easily cut the corner" came back with the prompt's own words inside
     it. The path is kept because a negative result you can reproduce is
     stronger evidence than a positive one you assert.

  4. **v1 silently truncated 92 clips.** `WhisperProcessor` pads *or truncates*
     to exactly 30.0s (n_samples=480000) and v1 passed whole clips with no
     chunking, so 21.9 minutes of audio was never transcribed at all. Worse,
     prosody read the *full* clip, so on those messages the words and the voice
     described different audio while the incongruence detector compared them.
     This is a plumbing bug, not a model-quality one, which is why fixing it
     matters more than the WER headline.

Two backends, one interface, chosen by `PITWALL_ASR_MODEL`:

  transformers    the incumbent path, whisper-small.en, runs anywhere including
                  the free ZeroGPU Space.
  faster-whisper  CTranslate2, used for the corpus build. It implements the
                  decode guards transformers' `generate` does not expose
                  conveniently - temperature fallback, compression-ratio and
                  no-speech thresholds - plus word timestamps and long-form
                  chunking, which is what removes finding (4).

The corpus and the live Space may therefore run different models. That is
disclosed in the response (`model_id`, `matches_corpus_model`) rather than
hidden, because a live result computed by a different model than the Race
Replay's would otherwise be a quiet inconsistency.
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass, field

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from pipeline import device

#: The model the shipped corpus was built with, on a GPU, once.
CORPUS_MODEL_ID = "openai/whisper-large-v3"

#: What a live request runs, and deliberately not the corpus model.
#:
#: large-v3 goes through CTranslate2, which wants CUDA and about 3GB resident.
#: A ZeroGPU slice transcribing one uploaded clip should not carry that, and a
#: CPU container cannot. So Live Analysis serves small.en and the response says
#: so: `matches_corpus_model` is False there, which is what that field exists
#: for. Disclosing the difference is the point - a deployment quietly running a
#: smaller model than the evidence screen describes is the failure being
#: avoided.
SERVING_MODEL_ID = "openai/whisper-small.en"

MODEL_ID = os.environ.get("PITWALL_ASR_MODEL", SERVING_MODEL_ID)
BACKEND = os.environ.get(
    "PITWALL_ASR_BACKEND",
    "faster-whisper" if "large" in MODEL_ID or "distil" in MODEL_ID else "transformers",
)
SAMPLE_RATE = 16000

#: Whisper's feature extractor pads or truncates to exactly this. Anything
#: longer needs chunking, which is what v1 did not do.
WHISPER_WINDOW_S = 30.0

# Deliberately short. See finding (2) - a longer prompt measurably degrades
# output. Terms picked for frequency on radio x observed error rate.
F1_PROMPT = "F1 radio: box box box, supersoft, inters, undercut, DRS, deg, safety car."

#: Decode guards. Every one of these was absent in v1, which decoded greedily
#: with nothing to catch a repetition loop or a confabulation on silence.
#: `condition_on_previous_text` must be False: clips are independent bursts, and
#: conditioning is what drives repetition loops. faster-whisper defaults it True.
DECODE = {
    "beam_size": 5,
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "hallucination_silence_threshold": 2.0,
    "word_timestamps": True,
}


@dataclass
class Segment:
    """One decoded span, with the confidence signals that gate it."""
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    words: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3), "end": round(self.end, 3),
            "text": self.text,
            "avg_logprob": self.avg_logprob,
            "no_speech_prob": self.no_speech_prob,
            "compression_ratio": self.compression_ratio,
            "words": self.words,
        }


@dataclass
class Transcript:
    text: str
    text_unbiased: str | None
    duration_s: float
    elapsed_s: float
    # Everything below is new and defaults so v1 callers keep working unchanged.
    segments: list[Segment] = field(default_factory=list)
    model_id: str = MODEL_ID
    backend: str = BACKEND
    language: str | None = None
    language_probability: float | None = None
    truncated: bool = False
    no_speech: bool = False

    @property
    def rtf(self) -> float:
        """Real-time factor: <1.0 means faster than the audio it processed."""
        return round(self.elapsed_s / self.duration_s, 2) if self.duration_s else 0.0

    @property
    def matches_corpus_model(self) -> bool:
        return self.model_id == CORPUS_MODEL_ID

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "duration_s": self.duration_s,
            "elapsed_s": self.elapsed_s,
            "rtf": self.rtf,
            "model_id": self.model_id,
            "backend": self.backend,
            "matches_corpus_model": self.matches_corpus_model,
            "language": self.language,
            "language_probability": self.language_probability,
            "truncated": self.truncated,
            "no_speech": self.no_speech,
            "segments": [s.to_dict() for s in self.segments],
        }


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


# --------------------------------------------------------------------------
# transformers backend - the incumbent, and what the Space runs
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    # Whisper is safe and roughly twice as fast in fp16 on CUDA, unlike the
    # wav2vec2 affect model. See pipeline/device.py for why that is per-model.
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=device.dtype_for(prefer_fp16=True))
    return processor, device.place(model, prefer_fp16=True)


def _decode(processor, model, features, prompt: str | None) -> str:
    kwargs = {"max_new_tokens": 128}
    if prompt:
        kwargs["prompt_ids"] = processor.get_prompt_ids(
            prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(features, **kwargs)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    # The decoded string is prefixed with the prompt itself; strip it.
    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text


def _transcribe_transformers(audio: np.ndarray, bias: bool, ab: bool) -> Transcript:
    """One clip through the incumbent path, chunked so nothing is truncated.

    v1 handed the whole clip to the processor, which cut it at 30s. Chunking
    with a small overlap keeps every backend transcribing the same audio.
    """
    processor, model = _load()
    duration = len(audio) / SAMPLE_RATE
    window = int(WHISPER_WINDOW_S * SAMPLE_RATE)
    overlap = int(1.0 * SAMPLE_RATE)

    chunks, start = [], 0
    while start < len(audio):
        chunks.append((start / SAMPLE_RATE, audio[start:start + window]))
        if start + window >= len(audio):
            break
        start += window - overlap
    if not chunks:
        chunks = [(0.0, audio)]

    t0 = time.perf_counter()
    parts, others, segments = [], [], []
    for offset, chunk in chunks:
        features = processor(
            chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(device=model.device, dtype=model.dtype)
        text = _decode(processor, model, features, F1_PROMPT if bias else None)
        parts.append(text)
        segments.append(Segment(start=offset,
                                end=offset + len(chunk) / SAMPLE_RATE,
                                text=text))
        if ab:
            others.append(_decode(processor, model, features,
                                  None if bias else F1_PROMPT))
    elapsed = time.perf_counter() - t0

    return Transcript(
        text=" ".join(p for p in parts if p).strip(),
        text_unbiased=" ".join(o for o in others if o).strip() if ab else None,
        duration_s=round(duration, 2),
        elapsed_s=round(elapsed, 2),
        segments=segments,
        truncated=False,
    )


# --------------------------------------------------------------------------
# faster-whisper backend - the corpus build
# --------------------------------------------------------------------------

#: OpenAI checkpoint -> the CTranslate2 conversion of the same weights.
#:
#: faster-whisper runs CTranslate2, which loads a `model.bin`. The `openai/*`
#: repos publish safetensors for transformers and contain no such file, so
#: passing MODEL_ID straight through fails on every single clip with "Unable to
#: open file 'model.bin'". faster-whisper resolves bare size aliases ("tiny",
#: "large-v3") to pre-converted repos itself, but takes a full `org/name` id
#: literally - which is why the tiny-model smoke test passed while the real
#: build could not load anything.
#:
#: Mapping rather than rewriting MODEL_ID: the recorded `asr_model_id` should
#: name the model, not its serialisation, so a v2 record stays comparable to
#: anything else built from the same weights.
CT2_EQUIVALENT = {
    "openai/whisper-large-v3": "Systran/faster-whisper-large-v3",
    "openai/whisper-large-v2": "Systran/faster-whisper-large-v2",
    "openai/whisper-medium.en": "Systran/faster-whisper-medium.en",
    "openai/whisper-small.en": "Systran/faster-whisper-small.en",
}


def ct2_repo_for(model_id: str) -> str:
    """The CTranslate2 repo to load for `model_id`, or `model_id` unchanged."""
    return CT2_EQUIVALENT.get(model_id, model_id)


@functools.lru_cache(maxsize=1)
def _load_faster():
    from faster_whisper import WhisperModel

    on_gpu = torch.cuda.is_available()
    return WhisperModel(
        ct2_repo_for(MODEL_ID),
        device="cuda" if on_gpu else "cpu",
        # int8 on CPU is a large speedup at negligible WER cost; fp16 on GPU.
        compute_type="float16" if on_gpu else "int8",
    )


def _transcribe_faster(audio: np.ndarray, bias: bool) -> Transcript:
    model = _load_faster()
    duration = len(audio) / SAMPLE_RATE
    opts = dict(DECODE)
    if bias:
        opts["initial_prompt"] = F1_PROMPT
    # `.en` checkpoints are English-only and reject a language hint; multilingual
    # ones get forced to English but still report what they detected, which is a
    # free artifact - code-switching on this corpus is real and worth counting.
    if not MODEL_ID.endswith(".en"):
        opts["language"] = "en"
        opts["task"] = "transcribe"

    t0 = time.perf_counter()
    segments_iter, info = model.transcribe(audio.astype(np.float32), **opts)
    segments = [
        Segment(
            start=s.start, end=s.end, text=s.text.strip(),
            avg_logprob=round(s.avg_logprob, 4) if s.avg_logprob is not None else None,
            no_speech_prob=round(s.no_speech_prob, 4) if s.no_speech_prob is not None else None,
            compression_ratio=round(s.compression_ratio, 4) if s.compression_ratio is not None else None,
            words=[{"word": w.word, "start": round(w.start, 3),
                    "end": round(w.end, 3), "probability": round(w.probability, 4)}
                   for w in (s.words or [])],
        )
        for s in segments_iter
    ]
    elapsed = time.perf_counter() - t0

    return Transcript(
        text=" ".join(s.text for s in segments if s.text).strip(),
        text_unbiased=None,
        duration_s=round(duration, 2),
        elapsed_s=round(elapsed, 2),
        segments=segments,
        language=getattr(info, "language", None),
        language_probability=round(getattr(info, "language_probability", 0.0) or 0.0, 4),
        truncated=False,
        no_speech=not segments,
    )


# --------------------------------------------------------------------------

def transcribe(audio: np.ndarray | str, bias: bool = False,
               ab: bool = False) -> Transcript:
    """Transcribe one radio clip.

    `bias` defaults to False - see finding (3). Pass bias=True only to reproduce
    the ablation. `ab=True` also runs the opposite pass for A/B evaluation; it
    doubles the cost, so production callers leave it off. It is only implemented
    on the transformers backend, which is where the ablation was measured.
    """
    if isinstance(audio, str):
        audio = load_audio(audio)
    audio = np.asarray(audio, dtype=np.float32)

    if BACKEND == "faster-whisper":
        return _transcribe_faster(audio, bias)
    return _transcribe_transformers(audio, bias, ab)
