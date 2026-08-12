"""Speech-to-text for F1 team radio, with domain vocabulary biasing.

Team radio is short, compressed, heavily distorted and dense with jargon that
general ASR models have never seen. The published transcripts in the source
dataset show exactly this failure: "supersoft" comes out as "SuperSalt",
"Vandoorne" as "Van der Waal".

Three findings shaped this module. All were measured, and the third overturned
the approach we set out to build.

  1. **Model choice is what actually mattered.** Against distil-whisper,
     openai/whisper-small.en recovers domain terms the distilled model mangles:
     "does not have DRS" vs "does not have the areas", "Hamilton's pitted" vs
     "I will turn to the pitted", "front tyres" vs "front tires". That gap is
     large and consistent. (see exp_prompting.py)

  2. **distil-whisper degenerates under prompt conditioning.** It was distilled
     without prompt training, so `prompt_ids` sends it into repetition loops
     ("and, and, and, ..."). Whisper proper was trained with prompting.

  3. **Domain prompting does not help, so it is off by default.** We expected it
     to be the differentiator. Measured over 40 clips (eval_asr.py):
     WER 0.2138 unbiased vs 0.2157 biased - no improvement, marginally worse.
     Jargon recall moved 9/10 -> 10/10, which is one term and therefore noise.
     Worse, the prompt leaks: "Maybe I can just easily cut the corner" came back
     as "F1 radio, the camera key just easily cut the cooler", with the prompt's
     own words inside the transcript.

So `bias` defaults to False. The path is kept because the ablation is worth
showing - a negative result you can reproduce is stronger evidence than a
positive one you assert - but the shipped transcripts do not use it.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from pipeline import device

MODEL_ID = "openai/whisper-small.en"
SAMPLE_RATE = 16000

# Deliberately short. See finding (2) above - a longer prompt measurably
# degrades output. Terms picked for frequency on radio x observed error rate.
F1_PROMPT = "F1 radio: box box box, supersoft, inters, undercut, DRS, deg, safety car."


@dataclass
class Transcript:
    text: str
    text_unbiased: str | None
    duration_s: float
    elapsed_s: float

    @property
    def rtf(self) -> float:
        """Real-time factor: <1.0 means faster than the audio it processed."""
        return round(self.elapsed_s / self.duration_s, 2) if self.duration_s else 0.0


@functools.lru_cache(maxsize=1)
def _load():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    # Whisper is safe and roughly twice as fast in fp16 on CUDA, unlike the
    # wav2vec2 affect model. See pipeline/device.py for why that is per-model.
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=device.dtype_for(prefer_fp16=True))
    return processor, device.place(model, prefer_fp16=True)


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


def _decode(processor, model, features, prompt: str | None) -> str:
    kwargs = {"max_new_tokens": 128}
    if prompt:
        kwargs["prompt_ids"] = processor.get_prompt_ids(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(features, **kwargs)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    # The decoded string is prefixed with the prompt itself; strip it.
    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text


def transcribe(audio: np.ndarray | str, bias: bool = False, ab: bool = False) -> Transcript:
    """Transcribe one radio clip.

    `bias` defaults to False - see finding (3) in the module docstring. Pass
    bias=True only to reproduce the ablation.

    ab=True also runs the opposite pass, for A/B evaluation. It doubles the cost,
    so production callers leave it off.
    """
    if isinstance(audio, str):
        audio = load_audio(audio)

    processor, model = _load()
    duration = len(audio) / SAMPLE_RATE

    features = processor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features.to(device=model.device, dtype=model.dtype)

    t0 = time.perf_counter()
    text = _decode(processor, model, features, F1_PROMPT if bias else None)
    other = _decode(processor, model, features, None if bias else F1_PROMPT) if ab else None
    elapsed = time.perf_counter() - t0

    return Transcript(
        text=text,
        text_unbiased=other,
        duration_s=round(duration, 2),
        elapsed_s=round(elapsed, 2),
    )
