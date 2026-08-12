"""Where the models run, decided in one place.

`asr.py` moved its model to CUDA. `prosody.py` and `sentiment.py` never did, so
on any GPU machine two thirds of the pipeline silently stayed on the CPU. That
cost nothing while everything was built on a CPU box, and costs a great deal in
the two places this now runs on a GPU:

  - the Colab build, where prosody is the second most expensive stage
  - the ZeroGPU Space, where the per-call budget is 60s and prosody alone was
    spending ~25s of it on the wrong device

One subtlety that is worth the separate `dtype` argument: wav2vec2-large is not
safe in fp16. The feature encoder's group norm can overflow and emit NaNs, which
then propagate silently through mean-pooling into a plausible-looking affect
score. Whisper is fine in fp16 and roughly twice as fast for it. So the choice is
per-model, not global, and `prefer_fp16=False` is a correctness requirement for
prosody rather than a conservative default.
"""

from __future__ import annotations

import torch


def device() -> torch.device:
    """The device models should live on. CUDA when present, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dtype_for(prefer_fp16: bool) -> torch.dtype:
    """Half precision only on CUDA, and only where the model tolerates it."""
    return torch.float16 if (prefer_fp16 and torch.cuda.is_available()) else torch.float32


def place(model, prefer_fp16: bool = False):
    """Move a loaded model to the inference device and put it in eval mode."""
    model = model.to(device())
    model.eval()
    return model


def inputs_to(batch, dev: torch.device | None = None):
    """Move every tensor in a processor/tokenizer output to the model's device.

    Accepts a BatchEncoding/BatchFeature or a plain dict. Non-tensor values are
    passed through untouched.
    """
    dev = dev or device()
    if hasattr(batch, "to"):
        return batch.to(dev)
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}


def describe() -> str:
    """One line for logs and for the /api/health payload."""
    if not torch.cuda.is_available():
        return f"cpu ({torch.get_num_threads()} threads)"
    return f"cuda ({torch.cuda.get_device_name(0)})"
