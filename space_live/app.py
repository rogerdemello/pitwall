"""PIT WALL — model backend, on ZeroGPU.

Why this exists. The main app is a Next.js frontend against a FastAPI backend,
and both run together in the Docker image. Hugging Face now charges for Docker
and Gradio Spaces on personal accounts, so the public deployment is a *static*
Space — which serves the precomputed corpus perfectly, but has no process to run
a model in. Live Analysis was therefore disabled in public, which is the one
thing about this submission that looked like a missing backend.

Free personal accounts in good standing may host two ZeroGPU Gradio Spaces. So
the model backend moves here, and the static frontend calls it cross-origin.
Gradio's CORS middleware accepts any origin when the host is not a localhost
alias, which is exactly the case on *.hf.space, so no proxy is involved.

What this file is NOT. It is not a second implementation. Every model call below
goes through `pipeline/`, the same package `backend/main.py` imports, copied in
verbatim at build time by `backend/data/build_space_live.py`. If the two ever
disagree, that is a bug in the build script, not a fork to reconcile.

The response is byte-compatible with `POST /api/analyze` in backend/main.py, so
the frontend has one result renderer rather than two.
"""

from __future__ import annotations

import os
import time

import gradio as gr
import librosa
import numpy as np

import spaces  # noqa: F401  — must be imported before torch on ZeroGPU

from pipeline import asr, device, fusion, prosody, sentiment
from pipeline.calibration import Calibrator

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION = os.path.join(HERE, "_pooled.calibration.json")

#: The corpus was built with whisper-small.en. The Space runs the same model on
#: purpose: a live result computed by a different model than the one behind the
#: Race Replay would be a quiet inconsistency, and the response says which was
#: used either way. Override to compare.
ASR_MODEL = os.environ.get("PITWALL_ASR_MODEL", asr.MODEL_ID)
CORPUS_ASR_MODEL = "openai/whisper-small.en"

MAX_SECONDS = 120.0

# ZeroGPU requires models to be resident before the first decorated call, so
# these load at import rather than lazily on first request.
_CAL: Calibrator | None = None
try:
    _CAL = Calibrator.from_json(CALIBRATION)
    print(f"[startup] calibration loaded from {CALIBRATION}")
except (OSError, ValueError) as e:
    print(f"[startup] no calibration ({e}); percentiles will fall back to raw scores")

print(f"[startup] device: {device.describe()}")
asr._load()
prosody._load()
sentiment._load()
print("[startup] models resident")


@spaces.GPU(duration=45)
def _run(audio: np.ndarray) -> dict:
    """The whole pipeline, on the GPU slice. Identical to backend/main.py."""
    tr = asr.transcribe(audio)
    af = prosody.analyse(audio)
    se = sentiment.analyse(tr.text)
    st = fusion.fuse(af, se, calibrator=_CAL, transcript=tr.text)
    return {
        "transcript": tr.text,
        "duration_s": tr.duration_s,
        "elapsed_s": tr.elapsed_s,
        "rtf": tr.rtf,
        "text_sentiment": {"label": se.label, "polarity": se.polarity},
        "state": st.to_dict(),
        "calibrated_against": "pooled" if _CAL else None,
        "model_id": ASR_MODEL,
        "matches_corpus_model": ASR_MODEL == CORPUS_ASR_MODEL,
    }


def analyze(path: str | None) -> dict:
    """Entry point. Returns the error in the payload rather than raising.

    A Gradio exception surfaces to an API caller as an opaque 500, and this is
    called cross-origin by the frontend, so failures are described instead.
    """
    if not path:
        return {"error": "no audio supplied"}
    started = time.time()
    try:
        audio, _ = librosa.load(path, sr=asr.SAMPLE_RATE, mono=True)
    except Exception as e:
        return {"error": f"could not decode audio: {e}"}

    seconds = len(audio) / asr.SAMPLE_RATE
    if seconds < 0.2:
        return {"error": f"clip is only {seconds:.2f}s; nothing to analyse"}
    if seconds > MAX_SECONDS:
        return {"error": f"clip is {seconds:.0f}s; limit is {MAX_SECONDS:.0f}s"}

    try:
        out = _run(audio)
    except Exception as e:  # ZeroGPU quota exhausted, OOM, model failure
        return {"error": f"{type(e).__name__}: {e}"}
    out["total_s"] = round(time.time() - started, 2)
    return out


DESCRIPTION = """
# PIT WALL — the model backend

Upload or record a few seconds of speech. This runs the same four-stage pipeline
the [PIT WALL app](https://huggingface.co/spaces/rogerdemello/pitwall) uses:
Whisper for the words, a wav2vec2 dimensional-affect model for the voice, a
RoBERTa sentiment model for the text, and a calibration layer that places the
result against 2,042 real F1 team-radio messages.

**This Space is the backend.** It exists so the app's Live Analysis works for
anyone, without running anything locally. The app itself is the place to look —
this page is the raw endpoint, and is also callable as an API at
`/gradio_api/call/analyze`.

Two things it will tell you honestly: the valence axis of the affect model
scores at chance against gold labels, so a state's calm/stressed *direction* is
much less reliable than its high/low activation; and the index is calibrated
against F1 radio, so a clip of ordinary speech will be scored as unusually calm.
"""


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Audio(type="filepath", sources=["upload", "microphone"],
                    label="Radio clip"),
    outputs=gr.JSON(label="Analysis"),
    title="PIT WALL — model backend",
    description=DESCRIPTION,
    article=(
        "Source: [github/pitwall](https://huggingface.co/spaces/rogerdemello/pitwall) · "
        "Dataset: [pitwall-f1-radio-analysis]"
        "(https://huggingface.co/datasets/rogerdemello/pitwall-f1-radio-analysis)"
    ),
    api_name="analyze",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
