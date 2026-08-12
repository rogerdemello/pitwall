"""Experiment: can we bias F1 vocabulary into the ASR without breaking it?

The long-prompt approach collapsed distil-whisper into repetition loops. This
compares, on the same clips:
    A. distil-small.en, no prompt
    B. distil-small.en, short prompt
    C. whisper-small.en, short prompt   (original Whisper: trained with prompting)

Prints results side by side so we can pick on evidence rather than assumption.
"""

import json
import os
import sys
import time

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import asr  # noqa: E402

RACE = "2021_Abu_Dhabi_Grand_Prix"
CLIP_DIR = os.path.join(os.path.dirname(__file__), "..", "clips", RACE)
SHORT_PROMPT = "F1 radio: box box box, supersoft, inters, undercut, DRS, deg, safety car."

manifest = json.load(open(os.path.join(CLIP_DIR, "manifest.json"), encoding="utf-8"))

# Short, in-race, jargon-bearing clips.
JARGON = ("box", "soft", "medium", "hard", "inter", "drs", "safety car", "deg", "tyre", "pit")
cands = [
    m for m in manifest
    if any(j in m["reference_transcription"].lower() for j in JARGON)
    and m["message_timestamp"] > "2021-12-12T13:00"
    and 2 < len(m["reference_transcription"]) < 120
][:5]

print(f"{len(cands)} candidate clips\n")


def run(model_id, prompt, audio):
    processor = WhisperProcessor.from_pretrained(model_id)
    model = _cache.get(model_id)
    if model is None:
        model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32)
        model.eval()
        _cache[model_id] = model
    feats = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    kwargs = {"max_new_tokens": 96}
    if prompt:
        kwargs["prompt_ids"] = processor.get_prompt_ids(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        ids = model.generate(feats, **kwargs)
    out = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if prompt and out.startswith(prompt):
        out = out[len(prompt):].strip()
    return out, time.perf_counter() - t0


_cache = {}

for m in cands:
    audio = asr.load_audio(os.path.join(CLIP_DIR, m["audio_file"]))
    print("=" * 78)
    print(f"[{m['driver_id']}] {len(audio)/16000:.1f}s")
    print(f"  REFERENCE : {m['reference_transcription']}")
    for tag, mid, pr in [
        ("A distil/none ", "distil-whisper/distil-small.en", None),
        ("B distil/short", "distil-whisper/distil-small.en", SHORT_PROMPT),
        ("C small.en/sht", "openai/whisper-small.en", SHORT_PROMPT),
    ]:
        try:
            text, dt = run(mid, pr, audio)
            print(f"  {tag}: {text}   [{dt:.1f}s]")
        except Exception as e:
            print(f"  {tag}: FAILED {type(e).__name__}: {e}")
