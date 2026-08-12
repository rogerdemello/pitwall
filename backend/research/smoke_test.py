"""Smoke test: run ASR + prosody over a few real Abu Dhabi 2021 clips."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import asr, prosody  # noqa: E402

RACE = "2021_Abu_Dhabi_Grand_Prix"
CLIP_DIR = os.path.join(os.path.dirname(__file__), "..", "clips", RACE)

manifest = json.load(open(os.path.join(CLIP_DIR, "manifest.json"), encoding="utf-8"))

# Verstappen and Hamilton only - the two that matter for this race.
picks = [m for m in manifest if m["driver_id"] in ("MAXVER01", "LEWHAM01")][:8]

print(f"Testing {len(picks)} clips\n" + "=" * 78)
t_start = time.perf_counter()

for m in picks:
    path = os.path.join(CLIP_DIR, m["audio_file"])
    audio = asr.load_audio(path)

    t0 = time.perf_counter()
    tr = asr.transcribe(audio)
    t_asr = time.perf_counter() - t0

    t0 = time.perf_counter()
    af = prosody.analyse(audio)
    t_pros = time.perf_counter() - t0

    print(f"\n[{m['driver_id']}] {m['message_timestamp']}  ({tr.duration_s}s audio)")
    print(f"  reference : {m['reference_transcription']}")
    print(f"  unbiased  : {tr.text_unbiased}")
    print(f"  F1-biased : {tr.text}")
    print(f"  affect    : arousal={af.arousal}  valence={af.valence}  dominance={af.dominance}")
    print(f"  timing    : asr={t_asr:.2f}s  prosody={t_pros:.2f}s")

print("\n" + "=" * 78)
print(f"total {time.perf_counter() - t_start:.1f}s for {len(picks)} clips")
