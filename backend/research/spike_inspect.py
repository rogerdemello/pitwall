"""Day-1 spike, part 1: what is actually inside the HF dataset?

Streams a handful of rows so we don't pull 2.57 GB just to read the schema.
Audio is taken as raw bytes (decode=False) and decoded with librosa, which
avoids a torchcodec dependency that is awkward on Windows.
"""

import io

import librosa
from datasets import Audio, load_dataset

DATASET_ID = "MikCil/f1-team-radio"

ds = load_dataset(DATASET_ID, split="train", streaming=True)

print("=== FEATURES ===")
for name, feat in ds.features.items():
    print(f"  {name}: {feat}")

ds = ds.cast_column("audio", Audio(decode=False))

print("\n=== FIRST 5 ROWS ===")
for i, row in enumerate(ds.take(5)):
    print(f"\n--- row {i} ---")
    for k, v in row.items():
        if k == "audio":
            data = v.get("bytes")
            path = v.get("path")
            if data:
                y, sr = librosa.load(io.BytesIO(data), sr=16000)
                print(f"  audio: path={path} bytes={len(data)} dur={len(y) / sr:.2f}s sr={sr}")
            else:
                print(f"  audio: path={path} (no inline bytes)")
        else:
            print(f"  {k}: {v!r}")
