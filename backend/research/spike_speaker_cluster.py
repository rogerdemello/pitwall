"""Day-1 spike: can we separate the driver from their engineer, acoustically?

The premise: a car's radio channel only ever carries two voices - that driver and
their race engineer. So rather than diarizing a 3-second clip in isolation (hard,
unreliable), pool every clip for one driver, window it, embed each window, and
cluster into two.

Three rounds, each fixing what the previous one exposed:

  R1  Strong acoustic separation (cosine ~0.78 within vs ~0.35 between) but the
      clusters were mislabelled. Cause: the weak supervision applied a clip's
      *text* label to every window in it, and dialogue clips contain both
      speakers - the exact contamination this exists to remove.

  R2  Restricted anchors to short, unambiguous clips. That fixed the poisoning
      but starved the signal: 5-7 anchor windows per driver, and zero for Perez.
      It also surfaced a real bug - a cluster with NO anchors scored 0.0 and
      could still beat a cluster that was overwhelmingly engineer, which is how
      Verstappen's engineer cluster (6 engineer votes vs 1 driver) got labelled
      "driver".

  R3  (this) Two fixes. Label each cluster by its anchor *majority*, resolving a
      no-anchor cluster by elimination rather than by a meaningless 0.0 share.
      And pool a driver's clips across ALL downloaded races, which multiplies the
      anchor count without changing the two voices on the channel.

Stopping criterion, set before running: the driver cluster needs a clear anchor
majority (driver-share >= 0.65 against <= 0.35) on at least 15 anchor windows.
Short of that, this is reported as not working and the text heuristic stands -
the same rule that retired the ASR prompt-biasing idea.

Uses microsoft/wavlm-base-plus-sv (ungated, transformers-native, no torchaudio).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import AutoFeatureExtractor, WavLMForXVector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import asr, speaker  # noqa: E402

MODEL_ID = "microsoft/wavlm-base-plus-sv"
CLIP_ROOT = os.path.join(os.path.dirname(__file__), "..", "clips")
SR = 16000
WIN_S = 3.0
MIN_WIN_S = 1.2
ANCHOR_MAX_S = 7.0
SILENCE_RMS = 0.005

# Pass criteria, fixed in advance.
MIN_ANCHORS = 15
MIN_SHARE_GAP = (0.65, 0.35)


def load_model():
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = WavLMForXVector.from_pretrained(MODEL_ID)
    model.eval()
    return fe, model


def windows(audio: np.ndarray) -> list[np.ndarray]:
    n = int(WIN_S * SR)
    out = []
    for i in range(0, len(audio), n):
        w = audio[i:i + n]
        if len(w) < MIN_WIN_S * SR:
            continue
        if float(np.sqrt(np.mean(w ** 2))) < SILENCE_RMS:
            continue
        out.append(w)
    return out


def embed(fe, model, chunks: list[np.ndarray]) -> np.ndarray:
    vecs = []
    for i in range(0, len(chunks), 8):
        inputs = fe(chunks[i:i + 8], sampling_rate=SR, padding=True, return_tensors="pt")
        with torch.no_grad():
            e = model(**inputs).embeddings
        vecs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.vstack(vecs)


def collect(driver_id: str) -> list[dict]:
    """Every clip for this driver across every race downloaded so far."""
    out = []
    for race in sorted(os.listdir(CLIP_ROOT)):
        mpath = os.path.join(CLIP_ROOT, race, "manifest.json")
        if not os.path.exists(mpath):
            continue
        for m in json.load(open(mpath, encoding="utf-8")):
            if m["driver_id"] == driver_id:
                out.append({**m, "_race": race})
    return out


def label_clusters(votes: dict[int, dict]) -> tuple[int, float, float]:
    """Name each cluster from its anchor majority; resolve an empty one by elimination."""
    def share(v):
        t = v["driver"] + v["engineer"]
        return (v["driver"] / t) if t else None  # None = no evidence

    s0, s1 = share(votes[0]), share(votes[1])
    if s0 is None and s1 is None:
        return -1, 0.0, 0.0
    if s0 is None:                      # only c1 has evidence
        return (1, s1, 0.0) if s1 > 0.5 else (0, 1 - s1, s1)
    if s1 is None:
        return (0, s0, 0.0) if s0 > 0.5 else (1, 1 - s0, s0)
    return (0, s0, s1) if s0 > s1 else (1, s1, s0)


def run(driver_id: str, fe, model) -> dict | None:
    clips = collect(driver_id)
    if not clips:
        print(f"{driver_id}: no clips")
        return None

    chunks, origin = [], []
    for m in clips:
        path = os.path.join(CLIP_ROOT, m["_race"], m["audio_file"])
        if not os.path.exists(path):
            continue
        audio = asr.load_audio(path)
        dur = len(audio) / SR
        who, _ = speaker.classify(m["reference_transcription"])
        is_anchor = dur <= ANCHOR_MAX_S and who in ("driver", "engineer")
        for w in windows(audio):
            chunks.append(w)
            origin.append({"clip": m, "anchor": who if is_anchor else None})

    races = len({m["_race"] for m in clips})
    if len(chunks) < 12:
        print(f"{driver_id}: only {len(chunks)} speech windows, skipping")
        return None

    X = embed(fe, model, chunks)
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)

    c0, c1 = X[labels == 0], X[labels == 1]
    within = (float(np.mean(c0 @ c0.T)) + float(np.mean(c1 @ c1.T))) / 2
    between = float(np.mean(c0 @ c1.T))

    votes = {0: {"driver": 0, "engineer": 0}, 1: {"driver": 0, "engineer": 0}}
    for lab, o in zip(labels, origin):
        if o["anchor"]:
            votes[int(lab)][o["anchor"]] += 1
    n_anchor = sum(sum(v.values()) for v in votes.values())

    dc, hi, lo = label_clusters(votes)
    hi_ok = hi >= MIN_SHARE_GAP[0] and lo <= MIN_SHARE_GAP[1]
    passed = dc >= 0 and n_anchor >= MIN_ANCHORS and hi_ok

    print(f"\n=== {driver_id}: {len(clips)} clips over {races} races -> {len(chunks)} windows ===")
    print(f"  cluster sizes: {len(c0)} / {len(c1)}")
    print(f"  cosine within {within:.3f} | between {between:.3f} | gap {within - between:.3f}")
    print(f"  anchors: {n_anchor}  c0={votes[0]}  c1={votes[1]}")
    if dc < 0:
        print("  -> NO ANCHORS: cannot name the clusters")
    else:
        print(f"  -> cluster {dc} = DRIVER (share {hi:.2f} vs {lo:.2f})  "
              f"[{'PASS' if passed else 'FAIL'}]")

    return {"driver": driver_id, "anchors": n_anchor, "gap": within - between,
            "hi": hi, "lo": lo, "passed": passed}


if __name__ == "__main__":
    fe, model = load_model()
    results = []
    for d in (sys.argv[1:] or ["MAXVER01", "LEWHAM01", "SERPER01", "LANNOR01"]):
        r = run(d, fe, model)
        if r:
            results.append(r)

    print("\n" + "=" * 60)
    ok = sum(1 for r in results if r["passed"])
    print(f"VERDICT: {ok}/{len(results)} drivers meet the pre-set criteria "
          f"(>={MIN_ANCHORS} anchors, driver-share >={MIN_SHARE_GAP[0]} vs <={MIN_SHARE_GAP[1]})")
    if ok < len(results):
        print("Not a clean pass. Report honestly and keep the text heuristic unless")
        print("the failures are only the low-data drivers.")
