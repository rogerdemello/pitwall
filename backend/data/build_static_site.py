"""Snapshot the API to static files, so the app can run with no backend.

Hugging Face now charges for Docker Spaces; Static Spaces are free. That is
workable here because almost nothing in this app actually needs a server: Race
Replay and Evidence read precomputed JSON, and only Live Analysis runs a model.

So we freeze every GET endpoint to a file. The mapping is deliberately uniform -
`/api/<anything>` becomes `/data/<anything>.json` - so the frontend switches
modes with one string replacement instead of a per-endpoint lookup table that
would drift the moment an endpoint is added.

Audio: only the showcase race ships. All twelve would be 327 MB, and the other
eleven contribute to the Evidence screens, which need no audio whatsoever. One
race at ~36 MB keeps the Space light and the demo intact.

Usage:
    python backend/data/build_static_site.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(HERE, "..", "..", "frontend", "public")
DATA = os.path.join(PUBLIC, "data")
AUDIO = os.path.join(PUBLIC, "audio")
CLIPS = os.path.join(HERE, "..", "clips")

SHOWCASE = "2021_Abu_Dhabi_Grand_Prix"


def write(rel: str, payload) -> None:
    """Write /api/<rel> to /data/<rel>.json."""
    path = os.path.join(DATA, *rel.split("/")) + ".json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def main() -> None:
    import main as api  # the FastAPI module; call its handlers directly

    shutil.rmtree(DATA, ignore_errors=True)
    os.makedirs(DATA, exist_ok=True)

    races = api.list_races()
    write("races", races)
    ids = [r["race_id"] for r in races["races"]]
    print(f"{len(ids)} races")

    for rid in ids:
        write(f"race/{rid}", api.get_race(rid))
        write(f"evidence/{rid}", api.get_evidence(rid))
        write(f"evidence/{rid}/asr", api.get_asr_eval(rid))
        write(f"evidence/{rid}/affect", api.get_affect_eval(rid))

    for rel, fn in [
        ("compare", api.compare_races),
        ("corpus-finding", api.get_corpus_finding),
        ("corpus-analysis", api.get_corpus_analysis),
        ("corpus-asr", api.get_corpus_asr),
        ("gold-affect", api.get_gold_affect),
        ("convergent", api.get_convergent),
        ("era-analysis", api.get_era_analysis),
        ("experiments", api.get_experiments),
    ]:
        write(rel, fn())
        print(f"  wrote {rel}")

    # Audio for the showcase race only.
    shutil.rmtree(AUDIO, ignore_errors=True)
    src = os.path.join(CLIPS, SHOWCASE)
    dst = os.path.join(AUDIO, SHOWCASE)
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        n = 0
        for fn in os.listdir(src):
            if fn.endswith(".mp3"):
                shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
                n += 1
        print(f"copied {n} clips for {SHOWCASE}")
    else:
        print(f"!! no clips found for {SHOWCASE}; audio will not play")

    # Tell the frontend which races can actually play audio, so it can hide the
    # player rather than showing one that will silently fail.
    write("audio-manifest", {"races_with_audio": [SHOWCASE] if os.path.isdir(src) else []})

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for root in (DATA, AUDIO)
        for dp, _, fs in os.walk(root)
        for f in fs
    )
    print(f"\nstatic payload: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
