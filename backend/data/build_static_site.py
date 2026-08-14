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
    python backend/data/build_static_site.py --check

`--check` exists because this snapshot is the one published artifact nothing
guarded. `test_evidence_is_measured.py` stops a *document* quoting a number the
evidence files no longer support, but the deployed Space reads neither: it reads
this snapshot. So the corpus was recalibrated, `frontend/public/data/` was not
rebuilt, and the live site served pre-recalibration numbers while every test
passed. `--check` regenerates to a temp directory and diffs, which turns that
into a build failure instead of a thing somebody has to remember.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(HERE, "..", "..", "frontend", "public")
DATA = os.path.join(PUBLIC, "data")
AUDIO = os.path.join(PUBLIC, "audio")
CLIPS = os.path.join(HERE, "..", "clips")

SHOWCASE = "2021_Abu_Dhabi_Grand_Prix"


def write(dest: str, rel: str, payload) -> None:
    """Write /api/<rel> to <dest>/<rel>.json."""
    path = os.path.join(dest, *rel.split("/")) + ".json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def build_data(dest: str, quiet: bool = False) -> None:
    """Freeze every GET endpoint under `dest`.

    Split out from `main` so `--check` can generate a second copy somewhere
    harmless and compare, rather than overwriting the thing it is checking.
    """
    import main as api  # the FastAPI module; call its handlers directly

    def say(msg: str) -> None:
        if not quiet:
            print(msg)

    races = api.list_races()
    write(dest, "races", races)
    ids = [r["race_id"] for r in races["races"]]
    say(f"{len(ids)} races")

    for rid in ids:
        write(dest, f"race/{rid}", api.get_race(rid))
        write(dest, f"evidence/{rid}", api.get_evidence(rid))
        write(dest, f"evidence/{rid}/asr", api.get_asr_eval(rid))
        write(dest, f"evidence/{rid}/affect", api.get_affect_eval(rid))

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
        write(dest, rel, fn())
        say(f"  wrote {rel}")

    # Tell the frontend which races can actually play audio, so it can hide the
    # player rather than showing one that will silently fail.
    have_audio = os.path.isdir(os.path.join(CLIPS, SHOWCASE))
    write(dest, "audio-manifest", {"races_with_audio": [SHOWCASE] if have_audio else []})


def _tree(root: str) -> dict[str, bytes]:
    """Every JSON file under `root`, keyed by its path relative to it."""
    out: dict[str, bytes] = {}
    for dp, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            with open(full, "rb") as f:
                out[rel] = f.read()
    return out


def check() -> int:
    """Is the published snapshot what the current corpus would produce?"""
    if not os.path.isdir(DATA):
        print(f"!! {DATA} does not exist - run this script without --check")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        build_data(tmp, quiet=True)
        fresh, published = _tree(tmp), _tree(DATA)

    missing = sorted(set(fresh) - set(published))
    extra = sorted(set(published) - set(fresh))
    changed = sorted(
        rel for rel in set(fresh) & set(published) if fresh[rel] != published[rel]
    )

    if not (missing or extra or changed):
        print(f"static snapshot is current ({len(fresh)} files)")
        return 0

    print("frontend/public/data is stale - it does not match backend/races.\n")
    for rel in missing:
        print(f"  missing   {rel}")
    for rel in extra:
        print(f"  orphaned  {rel}")
    for rel in changed:
        print(f"  changed   {rel}  ({len(published[rel])} -> {len(fresh[rel])} bytes)")
    print("\nRun: python backend/data/build_static_site.py")
    print("Then rebuild frontend/out, or the deployed Space keeps serving the old numbers.")
    return 1


def main() -> None:
    shutil.rmtree(DATA, ignore_errors=True)
    os.makedirs(DATA, exist_ok=True)
    build_data(DATA)

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

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for root in (DATA, AUDIO)
        for dp, _, fs in os.walk(root)
        for f in fs
    )
    print(f"\nstatic payload: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    if "--check" in sys.argv:
        raise SystemExit(check())
    main()
