"""Assemble the ZeroGPU model-backend Space for upload.

`space_live/app.py` imports `pipeline.*` — the same package `backend/main.py`
imports. A Hugging Face Space is its own repository, so that package has to be
physically present in the upload. The tempting shortcuts are both wrong:

  Forking pipeline/ into space_live/ means two implementations that drift, and
  the Space would quietly start disagreeing with the Race Replay it sits next to.

  Symlinking does not survive `hf upload` and needs developer mode on Windows,
  which is the primary platform here.

So the copy is a *build artifact*: generated, gitignored, and overwritten every
time. `pipeline/` in backend/ stays the only source.

Usage:
    python backend/data/build_space_live.py            # assemble
    python backend/data/build_space_live.py --check    # verify without writing
"""

from __future__ import annotations

import filecmp
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(BACKEND, ".."))
PIPELINE = os.path.join(BACKEND, "pipeline")
SPACE = os.path.join(ROOT, "space_live")
CALIBRATION = os.path.join(BACKEND, "races", "_pooled.calibration.json")

#: Only what the Space actually calls. `analysis.py` and `strategy.py` operate on
#: whole races and have no live path; leaving them out keeps the image honest
#: about what runs here.
MODULES = [
    "__init__.py",
    "aggregate.py",
    "artifacts.py",
    "asr.py",
    "audio.py",      # prosody.analyse normalises against detected speech
    "calibration.py",
    "device.py",
    "fusion.py",
    "prosody.py",
    "sentiment.py",
    "vad.py",        # and needs the speech regions to window over
]

#: Files that must exist in space_live/ and are hand-written, not generated.
SOURCES = ["app.py", "README.md", "requirements.txt"]


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:12]


def build(check_only: bool = False) -> int:
    missing = [f for f in SOURCES if not os.path.exists(os.path.join(SPACE, f))]
    if missing:
        print(f"!! space_live/ is missing hand-written sources: {missing}")
        return 1

    dest_pipeline = os.path.join(SPACE, "pipeline")
    stale = []

    if check_only:
        for name in MODULES:
            src = os.path.join(PIPELINE, name)
            dst = os.path.join(dest_pipeline, name)
            if not os.path.exists(dst) or not filecmp.cmp(src, dst, shallow=False):
                stale.append(f"pipeline/{name}")
        dst_cal = os.path.join(SPACE, "_pooled.calibration.json")
        if not os.path.exists(dst_cal) or not filecmp.cmp(CALIBRATION, dst_cal,
                                                          shallow=False):
            stale.append("_pooled.calibration.json")
        if stale:
            print("!! space_live/ is out of date with backend/:")
            for s in stale:
                print(f"     {s}")
            print("   run: python backend/data/build_space_live.py")
            return 1
        print(f"space_live/ is in sync with backend/pipeline ({len(MODULES)} modules)")
        return 0

    if os.path.isdir(dest_pipeline):
        shutil.rmtree(dest_pipeline)
    os.makedirs(dest_pipeline)

    print(f"copying {len(MODULES)} modules from backend/pipeline ->")
    for name in MODULES:
        src = os.path.join(PIPELINE, name)
        if not os.path.exists(src):
            print(f"!! missing {src}")
            return 1
        shutil.copy2(src, os.path.join(dest_pipeline, name))
        print(f"   pipeline/{name:<20} {_sha(src)}")

    if not os.path.exists(CALIBRATION):
        print(f"!! no pooled calibration at {CALIBRATION} - run finish_corpus.py")
        return 1
    shutil.copy2(CALIBRATION, os.path.join(SPACE, "_pooled.calibration.json"))
    size_kb = os.path.getsize(CALIBRATION) // 1024
    print(f"   _pooled.calibration.json  {_sha(CALIBRATION)}  ({size_kb} KB)")

    # A stray __pycache__ from a local test run would otherwise ship.
    for root, dirs, _ in os.walk(SPACE):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))
                dirs.remove(d)

    print(f"\nspace_live/ assembled. To deploy:\n"
          f"    hf auth login\n"
          f"    hf upload <user>/pitwall-live {SPACE} . --repo-type=space\n"
          f"\nThen set NEXT_PUBLIC_LIVE_SPACE to the Space URL and rebuild the "
          f"frontend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(build(check_only="--check" in sys.argv))
