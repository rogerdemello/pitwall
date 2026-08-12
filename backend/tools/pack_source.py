"""Package the source the GPU build needs, for a runtime with no git remote.

The notebook's normal path clones the repository at a pinned commit, which is
the better option: it gives the manifest a SHA, so the artifact traces back to
exactly the code that produced it.

This is the fallback for when there is no remote to clone - which is the state
the project is in until it is pushed somewhere. It packs the ~720 KB of Python
the build actually imports, plus the built race JSON the smoke test compares
against, and records a content hash so provenance is still checkable even
without a commit to point at.

    python backend/tools/pack_source.py

Then upload `pitwall_src.zip` in the notebook's cell 3.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(BACKEND, ".."))
OUT = os.path.join(ROOT, "pitwall_src.zip")

#: Directories the runtime imports from. Deliberately not the whole repo: the
#: frontend, the clips and the FastF1 cache are all irrelevant to stage 1 and
#: would turn a 720 KB upload into gigabytes.
INCLUDE_DIRS = [
    ("backend/pipeline", (".py", ".json")),
    ("backend/data", (".py",)),
    # The smoke test in cell 7 diffs against v1, so the built races come along.
    # 6 MB, and being able to eyeball 20 clips before committing to a 40 minute
    # run is worth it.
    ("backend/races", (".json",)),
]
INCLUDE_FILES = ["requirements.txt"]

SKIP_PARTS = {"__pycache__", ".pytest_cache", "_loro"}


def _files() -> list[tuple[str, str]]:
    """(absolute path, archive name) for everything to pack."""
    out = []
    for rel_dir, suffixes in INCLUDE_DIRS:
        base = os.path.join(ROOT, rel_dir)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_PARTS]
            for fn in sorted(filenames):
                if not fn.endswith(suffixes):
                    continue
                full = os.path.join(dirpath, fn)
                out.append((full, os.path.relpath(full, ROOT).replace("\\", "/")))
    for rel in INCLUDE_FILES:
        full = os.path.join(ROOT, rel)
        if os.path.exists(full):
            out.append((full, rel))
    return sorted(out, key=lambda t: t[1])


def main() -> int:
    files = _files()
    if not files:
        print("!! nothing to pack - run from the repository")
        return 1

    # A hash over the packed content, so the build manifest can identify the
    # source even without a commit to name.
    digest = hashlib.sha256()
    for full, arc in files:
        digest.update(arc.encode())
        digest.update(open(full, "rb").read())
    content_hash = digest.hexdigest()[:12]

    manifest = {
        "packed_by": "backend/tools/pack_source.py",
        "content_sha256_12": content_hash,
        "n_files": len(files),
        "note": (
            "No git remote was available, so this bundle stands in for a pinned "
            "clone. content_sha256_12 identifies the exact source; prefer a real "
            "commit SHA once the repository has a remote."
        ),
    }

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for full, arc in files:
            z.write(full, arc)
        z.writestr("_source_manifest.json", json.dumps(manifest, indent=1))

    size = os.path.getsize(OUT) / 1e6
    by_dir: dict[str, int] = {}
    for _, arc in files:
        by_dir[arc.rsplit("/", 1)[0] if "/" in arc else "."] = \
            by_dir.get(arc.rsplit("/", 1)[0] if "/" in arc else ".", 0) + 1

    print(f"packed {len(files)} files -> {OUT}  ({size:.1f} MB)")
    for d, n in sorted(by_dir.items()):
        print(f"  {d:<24} {n:>4} files")
    print(f"\ncontent hash: {content_hash}")
    print("\nUpload this in the notebook's cell 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
