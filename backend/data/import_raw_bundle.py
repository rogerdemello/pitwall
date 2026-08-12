"""Bring the GPU build's stage-1 output back into the repository.

The Colab notebook (`backend/tools/build_corpus_colab.ipynb`) returns one
tarball of JSONL. This verifies it, converts it to the shape stage 2 already
reads, and puts v1 somewhere safe.

That last part matters more than it sounds. v2 covers the *identical* 2,042
clips, so keeping v1 gives a paired comparison for free - the strongest form of
"the improvement was measured", and the only way to report honestly if some part
of it made things worse. Overwriting v1 would throw that away permanently, so
this refuses to do it without --force.

Usage:
    python backend/data/import_raw_bundle.py pitwall_raw_v2.tar.gz
    python backend/data/import_raw_bundle.py bundle.tar.gz --dry-run
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, "..")
RAW = os.path.join(BACKEND, "raw")
RAW_V1 = os.path.join(BACKEND, "raw_v1")

#: Fields stage 2 (calibrate.py) requires. A bundle missing any of these would
#: fail later and further from the cause.
REQUIRED = ["id", "transcript", "arousal", "valence", "dominance",
            "text_label", "text_polarity", "text_negative", "text_positive"]


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def verify(work: str) -> tuple[dict, list[str]]:
    """Checksums and required fields, before anything is written."""
    problems = []

    manifest_path = os.path.join(work, "_manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8")) \
        if os.path.exists(manifest_path) else {}
    if not manifest:
        problems.append("no _manifest.json - cannot trace this bundle to a commit")

    sums = os.path.join(work, "checksums.sha256")
    if os.path.exists(sums):
        for line in open(sums, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            expected, name = line.split(None, 1)
            path = os.path.join(work, name.strip())
            if not os.path.exists(path):
                problems.append(f"checksummed file missing: {name}")
            elif _sha(path) != expected:
                problems.append(f"CHECKSUM MISMATCH: {name}")
    else:
        problems.append("no checksums.sha256 - bundle integrity unverifiable")

    journals = [f for f in os.listdir(work) if f.endswith(".raw.jsonl")]
    if not journals:
        problems.append("bundle contains no .raw.jsonl files")

    for name in journals:
        rows, bad = 0, 0
        for line in open(os.path.join(work, name), encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            rows += 1
            if "error" in rec:
                continue
            missing = [f for f in REQUIRED if rec.get(f) is None]
            if missing:
                problems.append(f"{name}: {rec.get('id')} missing {missing}")
                break
        if bad:
            problems.append(f"{name}: {bad} unparseable line(s)")
        if rows == 0:
            problems.append(f"{name}: empty")

    return manifest, problems


def convert(work: str, dry_run: bool) -> list[tuple[str, int]]:
    """JSONL -> the raw/<race>.raw.json shape stage 2 already reads."""
    written = []
    for name in sorted(f for f in os.listdir(work) if f.endswith(".raw.jsonl")):
        race_id = name[:-len(".raw.jsonl")]
        rows = [json.loads(l) for l in open(os.path.join(work, name),
                                            encoding="utf-8") if l.strip()]
        # Later lines win: a resumed run may retry a clip that previously failed.
        by_id = {}
        for r in rows:
            by_id[r["id"]] = r
        payload = {"race_id": race_id, "messages": list(by_id.values())}
        if not dry_run:
            json.dump(payload, open(os.path.join(RAW, f"{race_id}.raw.json"), "w",
                                    encoding="utf-8"), indent=1)
            shutil.copy(os.path.join(work, name), os.path.join(RAW, name))
        written.append((race_id, len(by_id)))
    return written


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    force = "--force" in argv
    dry_run = "--dry-run" in argv
    if not args:
        print(__doc__)
        return 1
    bundle = args[0]
    if not os.path.exists(bundle):
        print(f"!! no such bundle: {bundle}")
        return 1

    with tempfile.TemporaryDirectory() as work:
        with tarfile.open(bundle) as tar:
            tar.extractall(work)
        # Tolerate a bundle packed with a top-level directory.
        entries = os.listdir(work)
        if len(entries) == 1 and os.path.isdir(os.path.join(work, entries[0])):
            work = os.path.join(work, entries[0])

        manifest, problems = verify(work)
        if problems:
            print(f"!! bundle failed verification ({len(problems)} problem(s)):")
            for p in problems[:12]:
                print(f"     {p}")
            return 1

        print("bundle verified")
        for key in ("repo_sha", "asr_model", "asr_backend", "prosody_model",
                    "gpu", "clips", "races"):
            if key in manifest:
                print(f"  {key:<16} {manifest[key]}")

        existing = [f for f in os.listdir(RAW) if f.endswith(".raw.json")] \
            if os.path.isdir(RAW) else []
        if existing and not force and not dry_run:
            if os.path.isdir(RAW_V1) and os.listdir(RAW_V1):
                print(f"\n!! {RAW_V1} already holds a previous version. Passing "
                      "--force would overwrite it and destroy the paired "
                      "v1-vs-v2 comparison. Move it aside by hand first.")
                return 1
            os.makedirs(RAW_V1, exist_ok=True)
            for f in existing:
                shutil.copy2(os.path.join(RAW, f), os.path.join(RAW_V1, f))
            print(f"\nv1 preserved: {len(existing)} file(s) -> {RAW_V1}")
            print("  (v2 covers the identical clips, so this is what makes the "
                  "paired upgrade comparison possible)")

        os.makedirs(RAW, exist_ok=True)
        written = convert(work, dry_run)

    print(f"\n{'would write' if dry_run else 'wrote'} {len(written)} race(s):")
    for race_id, n in written:
        print(f"  {race_id:<34} {n:>4} clips")
    if not dry_run:
        print("\nnext:  python backend/data/finish_corpus.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
