"""Which files in races/ are races, and which are something else.

This lived in six places in two mutually incompatible forms. `main.py` had the
correct one; five scripts inlined

    base.startswith("_") or base.count(".") > 1

which agrees on today's filenames but accepts anything without a `.json` suffix,
so a stray `notes.txt` would have been parsed as a race. Getting this wrong once
already made /api/races return a 500, when the diarization experiment was
written into races/ and the underscore case was missing.

One definition, one set of tests.
"""

from __future__ import annotations

import os


def is_race_file(fn: str) -> bool:
    """Is this a race file, as opposed to something else living in races/?

    Two kinds of neighbour share the directory and must never be parsed as races:
      - sidecars with a dotted suffix (.calibration.json, .asr_eval.json, ...)
      - corpus-level artifacts prefixed with an underscore
        (_pooled.calibration.json, _diarization_experiment.json)
    """
    if not fn.endswith(".json") or fn.startswith("_"):
        return False
    return "." not in fn[:-5]


def race_ids(races_dir: str) -> list[str]:
    """Every built race id, sorted. Empty if the directory does not exist."""
    if not os.path.isdir(races_dir):
        return []
    return sorted(fn[:-5] for fn in os.listdir(races_dir) if is_race_file(fn))


def iter_race_files(races_dir: str):
    """Full paths to every built race JSON, sorted by race id."""
    for race_id in race_ids(races_dir):
        yield os.path.join(races_dir, f"{race_id}.json")


def sidecar_path(races_dir: str, race_id: str, kind: str) -> str:
    """Path to a per-race sidecar, e.g. kind='asr_eval' or 'calibration'."""
    return os.path.join(races_dir, f"{race_id}.{kind}.json")
