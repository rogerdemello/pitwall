"""What changed when published evidence was re-measured, derived rather than typed.

Three gold-label artifacts - `_gold_affect_eval`, `_valence_diagnostic` and
`_valence_boundary` - were produced before `pipeline/prosody.py` gained VAD and
per-window scoring. Both changes are upstream of every number in them, so they
described a `prosody.analyse` that no longer runs. Nothing detected it: the
suite checks that documents match the artifacts, and the artifacts matched
themselves.

Re-running them is straightforward. Reporting the result is the part worth
getting right, because "we re-ran it and here is the new number" quietly
discards the old one, and the difference between the two *is* the finding.

So this reads both versions out of git and records the delta. The old values
come from a named revision rather than from memory, which means the record can
be regenerated and checked by anyone:

    python backend/data/remeasurement.py --against v1-submission-good

Nothing here is hand-entered. That is the same rule `facts.py` follows, applied
to a class of number that rule did not previously cover: not "what do we
publish" but "what did we publish before, and does it still hold".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(RACES, "_remeasurement.json")

#: Why the re-measurement was necessary, and what makes it checkable.
CAUSE = {
    "artifacts_predate": ["VAD conditioning", "per-window prosody"],
    "how_it_was_found": (
        "Ancestry, not timestamps: each artifact's last commit is a strict "
        "ancestor of both the VAD commit and the per-window prosody commit, so "
        "every figure in them was produced by a prosody.analyse that no longer "
        "exists in the tree."
    ),
    "why_nothing_caught_it": (
        "test_evidence_is_measured.py checks that documents match the artifacts. "
        "The artifacts matched themselves. Nothing checked that an artifact still "
        "matched the code that generated it."
    ),
}

#: The published figures, and where each lives. Dotted paths, so the extraction
#: is declarative and a schema change surfaces as a missing key rather than as a
#: silently absent comparison.
TRACKED: dict[str, dict[str, str]] = {
    "_gold_affect_eval.json": {
        "four_way_accuracy": "accuracy",
        "majority_baseline": "majority_class_baseline",
        "arousal_accuracy": "axes.arousal_high_vs_low.accuracy",
        "arousal_lift": "axes.arousal_high_vs_low.lift",
        "valence_accuracy": "axes.valence_negative_vs_positive.accuracy",
        "valence_lift": "axes.valence_negative_vs_positive.lift",
    },
    "_valence_diagnostic.json": {
        "verdict": "verdict",
        "happy_anger_auc_naive": "contrasts.happy_vs_anger.valence_naive.auc",
        "happy_anger_auc_stratified":
            "contrasts.happy_vs_anger.valence_arousal_stratified.pooled_auc",
        "neutral_sad_auc_stratified":
            "contrasts.neutral_vs_sad.valence_arousal_stratified.pooled_auc",
    },
    "_valence_boundary.json": {
        "finding": "finding",
        "recommendation": "recommendation",
        "valence_auc": "valence_raw_space.auc",
        "lift_over_baseline_median": "valence_raw_space.lift_over_baseline_median",
        "lift_over_baseline_fitted": "valence_raw_space.lift_over_baseline_fitted",
        "lift_over_median": "valence_raw_space.lift_over_median",
        "fitted_cut_mean": "valence_raw_space.fitted_cut_mean",
        "domain_shift_in_gold_sds": "transfer_check.median_shift_in_gold_sds",
    },
}


def _dig(doc, path: str):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _at_revision(ref: str, rel: str):
    """The artifact as it was at `ref`, or None if it did not exist there."""
    try:
        blob = subprocess.run(
            ["git", "show", f"{ref}:backend/races/{rel}"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
        return json.loads(blob)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def _current(rel: str):
    try:
        with open(os.path.join(RACES, rel), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def build(ref: str) -> dict:
    sha = subprocess.run(["git", "rev-parse", ref], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    artifacts = {}
    moved_total = held_total = 0

    for rel, fields in TRACKED.items():
        before, after = _at_revision(ref, rel), _current(rel)
        if before is None or after is None:
            artifacts[rel] = {"comparable": False,
                              "reason": f"not readable at {ref}" if before is None
                                        else "not present in the working tree"}
            continue
        rows = {}
        for name, path in fields.items():
            b, a = _dig(before, path), _dig(after, path)
            moved = b != a
            rows[name] = {"before": b, "after": a, "moved": moved}
            moved_total += moved
            held_total += not moved
        artifacts[rel] = {"comparable": True, "fields": rows,
                          "n_moved": sum(1 for r in rows.values() if r["moved"]),
                          "n_held": sum(1 for r in rows.values() if not r["moved"])}

    return {
        "generated_by": "backend/data/remeasurement.py",
        "purpose": (
            "Record what moved when gold-label evidence was re-measured against "
            "the prosody path that actually ships. Both sides are read from "
            "files - the old ones out of git - so the comparison can be "
            "regenerated rather than taken on trust."
        ),
        "compared_against": {"ref": ref, "sha": sha},
        "cause": CAUSE,
        "artifacts": artifacts,
        "totals": {"figures_moved": moved_total, "figures_held": held_total},
        "how_to_reproduce": (
            f"python backend/data/remeasurement.py --against {ref}"
        ),
    }


def main(argv: list[str]) -> int:
    ref = "HEAD"
    if "--against" in argv:
        ref = argv[argv.index("--against") + 1]

    out = build(ref)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"re-measurement against {ref} ({out['compared_against']['sha'][:12]})\n")
    for rel, a in out["artifacts"].items():
        if not a["comparable"]:
            print(f"{rel}: {a['reason']}")
            continue
        print(f"{rel}   {a['n_moved']} moved, {a['n_held']} held")
        for name, r in a["fields"].items():
            mark = "->" if r["moved"] else "  "
            print(f"   {name:<28} {r['before']} {mark} {r['after']}"
                  if r["moved"] else
                  f"   {name:<28} {r['after']}  (unchanged)")
        print()
    t = out["totals"]
    print(f"{t['figures_moved']} figure(s) moved, {t['figures_held']} held")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
