"""Replace typed numbers in the docs with measured ones.

Every figure in this project's documentation should come from the file that
measured it. They did not: the corpus doubled from six races to twelve and the
docs did not all follow, so README said 29 tests against a real 277, the
published dataset card said 556 paired observations against a real 1,155, and
README contradicted itself on the same figure thirty lines apart.

Rather than a full template engine, this does the narrow thing that actually
prevents recurrence: it rewrites known stale tokens to their measured values,
and `test_docs_numbers.py` fails the build if any of them reappear.

    python backend/data/render_docs.py            # rewrite
    python backend/data/render_docs.py --check    # fail if anything is stale
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.facts import facts  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))

#: Files that carry published numbers. `backend/hub_dataset/README.md` is the
#: generated dataset card and is fixed at its source in push_to_hub.py instead.
DOCS = [
    "README.md",
    "docs/DEMO.md",
    "docs/SUBMISSION.md",
    "docs/DEPLOY.md",
    "space/README.md",
    "space/README.static.md",
    "backend/data/push_to_hub.py",
]


def rules(f: dict) -> list[tuple[str, str]]:
    """(pattern, replacement), applied in order. Patterns are regexes."""
    n_races, n_msg = f["n_races"], f["n_messages"]
    paired, r = f["paired_n"], f["pooled_r"]
    return [
        # The headline null, in every phrasing it was written in.
        (r"\b556\b", str(paired)),
        (r"\b1,155\b", f"{paired:,}"),
        (r"\br = 0\.047\b", f"r = {r}"),
        (r"\bpooled r = 0\.047\b", f"pooled r = {r}"),
        (r"\br=0\.047\b", f"r={r}"),
        (r"\b0\.043\b", str(r)),
        # Corpus size.
        (r"\b1,042\b", f"{n_msg:,}"),
        (r"\b1042\b", str(n_msg)),
        (r"\bsix races\b", f"{n_races} races"),
        (r"\bsix-race\b", f"{n_races}-race"),
        (r"\bfrom six\b", f"from {n_races}"),
        # Test count.
        # Deliberately no *count* here. An exact number of tests in prose is
        # stale the moment anyone adds one - it invalidates itself, and a doc
        # that cannot stay true is worse than a doc that says less. The docs
        # describe what the suite covers instead.
        (r"\b\d+ tests over the logic\b", "tests over the logic"),
        (r"\(\d+ tests, no models needed\)", "(no models needed)"),
        (r"\b\d+ tests pass\b", "the suite passes"),
        # Sign test and tercile, which moved with the speaker-attribution change.
        (r"\b38/80\b", f"{f['drivers_slower']}/{f['drivers_total']}"),
        (r"p = 0\.7376", f"p = {f['sign_test_p']}"),
        (r"p=0\.7376", f"p={f['sign_test_p']}"),
    ]


def apply(text: str, f: dict) -> tuple[str, int]:
    """Rewrite, counting only substitutions that actually change something.

    Several rules normalise a value to itself once the doc is already correct -
    `1,155` -> `1,155`. Counting those would make `--check` fail forever on a
    perfectly up-to-date file, so the count is of real changes.
    """
    n = 0
    for pattern, repl in rules(f):
        after = re.sub(pattern, repl, text)
        if after != text:
            n += len(re.findall(pattern, text))
            text = after
    return text, n


def main(check_only: bool = False) -> int:
    f = facts()
    if f.get("paired_n") is None:
        print("!! evidence files missing - cannot render docs")
        return 1

    stale, changed = [], []
    for rel in DOCS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        before = open(path, encoding="utf-8").read()
        after, n = apply(before, f)
        if n:
            (stale if check_only else changed).append((rel, n))
            if not check_only:
                open(path, "w", encoding="utf-8").write(after)

    if check_only:
        if stale:
            print(f"!! {len(stale)} file(s) carry stale numbers:")
            for rel, n in stale:
                print(f"     {rel:<34} {n} occurrence(s)")
            print("   run: python backend/data/render_docs.py")
            return 1
        print(f"all {len(DOCS)} doc(s) match the measured evidence")
        return 0

    if changed:
        print(f"rewrote {sum(n for _, n in changed)} number(s) across "
              f"{len(changed)} file(s):")
        for rel, n in changed:
            print(f"  {rel:<34} {n}")
    else:
        print("nothing stale")
    print(f"\nauthoritative: {f['n_races']} races, {f['n_messages']:,} messages, "
          f"{f['paired_n']:,} paired, r = {f['pooled_r']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(check_only="--check" in sys.argv))
