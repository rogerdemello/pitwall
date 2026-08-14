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
    "space_live/README.md",
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
        # The gold-label axis figures.
        #
        # These had no rules at all, and every one of them was wrong: the three
        # CREMA-D artifacts were generated before prosody.py gained VAD and
        # per-window scoring, so they described a model that no longer ran. The
        # docs faithfully quoted numbers no current code produced, and nothing
        # failed, because the tests compared documents to artifacts and the
        # artifacts agreed with themselves. Anchored to their surrounding words
        # so a bare "78.1%" elsewhere is not rewritten by accident.
        (r"(?<=\| )78\.1%", _pctv(f, "arousal_acc")),
        (r"(?<=\| )62\.9%", _pctv(f, "valence_acc")),
        (r"arousal (?:axis )?scores 78\.1%", f"arousal scores {_pctv(f, 'arousal_acc')}"),
        (r"\(78\.1% vs a 61\.8% baseline\)",
         f"({_pctv(f, 'arousal_acc')} vs a {_pctv(f, 'arousal_baseline')} baseline)"),
        (r"\(62\.9% vs 61\.9%\)",
         f"({_pctv(f, 'valence_acc')} vs {_pctv(f, 'valence_baseline')})"),
        (r"scores 62\.9% against 61\.9%",
         f"scores {_pctv(f, 'valence_acc')} against {_pctv(f, 'valence_baseline')}"),
        (r"78\.1% against a 61\.8% baseline",
         f"{_pctv(f, 'arousal_acc')} against a {_pctv(f, 'arousal_baseline')} baseline"),
        (r"\b61\.8%", _pctv(f, "arousal_baseline")),
        (r"\b61\.9%", _pctv(f, "valence_baseline")),
        (r"\*\*49\.2% against a 41\.3% majority-class baseline\*\*",
         f"**{_pctv(f, 'gold_accuracy')} against a {_pctv(f, 'gold_baseline')} "
         "majority-class baseline**"),
        (r"(?<=\| )\*\*\+16\.3\*\*", f"**{_signed(f, 'arousal_lift')}**"),
        (r"(?<=\| )\*\*\+1\.0\*\*", f"**{_signed(f, 'valence_lift')}**"),
        # The confirmatory figures, which the GPU rebuild moved. Every one of
        # these was a literal in prose with no rule behind it, which is how the
        # v1 values would have survived a rebuild that changed all of them.
        (r"\br = 0\.045\b", f"r = {r}"),
        (r"\bpooled r = 0\.045\b", f"pooled r = {r}"),
        (r"\bpooled Pearson \*\*r = 0\.045\*\*", f"pooled Pearson **r = {r}**"),
        (r"\b0\.045\b", str(r)),
        (r"\*\*[−-]0\.07s\*\*", f"**{f['tercile_gap_s']}s**"),
        (r"\b[−-]0\.07s\b", f"{f['tercile_gap_s']}s"),
        (r"\b38 of 80\b", f"{f['drivers_slower']} of {f['drivers_total']}"),
        (r"\b37 of 80\b", f"{f['drivers_slower']} of {f['drivers_total']}"),
        (r"p = 0\.5764", f"p = {f['sign_test_p']}"),
        # Calibration leakage: in-sample vs held-out spread.
        (r"spread 5\.5 → 7\.7 points",
         f"spread {f['leakage_spread_in_sample']} → {f['leakage_spread_held_out']} points"),
        (r"spread 7\.7 → 8\.5 points",
         f"spread {f['leakage_spread_in_sample']} → {f['leakage_spread_held_out']} points"),
        # The corpus ASR model, where a document is stating what built the
        # corpus. NOT applied blanket: space_live serves small.en on purpose and
        # says so, and the distil-whisper contrast table is a v1 measurement
        # that stays true. Anchored to the models-table row only.
        (r"\| ASR \| \[`openai/whisper-small\.en`\]\(https://huggingface\.co/openai/whisper-small\.en\) \|",
         f"| ASR | [`{f['corpus_asr_model']}`](https://huggingface.co/{f['corpus_asr_model']}) |"),
        (r"`pipeline/asr\.py` — `openai/whisper-small\.en`",
         f"`pipeline/asr.py` — `{f['corpus_asr_model']}`"),
        (r"our ASR output \(`openai/whisper-small\.en`\)",
         f"our ASR output (`{f['corpus_asr_model']}`)"),
    ]


def _pctv(f: dict, key: str) -> str:
    """A fact as a percentage string, or the placeholder if it is unmeasured."""
    v = f.get(key)
    return "—" if v is None else f"{v * 100:.1f}%"


def _signed(f: dict, key: str) -> str:
    """A lift in percentage points, always signed - the sign is the finding."""
    v = f.get(key)
    return "—" if v is None else f"{v * 100:+.1f}"


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
