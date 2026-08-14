"""No document may state a number the evidence does not support.

The corpus doubled from six races to twelve and the docs did not all follow.
README said 29 tests against a real 277. The *published* dataset card said 556
paired observations against a real 1,155. README contradicted itself on the same
figure thirty lines apart, because nobody re-reads 385 lines of prose.

In a project whose entire pitch is numeric honesty, that is the most damaging
kind of defect: every individual claim is checkable and the aggregate is wrong.

This makes the claim mechanical rather than asserted.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

from data import render_docs  # noqa: E402
from data.facts import facts  # noqa: E402

#: Numbers that were published, are now wrong, and must never come back.
RETIRED = {
    r"\b556\b": "paired observations (now 1,155)",
    r"\b0\.047\b": "pooled r (now measured)",
    r"\b1,042\b": "corpus size (now 2,042)",
    r"\b29 tests\b": "test count",
    r"\bsix races\b": "corpus size (now 12 races)",
    # The gold-label axis figures, retired when the three CREMA-D artifacts were
    # re-measured against the prosody path that actually ships. See
    # _remeasurement.json. `78.1%` is barred rather than `78.1` because the
    # latter appears legitimately inside quoted history.
    r"\b78\.1%": "arousal accuracy (re-measured with VAD and windowing)",
    r"\b62\.9%": "valence accuracy (re-measured)",
    r"\b49\.2%": "four-way gold accuracy (re-measured)",
    r"\+16\.3\b": "arousal lift (re-measured)",
    r"\+0\.0605\b": "valence lift at the fitted boundary (re-measured)",
    r"\+0\.061\b": "valence lift at the fitted boundary (re-measured)",
    r"\bsix times the published figure\b":
        "the boundary refit no longer clears its own materiality bar",
}

DOCS = [d for d in render_docs.DOCS if d.endswith(".md")]

#: (file, pattern) -> why this one occurrence is allowed to survive retirement.
#:
#: A retired number must never be *asserted* again. But a superseded claim that
#: is being withdrawn has to name the figure it is withdrawing, or the retraction
#: says nothing - and "we quietly stopped mentioning it" is exactly the move this
#: project exists not to make.
#:
#: So the ban stays absolute by default and every exception is written down here
#: with a reason. An allowlist that costs a line of justification per entry is
#: hard to abuse by accident; a blanket "skip anything near the word withdrawn"
#: would not be.
QUOTED_HISTORY = {
    ("docs/SUBMISSION.md", r"\+0\.061\b"):
        "names the withdrawn lift in the paragraph that withdraws it",
    ("docs/SUBMISSION.md", r"\+0\.0605\b"):
        "quotes the pre-registration, which is committed-before-the-fact and "
        "must not be edited to match a later measurement",
    ("docs/SUBMISSION.md", r"\bsix times the published figure\b"):
        "quotes the exact wording being retracted",
    ("README.md", r"\+16\.3\b"):
        "contrasts the superseded arousal lift with the re-measured one",
}


@pytest.fixture(scope="module")
def measured():
    f = facts()
    if f.get("paired_n") is None:
        pytest.skip("evidence files not built")
    return f


def _read(rel: str) -> str | None:
    path = os.path.join(ROOT, rel)
    return open(path, encoding="utf-8").read() if os.path.exists(path) else None


class TestNoRetiredNumbers:
    @pytest.mark.parametrize("rel", DOCS)
    def test_doc_has_no_retired_token(self, rel):
        text = _read(rel)
        if text is None:
            pytest.skip(f"{rel} not present")
        found = [why for pat, why in RETIRED.items()
                 if re.search(pat, text) and (rel, pat) not in QUOTED_HISTORY]
        assert not found, f"{rel} still states retired {found}"

    @pytest.mark.parametrize("key", sorted(QUOTED_HISTORY))
    def test_every_exemption_is_still_needed(self, key):
        """An allowlist nobody prunes becomes a list of things nobody checks."""
        rel, pat = key
        text = _read(rel)
        if text is None:
            pytest.skip(f"{rel} not present")
        assert re.search(pat, text), (
            f"{rel} no longer contains {pat!r} - drop this entry from "
            "QUOTED_HISTORY so the ban goes back to being absolute"
        )

    def test_the_generated_dataset_card_template_is_clean(self):
        """The card is published to the Hub, so a stale number there is public.

        Editing backend/hub_dataset/README.md achieves nothing - it is generated
        - so the template inside push_to_hub.py is what has to be right.
        """
        text = _read("backend/data/push_to_hub.py")
        if text is None:
            pytest.skip("push_to_hub.py not present")
        for pat, why in RETIRED.items():
            assert not re.search(pat, text), f"push_to_hub.py template states {why}"


class TestDocsMatchTheEvidence:
    def test_render_is_idempotent(self, measured):
        """Running the renderer on up-to-date docs must change nothing.

        Without this the check would fail forever on a correct file, because
        several rules normalise a value to itself.
        """
        assert render_docs.main(check_only=True) == 0

    @pytest.mark.parametrize("rel", DOCS)
    def test_headline_figures_appear_correctly_where_stated(self, rel, measured):
        """Where a doc states the headline null, it must state the real one."""
        text = _read(rel)
        if text is None:
            pytest.skip(f"{rel} not present")
        if "paired observation" not in text:
            pytest.skip("does not state the paired-observation figure")
        assert f"{measured['paired_n']:,}" in text or str(measured["paired_n"]) in text

    def test_readme_does_not_contradict_itself(self):
        """The specific failure: 1,155 on one line and 556 thirty lines later."""
        text = _read("README.md")
        if text is None:
            pytest.skip("no README")
        # The negative lookbehind matters: \b would match "155" inside "1,155"
        # and report the file as contradicting itself with its own number.
        figures = set(re.findall(r"(?<![\d,])([\d,]{3,6}) paired observations?\b", text))
        assert len(figures) <= 1, f"README states {figures} paired observations"


class TestDocsExist:
    @pytest.mark.parametrize("rel", [
        "README.md", "LICENSE",
        "docs/DEMO.md", "docs/DEPLOY.md", "docs/SUBMISSION.md",
        "docs/problem-statement.pdf",
    ])
    def test_present(self, rel):
        assert os.path.exists(os.path.join(ROOT, rel)), f"{rel} is missing"


class TestNotebooksCarryNoExecutionState:
    """A committed notebook should hold source, not whatever was on screen.

    Running a cell in VS Code, Jupyter or Colab writes the output back into the
    file, and it gets committed with everything else. That is how a
    FileNotFoundError traceback from a GPU-less runtime ended up inside the
    build tool - in a repository a judge may well open.
    """

    def test_no_notebook_has_saved_outputs(self):
        sys.path.insert(0, os.path.join(BACKEND, "tools"))
        import clean_notebooks
        assert clean_notebooks.main(check_only=True) == 0, (
            "a notebook carries execution state; run "
            "python backend/tools/clean_notebooks.py"
        )
