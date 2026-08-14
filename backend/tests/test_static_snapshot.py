"""The published snapshot must be what the current corpus would produce.

This is the gap that let the deployed Space serve numbers the repository had
already superseded.

`test_evidence_is_measured.py` stops a *document* quoting a figure the evidence
files no longer support, and `test_docs_numbers.py` stops a retired number
coming back. Neither watches `frontend/public/data/`, which is the only thing
the static Space actually reads. So when the corpus was recalibrated and the
snapshot was not rebuilt, every test passed and the live site was wrong — on a
project whose whole claim is that its numbers are measured rather than asserted.

The same argument as everywhere else in this suite: the guarantee has to be
mechanical, because "remember to re-run the build script" is not one.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

from data import build_static_site as bss  # noqa: E402

RACES = os.path.join(BACKEND, "races")


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    """One regeneration, shared: it calls every API handler over 12 races."""
    dest = tmp_path_factory.mktemp("static")
    bss.build_data(str(dest), quiet=True)
    return str(dest)


class TestTheSnapshotMatchesTheCorpus:
    def test_public_data_exists(self):
        assert os.path.isdir(bss.DATA), (
            "frontend/public/data is missing entirely - the static Space would "
            "have no data at all. Run build_static_site.py."
        )

    def test_snapshot_is_current(self):
        """The whole point. A stale snapshot is a published wrong number."""
        assert bss.check() == 0, (
            "frontend/public/data does not match backend/races. The deployed "
            "static Space reads this snapshot, not the race files, so it is "
            "serving superseded numbers. Run:\n"
            "  python backend/data/build_static_site.py\n"
            "then rebuild frontend/out."
        )

    def test_every_race_is_snapshotted(self, fresh):
        """A race present in the corpus but absent from the export is a race the
        deployed app cannot show at all."""
        from pipeline.artifacts import race_ids

        for rid in race_ids(RACES):
            for rel in (f"race/{rid}.json", f"evidence/{rid}.json"):
                assert os.path.exists(os.path.join(fresh, *rel.split("/"))), rel

    def test_the_uniform_endpoint_mapping_holds(self, fresh):
        """/api/<x> -> /data/<x>.json is what lets the frontend switch modes with
        one string replacement. A missing corpus-level file breaks a screen."""
        for rel in (
            "races",
            "compare",
            "corpus-finding",
            "corpus-analysis",
            "corpus-asr",
            "gold-affect",
            "convergent",
            "era-analysis",
            "experiments",
            "audio-manifest",
        ):
            assert os.path.exists(os.path.join(fresh, rel + ".json")), rel

    def test_regenerating_twice_gives_the_same_bytes(self, fresh, tmp_path):
        """--check compares bytes, so it is only meaningful if the build is
        deterministic. If it is not, this test fails here rather than as a
        mystery diff in someone's working tree."""
        again = tmp_path / "again"
        bss.build_data(str(again), quiet=True)
        assert bss._tree(fresh) == bss._tree(str(again))


class TestTheSnapshotIsNotEmpty:
    """A build that silently produced valid, empty files would pass every test
    above. These are the sanity floors."""

    def test_races_are_listed(self, fresh):
        with open(os.path.join(fresh, "races.json"), encoding="utf-8") as f:
            assert len(json.load(f)["races"]) >= 12

    def test_the_corpus_analysis_carries_its_verdict(self, fresh):
        with open(os.path.join(fresh, "corpus-analysis.json"), encoding="utf-8") as f:
            ca = json.load(f)
        assert ca.get("verdict"), "the central finding is missing from the export"
        assert ca.get("messages_pooled", 0) >= 2000
