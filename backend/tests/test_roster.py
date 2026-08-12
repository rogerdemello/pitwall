"""Driver identity must cover the corpus, and must fail loudly when it does not.

The bug this guards against was silent by construction: `calibrate.py` held a
hand-typed map covering 22 of the 30 drivers, and anything missing fell through
to `driver_id[:3]`. That fallback does not look like a failure - it produces
"NIC" for Hulkenberg and "GUA" for Zhou, which are plausible-looking wrong
codes, on 23.5% of the corpus. Nothing raised, nothing logged, and the frontend
rendered a driver called "NICHUL01" for months.

So the tests below assert coverage rather than mechanics.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

from pipeline import roster, speaker  # noqa: E402
from pipeline.artifacts import iter_race_files  # noqa: E402

RACES = os.path.join(BACKEND, "races")


@pytest.fixture(scope="module")
def corpus_messages():
    msgs = []
    for path in iter_race_files(RACES):
        msgs.extend(json.load(open(path, encoding="utf-8"))["messages"])
    if not msgs:
        pytest.skip("no races built")
    return msgs


class TestRosterCoversTheCorpus:
    def test_every_driver_id_is_known(self, corpus_messages):
        unknown = {m["driver_id"] for m in corpus_messages if not roster.known(m["driver_id"])}
        assert not unknown, (
            f"{len(unknown)} driver_id(s) missing from roster.json: {sorted(unknown)}. "
            "Run: python backend/data/build_roster.py"
        )

    def test_no_message_renders_a_raw_id_as_a_name(self, corpus_messages):
        """The exact symptom: driver_name == driver_id."""
        broken = [m["id"] for m in corpus_messages if m["driver_name"] == m["driver_id"]]
        assert not broken, f"{len(broken)} messages render a raw driver_id as a name"

    def test_codes_are_three_letters(self, corpus_messages):
        bad = {m["driver_code"] for m in corpus_messages
               if len(m["driver_code"]) != 3 or not m["driver_code"].isupper()}
        assert not bad, f"malformed driver codes: {sorted(bad)}"

    @pytest.mark.parametrize("driver_id,code", [
        # Every one of these was wrong under the driver_id[:3] fallback, and
        # wrong in a way that looked right.
        ("NICHUL01", "HUL"),   # was "NIC"
        ("GUAZHO01", "ZHO"),   # was "GUA"
        ("LOGSAR01", "SAR"),   # was "LOG"
        ("ALEALB01", "ALB"),   # was "ALE"
        ("OSCPIA01", "PIA"),   # was "OSC"
        ("KEVMAG01", "MAG"),   # was "KEV"
        ("NYCDEV01", "DEV"),   # was "NYC"
        ("LIALAW01", "LAW"),   # was "LIA"
        ("DANKVY01", "KVY"),   # was "DAN"
        ("ROMGRO01", "GRO"),   # was "ROM"
        ("ROBKUB01", "KUB"),   # was "ROB"
        # These two are irregular dataset ids resolved through ALIASES.
        ("NICLAF01", "LAT"),   # dataset says LAF, the FIA code is LAT
        ("MICSCH02", "MSC"),   # 02 because Michael Schumacher holds 01
    ])
    def test_known_hard_cases(self, driver_id, code):
        assert roster.display(driver_id)[0] == code

    def test_unknown_id_falls_back_visibly(self):
        """A wrong answer must look wrong, not plausible."""
        code, name = roster.display("ZZZNON99")
        assert name == "ZZZNON99"


class TestVocatives:
    def test_surnames_are_available(self):
        """The old first-names-only set could not match 'Verstappen'."""
        assert roster.vocatives().get("verstappen") == "MAXVER01"
        assert roster.vocatives().get("hulkenberg") == "NICHUL01"

    def test_radio_nicknames_are_available(self):
        assert roster.vocatives().get("checo") == "SERPER01"
        assert roster.vocatives().get("hulk") == "NICHUL01"

    def test_previously_missing_drivers_now_match(self):
        """These six had no entry at all, so the vocative cue never fired."""
        for name in ("nyck", "liam", "guanyu", "romain", "daniil", "robert"):
            assert name in roster.vocatives(), f"{name} still missing"

    def test_longest_match_wins(self):
        """Without longest-first ordering, 'danny' shadows 'danny ric'."""
        found = dict(roster.find_vocatives("okay danny ric, box this lap"))
        assert "danny ric" in found

    def test_finds_nothing_in_empty_text(self):
        assert roster.find_vocatives("") == []


class TestSpeakerUsesTheClipsOwnDriver:
    """driver_id is free metadata that the old classifier ignored entirely."""

    def test_own_name_is_the_engineer_addressing_them(self):
        who, why = speaker.classify("Okay Lewis, box box", driver_id="LEWHAM01")
        assert who == "engineer"
        assert "lewis" in why

    def test_another_drivers_name_is_not(self):
        """'Checo is a legend' on Verstappen's channel is Verstappen talking."""
        who, _ = speaker.classify("Checo is a legend. Absolute animal.",
                                  driver_id="MAXVER01")
        assert who != "engineer"

    def test_a_mention_does_not_override_an_explicit_you(self):
        """Weak evidence must not beat strong evidence.

        'We are locking both axles, Danny. You are able to afford more front
        lock' mentions another car and is still plainly the engineer.
        """
        who, _ = speaker.classify(
            "Your front lock is fine, Danny. You are able to afford more.",
            driver_id="MAXVER01")
        assert who == "engineer"

    def test_still_works_without_a_driver_id(self):
        """calibrate.py passes one; research scripts have only text."""
        assert speaker.classify("Okay Lewis, box box")[0] == "engineer"
