"""The in-domain labelling pass is the one gap this project still declares.

`label_affect.py score` never touches audio - it needs only
{message_id: state} - so the listening pass does not have to happen through the
tool's own loop. That makes the tool decomposable, and these tests hold the
seams open: the sample must be reproducible, the import must refuse input it
does not understand rather than silently keeping the half it does, and the
server must not hand out files outside backend/clips.

The sampling property matters most. The sample is stratified over the model's
*own predicted state*, which is what stops a model that guesses one class for
everything from looking good on a sample it chose. It also means the result is
per-class precision and not a base rate, and nothing here should imply otherwise.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

from data import label_affect as la  # noqa: E402
from pipeline.artifacts import race_ids  # noqa: E402

RACE = "2021_Abu_Dhabi_Grand_Prix"
have_race = pytest.mark.skipif(
    not os.path.exists(os.path.join(la.RACES, f"{RACE}.json")),
    reason="showcase race not built",
)


@pytest.fixture
def labels_dir(tmp_path, monkeypatch):
    """Never write to backend/labels/ from a test."""
    monkeypatch.setattr(la, "LABELS", str(tmp_path))
    return tmp_path


@have_race
class TestTheSampleIsHonest:
    def test_it_is_stratified_over_predicted_state(self, labels_dir):
        counts = Counter(m["state"] for m in la.sample_for(RACE, 60))
        assert set(counts) == set(la.STATES), "a class is missing from the sample"
        assert max(counts.values()) - min(counts.values()) <= 1, counts

    def test_it_is_reproducible(self, labels_dir):
        a = [m["id"] for m in la.sample_for(RACE, 40)]
        b = [m["id"] for m in la.sample_for(RACE, 40)]
        assert a == b, "fixed seed, so an interrupted pass resumes on the same clips"

    def test_already_labelled_clips_are_not_offered_again(self, labels_dir):
        first = la.sample_for(RACE, 20)
        la._save_labels(RACE, {first[0]["id"]: "Calm"})
        assert first[0]["id"] not in {m["id"] for m in la.sample_for(RACE, 20)}

    def test_clips_too_short_to_judge_are_excluded(self, labels_dir):
        for m in la.sample_for(RACE, 60):
            assert len(m["transcript"].split()) >= 2


@have_race
class TestImportRefusesWhatItCannotRead:
    def _race_ids(self):
        race = json.load(open(os.path.join(la.RACES, f"{RACE}.json"), encoding="utf-8"))
        return [m["id"] for m in race["messages"]]

    def test_a_json_object_is_accepted(self, labels_dir, tmp_path):
        ids = self._race_ids()[:3]
        f = tmp_path / "j.json"
        f.write_text(json.dumps({ids[0]: "Calm", ids[1]: "stressed"}), encoding="utf-8")
        la.import_labels(RACE, str(f))
        got = la._load_labels(RACE)
        assert got == {ids[0]: "Calm", ids[1]: "Stressed"}, "state names are case-folded"

    def test_a_json_list_is_accepted(self, labels_dir, tmp_path):
        ids = self._race_ids()[:2]
        f = tmp_path / "l.json"
        f.write_text(json.dumps([{"id": ids[0], "label": "Fatigued"}]), encoding="utf-8")
        la.import_labels(RACE, str(f))
        assert la._load_labels(RACE) == {ids[0]: "Fatigued"}

    def test_a_csv_is_accepted_and_its_header_ignored(self, labels_dir, tmp_path):
        ids = self._race_ids()[:2]
        f = tmp_path / "c.csv"
        f.write_text(f"id,label\n{ids[0]},Energised\n", encoding="utf-8")
        la.import_labels(RACE, str(f))
        assert la._load_labels(RACE) == {ids[0]: "Energised"}

    def test_an_unknown_state_is_not_imported(self, labels_dir, tmp_path):
        ids = self._race_ids()[:2]
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({ids[0]: "Furious", ids[1]: "Calm"}), encoding="utf-8")
        la.import_labels(RACE, str(f))
        # The good row lands; the bad one does not become a silent omission.
        assert la._load_labels(RACE) == {ids[1]: "Calm"}

    def test_an_unknown_id_is_not_imported(self, labels_dir, tmp_path):
        f = tmp_path / "ghost.json"
        f.write_text(json.dumps({"not-a-real-message": "Calm"}), encoding="utf-8")
        la.import_labels(RACE, str(f))
        assert la._load_labels(RACE) == {}

    def test_importing_twice_updates_rather_than_duplicates(self, labels_dir, tmp_path):
        ids = self._race_ids()[:1]
        for state in ("Calm", "Stressed"):
            f = tmp_path / f"{state}.json"
            f.write_text(json.dumps({ids[0]: state}), encoding="utf-8")
            la.import_labels(RACE, str(f))
        assert la._load_labels(RACE) == {ids[0]: "Stressed"}


@have_race
class TestScoringWritesWhatTheEvidencePageReads:
    def test_score_writes_the_sidecar_the_api_serves(self, labels_dir, tmp_path,
                                                     monkeypatch):
        # main.py:/api/evidence/{race}/affect reads races/<race>.affect_eval.json,
        # and evidence/page.tsx renders "Still outstanding" until it exists. So
        # producing that file is the whole deliverable of a labelling pass.
        races = tmp_path / "races"
        races.mkdir()
        src = os.path.join(la.RACES, f"{RACE}.json")
        race = json.load(open(src, encoding="utf-8"))
        (races / f"{RACE}.json").write_text(json.dumps(race), encoding="utf-8")
        monkeypatch.setattr(la, "RACES", str(races))

        truth = {m["id"]: m["state"] for m in race["messages"][:12]}
        la._save_labels(RACE, truth)
        la.score(RACE)

        out = races / f"{RACE}.affect_eval.json"
        assert out.exists()
        ev = json.loads(out.read_text(encoding="utf-8"))
        assert ev["n"] == len(truth)
        assert ev["accuracy"] == 1.0, "labels copied from predictions must score 1.0"
        assert set(ev["confusion"]) == set(la.STATES)
        assert set(ev["per_class"]) == set(la.STATES)


class TestTheServerDoesNotLeakTheFilesystem:
    """The labelling server reads from backend/clips by request. It is bound to
    127.0.0.1 and short-lived, but it is still a file server, and the API it
    mirrors (main.py:159) has the same guard for the same reason."""

    def test_every_race_id_is_a_real_race(self):
        assert set(la._race_ids()) == set(race_ids(la.RACES))

    @pytest.mark.parametrize("bad", [
        "../../requirements.txt",
        "../races/_preregistration.json",
        f"{RACE}/../../races/_v1_baseline.json",
        "..",
        "",
        "a",
        "a/b/c",
        f"{RACE}/",
        f"{RACE}/nope.mp3",
    ])
    def test_it_refuses_anything_that_is_not_a_clip(self, bad):
        assert la.clip_path(bad) is None, f"{bad!r} resolved to a file"

    @have_race
    def test_it_resolves_a_real_clip(self):
        race = json.load(
            open(os.path.join(la.RACES, f"{RACE}.json"), encoding="utf-8"))
        name = race["messages"][0]["audio_file"]
        got = la.clip_path(f"{RACE}/{name}")
        if got is None:
            pytest.skip("clips not present on this machine")
        assert os.path.isfile(got)
        assert os.path.commonpath([got, os.path.abspath(la.CLIPS)]) == \
            os.path.abspath(la.CLIPS)
