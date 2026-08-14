"""The GPU handoff: journal -> tarball -> repository.

This path is only exercised once, after an hour of GPU time, at which point a
bug in it is expensive. So it is tested against a synthetic bundle instead.

The property that matters most is the refusal: v2 covers the identical 2,042
clips, so keeping v1 gives a paired comparison for free - the only way to report
honestly if part of the upgrade made something worse. An import that silently
overwrote v1 would destroy that permanently.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

from data import build_race, import_raw_bundle  # noqa: E402
from pipeline import asr  # noqa: E402


def record(i: int, **over) -> dict:
    r = {
        "id": f"race_clip_{i}", "audio_file": f"clip{i}.mp3",
        "racing_number": "44", "message_timestamp": "2021-12-12T12:00:00Z",
        "transcript": f"box box box {i}", "duration_s": 5.0,
        "arousal": 0.6, "valence": 0.4, "dominance": 0.55,
        "text_label": "neutral", "text_polarity": 0.0,
        "text_negative": 0.1, "text_positive": 0.1,
        "windows": 2, "voiced_fraction": 0.7,
        # Resume is scoped to the model that produced the record, so a record
        # without this is not a checkpoint - see build_race._is_done.
        "asr_model_id": asr.MODEL_ID,
    }
    r.update(over)
    return r


def make_bundle(tmp_path, records, *, manifest=True, checksums=True,
                race="2021_Abu_Dhabi_Grand_Prix"):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    journal = work / f"{race}.raw.jsonl"
    journal.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    files = [journal]
    if manifest:
        m = work / "_manifest.json"
        m.write_text(json.dumps({
            "repo_sha": "deadbeef", "asr_model": "openai/whisper-large-v3",
            "asr_backend": "faster-whisper", "gpu": "Tesla T4",
            "clips": len(records), "races": 1,
        }), encoding="utf-8")
        files.append(m)
    if checksums:
        lines = []
        for f in files:
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            lines.append(f"{h}  {f.name}")
        c = work / "checksums.sha256"
        c.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files.append(c)

    bundle = tmp_path / "bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f.name)
    return bundle


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(import_raw_bundle, "RAW", str(raw))
    monkeypatch.setattr(import_raw_bundle, "RAW_V1", str(tmp_path / "raw_v1"))
    return raw


class TestJournal:
    def test_reads_completed_records(self, tmp_path):
        p = tmp_path / "j.jsonl"
        p.write_text("".join(json.dumps(record(i)) + "\n" for i in range(3)),
                     encoding="utf-8")
        assert len(build_race._read_journal(str(p))) == 3

    def test_skips_a_truncated_final_line(self, tmp_path):
        """The expected shape of a killed process, not corruption."""
        p = tmp_path / "j.jsonl"
        good = "".join(json.dumps(record(i)) + "\n" for i in range(3))
        p.write_text(good + '{"id": "race_clip_3", "transc', encoding="utf-8")
        assert len(build_race._read_journal(str(p))) == 3

    def test_failed_clips_are_retried(self, tmp_path):
        p = tmp_path / "j.jsonl"
        p.write_text(
            json.dumps(record(0)) + "\n"
            + json.dumps({"id": "race_clip_1", "error": "boom"}) + "\n",
            encoding="utf-8")
        done = build_race._read_journal(str(p))
        assert "race_clip_0" in done and "race_clip_1" not in done

    def test_missing_journal_is_empty_not_an_error(self, tmp_path):
        assert build_race._read_journal(str(tmp_path / "nope.jsonl")) == {}

    def test_a_record_from_another_model_is_not_a_checkpoint(self, tmp_path):
        """Otherwise the GPU rebuild silently does nothing.

        backend/raw/*.raw.json is committed, so the Colab notebook's clone lands
        a full set of v1 records on the runtime before any model loads. Treating
        those as done skips every clip, leaves the journal empty, and finishes
        the "long run" in seconds - with a small tarball as the only symptom.
        """
        p = tmp_path / "j.jsonl"
        p.write_text(
            json.dumps(record(0, asr_model_id="openai/whisper-small.en")) + "\n"
            + json.dumps(record(1, asr_model_id="openai/whisper-large-v3")) + "\n",
            encoding="utf-8")
        done = build_race._read_journal(str(p))
        keep = f"race_clip_{0 if asr.MODEL_ID == 'openai/whisper-small.en' else 1}"
        drop = f"race_clip_{1 if keep.endswith('0') else 0}"
        assert keep in done and drop not in done

    def test_a_v1_record_predating_the_field_is_refused(self, tmp_path):
        """v1 raw output carries no asr_model_id at all. None must not match."""
        r = record(0)
        del r["asr_model_id"]
        p = tmp_path / "j.jsonl"
        p.write_text(json.dumps(r) + "\n", encoding="utf-8")
        assert build_race._read_journal(str(p)) == {}


class TestVerification:
    def test_a_good_bundle_imports(self, tmp_path, isolated, capsys):
        b = make_bundle(tmp_path, [record(i) for i in range(5)])
        assert import_raw_bundle.main([str(b)]) == 0
        out = json.load(open(isolated / "2021_Abu_Dhabi_Grand_Prix.raw.json",
                             encoding="utf-8"))
        assert len(out["messages"]) == 5

    def test_a_tampered_bundle_is_rejected(self, tmp_path, isolated):
        b = make_bundle(tmp_path, [record(i) for i in range(3)])
        # Repack with a changed journal but the original checksums.
        import shutil, tempfile
        with tempfile.TemporaryDirectory() as work:
            with tarfile.open(b) as tar:
                tar.extractall(work)
            j = os.path.join(work, "2021_Abu_Dhabi_Grand_Prix.raw.jsonl")
            with open(j, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record(99)) + "\n")
            with tarfile.open(b, "w:gz") as tar:
                for f in sorted(os.listdir(work)):
                    tar.add(os.path.join(work, f), arcname=f)
        assert import_raw_bundle.main([str(b)]) == 1

    def test_missing_checksums_is_rejected(self, tmp_path, isolated):
        b = make_bundle(tmp_path, [record(0)], checksums=False)
        assert import_raw_bundle.main([str(b)]) == 1

    def test_missing_manifest_is_rejected(self, tmp_path, isolated):
        """Without it the artifact cannot be traced to a commit."""
        b = make_bundle(tmp_path, [record(0)], manifest=False)
        assert import_raw_bundle.main([str(b)]) == 1

    def test_a_record_missing_stage2_fields_is_rejected(self, tmp_path, isolated):
        b = make_bundle(tmp_path, [record(0), record(1, arousal=None)])
        assert import_raw_bundle.main([str(b)]) == 1

    def test_empty_journal_is_rejected(self, tmp_path, isolated):
        b = make_bundle(tmp_path, [])
        assert import_raw_bundle.main([str(b)]) == 1


class TestV1IsPreserved:
    def test_v1_is_moved_aside_not_overwritten(self, tmp_path, isolated):
        """The paired v1-vs-v2 comparison depends entirely on this."""
        old = isolated / "2021_Abu_Dhabi_Grand_Prix.raw.json"
        old.write_text(json.dumps({"race_id": "x", "messages": [{"id": "v1"}]}),
                       encoding="utf-8")
        b = make_bundle(tmp_path, [record(i) for i in range(3)])
        assert import_raw_bundle.main([str(b)]) == 0

        preserved = json.load(open(
            os.path.join(import_raw_bundle.RAW_V1,
                         "2021_Abu_Dhabi_Grand_Prix.raw.json"), encoding="utf-8"))
        assert preserved["messages"][0]["id"] == "v1"

    def test_it_refuses_to_clobber_an_existing_v1(self, tmp_path, isolated):
        os.makedirs(import_raw_bundle.RAW_V1, exist_ok=True)
        with open(os.path.join(import_raw_bundle.RAW_V1, "old.raw.json"), "w") as fh:
            fh.write("{}")
        (isolated / "2021_Abu_Dhabi_Grand_Prix.raw.json").write_text("{}",
                                                                    encoding="utf-8")
        b = make_bundle(tmp_path, [record(0)])
        assert import_raw_bundle.main([str(b)]) == 1, (
            "importing over an existing v1 must be refused"
        )


class TestConversion:
    def test_dry_run_writes_nothing(self, tmp_path, isolated):
        b = make_bundle(tmp_path, [record(i) for i in range(3)])
        assert import_raw_bundle.main([str(b), "--dry-run"]) == 0
        assert not list(isolated.iterdir())

    def test_a_retried_clip_keeps_the_later_record(self, tmp_path, isolated):
        """A resumed run appends; the successful retry must win."""
        rows = [{"id": "race_clip_0", "error": "boom"}, record(0, transcript="good")]
        b = make_bundle(tmp_path, rows)
        assert import_raw_bundle.main([str(b)]) == 0
        out = json.load(open(isolated / "2021_Abu_Dhabi_Grand_Prix.raw.json",
                             encoding="utf-8"))
        assert out["messages"][0]["transcript"] == "good"

    def test_the_journal_ships_alongside_the_converted_file(self, tmp_path, isolated):
        b = make_bundle(tmp_path, [record(0)])
        import_raw_bundle.main([str(b)])
        assert (isolated / "2021_Abu_Dhabi_Grand_Prix.raw.jsonl").exists()
