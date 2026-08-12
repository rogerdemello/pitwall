"""API contract and the failure modes that would show up in front of judges.

Models are monkeypatched throughout, so these run in milliseconds and test the
*serving* rather than the pipeline.

The one that matters most is the concurrency guard. /api/analyze is an
`async def` that ran ~25s of CPU inline, which blocked the event loop outright:
a single upload froze every other request, including /api/health and the
precomputed race JSON the whole demo runs on.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

import main  # noqa: E402
from pipeline.artifacts import race_ids  # noqa: E402

RACES = os.path.join(BACKEND, "races")


@pytest.fixture(scope="module")
def client():
    # Neutralise the warm-up thread rather than the lifespan itself: nulling
    # the lifespan context makes TestClient fail to start at all, and loading
    # three real models would turn a millisecond suite into a minute.
    main._warm_models = lambda: None
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def a_race():
    ids = race_ids(RACES)
    if not ids:
        pytest.skip("no races built")
    return ids[0]


class TestRaceEndpoints:
    def test_lists_only_races(self, client):
        """races/ holds sidecars and corpus artifacts too. Leaking one 500s."""
        r = client.get("/api/races")
        assert r.status_code == 200
        for row in r.json()["races"]:
            assert not row["race_id"].startswith("_")
            assert "." not in row["race_id"]

    def test_unknown_race_is_404(self, client):
        assert client.get("/api/race/not_a_race").status_code == 404

    def test_race_has_the_shape_the_frontend_reads(self, client, a_race):
        d = client.get(f"/api/race/{a_race}").json()
        assert {"messages", "drivers", "lap_traces", "calibration"} <= set(d)

    def test_unknown_driver_is_404(self, client, a_race):
        assert client.get(f"/api/race/{a_race}/driver/NOBODY99").status_code == 404


class TestAudioPathTraversal:
    @pytest.mark.parametrize("name", [
        "../../etc/passwd", "..\\..\\windows\\system32", "a/b.mp3", "a\\b.mp3",
    ])
    def test_rejected(self, client, a_race, name):
        assert client.get(f"/api/audio/{a_race}/{name}").status_code in (400, 404)


class TestEvidence:
    def test_invariants(self, client, a_race):
        d = client.get(f"/api/evidence/{a_race}").json()
        assert 0 <= d["join_rate"] <= 1
        assert d["suppressed_stress_count"] <= d["suppressed_stress_eligible"]

    def test_empty_race_does_not_500(self, client, monkeypatch, a_race):
        """min()/max() on an empty sequence used to raise a traceback."""
        monkeypatch.setattr(main, "_load_race", lambda rid: {
            "race_id": rid, "messages": [], "drivers": [], "lap_traces": {},
            "calibration": {}, "calibration_own_race": {},
        })
        r = client.get(f"/api/evidence/{a_race}")
        assert r.status_code == 200
        assert r.json()["dsi"]["min"] is None

    def test_compare_uses_a_shared_calibration(self, client):
        """Cross-race comparison is meaningless without one."""
        d = client.get("/api/compare").json()
        if d.get("races"):
            assert d.get("comparable") is True


class TestAnalyze:
    def test_empty_upload_is_400(self, client):
        r = client.post("/api/analyze", files={"file": ("x.mp3", b"", "audio/mpeg")})
        assert r.status_code == 400

    def test_oversize_upload_is_413(self, client):
        big = b"\x00" * (main.MAX_UPLOAD_BYTES + 1024)
        r = client.post("/api/analyze", files={"file": ("x.mp3", big, "audio/mpeg")})
        assert r.status_code == 413

    def test_undecodable_audio_is_400(self, client):
        r = client.post("/api/analyze",
                        files={"file": ("x.mp3", b"not audio at all", "audio/mpeg")})
        assert r.status_code == 400

    def test_response_shape(self, client, monkeypatch, a_race):
        """What frontend/lib/api.ts depends on."""
        import numpy as np
        monkeypatch.setattr(main.librosa, "load",
                            lambda *a, **k: (np.zeros(16000, dtype="float32"), 16000))
        monkeypatch.setattr(main, "_run_models", lambda audio: (
            main.asr.Transcript(text="box box", text_unbiased=None,
                                duration_s=1.0, elapsed_s=0.1),
            main.prosody.Affect(0.6, 0.5, 0.4),
            main.sentiment.TextSentiment("neutral", 0.1, 0.8, 0.1),
        ))
        r = client.post("/api/analyze",
                        files={"file": ("x.mp3", b"xxxx", "audio/mpeg")})
        assert r.status_code == 200
        d = r.json()
        assert {"transcript", "duration_s", "rtf", "text_sentiment", "state",
                "model_id", "matches_corpus_model"} <= set(d)
        assert 0 <= d["state"]["dsi"] <= 100


class TestConcurrencyGuard:
    """The demo-killer. One 25s upload used to freeze the whole app."""

    def test_a_slow_analysis_does_not_block_other_requests(self, client, monkeypatch):
        import threading

        import numpy as np
        monkeypatch.setattr(main.librosa, "load",
                            lambda *a, **k: (np.zeros(16000, dtype="float32"), 16000))

        def slow(audio):
            time.sleep(2.0)
            return (main.asr.Transcript(text="x", text_unbiased=None,
                                        duration_s=1.0, elapsed_s=2.0),
                    main.prosody.Affect(0.5, 0.5, 0.5),
                    main.sentiment.TextSentiment("neutral", 0.1, 0.8, 0.1))
        monkeypatch.setattr(main, "_run_models", slow)

        done = threading.Event()
        threading.Thread(
            target=lambda: (client.post(
                "/api/analyze", files={"file": ("x.mp3", b"xxxx", "audio/mpeg")}),
                done.set()),
            daemon=True).start()

        time.sleep(0.4)               # let the analysis get going
        t0 = time.perf_counter()
        r = client.get("/api/health")
        elapsed = time.perf_counter() - t0

        assert r.status_code == 200
        assert elapsed < 1.0, (
            f"/api/health took {elapsed:.2f}s while an analysis was running - "
            "the event loop is still blocked"
        )
        done.wait(timeout=10)

    def test_timeout_is_bounded(self):
        assert 0 < main._ANALYZE_TIMEOUT_S <= 300

    def test_concurrency_is_bounded(self):
        assert isinstance(main._ANALYZE_SLOTS, asyncio.Semaphore)


class TestCors:
    def test_no_wildcard_origin(self):
        """A wildcard on an endpoint that burns 25s of CPU is an amplifier."""
        assert "*" not in main._ORIGINS

    def test_deployment_origin_is_configurable(self, monkeypatch):
        assert "PITWALL_ORIGINS" in open(
            os.path.join(BACKEND, "main.py"), encoding="utf-8").read()


class TestCacheInvalidation:
    def test_a_rebuilt_race_is_reloaded(self, client, a_race, tmp_path):
        """Serving yesterday's numbers forever is a credibility bug."""
        path = os.path.join(RACES, f"{a_race}.json")
        first = main._load_race(a_race)
        os.utime(path, (time.time(), time.time()))   # touch: new mtime
        second = main._load_race(a_race)
        assert second is not first, "cache ignored the file changing"
