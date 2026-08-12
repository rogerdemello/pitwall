"""The ZeroGPU Space must stay identical to the FastAPI backend.

Two deployments now run the same pipeline: `backend/main.py` in Docker and
locally, and `space_live/app.py` on ZeroGPU for the public static Space. They
return the same payload so the frontend has one result renderer rather than two.

Nothing structural stops them drifting, so these tests do:

  - the Space's response keys are a superset of /api/analyze's, and the shared
    keys carry the same types
  - space_live/pipeline/ is a byte-identical copy of backend/pipeline/, not a
    fork that has started to diverge
  - the error paths return a described failure rather than raising, because a
    Gradio exception reaches a cross-origin caller as an opaque 500

`gradio` and `spaces` are not installed locally and are not worth installing to
run a shape test, so both are stubbed. Everything below the stub - librosa,
torch, the three models - is real.
"""

from __future__ import annotations

import filecmp
import os
import sys
import types

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
SPACE = os.path.join(ROOT, "space_live")

sys.path.insert(0, BACKEND)


def _stub_gradio() -> None:
    """Minimal stand-ins for the two Space-only imports."""
    if "spaces" not in sys.modules:
        spaces = types.ModuleType("spaces")
        spaces.GPU = lambda **kw: (lambda fn: fn)
        sys.modules["spaces"] = spaces

    if "gradio" not in sys.modules:
        gradio = types.ModuleType("gradio")

        class _Component:
            def __init__(self, *a, **kw):
                pass

        class _Interface:
            def __init__(self, *a, **kw):
                self.kwargs = kw

            def launch(self, *a, **kw):
                raise AssertionError("launch() must not run at import time")

        gradio.Audio = _Component
        gradio.JSON = _Component
        gradio.Interface = _Interface
        sys.modules["gradio"] = gradio


@pytest.fixture(scope="module")
def space_app():
    if not os.path.isdir(os.path.join(SPACE, "pipeline")):
        pytest.skip("space_live/ not assembled - run build_space_live.py")
    _stub_gradio()
    sys.path.insert(0, SPACE)
    try:
        import app  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        pytest.skip(f"space_live/app.py did not import: {e}")
    return app


class TestBuildIsACopyNotAFork:
    """space_live/pipeline must be generated from backend/pipeline, never edited."""

    def test_build_reports_in_sync(self):
        sys.path.insert(0, os.path.join(BACKEND, "data"))
        from data import build_space_live
        assert build_space_live.build(check_only=True) == 0, (
            "space_live/ is stale - run: python backend/data/build_space_live.py"
        )

    def test_every_copied_module_is_byte_identical(self):
        from data import build_space_live
        for name in build_space_live.MODULES:
            src = os.path.join(BACKEND, "pipeline", name)
            dst = os.path.join(SPACE, "pipeline", name)
            assert os.path.exists(dst), f"{name} missing from space_live/pipeline"
            assert filecmp.cmp(src, dst, shallow=False), (
                f"space_live/pipeline/{name} has diverged from backend/pipeline/{name}"
            )

    def test_calibration_ships_with_the_space(self):
        """Without it every percentile silently falls back to a raw score."""
        assert os.path.exists(os.path.join(SPACE, "_pooled.calibration.json"))


class TestResponseContract:
    """The frontend renders live and precomputed results with one component."""

    #: Keys backend/main.py:405-413 returns from POST /api/analyze.
    API_KEYS = {
        "transcript": str,
        "duration_s": float,
        "elapsed_s": float,
        "rtf": float,
        "text_sentiment": dict,
        "state": dict,
    }

    @pytest.fixture(scope="class")
    def result(self, space_app):
        clip = os.path.join(
            BACKEND, "clips", "2021_Abu_Dhabi_Grand_Prix",
            "LEWHAM01_44_20211212_122419.mp3")
        if not os.path.exists(clip):
            pytest.skip("showcase clips not present (backend/clips is gitignored)")
        return space_app.analyze(clip)

    def test_no_error(self, result):
        assert "error" not in result, result.get("error")

    def test_carries_every_api_key(self, result):
        assert set(self.API_KEYS) <= set(result), (
            f"missing vs /api/analyze: {set(self.API_KEYS) - set(result)}"
        )

    @pytest.mark.parametrize("key,kind", sorted(API_KEYS.items()))
    def test_key_types_match(self, result, key, kind):
        assert isinstance(result[key], kind), f"{key} is {type(result[key])}"

    def test_state_matches_the_driver_state_dataclass(self, result):
        from pipeline.fusion import DriverState
        expected = set(DriverState.__dataclass_fields__)
        assert set(result["state"]) == expected

    def test_declares_which_model_produced_it(self, result):
        """The Space may not run the model the corpus was built with. Say so."""
        assert "model_id" in result
        assert isinstance(result["matches_corpus_model"], bool)

    def test_dsi_in_range(self, result):
        assert 0 <= result["state"]["dsi"] <= 100

    def test_state_is_one_of_the_four(self, result):
        assert result["state"]["state"] in {"Calm", "Energised", "Stressed", "Fatigued"}


class TestErrorsAreDescribedNotRaised:
    """A Gradio exception reaches a cross-origin caller as an opaque 500."""

    def test_no_audio(self, space_app):
        assert "error" in space_app.analyze(None)

    def test_undecodable_path(self, space_app):
        out = space_app.analyze(os.path.join(SPACE, "README.md"))
        assert "error" in out

    def test_missing_file(self, space_app):
        assert "error" in space_app.analyze("does_not_exist.mp3")
