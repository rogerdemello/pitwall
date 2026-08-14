"""The demo script must describe the corpus that actually exists.

`docs/DEMO.md` names specific clips with specific numbers — a DSI, a lap time,
a transcript, a percentile. Those are read off the corpus, and the corpus gets
rebuilt. After the v2 rebuild every one of them had moved: the showcase clip
went from DSI 82 to 92, its transcript changed from "I mean, I have to stop" to
"I'm in a two-stop", and the suppressed-stress rate went from 8% to 6.5%.

Nothing caught that, because no test had any reason to read the presentation
script. The failure mode is specific and expensive: you stand in front of
judges, point at the screen, and say a number that is not on it.

So the beats are pinned here — asserted against the race files *and* against the
prose that quotes them, the same way `test_docs_numbers.py` pins the evidence.
If a rebuild moves them again this fails, which is the cheapest possible place
to find out.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

RACES = os.path.join(BACKEND, "races")
SHOWCASE = "2021_Abu_Dhabi_Grand_Prix"
DEMO = os.path.join(ROOT, "docs", "DEMO.md")

needs_corpus = pytest.mark.skipif(
    not os.path.exists(os.path.join(RACES, f"{SHOWCASE}.json")),
    reason="showcase race not built",
)


@pytest.fixture(scope="module")
def race():
    with open(os.path.join(RACES, f"{SHOWCASE}.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def demo():
    with open(DEMO, encoding="utf-8") as f:
        return f.read()


def by_driver_lap(race, code, lap):
    return [m for m in race["messages"]
            if m.get("driver_code") == code
            and (m.get("lap") or {}).get("lap_number") == lap]


@needs_corpus
class TestBeatOneTheShowcaseClip:
    """Race Replay → SAI lap 35. The clip the whole demo opens on."""

    @pytest.fixture(scope="class")
    def clip(self, race):
        found = by_driver_lap(race, "SAI", 35)
        assert found, "no SAI message on lap 35 - the demo opens on a clip that is gone"
        return max(found, key=lambda m: m["dsi"])

    def test_it_is_the_high_dsi_stressed_moment(self, clip):
        assert clip["state"] == "Stressed"
        assert clip["dsi"] >= 85, "the demo sells this as a peak-stress call"

    def test_the_demo_quotes_its_actual_dsi(self, clip, demo):
        assert f"DSI {clip['dsi']}" in demo, (
            f"DEMO.md does not mention DSI {clip['dsi']}; the corpus moved under it")

    def test_the_demo_quotes_its_actual_transcript(self, clip, demo):
        # A distinctive fragment, so ordinary rewording of the script is fine
        # but a changed transcription is not.
        head = " ".join(clip["transcript"].split()[:4]).rstrip(",.")
        assert head in demo, f"DEMO.md does not quote the current transcript ({head!r}...)"

    def test_the_demo_quotes_its_actual_percentiles(self, clip, demo):
        for pct in (clip["arousal_pct"], clip["valence_pct"]):
            nth = round(pct * 100)
            assert re.search(rf"\b{nth}(st|nd|rd|th)\b", demo), (
                f"DEMO.md does not state the {nth}th percentile")

    def test_the_audio_actually_ships(self, clip):
        """The demo plays it. A missing file is a dead click on stage."""
        p = os.path.join(BACKEND, "clips", SHOWCASE, clip["audio_file"])
        if not os.path.isdir(os.path.join(BACKEND, "clips", SHOWCASE)):
            pytest.skip("clips not present on this machine")
        assert os.path.exists(p), f"{clip['audio_file']} is missing"


@needs_corpus
class TestBeatTwoTheTelemetryJoin:
    """HAM lap 15 — the moment that proves the join, and the demo's strongest."""

    @pytest.fixture(scope="class")
    def stop(self, race):
        found = by_driver_lap(race, "HAM", 15)
        assert found, "no HAM message on lap 15 - the join demo is gone"
        return found[0]

    def test_the_radio_orders_the_stop(self, stop):
        assert "box" in stop["transcript"].lower()

    def test_the_telemetry_confirms_it_independently(self, stop, race):
        lap = stop["lap"]
        assert lap.get("compound") == "HARD", "the compound change is the evidence"
        assert lap.get("stint", 0) >= 2, "a stop increments the stint"

    def test_the_demo_quotes_a_delta_that_matches_the_screen(self, stop, race, demo):
        """The delta must be real, and must be the one the app displays.

        Two medians are defensible - over all laps, or over racing laps only -
        and the app uses the second. They differ by about 0.2s here. The demo
        has to quote what is on screen, so this checks the figure in the script
        is a plausible delta rather than pinning one definition: too loose to
        forbid the app's choice, too tight to let a wrong number through.
        """
        import re
        import statistics as st
        trace = race["lap_traces"]["LEWHAM01"]
        median = st.median(x["seconds"] for x in trace if x.get("seconds"))
        lap15 = next(x for x in trace if x["lap"] == 15)
        delta = lap15["seconds"] - median
        assert delta > 15, "lap 15 should be dramatically slow - it is a pit stop"

        quoted = [float(m) for m in re.findall(r"\+(\d+\.\d+)s", demo)]
        assert any(abs(q - delta) < 1.0 for q in quoted), (
            f"DEMO.md quotes {quoted}; none is within 1s of the measured "
            f"+{delta:.2f}s")

    def test_the_demo_names_the_engineer_banner(self, demo):
        """The app suppresses driver-state claims on engineer transmissions, and
        this beat is the clearest place to show it."""
        assert "engineer voice" in demo.lower()


@needs_corpus
class TestBeatThreeSuppressedStress:
    """The feature the project claims nobody else built."""

    def test_the_showcase_race_has_one_to_point_at(self, race):
        flagged = [m for m in race["messages"] if m.get("suppressed_stress")]
        assert flagged, "no suppressed-stress example in the showcase race"

    def test_the_demo_points_at_a_real_one(self, race, demo):
        flagged = [m for m in race["messages"] if m.get("suppressed_stress")]
        best = max(flagged, key=lambda m: m["dsi"])
        assert f"DSI {best['dsi']}" in demo, (
            f"DEMO.md should point at the strongest example (DSI {best['dsi']})")
        head = " ".join(best["transcript"].split()[:3]).rstrip(",.")
        assert head in demo, f"DEMO.md does not quote it ({head!r}...)"

    def test_the_demo_states_the_real_rate(self, demo):
        """Over eligible messages, not all messages - engineers can never be
        flagged, so the wider denominator would understate it."""
        from pipeline.artifacts import iter_race_files
        elig = flag = 0
        for p in iter_race_files(RACES):
            with open(p, encoding="utf-8") as f:
                for m in json.load(f)["messages"]:
                    if m.get("speaker") != "engineer":
                        elig += 1
                        flag += bool(m.get("suppressed_stress"))
        rate = flag / elig
        assert f"{rate:.1%}" in demo, (
            f"DEMO.md does not state the current rate of {rate:.1%} "
            f"({flag} of {elig} eligible)")
