"""Join radio messages to the lap they were actually spoken on.

This is the piece that separates the project from a clip-labelling demo. Each
message carries a UTC timestamp; FastF1 gives every lap an absolute
`LapStartDate`. Intersect them and a radio message inherits the real lap number,
lap time, tyre compound and stint age from the race it came from.

Verified on 2018 Australian GP: a "Box, box, box" call landed on lap 1 (stint 1,
SUPERSOFT) and the next message on lap 2 (stint 2, SOFT) - the telemetry
independently confirms the pit stop the radio ordered.

Gotcha worth keeping: `LapStartDate` is derived from `t0_date`, which is only
populated when telemetry is loaded. With `telemetry=False` the whole column is
NaT and every join silently misses.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import fastf1
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".fastf1_cache")
os.makedirs(CACHE, exist_ok=True)
fastf1.Cache.enable_cache(CACHE)


@dataclass
class LapContext:
    lap_number: int | None
    lap_time_s: float | None
    compound: str | None
    tyre_life: float | None
    stint: int | None
    position: float | None
    delta_to_median_s: float | None  # this lap vs the driver's own race median
    in_race: bool
    note: str = ""
    # True only for a green-flag racing lap with no pit entry or exit. A pit stop
    # costs ~20s and a safety car ~30s; leaving those laps in any stress-vs-pace
    # comparison drowns the effect being measured in noise an order of magnitude
    # larger. Charts still show every lap - only the statistics exclude these.
    is_representative: bool = False
    track_status: str | None = None
    pit_lap: bool = False


def parse_race_id(race_id: str) -> tuple[int, str]:
    """'2021_Abu_Dhabi_Grand_Prix' -> (2021, 'Abu Dhabi Grand Prix')."""
    m = re.match(r"^(\d{4})_(.+)$", race_id)
    if not m:
        raise ValueError(f"unrecognised race_id: {race_id}")
    return int(m.group(1)), m.group(2).replace("_", " ")


def load_session(race_id: str):
    year, gp = parse_race_id(race_id)
    session = fastf1.get_session(year, gp, "R")
    # telemetry=True is required for LapStartDate. See module docstring.
    session.load(telemetry=True, weather=False, messages=False)
    return session


def driver_laps(session, racing_number: str) -> pd.DataFrame:
    laps = session.laps
    dl = laps[laps["DriverNumber"] == str(racing_number)].copy()
    return dl.dropna(subset=["LapStartDate"])


def _is_representative(lap) -> bool:
    """A green-flag racing lap: no pit entry/exit, all-clear track status.

    FastF1 TrackStatus is a concatenation of codes seen during the lap:
    1 all clear, 2 yellow, 4 safety car, 5 red, 6/7 VSC. Anything other than a
    pure '1' means the lap was not run at racing speed.
    """
    if pd.isna(lap["LapTime"]):
        return False
    if pd.notna(lap["PitInTime"]) or pd.notna(lap["PitOutTime"]):
        return False
    status = str(lap["TrackStatus"]) if pd.notna(lap["TrackStatus"]) else ""
    return set(status) <= {"1"} and status != ""


def lap_for_timestamp(dl: pd.DataFrame, iso_ts: str) -> LapContext:
    """Find which of this driver's laps contains the given UTC timestamp."""
    if dl.empty:
        return LapContext(None, None, None, None, None, None, None, False, "no lap data")

    t = pd.Timestamp(iso_ts).tz_localize(None)

    # Pace baseline: the median of this driver's *representative* laps only, so
    # pit stops and safety-car laps don't drag the baseline they're compared to.
    rep = dl[dl.apply(_is_representative, axis=1)].dropna(subset=["LapTime"])
    base = rep if not rep.empty else dl.dropna(subset=["LapTime"])
    median_s = base["LapTime"].dt.total_seconds().median() if not base.empty else None

    for _, lap in dl.iterrows():
        start = lap["LapStartDate"]
        lt = lap["LapTime"]
        if pd.isna(lt):
            continue
        if start <= t <= start + lt:
            secs = lt.total_seconds()
            status = str(lap["TrackStatus"]) if pd.notna(lap["TrackStatus"]) else None
            is_pit = pd.notna(lap["PitInTime"]) or pd.notna(lap["PitOutTime"])
            return LapContext(
                lap_number=int(lap["LapNumber"]),
                lap_time_s=round(secs, 3),
                compound=lap["Compound"] if pd.notna(lap["Compound"]) else None,
                tyre_life=float(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) else None,
                stint=int(lap["Stint"]) if pd.notna(lap["Stint"]) else None,
                position=float(lap["Position"]) if pd.notna(lap["Position"]) else None,
                delta_to_median_s=round(secs - median_s, 3) if median_s else None,
                in_race=True,
                is_representative=bool(_is_representative(lap)),
                track_status=status,
                pit_lap=bool(is_pit),
            )

    # Outside any lap: pre-race build-up, a pit stop, or after retirement.
    first, last = dl["LapStartDate"].min(), dl["LapStartDate"].max()
    if t < first:
        note = "before race start"
    elif t > last:
        note = "after final lap"
    else:
        note = "between laps (pit stop or stationary)"
    return LapContext(None, None, None, None, None, None, None, False, note)
