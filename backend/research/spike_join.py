"""Day-1 spike, part 2: THE make-or-break test.

Does `message_timestamp` from the HF dataset land on a real lap in FastF1?

If this works, every radio message can be placed on a specific lap of a real
race, carrying that lap's time, tyre compound and stint age. That join is the
whole moat. If it doesn't, we fall back to OpenF1 or manual alignment.
"""

import os

import fastf1
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), "..", ".fastf1_cache")
os.makedirs(CACHE, exist_ok=True)
fastf1.Cache.enable_cache(CACHE)

YEAR, GP = 2018, "Australian Grand Prix"

session = fastf1.get_session(YEAR, GP, "R")
# telemetry=True is required: LapStartDate is derived from t0_date, which is only
# populated by the telemetry sync. With telemetry=False the column is all NaT.
session.load(telemetry=True, weather=False, messages=False)

print("=== SESSION ===")
print("  name:", session.event["EventName"], session.name)
print("  t0_date (UTC):", session.t0_date)

laps = session.laps
print("\n=== LAPS TABLE ===")
print("  rows:", len(laps))
print("  columns:", list(laps.columns))

# Radio messages observed in the dataset for this race (UTC).
probes = [
    ("BREHAR01", "28", "2018-03-25T05:14:31.022Z", "Are you folks?"),
    ("BREHAR01", "28", "2018-03-25T05:14:51.850Z", "Box, Brendan, box, box."),
    ("BREHAR01", "28", "2018-03-25T05:15:43.088Z", "Okay, Brennan, we've got to make this work now."),
    ("CARSAI01", "55", "2018-03-25T06:10:48.903Z", "You're 7 tenths quicker than Van der Waal ahead on SuperSalt."),
]

for drv_id, num, ts, text in probes:
    t = pd.Timestamp(ts).tz_localize(None)  # FastF1 dates are naive UTC
    dl = laps[laps["DriverNumber"] == num]
    if dl.empty:
        print(f"\n[{drv_id}] no laps found for car #{num}")
        continue

    print(f"\n--- {drv_id} (#{num}) @ {ts} ---")
    print(f'    "{text}"')
    print(f"    driver laps: {len(dl)}  window: {dl['LapStartDate'].min()} -> {dl['LapStartDate'].max()}")

    # A lap contains the message if it starts at/before it and the next lap starts after.
    hit = None
    for _, lap in dl.iterrows():
        start = lap["LapStartDate"]
        if pd.isna(start) or pd.isna(lap["LapTime"]):
            continue
        end = start + lap["LapTime"]
        if start <= t <= end:
            hit = lap
            break

    if hit is not None:
        print(f"    >>> MATCH: lap {int(hit['LapNumber'])}  "
              f"laptime={hit['LapTime']}  compound={hit['Compound']}  "
              f"tyrelife={hit['TyreLife']}  stint={hit['Stint']}")
    else:
        # Not inside a lap (pit stop, formation, post-retirement). Report nearest.
        dl2 = dl.dropna(subset=["LapStartDate"]).copy()
        dl2["delta"] = (dl2["LapStartDate"] - t).abs()
        near = dl2.nsmallest(1, "delta").iloc[0]
        print(f"    >>> no containing lap; nearest is lap {int(near['LapNumber'])} "
              f"(off by {near['delta']})")
