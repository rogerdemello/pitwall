"""End-to-end: real clip -> ASR -> prosody -> sentiment -> fusion -> real lap.

Runs the whole chain over in-race Abu Dhabi 2021 messages from the two title
contenders and prints what the pit wall would actually see.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import race_data  # noqa: E402
from pipeline import asr, fusion, prosody, sentiment  # noqa: E402

RACE = "2021_Abu_Dhabi_Grand_Prix"
CLIP_DIR = os.path.join(os.path.dirname(__file__), "..", "clips", RACE)

manifest = json.load(open(os.path.join(CLIP_DIR, "manifest.json"), encoding="utf-8"))

print("loading FastF1 session...")
session = race_data.load_session(RACE)

DRIVERS = {"MAXVER01": "33", "LEWHAM01": "44"}
laps_by_driver = {d: race_data.driver_laps(session, num) for d, num in DRIVERS.items()}

# In-race only: Abu Dhabi 2021 started at 13:00 UTC.
picks = [
    m for m in manifest
    if m["driver_id"] in DRIVERS and m["message_timestamp"] > "2021-12-12T13:00"
][:12]

print(f"\n{len(picks)} in-race clips from VER/HAM\n" + "=" * 78)
t_start = time.perf_counter()

for m in picks:
    path = os.path.join(CLIP_DIR, m["audio_file"])
    audio = asr.load_audio(path)

    tr = asr.transcribe(audio)
    af = prosody.analyse(audio)
    se = sentiment.analyse(tr.text)
    st = fusion.fuse(af, se)
    lap = race_data.lap_for_timestamp(laps_by_driver[m["driver_id"]], m["message_timestamp"])

    where = (
        f"LAP {lap.lap_number}  {lap.lap_time_s}s "
        f"(delta {lap.delta_to_median_s:+.2f}s)  {lap.compound} "
        f"age {lap.tyre_life}  P{lap.position:.0f}"
        if lap.in_race else f"[{lap.note}]"
    )

    flag = "  <<< SUPPRESSED STRESS" if st.suppressed_stress else ""
    print(f"\n[{m['driver_id']}] {m['message_timestamp'][11:19]}  {where}")
    print(f'  "{tr.text}"')
    print(f"  DSI {st.dsi:>3}  {st.state:<9} | arousal {st.arousal:.2f} valence {st.valence:.2f} "
          f"| words {se.label} ({se.polarity:+.2f}) | incong {st.incongruence:.2f}{flag}")

print("\n" + "=" * 78)
print(f"total {time.perf_counter() - t_start:.1f}s for {len(picks)} clips")
