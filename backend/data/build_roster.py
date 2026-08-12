"""Generate the driver roster from FastF1, instead of typing it a third time.

Driver identity was hand-maintained in two places that had drifted apart:

  calibrate.py    driver_id -> (code, full name), covering 22 of the 30 drivers
                  in the corpus. The other 8 fell through to a `driver_id[:3]`
                  fallback, so 479 messages (23.5%) rendered a driver named
                  "NICHUL01" with the code "NIC" - and the code was wrong as
                  well as ugly: Hulkenberg is HUL, Zhou is ZHO, Sargeant is SAR.

  speaker.py      a flat set of ~30 lowercase first names used to detect the
                  vocative cue. It had the mirror gap, so for nyck, liam,
                  guanyu, romain, daniil and robert the strongest single
                  attribution signal never fired at all.

Hand-typing is how both gaps appeared, so this reads the truth out of FastF1's
session results and writes it once. The output is committed, so nothing at
runtime depends on FastF1 or the network.

The dataset's driver_id is FIRST3 + LAST3 + NN, uppercased and ASCII-folded -
MAXVER01, NICHUL01, GUAZHO01. That is reconstructible from FirstName/LastName,
which is what lets this match without a hand-written mapping.

Usage:
    python backend/data/build_roster.py
    python backend/data/build_roster.py --check     # fail if the corpus has
                                                    # drivers the roster lacks
"""

from __future__ import annotations

import collections
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.artifacts import iter_race_files  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, "..")
RACES = os.path.join(BACKEND, "races")
OUT = os.path.join(BACKEND, "pipeline", "roster.json")

#: Radio nicknames and shortenings the pit wall actually uses. FastF1 knows the
#: legal name; it does not know that Perez is "Checo" on the radio or that
#: Ricciardo is "Danny Ric". This is the only hand-entered content here, it is
#: additive, and a missing entry costs recall on one cue rather than breaking a
#: name.
#: Dataset ids that FIRST3+LAST3+NN does not reconstruct. Both are genuine
#: irregularities in the source data rather than bugs here, so they are declared
#: rather than worked around:
#:   NICLAF01  the dataset abbreviates Latifi as LAF; the FIA code is LAT
#:   MICSCH02  Mick Schumacher takes 02 because Michael Schumacher holds 01
#: Anything else that fails to match is a real error and stops the build.
ALIASES = {
    "NICLAF01": "NICLAT01",
    "MICSCH02": "MICSCH01",
}

NICKNAMES = {
    "SERPER01": ["checo"],
    "DANRIC01": ["danny", "danny ric"],
    "SEBVET01": ["seb"],
    "VALBOT01": ["val"],
    "GUAZHO01": ["zhou"],          # Zhou Guanyu - surname first, addressed as Zhou
    "KIMRAI01": ["kimi"],
    "ANTGIO01": ["gio"],
    "MICSCH02": ["mick"],
    "NICHUL01": ["hulk", "nico"],
    "ALEALB01": ["alex"],
    "OSCPIA01": ["osc"],
    "FERALO01": ["nando"],
}


def fold(s: str) -> str:
    """Strip accents and non-letters: Hulkenberg from Hülkenberg, Perez from Pérez."""
    n = unicodedata.normalize("NFKD", s)
    return "".join(c for c in n if c.isalpha() and not unicodedata.combining(c)).upper()


def dataset_id(first: str, last: str, seq: str = "01") -> str:
    return f"{fold(first)[:3]}{fold(last)[:3]}{seq}"


def corpus_driver_ids() -> dict[str, int]:
    """Every driver_id in the built corpus, with its message count."""
    counts: collections.Counter = collections.Counter()
    for path in iter_race_files(RACES):
        d = json.load(open(path, encoding="utf-8"))
        for m in d["messages"]:
            counts[m["driver_id"]] += 1
    return dict(counts)


def from_fastf1(race_ids: list[str]) -> dict[str, dict]:
    """Pull every driver who appeared in any session in the corpus."""
    import fastf1

    fastf1.Cache.enable_cache(os.path.join(BACKEND, ".fastf1_cache"))
    out: dict[str, dict] = {}

    for race_id in race_ids:
        year = int(race_id[:4])
        gp = race_id[5:].replace("_Grand_Prix", "").replace("_", " ")
        try:
            s = fastf1.get_session(year, gp, "R")
            s.load(telemetry=False, weather=False, messages=False)
        except Exception as e:
            print(f"   !! {race_id}: {type(e).__name__}: {e}")
            continue

        for _, row in s.results.iterrows():
            first, last = str(row["FirstName"]), str(row["LastName"])
            did = dataset_id(first, last)
            if did in out:
                out[did]["numbers"].add(str(row["DriverNumber"]))
                out[did]["seasons"].add(year)
                continue
            out[did] = {
                "driver_id": did,
                "code": str(row["Abbreviation"]),
                "full_name": str(row["FullName"]),
                "first_name": first,
                "last_name": last,
                "numbers": {str(row["DriverNumber"])},
                "seasons": {year},
            }
        print(f"   {race_id:<34} {len(s.results):>3} drivers")
    return out


def build(check_only: bool = False) -> int:
    corpus = corpus_driver_ids()
    if not corpus:
        print("no races built")
        return 1

    if check_only:
        if not os.path.exists(OUT):
            print(f"!! no roster at {OUT} - run build_roster.py")
            return 1
        roster = json.load(open(OUT, encoding="utf-8"))["drivers"]
        missing = {k: v for k, v in corpus.items() if k not in roster}
        if missing:
            print(f"!! roster is missing {len(missing)} driver(s) present in the "
                  f"corpus ({sum(missing.values())} messages):")
            for k, v in sorted(missing.items(), key=lambda x: -x[1]):
                print(f"     {k}  {v} messages")
            return 1
        print(f"roster covers all {len(corpus)} drivers in the corpus")
        return 0

    race_ids = [os.path.basename(p)[:-5] for p in iter_race_files(RACES)]
    print(f"reading {len(race_ids)} sessions from FastF1 ...")
    found = from_fastf1(race_ids)

    drivers, unmatched = {}, []
    for did, n in sorted(corpus.items(), key=lambda x: -x[1]):
        rec = found.get(did) or found.get(ALIASES.get(did, ""))
        if rec is None:
            unmatched.append((did, n))
            continue
        first_names = [rec["first_name"].lower()]
        drivers[did] = {
            "code": rec["code"],
            "full_name": rec["full_name"],
            "first_name": rec["first_name"],
            "last_name": rec["last_name"],
            "numbers": sorted(rec["numbers"]),
            "seasons": sorted(rec["seasons"]),
            # Lowercased forms an engineer might use as a vocative on the radio.
            "vocatives": sorted(set(
                first_names
                + [rec["last_name"].lower()]
                + NICKNAMES.get(did, [])
            )),
            "messages": n,
        }

    if unmatched:
        print(f"\n!! {len(unmatched)} driver_id(s) in the corpus not found in any "
              "session:")
        for did, n in unmatched:
            print(f"     {did}  {n} messages")
        print("   These would render as raw IDs. Add the session that contains "
              "them, or map them by hand in NICKNAMES' sibling.")
        return 1

    payload = {
        "generated_by": "backend/data/build_roster.py",
        "source": "FastF1 session.results across every race in backend/races/",
        "note": (
            "Committed on purpose: nothing at runtime should need FastF1 or the "
            "network to render a driver's name. Regenerate when a race is added."
        ),
        "n_drivers": len(drivers),
        "n_messages": sum(d["messages"] for d in drivers.values()),
        "drivers": drivers,
    }
    json.dump(payload, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"\n{len(drivers)} drivers, {payload['n_messages']} messages")
    print(f"{'driver_id':<12}{'code':>6}  {'name':<24}{'msgs':>6}  vocatives")
    for did, d in sorted(drivers.items(), key=lambda x: -x[1]["messages"]):
        print(f"{did:<12}{d['code']:>6}  {d['full_name']:<24}{d['messages']:>6}  "
              f"{', '.join(d['vocatives'])}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build(check_only="--check" in sys.argv))
