"""Settle the era confound: is race separation real, or just recording era?

The cross-era corpus showed DSI separating races by 5.6 points. But it spanned
2019-2023, and radio broadcast encoding is not constant across seasons, so some
of that could have been the recording rather than the driving. We flagged it as
an open confound rather than pretending otherwise - and notably, the one race
that failed its prediction was the oldest in the set, which is exactly what the
confound would produce.

There is only one clean way to test it: hold the season fixed. Nine of the
twelve races now built are from 2023 alone.

Two tests:
  1. **Within-season spread.** If races still separate by a similar margin when
     every recording comes from the same season, era does not explain it.
  2. **Direct era comparison.** Pre-2023 races versus 2023 races. A systematic
     offset between them is the confound showing up directly.

Usage:
    python backend/data/era_analysis.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.calibration import CROSS_RACE_SOURCES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
OUT = os.path.join(RACES, "_era_analysis.json")


def load() -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(RACES, "*.json"))):
        base = os.path.basename(path)
        if base.startswith("_") or base.count(".") > 1:
            continue
        d = json.load(open(path, encoding="utf-8"))
        dsis = [m["dsi"] for m in d["messages"] if m.get("speaker") != "engineer"]
        if len(dsis) < 20:
            continue
        out.append({
            "race_id": d["race_id"],
            "race": d["grand_prix"].replace(" Grand Prix", ""),
            "season": int(d["race_id"][:4]),
            "n": len(dsis),
            "mean": st.fmean(dsis),
            "sd": st.pstdev(dsis),
            "se": st.pstdev(dsis) / math.sqrt(len(dsis)),
            "calibration": d.get("calibration_source"),
            "_dsis": dsis,
        })
    return out


def z_test(a: dict, b: dict) -> tuple[float, float, float]:
    diff = a["mean"] - b["mean"]
    sed = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    z = diff / sed if sed else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return diff, z, p


def spread(rows: list[dict]) -> float:
    return max(r["mean"] for r in rows) - min(r["mean"] for r in rows)


def main() -> None:
    rows = load()
    if len(rows) < 4:
        print("not enough races built")
        return

    sources = {r["calibration"] for r in rows}
    if not sources <= CROSS_RACE_SOURCES:
        print(f"!! calibration sources are {sources}; cross-race comparison needs "
              f"a shared reference ({sorted(CROSS_RACE_SOURCES)}). "
              "Run finish_corpus.py.")
        return

    rows.sort(key=lambda r: -r["mean"])
    season_2023 = [r for r in rows if r["season"] == 2023]
    older = [r for r in rows if r["season"] != 2023]

    all_spread = spread(rows)
    within_spread = spread(season_2023) if len(season_2023) > 1 else None

    print(f"{len(rows)} races, {len(season_2023)} of them from 2023\n")
    print(f"{'race':<22}{'season':>7}{'n':>6}{'mean':>8}   95% CI")
    for r in rows:
        lo, hi = r["mean"] - 1.96 * r["se"], r["mean"] + 1.96 * r["se"]
        print(f"{r['race']:<22}{r['season']:>7}{r['n']:>6}{r['mean']:>8.1f}   [{lo:.1f}, {hi:.1f}]")

    # Test 1: does separation survive with the season held fixed?
    contrasts = []
    if len(season_2023) > 1:
        hi_r = max(season_2023, key=lambda r: r["mean"])
        lo_r = min(season_2023, key=lambda r: r["mean"])
        diff, z, p = z_test(hi_r, lo_r)
        n_tests = len(season_2023) - 1
        alpha = 0.05 / max(1, n_tests)
        contrasts.append({
            "comparison": f"{hi_r['race']} vs {lo_r['race']} (both 2023)",
            "diff": round(diff, 2), "z": round(z, 2), "p": round(p, 5),
            "bonferroni_alpha": round(alpha, 4),
            "survives": p < alpha,
        })
        print(f"\nwithin-2023 spread: {within_spread:.1f} points")
        print(f"  {hi_r['race']} vs {lo_r['race']}: {diff:+.1f}, z={z:.2f}, p={p:.5f} "
              f"({'holds' if p < alpha else 'not significant'} at Bonferroni {alpha:.4f})")
    print(f"cross-era spread:   {all_spread:.1f} points")

    # Test 2: is there a systematic offset between eras?
    era_gap = None
    if older and season_2023:
        a = {"mean": st.fmean([r["mean"] for r in season_2023]),
             "se": st.fmean([r["se"] for r in season_2023]) / math.sqrt(len(season_2023))}
        b = {"mean": st.fmean([r["mean"] for r in older]),
             "se": st.fmean([r["se"] for r in older]) / math.sqrt(len(older))}
        diff, z, p = z_test(a, b)
        era_gap = {"mean_2023": round(a["mean"], 2), "mean_pre2023": round(b["mean"], 2),
                   "diff": round(diff, 2), "z": round(z, 2), "p": round(p, 4),
                   "systematic_era_effect": p < 0.05}
        print(f"\nera comparison: 2023 mean {a['mean']:.1f} vs pre-2023 mean {b['mean']:.1f} "
              f"({diff:+.1f}, p={p:.4f})")

    verdict = _verdict(within_spread, all_spread, contrasts, era_gap)
    print(f"\nVERDICT: {verdict}")

    for r in rows:
        r.pop("_dsis", None)
    json.dump({
        "races": rows,
        "n_races": len(rows),
        "n_2023": len(season_2023),
        "within_season_spread": round(within_spread, 2) if within_spread else None,
        "cross_era_spread": round(all_spread, 2),
        "contrasts": contrasts,
        "era_comparison": era_gap,
        "verdict": verdict,
    }, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"wrote {OUT}")


def _verdict(within, across, contrasts, era_gap) -> str:
    if within is None:
        return "Not enough same-season races to test the confound."

    bits = []
    ratio = within / across if across else 0
    if ratio >= 0.8:
        bits.append(
            f"Holding the season fixed barely changes the picture: races from 2023 "
            f"alone spread {within:.1f} points, against {across:.1f} across all "
            "seasons. Recording era does not explain the separation."
        )
    elif ratio >= 0.5:
        bits.append(
            f"Within one season races still spread {within:.1f} points against "
            f"{across:.1f} across eras, so era accounts for some of the gap but "
            "not most of it."
        )
    else:
        bits.append(
            f"Within one season the spread collapses to {within:.1f} points from "
            f"{across:.1f} across eras. Most of the apparent separation was "
            "recording era, and the earlier claim should be retracted."
        )

    if contrasts and contrasts[0]["survives"]:
        c = contrasts[0]
        bits.append(f"{c['comparison']} differs by {c['diff']:+.1f} (p={c['p']:.5f}), "
                    "surviving Bonferroni correction.")
    elif contrasts:
        bits.append(f"{contrasts[0]['comparison']} does not survive correction, though.")

    if era_gap:
        if era_gap["systematic_era_effect"]:
            bits.append(
                f"There IS a systematic era offset ({era_gap['diff']:+.1f} points, "
                f"p={era_gap['p']}), so cross-era comparisons stay suspect."
            )
        else:
            bits.append(
                f"And there is no systematic offset between eras "
                f"({era_gap['diff']:+.1f} points, p={era_gap['p']}), which is the "
                "confound tested directly."
            )
    return " ".join(bits)


if __name__ == "__main__":
    main()
