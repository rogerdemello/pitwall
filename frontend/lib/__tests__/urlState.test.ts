/**
 * URL state: the round-trip, and the promise that a hand-edited URL degrades
 * rather than breaking the page.
 *
 * This is the highest-value unit test in the frontend suite. Every shareable
 * link depends on `parse(serialise(x)) === x`, and every link someone types by
 * hand depends on `parse` never throwing.
 */

import { describe, expect, it } from "vitest";

import { SHOWCASE_RACE } from "../constants";
import {
  DEFAULTS, NAVIGATIONAL, activeFilters, isDefault, parseUrlState,
  serialiseUrlState, type UrlState,
} from "../urlState";

const state = (over: Partial<UrlState> = {}): UrlState => ({ ...DEFAULTS, ...over });

describe("round trip", () => {
  const cases: [string, UrlState][] = [
    ["defaults", state()],
    ["a race", state({ race: "2023_Monaco_Grand_Prix" })],
    ["a driver", state({ driver: "LEWHAM01" })],
    ["a message", state({ msg: "2021_Abu_Dhabi_Grand_Prix_LEWHAM01_44_x" })],
    ["a lap range", state({ laps: [12, 40] })],
    ["states", state({ state: ["Stressed", "Fatigued"] })],
    ["a dsi range", state({ dsi: [60, 90] })],
    ["speakers", state({ speaker: ["driver", "unknown"] })],
    ["severities", state({ sev: ["act", "watch"] })],
    ["flags", state({ flags: ["suppressed", "hasrec"] })],
    ["a query", state({ q: "box box" })],
    ["a sort", state({ sort: "dsi:desc" })],
    ["table twins", state({ tables: ["pace", "dsi"] })],
    ["an evidence section", state({ sec: "calibration" })],
    ["everything at once", state({
      race: "2023_Qatar_Grand_Prix", driver: "MAXVER01", msg: "abc",
      laps: [5, 30], state: ["Stressed"], dsi: [40, 80], speaker: ["driver"],
      sev: ["act"], flags: ["onlap"], q: "grip", sort: "dsi:desc",
      tables: ["pace"], sec: "gold",
    })],
  ];

  it.each(cases)("survives %s", (_name, s) => {
    expect(parseUrlState(serialiseUrlState(s))).toEqual(s);
  });
});

describe("serialisation is minimal", () => {
  it("emits nothing for a default state", () => {
    // A fresh URL should stay clean; a shared one should carry only what
    // was actually changed.
    expect(serialiseUrlState(DEFAULTS)).toBe("");
    expect(isDefault(DEFAULTS)).toBe(true);
  });

  it("omits the default race", () => {
    expect(serialiseUrlState(state({ race: SHOWCASE_RACE }))).toBe("");
  });

  it("omits a full dsi range", () => {
    expect(serialiseUrlState(state({ dsi: [0, 100] }))).toBe("");
  });

  it("includes only what changed", () => {
    expect(serialiseUrlState(state({ driver: "LEWHAM01" }))).toBe("?driver=LEWHAM01");
  });
});

describe("parsing never throws", () => {
  const nasty = [
    "", "?", "???", "&&&", "?=", "?race", "?race=", "?%", "?%zz",
    "?dsi=abc", "?dsi=1", "?dsi=1-2-3", "?dsi=-", "?laps=x-y",
    "?state=Nonsense", "?state=", "?speaker=alien", "?sev=urgent",
    "?flags=nope", "?sort=", "?t=", "?msg=", `?q=${"x".repeat(5000)}`,
    "?race=../../etc/passwd", "?driver=<script>alert(1)</script>",
  ];

  it.each(nasty)("survives %j", (raw) => {
    expect(() => parseUrlState(raw)).not.toThrow();
  });

  it("falls back to defaults on garbage", () => {
    const s = parseUrlState("?dsi=abc&state=Nonsense&sort=");
    expect(s.dsi).toEqual(DEFAULTS.dsi);
    expect(s.state).toEqual([]);
    expect(s.sort).toBe(DEFAULTS.sort);
  });

  it("drops unknown enum values but keeps valid ones", () => {
    expect(parseUrlState("?state=Stressed,Nonsense,Calm").state)
      .toEqual(["Stressed", "Calm"]);
  });

  it("de-duplicates repeated values", () => {
    expect(parseUrlState("?speaker=driver,driver,engineer").speaker)
      .toEqual(["driver", "engineer"]);
  });
});

describe("ranges are repaired rather than rejected", () => {
  it("swaps an inverted range", () => {
    expect(parseUrlState("?dsi=90-10").dsi).toEqual([10, 90]);
  });

  it("clamps to bounds", () => {
    expect(parseUrlState("?dsi=-50-500").dsi).toEqual([0, 100]);
  });

  it("keeps a valid range", () => {
    expect(parseUrlState("?dsi=30-70").dsi).toEqual([30, 70]);
  });

  it("treats an unparseable lap range as absent", () => {
    expect(parseUrlState("?laps=nonsense").laps).toBeNull();
  });
});

describe("history mode", () => {
  it("treats navigation as navigational", () => {
    // Back should return to the previous message, not undo a keystroke.
    for (const k of ["race", "driver", "msg", "laps"]) {
      expect(NAVIGATIONAL.has(k)).toBe(true);
    }
  });

  it("treats refinements as replacements", () => {
    // Typing in a search box must not bury the previous page under thirty
    // history entries.
    for (const k of ["q", "dsi", "state", "speaker", "sev", "flags", "sort", "tables"]) {
      expect(NAVIGATIONAL.has(k)).toBe(false);
    }
  });
});

describe("activeFilters", () => {
  it("is empty for a default state", () => {
    expect(activeFilters(DEFAULTS)).toEqual([]);
  });

  it("does not list the race", () => {
    // The race is context, not a filter to clear.
    expect(activeFilters(state({ race: "2023_Monaco_Grand_Prix" }))).toEqual([]);
  });

  it("lists each narrowing filter once", () => {
    const out = activeFilters(state({
      driver: "LEWHAM01", state: ["Stressed"], dsi: [60, 100], q: "grip",
    }));
    expect(out.map((f) => f.key)).toEqual(["driver", "state", "dsi", "q"]);
  });

  it("quotes the search term so it reads as text", () => {
    expect(activeFilters(state({ q: "box box" }))[0].label).toBe('"box box"');
  });
});
