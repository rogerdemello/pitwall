/**
 * Filtering and sorting: each predicate alone, then combined.
 *
 * The cases that matter are the ones where a reasonable implementation quietly
 * does the wrong thing - a message with no lap satisfying a lap range, a null
 * parading to the top when the sort flips, a facet count reading zero for an
 * option you could still pick.
 */

import { describe, expect, it } from "vitest";

import { applyFilters, facetCounts, filterMessages, parseSort, sortMessages, withSearchIndex, type Filterable } from "../filters";
import { DEFAULTS, type UrlState } from "../urlState";

function msg(over: Partial<Filterable> & { id: string }): Filterable {
  return {
    driver_id: "LEWHAM01", driver_code: "HAM",
    transcript: "box box box", dsi: 50, state: "Calm", speaker: "driver",
    suppressed_stress: false,
    lap: { lap_number: 10, in_race: true },
    recommendation: null,
    ...over,
  };
}

const CORPUS = withSearchIndex([
  msg({ id: "a", dsi: 20, state: "Calm", speaker: "driver", transcript: "all good mate", lap: { lap_number: 1, in_race: true } }),
  msg({ id: "b", dsi: 55, state: "Energised", speaker: "engineer", transcript: "box box box", lap: { lap_number: 5, in_race: true } }),
  msg({ id: "c", dsi: 80, state: "Stressed", speaker: "driver", transcript: "I have no grip", lap: { lap_number: 20, in_race: true }, suppressed_stress: true, recommendation: { severity: "act" }, driver_id: "MAXVER01", driver_code: "VER" }),
  msg({ id: "d", dsi: 65, state: "Fatigued", speaker: "unknown", transcript: "yeah copy", lap: { lap_number: null, in_race: false } }),
  msg({ id: "e", dsi: 45, state: "Calm", speaker: "driver", transcript: "understood", lap: { lap_number: 30, in_race: true }, recommendation: { severity: "watch" } }),
]);

const f = (over: Partial<UrlState> = {}): UrlState => ({ ...DEFAULTS, ...over });
const ids = (ms: Filterable[]) => ms.map((m) => m.id);

describe("individual predicates", () => {
  it("passes everything by default", () => {
    expect(filterMessages(CORPUS, f())).toHaveLength(5);
  });

  it("filters by driver", () => {
    expect(ids(filterMessages(CORPUS, f({ driver: "MAXVER01" })))).toEqual(["c"]);
  });

  it("filters by state", () => {
    expect(ids(filterMessages(CORPUS, f({ state: ["Calm"] })))).toEqual(["a", "e"]);
  });

  it("treats multiple states as OR", () => {
    expect(ids(filterMessages(CORPUS, f({ state: ["Calm", "Stressed"] }))))
      .toEqual(["a", "c", "e"]);
  });

  it("filters by speaker", () => {
    expect(ids(filterMessages(CORPUS, f({ speaker: ["engineer"] })))).toEqual(["b"]);
  });

  it("filters by dsi range inclusively", () => {
    expect(ids(filterMessages(CORPUS, f({ dsi: [45, 65] })))).toEqual(["b", "d", "e"]);
  });

  it("filters by severity", () => {
    expect(ids(filterMessages(CORPUS, f({ sev: ["act"] })))).toEqual(["c"]);
  });

  it("filters by suppressed-stress flag", () => {
    expect(ids(filterMessages(CORPUS, f({ flags: ["suppressed"] })))).toEqual(["c"]);
  });

  it("filters by has-recommendation flag", () => {
    expect(ids(filterMessages(CORPUS, f({ flags: ["hasrec"] })))).toEqual(["c", "e"]);
  });

  it("filters by on-lap flag", () => {
    expect(ids(filterMessages(CORPUS, f({ flags: ["onlap"] })))).toEqual(["a", "b", "c", "e"]);
  });
});

describe("lap range", () => {
  it("includes only laps inside it", () => {
    expect(ids(filterMessages(CORPUS, f({ laps: [1, 10] })))).toEqual(["a", "b"]);
  });

  it("excludes a message with no lap", () => {
    // "unknown lap" is not "inside the range". Including it would quietly
    // widen every brushed selection on the chart.
    expect(ids(filterMessages(CORPUS, f({ laps: [1, 100] })))).not.toContain("d");
  });

  it("is inclusive at both ends", () => {
    expect(ids(filterMessages(CORPUS, f({ laps: [5, 20] })))).toEqual(["b", "c"]);
  });
});

describe("search", () => {
  it("matches transcript text", () => {
    expect(ids(filterMessages(CORPUS, f({ q: "grip" })))).toEqual(["c"]);
  });

  it("is case insensitive", () => {
    expect(ids(filterMessages(CORPUS, f({ q: "BOX" })))).toEqual(["b"]);
  });

  it("matches driver code", () => {
    expect(ids(filterMessages(CORPUS, f({ q: "ver" })))).toEqual(["c"]);
  });

  it("ignores surrounding whitespace", () => {
    expect(ids(filterMessages(CORPUS, f({ q: "  grip  " })))).toEqual(["c"]);
  });

  it("returns an empty array, not undefined, when nothing matches", () => {
    const out = filterMessages(CORPUS, f({ q: "zzzz" }));
    expect(out).toEqual([]);
  });

  it("works without a precomputed index", () => {
    const raw = [msg({ id: "x", transcript: "understeer in turn 3" })];
    expect(ids(filterMessages(raw, f({ q: "understeer" })))).toEqual(["x"]);
  });
});

describe("combined filters are AND", () => {
  it("narrows across facets", () => {
    expect(ids(filterMessages(CORPUS, f({ speaker: ["driver"], dsi: [40, 100] }))))
      .toEqual(["c", "e"]);
  });

  it("can produce nothing", () => {
    expect(filterMessages(CORPUS, f({ state: ["Calm"], dsi: [90, 100] }))).toEqual([]);
  });
});

describe("sorting", () => {
  it("defaults to lap ascending", () => {
    expect(parseSort("lap:asc")).toEqual({ key: "lap", dir: 1 });
  });

  it("falls back on an unknown key rather than throwing", () => {
    expect(parseSort("vibes:desc").key).toBe("lap");
  });

  it("sorts numerically by dsi", () => {
    expect(ids(sortMessages(CORPUS, "dsi:desc")).slice(0, 2)).toEqual(["c", "d"]);
  });

  it("sorts strings by locale", () => {
    expect(ids(sortMessages(CORPUS, "state:asc"))[0]).toBe("a");
  });

  it("puts nulls last regardless of direction", () => {
    // Message "d" has no lap. Flipping the sort should not parade it to the
    // top - it is unplaced, not "before lap 1".
    expect(ids(sortMessages(CORPUS, "lap:asc")).at(-1)).toBe("d");
    expect(ids(sortMessages(CORPUS, "lap:desc")).at(-1)).toBe("d");
  });

  it("does not mutate the input", () => {
    const before = ids([...CORPUS]);
    sortMessages(CORPUS, "dsi:desc");
    expect(ids([...CORPUS])).toEqual(before);
  });
});

describe("applyFilters", () => {
  it("reports the true total even when capped", () => {
    const many = withSearchIndex(
      Array.from({ length: 500 }, (_, i) => msg({ id: `m${i}` })),
    );
    const out = applyFilters(many, f(), 200);
    expect(out.total).toBe(500);
    expect(out.shown).toBe(200);
    expect(out.truncated).toBe(true);
  });

  it("is not truncated when everything fits", () => {
    const out = applyFilters(CORPUS, f(), 200);
    expect(out.truncated).toBe(false);
    expect(out.shown).toBe(out.total);
  });

  it("applies the sort before the cap", () => {
    const many = withSearchIndex(
      Array.from({ length: 10 }, (_, i) => msg({ id: `m${i}`, dsi: i * 10 })),
    );
    expect(applyFilters(many, f({ sort: "dsi:desc" }), 3).rows[0].dsi).toBe(90);
  });
});

describe("facetCounts", () => {
  it("counts each option", () => {
    expect(facetCounts(CORPUS, f(), "state")).toEqual({
      Calm: 2, Energised: 1, Stressed: 1, Fatigued: 1,
    });
  });

  it("ignores that facet's own filter", () => {
    // With Calm selected, the other states must still show their counts -
    // otherwise every unselected option reads zero and the UI looks broken.
    const counts = facetCounts(CORPUS, f({ state: ["Calm"] }), "state");
    expect(counts.Stressed).toBe(1);
    expect(counts.Calm).toBe(2);
  });

  it("still respects other facets", () => {
    const counts = facetCounts(CORPUS, f({ speaker: ["driver"] }), "state");
    expect(counts.Energised).toBeUndefined();
    expect(counts.Calm).toBe(2);
  });
});
