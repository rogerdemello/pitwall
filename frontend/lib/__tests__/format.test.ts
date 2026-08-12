/**
 * Formatting, and the two render bugs it exists to fix.
 *
 * Both shipped, both are the same mistake - an absent value formatted as though
 * it were a number - and both had the symptom of looking like data rather than
 * looking broken.
 */

import { describe, expect, it } from "vitest";

import {
  EMPTY, clock, correlation, count, duration, int, lapTime, num, pValue, pct,
  pctOf, safeMax, safeMin, signed, truncate,
} from "../format";

describe("pct", () => {
  it("renders a percentage", () => {
    expect(pct(1, 4)).toBe("25%");
  });

  it("returns EMPTY on a zero denominator rather than NaN%", () => {
    // evidence/page.tsx:134 shipped this: a race with no eligible messages
    // rendered "NaN%" where a number belonged.
    expect(pct(0, 0)).toBe(EMPTY);
    expect(pct(5, 0)).toBe(EMPTY);
  });

  it("returns EMPTY for null or undefined", () => {
    expect(pct(null, 10)).toBe(EMPTY);
    expect(pct(10, undefined)).toBe(EMPTY);
  });

  it("returns EMPTY for non-finite input", () => {
    expect(pct(Infinity, 10)).toBe(EMPTY);
    expect(pct(NaN, 10)).toBe(EMPTY);
  });

  it("honours digits", () => {
    expect(pct(1, 3, 1)).toBe("33.3%");
  });
});

describe("pctOf", () => {
  it("scales a fraction", () => {
    expect(pctOf(0.525, 1)).toBe("52.5%");
  });

  it("returns EMPTY for absent values", () => {
    expect(pctOf(null)).toBe(EMPTY);
    expect(pctOf(NaN)).toBe(EMPTY);
  });
});

describe("safeMin / safeMax", () => {
  it("return null on an empty list, never Infinity", () => {
    // RaceTimeline.tsx:43-46 shipped Math.min(...[]) === Infinity, which made
    // every path coordinate NaN and the chart invisible.
    expect(safeMin([])).toBeNull();
    expect(safeMax([])).toBeNull();
  });

  it("find the extremes", () => {
    expect(safeMin([3, 1, 2])).toBe(1);
    expect(safeMax([3, 1, 2])).toBe(3);
  });

  it("ignore non-finite entries", () => {
    expect(safeMin([NaN, 5, Infinity])).toBe(5);
    expect(safeMax([NaN, 5, -Infinity])).toBe(5);
  });

  it("return null when everything is non-finite", () => {
    expect(safeMin([NaN, Infinity])).toBeNull();
  });

  it("handle a long array without blowing the argument limit", () => {
    // Math.max(...xs) throws on very large arrays; the reduce does not.
    const big = Array.from({ length: 200_000 }, (_, i) => i);
    expect(safeMax(big)).toBe(199_999);
  });
});

describe("num / int / signed", () => {
  it("format finite numbers", () => {
    expect(num(1.234, 2)).toBe("1.23");
    expect(int(1.6)).toBe("2");
  });

  it("return EMPTY for absent values", () => {
    for (const f of [num, int]) {
      expect(f(null)).toBe(EMPTY);
      expect(f(undefined)).toBe(EMPTY);
      expect(f(NaN)).toBe(EMPTY);
    }
  });

  it("always shows the direction of a delta", () => {
    expect(signed(0.35, 2, "s")).toBe("+0.35s");
    expect(signed(-0.07, 2, "s")).toBe("-0.07s");
    expect(signed(0, 2)).toBe("+0.00");
  });
});

describe("lapTime", () => {
  it("renders m:ss.mmm above a minute", () => {
    expect(lapTime(83.456)).toBe("1:23.456");
  });

  it("renders bare seconds below a minute", () => {
    expect(lapTime(45.5)).toBe("45.500");
  });

  it("pads the seconds so 1:05 does not read as 1:5", () => {
    expect(lapTime(65.25)).toBe("1:05.250");
  });

  it("rejects absent and negative values", () => {
    expect(lapTime(null)).toBe(EMPTY);
    expect(lapTime(-1)).toBe(EMPTY);
  });
});

describe("duration", () => {
  it("uses seconds under a minute", () => {
    expect(duration(10.44)).toBe("10.4s");
  });

  it("uses minutes above one", () => {
    expect(duration(105.4)).toBe("1m 45s");
  });

  it("returns EMPTY for absent values", () => {
    expect(duration(undefined)).toBe(EMPTY);
  });
});

describe("clock", () => {
  it("renders UTC time of day", () => {
    expect(clock("2021-12-12T13:55:03.000Z")).toBe("13:55:03");
  });

  it("returns EMPTY for an unparseable timestamp", () => {
    expect(clock("not a date")).toBe(EMPTY);
    expect(clock(null)).toBe(EMPTY);
  });
});

describe("correlation", () => {
  it("carries the sample size", () => {
    // r=0.043 on n=1155 and r=0.62 on n=10 are not comparable claims, and the
    // n is what makes that visible.
    expect(correlation(0.0428, 1155)).toBe("r = 0.043 (n = 1155)");
  });

  it("omits n when absent", () => {
    expect(correlation(0.5, null)).toBe("r = 0.500");
  });

  it("returns EMPTY without an r", () => {
    expect(correlation(null, 100)).toBe(EMPTY);
  });
});

describe("pValue", () => {
  it("bounds tiny values rather than rounding them to zero", () => {
    // "p = 0.000" reads as certainty; "p < 0.001" reads as a bound.
    expect(pValue(0.0000001)).toBe("p < 0.001");
  });

  it("renders ordinary values", () => {
    expect(pValue(0.7376)).toBe("p = 0.738");
  });

  it("returns EMPTY for absent values", () => {
    expect(pValue(null)).toBe(EMPTY);
  });
});

describe("truncate", () => {
  it("leaves short text alone", () => {
    expect(truncate("box box box", 80)).toBe("box box box");
  });

  it("cuts on a word boundary when there is a sensible one", () => {
    const out = truncate("the quick brown fox jumps over the lazy dog", 20);
    expect(out.endsWith("…")).toBe(true);
    expect(out.length).toBeLessThanOrEqual(21);
    expect(out).not.toContain("jum…");
  });

  it("returns EMPTY for absent text", () => {
    expect(truncate(null)).toBe(EMPTY);
    expect(truncate("")).toBe(EMPTY);
  });
});

describe("count", () => {
  it("groups thousands", () => {
    expect(count(2042)).toBe("2,042");
  });

  it("returns EMPTY for absent values", () => {
    expect(count(null)).toBe(EMPTY);
  });
});
