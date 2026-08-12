/**
 * Number and text formatting, with the two known render bugs fixed at source.
 *
 * `evidence/page.tsx:134` divided by `suppressed_stress_eligible` without
 * guarding zero, so a race with no eligible messages rendered "NaN%".
 *
 * `RaceTimeline.tsx:43-46` called `Math.min(...[])`, which is `Infinity`, so an
 * empty lap array produced `M NaN,NaN` path data and an invisible chart. The
 * parent guarded `onLap.length` but not `lapTimes.length`.
 *
 * Both are the same mistake: a value that is *absent* being formatted as though
 * it were a number. So the rule here is that absent formats as an em dash and
 * never as a digit.
 */

/** What every formatter renders when there is nothing to render. */
export const EMPTY = "—";

/** A percentage, or EMPTY when the denominator makes it meaningless.
 *  This is the exact fix for the NaN% bug. */
export function pct(numerator: number | null | undefined,
                    denominator: number | null | undefined,
                    digits = 0): string {
  if (numerator == null || denominator == null) return EMPTY;
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator)) return EMPTY;
  if (denominator === 0) return EMPTY;
  return `${((numerator / denominator) * 100).toFixed(digits)}%`;
}

/** A already-fractional value as a percentage. */
export function pctOf(value: number | null | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return EMPTY;
  return `${(value * 100).toFixed(digits)}%`;
}

/** Minimum of a possibly-empty list. `null`, never `Infinity`.
 *  This is the exact fix for the NaN chart-path bug. */
export function safeMin(values: readonly number[]): number | null {
  const finite = values.filter((v) => Number.isFinite(v));
  return finite.length ? finite.reduce((a, b) => (b < a ? b : a)) : null;
}

/** Maximum of a possibly-empty list. `null`, never `-Infinity`.
 *  Uses a reduce rather than `Math.max(...xs)` so a long array cannot blow the
 *  argument-list limit. */
export function safeMax(values: readonly number[]): number | null {
  const finite = values.filter((v) => Number.isFinite(v));
  return finite.length ? finite.reduce((a, b) => (b > a ? b : a)) : null;
}

/** A number to fixed digits, or EMPTY. */
export function num(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return EMPTY;
  return value.toFixed(digits);
}

/** An integer, or EMPTY. */
export function int(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return EMPTY;
  return String(Math.round(value));
}

/** A signed value, so a delta always shows its direction. */
export function signed(value: number | null | undefined, digits = 2,
                       unit = ""): string {
  if (value == null || !Number.isFinite(value)) return EMPTY;
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}${unit}`;
}

/** Lap time as m:ss.mmm, which is how a pit wall reads it. */
export function lapTime(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return EMPTY;
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return m > 0 ? `${m}:${s.toFixed(3).padStart(6, "0")}` : s.toFixed(3);
}

/** Clip duration, in the units a human says out loud. */
export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return EMPTY;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  return `${m}m ${Math.round(seconds - m * 60)}s`;
}

/** UTC clock time from an ISO timestamp. The corpus spans several years, so the
 *  date is rarely the interesting part; the time of day within the race is. */
export function clock(iso: string | null | undefined): string {
  if (!iso) return EMPTY;
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? EMPTY
    : d.toISOString().slice(11, 19);
}

/** A correlation with its sample size, because one without the other invites
 *  reading r=0.043 on n=1155 and r=0.62 on n=10 as comparable claims. */
export function correlation(r: number | null | undefined,
                            n: number | null | undefined): string {
  if (r == null || !Number.isFinite(r)) return EMPTY;
  const base = `r = ${r.toFixed(3)}`;
  return n != null && Number.isFinite(n) ? `${base} (n = ${n})` : base;
}

/** A p-value, floored rather than rounded to zero - "p = 0.000" reads as
 *  certainty and "p < 0.001" reads as a bound. */
export function pValue(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return EMPTY;
  if (p < 0.001) return "p < 0.001";
  return `p = ${p.toFixed(3)}`;
}

/** Truncate for a table cell without cutting mid-word where avoidable. */
export function truncate(text: string | null | undefined, max = 80): string {
  if (!text) return EMPTY;
  if (text.length <= max) return text;
  const cut = text.slice(0, max);
  const space = cut.lastIndexOf(" ");
  return `${(space > max * 0.6 ? cut.slice(0, space) : cut).trimEnd()}…`;
}

/** "1,155" rather than "1155" for anything a person will read aloud. */
export function count(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return EMPTY;
  return Math.round(value).toLocaleString("en-GB");
}
