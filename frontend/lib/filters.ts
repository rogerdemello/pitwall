/**
 * Filtering and sorting messages. Pure, so it is testable and so the dashboard
 * has one definition of what a filter means rather than one per view.
 *
 * Performance is not a concern at this size and pretending otherwise would be
 * over-engineering: a race is 170-250 messages, and six predicates plus a
 * substring test over a precomputed lowercase field is sub-millisecond. What
 * *is* a concern is rendering, which is why the table caps its rows.
 */

import { MAX_TABLE_ROWS } from "./constants";
import type { UrlState } from "./urlState";

/** The subset of a message this module needs. Kept structural so it works with
 *  both the full `Message` and anything narrower a view hands it. */
export interface Filterable {
  id: string;
  driver_id: string;
  driver_code?: string;
  transcript: string;
  dsi: number;
  state: string;
  speaker: string;
  suppressed_stress: boolean;
  lap?: { lap_number: number | null; in_race: boolean } | null;
  recommendation?: { severity: string } | null;
  /** Precomputed lowercase haystack. Built once per race in the query layer,
   *  because lowercasing 2,000 transcripts on every keystroke is the one thing
   *  here that would actually be slow. */
  _search?: string;
}

/** The haystack a search matches against. */
export function searchIndex(m: Filterable): string {
  return [m.transcript, m.driver_code ?? "", m.driver_id, m.state]
    .join(" ")
    .toLowerCase();
}

/** Attach `_search` once, at load. */
export function withSearchIndex<T extends Filterable>(messages: T[]): T[] {
  return messages.map((m) => (m._search ? m : { ...m, _search: searchIndex(m) }));
}

function matchesLaps(m: Filterable, laps: UrlState["laps"]): boolean {
  if (!laps) return true;
  const n = m.lap?.lap_number;
  // A message with no lap cannot satisfy a lap range. Treating "unknown" as
  // "included" would quietly widen every brushed selection.
  if (n == null) return false;
  return n >= laps[0] && n <= laps[1];
}

export function filterMessages<T extends Filterable>(
  messages: readonly T[],
  f: UrlState,
): T[] {
  const q = f.q.trim().toLowerCase();
  // Widened to string on purpose. The URL values are already validated against
  // the vocabularies by `parseUrlState`; the message values arrive from the API
  // as plain strings, and a message carrying a state we do not recognise should
  // fail to match rather than fail to compile.
  const wantState: ReadonlySet<string> = new Set<string>(f.state);
  const wantSpeaker: ReadonlySet<string> = new Set<string>(f.speaker);
  const wantSev: ReadonlySet<string> = new Set<string>(f.sev);
  const flags: ReadonlySet<string> = new Set<string>(f.flags);

  return messages.filter((m) => {
    if (f.driver !== "all" && m.driver_id !== f.driver) return false;
    if (wantState.size && !wantState.has(m.state)) return false;
    if (wantSpeaker.size && !wantSpeaker.has(m.speaker)) return false;
    if (m.dsi < f.dsi[0] || m.dsi > f.dsi[1]) return false;
    if (!matchesLaps(m, f.laps)) return false;

    if (wantSev.size) {
      const sev = m.recommendation?.severity;
      if (!sev || !wantSev.has(sev)) return false;
    }
    if (flags.has("suppressed") && !m.suppressed_stress) return false;
    if (flags.has("onlap") && !m.lap?.in_race) return false;
    if (flags.has("hasrec") && !m.recommendation) return false;

    if (q && !(m._search ?? searchIndex(m)).includes(q)) return false;
    return true;
  });
}

type Getter = (m: Filterable) => number | string | null;

/** Sortable columns. A key not in here is ignored rather than throwing, so a
 *  hand-edited `?sort=` cannot break the table. */
export const SORT_KEYS: Record<string, Getter> = {
  lap: (m) => m.lap?.lap_number ?? null,
  dsi: (m) => m.dsi,
  driver: (m) => m.driver_code ?? m.driver_id,
  state: (m) => m.state,
  speaker: (m) => m.speaker,
  transcript: (m) => m.transcript,
};

export function parseSort(sort: string): { key: string; dir: 1 | -1 } {
  const [key, dir] = sort.split(":");
  return {
    key: key in SORT_KEYS ? key : "lap",
    dir: dir === "desc" ? -1 : 1,
  };
}

export function sortMessages<T extends Filterable>(
  messages: readonly T[],
  sort: string,
): T[] {
  const { key, dir } = parseSort(sort);
  const get = SORT_KEYS[key];
  // Copy: sorting the array a query cache handed us would mutate shared state.
  return [...messages].sort((a, b) => {
    const va = get(a);
    const vb = get(b);
    // Nulls sort last regardless of direction. A message with no lap is not
    // "before lap 1", it is unplaced, and flipping the sort should not
    // parade it to the top.
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "string" || typeof vb === "string") {
      return String(va).localeCompare(String(vb)) * dir;
    }
    return (va - vb) * dir;
  });
}

export interface FilterResult<T> {
  /** Everything that matched. */
  all: T[];
  /** What the table should render. */
  rows: T[];
  total: number;
  shown: number;
  truncated: boolean;
}

/** Filter, sort, and cap. The cap is explicit in the return value so the UI can
 *  say "showing 200 of 1,847" rather than silently lying about the total. */
export function applyFilters<T extends Filterable>(
  messages: readonly T[],
  f: UrlState,
  limit = MAX_TABLE_ROWS,
): FilterResult<T> {
  const all = sortMessages(filterMessages(messages, f), f.sort);
  const rows = all.slice(0, limit);
  return {
    all,
    rows,
    total: all.length,
    shown: rows.length,
    truncated: all.length > rows.length,
  };
}

/** Counts per facet, computed against everything *except* that facet's own
 *  filter - so a count never reads zero for an option you could still pick.
 *  This is what stops a filter UI from looking broken. */
export function facetCounts<T extends Filterable>(
  messages: readonly T[],
  f: UrlState,
  facet: "state" | "speaker",
): Record<string, number> {
  const without: UrlState = { ...f, [facet]: [] };
  const out: Record<string, number> = {};
  for (const m of filterMessages(messages, without)) {
    const key = facet === "state" ? m.state : m.speaker;
    out[key] = (out[key] ?? 0) + 1;
  }
  return out;
}
