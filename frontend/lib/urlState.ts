/**
 * Filter state lives in the URL, so every view is shareable.
 *
 * Nothing in the old app was: race, driver, selected message and chart mode were
 * all local `useState`, so no view could be linked, bookmarked, or reached with
 * the back button.
 *
 * **Why not `useSearchParams`.** Under `output: export` every route is
 * prerendered, and calling it in a prerendered route forces a client-side
 * bailout up to the nearest `<Suspense>` - Next 16 errors the build outright if
 * there isn't one. It is also read-only, so writes go through
 * `router.replace()`, which is a router transition per keystroke in a search
 * box. A `useSyncExternalStore` over `location.search` sidesteps both, and it is
 * the pattern `ThemeToggle` already uses for the theme.
 *
 * Three details that will bite if changed:
 *
 *   `getSnapshot` must return a cached reference. Returning a fresh object each
 *   call makes React re-render forever.
 *
 *   `getServerSnapshot` returns the frozen defaults, because during export
 *   prerender there is no URL. Hydration then re-renders once with the real
 *   snapshot, which is invisible behind the loading skeletons.
 *
 *   push vs replace is a design decision, not a detail. Navigational changes
 *   (race, driver, message, lap range) push, so Back returns to the previous
 *   message. Refinements (search text, ranges, sorts) replace, so typing does
 *   not bury the previous page under thirty history entries.
 */

import { useCallback, useSyncExternalStore } from "react";

import {
  FLAGS, SEVERITIES, SHOWCASE_RACE, SPEAKERS, STATES,
  type Flag, type Severity, type Speaker, type State,
} from "./constants";

export interface UrlState {
  race: string;
  driver: string;          // driver_id, or "all"
  msg: string | null;      // selected message id
  laps: [number, number] | null;
  state: State[];
  dsi: [number, number];
  speaker: Speaker[];
  sev: Severity[];
  flags: Flag[];
  q: string;
  sort: string;            // "field:asc" | "field:desc"
  tables: string[];        // panel ids currently showing their table twin
  sec: string | null;      // evidence section anchor
}

export const DEFAULTS: Readonly<UrlState> = Object.freeze({
  race: SHOWCASE_RACE,
  driver: "all",
  msg: null,
  laps: null,
  state: [],
  dsi: [0, 100] as [number, number],
  speaker: [],
  sev: [],
  flags: [],
  q: "",
  sort: "lap:asc",
  tables: [],
  sec: null,
});

/** Params that push a history entry. Everything else replaces. */
export const NAVIGATIONAL = new Set(["race", "driver", "msg", "laps"]);

const KEYS = {
  race: "race", driver: "driver", msg: "msg", laps: "laps", state: "state",
  dsi: "dsi", speaker: "speaker", sev: "sev", flags: "flags", q: "q",
  sort: "sort", tables: "t", sec: "sec",
} as const;

// --- parsing ---------------------------------------------------------------

function csv<T extends string>(raw: string | null, allowed: readonly T[]): T[] {
  if (!raw) return [];
  const set = new Set(allowed as readonly string[]);
  const seen = new Set<string>();
  const out: T[] = [];
  for (const part of raw.split(",")) {
    const v = part.trim();
    // Unknown values are dropped rather than throwing: a hand-edited URL should
    // degrade, not break the page.
    if (set.has(v) && !seen.has(v)) { seen.add(v); out.push(v as T); }
  }
  return out;
}

function range(raw: string | null, lo: number, hi: number): [number, number] | null {
  if (!raw) return null;
  const m = raw.match(/^(-?\d+)-(-?\d+)$/);
  if (!m) return null;
  let a = Number(m[1]);
  let b = Number(m[2]);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  if (a > b) [a, b] = [b, a];                       // swap rather than reject
  return [Math.max(lo, a), Math.min(hi, b)];        // clamp rather than reject
}

/** Never throws. A URL someone typed by hand must degrade, not break. */
export function parseUrlState(search: string): UrlState {
  let p: URLSearchParams;
  try {
    p = new URLSearchParams(search.startsWith("?") ? search.slice(1) : search);
  } catch {
    return { ...DEFAULTS };
  }

  const dsi = range(p.get(KEYS.dsi), 0, 100) ?? DEFAULTS.dsi;
  const laps = range(p.get(KEYS.laps), 0, 1000);

  return {
    race: p.get(KEYS.race)?.trim() || DEFAULTS.race,
    driver: p.get(KEYS.driver)?.trim() || DEFAULTS.driver,
    msg: p.get(KEYS.msg)?.trim() || null,
    laps,
    state: csv(p.get(KEYS.state), STATES),
    dsi,
    speaker: csv(p.get(KEYS.speaker), SPEAKERS),
    sev: csv(p.get(KEYS.sev), SEVERITIES),
    flags: csv(p.get(KEYS.flags), FLAGS),
    q: p.get(KEYS.q) ?? DEFAULTS.q,
    sort: p.get(KEYS.sort)?.trim() || DEFAULTS.sort,
    tables: (p.get(KEYS.tables) ?? "").split(",").map((s) => s.trim()).filter(Boolean),
    sec: p.get(KEYS.sec)?.trim() || null,
  };
}

/** Params equal to their default are omitted, so a fresh URL stays clean and a
 *  shared one carries only what was actually changed. */
export function serialiseUrlState(s: UrlState): string {
  const p = new URLSearchParams();
  const put = (k: string, v: string) => { if (v) p.set(k, v); };

  if (s.race !== DEFAULTS.race) put(KEYS.race, s.race);
  if (s.driver !== DEFAULTS.driver) put(KEYS.driver, s.driver);
  if (s.msg) put(KEYS.msg, s.msg);
  if (s.laps) put(KEYS.laps, `${s.laps[0]}-${s.laps[1]}`);
  if (s.state.length) put(KEYS.state, s.state.join(","));
  if (s.dsi[0] !== DEFAULTS.dsi[0] || s.dsi[1] !== DEFAULTS.dsi[1]) {
    put(KEYS.dsi, `${s.dsi[0]}-${s.dsi[1]}`);
  }
  if (s.speaker.length) put(KEYS.speaker, s.speaker.join(","));
  if (s.sev.length) put(KEYS.sev, s.sev.join(","));
  if (s.flags.length) put(KEYS.flags, s.flags.join(","));
  if (s.q) put(KEYS.q, s.q);
  if (s.sort !== DEFAULTS.sort) put(KEYS.sort, s.sort);
  if (s.tables.length) put(KEYS.tables, s.tables.join(","));
  if (s.sec) put(KEYS.sec, s.sec);

  const out = p.toString();
  return out ? `?${out}` : "";
}

/** True when a state is entirely default - used to decide whether to show a
 *  "clear filters" affordance. */
export function isDefault(s: UrlState): boolean {
  return serialiseUrlState(s) === "";
}

/** Which filters are actually narrowing the view, for the active-filter chips. */
export function activeFilters(s: UrlState): { key: keyof UrlState; label: string }[] {
  const out: { key: keyof UrlState; label: string }[] = [];
  if (s.driver !== DEFAULTS.driver) out.push({ key: "driver", label: s.driver });
  if (s.laps) out.push({ key: "laps", label: `laps ${s.laps[0]}-${s.laps[1]}` });
  if (s.state.length) out.push({ key: "state", label: s.state.join(", ") });
  if (s.dsi[0] !== 0 || s.dsi[1] !== 100) {
    out.push({ key: "dsi", label: `DSI ${s.dsi[0]}-${s.dsi[1]}` });
  }
  if (s.speaker.length) out.push({ key: "speaker", label: s.speaker.join(", ") });
  if (s.sev.length) out.push({ key: "sev", label: s.sev.join(", ") });
  if (s.flags.length) out.push({ key: "flags", label: s.flags.join(", ") });
  if (s.q) out.push({ key: "q", label: `"${s.q}"` });
  return out;
}

// --- the store -------------------------------------------------------------

const EVENT = "pitwall:urlstate";

/** Cached so `getSnapshot` returns a stable reference. Re-parsed only when the
 *  raw search string actually changes; without this React loops forever. */
let cache: { raw: string; parsed: UrlState } | null = null;

function getSnapshot(): UrlState {
  const raw = typeof window === "undefined" ? "" : window.location.search;
  if (!cache || cache.raw !== raw) cache = { raw, parsed: parseUrlState(raw) };
  return cache.parsed;
}

function getServerSnapshot(): UrlState {
  return DEFAULTS;
}

function subscribe(onChange: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  window.addEventListener("popstate", onChange);
  window.addEventListener(EVENT, onChange);
  return () => {
    window.removeEventListener("popstate", onChange);
    window.removeEventListener(EVENT, onChange);
  };
}

function write(next: UrlState, history: "push" | "replace") {
  if (typeof window === "undefined") return;
  const search = serialiseUrlState(next);
  const url = `${window.location.pathname}${search}${window.location.hash}`;
  if (history === "push") window.history.pushState(null, "", url);
  else window.history.replaceState(null, "", url);
  // pushState/replaceState do not fire popstate, so subscribers need telling.
  window.dispatchEvent(new Event(EVENT));
}

export interface UrlStateApi {
  state: UrlState;
  /** Merge a partial update. History mode is inferred from which keys changed
   *  unless overridden. */
  patch: (next: Partial<UrlState>, opts?: { history?: "push" | "replace" }) => void;
  /** Back to defaults, keeping the race - clearing filters should not also
   *  navigate away from what you were looking at. */
  clear: () => void;
}

export function useUrlState(): UrlStateApi {
  const state = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const patch = useCallback<UrlStateApi["patch"]>((next, opts) => {
    const current = getSnapshot();
    const merged = { ...current, ...next };
    const mode = opts?.history
      ?? (Object.keys(next).some((k) => NAVIGATIONAL.has(k)) ? "push" : "replace");
    write(merged, mode);
  }, []);

  const clear = useCallback(() => {
    write({ ...DEFAULTS, race: getSnapshot().race }, "push");
  }, []);

  return { state, patch, clear };
}
