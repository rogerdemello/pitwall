/**
 * Route paths and href construction.
 *
 * There is one non-obvious rule and it is load-bearing: Hugging Face's static
 * host does not resolve a nested `index.html`, so under a static export every
 * internal link has to point at the file rather than the directory.
 * `Nav.tsx:20-23` did this inline for the three routes it knew about, which
 * meant any new route silently 404d on the deployed Space while working
 * perfectly in dev. Centralising it here is the only way that cannot happen
 * again - and `DEPLOY.md` says explicitly not to "tidy" the rewrite away.
 */

import { STATIC_MODE } from "./http";
import { DEFAULTS, serialiseUrlState, type UrlState } from "./urlState";

export const ROUTES = {
  radio: "/",
  race: "/race",
  drivers: "/drivers",
  compare: "/compare",
  evidence: "/evidence",
} as const;

export type RouteKey = keyof typeof ROUTES;

/** What the nav shows, in order, with what each one is for. */
export const NAV: { key: RouteKey; label: string; blurb: string }[] = [
  { key: "radio", label: "Radio", blurb: "One call, heard and read" },
  { key: "race", label: "Race", blurb: "What happened across a race" },
  { key: "drivers", label: "Drivers", blurb: "Who, and how they differ" },
  { key: "compare", label: "Compare", blurb: "Do races differ, and is it real" },
  { key: "evidence", label: "Evidence", blurb: "Why any of this should be believed" },
];

/**
 * Routes kept only so links that already exist keep working.
 * `/live` was the analyse screen before it moved to the root.
 */
export const LEGACY: Record<string, RouteKey> = { "/live": "radio" };

/** The path as it must appear in an href for the current build shape. */
export function path(route: RouteKey): string {
  const p = ROUTES[route];
  if (!STATIC_MODE) return p;
  // The static host needs the file, not the directory.
  return p === "/" ? "/" : `${p}/index.html`;
}

/** An href for a route, optionally carrying filter state.
 *
 * Passing a partial state builds the query from defaults, so a link only ever
 * carries what it actually sets. */
export function href(route: RouteKey, params?: Partial<UrlState>): string {
  const base = path(route);
  if (!params || Object.keys(params).length === 0) return base;
  const search = serialiseUrlState({ ...DEFAULTS, ...params } as UrlState);
  return `${base}${search}`;
}

/** Which route a pathname belongs to, for active-link marking.
 *
 * Matches on the logical route rather than the href, because the href differs
 * between build shapes and `/race/index.html` is still `/race`. */
export function activeRoute(pathname: string): RouteKey | null {
  const clean = pathname.replace(/\/index\.html$/, "").replace(/\/+$/, "") || "/";
  if (clean in LEGACY) return LEGACY[clean];
  const hit = (Object.keys(ROUTES) as RouteKey[])
    .filter((k) => ROUTES[k] !== "/")
    .find((k) => clean === ROUTES[k] || clean.startsWith(`${ROUTES[k]}/`));
  return hit ?? (clean === "/" ? "radio" : null);
}
