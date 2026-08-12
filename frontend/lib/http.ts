/**
 * The one place that knows how to reach the backend.
 *
 * The static/live switch is a single uniform rewrite rather than a lookup
 * table, and that uniformity is the best thing about the original data layer:
 * `build_static_site.py` freezes every GET endpoint under the same shape, so
 * `/api/<anything>` becomes `/data/<anything>.json` and adding an endpoint
 * needs no change here. A table would drift the moment someone forgot to
 * update it.
 */

/**
 * Empty by default, so the browser uses same-origin `/api` paths.
 *
 * This used to default to `http://localhost:8000`, which meant the `rewrites()`
 * proxy in next.config.ts was dead code in dev - requests went cross-origin
 * straight to the backend and depended on its CORS allowlist. Defaulting to ""
 * makes dev, Docker and the export all take the same path and removes CORS from
 * the critical path entirely.
 */
export const API = process.env.NEXT_PUBLIC_API ?? "";

/**
 * Static mode: the deployed Space is `sdk: static`, so there is no server and
 * every GET reads a frozen JSON file instead.
 */
export const STATIC_MODE = process.env.NEXT_PUBLIC_STATIC === "1";

/**
 * The ZeroGPU Gradio Space that runs the models for Live Analysis.
 *
 * Set, the analyse panel posts here cross-origin - Gradio accepts any origin
 * when the host is not a localhost alias, which is the case on *.hf.space, so
 * no proxy is involved. Unset, the panel falls back to same-origin `/api` (dev
 * and Docker) or explains itself (static with no live Space).
 */
export const LIVE_SPACE = process.env.NEXT_PUBLIC_LIVE_SPACE ?? "";

/** Can this build actually analyse an uploaded clip? */
export const CAN_ANALYSE = !STATIC_MODE || Boolean(LIVE_SPACE);

export function resolve(path: string): string {
  if (!STATIC_MODE) return `${API}${path}`;
  return `/data${path.slice("/api".length)}.json`;
}

/** An error that says what failed and where, so the UI can render a sentence
 *  rather than `String(e)`. The old pages printed raw exception text at the
 *  user - "Error: TypeError: Failed to fetch" is not a message. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly path: string,
    message?: string,
  ) {
    super(message ?? `${status} on ${path}`);
    this.name = "ApiError";
  }

  /** What to show a person. */
  get human(): string {
    if (this.status === 0) {
      return STATIC_MODE
        ? "That data file is missing from this build."
        : "Could not reach the backend. Is it running?";
    }
    if (this.status === 404) return "That data has not been built yet.";
    if (this.status === 413) return "That clip is too large.";
    if (this.status === 503) return "The analyser is busy. Try again in a moment.";
    if (this.status === 504) return "The analysis took too long and was stopped.";
    if (this.status >= 500) return "The backend failed while handling that.";
    return `Request failed (${this.status}).`;
  }
}

/** Requests that hang forever look identical to requests that are slow. A cold
 *  ZeroGPU Space can take a while to wake, but not this long. */
export const TIMEOUT_MS = 15_000;

export async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const url = resolve(path);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort);

  try {
    const res = await fetch(url, { cache: "no-store", signal: controller.signal });
    if (!res.ok) throw new ApiError(res.status, path);
    return (await res.json()) as T;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    // A network failure, a CORS rejection and a timeout are indistinguishable
    // from here; status 0 means "never got a response".
    throw new ApiError(0, path, e instanceof Error ? e.message : String(e));
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

/** Audio for a message. Static builds ship only the showcase race's clips -
 *  all twelve would be 327 MB and the evidence screens need none of them. */
export function audioUrl(raceId: string, file: string): string {
  const enc = encodeURIComponent(file);
  return STATIC_MODE ? `/audio/${raceId}/${enc}` : `${API}/api/audio/${raceId}/${enc}`;
}
