/**
 * Reaching the model backend from a build that has no backend of its own.
 *
 * The deployed app is a `sdk: static` Space, so there is no process to run a
 * model in. `space_live/` is a second Space — Gradio on ZeroGPU — that imports
 * the same `pipeline/` package and answers with the same fields as
 * `POST /api/analyze`. This module is the client for it.
 *
 * Gradio's REST protocol is three hops, not one, and the reason is the GPU
 * queue: the call returns an id immediately and the result arrives later over
 * a stream, so a request never sits on an open socket waiting for a slot.
 *
 *   1. POST /gradio_api/upload            multipart -> ["/tmp/gradio/../clip.webm"]
 *   2. POST /gradio_api/call/analyze      {data:[FileData]} -> {event_id}
 *   3. GET  /gradio_api/call/analyze/<id> server-sent events -> the result
 *
 * `fetchImpl` is injectable for the same reason the pipeline is imported rather
 * than reimplemented: the three-hop sequence is the part that can silently
 * break, so it is the part that gets tested.
 */

/** The upload and enqueue hops are fast; only the third waits on a GPU slice.
 *  A cold ZeroGPU Space has to wake and load three models before it answers,
 *  which is slow but not unbounded — and a stream that never closes looks
 *  exactly like one that is merely slow, so it needs a ceiling. */
export const LIVE_TIMEOUT_MS = 180_000;

export class LiveSpaceError extends Error {
  constructor(
    message: string,
    readonly stage: "upload" | "enqueue" | "stream" | "result",
  ) {
    super(message);
    this.name = "LiveSpaceError";
  }
}

interface Options {
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
  signal?: AbortSignal;
}

/** Pull the completed payload out of a Gradio event stream.
 *
 *  Exported because the parsing is the fiddly half: the stream carries
 *  heartbeats and progress events too, and taking the first `data:` line
 *  rather than the one belonging to `event: complete` is the obvious way to
 *  get a plausible wrong answer. */
export function parseEventStream(text: string): unknown {
  let event = "";
  let completed: string | null = null;
  let errored: string | null = null;

  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trimEnd();
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
      continue;
    }
    if (!line.startsWith("data:")) continue;
    const payload = line.slice("data:".length).trim();
    if (event === "complete") completed = payload;
    else if (event === "error") errored = payload;
  }

  if (completed === null && errored !== null) {
    throw new LiveSpaceError(
      `The Space reported an error: ${errored || "no detail given"}`,
      "result",
    );
  }
  if (completed === null) {
    throw new LiveSpaceError(
      "The Space closed the stream without returning a result.",
      "stream",
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(completed);
  } catch {
    throw new LiveSpaceError("The Space returned a result that was not JSON.", "result");
  }
  // gr.Interface has one output, so the payload is a one-element array.
  return Array.isArray(parsed) ? parsed[0] : parsed;
}

/** Run one clip through the ZeroGPU Space. Resolves to the analysis object,
 *  which carries the same fields as `POST /api/analyze`. */
export async function analyzeOnSpace(
  base: string,
  file: File,
  opts: Options = {},
): Promise<unknown> {
  const f = opts.fetchImpl ?? fetch;
  const root = base.replace(/\/+$/, "");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), opts.timeoutMs ?? LIVE_TIMEOUT_MS);
  const onAbort = () => controller.abort();
  opts.signal?.addEventListener("abort", onAbort);
  const sig = controller.signal;

  try {
    // 1. Upload. Gradio wants the field named `files`, and answers with the
    //    paths it stored them at.
    const form = new FormData();
    form.append("files", file);
    const up = await f(`${root}/gradio_api/upload`, {
      method: "POST",
      body: form,
      signal: sig,
    });
    if (!up.ok) {
      throw new LiveSpaceError(`Could not upload the clip (${up.status}).`, "upload");
    }
    const paths = (await up.json()) as unknown;
    if (!Array.isArray(paths) || paths.length === 0 || typeof paths[0] !== "string") {
      throw new LiveSpaceError(
        "The Space accepted the upload but returned no file path.",
        "upload",
      );
    }

    // 2. Enqueue. `analyze` is the api_name in space_live/app.py.
    const call = await f(`${root}/gradio_api/call/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data: [{ path: paths[0], meta: { _type: "gradio.FileData" } }],
      }),
      signal: sig,
    });
    if (!call.ok) {
      throw new LiveSpaceError(`The Space refused the request (${call.status}).`, "enqueue");
    }
    const queued = (await call.json()) as { event_id?: string };
    if (!queued?.event_id) {
      throw new LiveSpaceError("The Space did not return an event id.", "enqueue");
    }

    // 3. Wait for the result.
    const stream = await f(`${root}/gradio_api/call/analyze/${queued.event_id}`, {
      signal: sig,
    });
    if (!stream.ok) {
      throw new LiveSpaceError(
        `Lost the result stream (${stream.status}).`,
        "stream",
      );
    }
    const result = parseEventStream(await stream.text());

    // The Space describes failures in the payload rather than raising, because
    // a Gradio exception reaches a cross-origin caller as an opaque 500.
    if (result && typeof result === "object" && "error" in result) {
      throw new LiveSpaceError(String((result as { error: unknown }).error), "result");
    }
    return result;
  } catch (e) {
    if (e instanceof LiveSpaceError) throw e;
    if (sig.aborted) {
      throw new LiveSpaceError(
        "The Space did not answer in time. A cold ZeroGPU Space has to wake and "
          + "load three models on the first call — try once more.",
        "stream",
      );
    }
    throw new LiveSpaceError(
      e instanceof Error ? e.message : String(e),
      "stream",
    );
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener("abort", onAbort);
  }
}
