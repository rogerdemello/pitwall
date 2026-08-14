import { describe, expect, it, vi } from "vitest";

import {
  analyzeOnSpace,
  LiveSpaceError,
  parseEventStream,
} from "../liveSpace";

/* The three-hop Gradio sequence is the part of this project that can break
 * without anything failing loudly: a wrong field name on the upload, or reading
 * the first `data:` line instead of the completed one, both produce a plausible
 * wrong answer rather than an error. So the sequence itself is what is asserted
 * here - the URLs hit, in order, with the bodies Gradio expects. */

const OK = (body: unknown) =>
  ({ ok: true, status: 200, json: async () => body }) as unknown as Response;

const STREAM = (text: string) =>
  ({ ok: true, status: 200, text: async () => text }) as unknown as Response;

const RESULT = { transcript: "box box", state: { dsi: 71 } };

const COMPLETE = `event: complete\ndata: ${JSON.stringify([RESULT])}\n\n`;

/** A Space that answers all three hops correctly. */
function happyFetch(stream = COMPLETE) {
  return vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    void init;
    const u = String(url);
    if (u.endsWith("/gradio_api/upload")) return OK(["/tmp/gradio/abc/clip.webm"]);
    if (u.endsWith("/gradio_api/call/analyze")) return OK({ event_id: "ev-1" });
    if (u.includes("/gradio_api/call/analyze/ev-1")) return STREAM(stream);
    throw new Error(`unexpected url ${u}`);
  });
}

const clip = () => new File([new Uint8Array([1, 2, 3])], "clip.webm", { type: "audio/webm" });

/** The rejection, typed - and a failure if the call unexpectedly succeeds. */
async function rejection(p: Promise<unknown>): Promise<LiveSpaceError> {
  return p.then(
    () => {
      throw new Error("expected the call to reject, but it resolved");
    },
    (e: unknown) => e as LiveSpaceError,
  );
}

describe("parseEventStream", () => {
  it("returns the payload of the completed event", () => {
    expect(parseEventStream(COMPLETE)).toEqual(RESULT);
  });

  it("ignores heartbeats and progress events before the result", () => {
    const noisy =
      `event: heartbeat\ndata: null\n\n` +
      `event: generating\ndata: [{"transcript":"partial"}]\n\n` +
      COMPLETE;
    // Taking the first data: line would return the partial and look fine.
    expect(parseEventStream(noisy)).toEqual(RESULT);
  });

  it("raises the Space's error event rather than reporting no result", () => {
    const stream = `event: error\ndata: "GPU quota exhausted"\n\n`;
    expect(() => parseEventStream(stream)).toThrow(/GPU quota exhausted/);
  });

  it("raises when the stream closes with no completed event", () => {
    expect(() => parseEventStream("event: heartbeat\ndata: null\n\n")).toThrow(
      /without returning a result/,
    );
  });

  it("unwraps the single-output array gr.Interface produces", () => {
    expect(parseEventStream(`event: complete\ndata: [{"a":1}]\n`)).toEqual({ a: 1 });
  });
});

describe("analyzeOnSpace", () => {
  it("uploads, enqueues and streams, in that order", async () => {
    const f = happyFetch();
    const out = await analyzeOnSpace("https://x.hf.space", clip(), { fetchImpl: f });

    expect(out).toEqual(RESULT);
    const urls = f.mock.calls.map((c) => String(c[0]));
    expect(urls).toEqual([
      "https://x.hf.space/gradio_api/upload",
      "https://x.hf.space/gradio_api/call/analyze",
      "https://x.hf.space/gradio_api/call/analyze/ev-1",
    ]);
  });

  it("sends the uploaded path back as gradio.FileData", async () => {
    const f = happyFetch();
    await analyzeOnSpace("https://x.hf.space", clip(), { fetchImpl: f });

    const body = JSON.parse(String(f.mock.calls[1][1]?.body));
    expect(body).toEqual({
      data: [{ path: "/tmp/gradio/abc/clip.webm", meta: { _type: "gradio.FileData" } }],
    });
  });

  it("names the upload field `files`, which is what Gradio reads", async () => {
    const f = happyFetch();
    await analyzeOnSpace("https://x.hf.space", clip(), { fetchImpl: f });

    const form = f.mock.calls[0][1]?.body as FormData;
    expect(form).toBeInstanceOf(FormData);
    expect((form.get("files") as File).name).toBe("clip.webm");
  });

  it("tolerates a trailing slash on the Space url", async () => {
    const f = happyFetch();
    await analyzeOnSpace("https://x.hf.space/", clip(), { fetchImpl: f });
    expect(String(f.mock.calls[0][0])).toBe("https://x.hf.space/gradio_api/upload");
  });

  it("surfaces an error the Space described in its payload", async () => {
    // space_live/app.py returns errors in the body rather than raising, because
    // a Gradio exception reaches a cross-origin caller as an opaque 500.
    const f = happyFetch(
      `event: complete\ndata: [{"error":"clip is 0.05s; nothing to analyse"}]\n\n`,
    );
    await expect(analyzeOnSpace("https://x.hf.space", clip(), { fetchImpl: f }))
      .rejects.toThrow(/nothing to analyse/);
  });

  it("reports which hop failed", async () => {
    const f = vi.fn(async () => ({ ok: false, status: 503 }) as unknown as Response);
    const err = await rejection(
      analyzeOnSpace("https://x.hf.space", clip(), { fetchImpl: f }),
    );
    expect(err).toBeInstanceOf(LiveSpaceError);
    expect(err.stage).toBe("upload");
    expect(err.message).toMatch(/503/);
  });

  it("does not enqueue when the upload returned no path", async () => {
    const f = vi.fn(async () => OK([]));
    await expect(analyzeOnSpace("https://x.hf.space", clip(), { fetchImpl: f }))
      .rejects.toThrow(/returned no file path/);
    expect(f).toHaveBeenCalledTimes(1);
  });

  it("explains a timeout as a cold Space rather than as a network failure", async () => {
    const f = vi.fn(async (_u: unknown, init?: RequestInit) => {
      await new Promise((_, reject) =>
        init?.signal?.addEventListener("abort", () =>
          reject(Object.assign(new Error("aborted"), { name: "AbortError" })),
        ),
      );
      return OK({});
    });
    const err = await rejection(
      analyzeOnSpace("https://x.hf.space", clip(), {
        fetchImpl: f as unknown as typeof fetch,
        timeoutMs: 10,
      }),
    );
    expect(err.message).toMatch(/cold ZeroGPU Space/);
  });
});
