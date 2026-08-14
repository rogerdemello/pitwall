"use client";

/* Live Analysis exists to prove the Race Replay is a real analysis and not a
 * recording. Upload or record a clip and watch the same pipeline run on it. */

import { useRef, useState } from "react";
import { analyzeClip, CAN_ANALYSE, dsiColor, LIVE_SPACE, STATIC_MODE } from "@/lib/api";

const RACE = "2021_Abu_Dhabi_Grand_Prix";

interface Result {
  transcript: string;
  duration_s: number;
  elapsed_s: number;
  rtf: number;
  text_sentiment: { label: string; polarity: number };
  state: {
    dsi: number;
    state: string;
    descriptor: string;
    arousal_pct: number;
    valence_pct: number;
    dominance_pct: number;
    incongruence: number;
    suppressed_stress: boolean;
    note: string;
  };
  calibrated_against: string | null;
}

export default function LivePage() {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [name, setName] = useState<string>("");
  const [recording, setRecording] = useState(false);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  async function run(file: File) {
    setBusy(true);
    setErr(null);
    setResult(null);
    setName(file.name);
    try {
      setResult(await analyzeClip(file, RACE));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function toggleRecord() {
    if (recording) {
      recorder.current?.stop();
      setRecording(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunks.current = [];
      mr.ondataavailable = (e) => chunks.current.push(e.data);
      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks.current, { type: mr.mimeType });
        run(new File([blob], "recording.webm", { type: mr.mimeType }));
      };
      mr.start();
      recorder.current = mr;
      setRecording(true);
    } catch {
      setErr("Microphone unavailable. Upload a clip instead.");
    }
  }

  const s = result?.state;

  return (
    <>
      <div style={{ marginBottom: 18 }}>
        <h1 style={{ fontSize: 21, margin: "0 0 4px", fontWeight: 650 }}>Live Analysis</h1>
        <p className="muted" style={{ margin: 0, fontSize: 13 }}>
          Run the full pipeline on any clip: speech-to-text, vocal affect, text sentiment,
          then fusion. Calibrated against the {RACE.replace(/_/g, " ")} corpus.
        </p>
      </div>

      {STATIC_MODE && CAN_ANALYSE && (
        <div className="card" style={{ marginBottom: 18 }}>
          <p className="card-title">Running on the ZeroGPU backend</p>
          <p style={{ fontSize: 13, lineHeight: 1.6, margin: 0 }}>
            This page is served by a <strong>static</strong> Space, which has no process
            to run a model in. The clip is sent to{" "}
            <a href={LIVE_SPACE} target="_blank" rel="noreferrer">
              a second Space
            </a>{" "}
            running the pipeline on ZeroGPU — the same <code>pipeline/</code> package
            behind the Race Replay, copied in verbatim at build time rather than
            reimplemented.
          </p>
          <p
            className="muted"
            style={{ fontSize: 12.5, lineHeight: 1.6, marginTop: 8, marginBottom: 0 }}
          >
            The first call of the day wakes the Space and loads three models, so it can
            take a minute; later ones are quick. ZeroGPU time is charged to you, not to
            the Space — signing in to Hugging Face raises the allowance.
          </p>
        </div>
      )}

      {!CAN_ANALYSE && (
        <div
          className="card"
          style={{
            marginBottom: 18,
            background: "rgba(250,178,25,0.08)",
            borderColor: "rgba(250,178,25,0.3)",
          }}
        >
          <div className="sev watch" style={{ marginBottom: 8 }}>
            <span aria-hidden>▲</span> Not available in this deployment
          </div>
          <p style={{ fontSize: 13, lineHeight: 1.6, margin: 0 }}>
            Live Analysis runs three models, which needs a backend. This is a{" "}
            <strong>static</strong> Space — Hugging Face hosts those for free, while
            Docker Spaces require a paid plan, so the app here serves precomputed
            results only.
          </p>
          <p
            className="muted"
            style={{ fontSize: 12.5, lineHeight: 1.6, marginTop: 8, marginBottom: 0 }}
          >
            Nothing is lost from the analysis itself: <strong>Race Replay</strong> is the
            output of exactly this pipeline over 2,042 real radio messages, and{" "}
            <strong>Evidence</strong> is fully live. To run a clip yourself, clone the
            repo and start it locally — the same page then works against the API.
          </p>
        </div>
      )}

      <div className="split even">
        <div className="card">
          <p className="card-title">Input</p>

          <label
            style={{
              display: "block",
              padding: "26px 16px",
              border: "1px dashed var(--baseline)",
              borderRadius: 8,
              textAlign: "center",
              cursor: "pointer",
              marginBottom: 12,
            }}
          >
            <input
              type="file"
              accept="audio/*"
              style={{ display: "none" }}
              onChange={(e) => e.target.files?.[0] && run(e.target.files[0])}
            />
            <div style={{ fontSize: 13.5, marginBottom: 4 }}>Drop an audio clip or click to browse</div>
            <div className="muted" style={{ fontSize: 11.5 }}>mp3, wav, m4a, webm</div>
          </label>

          <button
            className="chip"
            style={{
              width: "100%",
              padding: "11px",
              background: recording ? "var(--critical)" : "var(--surface-2)",
              borderColor: recording ? "var(--critical)" : "var(--hairline)",
              color: recording ? "#fff" : "var(--ink-secondary)",
              fontWeight: recording ? 600 : 400,
            }}
            onClick={toggleRecord}
            disabled={busy}
          >
            {recording ? "■  Stop recording" : "●  Record from microphone"}
          </button>

          {name && (
            <p className="muted mono" style={{ fontSize: 11.5, marginTop: 12, wordBreak: "break-all" }}>
              {name}
            </p>
          )}

          <p className="muted" style={{ fontSize: 11.5, lineHeight: 1.55, marginTop: 16, marginBottom: 0 }}>
            {STATIC_MODE && CAN_ANALYSE
              ? "Inference runs on a shared ZeroGPU slice, so a warm call takes a few seconds and a cold one takes a minute. The Race Replay is precomputed for exactly this reason."
              : "Inference runs on CPU, so expect a few seconds per clip. The Race Replay is precomputed for exactly this reason."}
          </p>
        </div>

        <div className="card">
          <p className="card-title">Result</p>

          {busy && <p className="muted" style={{ fontSize: 13 }}>Running the pipeline…</p>}
          {err && <p style={{ color: "var(--critical)", fontSize: 13 }}>{err}</p>}
          {!busy && !err && !result && (
            <p className="muted" style={{ fontSize: 13, lineHeight: 1.6 }}>
              Nothing analysed yet. Upload or record a clip to see the transcript, the
              vocal affect, and the fused driver state.
            </p>
          )}

          {result && s && (
            <>
              <blockquote
                style={{
                  margin: "0 0 16px",
                  padding: "11px 13px",
                  background: "var(--surface-2)",
                  borderLeft: "2px solid var(--series-1)",
                  borderRadius: 6,
                  fontSize: 13.5,
                  lineHeight: 1.5,
                }}
              >
                &ldquo;{result.transcript || "(no speech detected)"}&rdquo;
              </blockquote>

              <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 16 }}>
                <div>
                  <div className="hero-figure" style={{ color: dsiColor(s.dsi) }}>{s.dsi}</div>
                  <div className="muted" style={{ fontSize: 10.5, letterSpacing: "0.08em", marginTop: 4 }}>
                    DSI
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 3 }}>{s.state}</div>
                  <div className="muted" style={{ fontSize: 12 }}>{s.descriptor}</div>
                </div>
              </div>

              {s.suppressed_stress && (
                <div
                  style={{
                    padding: "11px 13px",
                    background: "rgba(250,178,25,0.10)",
                    border: "1px solid rgba(250,178,25,0.3)",
                    borderRadius: 8,
                    marginBottom: 16,
                  }}
                >
                  <div className="sev watch" style={{ marginBottom: 7 }}>
                    <span aria-hidden>▲</span> Suppressed stress
                  </div>
                  <div style={{ fontSize: 12.5, lineHeight: 1.5, color: "var(--ink-secondary)" }}>{s.note}</div>
                </div>
              )}

              <div className="tablewrap">
                <table className="data">
                  <tbody>
                    <tr><td>Arousal (percentile)</td><td>{(s.arousal_pct * 100).toFixed(0)}</td></tr>
                    <tr><td>Valence (percentile)</td><td>{(s.valence_pct * 100).toFixed(0)}</td></tr>
                    <tr><td>Control (percentile)</td><td>{(s.dominance_pct * 100).toFixed(0)}</td></tr>
                    <tr><td>Text sentiment</td><td>{result.text_sentiment.label} ({result.text_sentiment.polarity.toFixed(2)})</td></tr>
                    <tr><td>Words/voice gap</td><td>{s.incongruence.toFixed(2)}</td></tr>
                    <tr><td>Audio duration</td><td>{result.duration_s.toFixed(2)}s</td></tr>
                    <tr><td>Processing time</td><td>{result.elapsed_s.toFixed(2)}s (RTF {result.rtf})</td></tr>
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
