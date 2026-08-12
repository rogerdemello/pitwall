"use client";

import type { Message } from "@/lib/api";
import { audioUrl, dsiColor } from "@/lib/api";

/* The panel that opens when a radio pin is clicked: hear it, read it, see the
 * state it produced and the lap it landed on. */

export default function MessageDetail({
  message,
  raceId,
  hasAudio = true,
}: {
  message: Message | null;
  raceId: string;
  hasAudio?: boolean;
}) {
  if (!message) {
    return (
      <div className="card" style={{ minHeight: 300 }}>
        <p className="card-title">Radio message</p>
        <p className="muted" style={{ fontSize: 13, lineHeight: 1.6 }}>
          Select a point on the Driver State Index to play the radio call, read the
          transcript, and see the lap it was spoken on.
        </p>
      </div>
    );
  }

  const lap = message.lap;
  const bars: [string, number][] = [
    ["Arousal", message.arousal_pct],
    ["Valence", message.valence_pct],
    ["Control", message.dominance_pct],
  ];

  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 10 }}>
        <span style={{ fontWeight: 700, fontSize: 15 }}>{message.driver_code}</span>
        <span className="muted" style={{ fontSize: 12 }}>
          {lap.in_race ? `Lap ${lap.lap_number}` : lap.note}
        </span>
        <span className="muted mono" style={{ fontSize: 11, marginLeft: "auto" }}>
          {message.timestamp.slice(11, 19)} UTC
        </span>
      </div>

      {/* Who is speaking changes what the affect score is evidence *of*. */}
      <div
        title={message.speaker_reason}
        style={{
          display: "inline-block",
          fontSize: 10.5,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          padding: "3px 8px",
          borderRadius: 5,
          marginBottom: 12,
          background: "var(--surface-3)",
          color: message.speaker === "driver" ? "var(--series-3)" : "var(--ink-muted)",
        }}
      >
        {message.speaker === "driver"
          ? "Driver voice"
          : message.speaker === "engineer"
            ? "Engineer voice — not scored as driver state"
            : "Speaker unclear"}
      </div>

      {hasAudio ? (
        <audio
          controls
          src={audioUrl(raceId, message.audio_file)}
          style={{ width: "100%", height: 34, marginBottom: 14 }}
        />
      ) : (
        // Better to say why than to render a player that will silently fail.
        <p
          className="muted"
          style={{
            fontSize: 11.5,
            lineHeight: 1.5,
            padding: "8px 11px",
            background: "var(--surface-2)",
            borderRadius: 6,
            marginTop: 0,
            marginBottom: 14,
          }}
        >
          Audio for this race isn&rsquo;t bundled in this deployment — all twelve races
          would be 327 MB. The showcase race plays; the analysis below is identical
          either way.
        </p>
      )}

      <blockquote
        style={{
          margin: "0 0 14px",
          padding: "11px 13px",
          background: "var(--surface-2)",
          borderLeft: `2px solid var(--series-1)`,
          borderRadius: 6,
          fontSize: 13.5,
          lineHeight: 1.5,
        }}
      >
        &ldquo;{message.transcript}&rdquo;
      </blockquote>

      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16 }}>
        <div>
          <div className="hero-figure" style={{ color: dsiColor(message.dsi) }}>
            {message.dsi}
          </div>
          <div className="muted" style={{ fontSize: 10.5, letterSpacing: "0.08em", marginTop: 4 }}>
            DSI
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 3 }}>
            {message.state}
            {/* Validation on gold labels showed arousal is recovered (+16pts over
                baseline) but valence is at chance (+1pt). Stressed/Energised and
                Calm/Fatigued differ ONLY in valence, so those pairs are the
                unreliable ones and the UI should not pretend otherwise. */}
            <span
              title="Arousal is validated (+16 pts over baseline); valence is at chance (+1 pt). Stressed vs Energised, and Calm vs Fatigued, differ only in valence — treat that distinction as unreliable. See Evidence."
              style={{
                marginLeft: 7,
                fontSize: 10,
                fontWeight: 500,
                padding: "2px 6px",
                borderRadius: 4,
                background: "var(--surface-3)",
                color: "var(--ink-muted)",
                cursor: "help",
                verticalAlign: "middle",
              }}
            >
              valence unvalidated
            </span>
          </div>
          <div className="muted" style={{ fontSize: 12, lineHeight: 1.45 }}>
            {message.descriptor}
          </div>
        </div>
      </div>

      {bars.map(([label, v]) => (
        <div key={label} style={{ marginBottom: 9 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
            <span className="muted">{label}</span>
            <span className="mono" style={{ color: "var(--ink-secondary)" }}>
              {(v * 100).toFixed(0)}
              <span className="muted" style={{ fontSize: 10 }}> pct</span>
            </span>
          </div>
          <div style={{ height: 5, background: "var(--surface-3)", borderRadius: 3 }}>
            <div
              style={{
                width: `${v * 100}%`,
                height: "100%",
                background: "var(--series-1)",
                borderRadius: 3,
              }}
            />
          </div>
        </div>
      ))}

      {message.suppressed_stress && (
        <div
          style={{
            marginTop: 14,
            padding: "11px 13px",
            background: "rgba(250,178,25,0.10)",
            border: "1px solid rgba(250,178,25,0.3)",
            borderRadius: 8,
          }}
        >
          <div className="sev watch" style={{ marginBottom: 7 }}>
            <span aria-hidden>▲</span> Suppressed stress
          </div>
          <div style={{ fontSize: 12.5, lineHeight: 1.5, color: "var(--ink-secondary)" }}>
            {message.note}
          </div>
        </div>
      )}

      {message.recommendation && (
        <div
          style={{
            marginTop: 14,
            padding: "12px 13px",
            background: "var(--surface-2)",
            borderRadius: 8,
            border: "1px solid var(--hairline)",
          }}
        >
          <div className={`sev ${message.recommendation.severity}`} style={{ marginBottom: 8 }}>
            <span aria-hidden>{message.recommendation.severity === "act" ? "●" : "▲"}</span>
            {message.recommendation.severity}
          </div>
          <div style={{ fontSize: 13.5, fontWeight: 600, marginBottom: 5 }}>
            {message.recommendation.headline}
          </div>
          <div style={{ fontSize: 12.5, lineHeight: 1.5, color: "var(--ink-secondary)", marginBottom: 9 }}>
            {message.recommendation.detail}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {message.recommendation.evidence.map((e) => (
              <span
                key={e}
                className="mono"
                style={{
                  fontSize: 10.5,
                  padding: "3px 7px",
                  background: "var(--surface-3)",
                  borderRadius: 4,
                  color: "var(--ink-muted)",
                }}
              >
                {e}
              </span>
            ))}
          </div>
        </div>
      )}

      {lap.in_race && (
        <div className="stat-row" style={{ marginTop: 14 }}>
          <div className="stat">
            <div className="label">Lap time</div>
            <div className="value mono" style={{ fontSize: 19 }}>
              {lap.lap_time_s?.toFixed(2)}s
            </div>
            {lap.delta_to_median_s != null && (
              <div className="foot">
                {lap.delta_to_median_s > 0 ? "+" : ""}
                {lap.delta_to_median_s.toFixed(2)}s vs median
              </div>
            )}
          </div>
          <div className="stat">
            <div className="label">Tyres</div>
            <div className="value" style={{ fontSize: 19 }}>
              {lap.compound ? lap.compound.charAt(0) + lap.compound.slice(1).toLowerCase() : "—"}
            </div>
            <div className="foot">{lap.tyre_life ?? "—"} laps old</div>
          </div>
        </div>
      )}
    </div>
  );
}
