"use client";

/* Two stacked panels sharing one x-axis (lap number).
 *
 * Deliberately NOT a dual-axis chart. Lap time (seconds) and DSI (0-100) have
 * no common scale, and overlaying them on twin y-axes would invent a visual
 * correlation the data may not support - the exact claim this project sets out
 * to measure honestly. Small multiples let the reader line the two up
 * themselves, and the correlation is reported as a number on the Evidence page.
 */

import { useMemo, useState } from "react";
import type { Message } from "@/lib/api";
import { dsiColor } from "@/lib/api";
import { safeMax, safeMin } from "@/lib/format";

interface Props {
  messages: Message[];
  lapTimes: { lap: number; seconds: number }[];
  onSelect: (m: Message) => void;
  selectedId?: string;
}

const W = 1000;
const H_PACE = 150;
const H_DSI = 150;
const PAD_L = 52;
const PAD_R = 16;
const PAD_T = 14;
const AXIS_H = 26;

export default function RaceTimeline({ messages, lapTimes, onSelect, selectedId }: Props) {
  const [hover, setHover] = useState<{ x: number; y: number; m: Message } | null>(null);

  const onLap = useMemo(
    () => messages.filter((m) => m.lap.in_race && m.lap.lap_number != null),
    [messages],
  );

  const maxLap = Math.max(...lapTimes.map((l) => l.lap), ...onLap.map((m) => m.lap.lap_number!), 1);

  // Clip the pace axis to sane racing laps: pit stops and safety cars produce
  // 100s+ outliers that would otherwise flatten the entire trace.
  const racing = lapTimes.map((l) => l.seconds).sort((a, b) => a - b);
  const p95 = racing[Math.floor(racing.length * 0.95)] ?? 100;

  // safeMin/safeMax return null on an empty list rather than the +/-Infinity
  // that `Math.min(...[])` gives. That distinction is the whole fix: Infinity
  // propagated silently into the tick positions and rendered "M NaN,NaN" path
  // data with Infinity axis labels. A race with no lap times now draws no pace
  // panel, which is the honest rendering of having no pace data.
  const lo = safeMin(racing);
  const hi = safeMax(racing);
  const hasPace = lo != null && hi != null;
  const loT = lo ?? 0;
  const hiT = hasPace ? Math.min(hi, p95 * 1.15) : 1;

  const x = (lap: number) => PAD_L + ((lap - 1) / Math.max(1, maxLap - 1)) * (W - PAD_L - PAD_R);
  const yPace = (s: number) =>
    PAD_T + (1 - (Math.min(s, hiT) - loT) / Math.max(0.001, hiT - loT)) * (H_PACE - PAD_T - 10);
  const yDsi = (d: number) => PAD_T + (1 - d / 100) * (H_DSI - PAD_T - 10);

  const paceLine = lapTimes
    .filter((l) => l.seconds <= hiT)
    .map((l, i) => `${i === 0 ? "M" : "L"}${x(l.lap).toFixed(1)},${yPace(l.seconds).toFixed(1)}`)
    .join(" ");

  const dsiPoints = [...onLap].sort((a, b) => a.lap.lap_number! - b.lap.lap_number!);
  const dsiLine = dsiPoints
    .map((m, i) => `${i === 0 ? "M" : "L"}${x(m.lap.lap_number!).toFixed(1)},${yDsi(m.dsi).toFixed(1)}`)
    .join(" ");

  const lapTicks = Array.from({ length: Math.min(12, maxLap) }, (_, i) =>
    Math.round(1 + (i * (maxLap - 1)) / Math.max(1, Math.min(12, maxLap) - 1)),
  );

  const paceTicks = hasPace ? [loT, (loT + hiT) / 2, hiT] : [];

  return (
    <div style={{ position: "relative" }}>
      <svg
        viewBox={`0 0 ${W} ${H_PACE + H_DSI + AXIS_H + 22}`}
        style={{ width: "100%", height: "auto", display: "block" }}
        role="img"
        aria-label="Lap time and driver stress index across the race"
      >
        {/* ---------- panel 1: pace ---------- */}
        <text x={PAD_L} y={11} fill="var(--ink-muted)" fontSize={10.5} letterSpacing="0.08em">
          LAP TIME (s)
        </text>
        {paceTicks.map((t, i) => (
          <g key={i}>
            <line
              x1={PAD_L}
              x2={W - PAD_R}
              y1={yPace(t)}
              y2={yPace(t)}
              stroke="var(--grid)"
              strokeWidth={1}
            />
            <text
              x={PAD_L - 8}
              y={yPace(t) + 3.5}
              fill="var(--ink-muted)"
              fontSize={10}
              textAnchor="end"
              style={{ fontVariantNumeric: "tabular-nums" }}
            >
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <path d={paceLine} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" />

        {/* ---------- panel 2: DSI ---------- */}
        <g transform={`translate(0, ${H_PACE})`}>
          <text x={PAD_L} y={11} fill="var(--ink-muted)" fontSize={10.5} letterSpacing="0.08em">
            DRIVER STATE INDEX
          </text>
          {[0, 50, 100].map((t) => (
            <g key={t}>
              <line
                x1={PAD_L}
                x2={W - PAD_R}
                y1={yDsi(t)}
                y2={yDsi(t)}
                stroke="var(--grid)"
                strokeWidth={1}
              />
              <text
                x={PAD_L - 8}
                y={yDsi(t) + 3.5}
                fill="var(--ink-muted)"
                fontSize={10}
                textAnchor="end"
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {t}
              </text>
            </g>
          ))}
          <path
            d={dsiLine}
            fill="none"
            stroke="var(--series-2)"
            strokeWidth={2}
            strokeLinejoin="round"
            opacity={0.85}
          />

          {dsiPoints.map((m) => {
            const cx = x(m.lap.lap_number!);
            const cy = yDsi(m.dsi);
            const sel = m.id === selectedId;
            return (
              <g key={m.id}>
                {/* 2px surface ring so overlapping markers stay separable */}
                <circle cx={cx} cy={cy} r={sel ? 7 : 5} fill="var(--surface-1)" />
                <circle
                  cx={cx}
                  cy={cy}
                  r={sel ? 5.5 : 3.8}
                  fill={dsiColor(m.dsi)}
                  stroke={m.suppressed_stress ? "var(--ink-primary)" : "none"}
                  strokeWidth={m.suppressed_stress ? 1.6 : 0}
                />
                {/* Generous invisible hit area - never make people hit a 4px dot.
                    It is also the keyboard target: these are real controls, so
                    they take focus and respond to Enter/Space, and carry a label
                    a screen reader can actually use. */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={13}
                  fill="transparent"
                  className="chart-point"
                  tabIndex={0}
                  role="button"
                  aria-label={
                    `Lap ${m.lap.lap_number}, ${m.state}, DSI ${m.dsi}` +
                    (m.suppressed_stress ? ", suppressed stress" : "") +
                    `: ${m.transcript.slice(0, 60)}`
                  }
                  onMouseEnter={() => setHover({ x: cx, y: cy + H_PACE, m })}
                  onMouseLeave={() => setHover(null)}
                  onFocus={() => setHover({ x: cx, y: cy + H_PACE, m })}
                  onBlur={() => setHover(null)}
                  onClick={() => onSelect(m)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect(m);
                    }
                  }}
                />
              </g>
            );
          })}
        </g>

        {/* ---------- shared x axis ---------- */}
        <g transform={`translate(0, ${H_PACE + H_DSI})`}>
          <line
            x1={PAD_L}
            x2={W - PAD_R}
            y1={4}
            y2={4}
            stroke="var(--baseline)"
            strokeWidth={1}
          />
          {lapTicks.map((l) => (
            <text
              key={l}
              x={x(l)}
              y={19}
              fill="var(--ink-muted)"
              fontSize={10}
              textAnchor="middle"
              style={{ fontVariantNumeric: "tabular-nums" }}
            >
              {l}
            </text>
          ))}
          <text x={(W - PAD_R + PAD_L) / 2} y={36} fill="var(--ink-muted)" fontSize={10.5} textAnchor="middle">
            LAP
          </text>
        </g>
      </svg>

      {hover && (
        <div
          style={{
            position: "absolute",
            left: `${(hover.x / W) * 100}%`,
            top: hover.y + 18,
            transform: "translateX(-50%)",
            background: "var(--surface-3)",
            border: "1px solid var(--hairline)",
            borderRadius: 8,
            padding: "9px 11px",
            pointerEvents: "none",
            zIndex: 20,
            minWidth: 190,
            maxWidth: 280,
          }}
        >
          <div style={{ fontSize: 11, color: "var(--ink-muted)", marginBottom: 4 }}>
            Lap {hover.m.lap.lap_number} · {hover.m.state}
          </div>
          <div style={{ fontSize: 12.5, marginBottom: 6, lineHeight: 1.35 }}>
            &ldquo;{hover.m.transcript.slice(0, 90)}
            {hover.m.transcript.length > 90 ? "…" : ""}&rdquo;
          </div>
          <div style={{ fontSize: 11, color: "var(--ink-secondary)" }}>
            DSI <strong style={{ color: dsiColor(hover.m.dsi) }}>{hover.m.dsi}</strong>
            {hover.m.lap.delta_to_median_s != null && (
              <> · pace {hover.m.lap.delta_to_median_s > 0 ? "+" : ""}
                {hover.m.lap.delta_to_median_s.toFixed(2)}s</>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
