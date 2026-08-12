"use client";

import { useEffect, useMemo, useState } from "react";
import RaceTimeline from "@/components/RaceTimeline";
import MessageDetail from "@/components/MessageDetail";
import RacePicker, { useRaces } from "@/components/RacePicker";
import {
  dsiColor,
  getAudioManifest,
  getRace,
  STATIC_MODE,
  type Race,
} from "@/lib/api";

// The showcase race, if it has been built. Falls back to whatever exists.
const DEFAULT_RACE = "2021_Abu_Dhabi_Grand_Prix";

export default function ReplayPage() {
  const { races, raceId, setRaceId } = useRaces(DEFAULT_RACE);
  const [race, setRace] = useState<Race | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [driverId, setDriverId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showTable, setShowTable] = useState(false);
  // Static builds bundle audio for the showcase race only.
  const [audioRaces, setAudioRaces] = useState<string[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    getRace(raceId)
      .then((r) => {
        if (cancelled) return;
        setRace(r);
        const top = r.drivers.find((d) => d.in_race_count > 0) ?? r.drivers[0];
        setDriverId(top?.driver_id ?? null);
      })
      .catch((e) => !cancelled && setErr(String(e)));
    // Guards against a slow response for a race the user has already navigated
    // away from overwriting the newer one.
    return () => {
      cancelled = true;
    };
  }, [raceId]);

  useEffect(() => {
    if (!STATIC_MODE) return;
    getAudioManifest().then((m) => setAudioRaces(m.races_with_audio));
  }, []);

  const messages = useMemo(
    () => (race && driverId ? race.messages.filter((m) => m.driver_id === driverId) : []),
    [race, driverId],
  );

  // The driver's complete lap trace, straight from telemetry - not only the laps
  // that happen to carry a radio message.
  const lapTimes = useMemo(
    () =>
      race && driverId
        ? (race.lap_traces?.[driverId] ?? []).map((l) => ({ lap: l.lap, seconds: l.seconds }))
        : [],
    [race, driverId],
  );

  // Derived, not stored: switching driver drops a selection that no longer
  // belongs to the visible timeline without needing an effect to reset it.
  const selected = useMemo(
    () => messages.find((m) => m.id === selectedId) ?? null,
    [messages, selectedId],
  );

  // Shortcuts to the moments worth showing: every suppressed-stress flag, plus
  // the highest-stress calls. Saves hunting for them during a live demo.
  const keyMoments = useMemo(() => {
    const onLapMsgs = messages.filter((m) => m.lap.in_race);
    const flaggedM = onLapMsgs.filter((m) => m.suppressed_stress);
    const topM = [...onLapMsgs]
      .sort((a, b) => b.dsi - a.dsi)
      .filter((m) => !flaggedM.includes(m))
      .slice(0, 4);
    return [...flaggedM, ...topM]
      .sort((a, b) => (a.lap.lap_number ?? 0) - (b.lap.lap_number ?? 0))
      .slice(0, 7);
  }, [messages]);

  if (err) {
    return (
      <div className="card">
        <p className="card-title">Cannot reach the API</p>
        <p style={{ fontSize: 13, lineHeight: 1.6 }} className="muted">
          {err}
          <br />
          Start the backend with{" "}
          <code className="mono">uvicorn main:app --reload</code> from the{" "}
          <code className="mono">backend/</code> directory.
        </p>
      </div>
    );
  }

  if (!race) return <p className="muted">Loading race…</p>;

  const driver = race.drivers.find((d) => d.driver_id === driverId);
  const onLap = messages.filter((m) => m.lap.in_race);
  const flagged = messages.filter((m) => m.suppressed_stress);
  const calls = messages.filter((m) => m.recommendation);

  return (
    <>
      <div style={{ marginBottom: 18 }}>
        <h1 style={{ fontSize: 21, margin: "0 0 4px", fontWeight: 650 }}>{race.grand_prix}</h1>
        <p className="muted" style={{ margin: 0, fontSize: 13 }}>
          {race.message_count} radio messages · {race.in_race_count} placed on a race lap ·{" "}
          {race.session_date}
        </p>
      </div>

      <RacePicker races={races} raceId={raceId} onChange={setRaceId} />

      {/* One filter row above everything it scopes */}
      <div className="filter-row">
        {race.drivers
          .filter((d) => d.in_race_count > 0)
          .map((d) => (
            <button
              key={d.driver_id}
              className={`chip ${d.driver_id === driverId ? "on" : ""}`}
              onClick={() => setDriverId(d.driver_id)}
            >
              {d.code}
              <span style={{ opacity: 0.65, marginLeft: 6, fontSize: 11 }}>{d.in_race_count}</span>
            </button>
          ))}
      </div>

      {driver && (
        <div className="stat-row" style={{ marginBottom: 18 }}>
          <div className="stat">
            <div className="label">Driver</div>
            <div className="value" style={{ fontSize: 20 }}>{driver.name}</div>
            <div className="foot">Car #{driver.racing_number}</div>
          </div>
          <div className="stat">
            <div className="label">Mean DSI</div>
            <div className="value" style={{ color: dsiColor(driver.mean_dsi ?? 0) }}>
              {driver.mean_dsi ?? "—"}
            </div>
            <div className="foot">across {driver.in_race_count} on-lap calls</div>
          </div>
          <div className="stat">
            <div className="label">Peak DSI</div>
            <div className="value" style={{ color: dsiColor(driver.peak_dsi ?? 0) }}>
              {driver.peak_dsi ?? "—"}
            </div>
            <div className="foot">highest single message</div>
          </div>
          <div className="stat">
            <div className="label">Suppressed stress</div>
            <div className="value">{flagged.length}</div>
            <div className="foot">words/voice mismatch</div>
          </div>
          <div className="stat">
            <div className="label">Pit-wall calls</div>
            <div className="value">{calls.length}</div>
            <div className="foot">recommendations raised</div>
          </div>
        </div>
      )}

      <div className="split">
        <div className="card">
          <div style={{ display: "flex", alignItems: "center", marginBottom: 10 }}>
            <p className="card-title" style={{ margin: 0 }}>Race timeline</p>
            <button className="linkish" style={{ marginLeft: "auto" }} onClick={() => setShowTable((s) => !s)}>
              {showTable ? "Show chart" : "Show table"}
            </button>
          </div>

          {/* legend: two panels, one series each - identity is never colour-alone */}
          <div style={{ display: "flex", gap: 18, marginBottom: 10, fontSize: 11.5 }}>
            <span className="muted">
              <span style={{ display: "inline-block", width: 10, height: 2, background: "var(--series-1)", marginRight: 6, verticalAlign: "middle" }} />
              Lap time
            </span>
            <span className="muted">
              <span style={{ display: "inline-block", width: 10, height: 2, background: "var(--series-2)", marginRight: 6, verticalAlign: "middle" }} />
              Driver State Index
            </span>
            <span className="muted">
              <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 4, border: "1.5px solid var(--ink-primary)", marginRight: 6, verticalAlign: "middle" }} />
              Suppressed stress
            </span>
          </div>

          {showTable ? (
            <div className="tablewrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>Lap</th><th>Time</th><th>DSI</th><th>State</th><th>Pace Δ</th><th>Transcript</th>
                  </tr>
                </thead>
                <tbody>
                  {onLap.map((m) => (
                    <tr key={m.id}>
                      <td>{m.lap.lap_number}</td>
                      <td>{m.timestamp.slice(11, 19)}</td>
                      <td style={{ color: dsiColor(m.dsi), fontWeight: 600 }}>{m.dsi}</td>
                      <td>{m.state}</td>
                      <td>
                        {m.lap.delta_to_median_s != null
                          ? `${m.lap.delta_to_median_s > 0 ? "+" : ""}${m.lap.delta_to_median_s.toFixed(2)}s`
                          : "—"}
                      </td>
                      <td style={{ maxWidth: 320 }}>{m.transcript.slice(0, 70)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : onLap.length ? (
            <>
              <RaceTimeline
                messages={messages}
                lapTimes={lapTimes}
                onSelect={(m) => setSelectedId(m.id)}
                selectedId={selected?.id}
              />
              {keyMoments.length > 0 && (
                <div style={{ marginTop: 14, borderTop: "1px solid var(--grid)", paddingTop: 12 }}>
                  <div
                    className="muted"
                    style={{ fontSize: 10.5, letterSpacing: "0.09em", marginBottom: 8 }}
                  >
                    KEY MOMENTS
                  </div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {keyMoments.map((m) => (
                      <button
                        key={m.id}
                        className={`chip ${m.id === selectedId ? "on" : ""}`}
                        style={{ fontSize: 12, padding: "6px 12px" }}
                        onClick={() => setSelectedId(m.id)}
                      >
                        <span style={{ opacity: 0.7 }}>L{m.lap.lap_number}</span>
                        <span style={{ margin: "0 7px", fontWeight: 600 }}>{m.dsi}</span>
                        <span style={{ opacity: 0.85 }}>
                          {m.suppressed_stress ? "suppressed stress" : m.state.toLowerCase()}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <p className="muted" style={{ fontSize: 13 }}>
              No messages from this driver landed on a race lap.
            </p>
          )}
        </div>

        <MessageDetail
          message={selected}
          raceId={race.race_id}
          hasAudio={!STATIC_MODE || (audioRaces ?? []).includes(race.race_id)}
        />
      </div>
    </>
  );
}
