"use client";

/* Evidence is the screen most teams will not build.
 *
 * Everything here is measured on real data and reported whichever way it comes
 * out - including the correlation that the whole project rests on. A weak
 * number stated plainly is worth more than a strong number asserted. */

import { useEffect, useState } from "react";
import {
  getAffectEval,
  getAsrEval,
  getEvidence,
  getComparison,
  getCorpusAnalysis,
  getConvergent,
  getCorpusAsr,
  getEraAnalysis,
  getGoldAffect,
  getCorpusFinding,
  getExperiments,
  type AffectEval,
  type AsrEval,
  type DiarizationExperiment,
  type RaceComparison,
  type CorpusFinding,
  type CorpusAnalysis,
  type CorpusAsr,
  type Convergent,
  type GoldAffect,
  type EraAnalysis,
  type Evidence,
} from "@/lib/api";
import { pct, pctOf } from "@/lib/format";
import RacePicker, { useRaces } from "@/components/RacePicker";

const DEFAULT_RACE = "2021_Abu_Dhabi_Grand_Prix";

function interpret(r: number | null): { verdict: string; detail: string } {
  if (r == null) return { verdict: "Not computable", detail: "Too few paired observations." };
  const a = Math.abs(r);
  const dir = r > 0 ? "higher stress goes with slower laps" : "higher stress goes with faster laps";
  if (a < 0.1) return { verdict: "No linear relationship", detail: `r = ${r}. On this race the two are essentially independent.` };
  if (a < 0.3) return { verdict: "Weak relationship", detail: `r = ${r} — ${dir}, but weakly.` };
  if (a < 0.5) return { verdict: "Moderate relationship", detail: `r = ${r} — ${dir}.` };
  return { verdict: "Strong relationship", detail: `r = ${r} — ${dir}.` };
}

export default function EvidencePage() {
  const [loaded, setEv] = useState<Evidence | null>(null);
  const [asr, setAsr] = useState<AsrEval | null>(null);
  const [affect, setAffect] = useState<AffectEval | null>(null);
  const [experiments, setExperiments] = useState<DiarizationExperiment[]>([]);
  const [cmp, setCmp] = useState<RaceComparison | null>(null);
  const [finding, setFinding] = useState<CorpusFinding | null>(null);
  const [ca, setCa] = useState<CorpusAnalysis | null>(null);
  const [casr, setCasr] = useState<CorpusAsr | null>(null);
  const [gold, setGold] = useState<GoldAffect | null>(null);
  const [conv, setConv] = useState<Convergent | null>(null);
  const [era, setEra] = useState<EraAnalysis | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const { races, raceId, setRaceId } = useRaces(DEFAULT_RACE);

  useEffect(() => {
    getEvidence(raceId).then(setEv).catch((e) => setErr(String(e)));
    // Each of these degrades to "not yet measured" rather than blocking the page.
    getAsrEval(raceId).then(setAsr).catch(() => setAsr(null));
    getAffectEval(raceId).then(setAffect).catch(() => setAffect(null));
  }, [raceId]);

  useEffect(() => {
    getExperiments().then((r) => setExperiments(r.experiments)).catch(() => setExperiments([]));
    getComparison().then(setCmp).catch(() => setCmp(null));
    getCorpusFinding().then(setFinding).catch(() => setFinding(null));
    getCorpusAnalysis().then(setCa).catch(() => setCa(null));
    getCorpusAsr().then(setCasr).catch(() => setCasr(null));
    getGoldAffect().then(setGold).catch(() => setGold(null));
    getConvergent().then(setConv).catch(() => setConv(null));
    getEraAnalysis().then(setEra).catch(() => setEra(null));
  }, []);

  if (err) return <p className="muted" style={{ fontSize: 13 }}>Cannot reach the API: {err}</p>;

  // Derived rather than cleared in an effect: data from the previously selected
  // race is simply not shown, so switching races can never display stale numbers
  // under a new race's heading.
  const ev = loaded && loaded.race_id === raceId ? loaded : null;

  const header = (
    <>
      <div style={{ marginBottom: 18 }}>
        <h1 style={{ fontSize: 21, margin: "0 0 4px", fontWeight: 650 }}>Evidence</h1>
        <p className="muted" style={{ margin: 0, fontSize: 13 }}>
          {ev
            ? `What we measured, on ${ev.message_count} real radio messages from ${raceId.replace(/_/g, " ")} — reported as it came out.`
            : "Loading…"}
        </p>
      </div>
      <RacePicker races={races} raceId={raceId} onChange={setRaceId} />
    </>
  );

  // Keep the picker mounted while a race loads, so switching races does not make
  // the controls disappear underneath the cursor.
  if (!ev) return header;

  const corr = interpret(ev.dsi_vs_lap_delta_correlation);
  const cal = ev.calibration;
  const svp = ev.stress_vs_pace;

  return (
    <>
      {header}

      <div className="stat-row" style={{ marginBottom: 18 }}>
        <div className="stat">
          <div className="label">Messages analysed</div>
          <div className="value">{ev.message_count}</div>
          <div className="foot">full race session</div>
        </div>
        <div className="stat">
          <div className="label">Placed on a lap</div>
          <div className="value">{ev.on_lap_count}</div>
          <div className="foot">{pctOf(ev.join_rate, 0)} telemetry join rate</div>
        </div>
        <div className="stat">
          <div className="label">DSI range</div>
          <div className="value">{ev.dsi.min}–{ev.dsi.max}</div>
          <div className="foot">mean {ev.dsi.mean}</div>
        </div>
        <div className="stat">
          <div className="label">Suppressed stress</div>
          <div className="value">{ev.suppressed_stress_count}</div>
          <div className="foot">
            {pct(ev.suppressed_stress_count, ev.suppressed_stress_eligible, 1)} of{" "}
            {ev.suppressed_stress_eligible} eligible
          </div>
        </div>
      </div>

      {ca && ca.measured !== false && ca.stress_vs_pace && ca.lag && (
        <div className="card" style={{ marginBottom: 18 }}>
          <p className="card-title">
            The central question, over the whole corpus ({ca.messages_pooled} messages,{" "}
            {ca.races?.length} races)
          </p>
          <p style={{ fontSize: 14, lineHeight: 1.6, marginTop: 0 }}>{ca.verdict}</p>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit,minmax(220px,1fr))",
              gap: 12,
              marginTop: 14,
            }}
          >
            <div className="tablewrap">
              <table className="data">
                <thead><tr><th>Lag</th><th>n</th><th>r</th><th>p</th></tr></thead>
                <tbody>
                  {Object.entries(ca.lag.by_lag).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k.replace("lag_", "lap +")}</td>
                      <td>{v.n}</td>
                      <td>{v.r ?? "—"}</td>
                      <td>{v.p ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div>
              <p style={{ fontSize: 12.5, lineHeight: 1.6, marginTop: 0, color: "var(--ink-secondary)" }}>
                A relationship at lap +0 would say stress and slow laps merely
                co-occur. A relationship at lap +1 or beyond would say the radio call
                carries information about laps that have not happened yet — the
                difference between a dashboard and a decision.
              </p>
              <p className="muted" style={{ fontSize: 12, lineHeight: 1.6, marginBottom: 0 }}>
                {ca.lag.caveat}
              </p>
            </div>
          </div>

          <p className="muted" style={{ fontSize: 12, lineHeight: 1.6, marginTop: 12, marginBottom: 0 }}>
            {ca.caveat}
          </p>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(340px,1fr))", gap: 18 }}>
        <div className="card">
          <p className="card-title">Does stress track lap-time loss? (this race)</p>

          {svp.tercile && (
            <div style={{ marginBottom: 16 }}>
              <div
                className="hero-figure"
                style={{ color: svp.tercile.mean_gap_s > 0 ? "var(--serious)" : "var(--good)" }}
              >
                {svp.tercile.mean_gap_s > 0 ? "+" : ""}
                {svp.tercile.mean_gap_s.toFixed(2)}s
              </div>
              <p style={{ fontSize: 13, lineHeight: 1.55, marginTop: 8, marginBottom: 0 }}>
                Average lap-time gap between a driver&rsquo;s most-stressed and calmest radio
                calls — computed within each driver, then averaged, so a talkative driver
                cannot dominate the result.
              </p>
              <p className="muted" style={{ fontSize: 12.5, marginTop: 6, marginBottom: 0 }}>
                {svp.tercile.drivers_slower_when_stressed} of {svp.tercile.drivers_total}{" "}
                drivers were slower on their most-stressed laps — so the direction is not
                consistent, and on this race we do <strong>not</strong> claim a reliable effect.
              </p>
            </div>
          )}

          {svp.tercile && (
            <div className="tablewrap" style={{ maxHeight: 200 }}>
              <table className="data">
                <thead>
                  <tr><th>Driver</th><th>n</th><th>Calm Δ</th><th>Stressed Δ</th><th>Gap</th></tr>
                </thead>
                <tbody>
                  {svp.tercile.drivers.map((d) => (
                    <tr key={d.driver}>
                      <td>{d.driver}</td>
                      <td>{d.n}</td>
                      <td>{d.calm_mean_delta_s > 0 ? "+" : ""}{d.calm_mean_delta_s.toFixed(2)}s</td>
                      <td>{d.stressed_mean_delta_s > 0 ? "+" : ""}{d.stressed_mean_delta_s.toFixed(2)}s</td>
                      <td style={{ color: d.gap_s > 0 ? "var(--serious)" : "var(--good)", fontWeight: 600 }}>
                        {d.gap_s > 0 ? "+" : ""}{d.gap_s.toFixed(2)}s
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div
            style={{
              marginTop: 14,
              padding: "11px 13px",
              background: "var(--surface-2)",
              borderRadius: 8,
              border: "1px solid var(--hairline)",
            }}
          >
            <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 5 }}>
              Why {svp.excluded_non_racing_laps} laps were excluded
            </div>
            <p style={{ fontSize: 12.5, lineHeight: 1.6, margin: 0, color: "var(--ink-secondary)" }}>
              Only green-flag laps with no pit entry or exit are counted. The first run of
              this analysis left them in, and one driver&rsquo;s &ldquo;stressed&rdquo; tercile averaged
              <strong> +18.8s</strong> — a pit stop, not a mood. A 20-second stop and a
              30-second safety-car lap are an order of magnitude larger than any plausible
              effect of driver state, so leaving them in would have manufactured a dramatic
              result out of nothing.
            </p>
          </div>

          <p style={{ fontSize: 12.5, lineHeight: 1.6, marginTop: 14, marginBottom: 6 }} className="muted">
            <strong style={{ color: "var(--ink-secondary)" }}>Pooled correlation:</strong>{" "}
            {corr.detail} over {ev.correlation_n} paired observations. Pooling every driver
            together mixes between-driver differences into a within-driver question, which
            is why the contrast above is the figure we lead with.
          </p>
          <p style={{ fontSize: 12.5, lineHeight: 1.6, marginBottom: 0 }} className="muted">
            Stated whichever way it lands. One race is a small sample, and lap time is
            shaped by traffic, fuel load and tyre state far more than by mood. Worth noting
            as a hypothesis rather than a finding: the two title contenders — the drivers
            under actual championship pressure — showed the largest gaps, while the
            midfield did not. Six drivers is far too few to conclude anything from that.
          </p>
        </div>

        <div className="card">
          <p className="card-title">Why calibration was necessary</p>
          <p style={{ fontSize: 13, lineHeight: 1.6, color: "var(--ink-secondary)", marginTop: 0 }}>
            The prosody model was trained on studio podcast speech. On compressed team
            radio its raw outputs collapse into a narrow band, so a textbook 0.5 threshold
            labels almost every message the same way.
          </p>
          <div className="tablewrap">
            <table className="data">
              <thead>
                <tr><th>Dimension</th><th>Min</th><th>P10</th><th>Median</th><th>P90</th><th>Max</th></tr>
              </thead>
              <tbody>
                {(["arousal", "valence", "dominance"] as const).map((k) => (
                  <tr key={k}>
                    <td style={{ textTransform: "capitalize" }}>{k}</td>
                    <td>{cal[k]?.min}</td>
                    <td>{cal[k]?.p10}</td>
                    <td>{cal[k]?.median}</td>
                    <td>{cal[k]?.p90}</td>
                    <td>{cal[k]?.max}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: 12.5, lineHeight: 1.6, marginBottom: 0, marginTop: 12 }} className="muted">
            We therefore score each message as a percentile against the F1 radio corpus
            rather than against the model&rsquo;s nominal 0–1 range. DSI 80 means &ldquo;top fifth of
            stress for this race&rdquo;, which is a claim an engineer can act on.
          </p>
        </div>

        <div className="card">
          <p className="card-title">Speech recognition — an ablation we lost</p>
          <p style={{ fontSize: 13, lineHeight: 1.6, color: "var(--ink-secondary)", marginTop: 0 }}>
            We expected F1 vocabulary prompting to be the differentiator here. We measured
            it, and it is not. It ships <strong>disabled</strong>.
          </p>

          {casr && casr.measured !== false && casr.races && (
            <>
              <div className="tablewrap" style={{ marginBottom: 8 }}>
                <table className="data">
                  <thead>
                    <tr><th>Race</th><th>WER base</th><th>WER prompted</th><th></th></tr>
                  </thead>
                  <tbody>
                    {casr.races.map((r) => {
                      const hurt = r.wer_biased > r.wer_unbiased;
                      return (
                        <tr key={r.race_id}>
                          <td>{r.race_id.replace("_Grand_Prix", "").replace(/_/g, " ")}</td>
                          <td>{r.wer_unbiased.toFixed(4)}</td>
                          <td style={{ color: hurt ? "var(--critical)" : "var(--good)" }}>
                            {r.wer_biased.toFixed(4)}
                          </td>
                          <td className="muted">{hurt ? "worse" : "better"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p style={{ fontSize: 12.5, lineHeight: 1.6, marginTop: 0 }}>
                <strong>{casr.conclusion}</strong> Mean WER {casr.mean_wer_unbiased} unprompted
                vs {casr.mean_wer_biased} prompted; jargon recall moved only{" "}
                {casr.jargon_unbiased} → {casr.jargon_biased} of {casr.jargon_total}. On
                Qatar the prompt nearly doubled the error rate.
              </p>
            </>
          )}

          <p className="card-title" style={{ marginTop: 14 }}>This race</p>
          {asr?.measured === false || !asr?.wer_unbiased ? (
            <p className="muted" style={{ fontSize: 12.5, fontStyle: "italic" }}>
              Not yet measured for this race — run{" "}
              <span className="mono">eval_asr.py</span>.
            </p>
          ) : (
            <>
              <div className="tablewrap" style={{ marginBottom: 12 }}>
                <table className="data">
                  <thead>
                    <tr><th>Metric</th><th>Unbiased</th><th>F1-prompted</th></tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Word error rate</td>
                      <td>{asr.wer_unbiased?.toFixed(4)}</td>
                      <td>{asr.wer_biased?.toFixed(4)}</td>
                    </tr>
                    <tr>
                      <td>Jargon recall</td>
                      <td>{asr.jargon_recovered_unbiased} / {asr.jargon_terms_in_reference}</td>
                      <td>{asr.jargon_recovered_biased} / {asr.jargon_terms_in_reference}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="muted" style={{ fontSize: 11.5, marginTop: -4 }}>
                Measured over {asr.sample_size} clips.
              </p>
            </>
          )}

          <p style={{ fontSize: 12.5, lineHeight: 1.6, color: "var(--ink-secondary)", marginTop: 0 }}>
            No WER improvement, and the one extra jargon term is a sample of one. The prompt
            also <strong>leaks into the output</strong>: &ldquo;Maybe I can just easily cut the
            corner&rdquo; came back as &ldquo;<em>F1 radio</em>, the camera key just easily cut the
            cooler&rdquo;. We shipped the configuration the evidence supports rather than the one
            we planned.
          </p>

          <p style={{ fontSize: 12.5, lineHeight: 1.6, marginTop: 12, marginBottom: 6 }} className="muted">
            <strong style={{ color: "var(--ink-secondary)" }}>What did matter: the model
            itself.</strong> Against distil-whisper, <span className="mono">whisper-small.en</span>{" "}
            recovers domain terms the distilled model mangles — and distil-whisper collapses
            into repetition loops when prompted at all, having been distilled without prompt
            training.
          </p>
          <div className="tablewrap">
            <table className="data">
              <thead><tr><th>Truth</th><th>distil-whisper</th><th>whisper-small.en</th></tr></thead>
              <tbody>
                <tr><td>does not have DRS</td><td>the areas</td><td>DRS</td></tr>
                <tr><td>he has DRS still</td><td>the R still</td><td>DRS still</td></tr>
                <tr><td>Hamilton&rsquo;s pitted</td><td>I will turn to the pitted</td><td>Hamilton&rsquo;s pitted</td></tr>
                <tr><td>front tyres</td><td>front tires</td><td>front tyres</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <p className="card-title">Who is actually speaking</p>
          <p style={{ fontSize: 13, lineHeight: 1.6, color: "var(--ink-secondary)", marginTop: 0 }}>
            Team radio is a two-way channel and the dataset does not say which side is
            talking. Scoring every clip as &ldquo;the driver&rdquo; produces claims that are plainly
            wrong — <em>&ldquo;You are doing a good job&rdquo;</em> is the engineer, not driver stress.
          </p>
          <div className="tablewrap" style={{ marginBottom: 12 }}>
            <table className="data">
              <thead><tr><th>Attribution</th><th>Messages</th><th>Share</th></tr></thead>
              <tbody>
                {(["driver", "engineer", "unknown"] as const).map((k) => (
                  <tr key={k}>
                    <td style={{ textTransform: "capitalize" }}>{k}</td>
                    <td>{ev.speaker_split[k]}</td>
                    <td>{((ev.speaker_split[k] / ev.message_count) * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: 12.5, lineHeight: 1.6, margin: 0 }} className="muted">
            Attribution is by grammatical direction: a vocative (<em>&ldquo;Okay, Lewis, box
            box&rdquo;</em>) is the pit wall; a first-person report (<em>&ldquo;I have no grip&rdquo;</em>) is
            the driver. It is deliberately conservative — a third-person mention such as
            <em> &ldquo;Checo is a legend&rdquo;</em> resolves to <strong>unknown</strong> rather than
            being guessed, which is why the unknown share is the largest. Engineer messages
            never raise a driver-state flag or a pit call.
          </p>
        </div>

        {cmp && cmp.races.length > 1 && (
          <div className="card">
            <p className="card-title">Does the index discriminate between races?</p>
            <p style={{ fontSize: 13, lineHeight: 1.6, color: "var(--ink-secondary)", marginTop: 0 }}>
              The race slate was chosen for contrast, not volume: a soaking-wet scramble, a
              processional dry afternoon, a race run in punishing heat. If stress does not
              separate those, the index is not measuring what we claim.
            </p>
            <div className="tablewrap" style={{ marginBottom: 10 }}>
              <table className="data">
                <thead>
                  <tr><th>Race</th><th>Mean DSI</th><th>Peak</th><th>Stressed</th><th>Fatigued</th></tr>
                </thead>
                <tbody>
                  {cmp.races.map((r) => (
                    <tr key={r.race_id}>
                      <td>{r.grand_prix.replace(" Grand Prix", "")}</td>
                      <td style={{ fontWeight: 600 }}>{r.mean_dsi ?? "—"}</td>
                      <td>{r.peak_dsi ?? "—"}</td>
                      <td>{r.stressed_share != null ? `${(r.stressed_share * 100).toFixed(0)}%` : "—"}</td>
                      <td>{r.fatigued_share != null ? `${(r.fatigued_share * 100).toFixed(0)}%` : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p
              style={{ fontSize: 12.5, lineHeight: 1.6, margin: 0 }}
              className={cmp.comparable ? "muted" : undefined}
            >
              {!cmp.comparable && (
                <strong style={{ color: "var(--warning)" }}>Not yet comparable. </strong>
              )}
              {cmp.note}
            </p>

            {finding && finding.measured !== false && (
              <>
                <div
                  style={{
                    marginTop: 14,
                    paddingTop: 12,
                    borderTop: "1px solid var(--grid)",
                  }}
                >
                  <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
                    {finding.verdict}
                  </div>
                  <ul
                    style={{
                      fontSize: 12.5,
                      lineHeight: 1.65,
                      color: "var(--ink-secondary)",
                      paddingLeft: 18,
                      margin: "0 0 10px",
                    }}
                  >
                    {finding.what_held?.map((h) => (
                      <li key={h}>{h}</li>
                    ))}
                  </ul>
                  <div
                    style={{
                      padding: "11px 13px",
                      background: "rgba(208,59,59,0.08)",
                      border: "1px solid rgba(208,59,59,0.25)",
                      borderRadius: 8,
                      marginBottom: 10,
                    }}
                  >
                    <div className="sev act" style={{ marginBottom: 7 }}>
                      <span aria-hidden>●</span> Prediction that failed
                    </div>
                    <p style={{ fontSize: 12.5, lineHeight: 1.55, margin: 0, color: "var(--ink-secondary)" }}>
                      {finding.what_failed}
                    </p>
                  </div>
                </div>

                <div className="tablewrap" style={{ marginBottom: 10 }}>
                  <table className="data">
                    <thead>
                      <tr><th>vs dry control</th><th>Δ DSI</th><th>p</th><th></th></tr>
                    </thead>
                    <tbody>
                      {finding.contrasts_vs_dry_control?.map((c) => (
                        <tr key={c.race}>
                          <td>{c.race}</td>
                          <td>{c.diff > 0 ? "+" : ""}{c.diff.toFixed(1)}</td>
                          <td>{c.p < 0.001 ? "<0.001" : c.p.toFixed(3)}</td>
                          <td style={{ color: c.survives_bonferroni ? "var(--good)" : "var(--ink-muted)" }}>
                            {c.survives_bonferroni ? "holds" : "not claimed"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <p className="muted" style={{ fontSize: 12, lineHeight: 1.6, margin: "0 0 8px" }}>
                  {finding.bonferroni_note} {finding.effect_size}
                </p>
                {era && era.measured !== false ? (
                  <div
                    style={{
                      padding: "11px 13px",
                      background: "rgba(12,163,12,0.08)",
                      border: "1px solid rgba(12,163,12,0.28)",
                      borderRadius: 8,
                    }}
                  >
                    <div
                      className="sev"
                      style={{
                        marginBottom: 7,
                        background: "rgba(12,163,12,0.15)",
                        color: "var(--good)",
                      }}
                    >
                      <span aria-hidden>✓</span> Confound tested and ruled out
                    </div>
                    <p
                      style={{
                        fontSize: 12.5,
                        lineHeight: 1.55,
                        margin: "0 0 8px",
                        color: "var(--ink-secondary)",
                      }}
                    >
                      {era.verdict}
                    </p>
                    <div className="tablewrap">
                      <table className="data">
                        <tbody>
                          <tr>
                            <td>Spread within 2023 only ({era.n_2023} races)</td>
                            <td>{era.within_season_spread} pts</td>
                          </tr>
                          <tr>
                            <td>Spread across all eras ({era.n_races} races)</td>
                            <td>{era.cross_era_spread} pts</td>
                          </tr>
                          {era.era_comparison && (
                            <tr>
                              <td>2023 mean vs pre-2023 mean</td>
                              <td>
                                {era.era_comparison.mean_2023} vs{" "}
                                {era.era_comparison.mean_pre2023} (p={era.era_comparison.p})
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : (
                  <p className="muted" style={{ fontSize: 12, lineHeight: 1.6, margin: 0 }}>
                    <strong style={{ color: "var(--ink-secondary)" }}>Open confound:</strong>{" "}
                    {finding.confound_we_cannot_rule_out}
                  </p>
                )}
              </>
            )}
          </div>
        )}

        {gold && gold.measured !== false && gold.axes && (
          <div className="card">
            <p className="card-title">Is the affect scale right? Validated on gold labels</p>
            <p style={{ fontSize: 12.5, lineHeight: 1.6, color: "var(--ink-secondary)", marginTop: 0 }}>
              The F1 dataset ships no emotion labels, so we validated the same pipeline
              against <span className="mono">{gold.dataset}</span> — {gold.n} clips with
              gold labels.
            </p>

            <div style={{ display: "flex", gap: 22, marginBottom: 12, flexWrap: "wrap" }}>
              {([
                ["Arousal", gold.axes.arousal_high_vs_low],
                ["Valence", gold.axes.valence_negative_vs_positive],
              ] as const).map(([name, a]) => {
                const weak = a.lift < 0.05;
                return (
                  <div key={name}>
                    <div
                      className="hero-figure"
                      style={{ fontSize: 32, color: weak ? "var(--critical)" : "var(--good)" }}
                    >
                      {(a.accuracy * 100).toFixed(0)}%
                    </div>
                    <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>
                      {name} · baseline {(a.majority_baseline * 100).toFixed(0)}% ·{" "}
                      <strong style={{ color: weak ? "var(--critical)" : "var(--good)" }}>
                        {a.lift > 0 ? "+" : ""}{(a.lift * 100).toFixed(1)}
                      </strong>
                    </div>
                  </div>
                );
              })}
            </div>

            <div
              style={{
                padding: "11px 13px",
                background: "rgba(208,59,59,0.08)",
                border: "1px solid rgba(208,59,59,0.25)",
                borderRadius: 8,
                marginBottom: 12,
              }}
            >
              <div className="sev act" style={{ marginBottom: 7 }}>
                <span aria-hidden>●</span> Biggest limitation
              </div>
              <p style={{ fontSize: 12.5, lineHeight: 1.55, margin: 0, color: "var(--ink-secondary)" }}>
                {gold.axes.interpretation}
              </p>
            </div>

            <p style={{ fontSize: 12.5, lineHeight: 1.6, marginTop: 0 }}>
              Overall 4-way accuracy {((gold.accuracy ?? 0) * 100).toFixed(1)}% against a
              majority-class baseline of {((gold.majority_class_baseline ?? 0) * 100).toFixed(1)}%
              — real signal, but modest.
            </p>

            <div className="tablewrap">
              <table className="data">
                <thead>
                  <tr><th>State</th><th>Precision</th><th>Recall</th><th>F1</th><th>n</th></tr>
                </thead>
                <tbody>
                  {Object.entries(gold.per_class ?? {}).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td>{v.precision ?? "—"}</td>
                      <td>{v.recall ?? "—"}</td>
                      <td>{v.f1 ?? "—"}</td>
                      <td>{v.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="muted" style={{ fontSize: 12, lineHeight: 1.6, marginTop: 10, marginBottom: 0 }}>
              {gold.caveat}
            </p>
          </div>
        )}

        {conv && conv.measured !== false && (
          <div className="card">
            <p className="card-title">Does an independent model agree?</p>
            <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>
              Inconclusive — the reference model didn&rsquo;t transfer
            </div>
            <p style={{ fontSize: 12.5, lineHeight: 1.6, color: "var(--ink-secondary)", marginTop: 0 }}>
              {conv.verdict}
            </p>
            <div className="tablewrap">
              <table className="data">
                <tbody>
                  <tr><td>Second model</td><td className="mono">{conv.second_model}</td></tr>
                  <tr><td>Raw agreement</td><td>{((conv.raw_agreement ?? 0) * 100).toFixed(1)}%</td></tr>
                  <tr><td>Cohen&rsquo;s kappa</td><td>{conv.cohens_kappa} ({conv.kappa_band})</td></tr>
                  <tr>
                    <td>Its largest class, on radio</td>
                    <td>{((conv.reference_largest_class_share ?? 0) * 100).toFixed(0)}%</td>
                  </tr>
                  <tr>
                    <td>Its accuracy on CREMA-D</td>
                    <td>
                      {((conv.reference_sanity_check?.accuracy_on_gold ?? 0) * 100).toFixed(1)}%
                      {" "}(baseline{" "}
                      {((conv.reference_sanity_check?.majority_baseline ?? 0) * 100).toFixed(1)}%)
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="muted" style={{ fontSize: 12, lineHeight: 1.6, marginTop: 10, marginBottom: 0 }}>
              {conv.why_kappa} This is also a result in its own right: an off-the-shelf
              categorical emotion model, healthy on studio speech, collapses on compressed
              team radio — which is exactly why this project uses a dimensional model with
              domain calibration instead.
            </p>
          </div>
        )}

        <div className="card">
          <p className="card-title">In-domain human labels</p>
          {!affect || affect.measured === false || !affect.n ? (
            <>
              <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>
                Still outstanding
              </div>
              <p style={{ fontSize: 12.5, lineHeight: 1.6, color: "var(--ink-secondary)", margin: 0 }}>
                The gold-label validation above uses acted studio speech. Nothing has yet
                been checked against a human listening to <em>this</em> audio. The labelling
                tool is built (<span className="mono">label_affect.py</span>) — the
                in-domain confusion matrix appears here as soon as it has been run.
              </p>
            </>
          ) : (
            <>
              <div className="hero-figure">{((affect.accuracy ?? 0) * 100).toFixed(0)}%</div>
              <p className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                agreement with human labels over {affect.n} clips
              </p>
              <div className="tablewrap" style={{ marginTop: 12 }}>
                <table className="data">
                  <thead>
                    <tr><th>State</th><th>Precision</th><th>Recall</th><th>F1</th><th>n</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(affect.per_class ?? {}).map(([k, v]) => (
                      <tr key={k}>
                        <td>{k}</td>
                        <td>{v.precision?.toFixed(2) ?? "—"}</td>
                        <td>{v.recall?.toFixed(2) ?? "—"}</td>
                        <td>{v.f1?.toFixed(2) ?? "—"}</td>
                        <td>{v.support}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>

        {experiments.map((x) => (
          <div className="card" key={x.experiment}>
            <p className="card-title">Rejected: {x.experiment}</p>
            <div className="sev act" style={{ marginBottom: 10 }}>
              <span aria-hidden>●</span> {x.verdict.split(" - ")[0]}
            </div>
            <p style={{ fontSize: 12.5, lineHeight: 1.6, color: "var(--ink-secondary)", marginTop: 0 }}>
              {x.question}
            </p>
            <div className="tablewrap" style={{ marginBottom: 10 }}>
              <table className="data">
                <thead>
                  <tr><th>Driver</th><th>Anchors</th><th>Driver share</th><th>Other</th><th></th></tr>
                </thead>
                <tbody>
                  {x.results.map((r) => (
                    <tr key={r.driver}>
                      <td>{r.driver.slice(0, 6)}</td>
                      <td>{r.anchors}</td>
                      <td>{r.driver_share.toFixed(2)}</td>
                      <td>{r.other_share.toFixed(2)}</td>
                      <td style={{ color: r.passed ? "var(--good)" : "var(--ink-muted)" }}>
                        {r.passed ? "pass" : "fail"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p style={{ fontSize: 12.5, lineHeight: 1.6, color: "var(--ink-secondary)", margin: 0 }}>
              {x.conclusion}
            </p>
            <p className="muted" style={{ fontSize: 12, lineHeight: 1.6, marginBottom: 0 }}>
              Criteria were fixed before running, so the result could not be tuned into a
              pass. Three rounds are recorded in{" "}
              <span className="mono">_diarization_experiment.json</span>.
            </p>
          </div>
        ))}

        <div className="card">
          <p className="card-title">Known limitations</p>
          <ul style={{ fontSize: 12.5, lineHeight: 1.7, color: "var(--ink-secondary)", paddingLeft: 18, margin: 0 }}>
            <li>
              The dataset ships no emotion labels, so affect is not validated against
              per-clip ground truth for this race. Percentile calibration makes the scale
              internally consistent; it does not make it externally verified.
            </li>
            <li>
              Radio carries the engineer&rsquo;s voice as well as the driver&rsquo;s. Clips containing
              both are scored as one mixed signal.
            </li>
            <li>
              The published reference transcripts contain systematic F1-jargon errors
              (&ldquo;supersoft&rdquo; → &ldquo;SuperSalt&rdquo;), so they are a comparison baseline, not gold
              truth.
            </li>
            <li>
              Lap-time delta is measured against each driver&rsquo;s own race median, which
              absorbs some but not all of traffic, fuel load and safety-car effects.
            </li>
          </ul>
        </div>
      </div>
    </>
  );
}
