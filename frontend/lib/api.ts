export const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

/* Static mode.
 *
 * Docker Spaces need a paid plan; Static Spaces are free. Almost nothing here
 * needs a server - Replay and Evidence read precomputed JSON, and only Live
 * Analysis runs a model - so a build can point at snapshotted files instead.
 *
 * build_static_site.py freezes every GET endpoint under the same shape:
 * /api/<anything> becomes /data/<anything>.json. Keeping that mapping uniform
 * means one rewrite here rather than a lookup table that drifts as soon as
 * someone adds an endpoint. */
export const STATIC_MODE = process.env.NEXT_PUBLIC_STATIC === "1";

function resolve(path: string): string {
  if (!STATIC_MODE) return `${API}${path}`;
  return `/data${path.slice("/api".length)}.json`;
}

export interface LapContext {
  lap_number: number | null;
  lap_time_s: number | null;
  compound: string | null;
  tyre_life: number | null;
  stint: number | null;
  position: number | null;
  delta_to_median_s: number | null;
  in_race: boolean;
  note: string;
}

export interface Recommendation {
  severity: "info" | "watch" | "act";
  headline: string;
  detail: string;
  evidence: string[];
}

export interface Message {
  id: string;
  driver_id: string;
  driver_code: string;
  driver_name: string;
  racing_number: string;
  timestamp: string;
  audio_file: string;
  transcript: string;
  reference_transcription: string;
  duration_s: number;
  lap: LapContext;
  speaker: "driver" | "engineer" | "unknown";
  speaker_reason: string;
  dsi: number;
  state: "Calm" | "Energised" | "Stressed" | "Fatigued";
  descriptor: string;
  arousal_pct: number;
  valence_pct: number;
  dominance_pct: number;
  arousal_raw: number;
  valence_raw: number;
  text_polarity: number;
  incongruence: number;
  suppressed_stress: boolean;
  note: string;
  recommendation: Recommendation | null;
}

export interface DriverSummary {
  driver_id: string;
  code: string;
  name: string;
  racing_number: string;
  message_count: number;
  in_race_count: number;
  mean_dsi: number | null;
  peak_dsi: number | null;
  suppressed_count: number;
}

export interface LapPoint {
  lap: number;
  seconds: number;
  compound: string | null;
  tyre_life: number | null;
  position: number | null;
}

export interface Race {
  race_id: string;
  grand_prix: string;
  session_date: string;
  message_count: number;
  in_race_count: number;
  drivers: DriverSummary[];
  lap_traces: Record<string, LapPoint[]>;
  calibration: Record<string, Record<string, number>>;
  messages: Message[];
}

export interface TercileRow {
  driver: string;
  n: number;
  calm_mean_delta_s: number;
  stressed_mean_delta_s: number;
  gap_s: number;
}

export interface StressVsPace {
  n: number;
  excluded_non_racing_laps: number;
  pooled_r: number | null;
  per_driver: { driver: string; n: number; r: number }[];
  tercile: {
    drivers: TercileRow[];
    mean_gap_s: number;
    drivers_slower_when_stressed: number;
    drivers_total: number;
  } | null;
}

export interface Evidence {
  race_id: string;
  message_count: number;
  on_lap_count: number;
  join_rate: number;
  dsi: { min: number; max: number; mean: number };
  suppressed_stress_count: number;
  suppressed_stress_eligible: number;
  speaker_split: { driver: number; engineer: number; unknown: number };
  dsi_vs_lap_delta_correlation: number | null;
  correlation_n: number;
  stress_vs_pace: StressVsPace;
  calibration: Record<string, Record<string, number>>;
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(resolve(path), { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

export interface AsrEval {
  measured?: false;
  race_id: string;
  sample_size?: number;
  wer_unbiased?: number;
  wer_biased?: number;
  wer_delta?: number;
  jargon_terms_in_reference?: number;
  jargon_recovered_unbiased?: number;
  jargon_recovered_biased?: number;
  note?: string;
}

export interface AffectEval {
  measured?: false;
  race_id: string;
  n?: number;
  accuracy?: number;
  confusion?: Record<string, Record<string, number>>;
  per_class?: Record<
    string,
    { precision: number | null; recall: number | null; f1: number | null; support: number }
  >;
}

export interface DiarizationExperiment {
  experiment: string;
  question: string;
  motivation: string;
  method: string;
  verdict: string;
  conclusion: string;
  what_we_kept: string;
  what_would_fix_it: string;
  model: string;
  rounds: { round: number; change: string; result: string; why: string }[];
  results: {
    driver: string;
    clips: number;
    windows: number;
    anchors: number;
    cosine_gap: number;
    driver_share: number;
    other_share: number;
    passed: boolean;
    note?: string;
  }[];
  pre_registered_criteria: Record<string, unknown>;
}

export interface RaceComparison {
  races: {
    race_id: string;
    grand_prix: string;
    message_count: number;
    mean_dsi: number | null;
    peak_dsi: number | null;
    stressed_share: number | null;
    fatigued_share: number | null;
    suppressed: number;
  }[];
  comparable: boolean;
  calibration_sources: string[];
  note: string;
}

export interface CorpusFinding {
  measured?: false;
  verdict?: string;
  what_held?: string[];
  what_failed?: string;
  effect_size?: string;
  confound_we_cannot_rule_out?: string;
  honest_summary?: string;
  bonferroni_note?: string;
  contrasts_vs_dry_control?: {
    race: string;
    diff: number;
    z: number;
    p: number;
    survives_bonferroni: boolean;
  }[];
}

export interface CorpusAnalysis {
  measured?: false;
  races?: string[];
  messages_pooled?: number;
  verdict?: string;
  caveat?: string;
  stress_vs_pace?: {
    n: number;
    pooled_r: number | null;
    excluded_non_racing_laps: number;
    tercile: {
      mean_gap_s: number;
      drivers_slower_when_stressed: number;
      drivers_total: number;
      sign_test_p: number | null;
    } | null;
  };
  lag?: {
    by_lag: Record<string, { n: number; r: number | null; p: number | null }>;
    best_lag: string | null;
    best_r: number | null;
    best_p: number | null;
    predictive: boolean;
    bonferroni_alpha: number;
    caveat: string;
  };
}

export interface CorpusAsr {
  measured?: false;
  races?: {
    race_id: string;
    n: number;
    wer_unbiased: number;
    wer_biased: number;
    jargon_total: number;
    jargon_unbiased: number;
    jargon_biased: number;
  }[];
  race_count?: number;
  mean_wer_unbiased?: number;
  mean_wer_biased?: number;
  races_where_prompting_hurt?: number;
  jargon_unbiased?: number;
  jargon_biased?: number;
  jargon_total?: number;
  conclusion?: string;
}

export interface AxisScore {
  accuracy: number;
  majority_baseline: number;
  lift: number;
  n: number;
}

export interface GoldAffect {
  measured?: false;
  dataset?: string;
  n?: number;
  accuracy?: number;
  majority_class_baseline?: number;
  beats_baseline?: boolean;
  confusion?: Record<string, Record<string, number>>;
  per_class?: Record<
    string,
    { precision: number | null; recall: number | null; f1: number | null; support: number }
  >;
  axes?: {
    arousal_high_vs_low: AxisScore;
    valence_negative_vs_positive: AxisScore;
    near_chance_axes: string[];
    interpretation: string;
  };
  verdict?: string;
  caveat?: string;
}

export interface Convergent {
  measured?: false;
  second_model?: string;
  n?: number;
  raw_agreement?: number;
  cohens_kappa?: number | null;
  kappa_band?: string;
  reference_collapsed_on_radio?: boolean;
  reference_largest_class_share?: number;
  reference_sanity_check?: {
    ran: boolean;
    n?: number;
    accuracy_on_gold?: number;
    majority_baseline?: number;
    largest_class_share?: number;
  };
  verdict?: string;
  why_kappa?: string;
}

export interface EraAnalysis {
  measured?: false;
  n_races?: number;
  n_2023?: number;
  within_season_spread?: number | null;
  cross_era_spread?: number;
  contrasts?: {
    comparison: string;
    diff: number;
    z: number;
    p: number;
    bonferroni_alpha: number;
    survives: boolean;
  }[];
  era_comparison?: {
    mean_2023: number;
    mean_pre2023: number;
    diff: number;
    p: number;
    systematic_era_effect: boolean;
  } | null;
  verdict?: string;
}

export const getEraAnalysis = () => get<EraAnalysis>("/api/era-analysis");
export const getGoldAffect = () => get<GoldAffect>("/api/gold-affect");
export const getConvergent = () => get<Convergent>("/api/convergent");
export const getCorpusAsr = () => get<CorpusAsr>("/api/corpus-asr");
export const getCorpusAnalysis = () => get<CorpusAnalysis>("/api/corpus-analysis");
export const getCorpusFinding = () => get<CorpusFinding>("/api/corpus-finding");
export const getComparison = () => get<RaceComparison>("/api/compare");
export const getAsrEval = (id: string) => get<AsrEval>(`/api/evidence/${id}/asr`);
export const getAffectEval = (id: string) => get<AffectEval>(`/api/evidence/${id}/affect`);
export const getExperiments = () =>
  get<{ experiments: DiarizationExperiment[] }>("/api/experiments");

export const getRaces = () =>
  get<{ races: { race_id: string; grand_prix: string; message_count: number }[] }>(
    "/api/races",
  );
export const getRace = (id: string) => get<Race>(`/api/race/${id}`);
export const getEvidence = (id: string) => get<Evidence>(`/api/evidence/${id}`);
export const audioUrl = (raceId: string, file: string) =>
  STATIC_MODE
    ? `/audio/${raceId}/${encodeURIComponent(file)}`
    : `${API}/api/audio/${raceId}/${encodeURIComponent(file)}`;

/** Which races shipped audio. Only the showcase race does in static mode -
 *  all twelve would be 327 MB, and the Evidence screens need none of it. */
export const getAudioManifest = () =>
  get<{ races_with_audio: string[] }>("/api/audio-manifest").catch(
    () => ({ races_with_audio: [] as string[] }),
  );

export async function analyzeClip(file: File, raceId: string) {
  if (STATIC_MODE) {
    throw new Error(
      "Live Analysis needs the model backend, which a static Space cannot host. " +
        "Run the app locally to use it - the Race Replay here is the same pipeline's output.",
    );
  }
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/api/analyze?race_id=${encodeURIComponent(raceId)}`, {
    method: "POST",
    body: fd,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/** Status colour for a DSI value. Always paired with the number or a word -
 *  colour never carries the meaning on its own. */
export function dsiColor(dsi: number): string {
  if (dsi >= 75) return "var(--critical)";
  if (dsi >= 60) return "var(--serious)";
  if (dsi >= 45) return "var(--warning)";
  return "var(--good)";
}
