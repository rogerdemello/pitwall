/**
 * Values that were hardcoded in three separate files.
 *
 * `DEFAULT_RACE` / `RACE` appeared verbatim in app/page.tsx, app/live/page.tsx
 * and app/evidence/page.tsx, so changing the showcase race meant finding all
 * three.
 */

/** The race the app opens on. Abu Dhabi 2021 is the only one with audio in the
 *  static export - all twelve would be 327 MB and the Evidence screens need
 *  none of it. */
export const SHOWCASE_RACE = "2021_Abu_Dhabi_Grand_Prix";

/** The four states the affect plane produces. */
export const STATES = ["Calm", "Energised", "Stressed", "Fatigued"] as const;
export type State = (typeof STATES)[number];

/** Who is speaking, as `speaker.py` reports it. Roughly half the corpus is
 *  `unknown`, and that is a measured limitation rather than a bug to hide. */
export const SPEAKERS = ["driver", "engineer", "unknown"] as const;
export type Speaker = (typeof SPEAKERS)[number];

/** Recommendation severities, ordered by urgency. */
export const SEVERITIES = ["info", "watch", "act"] as const;
export type Severity = (typeof SEVERITIES)[number];

/** Extra boolean filters, which are cheaper as flags than as separate params. */
export const FLAGS = ["suppressed", "onlap", "hasrec"] as const;
export type Flag = (typeof FLAGS)[number];

/** Rows rendered before the table asks for a narrower filter. 2,042 messages
 *  is fine to filter and far too many to render. */
export const MAX_TABLE_ROWS = 200;

/** The valence axis of the affect model scores at chance against gold labels at
 *  the 0.5 split. The model does rank valence (AUC 0.687) - the boundary is
 *  what loses it - but the shipped split is the median one, so a state's
 *  calm/stressed *direction* is much less reliable than its high/low
 *  *activation*. Anything that renders a state should say so. */
export const VALENCE_CAVEAT =
  "The valence axis scores at chance against gold labels at this boundary, so " +
  "the calm-versus-stressed direction is far less reliable than the high-" +
  "versus-low activation.";
