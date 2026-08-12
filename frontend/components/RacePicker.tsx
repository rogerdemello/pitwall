"use client";

/* Race selector.
 *
 * The corpus is only useful if you can actually reach it - the pages used to
 * hardcode Abu Dhabi, which made every other race invisible in the product.
 * Kept as one row above everything it scopes, so both screens read the same way.
 */

import { useEffect, useState } from "react";
import { getRaces } from "@/lib/api";

export interface RaceOption {
  race_id: string;
  grand_prix: string;
  message_count: number;
}

export function useRaces(preferred: string) {
  const [races, setRaces] = useState<RaceOption[]>([]);
  const [raceId, setRaceId] = useState<string>(preferred);

  useEffect(() => {
    getRaces()
      .then((r) => {
        setRaces(r.races);
        // Keep the preferred race if it exists, otherwise fall back to the first
        // one that is actually built.
        if (r.races.length && !r.races.some((x) => x.race_id === preferred)) {
          setRaceId(r.races[0].race_id);
        }
      })
      .catch(() => setRaces([]));
  }, [preferred]);

  return { races, raceId, setRaceId };
}

export default function RacePicker({
  races,
  raceId,
  onChange,
}: {
  races: RaceOption[];
  raceId: string;
  onChange: (id: string) => void;
}) {
  if (races.length <= 1) return null;

  return (
    <div className="filter-row" style={{ marginBottom: 14 }}>
      <span
        className="muted"
        style={{ fontSize: 10.5, letterSpacing: "0.09em", marginRight: 2 }}
      >
        RACE
      </span>
      {races.map((r) => (
        <button
          key={r.race_id}
          className={`chip ${r.race_id === raceId ? "on" : ""}`}
          onClick={() => onChange(r.race_id)}
        >
          {r.grand_prix.replace(" Grand Prix", "")}
          <span style={{ opacity: 0.65, marginLeft: 6, fontSize: 11 }}>
            {r.message_count}
          </span>
        </button>
      ))}
    </div>
  );
}
