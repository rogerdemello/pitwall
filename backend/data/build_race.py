"""Precompute a whole race: every radio message, analysed and placed on its lap.

Deliberately split into two stages.

  Stage 1 (this file, expensive): run the models over every clip and write the
  RAW outputs to disk. ~25s per clip on CPU, so a race is roughly an hour.

  Stage 2 (calibrate.py, cheap): turn raw outputs into DSI, states and pit
  calls. Because stage 1 is cached, the interpretation layer can be re-tuned in
  seconds instead of re-running an hour of inference.

That split is what makes iteration on the fusion thresholds practical at all.

Usage:
    python backend/data/build_race.py 2021_Abu_Dhabi_Grand_Prix
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import race_data  # noqa: E402
from pipeline import asr, prosody, sentiment  # noqa: E402

CLIP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clips")
RAW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")


def _is_done(rec: dict) -> bool:
    """Is this cached record a finished clip *from the model now selected*?

    The model check is not defensive tidiness; without it the GPU rebuild is a
    silent no-op. `backend/raw/*.raw.json` is committed, so the Colab notebook's
    `git clone` puts a full set of v1 records on the runtime before a single
    model is loaded. Resume then sees every clip as already done, the journal
    stays empty, the "long run" finishes in seconds, and the only symptom is a
    tarball that looks small.

    Cached output from a different model is not a checkpoint, it is the previous
    experiment. v1 records predate this field entirely, so they read as None and
    are correctly refused.
    """
    return ("id" in rec and "error" not in rec
            and rec.get("asr_model_id") == asr.MODEL_ID)


def _read_journal(path: str) -> dict[str, dict]:
    """Completed records from an append-only journal.

    A partial final line is the expected shape of an interrupted write, so it is
    skipped rather than treated as corruption. Failed clips are not returned, so
    a rerun retries them.
    """
    if not os.path.exists(path):
        return {}
    done: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail from a killed process
            if _is_done(rec):
                done[rec["id"]] = rec
    return done


def _append(journal, record: dict) -> None:
    """One line, flushed and fsynced, so a hard kill cannot lose it."""
    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
    journal.flush()
    try:
        os.fsync(journal.fileno())
    except OSError:
        pass  # some filesystems (and Drive mounts) refuse fsync; the flush stands


def build(race_id: str, limit: int | None = None,
          join_laps: bool = True) -> str:
    clip_dir = os.path.join(CLIP_ROOT, race_id)
    manifest = json.load(open(os.path.join(clip_dir, "manifest.json"), encoding="utf-8"))
    if limit:
        manifest = manifest[:limit]

    os.makedirs(RAW_ROOT, exist_ok=True)
    out_path = os.path.join(RAW_ROOT, f"{race_id}.raw.json")
    journal_path = os.path.join(RAW_ROOT, f"{race_id}.raw.jsonl")

    # Resume support: an hour-long job should never be restarted from zero.
    #
    # The journal is append-only, and that is the point. This used to rewrite the
    # entire results list to one path every five clips, so a disconnect during a
    # write truncated the file and lost the whole run - which is exactly the
    # failure mode a Colab session produces when it drops. One line per clip
    # cannot be corrupted by an interrupted write; at worst the last line is
    # partial and is skipped on reload.
    # Resume only across records this same model produced - see _is_done.
    done: dict[str, dict] = _read_journal(journal_path)
    if not done and os.path.exists(out_path):
        prev = json.load(open(out_path, encoding="utf-8")).get("messages", [])
        # Failed clips are deliberately not treated as done, so a rerun retries them.
        done = {r["id"]: r for r in prev if _is_done(r)}
        stale = len(prev) - len(done)
        if stale:
            print(f"ignoring {stale} record(s) from another model; "
                  f"rebuilding them with {asr.MODEL_ID}")
    if done:
        print(f"resuming: {len(done)} already processed")
    journal = open(journal_path, "a", encoding="utf-8")

    # The lap join here is redundant and optional.
    #
    # calibrate.py recomputes it from scratch (`m["lap"] = lap_for_timestamp(...)`),
    # so whatever stage 1 writes is overwritten in stage 2 regardless. It stays
    # by default because it is free when the FastF1 cache is warm and makes the
    # raw file readable on its own.
    #
    # It must be skippable because stage 1 now runs on a GPU box that has no
    # cache, and FastF1's live-timing API no longer serves older seasons - a
    # 2019 session there returns SessionNotAvailableError for every endpoint and
    # every message lands on no lap. That is harmless, since stage 2 fixes it on
    # a machine that has the cache, but spending the API calls and the wall time
    # to produce a result that is thrown away is not.
    session = None
    if join_laps:
        print(f"loading FastF1 session for {race_id}...")
        try:
            session = race_data.load_session(race_id)
        except Exception as e:
            print(f"  !! FastF1 unavailable ({type(e).__name__}: {e})")
            print("     continuing without the lap join - calibrate.py redoes it")
            session = None
    else:
        print("skipping the lap join (calibrate.py recomputes it in stage 2)")

    # Cache lap frames per car number rather than per message.
    lap_cache: dict[str, object] = {}

    def laps_for(num: str):
        if session is None:
            return None
        if num not in lap_cache:
            lap_cache[num] = race_data.driver_laps(session, num)
        return lap_cache[num]

    results = []
    t_start = time.perf_counter()

    for i, m in enumerate(manifest, 1):
        if m["id"] in done:
            results.append(done[m["id"]])
            continue

        path = os.path.join(clip_dir, m["audio_file"])
        try:
            audio = asr.load_audio(path)
            tr = asr.transcribe(audio)
            af = prosody.analyse(audio)
            se = sentiment.analyse(tr.text)
            frames = laps_for(m["racing_number"])
            lap = (race_data.lap_for_timestamp(frames, m["message_timestamp"])
                   if frames is not None else None)

            record = {
                **m,
                "transcript": tr.text,
                "duration_s": tr.duration_s,
                "asr_elapsed_s": tr.elapsed_s,
                "arousal": af.arousal,
                "valence": af.valence,
                "dominance": af.dominance,
                "peak_arousal": af.peak_arousal,
                "min_valence": af.min_valence,
                "text_label": se.label,
                "text_polarity": se.polarity,
                "text_negative": se.negative,
                "text_positive": se.positive,
                "lap": lap.__dict__ if lap is not None else None,
                # v2 additions. Every aggregation choice becomes a stage-2
                # decision instead of being baked into an hour of inference.
                "windows": af.windows,
                "arousal_range": af.arousal_range,
                "arousal_slope": af.arousal_slope,
                "voiced_fraction": af.voiced_fraction,
                "speech_s": round(af.speech_s, 3),
                "window_scores": [w.to_dict() for w in af.window_scores],
                "asr_model_id": tr.model_id,
                "asr_backend": tr.backend,
                "asr_segments": [s.to_dict() for s in tr.segments],
                "asr_language": tr.language,
                "asr_language_probability": tr.language_probability,
                "no_speech": tr.no_speech,
            }
            results.append(record)
        except Exception as e:  # one bad clip must not lose the whole run
            print(f"  !! {m['id']}: {type(e).__name__}: {e}")
            record = {**m, "error": f"{type(e).__name__}: {e}"}
            results.append(record)

        _append(journal, record)

        if i % 5 == 0 or i == len(manifest):
            rate = (time.perf_counter() - t_start) / max(1, i - len(done))
            left = rate * (len(manifest) - i)
            print(f"  {i}/{len(manifest)}  ~{left/60:.1f} min remaining", flush=True)

    journal.close()
    json.dump({"race_id": race_id, "messages": results},
              open(out_path, "w", encoding="utf-8"), indent=1)

    ok = sum(1 for r in results if "error" not in r)
    in_race = sum(1 for r in results if r.get("lap", {}).get("in_race"))
    print(f"\ndone: {ok}/{len(results)} analysed, {in_race} landed on a race lap")
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    race = sys.argv[1] if len(sys.argv) > 1 else "2021_Abu_Dhabi_Grand_Prix"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    build(race, lim)
