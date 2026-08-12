"""PIT WALL API.

Two very different paths, on purpose:

  /race/*    serves precomputed race JSON. Instant, no model inference, works
             with the network unplugged. This is what the demo runs on.
  /analyze   runs the full pipeline on an uploaded clip. Slower, but it is the
             proof that the replay is a real analysis and not a recording.
"""

from __future__ import annotations

import io
import json
import os
import threading
from contextlib import asynccontextmanager

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pipeline import analysis, asr, fusion, prosody, sentiment
from pipeline.calibration import Calibrator

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "races")
CLIPS = os.path.join(HERE, "clips")

_WARM = False


def _warm_models() -> None:
    global _WARM
    try:
        silence = np.zeros(asr.SAMPLE_RATE * 2, dtype=np.float32)
        asr.transcribe(silence)
        prosody.analyse(silence)
        sentiment.analyse("warm up")
        _WARM = True
        print("[startup] models warm")
    except Exception as e:  # never let warm-up affect serving
        print(f"[startup] warm-up skipped: {type(e).__name__}: {e}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the models in the background, without holding up the server.

    Loading them on the first request cost ~20s, which is a bad thing to
    discover in front of judges. But doing it *inside* the lifespan was worse:
    uvicorn accepts no connections until lifespan yields, so in a fresh
    container - where the weights must be downloaded first - the whole app was
    unreachable for about seven minutes.

    Nothing about the Race Replay needs a model: it reads precomputed JSON. Only
    Live Analysis does. So warm-up runs on a background thread, the app serves
    immediately, and /api/health reports `models_warm` so a caller can tell
    whether the first analysis will be slow.
    """
    threading.Thread(target=_warm_models, name="warm-models", daemon=True).start()
    yield


app = FastAPI(title="PIT WALL", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # Next.js falls back to 3001+ when 3000 is taken; allow the usual dev range.
    allow_origins=[f"http://{h}:{p}" for h in ("localhost", "127.0.0.1")
                   for p in (3000, 3001, 3002)],
    allow_methods=["*"],
    allow_headers=["*"],
)

_race_cache: dict[str, dict] = {}
_cal_cache: dict[str, Calibrator] = {}


def _load_race(race_id: str) -> dict:
    if race_id not in _race_cache:
        path = os.path.join(RACES, f"{race_id}.json")
        if not os.path.exists(path):
            raise HTTPException(404, f"race not built: {race_id}")
        _race_cache[race_id] = json.load(open(path, encoding="utf-8"))
    return _race_cache[race_id]


def _calibrator(race_id: str) -> Calibrator | None:
    if race_id not in _cal_cache:
        path = os.path.join(RACES, f"{race_id}.calibration.json")
        _cal_cache[race_id] = Calibrator.from_json(path) if os.path.exists(path) else None
    return _cal_cache[race_id]


def _is_race_file(fn: str) -> bool:
    """Is this a race file, as opposed to something else living in races/?

    Two kinds of neighbour share the directory and must never be parsed as races:
      - sidecars with a dotted suffix (.calibration.json, .asr_eval.json, ...)
      - corpus-level artifacts prefixed with an underscore
        (_pooled.calibration.json, _diarization_experiment.json)
    Missing the underscore case here made /api/races return a 500 once the
    diarization experiment was written.
    """
    if not fn.endswith(".json") or fn.startswith("_"):
        return False
    return "." not in fn[:-5]


@app.get("/api/races")
def list_races():
    """Which races have been precomputed."""
    if not os.path.isdir(RACES):
        return {"races": []}
    out = []
    for fn in sorted(os.listdir(RACES)):
        if not _is_race_file(fn):
            continue
        r = _load_race(fn[:-5])
        out.append({
            "race_id": r["race_id"],
            "grand_prix": r["grand_prix"],
            "session_date": r["session_date"],
            "message_count": r["message_count"],
            "driver_count": len(r["drivers"]),
        })
    return {"races": out}


@app.get("/api/race/{race_id}")
def get_race(race_id: str):
    """Full precomputed race: every message, state and pit call."""
    return _load_race(race_id)


@app.get("/api/race/{race_id}/driver/{driver_id}")
def get_driver(race_id: str, driver_id: str):
    """One driver's timeline, plus their lap trace for charting."""
    race = _load_race(race_id)
    msgs = [m for m in race["messages"] if m["driver_id"] == driver_id]
    if not msgs:
        raise HTTPException(404, f"no messages for {driver_id}")
    meta = next((d for d in race["drivers"] if d["driver_id"] == driver_id), None)
    return {"race_id": race_id, "driver": meta, "messages": msgs}


@app.get("/api/audio/{race_id}/{filename}")
def get_audio(race_id: str, filename: str):
    """Serve a radio clip. Path components are validated, not trusted."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "bad filename")
    path = os.path.join(CLIPS, race_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "clip not found")
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/api/evidence/{race_id}")
def get_evidence(race_id: str):
    """Numbers behind the claims, for the Evidence screen."""
    race = _load_race(race_id)
    msgs = race["messages"]
    on_lap = [m for m in msgs if m["lap"].get("in_race")]

    # Does stress actually track lap-time loss? Reported either way, three ways.
    stress_pace = analysis.analyse(msgs)

    # Engineer transmissions can never carry the flag - we clear it in stage 2,
    # because the pit wall's tone is not a claim about the driver. Reporting the
    # rate over all messages would therefore quietly understate it, so the
    # eligible denominator is published alongside the count.
    eligible = [m for m in msgs if m.get("speaker") != "engineer"]
    flagged = sum(1 for m in msgs if m["suppressed_stress"])

    dsis = [m["dsi"] for m in msgs]
    return {
        "race_id": race_id,
        "message_count": len(msgs),
        "on_lap_count": len(on_lap),
        "join_rate": round(len(on_lap) / len(msgs), 4) if msgs else 0,
        "dsi": {
            "min": min(dsis), "max": max(dsis),
            "mean": round(sum(dsis) / len(dsis), 1),
        },
        "suppressed_stress_count": flagged,
        "suppressed_stress_eligible": len(eligible),
        "speaker_split": {
            "driver": sum(1 for m in msgs if m.get("speaker") == "driver"),
            "engineer": sum(1 for m in msgs if m.get("speaker") == "engineer"),
            "unknown": sum(1 for m in msgs if m.get("speaker") == "unknown"),
        },
        "dsi_vs_lap_delta_correlation": stress_pace["pooled_r"],
        "correlation_n": stress_pace["n"],
        "stress_vs_pace": stress_pace,
        "calibration": race["calibration"],
    }


@app.get("/api/compare")
def compare_races():
    """Does the index discriminate between races, or just produce numbers?

    This is the single best test of whether DSI measures anything real. The slate
    was picked for contrast - a soaking-wet scramble (Turkey 2020), a
    processional dry afternoon (Monza 2023), a race run in punishing heat (Qatar
    2023). If stress does not separate those, the index is not working, and that
    is worth saying plainly.

    Note this is only meaningful once the pooled calibration is in use. Fitted
    per-race, every race centres on 50.0 by construction and this endpoint would
    show a flat line no matter what the audio contained.
    """
    if not os.path.isdir(RACES):
        return {"races": [], "comparable": False}

    rows = []
    sources = set()
    for fn in sorted(os.listdir(RACES)):
        if not _is_race_file(fn):
            continue
        r = _load_race(fn[:-5])
        msgs = r["messages"]
        driver_msgs = [m for m in msgs if m.get("speaker") != "engineer"]
        dsis = [m["dsi"] for m in driver_msgs] or [m["dsi"] for m in msgs]
        states = {}
        for m in driver_msgs:
            states[m["state"]] = states.get(m["state"], 0) + 1
        sources.add(r.get("calibration_source", "per-race"))
        rows.append({
            "race_id": r["race_id"],
            "grand_prix": r["grand_prix"],
            "message_count": len(msgs),
            "mean_dsi": round(sum(dsis) / len(dsis), 1) if dsis else None,
            "peak_dsi": max(dsis) if dsis else None,
            "stressed_share": round(states.get("Stressed", 0) / len(driver_msgs), 3)
            if driver_msgs else None,
            "fatigued_share": round(states.get("Fatigued", 0) / len(driver_msgs), 3)
            if driver_msgs else None,
            "suppressed": sum(1 for m in msgs if m["suppressed_stress"]),
        })

    rows.sort(key=lambda r: -(r["mean_dsi"] or 0))
    comparable = sources == {"pooled"} and len(rows) > 1
    return {
        "races": rows,
        "comparable": comparable,
        "calibration_sources": sorted(sources),
        "note": (
            "Comparable across races."
            if comparable
            else "Races are calibrated per-race, so every mean sits at 50 by "
                 "construction. Run pool_calibration.py then rebuild to compare."
        ),
    }


def _sidecar(race_id: str, suffix: str) -> dict | None:
    """Load an eval artifact written beside the race file, if it exists."""
    path = os.path.join(RACES, f"{race_id}.{suffix}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


@app.get("/api/evidence/{race_id}/asr")
def get_asr_eval(race_id: str):
    """ASR ablation results, or an explicit 'not measured' rather than silence.

    These numbers used to be typed by hand into the Evidence page, which meant
    re-running the eval left the page quietly lying. Now the page renders
    whatever eval_asr.py last wrote, or says nothing has been measured.
    """
    data = _sidecar(race_id, "asr_eval")
    return data or {"measured": False, "race_id": race_id}


@app.get("/api/evidence/{race_id}/affect")
def get_affect_eval(race_id: str):
    """Confusion matrix from the hand-labelled set, once labelling has been run."""
    data = _sidecar(race_id, "affect_eval")
    return data or {"measured": False, "race_id": race_id}


@app.get("/api/experiments")
def get_experiments():
    """Experiments we ran and rejected. Kept visible on purpose."""
    path = os.path.join(RACES, "_diarization_experiment.json")
    out = []
    if os.path.exists(path):
        out.append(json.load(open(path, encoding="utf-8")))
    return {"experiments": out}


@app.get("/api/corpus-finding")
def get_corpus_finding():
    """The cross-race result, with the prediction that failed left in."""
    path = os.path.join(RACES, "_corpus_finding.json")
    if not os.path.exists(path):
        return {"measured": False}
    return json.load(open(path, encoding="utf-8"))


@app.get("/api/era-analysis")
def get_era_analysis():
    """Was race separation real, or just recording era? Tested, not assumed."""
    path = os.path.join(RACES, "_era_analysis.json")
    if not os.path.exists(path):
        return {"measured": False}
    return json.load(open(path, encoding="utf-8"))


@app.get("/api/gold-affect")
def get_gold_affect():
    """Affect validated against CREMA-D's gold labels.

    The one claim that had no external check now has one. The axis breakdown is
    the part that matters: arousal is recovered, valence is not.
    """
    path = os.path.join(RACES, "_gold_affect_eval.json")
    if not os.path.exists(path):
        return {"measured": False}
    return json.load(open(path, encoding="utf-8"))


@app.get("/api/convergent")
def get_convergent():
    """Agreement with an independent emotion model, and whether it was usable."""
    path = os.path.join(RACES, "_convergent_eval.json")
    if not os.path.exists(path):
        return {"measured": False}
    return json.load(open(path, encoding="utf-8"))


@app.get("/api/corpus-asr")
def get_corpus_asr():
    """The ASR ablation aggregated over every race that has been measured.

    One race said domain prompting did not help. Six races say it actively hurts,
    which is a much harder result to wave away - and it is computed from the
    sidecar files rather than stored, so it can never drift from them.
    """
    import statistics

    rows = []
    for fn in sorted(os.listdir(RACES)) if os.path.isdir(RACES) else []:
        if not fn.endswith(".asr_eval.json"):
            continue
        d = json.load(open(os.path.join(RACES, fn), encoding="utf-8"))
        if d.get("wer_unbiased") is None:
            continue
        rows.append({
            "race_id": d["race_id"],
            "n": d["sample_size"],
            "wer_unbiased": d["wer_unbiased"],
            "wer_biased": d["wer_biased"],
            "jargon_total": d["jargon_terms_in_reference"],
            "jargon_unbiased": d["jargon_recovered_unbiased"],
            "jargon_biased": d["jargon_recovered_biased"],
        })

    if not rows:
        return {"measured": False}

    worse = sum(1 for r in rows if r["wer_biased"] > r["wer_unbiased"])
    return {
        "races": rows,
        "race_count": len(rows),
        "mean_wer_unbiased": round(statistics.fmean(r["wer_unbiased"] for r in rows), 4),
        "mean_wer_biased": round(statistics.fmean(r["wer_biased"] for r in rows), 4),
        "races_where_prompting_hurt": worse,
        "jargon_unbiased": sum(r["jargon_unbiased"] for r in rows),
        "jargon_biased": sum(r["jargon_biased"] for r in rows),
        "jargon_total": sum(r["jargon_total"] for r in rows),
        "conclusion": (
            f"Domain prompting made word error rate worse in {worse} of {len(rows)} "
            "races and barely moved jargon recall. It ships disabled."
        ),
    }


@app.get("/api/corpus-analysis")
def get_corpus_analysis():
    """Stress-vs-pace and the lag question, pooled over the whole corpus.

    Per-race these are underpowered; pooled they finally have enough
    observations to answer, and the answer is reported whichever way it lands.
    """
    path = os.path.join(RACES, "_corpus_analysis.json")
    if not os.path.exists(path):
        return {"measured": False}
    return json.load(open(path, encoding="utf-8"))


MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # a radio clip is tens of KB; 25 MB is generous


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), race_id: str = "2021_Abu_Dhabi_Grand_Prix"):
    """Run the full pipeline on an uploaded clip."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"clip too large ({len(data) // 1024 // 1024} MB); limit is 25 MB"
        )
    try:
        audio, _ = librosa.load(io.BytesIO(data), sr=asr.SAMPLE_RATE, mono=True)
    except Exception as e:
        raise HTTPException(400, f"could not decode audio: {e}")

    tr = asr.transcribe(audio)
    af = prosody.analyse(audio)
    se = sentiment.analyse(tr.text)
    st = fusion.fuse(af, se, calibrator=_calibrator(race_id), transcript=tr.text)

    return {
        "transcript": tr.text,
        "duration_s": tr.duration_s,
        "elapsed_s": tr.elapsed_s,
        "rtf": tr.rtf,
        "text_sentiment": {"label": se.label, "polarity": se.polarity},
        "state": st.to_dict(),
        "calibrated_against": race_id if _calibrator(race_id) else None,
    }


@app.get("/api/health")
def health():
    return {"ok": True, "models_warm": _WARM}


# In the Hugging Face Space the frontend is a static export served from here, so
# one container runs one process and the runtime image needs no Node.
#
# This mount is declared LAST on purpose: mounting "/" ahead of the route
# definitions above would shadow every /api path.
_STATIC = os.environ.get("PITWALL_STATIC")
if _STATIC and os.path.isdir(_STATIC):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="frontend")
    print(f"[startup] serving static frontend from {_STATIC}")
