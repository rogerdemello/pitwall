"""Hand-labelling tool for driver state, plus the evaluation that consumes it.

The dataset ships no emotion labels. Percentile calibration makes our scale
internally consistent, but nothing so far makes it *externally* correct. This
closes that gap the only way it can be closed: a human listens and says what
they hear, and we score the model against that.

Four modes.

    label   plays clips in a balanced sample and records your judgement (CLI)
    serve   the same sample in a browser, which is the one worth using
    import  ingest judgements collected some other way
    score   builds the confusion matrix and per-class precision/recall

Sampling is stratified across the model's own predicted states, so the sample
isn't dominated by whatever the model says most often - otherwise a model that
guesses "Energised" for everything would look good on a sample it chose itself.
Clips are presented without showing the prediction, so the label isn't anchored
by it.

**What this sample can and cannot measure.** Because it is stratified over
predicted state, per-class precision is honest but the class *base rates* are
not: this is not an estimate of how often drivers are actually stressed. Fitting
a decision boundary against it would import the stratification. Answering that
question needs a second, unstratified sample.

Usage:
    python backend/data/label_affect.py serve 2021_Abu_Dhabi_Grand_Prix 60
    python backend/data/label_affect.py serve all 300
    python backend/data/label_affect.py label 2021_Abu_Dhabi_Grand_Prix 60
    python backend/data/label_affect.py import 2021_Abu_Dhabi_Grand_Prix judged.json
    python backend/data/label_affect.py score 2021_Abu_Dhabi_Grand_Prix
    python backend/data/label_affect.py score all
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
CLIPS = os.path.join(HERE, "..", "clips")
LABELS = os.path.join(HERE, "..", "labels")

STATES = ["Calm", "Energised", "Stressed", "Fatigued"]
KEYS = {"1": "Calm", "2": "Energised", "3": "Stressed", "4": "Fatigued", "s": "SKIP"}


def _labels_path(race_id: str) -> str:
    os.makedirs(LABELS, exist_ok=True)
    return os.path.join(LABELS, f"{race_id}.labels.json")


def _load_labels(race_id: str) -> dict[str, str]:
    p = _labels_path(race_id)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def _save_labels(race_id: str, labels: dict[str, str]) -> None:
    with open(_labels_path(race_id), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=1, sort_keys=True)


def _race_ids() -> list[str]:
    from pipeline.artifacts import race_ids
    return race_ids(RACES)


def clip_path(rel: str) -> str | None:
    """Resolve `<race>/<clip>` under backend/clips, or None if it does not.

    The labelling server is a file server bound to localhost, and `main.py:159`
    guards the same thing for the same reason. A clip name is a leaf, never a
    path, so anything with a separator or a `..` in either half is refused
    before it is joined - and the resolved path is then checked to be inside
    the clips tree anyway, because one guard that has to be exactly right is
    worse than two that overlap.
    """
    parts = rel.split("/")
    if len(parts) != 2 or any(p in ("", "..", ".") or os.sep in p or "\\" in p
                              for p in parts):
        return None
    full = os.path.abspath(os.path.join(CLIPS, *parts))
    root = os.path.abspath(CLIPS)
    if os.path.commonpath([full, root]) != root or not os.path.isfile(full):
        return None
    return full


def sample_for(race_id: str, n: int = 60, seed: int = 11) -> list[dict]:
    """The stratified sample, as one function so every mode draws the same one.

    Fixed seed, so a pass interrupted and resumed presents the same clips, and
    so the sample is reproducible by anyone checking the result.
    """
    race = json.load(open(os.path.join(RACES, f"{race_id}.json"), encoding="utf-8"))
    done = _load_labels(race_id)

    # Stratify across predicted state so no single class dominates the sample.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for m in race["messages"]:
        if m["id"] not in done and len(m["transcript"].split()) >= 2:
            buckets[m["state"]].append(m)

    rng = random.Random(seed)
    per_class = max(1, n // len(STATES))
    sample: list[dict] = []
    for st in STATES:
        pool = buckets.get(st, [])
        rng.shuffle(pool)
        sample.extend(pool[:per_class])
    rng.shuffle(sample)
    return sample[:n]


def label(race_id: str, n: int = 60) -> None:
    sample = sample_for(race_id, n)
    done = _load_labels(race_id)

    if not sample:
        print("nothing left to label")
        return

    print(f"\nLabelling {len(sample)} clips from {race_id}")
    print("Play each clip, then press:")
    print("  1 Calm   2 Energised   3 Stressed   4 Fatigued   s Skip   q Save and quit")
    print("Judge the VOICE, not the words.\n")

    for i, m in enumerate(sample, 1):
        path = os.path.abspath(os.path.join(CLIPS, race_id, m["audio_file"]))
        print(f"[{i}/{len(sample)}] {m['driver_code']}  {m['duration_s']}s")
        print(f"    {path}")
        print(f'    "{m["transcript"][:100]}"')

        # Play it. Falls back to just printing the path if no player is available.
        try:
            if sys.platform == "win32":
                os.startfile(path)  # noqa: S606
            elif sys.platform == "darwin":
                os.system(f'afplay "{path}"')
            else:
                os.system(f'aplay -q "{path}" 2>/dev/null || ffplay -nodisp -autoexit -loglevel quiet "{path}"')
        except Exception:
            pass

        while True:
            k = input("    label> ").strip().lower()
            if k == "q":
                json.dump(done, open(_labels_path(race_id), "w", encoding="utf-8"), indent=1)
                print(f"\nsaved {len(done)} labels")
                return
            if k in KEYS:
                if KEYS[k] != "SKIP":
                    done[m["id"]] = KEYS[k]
                break
            print("    use 1/2/3/4/s/q")

        json.dump(done, open(_labels_path(race_id), "w", encoding="utf-8"), indent=1)

    print(f"\ndone - {len(done)} labels saved")


def import_labels(race_id: str, path: str) -> None:
    """Ingest judgements collected somewhere other than this tool.

    `score` never touches audio - it only needs {message_id: state} - so the
    listening pass does not have to happen through the loop above. This is the
    door for that: a JSON object, a JSON list of {id, label}, or a two-column
    CSV. Unknown ids and unknown states are reported rather than dropped
    silently, because a mis-keyed export that quietly labels forty clips and
    discards the rest is worse than one that fails.
    """
    race = json.load(open(os.path.join(RACES, f"{race_id}.json"), encoding="utf-8"))
    valid = {m["id"] for m in race["messages"]}

    raw = open(path, encoding="utf-8-sig").read().strip()
    incoming: dict[str, str] = {}
    if raw.startswith("{") or raw.startswith("["):
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            incoming = {str(k): str(v) for k, v in parsed.items()}
        else:
            for row in parsed:
                incoming[str(row["id"])] = str(row.get("label") or row.get("state"))
    else:
        for line in raw.splitlines():
            if not line.strip():
                continue
            parts = [c.strip().strip('"') for c in line.split(",")]
            if len(parts) >= 2 and parts[0].lower() not in ("id", "message_id"):
                incoming[parts[0]] = parts[1]

    canon = {s.lower(): s for s in STATES}
    accepted, unknown_id, unknown_state = {}, [], []
    for mid, state in incoming.items():
        st = canon.get(str(state).strip().lower())
        if st is None:
            unknown_state.append((mid, state))
        elif mid not in valid:
            unknown_id.append(mid)
        else:
            accepted[mid] = st

    if unknown_state:
        print(f"!! {len(unknown_state)} unrecognised state(s); expected one of {STATES}")
        for mid, state in unknown_state[:5]:
            print(f"     {mid} -> {state!r}")
    if unknown_id:
        print(f"!! {len(unknown_id)} id(s) not in {race_id}: {unknown_id[:5]}")
    if not accepted:
        print("nothing to import")
        return

    merged = _load_labels(race_id)
    added = sum(1 for k in accepted if k not in merged)
    changed = sum(1 for k, v in accepted.items() if k in merged and merged[k] != v)
    merged.update(accepted)
    _save_labels(race_id, merged)
    print(f"imported {len(accepted)} ({added} new, {changed} changed); "
          f"{len(merged)} labels total for {race_id}")
    print(f"next:  python backend/data/label_affect.py score {race_id}")


def serve(spec: str = "2021_Abu_Dhabi_Grand_Prix", n: int = 60, port: int = 8765) -> None:
    """The listening pass, in a browser.

    The CLI mode shells out to a media player per clip, which on Windows opens
    a window each time. That is tolerable for a demo and not for three hundred
    clips, and the labelling pass not happening is precisely the gap this tool
    was written to close - so the ergonomics are the feature.

    Every keystroke is written straight to backend/labels/<race>.labels.json,
    so closing the tab loses nothing.

    The transcript is hidden behind a key, not shown by default. The judgement
    asked for is about the voice, and reading "I have no grip" before deciding
    anchors it. The model's own prediction is never sent to the page at all.
    """
    import http.server
    import threading
    import urllib.parse
    import webbrowser

    races = _race_ids() if spec in ("all", "*") else [spec]
    per_race = max(1, n // len(races))

    queue: list[dict] = []
    for rid in races:
        for m in sample_for(rid, per_race):
            queue.append({
                "race_id": rid,
                "id": m["id"],
                "driver": m.get("driver_code") or m.get("driver_id"),
                "duration_s": m.get("duration_s"),
                "transcript": m.get("transcript", ""),
                "audio": f"/audio/{rid}/{urllib.parse.quote(m['audio_file'])}",
            })
    random.Random(11).shuffle(queue)

    if not queue:
        print("nothing left to label - every sampled clip already has a judgement")
        return

    already = sum(len(_load_labels(r)) for r in races)
    print(f"{len(queue)} clips queued across {len(races)} race(s); "
          f"{already} already labelled")

    page = _LABEL_PAGE.replace("__QUEUE__", json.dumps(queue))

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: D102 - one line per clip is enough
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            if path.startswith("/audio/"):
                full = clip_path(urllib.parse.unquote(path[len("/audio/"):]))
                if full is None:
                    return self._send(404, b"not found", "text/plain")
                with open(full, "rb") as f:
                    return self._send(200, f.read(), "audio/mpeg")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path != "/save":
                return self._send(404, b"not found", "text/plain")
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                rec = json.loads(body)
                rid, mid, state = rec["race_id"], rec["id"], rec["label"]
                if state not in STATES:
                    raise ValueError(f"unknown state {state!r}")
            except Exception as e:
                return self._send(400, json.dumps({"error": str(e)}).encode(),
                                  "application/json")
            labels = _load_labels(rid)
            labels[mid] = state
            _save_labels(rid, labels)
            total = sum(len(_load_labels(r)) for r in races)
            print(f"  {rid}  {mid} -> {state}   ({total} labelled)", flush=True)
            return self._send(200, json.dumps({"ok": True, "total": total}).encode(),
                              "application/json")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"\nlabelling at {url}   (ctrl-c when done)")
    print("keys:  1 Calm   2 Energised   3 Stressed   4 Fatigued   "
          "space replay   s skip   t transcript\n")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        for r in races:
            k = len(_load_labels(r))
            if k:
                print(f"  {r}: {k} labels")
        print("\nnext:  python backend/data/label_affect.py score all")
    finally:
        srv.server_close()


def score(race_id: str) -> None:
    race = json.load(open(os.path.join(RACES, f"{race_id}.json"), encoding="utf-8"))
    truth = _load_labels(race_id)
    if not truth:
        print("no labels yet - run `label` first")
        return

    pred = {m["id"]: m["state"] for m in race["messages"]}
    pairs = [(truth[i], pred[i]) for i in truth if i in pred]
    if not pairs:
        print("labels do not match any message ids")
        return

    matrix: dict[str, Counter] = {s: Counter() for s in STATES}
    for t, p in pairs:
        matrix[t][p] += 1

    correct = sum(1 for t, p in pairs if t == p)
    acc = correct / len(pairs)

    print(f"\n{race_id}: {len(pairs)} labelled clips")
    print(f"accuracy: {acc:.3f} ({correct}/{len(pairs)})\n")

    w = 11
    print(" " * w + "".join(f"{s[:9]:>{w}}" for s in STATES) + f"{'total':>{w}}")
    for t in STATES:
        row = matrix[t]
        print(f"{t:<{w}}" + "".join(f"{row.get(s, 0):>{w}}" for s in STATES)
              + f"{sum(row.values()):>{w}}")

    print("\nper-class:")
    report = {}
    for s in STATES:
        tp = matrix[s].get(s, 0)
        fp = sum(matrix[t].get(s, 0) for t in STATES if t != s)
        fn = sum(v for k, v in matrix[s].items() if k != s)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
        report[s] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        fmt = lambda v: f"{v:.3f}" if v is not None else "  n/a"  # noqa: E731
        print(f"  {s:<10} precision {fmt(prec)}   recall {fmt(rec)}   f1 {fmt(f1)}   n={tp + fn}")

    out = os.path.join(RACES, f"{race_id}.affect_eval.json")
    json.dump({
        "race_id": race_id,
        "n": len(pairs),
        "accuracy": round(acc, 4),
        "confusion": {t: dict(matrix[t]) for t in STATES},
        "per_class": report,
    }, open(out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {out}")


_LABEL_PAGE = """<!doctype html>
<meta charset="utf-8"><title>PIT WALL - affect labelling</title>
<style>
 :root{--bg:#0e1013;--card:#171a1f;--line:#272b32;--ink:#e6e8ea;--mut:#8b929c;
       --calm:#4ea6ff;--ener:#3ddc97;--stre:#ff5c5c;--fati:#fab219}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif;
      display:flex;align-items:center;justify-content:center;min-height:100vh;padding:24px}
 .wrap{width:min(680px,100%)}
 .bar{height:4px;background:var(--line);border-radius:2px;overflow:hidden;margin-bottom:20px}
 .bar>i{display:block;height:100%;background:var(--calm);width:0;transition:width .2s}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:26px}
 .meta{display:flex;justify-content:space-between;color:var(--mut);font-size:12.5px;margin-bottom:18px}
 .play{width:100%;padding:22px;border:1px dashed var(--line);border-radius:10px;background:#0b0d10;
       color:var(--ink);font-size:15px;cursor:pointer;margin-bottom:6px}
 .play:hover{border-color:var(--calm)}
 .hint{color:var(--mut);font-size:12px;text-align:center;margin:0 0 18px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
 button.st{padding:15px;border-radius:9px;border:1px solid var(--line);background:#0b0d10;
           color:var(--ink);font-size:14.5px;cursor:pointer;text-align:left}
 button.st:hover{border-color:currentColor}
 button.st b{display:block;font-size:11px;color:var(--mut);font-weight:500;margin-top:3px}
 #b1{color:var(--calm)}#b2{color:var(--ener)}#b3{color:var(--stre)}#b4{color:var(--fati)}
 .row{display:flex;gap:10px;margin-top:10px}
 .row button{flex:1;padding:11px;border-radius:9px;border:1px solid var(--line);
             background:transparent;color:var(--mut);font-size:13px;cursor:pointer}
 .tr{margin-top:16px;padding:12px;background:#0b0d10;border-radius:8px;font-size:13.5px;
     color:var(--mut);display:none}
 .done{text-align:center;padding:40px 0}
 kbd{background:#0b0d10;border:1px solid var(--line);border-radius:4px;padding:1px 6px;font-size:11.5px}
</style>
<div class="wrap">
 <div class="bar"><i id="pg"></i></div>
 <div class="card" id="card">
  <div class="meta"><span id="who"></span><span id="cnt"></span></div>
  <button class="play" id="play">&#9654;&nbsp; play</button>
  <p class="hint">Judge the <b>voice</b>, not the words.
     <kbd>space</kbd> replay &middot; <kbd>t</kbd> transcript &middot; <kbd>s</kbd> skip</p>
  <div class="grid">
   <button class="st" id="b1" data-s="Calm">1 &middot; Calm<b>settled, unhurried</b></button>
   <button class="st" id="b2" data-s="Energised">2 &middot; Energised<b>up, positive, driving hard</b></button>
   <button class="st" id="b3" data-s="Stressed">3 &middot; Stressed<b>tense, strained, under pressure</b></button>
   <button class="st" id="b4" data-s="Fatigued">4 &middot; Fatigued<b>flat, tired, resigned</b></button>
  </div>
  <div class="row"><button id="skip">skip</button><button id="tr">transcript</button></div>
  <div class="tr" id="trbox"></div>
 </div>
</div>
<script>
const Q = __QUEUE__;
let i = 0, saved = 0;
const audio = new Audio();
const $ = id => document.getElementById(id);

function render(){
  if (i >= Q.length){
    $("card").innerHTML = '<div class="done"><h2>Done</h2><p style="color:var(--mut)">'
      + saved + ' judgements saved.<br>Now run <code>python backend/data/'
      + 'label_affect.py score all</code></p></div>';
    $("pg").style.width = "100%";
    return;
  }
  const c = Q[i];
  $("who").textContent = c.driver + "  ·  " + c.duration_s + "s  ·  "
                       + c.race_id.replace(/_/g," ").replace(" Grand Prix","");
  $("cnt").textContent = (i+1) + " / " + Q.length;
  $("pg").style.width = (i / Q.length * 100) + "%";
  $("trbox").style.display = "none";
  $("trbox").textContent = c.transcript || "(no transcript)";
  audio.src = c.audio;
  audio.play().catch(()=>{});   // autoplay may need the first click
}

async function choose(state){
  const c = Q[i];
  if (state){
    try {
      const r = await fetch("/save", {method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({race_id:c.race_id, id:c.id, label:state})});
      if (r.ok) saved++;
    } catch(e){ alert("could not save: " + e); return; }
  }
  i++; render();
}

$("play").onclick = () => { audio.currentTime = 0; audio.play(); };
$("skip").onclick = () => choose(null);
$("tr").onclick   = () => { const b=$("trbox"); b.style.display = b.style.display==="none"?"block":"none"; };
document.querySelectorAll("button.st").forEach(b => b.onclick = () => choose(b.dataset.s));
addEventListener("keydown", e => {
  if (e.key === " "){ e.preventDefault(); audio.currentTime=0; audio.play(); return; }
  if (e.key === "t"){ $("tr").click(); return; }
  if (e.key === "s"){ choose(null); return; }
  const n = {"1":"Calm","2":"Energised","3":"Stressed","4":"Fatigued"}[e.key];
  if (n) choose(n);
});
render();
</script>
"""


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "label"
    race = sys.argv[2] if len(sys.argv) > 2 else "2021_Abu_Dhabi_Grand_Prix"
    if mode == "score":
        if race in ("all", "*"):
            for rid in _race_ids():
                if _load_labels(rid):
                    score(rid)
        else:
            score(race)
    elif mode == "serve":
        port = 8765
        if "--port" in sys.argv:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        rest = [a for a in sys.argv[3:] if not a.startswith("--")]
        serve(race, int(rest[0]) if rest and rest[0].isdigit() else 60, port)
    elif mode == "import":
        if len(sys.argv) < 4:
            raise SystemExit("usage: label_affect.py import <race_id> <file.json|csv>")
        import_labels(race, sys.argv[3])
    else:
        label(race, int(sys.argv[3]) if len(sys.argv) > 3 else 60)
