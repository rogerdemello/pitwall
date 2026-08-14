"""Record the demo walkthrough as video, by driving the real app.

This is not a mockup or a slideshow. It serves `frontend/out` - the exact bundle
deployed to the Space - and clicks through it with Playwright, so what you see
is the shipped application computing the numbers on screen.

Two limits worth knowing before you use the output:

  * Playwright's video has **no audio track**. The radio clips cannot be heard,
    so every beat carries the transcript as an on-screen caption instead. Record
    a voiceover over the top, or present live with this as a fallback.
  * There is no ffmpeg here, so the output is `.webm`. Browsers and every editor
    read it; convert if your submission needs mp4.

The beats and their numbers come from `docs/DEMO.md`, and
`backend/tests/test_demo_script.py` pins those to the corpus - so if a rebuild
moves a DSI, the tests fail before the recording does.

    python backend/tools/record_demo.py
    python backend/tools/record_demo.py --fast     # shorter holds, for iterating
    python backend/tools/record_demo.py --out demo.webm
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import shutil
import socketserver
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
OUT_DIR = os.path.join(ROOT, "frontend", "out")
PORT = 8899

#: The RaceTimeline maps lap -> x with these constants (frontend/components).
#: Inverting it is how a specific lap's pin gets clicked when it is not one of
#: the four "key moments" the app surfaces as buttons.
W, PAD_L, PAD_R = 1000, 52, 16
MAX_LAP = 58


def cx_for(lap: int) -> float:
    return PAD_L + ((lap - 1) / (MAX_LAP - 1)) * (W - PAD_L - PAD_R)


# --------------------------------------------------------------------------
# the caption layer
# --------------------------------------------------------------------------

OVERLAY_CSS = """
/* Anchored bottom-LEFT, under the chart, never full width.
   A full-width band covered the detail panel on the right - which is where the
   DSI, the percentiles and the tyre data live, i.e. exactly the evidence each
   caption is pointing at. Occluding your own proof is worse than no caption. */
#demo-cap {
  position: fixed; left: 24px; bottom: 22px; z-index: 2147483647;
  width: min(60%, 830px); padding: 16px 22px 18px; pointer-events: none;
  background: rgba(8,11,16,.94); border: 1px solid rgba(120,135,155,.22);
  border-left: 3px solid #ff4d3d; border-radius: 10px;
  box-shadow: 0 18px 48px rgba(0,0,0,.55);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  opacity: 0; transition: opacity .35s ease;
}
#demo-cap.on { opacity: 1; }
#demo-cap .eyebrow {
  font-size: 11px; letter-spacing: .16em; text-transform: uppercase;
  color: #8892a0; margin-bottom: 7px; font-weight: 600;
}
#demo-cap .line { font-size: 20px; line-height: 1.42; color: #f2f5f8; }
#demo-cap .line b { color: #ff4d3d; font-weight: 650; }
#demo-cap .quote { font-style: italic; color: #cfd6de; }
#demo-title {
  position: fixed; inset: 0; z-index: 2147483646; display: flex;
  flex-direction: column; align-items: center; justify-content: center;
  background: #06080c; color: #f2f5f8; gap: 20px; opacity: 0;
  transition: opacity .5s ease;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  /* Covers the whole viewport, so it must never intercept a click. Without
     this it sits invisibly over the app at opacity 0 and every subsequent
     Playwright click times out against it. */
  pointer-events: none; visibility: hidden;
}
#demo-title.on { opacity: 1; visibility: visible; }
#demo-title .big { font-size: 58px; font-weight: 700; letter-spacing: -.02em; }
#demo-title .sub { font-size: 20px; color: #98a2b3; max-width: 60ch; text-align: center;
                   line-height: 1.5; }
#demo-title .url { font-family: ui-monospace, Consolas, monospace; font-size: 16px;
                   color: #ff4d3d; margin-top: 8px; }
@media (prefers-reduced-motion: reduce) {
  #demo-cap, #demo-title { transition: none; }
}
"""

OVERLAY_JS = """
window.__demo = {
  ensure() {
    if (document.getElementById('demo-cap')) return;
    const s = document.createElement('style'); s.textContent = %s;
    document.head.appendChild(s);
    const c = document.createElement('div'); c.id = 'demo-cap';
    c.innerHTML = '<div class="eyebrow"></div><div class="line"></div>';
    document.body.appendChild(c);
    const t = document.createElement('div'); t.id = 'demo-title';
    t.innerHTML = '<div class="big"></div><div class="sub"></div><div class="url"></div>';
    document.body.appendChild(t);
  },
  say(eyebrow, html) {
    this.ensure();
    const c = document.getElementById('demo-cap');
    c.querySelector('.eyebrow').textContent = eyebrow || '';
    c.querySelector('.line').innerHTML = html || '';
    c.classList.add('on');
  },
  hide() { const c = document.getElementById('demo-cap'); if (c) c.classList.remove('on'); },
  card(big, sub, url) {
    this.ensure();
    const t = document.getElementById('demo-title');
    t.querySelector('.big').textContent = big || '';
    t.querySelector('.sub').textContent = sub || '';
    t.querySelector('.url').textContent = url || '';
    t.classList.add('on');
  },
  uncard() { const t = document.getElementById('demo-title'); if (t) t.classList.remove('on'); }
};
""" % repr(OVERLAY_CSS)


# --------------------------------------------------------------------------
# the walkthrough
# --------------------------------------------------------------------------

class Recorder:
    def __init__(self, page, speed: float):
        self.pg = page
        self.speed = speed

    def hold(self, seconds: float) -> None:
        self.pg.wait_for_timeout(int(seconds * 1000 * self.speed))

    def say(self, eyebrow: str, html: str, seconds: float) -> None:
        self.pg.evaluate("([e,h]) => window.__demo.say(e,h)", [eyebrow, html])
        self.hold(seconds)

    def card(self, big: str, sub: str = "", url: str = "", seconds: float = 3.0) -> None:
        self.pg.evaluate("([b,s,u]) => window.__demo.card(b,s,u)", [big, sub, url])
        self.hold(seconds)
        self.pg.evaluate("() => window.__demo.uncard()")
        self.hold(0.6)

    def driver(self, code: str) -> None:
        self.pg.get_by_role("button", name=code, exact=False).first.click()
        self.hold(1.1)

    def key_moment(self, label: str) -> None:
        self.pg.get_by_role("button", name=label).first.click()
        self.hold(1.3)

    def pin_at_lap(self, lap: int) -> None:
        self.pg.evaluate("""(cx) => {
            const cs = Array.from(document.querySelectorAll('svg circle'))
                .filter(c => c.getAttribute('r') === '13');
            if (!cs.length) return;
            const c = cs.reduce((a, b) =>
                Math.abs(+b.getAttribute('cx') - cx) < Math.abs(+a.getAttribute('cx') - cx) ? b : a);
            c.dispatchEvent(new MouseEvent('click', {bubbles: true}));
        }""", cx_for(lap))
        self.hold(1.3)

    def nav(self, name: str) -> None:
        self.pg.get_by_role("link", name=name).first.click()
        self.pg.wait_for_load_state("networkidle")
        self.hold(1.2)
        self.pg.evaluate("() => window.__demo && window.__demo.ensure()")

    def scroll_to(self, text: str) -> None:
        try:
            self.pg.get_by_text(text, exact=False).first.scroll_into_view_if_needed()
        except Exception:
            pass
        self.hold(0.8)


def walk(rec: Recorder) -> None:
    pg = rec.pg

    rec.card("PIT WALL", "The Silent Co-Driver  ·  2,042 real F1 team-radio messages, "
                         "placed on the lap they were spoken on",
             "huggingface.co/spaces/rogerdemello/pitwall", 3.6)

    # ---- 1. the showcase clip ------------------------------------------
    rec.say("Race Replay", "Pick a race and a driver. Every radio call is a point on the "
                           "<b>Driver State Index</b>, sharing one x-axis with lap time.", 3.4)
    rec.driver("SAI")
    rec.say("Carlos Sainz · 2021 Abu Dhabi", "Peak <b>DSI 92</b> of his race. "
            "The model found it, not a human.", 3.0)
    rec.key_moment("L3592")
    rec.say("Lap 35 · DSI 92 · Stressed",
            '<span class="quote">"I\'m in a two-stop, there\'s no way you meet these '
            'two guys in front of me."</span>', 4.2)
    rec.say("Lap 35 · DSI 92 · Stressed",
            "Arousal in the <b>96th percentile</b>, valence in the <b>12th</b>. "
            "And <b>lap 35</b> is the real lap he was driving.", 4.0)

    # ---- 2. the join ----------------------------------------------------
    rec.say("The join", "Two unrelated sources agreeing is the whole project. "
                        "Hamilton, lap 15.", 3.0)
    rec.driver("HAM")
    rec.pin_at_lap(15)
    rec.say("Lewis Hamilton · Lap 15",
            '<span class="quote">"Ok Lewis, so box, box."</span> &nbsp;The radio orders '
            'the stop.', 3.6)
    rec.say("Telemetry confirms it independently",
            "That lap took <b>108.75s</b> — <b>+20.92s</b> versus his median. Compound "
            "changes to <b>Hard</b>, tyre age resets to <b>2 laps</b>. "
            "Nobody told the code where the pit stops were.", 5.0)
    rec.say("Speaker attribution",
            "Note the banner: <b>engineer voice — not scored as driver state</b>. "
            "The pit wall talking is not the driver's stress.", 4.0)

    # ---- 3. suppressed stress -------------------------------------------
    rec.say("The feature nobody else built", "Everyone notices a driver shouting. "
            "What gets missed is the opposite.", 3.2)
    rec.driver("PER")
    rec.key_moment("L2197")
    rec.say("Pérez · Lap 21 · DSI 97 · suppressed stress",
            '<span class="quote">"Okay, back him up, back him up."</span>', 3.8)
    rec.say("Words positive. Voice bottom 1%.",
            "Text sentiment reads this as <b>positive</b>. The voice sits in the "
            "<b>bottom 1%</b> of the race for valence — the most negative-sounding "
            "message of the Grand Prix. We flag the gap.", 5.0)

    # ---- 4. evidence -----------------------------------------------------
    rec.nav("Evidence")
    rec.say("Evidence", "Every number here is read from the file that measured it. "
                        "Nothing on this screen is typed by hand.", 3.6)
    rec.say("The central claim failed",
            "<b>1,155</b> paired observations, 12 races. Pooled <b>r = 0.026</b>. "
            "Stress does <i>not</i> predict lap-time loss.", 4.2)
    rec.say("And a better model made it fail harder",
            "We rebuilt the corpus on <b>whisper-large-v3</b>. Every statistic moved "
            "<i>toward</i> zero. <b>41 of 80</b> drivers slower when stressed — "
            "a coin flip, p = 0.91.", 5.0)
    rec.say("We pre-registered five predictions. Four broke.",
            "Under the old pipeline <b>4 of 5</b> held. On the better one, <b>1 of 5</b>. "
            "So we published the falsification clause we wrote in advance.", 5.0)
    rec.say("Which half of the model works",
            "Arousal <b>79.4%</b> against a 61.8% baseline. Valence <b>60.9%</b> "
            "against 61.9% — <i>below</i> chance. The product says so.", 4.6)

    # ---- 5. live ---------------------------------------------------------
    rec.nav("Live Analysis")
    rec.say("Live Analysis",
            "The replay is precomputed so it cannot fail on stage. This is the same "
            "<b>pipeline/</b> package running on a ZeroGPU Space — proof it is an "
            "analysis, not a recording.", 5.0)

    rec.card("Measured, not asserted",
             "A null result with 1,155 observations and a retracted prediction, "
             "rather than a correlation we manufactured.",
             "huggingface.co/spaces/rogerdemello/pitwall", 4.4)


# --------------------------------------------------------------------------

def serve(directory: str, port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pitwall_demo.webm")
    ap.add_argument("--fast", action="store_true", help="shorter holds, for iterating")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args(argv)

    if not os.path.isdir(OUT_DIR):
        raise SystemExit("!! frontend/out is missing - run "
                         "python backend/tools/deploy.py --dry-run first")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("!! pip install playwright && playwright install chromium")

    httpd = serve(OUT_DIR, a.port)
    tmp = os.path.join(ROOT, ".demo_video")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    print(f"serving {OUT_DIR} on 127.0.0.1:{a.port}")

    t0 = time.time()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(
                viewport={"width": 1440, "height": 900},
                record_video_dir=tmp,
                record_video_size={"width": 1440, "height": 900},
                reduced_motion="no-preference",
                color_scheme="dark",
            )
            pg = ctx.new_page()
            pg.goto(f"http://127.0.0.1:{a.port}/", wait_until="networkidle")
            pg.wait_for_timeout(2200)
            pg.add_init_script(OVERLAY_JS)
            pg.evaluate(OVERLAY_JS)
            pg.evaluate("() => window.__demo.ensure()")

            walk(Recorder(pg, 0.45 if a.fast else 1.0))

            video = pg.video
            ctx.close()
            browser.close()
            src = video.path()
    finally:
        httpd.shutdown()

    dest = os.path.join(ROOT, a.out)
    shutil.move(src, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    mb = os.path.getsize(dest) / 1e6
    print(f"\nwrote {dest}  ({mb:.1f} MB, recorded in {time.time()-t0:.0f}s)")
    print("no audio track - Playwright records video only; narrate over it or "
          "use the on-screen captions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
