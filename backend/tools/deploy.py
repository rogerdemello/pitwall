"""Ship everything, in the one order that works.

Deploying this project by hand is six steps with three ways to quietly break it,
and all three were hit at least once:

  * `npm run check:export` runs its own `next build`. Run it *after* the real
    build and it silently replaces out/ with one that has no live-Space URL
    baked in and no README.md - an upload that looks fine, has Live Analysis
    dead, and has no Space frontmatter.
  * `space_live/pipeline/` is a gitignored build artifact. Upload without
    running build_space_live first and the Space ships with no pipeline.
  * `frontend/public/data/` is a snapshot. Forget to rebuild it and the Space
    serves numbers the repository has already superseded - which is exactly
    what happened, undetected, until a --check was added for it.

So the order is encoded here rather than remembered. Every gate that can fail
cheaply runs before anything is uploaded.

    python backend/tools/deploy.py              # gates, build, upload all three
    python backend/tools/deploy.py --fast       # skip pytest, keep cheap gates
    python backend/tools/deploy.py --dry-run    # build and verify, upload nothing
    python backend/tools/deploy.py --only live  # static | live | dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
FRONTEND = os.path.join(ROOT, "frontend")
OUT = os.path.join(FRONTEND, "out")

STATIC_SPACE = "rogerdemello/pitwall"
LIVE_SPACE = "rogerdemello/pitwall-live"
DATASET = "rogerdemello/pitwall-f1-radio-analysis"
LIVE_URL = "https://rogerdemello-pitwall-live.hf.space"

IGNORE = ["**/__pycache__/**", "**/*.pyc"]


def resolve(exe: str) -> str:
    """Absolute path to a command.

    Two Windows traps, both hit here. subprocess without shell=True does not
    consult PATHEXT, so bare "npm" raises FileNotFoundError. And under Git Bash
    the PATH also carries Node's extensionless *shell* script, which
    shutil.which happily returns and CreateProcess then rejects with "not a
    valid Win32 application". So on Windows the .cmd/.exe shim is preferred
    explicitly rather than left to which()'s ordering.
    """
    if sys.platform == "win32":
        for ext in (".cmd", ".exe", ".bat"):
            found = shutil.which(exe + ext)
            if found:
                return found
    found = shutil.which(exe)
    if found is None:
        raise SystemExit(f"!! {exe} is not on PATH - cannot deploy")
    return found


def run(cmd: list[str], cwd: str = ROOT, env: dict | None = None) -> None:
    """Run a step, or stop the deploy. A half-deployed app is worse than none."""
    cmd = [resolve(cmd[0]) if not os.path.isabs(cmd[0]) else cmd[0], *cmd[1:]]
    label = " ".join([os.path.basename(cmd[0]), *cmd[1:4]]) + (" ..." if len(cmd) > 4 else "")
    print(f"  $ {label}", flush=True)
    e = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stdout or "")[-1500:] + (r.stderr or "")[-1500:]
        raise SystemExit(f"\n!! step failed: {label}\n{tail}")


def step(n: int, total: int, what: str) -> None:
    print(f"\n[{n}/{total}] {what}", flush=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--fast", action="store_true",
                    help="skip pytest; the cheap --check gates still run")
    ap.add_argument("--dry-run", action="store_true", help="build and verify, upload nothing")
    ap.add_argument("--only", choices=["static", "live", "dataset"],
                    help="deploy one artifact instead of all three")
    a = ap.parse_args(argv)

    want = {"static", "live", "dataset"} if not a.only else {a.only}
    t0 = time.time()
    total = 7

    # ---- 1. Gates, before anything is built or uploaded --------------------
    step(1, total, "gates")
    if a.fast:
        print("  (skipping pytest - --fast)")
    else:
        run([sys.executable, "-m", "pytest", "backend/tests", "-q"])
    run([sys.executable, "backend/data/render_docs.py", "--check"])
    run([sys.executable, "backend/tools/clean_notebooks.py", "--check"])

    # ---- 2. The snapshot the static Space actually reads --------------------
    step(2, total, "freeze the API to frontend/public/data")
    run([sys.executable, "backend/data/build_static_site.py"])
    run([sys.executable, "backend/data/build_static_site.py", "--check"])

    # ---- 3. Export verification, BEFORE the real build ----------------------
    if "static" in want:
        step(3, total, "verify the export is buildable (runs its own next build)")
        run(["npm", "run", "check:export"], cwd=FRONTEND)
    else:
        step(3, total, "skipped - not deploying the static Space")

    # ---- 4. The real build, which must be the last one to touch out/ -------
    if "static" in want:
        step(4, total, "build the static export")
        shutil.rmtree(OUT, ignore_errors=True)
        run(["npx", "next", "build"], cwd=FRONTEND, env={
            "NEXT_PUBLIC_API": "",
            "NEXT_PUBLIC_STATIC": "1",
            "NEXT_PUBLIC_LIVE_SPACE": LIVE_URL,
            "NEXT_OUTPUT": "export",
        })
        shutil.copy(os.path.join(ROOT, "space", "README.static.md"),
                    os.path.join(OUT, "README.md"))
        verify_export()
    else:
        step(4, total, "skipped - not deploying the static Space")

    # ---- 5. Assemble the model backend -------------------------------------
    if "live" in want:
        step(5, total, "assemble space_live from backend/pipeline")
        run([sys.executable, "backend/data/build_space_live.py"])
    else:
        step(5, total, "skipped - not deploying the model backend")

    # ---- 6. Upload ----------------------------------------------------------
    step(6, total, "upload" + (" (dry run - nothing leaves this machine)" if a.dry_run else ""))
    if a.dry_run:
        for name in sorted(want):
            print(f"  would upload {name}")
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        who = api.whoami()["name"]
        print(f"  authenticated as {who}")
        if "live" in want:
            api.upload_folder(repo_id=LIVE_SPACE, repo_type="space",
                              folder_path=os.path.join(ROOT, "space_live"),
                              ignore_patterns=IGNORE,
                              commit_message="Deploy the model backend")
            print(f"  uploaded {LIVE_SPACE}")
        if "static" in want:
            api.upload_folder(repo_id=STATIC_SPACE, repo_type="space",
                              folder_path=OUT, ignore_patterns=IGNORE,
                              commit_message="Deploy the app")
            print(f"  uploaded {STATIC_SPACE}")
        if "dataset" in want:
            run([sys.executable, "backend/data/push_to_hub.py"])
            print(f"  uploaded {DATASET}")

    # ---- 7. Confirm it is actually serving ---------------------------------
    step(7, total, "verify what is live")
    if a.dry_run:
        print("  skipped - dry run")
    else:
        verify_live(want)

    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


def verify_export() -> None:
    """The two things that make an upload look fine and behave wrongly."""
    readme = os.path.join(OUT, "README.md")
    if not os.path.exists(readme):
        raise SystemExit("!! out/README.md missing - the Space would have no frontmatter")
    hits = 0
    for dp, _, files in os.walk(OUT):
        for fn in files:
            if fn.endswith((".html", ".js")):
                with open(os.path.join(dp, fn), encoding="utf-8", errors="ignore") as f:
                    if LIVE_URL in f.read():
                        hits += 1
    if not hits:
        raise SystemExit(
            "!! the live-Space URL is not in the bundle - Live Analysis would be "
            "dead in public. Something rebuilt out/ without NEXT_PUBLIC_LIVE_SPACE.")
    print(f"  ok  README.md present, live-Space URL in {hits} file(s)")


def verify_live(want: set[str]) -> None:
    import json
    import urllib.request

    if "static" in want:
        url = "https://rogerdemello-pitwall.static.hf.space/data/corpus-analysis.json"
        served = json.load(urllib.request.urlopen(url, timeout=60))
        local = json.load(open(os.path.join(ROOT, "backend", "races",
                                            "_corpus_analysis.json"), encoding="utf-8"))
        a = served["stress_vs_pace"]["pooled_r"]
        b = local["stress_vs_pace"]["pooled_r"]
        mark = "ok " if a == b else "!! "
        print(f"  {mark} static Space serving pooled_r {a} (repo says {b})")
        if a != b:
            print("      the CDN may still be catching up; re-check in a minute")

    if "live" in want:
        from huggingface_hub import HfApi
        rt = HfApi().space_info(LIVE_SPACE).runtime
        print(f"  ok  model backend {getattr(rt, 'stage', None)} "
              f"{getattr(rt, 'hardware', None) or ''}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
