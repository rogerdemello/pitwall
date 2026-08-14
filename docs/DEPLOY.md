# Publishing to Hugging Face

**Three artifacts are live under `rogerdemello`:**

- Dataset — https://huggingface.co/datasets/rogerdemello/pitwall-f1-radio-analysis
- App (static) — https://huggingface.co/spaces/rogerdemello/pitwall
- Model backend (ZeroGPU) — https://huggingface.co/spaces/rogerdemello/pitwall-live

## What shipped, and why it's a *static* Space

Docker Spaces now require a PRO subscription (`402 Payment Required`: *"Static
Spaces are free for everyone, but hosting Gradio and Docker Spaces on free
cpu-basic requires a PRO subscription"*). Static Spaces are free, and almost
nothing here needs a server — Replay and Evidence read precomputed JSON, and
only Live Analysis runs a model.

So `build_static_site.py` freezes every GET endpoint to a file (`/api/<x>` →
`/data/<x>.json`, one uniform rewrite rather than a per-endpoint table) and the
frontend builds with `NEXT_PUBLIC_STATIC=1`.

Three things worth knowing if you redeploy:

- **Rebuild the snapshot, or the Space serves superseded numbers.** This is not
  hypothetical: the corpus was recalibrated, `frontend/public/data/` was not
  rebuilt, and the live site served pre-recalibration figures while every test
  passed — because the tests watched `backend/races/` and the deployed artifact
  reads the snapshot. `build_static_site.py --check` now diffs the two and
  `backend/tests/test_static_snapshot.py` fails the build on a mismatch.

- **The static host serves exact file paths only.** It resolves `index.html` at
  the root but *not* in subdirectories, so `/evidence/` 404s and the browser
  falls back to huggingface.co. Static builds therefore link at
  `/evidence/index.html`. This is handled in `Nav.tsx`; don't "tidy" it.
- **`short_description` must be ≤ 60 characters** or `upload_folder` rejects the
  whole push on YAML validation.

To rebuild and redeploy:

```bash
python backend/tools/deploy.py            # gates, build, upload all three
python backend/tools/deploy.py --fast     # skip pytest, keep the cheap gates
python backend/tools/deploy.py --dry-run  # build and verify, upload nothing
python backend/tools/deploy.py --only live
```

About two minutes with `--fast`. It encodes the ordering rather than trusting
anyone to remember it, and refuses to upload an export that is missing its
README or its live-Space URL — the two ways an upload looks fine and behaves
wrongly. The manual sequence below is what it runs, kept for reference:


```bash
python backend/data/build_static_site.py
python backend/data/build_static_site.py --check      # must exit 0
cd frontend
npm run check:export                                  # FIRST - see below
NEXT_PUBLIC_API="" NEXT_PUBLIC_STATIC=1 \
  NEXT_PUBLIC_LIVE_SPACE="https://rogerdemello-pitwall-live.hf.space" \
  NEXT_OUTPUT=export npx next build
cp ../space/README.static.md out/README.md            # the YAML frontmatter is
                                                      # what configures the Space
hf upload rogerdemello/pitwall out . --repo-type=space
```

**`check:export` runs its own `next build`, so it must come before the real
one, not after.** Run it afterwards and it silently replaces `out/` with a build
that has no `NEXT_PUBLIC_LIVE_SPACE` baked in and no `README.md` — an upload
that looks fine, has Live Analysis dead again, and has no Space frontmatter.

## The model backend: `space_live/`

Live Analysis needs a process to run three models in, and a static Space has
none. Free accounts in good standing may host two **ZeroGPU** Gradio Spaces, so
the models live in a second Space and the static frontend calls it cross-origin.
Gradio's CORS middleware accepts any origin when the host is not a localhost
alias, which is the case on `*.hf.space`, so there is no proxy.

`space_live/` is assembled, not written: `build_space_live.py` copies
`backend/pipeline/` in verbatim and `backend/tests/test_space_live.py` asserts
every module is byte-identical and that the Gradio response matches
`POST /api/analyze` field for field. There is one implementation of every model
call, and a test that fails if that stops being true.

`space_live/pipeline/` and `space_live/_pooled.calibration.json` are build
artifacts and are gitignored, so a fresh clone **must** run the build before
uploading or the Space ships with no pipeline and no calibration:

```bash
python backend/data/build_space_live.py
hf upload rogerdemello/pitwall-live space_live . --repo-type=space
```

The Space must be on ZeroGPU hardware (`zero-a10g`); on CPU the wav2vec2 model
makes each request slow enough to time out. `startup_duration_timeout: 1h` in
its README frontmatter covers the first boot, which downloads three models.

Two consequences worth stating plainly. The frontend reaches it through
`NEXT_PUBLIC_LIVE_SPACE`; unset, Live Analysis falls back to explaining itself
rather than failing. And ZeroGPU time is charged to the visitor, not to the
Space, so an anonymous visitor gets a few GPU-minutes a day.

## If you later subscribe to PRO

The Docker setup is kept and verified — the image builds, runs, serves every
route and completes a live analysis in-container, which is how the `node: not
found` and blocking-warm-up bugs were caught. Use `space/README.md` (sdk:
docker) instead of `README.static.md`, and push the repo root rather than
`frontend/out`. That restores Live Analysis.

---

## Reference: the original push steps

---

## 0. Log in (once)

Create a **write** token at https://huggingface.co/settings/tokens, then:

```bash
hf auth login
```

`huggingface-cli login` also works on older versions of the CLI.

## 1. Push the dataset

```bash
python backend/data/push_to_hub.py
```

Defaults to `rogerdemello/pitwall-f1-radio-analysis`. It regenerates the files
first, so the push always matches the current corpus. The script checks you are
logged in and that you own the target before uploading, rather than failing with
a raw traceback halfway through.

What goes up: 2,042 analysed messages from 12 races — affect scores, DSI, driver
state, speaker attribution and the telemetry join — plus a dataset card that
documents the method, the valence limitation and the null result.

**Source audio is deliberately not redistributed.** It belongs to
`MikCil/f1-team-radio` (CC BY 4.0); each row carries the source `id` so anyone
can join back to it.

## 2. Create the Space

```bash
hf repo create pitwall --repo-type space --space_sdk docker
git clone https://huggingface.co/spaces/rogerdemello/pitwall hf-space
```

Then copy the app in — note `space/README.md` becomes the Space's `README.md`,
because the YAML frontmatter in it is what configures the Space:

```bash
cp -r backend frontend Dockerfile requirements.txt .dockerignore hf-space/
cp space/README.md hf-space/README.md

cd hf-space
git lfs install
git add -A && git commit -m "PIT WALL"
git push
```

The build takes a while (torch CPU is a large install). First boot then
downloads the models, but the app **serves immediately** — warm-up runs on a
background thread, and `/api/health` reports `models_warm` so you can tell
whether the first Live Analysis will be slow. Verified locally: 65s to first
response, models warm about 8 minutes later.

### Size warning

`backend/races/*.json` is a few MB and must ship — it is what makes the Replay
screen work offline. `backend/clips/` is several GB and is excluded by
`.dockerignore`; don't add it. Live Analysis works on uploads, so the Space
needs no source audio.

---

## Still outstanding for the hackathon

**Team accounts: done.** The brief requires every team member to have their own
Hugging Face account, and both do — [rogerdemello](https://huggingface.co/rogerdemello)
and [vynride](https://huggingface.co/vynride). This section used to say the
second one was still needed; it was stale, and `docs/SUBMISSION.md` had it right.

The one open item is the in-domain listening pass
(`python backend/data/label_affect.py label <race> 60`). The gold-label
validation uses acted studio speech; nobody has yet labelled *this* audio, and
the Evidence page says so rather than implying otherwise.
