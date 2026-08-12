# Publishing to Hugging Face

**Both artifacts are live under `rogerdemello`:**

- Dataset — https://huggingface.co/datasets/rogerdemello/pitwall-f1-radio-analysis
- Space — https://huggingface.co/spaces/rogerdemello/pitwall

## What shipped, and why it's a *static* Space

Docker Spaces now require a PRO subscription (`402 Payment Required`: *"Static
Spaces are free for everyone, but hosting Gradio and Docker Spaces on free
cpu-basic requires a PRO subscription"*). Static Spaces are free, and almost
nothing here needs a server — Replay and Evidence read precomputed JSON, and
only Live Analysis runs a model.

So `build_static_site.py` freezes every GET endpoint to a file (`/api/<x>` →
`/data/<x>.json`, one uniform rewrite rather than a per-endpoint table), the
frontend builds with `NEXT_PUBLIC_STATIC=1`, and Live Analysis shows an
explanation instead of failing.

Two things worth knowing if you redeploy:

- **The static host serves exact file paths only.** It resolves `index.html` at
  the root but *not* in subdirectories, so `/evidence/` 404s and the browser
  falls back to huggingface.co. Static builds therefore link at
  `/evidence/index.html`. This is handled in `Nav.tsx`; don't "tidy" it.
- **`short_description` must be ≤ 60 characters** or `upload_folder` rejects the
  whole push on YAML validation.

To rebuild and redeploy:

```bash
python backend/data/build_static_site.py
cd frontend
NEXT_PUBLIC_API="" NEXT_PUBLIC_STATIC=1 NEXT_OUTPUT=export npx next build
cp ../space/README.static.md out/README.md
# then upload out/ to spaces/rogerdemello/pitwall
```

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

**The brief requires every team member to have their own Hugging Face account.**
`rogerdemello` is one. Your second team member needs one too — that's a hard
submission rule, not a nice-to-have.

The other open item is the in-domain listening pass
(`python backend/data/label_affect.py label <race> 60`). The gold-label
validation uses acted studio speech; nobody has yet labelled *this* audio, and
the Evidence page says so rather than implying otherwise.
