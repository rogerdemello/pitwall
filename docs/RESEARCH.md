# Research scripts

These are not dead code. Each one answered a question that changed the build,
and the Evidence screen cites their results. They are kept runnable so any claim
on that screen can be reproduced rather than taken on trust.

| Script | Question it answered | Outcome |
|---|---|---|
| `spike_inspect.py` | What is actually in the HF dataset? | Confirmed UTC timestamps, race ids and reference transcripts — and that the published transcripts mangle F1 jargon ("supersoft" → "SuperSalt"). |
| `spike_join.py` | Do dataset timestamps land on real laps? | **Yes.** A "Box, box, box" call lands on lap 1 (SUPERSOFT, stint 1) and the next message on lap 2 (SOFT, stint 2) — telemetry independently confirms the stop. Also found that `LapStartDate` is all `NaT` unless telemetry is loaded. |
| `exp_prompting.py` | Does F1 vocabulary prompting improve ASR? | Model choice mattered; the prompt did not. distil-whisper collapses into repetition loops when prompted. Led to `whisper-small.en`. |
| `smoke_test.py` | Do ASR and prosody run on real radio at usable speed on CPU? | Yes, but revealed affect scores clustered near 0.5 — the finding that forced percentile calibration. |
| `e2e_test.py` | Does the whole chain hold together on in-race clips? | Yes. Also surfaced that DSI had no dynamic range (44–54) before calibration. |
| `spike_speaker_cluster.py` | Can we separate driver from engineer acoustically? | **No** — cleared the pre-registered bar for 1 of 4 drivers. Three rounds recorded in `races/_diarization_experiment.json`. Not shipped. |

Two of these produced negative results that changed what shipped. That is the
point of keeping them.
