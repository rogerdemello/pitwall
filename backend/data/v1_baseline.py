"""Record what v1 does, before the GPU rebuild changes it.

An improvement you cannot compare against a recorded baseline is an assertion.
This freezes the measurable behaviour of the current pipeline - whisper-small.en,
greedy, no VAD, 15s mean-pooled prosody windows, lexicon speaker attribution - so
that v2 can be scored against it on the identical 2,042 clips rather than
described in adjectives.

Three defects are measured here rather than argued about:

  Truncation. `WhisperProcessor` pads *or truncates* to 30.0s (n_samples=480000)
  and asr.py passes whole clips with no chunking. Clips longer than that are
  silently cut. Prosody meanwhile reads the full clip, so on those messages the
  transcript and the voice describe different audio - and the incongruence
  detector compares one against the other.

  Hallucination on non-speech. Whisper invents text on near-silence. The
  dataset's own published transcripts have the same problem, so the reference
  cannot be used to detect it.

  Scored noise. A clip that is really just squelch still gets a transcript, a
  sentiment, a DSI, a state and a place in the calibration distribution.

Usage:
    python backend/data/v1_baseline.py
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.artifacts import iter_race_files  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = os.path.join(HERE, "..", "races")
RACES_V1 = os.path.join(HERE, "..", "races_v1")
OUT = os.path.join(RACES, "_v1_baseline.json")

#: Whisper's feature extractor pads or truncates to exactly this many seconds.
WHISPER_WINDOW_S = 30.0

#: Artifact families Whisper emits on non-speech. `bare_you` dominates by an
#: order of magnitude and is the one that matters: it is what a squelch burst
#: decodes to.
ARTIFACTS = {
    "bare_you": r"(?i)^\W*you\W*$",
    "thanks_only": r"(?i)^\W*thank(s| you)\W*$",
    "subtitle_credit": r"(?i)(subtitle|amara\.org|thanks for watching|subscribe|www\.)",
    "bracketed_only": r"^\W*[\[(].*[\])]\W*$",
}


def has_repetition_loop(text: str, reps: int = 5, max_gram: int = 4) -> bool:
    """A 1-4 word phrase repeated `reps` times back to back.

    This is what an unguarded decode does when it loses the audio. v1 has no
    compression_ratio_threshold, so nothing catches it.
    """
    w = text.split()
    for n in range(1, max_gram + 1):
        for i in range(len(w) - n * reps + 1):
            gram = w[i:i + n]
            if all(w[i + k * n:i + (k + 1) * n] == gram for k in range(reps)):
                return True
    return False


def load_messages(races_dir: str | None = None) -> list[dict]:
    """Every message in a race tree.

    The directory is a parameter so `v2_scoring.py` can run the identical audit
    over `races_v1/` and `races/` and compare the results. A metric definition
    that drifts between a baseline and its follow-up manufactures exactly the
    kind of false positive this project keeps catching, so there is one
    definition and it is this one.
    """
    msgs = []
    for path in iter_race_files(races_dir or RACES):
        d = json.load(open(path, encoding="utf-8"))
        for m in d["messages"]:
            m["_race"] = d["race_id"]
            msgs.append(m)
    return msgs


def audit(msgs: list[dict]) -> dict:
    n = len(msgs)

    # --- Truncation -------------------------------------------------------
    over = [m for m in msgs if (m.get("duration_s") or 0) > WHISPER_WINDOW_S]
    lost_s = sum(m["duration_s"] - WHISPER_WINDOW_S for m in over)
    truncation = {
        "whisper_window_s": WHISPER_WINDOW_S,
        "clips_truncated": len(over),
        "share": round(len(over) / n, 4),
        "audio_never_transcribed_min": round(lost_s / 60, 1),
        "longest_clip_s": round(max((m.get("duration_s") or 0) for m in msgs), 1),
        "worst_examples": sorted(
            ({"race": m["_race"], "id": m["id"], "duration_s": m["duration_s"],
              "transcribed_share": round(WHISPER_WINDOW_S / m["duration_s"], 2)}
             for m in over), key=lambda r: -r["duration_s"])[:5],
        "note": (
            "asr.py passes whole clips to WhisperProcessor with no chunking, and "
            "the feature extractor truncates to 30s. prosody.analyse reads the "
            "full clip, so on these messages the transcript and the voice "
            "describe different audio."
        ),
    }

    # --- Hallucination ----------------------------------------------------
    families = collections.Counter()
    flagged: dict[str, dict] = {}
    for m in msgs:
        t = (m.get("transcript") or "").strip()
        for name, pat in ARTIFACTS.items():
            if re.search(pat, t):
                families[name] += 1
                flagged[m["id"]] = {"family": name, "duration_s": m.get("duration_s"),
                                    "transcript": t}
                break
        else:
            if has_repetition_loop(t):
                families["repetition_loop"] += 1
                flagged[m["id"]] = {"family": "repetition_loop",
                                    "duration_s": m.get("duration_s"),
                                    "transcript": t[:120]}

    # The dataset's published transcripts were themselves machine-produced and
    # hallucinate on the same clips, so WER against them cannot detect this.
    ref_agrees = sum(
        1 for m in msgs if m["id"] in flagged
        and re.search(ARTIFACTS["thanks_only"], (m.get("reference_transcription") or "").strip())
    )
    hallucination = {
        "flagged": len(flagged),
        "rate": round(len(flagged) / n, 4),
        "by_family": dict(families.most_common()),
        "median_duration_s": _median([v["duration_s"] for v in flagged.values()]),
        "reference_also_hallucinates": ref_agrees,
        "note": (
            "The published reference transcripts hallucinate on the same clips - "
            "a 0.9s squelch burst is transcribed 'Thank you.' in both. WER "
            "against that reference therefore cannot detect this class of error, "
            "which is why a gold set is needed to score the upgrade."
        ),
        "guard_in_v1": "none - v1 decodes greedily with no no_speech_threshold, "
                       "no compression_ratio_threshold and no temperature fallback",
    }

    # --- Scored noise -----------------------------------------------------
    # Everything downstream runs regardless: these clips carry a DSI, a state,
    # a text sentiment, and a slot in the calibration reference distribution.
    noise_states = collections.Counter(
        m.get("state") for m in msgs if m["id"] in flagged)
    scored_noise = {
        "clips": len(flagged),
        "share_of_corpus": round(len(flagged) / n, 4),
        "still_assigned_a_state": dict(noise_states.most_common()),
        "still_in_calibration_reference": True,
        "note": (
            "These clips are scored end to end. The largest family decodes to the "
            "single word 'you', on which the text sentiment model returns a fixed "
            "polarity, and the prosody model reads channel noise."
        ),
    }

    # --- Short clips ------------------------------------------------------
    short = [m for m in msgs if (m.get("duration_s") or 0) < 2.0]
    shorts = {
        "clips_under_2s": len(short),
        "share": round(len(short) / n, 4),
        "unknown_speaker_share": round(
            sum(1 for m in short if m.get("speaker") == "unknown") / len(short), 3
        ) if short else None,
        "note": "speaker.py returns 'unknown' below 2 words, so most of these are "
                "unattributed regardless of content.",
    }

    # --- Speaker attribution ---------------------------------------------
    spk = collections.Counter(m.get("speaker") for m in msgs)
    speaker = {
        "counts": dict(spk.most_common()),
        "shares": {k: round(v / n, 3) for k, v in spk.most_common()},
        "unknown_share": round(spk.get("unknown", 0) / n, 3),
        "fed_to_strategy_as_driver_voiced": round(
            sum(v for k, v in spk.items() if k != "engineer") / n, 3),
        "note": "strategy.py filters only on speaker != 'engineer', so unknowns "
                "are treated as possibly the driver.",
    }

    # --- Index distributions ---------------------------------------------
    dsis = [m["dsi"] for m in msgs]
    states = collections.Counter(m.get("state") for m in msgs)
    index = {
        "dsi_mean": round(sum(dsis) / n, 2),
        "dsi_sd": round(_sd(dsis), 2),
        "dsi_min": min(dsis), "dsi_max": max(dsis),
        "dsi_p05": _pct(dsis, 0.05), "dsi_p95": _pct(dsis, 0.95),
        "state_counts": dict(states.most_common()),
        "suppressed_stress": sum(1 for m in msgs if m.get("suppressed_stress")),
        "recommendations": sum(1 for m in msgs if m.get("recommendation")),
    }

    return {
        "generated_by": "backend/data/v1_baseline.py",
        "purpose": (
            "Frozen record of v1 behaviour, so the GPU rebuild can be scored "
            "against it on the identical clips rather than described in adjectives."
        ),
        "pipeline": {
            "asr": "openai/whisper-small.en, greedy, max_new_tokens=128, no VAD, "
                   "no chunking, no hallucination guards",
            "prosody": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim, "
                       "15s windows, mean-pooled over the whole clip, CPU",
            "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "speaker": "lexicon and regex, clip-level, no audio",
            "fusion": "0.55*a + 0.45*(1-v) - 0.10*(d-0.5), hand-set",
        },
        "n_messages": n,
        "n_races": len({m["_race"] for m in msgs}),
        "truncation": truncation,
        "hallucination": hallucination,
        "scored_noise": scored_noise,
        "short_clips": shorts,
        "speaker_attribution": speaker,
        "index": index,
        "what_v2_must_beat": {
            "clips_truncated": 0,
            "hallucination_rate": f"<= {round(len(flagged) / n, 4)}",
            "unknown_speaker_share": f"< {round(spk.get('unknown', 0) / n, 3)}",
            "note": "Any of these getting worse is a regression and must be "
                    "reported as one, not dropped.",
        },
    }


def _sd(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[len(xs) // 2], 1) if xs else None


def main(force: bool = False) -> None:
    # Refuse to re-derive the baseline once v1 has been frozen.
    #
    # This script reads races/, which after the v2 import holds v2. Running it
    # then would overwrite the recorded v1 defects - 92 truncated clips, a
    # 0.0269 hallucination rate, a 0.525 unknown share - with v2's own numbers
    # and label them the baseline. H1, H2 and H3 are all scored against those
    # three figures, so the pre-registration would silently start comparing v2
    # against itself. Nothing would error and every hypothesis would pass.
    if os.path.isdir(RACES_V1) and os.path.exists(OUT) and not force:
        raise SystemExit(
            f"refusing to overwrite {os.path.basename(OUT)}.\n"
            "\n"
            f"{RACES_V1} exists, so v1 is frozen and races/ now holds a later\n"
            "pipeline. Re-deriving the baseline from it would score v2 against\n"
            "itself.\n"
            "\n"
            "  To audit the current tree instead:  python backend/data/v2_scoring.py\n"
            "  To overwrite anyway (you will need a reason):  --force\n"
        )

    msgs = load_messages()
    if not msgs:
        print("no races built")
        return
    a = audit(msgs)

    print(f"v1 baseline over {a['n_messages']} messages, {a['n_races']} races\n")
    t = a["truncation"]
    print(f"TRUNCATION      {t['clips_truncated']:>4} clips over {t['whisper_window_s']:.0f}s "
          f"({t['share'] * 100:.1f}%) - {t['audio_never_transcribed_min']} min never "
          f"transcribed, longest clip {t['longest_clip_s']}s")
    h = a["hallucination"]
    print(f"HALLUCINATION   {h['flagged']:>4} clips ({h['rate'] * 100:.1f}%), "
          f"median duration {h['median_duration_s']}s")
    for fam, c in h["by_family"].items():
        print(f"                  {c:>4}  {fam}")
    print(f"                  reference transcripts hallucinate on "
          f"{h['reference_also_hallucinates']} of the same clips")
    s = a["speaker_attribution"]
    print(f"SPEAKER         unknown {s['unknown_share'] * 100:.1f}%, "
          f"{s['fed_to_strategy_as_driver_voiced'] * 100:.1f}% reach strategy as "
          "possibly-driver")
    print(f"                  {s['counts']}")
    n = a["scored_noise"]
    print(f"SCORED NOISE    {n['clips']} clips carry a DSI and a state: "
          f"{n['still_assigned_a_state']}")
    i = a["index"]
    print(f"INDEX           DSI mean {i['dsi_mean']} sd {i['dsi_sd']} "
          f"[{i['dsi_min']}-{i['dsi_max']}], p05-p95 {i['dsi_p05']}-{i['dsi_p95']}")
    print(f"                  {i['state_counts']}")
    print(f"                  {i['suppressed_stress']} suppressed-stress, "
          f"{i['recommendations']} recommendations")

    json.dump(a, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
