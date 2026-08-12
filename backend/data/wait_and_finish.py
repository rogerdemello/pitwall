"""Wait for the corpus build to finish, then run the pooling step.

Chaining this through a PowerShell wrapper failed silently: Set-Content wrote a
UTF-8 BOM, PowerShell parsed it as part of the first variable name, and the
process exited instantly without running anything. Doing it in Python avoids
shell-encoding surprises entirely.

Completion is detected from the data rather than from a process handle, so this
works no matter how the build was launched or whether it was restarted: a race
is done when its raw output holds as many messages as its manifest.

Usage:
    python backend/data/wait_and_finish.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: finish_corpus is imported lazily, inside main(), *after* the wait loop.
# Importing it at module level pulls in calibrate -> pipeline -> torch and
# transformers, so a process whose only job is to sleep and poll was holding
# ~2 GB of model libraries for hours. On a 15 GB machine already running the
# build and the API that was enough to exhaust memory and kill the build.
from data.fetch_many import SLATE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP_ROOT = os.path.join(HERE, "..", "clips")
RAW_ROOT = os.path.join(HERE, "..", "raw")

POLL_S = 120
MAX_WAIT_S = 10 * 60 * 60


def race_done(race: str) -> bool:
    manifest_path = os.path.join(CLIP_ROOT, race, "manifest.json")
    raw_path = os.path.join(RAW_ROOT, f"{race}.raw.json")
    if not (os.path.exists(manifest_path) and os.path.exists(raw_path)):
        return False
    try:
        expected = len(json.load(open(manifest_path, encoding="utf-8")))
        actual = len(json.load(open(raw_path, encoding="utf-8"))["messages"])
    except (json.JSONDecodeError, KeyError):
        return False  # mid-write; try again next poll
    return actual >= expected


def main() -> None:
    # Abu Dhabi is already built and is not part of the slate list.
    expected = ["2021_Abu_Dhabi_Grand_Prix", *SLATE]
    started = time.time()

    while time.time() - started < MAX_WAIT_S:
        pending = [r for r in expected if not race_done(r)]
        if not pending:
            print(f"all {len(expected)} races built after "
                  f"{(time.time() - started) / 60:.0f} min of waiting")
            break
        print(f"waiting on {len(pending)}: {', '.join(pending)}", flush=True)
        time.sleep(POLL_S)
    else:
        print("timed out waiting for the build; pooling over whatever finished")

    # Imported here, not at module scope - see the note at the top.
    from data import finish_corpus
    finish_corpus.main()


if __name__ == "__main__":
    main()
