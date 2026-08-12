"""Strip execution state from notebooks before they are committed.

Opening a notebook in VS Code, Jupyter or Colab and running a cell writes the
output back into the file. That output is then committed, so the repository
carries whatever happened to be on screen at the time - which in practice means
stale results, machine-specific paths, or, as happened here, a FileNotFoundError
traceback from a runtime with no GPU sitting in the build tool a judge might open.

Notebook source is the artifact worth versioning. Execution state is not.

    python backend/tools/clean_notebooks.py           # strip
    python backend/tools/clean_notebooks.py --check   # fail if any are dirty
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def clean(nb: dict) -> tuple[dict, int]:
    """Drop outputs and execution counts. Returns the notebook and what changed."""
    dirty = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            dirty += 1
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            dirty += 1
    # Colab stamps a per-session id that changes on every open, producing a diff
    # that says nothing about the notebook.
    nb.get("metadata", {}).pop("widgets", None)
    (nb.get("metadata", {}).get("colab") or {}).pop("authorship_tag", None)
    return nb, dirty


def main(check_only: bool = False) -> int:
    notebooks = sorted(p for p in ROOT.rglob("*.ipynb")
                       if "node_modules" not in p.parts
                       and ".ipynb_checkpoints" not in p.parts)
    if not notebooks:
        print("no notebooks found")
        return 0

    dirty_files = []
    for path in notebooks:
        nb = json.loads(path.read_text(encoding="utf-8"))
        nb, dirty = clean(nb)
        rel = path.relative_to(ROOT)
        if dirty:
            dirty_files.append((rel, dirty))
            if not check_only:
                path.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    if check_only:
        if dirty_files:
            print(f"!! {len(dirty_files)} notebook(s) carry execution state:")
            for rel, n in dirty_files:
                print(f"     {rel}  ({n} cell(s))")
            print("   run: python backend/tools/clean_notebooks.py")
            return 1
        print(f"all {len(notebooks)} notebook(s) are clean")
        return 0

    if dirty_files:
        for rel, n in dirty_files:
            print(f"cleaned {rel}  ({n} cell(s))")
    else:
        print(f"all {len(notebooks)} notebook(s) were already clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(check_only="--check" in sys.argv))
