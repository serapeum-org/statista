"""Strip outputs and execution counts from all notebooks under ``docs/notebook/``.

Used to keep committed notebooks small: developers run this (or the nbstripout
pre-commit hook) before committing. Notebook outputs are then re-generated in CI
and injected into the build by ``scripts/prep_notebooks.py``.

Usage:
    python scripts/strip_notebook_outputs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NOTEBOOK_ROOT = Path("docs/notebook")


def strip_notebook(path: Path) -> bool:
    """Strip outputs and execution counts from a single notebook.

    Returns True if the file was modified.
    """
    text = path.read_text(encoding="utf-8")
    nb = json.loads(text)

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

    # Strip transient kernel state from notebook-level metadata.
    metadata = nb.get("metadata", {})
    if "widgets" in metadata:
        del metadata["widgets"]
        changed = True

    if changed:
        # Preserve trailing newline, indent like nbformat's default (1 space).
        path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")

    return changed


def main() -> int:
    if not NOTEBOOK_ROOT.exists():
        print(f"Notebook root {NOTEBOOK_ROOT} does not exist.", file=sys.stderr)
        return 1

    notebooks = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    modified = 0
    for nb in notebooks:
        # Skip checkpoint files.
        if ".ipynb_checkpoints" in nb.parts:
            continue
        if strip_notebook(nb):
            modified += 1
            print(f"stripped: {nb}")

    print(f"Done: {modified}/{len(notebooks)} notebooks modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
