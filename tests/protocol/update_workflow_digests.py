"""Regenerate tests/protocol/workflow_digests.json — the shipped-workflow pins.

Run with ``uv run python tests/protocol/update_workflow_digests.py`` from the
repo root, then review the diff. Same discipline as the corpus pins: a changed
workflow digest means the workflow canonical form changed (workflow spec §7),
which is a loader migration, never a silent re-pin.

The pin exists so that "adding a step type changed no existing document" is a
test rather than a claim: a new step type that leaked a key into the canonical
form of documents that do not use it would move these digests.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol._env import FIXTURES, build_env, write_rot_fixture  # noqa: E402

from causalab.protocol.workflow import load_workflow  # noqa: E402

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "causalab/configs/workflows"


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(FIXTURES / "artifacts", tmp, dirs_exist_ok=True)
    write_rot_fixture(tmp)
    env = build_env(tmp)
    pins: dict[str, str] = {}
    for path in sorted(WORKFLOW_DIR.glob("*.json")):
        pins[path.name] = load_workflow(path, env).digest
    out = Path(__file__).parent / "workflow_digests.json"
    out.write_text(json.dumps(pins, indent=2) + "\n")
    print(f"wrote {out} ({len(pins)} workflows)")


if __name__ == "__main__":
    main()
