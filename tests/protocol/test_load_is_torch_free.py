"""``causalab validate`` of a script workflow must not import torch.

This is the property checklist rule 6 exists for: a script is **hashed, never
imported**, so a document is refused — or accepted — on a laptop with no
accelerator, before a single step runs. It is also what lets the script's
content hash sit in the digest without costing anything: hashing needs no
import.

The v1 version of this test guarded the transform-op registry's record/body
split. The registry is gone; the guarantee it protected is not, and this is
where it moved.

It has to run in a **subprocess**: ``tests/conftest.py`` imports torch at
session scope, so an in-process ``"torch" not in sys.modules`` check would be
false regardless of whether the loader behaved. The subprocess precedent is
``tests/neural/pytorch_hooks/test_end_to_end_iia.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.protocol._env import FIXTURES

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
METHODS = REPO / "causalab/configs/methods"

_PROBE = """
import json, sys
from causalab.protocol.cli import main

code = main(["validate", sys.argv[1], "--data-root", sys.argv[2],
             "--artifacts-root", sys.argv[3]])
print(json.dumps({"code": code, "torch": "torch" in sys.modules}))
"""

#: A script whose module scope imports torch. If the loader imported it, the
#: probe below would see torch in sys.modules — which is the whole point.
_TORCHY_SCRIPT = """
import torch


def main(inputs, outputs):
    outputs["out"].write_text("[]")
"""


def _workflow() -> dict:
    return {
        "version": "1",
        "output_dir": "probe",
        "steps": {
            "locate": {
                "type": "protocol",
                "document": str(METHODS / "weekdays_locate_scan.json"),
            },
            "reduce": {
                "type": "script",
                "script": "scripts/torchy.py",
                "inputs": {"table": {"step": "locate", "file": "iia.json"}},
                "outputs": {"out": "out.json"},
            },
        },
    }


def test_validate_of_a_script_workflow_never_imports_torch(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "torchy.py").write_text(_TORCHY_SCRIPT)
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps(_workflow(), indent=2))

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            str(wf),
            str(FIXTURES / "data"),
            str(FIXTURES / "artifacts"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["code"] == 0, "the document should validate"
    assert not result["torch"], (
        "validate imported torch — a script must be hashed and parsed, never "
        "imported (workflow spec §4.2)"
    )
