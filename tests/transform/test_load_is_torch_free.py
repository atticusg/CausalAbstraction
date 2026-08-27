"""``causalab validate`` of a transform workflow must not import torch.

This is the property the whole record-versus-body split exists for: a document
naming an unknown op, a bad parameter or a missing output slot is refused on a
laptop with no accelerator, before a single step runs. The registry holds the
op *records*; the numerics live inside the op function bodies.

It has to run in a **subprocess**: ``tests/conftest.py`` imports torch at
session scope, so an in-process ``"torch" not in sys.modules`` check would be
false regardless of whether the loader behaved. The subprocess precedent is
``tests/neural/pytorch_hooks/test_end_to_end_iia.py``.
"""

from __future__ import annotations

import json
import shutil
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
print(json.dumps({"code": code, "torch": "torch" in sys.modules,
                  "ops": "causalab.transform.registry" in sys.modules}))
"""


def _workflow() -> dict:
    return {
        "version": "1",
        "steps": {
            "harvest": {"type": "protocol", "document": str(METHODS / "harvest.json")},
            "fit": {
                "type": "transform",
                "op": "fit_pca@1",
                "inputs": {
                    "acts": {"step": "harvest", "value": "acts_L8_ans.safetensors"}
                },
                "params": {"k": 2},
                "outputs": {
                    "weight": "weight.safetensors",
                    "spectrum": "spectrum.json",
                },
            },
            "scree": {
                "type": "plot",
                "plot": "lines",
                "from": "fit",
                "table": "spectrum.json",
                "x": "pc",
                "value": "explained_variance_ratio",
                "file_path": "scree.png",
            },
        },
        "save": [{"step": "scree", "value": "scree.png", "file_path": "scree.png"}],
    }


def _validate(tmp_path: Path, workflow: dict) -> dict:
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps(workflow))
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(wf), str(FIXTURES / "data"), str(artifacts)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_validate_reads_op_records_without_importing_torch(tmp_path: Path) -> None:
    report = _validate(tmp_path, _workflow())
    assert report["code"] == 0
    assert report["ops"], "the registry was never consulted — the test proves nothing"
    assert not report["torch"], "validating a transform workflow imported torch"


def test_a_bad_op_is_refused_without_importing_torch(tmp_path: Path) -> None:
    """The refusal, not just the acceptance, has to be cheap: that is what
    lets an author check a document before booking an accelerator."""
    workflow = _workflow()
    workflow["steps"]["fit"]["params"]["k"] = "two"
    report = _validate(tmp_path, workflow)
    assert report["code"] != 0
    assert not report["torch"]
