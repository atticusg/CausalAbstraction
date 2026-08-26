"""A split run document executes, end to end, on tiny-random (CPU).

The composition tests in ``tests/protocol/test_method.py`` prove that an
``application`` + ``method`` document *is* a protocol document. This one
proves the boring half of the same claim: nothing downstream — planner,
executor, stamping — has to know, and the run's record names the method.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from causalab.protocol.cli import main

from tests.protocol._env import FIXTURES
from tests.neural.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[3]
RUN_DOCUMENT = REPO / "causalab/configs/runs/weekdays_8b_interchange.json"


@pytest.fixture(scope="module")
def artifacts_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    return root


def test_the_shipped_run_document_runs_and_records_its_method(artifacts_root, tmp_path):
    out = tmp_path / "out"
    code = main(
        [
            "run",
            str(RUN_DOCUMENT),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts_root),
            "--out",
            str(out),
            "--set",
            f"model.key={TINY_LLAMA}",
            "--set",
            "sites.target.layer=1",  # tiny-random is 2 layers deep
            "--dtype",
            "fp32",  # tiny-random on CPU; the shipped realization is bf16
        ]
    )
    assert code == 0

    iia = pd.read_parquet(out / "iia.parquet")["value"]
    assert len(iia) == 4  # the fixture table's rows
    assert set(iia.unique()) <= {0.0, 1.0}

    record = json.loads((out / "protocol.json").read_text())
    assert record["method"]["ref"] is None  # inlined: one file is one run
    assert len(record["method"]["digest"]) == 64
    assert record["canonical"]["model"]["dtype"] == "fp32"
    assert record["canonical"]["sites"]["target"]["layer"] == 1
