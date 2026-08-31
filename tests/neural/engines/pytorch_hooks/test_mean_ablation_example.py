"""The shipped mean-replacement example, end to end on tiny-random.

The causal protocol's step 2 says to use zero and mean replacement as a matter
of course, and the mean is a **two-document handoff** — a harvest that reduces
its read at save time, then an ablation that swaps the resulting vector in as a
``params`` operand. That idiom worked and appeared in no shipped example, so
every run re-derived it. This runs the files a reader would copy, through the
real CLI, and checks the one thing a shipped example must promise: it runs, and
the mean is one vector rather than the whole corpus.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from safetensors.torch import load_file

from causalab.cli import main

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import FIXTURES
from tests.tables import frame as table_frame

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[4]
SHIPPED = REPO / "causalab/configs/workflows/mean_ablation.json"
PROTOCOLS = REPO / "causalab/configs/protocols"

TINY = {"model.key": TINY_LLAMA, "model.dtype": "fp32"}


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The shipped workflow, retargeted to tiny scale by per-step `set` — the
    documents themselves are the shipped bytes."""
    base = tmp_path_factory.mktemp("mean_ablation")
    document = json.loads(SHIPPED.read_text())
    for name, step in document["steps"].items():
        if step.get("type") != "intervention_protocol":
            continue
        # absolute, because the copy runs from a tmp directory
        step["document"] = str(PROTOCOLS / Path(step["document"]).name)
        step["set"] = {**step.get("set", {}), **TINY, "sites.target.layer": 0}
    wf_dir = base / "workflows"
    wf_dir.mkdir()
    path = wf_dir / "mean_ablation.json"
    path.write_text(json.dumps(document, indent=2))

    artifacts = base / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    out = base / "run"
    code = main(
        [
            "run",
            str(path),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    return out / document["output_dir"]


def test_the_harvest_saves_one_vector_not_the_corpus(run):
    """`reduce: mean` reduces **at save time**, so the un-reduced activations
    never reach disk — which is the whole reason the chain is two documents and
    not a harvest plus a notebook."""
    mean = load_file(str(run / "harvest/acts.safetensors"))["acts"]
    assert mean.ndim == 1, "the corpus mean is one vector, not a row per example"


def test_the_ablation_consumed_the_harvested_mean(run):
    """The `params` operand resolved against the run tree, and scored."""
    manifest = json.loads((run / "workflow.json").read_text())
    assert manifest["steps"]["ablate"]["status"] == "completed"
    scores = table_frame(run / "ablate/ld.json")
    assert len(scores) == 2  # the weekdays/test fixture rows
    assert scores["value"].dtype.kind == "f"
