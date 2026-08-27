"""A ``transform`` step end to end on tiny-random, through the real CLI.

Two directions in one pipeline, because they are the two halves of the
claim that a transform op is a first-class step and not a new mechanism:

* **protocol → transform → select/plot** — a harvested activation bundle is
  fitted by ``fit_pca@1``; the spectrum table it writes is then ranked by a
  ``select`` step and drawn by a ``plot`` step, against the columns the op
  *declared* (there are no sweep axes to group by).
* **protocol → transform → protocol** — the fitted basis is loaded back by a
  later protocol step as a ``pca`` featurizer through the ordinary run-tree
  ``file_path`` overlay, with its ArtifactIdentity checked. That is the
  handoff the manifold pipelines are built on, and the reason the transform
  layer stamps identity rather than writing a bare tensor.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from safetensors.torch import load_file

from causalab.protocol.cli import main
from causalab.protocol.resolve import read_safetensors_metadata

from tests.neural.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import FIXTURES
from tests.tables import frame as table_frame

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[3]
METHODS = str(REPO / "causalab/configs/methods")  # absolute: the workflow is in tmp

#: A pure-read document that loads the fitted basis as a ``pca`` featurizer.
#: Its ``file_path`` names the transform step's run tree, which is what makes
#: the edge a derived dependency (§3) — nothing about it is transform-specific.
PROJECT_DOC = {
    "version": "1",
    "description": "read L0 through the fitted PCA basis",
    "model": {"key": TINY_LLAMA, "revision": "main"},
    "data": {"base": {"dataset": "weekdays/train", "field": "input"}},
    "positions": {"answer_tok": {"index": -1}},
    "sites": {"L0": {"component": "block_output", "layer": 0}},
    "featurizers": {
        "basis": {"kind": "pca", "k": 2, "file_path": "fit/weight.safetensors"}
    },
    "reads": {
        "coords": {
            "site": "L0",
            "pos": "answer_tok",
            "model": "original",
            "input": "base",
            "featurizer": "basis",
        }
    },
    "save": [
        {
            "value": "coords",
            "model": "original",
            "input": "base",
            "file_path": "coords.safetensors",
        }
    ],
}


def _workflow(project_doc: Path) -> dict:
    tiny = {"model.key": TINY_LLAMA}
    return {
        "version": "1",
        "description": "harvest -> fit_pca -> (select, plot) and back into a protocol step",
        "steps": {
            "harvest": {
                "type": "protocol",
                "document": f"{METHODS}/harvest.json",
                "set": {**tiny, "sites.L8.layer": 0, "sites.L24.layer": 1},
            },
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
            "top_pc": {
                "type": "select",
                "from": "fit",
                "table": "spectrum.json",
                "choose": "max",
                "value": "explained_variance_ratio",
                "emit": {"best_pc": "pc"},
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
            "project": {"type": "protocol", "document": str(project_doc)},
        },
        "save": [
            {
                "step": "fit",
                "value": "spectrum.json",
                "file_path": "spectrum.json",
            },
            {
                "step": "fit",
                "value": "weight.safetensors",
                "file_path": "basis.safetensors",
            },
            {"step": "top_pc", "value": "values.json", "file_path": "top_pc.json"},
            {"step": "scree", "value": "scree.png", "file_path": "scree.png"},
            {
                "step": "project",
                "value": "coords.safetensors",
                "file_path": "coords.safetensors",
            },
        ],
    }


@pytest.fixture(scope="module")
def transform_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One full CLI run of the pipeline; the assertions below share it."""
    root = tmp_path_factory.mktemp("transform-wf")
    artifacts = root / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    project_doc = root / "project.json"
    project_doc.write_text(json.dumps(PROJECT_DOC, indent=2))
    wf_path = root / "wf.json"
    wf_path.write_text(json.dumps(_workflow(project_doc), indent=2))
    out = root / "run"
    code = main(
        [
            "run",
            str(wf_path),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    return out


def test_transform_step_writes_its_declared_outputs(transform_run: Path) -> None:
    assert (transform_run / "fit" / "weight.safetensors").is_file()
    assert (transform_run / "fit" / "spectrum.json").is_file()
    weight = load_file(str(transform_run / "fit" / "weight.safetensors"))["weight"]
    # (d, k): the featurizer convention, so a protocol step can load it as-is
    assert weight.shape == (16, 2)


def test_spectrum_table_matches_the_declared_columns(transform_run: Path) -> None:
    spectrum = table_frame(transform_run / "fit" / "spectrum.json")
    assert list(spectrum.columns) == [
        "pc",
        "explained_variance",
        "explained_variance_ratio",
    ]
    assert len(spectrum) == 2  # k = 2
    assert spectrum["explained_variance_ratio"].is_monotonic_decreasing


def test_select_ranks_the_transform_table_as_written(transform_run: Path) -> None:
    """The rows are the op's, not a re-aggregation: the winning pc is 0, the
    leading component, which a collapse-to-one-row mean could not name."""
    values = json.loads((transform_run / "top_pc" / "values.json").read_text())
    assert values == {"best_pc": 0}


def test_fitted_basis_carries_a_checkable_identity(transform_run: Path) -> None:
    metadata = read_safetensors_metadata(transform_run / "fit" / "weight.safetensors")
    assert metadata is not None
    # inherited from the harvested activations, so the basis is provably a fit
    # on this model at the site the read came from
    assert metadata["model_key"] == TINY_LLAMA
    assert json.loads(metadata["site"]) == {"component": "block_output", "layer": 0}
    # from the op's params, and from the step itself
    assert metadata["k"] == "2"
    assert metadata["dtype"] == "fp32"
    assert metadata["backend"] == "transform"
    assert len(metadata["produced_by"]) == 64


def test_protocol_step_consumes_the_fitted_basis(transform_run: Path) -> None:
    """The transform → protocol direction: the identity check on the loaded
    featurizer is a real one, and the read comes back in the 2-d feature
    space the basis defines."""
    coords = load_file(str(transform_run / "project" / "coords.safetensors"))["coords"]
    assert coords.shape == (4, 1, 2)  # 4 examples x 1 position x k


def test_manifest_records_the_transform_step(transform_run: Path) -> None:
    manifest = json.loads((transform_run / "workflow.json").read_text())
    fit = manifest["steps"]["fit"]
    assert fit["type"] == "transform"
    assert fit["op"] == "fit_pca@1"
    assert fit["status"] == "completed"
    assert len(fit["digest"]) == 64
    assert fit["files"] == ["spectrum.json", "weight.safetensors"]
    assert sorted(manifest["published"]) == [
        "basis.safetensors",
        "coords.safetensors",
        "scree.png",
        "spectrum.json",
        "top_pc.json",
    ]
