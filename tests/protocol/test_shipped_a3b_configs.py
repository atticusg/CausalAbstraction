"""The shipped qwen3.6-35b-a3b protocol set loads, expands, and asks the
engine for exactly the capabilities its components imply.

These are the runnable counterparts of the six exploratory/testing methods the
Silico causalab documents describe, retargeted onto a hybrid MoE tower. They
are pinned here on *shape* rather than on numbers: the numbers need the 67 GB
checkpoint and belong in the golden tier, but "does the document still express
the experiment" is a load-time question and this is where it is cheap.

The point counts are the ones the campaign digests cover, so a change here is
either an intended edit to a shipped config or an unintended change to sweep
expansion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causalab.protocol.loader import load
from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv

pytestmark = pytest.mark.unit

CONFIGS = Path(__file__).resolve().parents[2] / "causalab" / "configs"
PROTOCOLS = CONFIGS / "protocols" / "qwen3_6_a3b"
WORKFLOW = CONFIGS / "workflows" / "qwen3_6_a3b_weekdays.json"

#: file stem -> expanded point count. `das_apply` is absent on purpose: it
#: loads a fitted rotation from the `fit` step's output, so it only resolves
#: inside the workflow that produces one.
POINTS = {
    "probe": 1,
    "interchange": 1,
    "control_positive": 1,
    "control_negative": 1,
    "locate_scan": 40,  # 20 layers x 2 positions
    "logit_lens": 77,  # 11 source layers x 7 prompt positions
    "knockout_head": 160,  # the TEN full-attention layers x 16 heads
    "knockout_mlp": 40,  # every layer of the tower
    "knockout_mlp_band3": 1,
    "head_attribution": 1,
    "harvest": 1,
    "das_sweep": 6,  # 3 ranks x 2 seeds
}


def _env() -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=CONFIGS / "data"),
        artifacts=FileArtifacts(root=CONFIGS),
    )


def test_every_shipped_document_is_covered_here():
    """A new file in the directory must land in POINTS, or it ships untested."""
    on_disk = {p.stem for p in PROTOCOLS.glob("*.json")}
    assert on_disk == set(POINTS) | {"das_apply"}


@pytest.mark.parametrize("stem", sorted(POINTS))
def test_a_shipped_document_loads_and_expands(stem: str):
    loaded = load(PROTOCOLS / f"{stem}.json", _env())
    assert len(loaded.expansion.points) == POINTS[stem]


def test_the_attention_scan_names_only_full_attention_layers():
    """The tower is 3:1 linear:full, so a head scan that swept the whole depth
    would refuse at 30 of 40 layers — architecturally, not as a gap. The
    document names the ten it can, and this is the check that keeps it in step
    with the registry entry's `num_layers`."""
    from causalab.protocol.registry import get_model_info

    loaded = load(PROTOCOLS / "knockout_head.json", _env())
    info = get_model_info("Qwen/Qwen3.6-35B-A3B")
    layers = {doc.sites["attn"].layer for doc in loaded.point_documents}
    assert layers == {3, 7, 11, 15, 19, 23, 27, 31, 35, 39}
    assert max(layers) == info.num_layers - 1


def test_the_workflow_chain_loads():
    from causalab.workflow.document import load_workflow

    workflow = load_workflow(WORKFLOW, _env())
    assert set(workflow.document.steps) == {
        "locate",
        "best",
        "fit",
        "best_fit",
        "apply",
        "scan_heatmap",
        "iia_by_k",
    }
