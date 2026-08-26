"""The weekdays pipeline shape end to end on tiny-random, through the
real CLI: locate scan → select the best cell → DAS fit at it (via in-run
artifact refs) → apply the fitted rotation (via a run-tree file_path,
ArtifactIdentity checked) → heatmap and lines plots → save publication
and the run manifest.

The workflow references the shipped method presets with per-step ``set``
overrides that retarget model/layers/positions at tiny scale — the same
mechanism the spec gives real campaigns. The tap axis overrides to two
index positions (the subject-variable window is ragged across weekday
rows, and ragged *writes* are a stated v1 backend boundary)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from safetensors.torch import load_file

from causalab.protocol.cli import main

from tests.neural.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import FIXTURES, write_rot_fixture

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[3]
METHODS = str(
    REPO / "causalab/configs/protocols"
)  # absolute: the workflow file lives in tmp


def _tiny_workflow() -> dict:
    tiny = {"model.key": TINY_LLAMA, "model.dtype": "fp32"}
    return {
        "version": "1",
        "description": "the weekdays pipeline shape at tiny scale",
        "steps": {
            "locate": {
                "type": "protocol",
                "document": f"{METHODS}/weekdays_locate_scan.json",
                "set": {
                    **tiny,
                    "sites.target.layer": {"sweep": {"range": [0, 2]}},
                    "positions.tap": {"sweep": [{"index": -1}, {"index": -2}]},
                },
            },
            "best": {
                "type": "select",
                "from": "locate",
                "table": "iia.parquet",
                "choose": "max",
                "emit": {
                    "best_layer": "sites.target.layer",
                    "best_pos": "positions.tap",
                },
            },
            "fit": {
                "type": "protocol",
                "document": f"{METHODS}/weekdays_das_sweep.json",
                "set": {
                    **tiny,
                    "positions.best": {"artifact": "best", "key": "best_pos"},
                    "sites.target.layer": {"artifact": "best", "key": "best_layer"},
                    "featurizers.rot.k": 2,
                    "train.seed": 0,
                    "train.steps": {"epochs": 1},
                    "train.batch": {"pairs": 2},
                },
            },
            "apply": {
                "type": "protocol",
                "document": f"{METHODS}/weekdays_das_apply.json",
                "set": {
                    **tiny,
                    "sites.target.layer": {"artifact": "best", "key": "best_layer"},
                    "featurizers.rot.k": 2,
                    "featurizers.rot.file_path": "fit/rot.safetensors",
                },
            },
            "scan_heatmap": {
                "type": "plot",
                "plot": "heatmap",
                "from": "locate",
                "table": "iia.parquet",
                "x": "sites.target.layer",
                "y": "positions.tap",
                "value": "value",
                "file_path": "scan_iia.png",
            },
            "iia_lines": {
                "type": "plot",
                "plot": "lines",
                "from": "locate",
                "table": "logit_diff.parquet",
                "x": "sites.target.layer",
                "series": "positions.tap",
                "value": "value",
                "file_path": "ld_by_layer.png",
            },
        },
        "save": [
            {"step": "best", "value": "values.json", "file_path": "best_cell.json"},
            {"step": "fit", "value": "iia.parquet", "file_path": "fit_iia.parquet"},
            {"step": "apply", "value": "iia.parquet", "file_path": "apply_iia.parquet"},
            {
                "step": "scan_heatmap",
                "value": "scan_iia.png",
                "file_path": "scan_iia.png",
            },
            {
                "step": "iia_lines",
                "value": "ld_by_layer.png",
                "file_path": "ld_by_layer.png",
            },
        ],
    }


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One full CLI run of the tiny pipeline; the assertions below share it."""
    base = tmp_path_factory.mktemp("wf")
    artifacts = base / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    write_rot_fixture(artifacts)
    wf_dir = base / "workflows"
    wf_dir.mkdir()
    (wf_dir / "tiny_weekdays.json").write_text(json.dumps(_tiny_workflow(), indent=2))
    out = base / "run"
    code = main(
        [
            "run",
            str(wf_dir / "tiny_weekdays.json"),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    return out, wf_dir


def test_run_tree_holds_every_step(pipeline_run):
    out, _ = pipeline_run
    assert (out / "locate/iia.parquet").is_file()
    assert (out / "best/values.json").is_file()
    assert (out / "fit/rot.safetensors").is_file()
    assert (out / "apply/iia.parquet").is_file()
    assert (out / "scan_heatmap/scan_iia.png").stat().st_size > 0
    assert (out / "iia_lines/ld_by_layer.png").stat().st_size > 0


def test_select_chose_the_argmax_cell(pipeline_run):
    out, _ = pipeline_run
    chosen = json.loads((out / "best/values.json").read_text())
    assert chosen["best_layer"] in (0, 1)
    assert chosen["best_pos"] in ({"index": -1}, {"index": -2})
    frame = pd.read_parquet(out / "locate/iia.parquet")
    grouped = frame.groupby(["sites.target.layer", "positions.tap"])["value"].mean()
    best_key = grouped.idxmax()
    assert chosen["best_layer"] == best_key[0]
    assert chosen["best_pos"] == json.loads(best_key[1])


def test_fit_consumed_the_selected_cell_and_stamped_it(pipeline_run):
    """The in-run artifact wiring is provable from the fit bundle's
    ArtifactIdentity: the stamped site carries the layer `best` chose."""
    from causalab.protocol.resolve import read_safetensors_metadata

    out, _ = pipeline_run
    chosen = json.loads((out / "best/values.json").read_text())
    stamped = read_safetensors_metadata(out / "fit/rot.safetensors")
    assert stamped is not None
    site = json.loads(stamped["site"])
    assert site["layer"] == chosen["best_layer"]
    assert stamped["model_key"] == TINY_LLAMA
    assert stamped["k"] == "2"
    weight = load_file(str(out / "fit/rot.safetensors"))["weight"]
    assert weight.shape == (16, 2)


def test_apply_scored_the_test_split_through_the_fitted_rotation(pipeline_run):
    out, _ = pipeline_run
    iia = pd.read_parquet(out / "apply/iia.parquet")
    assert len(iia) == 2  # the weekdays/test fixture rows
    assert iia["value"].dtype.kind == "f"


def test_locate_table_carries_coordinate_columns(pipeline_run):
    out, _ = pipeline_run
    frame = pd.read_parquet(out / "locate/iia.parquet")
    assert {"sites.target.layer", "positions.tap", "value", "produced_by"} <= set(
        frame.columns
    )
    assert len(frame) == 4 * 4  # 4 points x 4 examples


def test_save_manifest_published_and_stamped(pipeline_run):
    out, wf_dir = pipeline_run
    for published in (
        "best_cell.json",
        "fit_iia.parquet",
        "apply_iia.parquet",
        "scan_iia.png",
        "ld_by_layer.png",
    ):
        assert (out / published).is_file()
    manifest = json.loads((out / "workflow.json").read_text())
    assert len(manifest["workflow_digest"]) == 64
    assert manifest["steps"]["locate"]["points"] == 4
    assert len(manifest["steps"]["locate"]["point_digests"]) == 4  # provenance units
    assert manifest["steps"]["fit"]["backend"] == "pytorch_hooks"
    assert all(entry["status"] == "completed" for entry in manifest["steps"].values())
    # the run manifest stamps the fully resolved inner digests (§7)
    assert len(manifest["steps"]["apply"]["document_digest"]) == 64
    assert manifest["steps"]["best"]["chosen"] == json.loads(
        (out / "best_cell.json").read_text()
    )


def test_explain_reports_the_derived_schedule(pipeline_run, capsys):
    _, wf_dir = pipeline_run
    code = main(
        [
            "explain",
            str(wf_dir / "tiny_weekdays.json"),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(FIXTURES / "artifacts"),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "level 0: locate" in output
    assert "authored digest" in output  # fit/apply are step-dependent
