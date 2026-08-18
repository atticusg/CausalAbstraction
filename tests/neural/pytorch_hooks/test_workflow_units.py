"""Runner-service units from the adversarial review: overlay shadowing is
step-name-exact, select preserves integer coordinates and honors both
choose directions, and aggregation is the group-mean the spec defines."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from causalab.protocol.resolve import FileArtifacts
from causalab.workflow.runner import OverlayArtifacts, _aggregated, _run_select_step

pytestmark = pytest.mark.unit


def _axes(*ids: str) -> tuple[SimpleNamespace, ...]:
    return tuple(SimpleNamespace(id=axis_id) for axis_id in ids)


@pytest.fixture()
def stores(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "run"
    outer_root = tmp_path / "outer"
    (run_root / "fit").mkdir(parents=True)
    (run_root / "stray").mkdir()  # a non-step directory in the run tree
    outer_root.mkdir()
    (outer_root / "stray").mkdir()
    (outer_root / "stray" / "values.json").write_text('{"k": "outer"}')
    (run_root / "fit" / "values.json").write_text('{"k": "run"}')
    (run_root / "stray" / "values.json").write_text('{"k": "SHADOWED-WRONGLY"}')
    return run_root, outer_root


class TestOverlayShadowing:
    def test_step_names_shadow(self, stores):
        run_root, outer_root = stores
        overlay = OverlayArtifacts(
            run_root=run_root,
            outer=FileArtifacts(root=outer_root),
            step_names=frozenset({"fit"}),
        )
        assert overlay.read_value("fit", "k") == "run"

    def test_directory_existence_does_not_shadow(self, stores):
        """A run-tree directory that is NOT a step (a rerun leftover, a
        stray) must not capture external refs — §3 shadows by step NAME."""
        run_root, outer_root = stores
        overlay = OverlayArtifacts(
            run_root=run_root,
            outer=FileArtifacts(root=outer_root),
            step_names=frozenset({"fit"}),
        )
        assert overlay.read_value("stray", "k") == "outer"

    def test_resolve_path_falls_through(self, stores):
        run_root, outer_root = stores
        (outer_root / "bundle.safetensors").write_bytes(b"x")
        overlay = OverlayArtifacts(
            run_root=run_root,
            outer=FileArtifacts(root=outer_root),
            step_names=frozenset({"fit"}),
        )
        assert (
            overlay.resolve_path("bundle.safetensors")
            == outer_root / "bundle.safetensors"
        )
        assert (
            overlay.resolve_path("fit/rot.safetensors")
            == run_root / "fit/rot.safetensors"
        )


class TestSelect:
    def _table(self, out: Path) -> None:
        frame = pd.DataFrame(
            {
                "featurizers.rot.k": [2, 2, 16, 16],
                "train.seed": [0, 0, 1, 1],
                "value": [0.1, 0.3, 0.8, 0.6],
                "example": [0, 1, 0, 1],
            }
        )
        (out / "fit").mkdir(parents=True)
        frame.to_parquet(out / "fit" / "iia.parquet")

    def _step(self, choose: str) -> SimpleNamespace:
        return SimpleNamespace(
            from_="fit",
            table="iia.parquet",
            choose=choose,
            value="value",
            emit={"best_k": "featurizers.rot.k", "best_seed": "train.seed"},
        )

    def test_integer_coordinates_stay_integers(self, tmp_path: Path):
        """The emitted values feed strict int fields of the next document —
        a float 16.0 would refuse at its parse (the pandas row-Series
        upcast the review demonstrated)."""
        self._table(tmp_path)
        step_dir = tmp_path / "best"
        step_dir.mkdir()
        run_axes = {"fit": _axes("featurizers.rot.k", "train.seed")}
        entry = _run_select_step(
            "best", self._step("max"), tmp_path, step_dir, run_axes
        )
        chosen = json.loads((step_dir / "values.json").read_text())
        assert chosen == {"best_k": 16, "best_seed": 1}
        assert isinstance(chosen["best_k"], int)
        assert entry["score"] == pytest.approx(0.7)  # mean over the 2 examples

    def test_choose_min(self, tmp_path: Path):
        self._table(tmp_path)
        step_dir = tmp_path / "best"
        step_dir.mkdir()
        run_axes = {"fit": _axes("featurizers.rot.k", "train.seed")}
        _run_select_step("best", self._step("min"), tmp_path, step_dir, run_axes)
        chosen = json.loads((step_dir / "values.json").read_text())
        assert chosen == {"best_k": 2, "best_seed": 0}

    def test_aggregation_is_group_mean_over_examples(self, tmp_path: Path):
        self._table(tmp_path)
        table = _aggregated(
            tmp_path,
            "fit",
            "iia.parquet",
            "value",
            {"fit": _axes("featurizers.rot.k", "train.seed")},
        )
        by_k = dict(zip(table["featurizers.rot.k"], table["value"]))
        assert by_k[2] == pytest.approx(0.2)
        assert by_k[16] == pytest.approx(0.7)

    def test_unswept_producer_single_group(self, tmp_path: Path):
        frame = pd.DataFrame({"value": [0.2, 0.4], "example": [0, 1]})
        (tmp_path / "apply").mkdir()
        frame.to_parquet(tmp_path / "apply" / "iia.parquet")
        table = _aggregated(tmp_path, "apply", "iia.parquet", "value", {"apply": ()})
        assert len(table) == 1
        assert table["value"].iloc[0] == pytest.approx(0.3)
