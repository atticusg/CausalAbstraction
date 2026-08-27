"""``causalab:plot`` — both kinds, and the axis-coverage refusal.

The interesting assertion is the last one. v1 refused an uncovered sweep axis at
*load*; with the axes published as data rather than derived in the document
model, the refusal moved to the step. It had to survive the move: an uncovered
axis silently averages over a dimension the reader cannot see, which is a wrong
figure rather than a missing one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causalab.protocol.tables import read_table
from causalab.steps.builtin import plot
from causalab.steps.io import StepError
from tests.steps._run import put_sidecar, put_table, run_step

pytestmark = pytest.mark.unit

GRID = [
    {"sites.target.layer": 1, "positions.tap": 0, "value": 0.1, "example": 0},
    {"sites.target.layer": 1, "positions.tap": 0, "value": 0.3, "example": 1},
    {"sites.target.layer": 2, "positions.tap": 0, "value": 0.8, "example": 0},
    {"sites.target.layer": 2, "positions.tap": 1, "value": 0.5, "example": 0},
    {"sites.target.layer": 1, "positions.tap": 1, "value": 0.2, "example": 0},
]
AXES = ["sites.target.layer", "positions.tap"]


@pytest.fixture()
def grid(tmp_path: Path) -> Path:
    scan = tmp_path / "scan"
    put_table(scan / "iia.json", GRID)
    put_sidecar(scan, AXES)
    return scan / "iia.json"


def test_heatmap_writes_the_image_and_the_plotted_table(grid, tmp_path):
    """The declared output is the data; the image lands beside it, because a
    .png is neither of the two formats (§2.5)."""
    out = tmp_path / "out" / "scan.json"
    run_step(
        plot,
        {
            "table": grid,
            "plot": "heatmap",
            "x": "sites.target.layer",
            "y": "positions.tap",
            "figure": "scan.png",
        },
        {"plotted": out},
    )
    assert out.is_file()
    assert (out.parent / "scan.png").is_file()
    rows = read_table(out)
    # one row per (layer, tap) cell, mean over examples
    assert len(rows) == 4
    cell = next(
        r for r in rows if r["sites.target.layer"] == 1 and r["positions.tap"] == 0
    )
    assert cell["value"] == pytest.approx(0.2)


def test_lines_with_a_series(tmp_path):
    fit = tmp_path / "fit"
    put_table(
        fit / "iia.json",
        [
            {"featurizers.rot.k": 2, "train.seed": 0, "value": 0.1},
            {"featurizers.rot.k": 4, "train.seed": 0, "value": 0.4},
            {"featurizers.rot.k": 2, "train.seed": 1, "value": 0.2},
            {"featurizers.rot.k": 4, "train.seed": 1, "value": 0.5},
        ],
    )
    put_sidecar(fit, ["featurizers.rot.k", "train.seed"])
    out = tmp_path / "out" / "curve.json"
    run_step(
        plot,
        {
            "table": fit / "iia.json",
            "plot": "lines",
            "x": "featurizers.rot.k",
            "series": "train.seed",
            "figure": "curve.png",
        },
        {"plotted": out},
    )
    assert (out.parent / "curve.png").is_file()


def test_an_uncovered_axis_is_refused(grid, tmp_path):
    with pytest.raises(StepError) as err:
        run_step(
            plot,
            {"table": grid, "plot": "lines", "x": "sites.target.layer"},
            {"plotted": tmp_path / "out.json"},
        )
    assert "positions.tap" in str(err.value)
    assert "uncovered" in str(err.value)


def test_heatmap_needs_y(grid, tmp_path):
    with pytest.raises(StepError):
        run_step(
            plot,
            {"table": grid, "plot": "heatmap", "x": "sites.target.layer"},
            {"plotted": tmp_path / "out.json"},
        )


def test_unknown_kind_is_refused(grid, tmp_path):
    with pytest.raises(StepError):
        run_step(
            plot,
            {"table": grid, "plot": "violin", "x": "sites.target.layer"},
            {"plotted": tmp_path / "out.json"},
        )


def test_figure_must_be_a_bare_filename(grid, tmp_path):
    with pytest.raises(StepError):
        run_step(
            plot,
            {
                "table": grid,
                "plot": "heatmap",
                "x": "sites.target.layer",
                "y": "positions.tap",
                "figure": "../escape.png",
            },
            {"plotted": tmp_path / "out.json"},
        )
