"""``causalab.workflow.scripts.select`` — the reductions v1 had as spec rules.

These carry over the adversarial-review regressions from the v1 select step:
integer coordinates survive the aggregation, both choose directions work, the
aggregation is the group-mean, and an un-swept producer collapses to one row.
What changed is *where the axes come from* — the runner's ``_step.json`` record
instead of the document model (workflow spec §6) — so each test writes that
record the way the runner would.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.workflow.scripts import select
from causalab.io.step_io import StepError
from tests.step_scripts import put_sidecar, put_table, run_step

pytestmark = pytest.mark.unit

SWEPT = [
    {"featurizers.rot.k": 2, "train.seed": 0, "value": 0.1, "example": 0},
    {"featurizers.rot.k": 2, "train.seed": 0, "value": 0.3, "example": 1},
    {"featurizers.rot.k": 16, "train.seed": 1, "value": 0.8, "example": 0},
    {"featurizers.rot.k": 16, "train.seed": 1, "value": 0.6, "example": 1},
]

AXES = ["featurizers.rot.k", "train.seed"]


@pytest.fixture()
def swept(tmp_path: Path) -> Path:
    fit = tmp_path / "fit"
    put_table(fit / "iia.json", SWEPT)
    put_sidecar(fit, AXES)
    return fit / "iia.json"


def _select(table: Path, out: Path, choose: str = "max", **extra) -> dict:
    run_step(
        select,
        {
            "table": table,
            "choose": choose,
            "emit": {"best_k": "featurizers.rot.k", "best_seed": "train.seed"},
            **extra,
        },
        {"values": out},
    )
    return json.loads(out.read_text())


def test_integer_coordinates_stay_integers(swept, tmp_path):
    """The emitted values feed strict int fields of the next document — a
    float 16.0 would refuse at its parse (the pandas row-Series upcast the v1
    review demonstrated)."""
    chosen = _select(swept, tmp_path / "values.json")
    assert chosen == {"best_k": 16, "best_seed": 1}
    assert isinstance(chosen["best_k"], int)


def test_choose_min(swept, tmp_path):
    assert _select(swept, tmp_path / "values.json", choose="min") == {
        "best_k": 2,
        "best_seed": 0,
    }


def test_aggregation_is_the_group_mean_over_examples(swept, tmp_path):
    """k=16 wins on the mean (0.7) even though k=2 holds no single row above
    0.3 — averaging over examples is the reduction, not first-hit."""
    chosen = _select(swept, tmp_path / "values.json")
    assert chosen["best_k"] == 16


def test_unswept_producer_is_one_group(tmp_path):
    apply_dir = tmp_path / "apply"
    put_table(
        apply_dir / "iia.json",
        [{"value": 0.2, "example": 0}, {"value": 0.4, "example": 1}],
    )
    put_sidecar(apply_dir, [])
    run_step(
        select,
        {"table": apply_dir / "iia.json", "emit": {"v": "value"}},
        {"values": tmp_path / "values.json"},
    )
    chosen = json.loads((tmp_path / "values.json").read_text())
    assert chosen["v"] == pytest.approx(0.3)


def test_a_missing_sidecar_still_reduces(tmp_path):
    """A script may be pointed at a table no runner produced (a pinned file, a
    fixture). It then has no axes to group by, and that is not an error — with
    no `example` column either, the rows are already the unit, so the max is
    ranked as written."""
    loose = tmp_path / "loose"
    put_table(loose / "iia.json", [{"value": 1.0}, {"value": 3.0}])
    run_step(
        select,
        {"table": loose / "iia.json", "emit": {"v": "value"}},
        {"values": tmp_path / "values.json"},
    )
    assert json.loads((tmp_path / "values.json").read_text())["v"] == pytest.approx(3.0)


def test_a_script_written_table_is_ranked_as_written(tmp_path):
    """The case v1 special-cased by producer *type*: a table whose rows a
    script already decided must not be re-aggregated, or the very rows a
    consumer wants to choose between collapse. Here the discriminator is the
    data — no axes and no `example` column."""
    fit = tmp_path / "fit"
    put_table(
        fit / "spectrum.json",
        [
            {"pc": 0, "explained_variance_ratio": 0.8},
            {"pc": 1, "explained_variance_ratio": 0.2},
        ],
    )
    put_sidecar(fit, [])
    run_step(
        select,
        {
            "table": fit / "spectrum.json",
            "value": "explained_variance_ratio",
            "emit": {"best_pc": "pc"},
        },
        {"values": tmp_path / "values.json"},
    )
    assert json.loads((tmp_path / "values.json").read_text()) == {"best_pc": 0}


def test_structured_coordinates_are_decoded(tmp_path):
    """A swept position spec round-trips through the table as a JSON string;
    what lands in the values object must be the spec a document can parse."""
    scan = tmp_path / "scan"
    put_table(
        scan / "iia.json",
        [
            {"positions.tap": '{"index": -1}', "value": 0.9},
            {"positions.tap": '{"index": -2}', "value": 0.1},
        ],
    )
    put_sidecar(scan, ["positions.tap"])
    run_step(
        select,
        {"table": scan / "iia.json", "emit": {"best_pos": "positions.tap"}},
        {"values": tmp_path / "values.json"},
    )
    chosen = json.loads((tmp_path / "values.json").read_text())
    assert chosen["best_pos"] == {"index": -1}


def test_bad_choose_is_refused(swept, tmp_path):
    with pytest.raises(StepError):
        _select(swept, tmp_path / "values.json", choose="argmax")


def test_emit_column_absent_from_the_table(swept, tmp_path):
    with pytest.raises(StepError) as err:
        run_step(
            select,
            {"table": swept, "emit": {"x": "sites.ghost.layer"}},
            {"values": tmp_path / "values.json"},
        )
    assert "carried no such axis" in str(err.value)


def test_empty_table_is_refused(tmp_path):
    empty = tmp_path / "empty"
    put_table(empty / "iia.json", [])
    put_sidecar(empty, [])
    with pytest.raises(StepError) as err:
        run_step(
            select,
            {"table": empty / "iia.json", "emit": {"v": "value"}},
            {"values": tmp_path / "values.json"},
        )
    assert "no rows" in str(err.value)
