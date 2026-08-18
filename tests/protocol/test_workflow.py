"""The workflow-protocol document model (docs/workflow_protocol.md §5):
parse rules, the derived schedule, digest semantics, and the shipped
weekdays-8b worked example."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.workflow import (
    WorkflowError,
    is_workflow,
    load_workflow,
    parse_workflow,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
WEEKDAYS_WF = REPO / "causalab/configs/workflows/weekdays_8b.json"


def tiny_workflow(tmp_path: Path) -> dict[str, Any]:
    """A minimal two-step workflow over a copied method preset."""
    methods = tmp_path / "methods"
    methods.mkdir(exist_ok=True)
    shutil.copyfile(
        REPO / "causalab/configs/methods/weekdays_locate_scan.json",
        methods / "locate.json",
    )
    return {
        "version": "1",
        "steps": {
            "locate": {"type": "protocol", "document": "methods/locate.json"},
            "best": {
                "type": "select",
                "from": "locate",
                "table": "iia.parquet",
                "choose": "max",
                "emit": {"best_layer": "sites.target.layer"},
            },
        },
        "save": [{"step": "best", "value": "values.json", "file_path": "best.json"}],
    }


def expect_rule(rule: int, raw: dict[str, Any], env, tmp_path: Path) -> WorkflowError:
    with pytest.raises(WorkflowError) as err:
        load_workflow(raw, env, workflow_dir=tmp_path)
    assert err.value.rule == rule, f"expected W{rule}, got {err.value}"
    return err.value


# --------------------------------------------------------------------------- #
# parse rules
# --------------------------------------------------------------------------- #


def test_is_workflow_dispatches_on_steps():
    assert is_workflow({"version": "1", "steps": {}, "save": []})
    assert not is_workflow({"version": "1", "model": {}, "save": []})


def test_rule_1_unknown_step_type():
    raw = {
        "version": "1",
        "steps": {"a": {"type": "protocols", "document": "x.json"}},
        "save": [{"step": "a", "value": "x", "file_path": "x"}],
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 1 and "protocol" in str(err.value)


def test_rule_2_save_must_be_last():
    raw = {
        "version": "1",
        "save": [{"step": "a", "value": "x", "file_path": "x"}],
        "steps": {"a": {"type": "protocol", "document": "x.json"}},
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 2


def test_rule_3_step_names_filesystem_safe():
    raw = {
        "version": "1",
        "steps": {"a/b": {"type": "protocol", "document": "x.json"}},
        "save": [{"step": "a/b", "value": "x", "file_path": "x"}],
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 3


def test_rule_3_duplicate_save_paths():
    raw = {
        "version": "1",
        "steps": {"a": {"type": "protocol", "document": "x.json"}},
        "save": [
            {"step": "a", "value": "x", "file_path": "same"},
            {"step": "a", "value": "y", "file_path": "same"},
        ],
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 3


# --------------------------------------------------------------------------- #
# load rules
# --------------------------------------------------------------------------- #


def test_rule_4_from_names_unknown_step(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["from"] = "ghost"
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_missing_document(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["document"] = "methods/nowhere.json"
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_save_value_must_be_an_output(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["save"] = [
        {"step": "locate", "value": "ghost.parquet", "file_path": "g.parquet"}
    ]
    expect_rule(4, raw, env, tmp_path)


def test_rule_5_cycles_refuse(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["after"] = ["best"]
    expect_rule(5, raw, env, tmp_path)


def test_rule_6_dead_step(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["spare"] = {
        "type": "select",
        "from": "locate",
        "table": "iia.parquet",
        "choose": "max",
        "emit": {"x": "sites.target.layer"},
    }
    expect_rule(6, raw, env, tmp_path)


def test_rule_6_after_is_not_a_sink(env, tmp_path):
    """`after` orders without consuming — a step only anyone `after`s is
    still dead."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["spare"] = {
        "type": "select",
        "from": "locate",
        "table": "iia.parquet",
        "choose": "max",
        "emit": {"x": "sites.target.layer"},
    }
    raw["steps"]["best"]["after"] = ["spare"]
    expect_rule(6, raw, env, tmp_path)


def test_rule_7_emit_column_must_be_an_axis(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["emit"] = {"best_layer": "sites.target.head"}
    expect_rule(7, raw, env, tmp_path)


def test_rule_8_table_extension(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["table"] = "iia.json"
    expect_rule(8, raw, env, tmp_path)


def test_rule_9_set_override_must_target_existing_path(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["set"] = {"sites.ghost.layer": 3}
    expect_rule(9, raw, env, tmp_path)


def test_inner_load_errors_surface_as_rule_4(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["set"] = {"reads.v_src.site": "nowhere"}
    err = expect_rule(4, raw, env, tmp_path)
    assert "does not load" in str(err)


# --------------------------------------------------------------------------- #
# schedule + digests on the shipped worked example
# --------------------------------------------------------------------------- #


class TestWeekdaysWorkflow:
    def test_loads_with_the_spec_schedule(self, env):
        loaded = load_workflow(WEEKDAYS_WF, env)
        assert loaded.order == (
            "locate",
            "best",
            "fit",
            "apply",
            "scan_heatmap",
            "iia_by_k",
        )
        assert loaded.levels == (
            ("locate",),
            ("best", "scan_heatmap"),
            ("fit",),
            ("apply", "iia_by_k"),
        )

    def test_digest_kinds_split_by_dependency(self, env):
        loaded = load_workflow(WEEKDAYS_WF, env)
        assert loaded.inner_digest_kind == {
            "locate": "campaign",
            "fit": "authored",
            "apply": "authored",
        }
        # the independent step's stamp IS the standalone campaign digest
        assert loaded.inner_digests["locate"] == loaded.inner["locate"].document_digest

    def test_deterministic(self, env):
        first = load_workflow(WEEKDAYS_WF, env)
        second = load_workflow(WEEKDAYS_WF, env)
        assert first.digest == second.digest
        assert first.canonical == second.canonical

    def test_digest_tracks_the_inner_document(self, env, tmp_path):
        """Editing a referenced method file changes the workflow digest —
        the §7 content-addressing claim."""
        workflow_dir = tmp_path / "workflows"
        shutil.copytree(WEEKDAYS_WF.parent.parent, tmp_path, dirs_exist_ok=True)
        target = tmp_path / "methods/weekdays_locate_scan.json"
        original = load_workflow(workflow_dir / "weekdays_8b.json", env)
        doc = json.loads(target.read_text())
        doc["description"] = "same campaign, different words"
        target.write_text(json.dumps(doc))
        edited = load_workflow(workflow_dir / "weekdays_8b.json", env)
        assert edited.digest != original.digest


class TestCanonicalForm:
    def test_defaults_materialized(self, env):
        loaded = load_workflow(WEEKDAYS_WF, env)
        best = loaded.canonical["steps"]["best"]
        assert best["aggregate"] == "mean"
        assert best["value"] == "value"

    def test_protocol_steps_stamp_document_digests(self, env):
        loaded = load_workflow(WEEKDAYS_WF, env)
        for name in ("locate", "fit", "apply"):
            entry = loaded.canonical["steps"][name]
            assert len(entry["document_digest"]) == 64
            assert entry["digest_kind"] in ("campaign", "authored")
