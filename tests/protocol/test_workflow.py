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


# --------------------------------------------------------------------------- #
# regressions from the adversarial review of the workflow layer
# --------------------------------------------------------------------------- #


def test_redundant_after_on_a_data_edge_still_counts_as_consumption(env, tmp_path):
    """`after` naming the step's own data producer is redundant but legal —
    it must not erase the consumption edge (the W6 wrong-reject)."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["after"] = ["locate"]
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.order == ("locate", "best")


def test_inner_save_paths_are_outputs_not_loads(env, tmp_path):
    """An inner document whose SAVE paths sit under a step name must not
    fabricate dependency edges — no self-cycle, no schedule inversion, and
    the digest kind stays campaign."""
    raw = tiny_workflow(tmp_path)
    doc = json.loads((tmp_path / "methods/locate.json").read_text())
    doc["save"] = [
        {**entry, "file_path": f"locate/{entry['file_path']}"} for entry in doc["save"]
    ]
    (tmp_path / "methods/locate.json").write_text(json.dumps(doc))
    raw["save"].append(
        {"step": "locate", "value": "locate/iia.parquet", "file_path": "iia.parquet"}
    )
    raw["steps"]["best"]["table"] = "locate/iia.parquet"
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.inner_digest_kind["locate"] == "campaign"
    assert loaded.dependencies["locate"] == ()


def test_rule_10_artifact_ref_must_name_a_select_step(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["fit"] = {
        "type": "protocol",
        "document": "methods/locate.json",
        "set": {"sites.target.layer": {"artifact": "locate", "key": "anything"}},
    }
    raw["save"].append(
        {"step": "fit", "value": "iia.parquet", "file_path": "fit_iia.parquet"}
    )
    expect_rule(10, raw, env, tmp_path)


def test_rule_10_run_tree_file_path_must_be_saved(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    apply_doc = json.loads(
        (REPO / "causalab/configs/methods/weekdays_das_apply.json").read_text()
    )
    (tmp_path / "methods/apply.json").write_text(json.dumps(apply_doc))
    raw["steps"]["apply"] = {
        "type": "protocol",
        "document": "methods/apply.json",
        "set": {"featurizers.rot.file_path": "locate/GHOST.safetensors"},
    }
    raw["save"].append(
        {"step": "apply", "value": "iia.parquet", "file_path": "apply_iia.parquet"}
    )
    expect_rule(10, raw, env, tmp_path)


@pytest.mark.parametrize(
    "file_path", ["../escape.json", "/tmp/abs.json", "locate", "workflow.json"]
)
def test_rule_3_save_paths_contained_and_unreserved(env, tmp_path, file_path):
    raw = tiny_workflow(tmp_path)
    raw["save"] = [{"step": "best", "value": "values.json", "file_path": file_path}]
    with pytest.raises(WorkflowError) as err:
        load_workflow(raw, env, workflow_dir=tmp_path)
    assert err.value.rule == 3


def test_rule_8_plot_path_contained(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["fig"] = {
        "type": "plot",
        "plot": "lines",
        "from": "locate",
        "table": "iia.parquet",
        "x": "sites.target.layer",
        "series": "positions.tap",
        "file_path": "../../escape.png",
    }
    raw["save"].append(
        {"step": "fig", "value": "../../escape.png", "file_path": "f.png"}
    )
    with pytest.raises(WorkflowError) as err:
        load_workflow(raw, env, workflow_dir=tmp_path)
    assert err.value.rule == 8


def test_rule_1_plot_kind_strictness():
    base = {
        "type": "plot",
        "from": "a",
        "table": "t.parquet",
        "x": "x",
        "file_path": "f.png",
    }
    lines_with_y = {
        "version": "1",
        "steps": {
            "a": {"type": "protocol", "document": "d.json"},
            "p": {**base, "plot": "lines", "y": "y"},
        },
        "save": [{"step": "p", "value": "f.png", "file_path": "f.png"}],
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(lines_with_y)
    assert err.value.rule == 1
    heatmap_with_series = {
        "version": "1",
        "steps": {
            "a": {"type": "protocol", "document": "d.json"},
            "p": {**base, "plot": "heatmap", "y": "y", "series": "s"},
        },
        "save": [{"step": "p", "value": "f.png", "file_path": "f.png"}],
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(heatmap_with_series)
    assert err.value.rule == 1


def test_rule_7_value_column_checked(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["value"] = "GHOST_COLUMN"
    expect_rule(7, raw, env, tmp_path)


def test_rule_7_heatmap_must_cover_every_axis(env, tmp_path):
    """The locate document has two axes; a heatmap on one of them would
    collapse the other into duplicate cells — refused at load, not at the
    end of an expensive producer run."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["fig"] = {
        "type": "plot",
        "plot": "lines",
        "from": "locate",
        "table": "iia.parquet",
        "x": "sites.target.layer",  # positions.tap left uncovered
        "file_path": "f.png",
    }
    raw["save"].append({"step": "fig", "value": "f.png", "file_path": "f.png"})
    expect_rule(7, raw, env, tmp_path)


def test_deferred_doc_with_external_featurizer_loads(env, tmp_path):
    """A step-dependent document that ALSO loads an external fitted bundle
    must not have that bundle's ArtifactIdentity checked against the
    representative-substituted document at load — the check runs with real
    values at run time (the fixture is stamped layer 18; the representative
    layer is 0)."""
    raw = tiny_workflow(tmp_path)
    apply_doc = json.loads(
        (REPO / "causalab/configs/methods/weekdays_das_apply.json").read_text()
    )
    (tmp_path / "methods/apply.json").write_text(json.dumps(apply_doc))
    raw["steps"]["apply"] = {
        "type": "protocol",
        "document": "methods/apply.json",
        "set": {"sites.target.layer": {"artifact": "best", "key": "best_layer"}},
    }
    raw["save"].append(
        {"step": "apply", "value": "iia.parquet", "file_path": "apply_iia.parquet"}
    )
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.inner_digest_kind["apply"] == "authored"
