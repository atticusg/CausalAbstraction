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
                "table": "iia.json",
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
    raw["save"] = [{"step": "locate", "value": "ghost.json", "file_path": "g.json"}]
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
        "table": "iia.json",
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
        "table": "iia.json",
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
    """A ranked table is JSON. A third format is not merely unknown here — JSON
    and safetensors are the only two the stack has (IM spec §2.12)."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["table"] = "iia.csv"
    expect_rule(8, raw, env, tmp_path)


def test_rule_9_set_override_must_target_existing_path(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["set"] = {"sites.ghost.layer": 3}
    expect_rule(9, raw, env, tmp_path)


def test_inner_load_errors_surface_as_rule_4(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["set"] = {"reads.v_cf.site": "nowhere"}
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
            "best_fit",
            "apply",
            "scan_heatmap",
            "iia_by_k",
        )
        assert loaded.levels == (
            ("locate",),
            ("best", "scan_heatmap"),
            ("fit",),
            ("best_fit", "iia_by_k"),
            ("apply",),
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
        {"step": "locate", "value": "locate/iia.json", "file_path": "iia.json"}
    )
    raw["steps"]["best"]["table"] = "locate/iia.json"
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
        {"step": "fit", "value": "iia.json", "file_path": "fit_iia.json"}
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
        {"step": "apply", "value": "iia.json", "file_path": "apply_iia.json"}
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
        "table": "iia.json",
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
        "table": "t.json",
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
        "table": "iia.json",
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
        {"step": "apply", "value": "iia.json", "file_path": "apply_iia.json"}
    )
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.inner_digest_kind["apply"] == "authored"


# --------------------------------------------------------------------------- #
# transform steps (§2.4)
# --------------------------------------------------------------------------- #


def transform_workflow(tmp_path: Path) -> dict[str, Any]:
    """A minimal harvest → fit_pca → plot workflow."""
    methods = tmp_path / "methods"
    methods.mkdir(exist_ok=True)
    shutil.copyfile(
        REPO / "causalab/configs/methods/harvest.json", methods / "harvest.json"
    )
    return {
        "version": "1",
        "steps": {
            "harvest": {"type": "protocol", "document": "methods/harvest.json"},
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
            "scree": {
                "type": "plot",
                "plot": "lines",
                "from": "fit",
                "table": "spectrum.json",
                "x": "pc",
                "value": "explained_variance_ratio",
                "file_path": "scree.png",
            },
        },
        "save": [{"step": "scree", "value": "scree.png", "file_path": "scree.png"}],
    }


class TestTransformParse:
    def test_it_loads_and_schedules_after_its_input(self, env, tmp_path):
        """`inputs` ARE the dependency edges — nobody authored this order."""
        loaded = load_workflow(transform_workflow(tmp_path), env, workflow_dir=tmp_path)
        assert loaded.order == ("harvest", "fit", "scree")
        assert loaded.dependencies["fit"] == ("harvest",)

    def test_rule_1_unknown_op_suggests(self, env, tmp_path):
        raw = transform_workflow(tmp_path)
        raw["steps"]["fit"]["op"] = "fit_pcaa@1"
        err = expect_rule(1, raw, env, tmp_path)
        assert "fit_pca" in str(err)

    def test_rule_1_unknown_version_of_a_known_op(self, env, tmp_path):
        raw = transform_workflow(tmp_path)
        raw["steps"]["fit"]["op"] = "fit_pca@9"
        err = expect_rule(1, raw, env, tmp_path)
        assert "has no version 9" in str(err)

    def test_rule_1_a_missing_output_slot(self, env, tmp_path):
        raw = transform_workflow(tmp_path)
        del raw["steps"]["fit"]["outputs"]["spectrum"]
        err = expect_rule(1, raw, env, tmp_path)
        assert "spectrum" in str(err)

    def test_rule_1_an_undeclared_output_slot(self, env, tmp_path):
        raw = transform_workflow(tmp_path)
        raw["steps"]["fit"]["outputs"]["loadings"] = "loadings.json"
        err = expect_rule(1, raw, env, tmp_path)
        assert "no output slot 'loadings'" in str(err)

    def test_rule_1_an_undeclared_input_slot(self, env, tmp_path):
        raw = transform_workflow(tmp_path)
        raw["steps"]["fit"]["inputs"]["extra"] = {
            "step": "harvest",
            "value": "x.safetensors",
        }
        expect_rule(1, raw, env, tmp_path)

    def test_rule_1_a_bad_param_type(self, env, tmp_path):
        raw = transform_workflow(tmp_path)
        raw["steps"]["fit"]["params"]["k"] = "2"
        err = expect_rule(1, raw, env, tmp_path)
        assert "integer" in str(err)

    def test_rule_1_a_missing_required_param(self, env, tmp_path):
        raw = transform_workflow(tmp_path)
        raw["steps"]["fit"]["params"] = {}
        err = expect_rule(1, raw, env, tmp_path)
        assert "missing required parameter 'k'" in str(err)

    def test_rule_1_an_unknown_param_suggests(self, env, tmp_path):
        raw = transform_workflow(tmp_path)
        raw["steps"]["fit"]["params"]["kk"] = 2
        err = expect_rule(1, raw, env, tmp_path)
        assert "unknown parameter 'kk'" in str(err)

    def test_rule_8_an_output_slot_keeps_its_file_kind(self, env, tmp_path):
        """A tensor slot cannot be written as a .json — the record says
        which format the slot is, so a typo is a load error, not a surprise
        at read time."""
        raw = transform_workflow(tmp_path)
        raw["steps"]["fit"]["outputs"]["weight"] = "weight.json"
        expect_rule(8, raw, env, tmp_path)

    def test_rule_4_an_input_naming_an_unknown_step(self, env, tmp_path):
        raw = transform_workflow(tmp_path)
        raw["steps"]["fit"]["inputs"]["acts"]["step"] = "nowhere"
        expect_rule(4, raw, env, tmp_path)

    def test_rule_4_an_input_the_producer_does_not_save(self, env, tmp_path):
        raw = transform_workflow(tmp_path)
        raw["steps"]["fit"]["inputs"]["acts"]["value"] = "acts_L99.safetensors"
        err = expect_rule(4, raw, env, tmp_path)
        assert "produces no 'acts_L99.safetensors'" in str(err)


class TestTransformAsAProducer:
    def test_rule_7_columns_are_checked_against_the_declaration(self, env, tmp_path):
        """A transform producer has no sweep axes; its op's declared columns
        are what rule 7 checks instead."""
        raw = transform_workflow(tmp_path)
        raw["steps"]["scree"]["x"] = "component"
        err = expect_rule(7, raw, env, tmp_path)
        assert "explained_variance_ratio" in str(err)  # what it does have

    def test_rule_7_there_is_no_implicit_value_column(self, env, tmp_path):
        """`value` defaults to 'value' for a protocol producer's metric table;
        a transform table is only what its op declares, so the default must be
        refused rather than silently failing at run time."""
        raw = transform_workflow(tmp_path)
        del raw["steps"]["scree"]["value"]
        err = expect_rule(7, raw, env, tmp_path)
        assert "'value'" in str(err)

    def test_a_select_step_may_rank_a_transform_table(self, env, tmp_path):
        raw = transform_workflow(tmp_path)
        raw["steps"]["top"] = {
            "type": "select",
            "from": "fit",
            "table": "spectrum.json",
            "choose": "max",
            "value": "explained_variance_ratio",
            "emit": {"best_pc": "pc"},
        }
        raw["save"].append(
            {"step": "top", "value": "values.json", "file_path": "top.json"}
        )
        loaded = load_workflow(raw, env, workflow_dir=tmp_path)
        assert loaded.dependencies["top"] == ("fit",)

    def test_rule_4_a_tensor_slot_cannot_be_ranked(self, env, tmp_path):
        raw = transform_workflow(tmp_path)
        raw["steps"]["scree"]["table"] = "weight.safetensors"
        expect_rule(8, raw, env, tmp_path)  # .json is required first

    def test_a_protocol_step_may_load_a_transform_tensor(self, env, tmp_path):
        """Rule 10's run-tree half: the fitted-artifact direction #37 needs."""
        raw = transform_workflow(tmp_path)
        project = {
            "version": "1",
            "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main"},
            "data": {"base": {"dataset": "weekdays/train", "field": "input"}},
            "positions": {"tap": {"index": -1}},
            "sites": {"L0": {"component": "block_output", "layer": 0}},
            "featurizers": {
                "basis": {"kind": "pca", "k": 2, "file_path": "fit/weight.safetensors"}
            },
            "reads": {
                "coords": {
                    "site": "L0",
                    "pos": "tap",
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
        (tmp_path / "methods/project.json").write_text(json.dumps(project))
        raw["steps"]["project"] = {
            "type": "protocol",
            "document": "methods/project.json",
        }
        raw["save"].append(
            {
                "step": "project",
                "value": "coords.safetensors",
                "file_path": "coords.safetensors",
            }
        )
        loaded = load_workflow(raw, env, workflow_dir=tmp_path)
        assert "fit" in loaded.dependencies["project"]

    def test_rule_10_a_protocol_step_cannot_load_what_a_transform_never_writes(
        self, env, tmp_path
    ):
        raw = transform_workflow(tmp_path)
        project = {
            "version": "1",
            "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main"},
            "data": {"base": {"dataset": "weekdays/train", "field": "input"}},
            "positions": {"tap": {"index": -1}},
            "sites": {"L0": {"component": "block_output", "layer": 0}},
            "featurizers": {
                "basis": {"kind": "pca", "k": 2, "file_path": "fit/basis.safetensors"}
            },
            "reads": {
                "coords": {
                    "site": "L0",
                    "pos": "tap",
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
        (tmp_path / "methods/project.json").write_text(json.dumps(project))
        raw["steps"]["project"] = {
            "type": "protocol",
            "document": "methods/project.json",
        }
        raw["save"].append(
            {
                "step": "project",
                "value": "coords.safetensors",
                "file_path": "coords.safetensors",
            }
        )
        err = expect_rule(10, raw, env, tmp_path)
        assert "saves no 'basis.safetensors'" in str(err)


class TestTransformCanonicalForm:
    def test_the_op_version_and_params_enter_the_form(self, env, tmp_path):
        loaded = load_workflow(transform_workflow(tmp_path), env, workflow_dir=tmp_path)
        entry = loaded.canonical["steps"]["fit"]
        assert entry["op"] == "fit_pca@1"
        assert entry["params"] == {"k": 2}
        assert entry["outputs"] == {
            "weight": "weight.safetensors",
            "spectrum": "spectrum.json",
        }
        assert entry["inputs"] == {
            "acts": {"step": "harvest", "value": "acts_L8_ans.safetensors"}
        }

    def test_optional_selectors_add_no_key_when_unauthored(self, env, tmp_path):
        loaded = load_workflow(transform_workflow(tmp_path), env, workflow_dir=tmp_path)
        acts = loaded.canonical["steps"]["fit"]["inputs"]["acts"]
        assert "slot" not in acts and "entry" not in acts

    def test_a_param_change_moves_the_digest(self, env, tmp_path):
        original = load_workflow(
            transform_workflow(tmp_path), env, workflow_dir=tmp_path
        )
        raw = transform_workflow(tmp_path)
        raw["steps"]["fit"]["params"]["k"] = 3
        edited = load_workflow(raw, env, workflow_dir=tmp_path)
        assert edited.digest != original.digest

    def test_each_transform_step_gets_a_provenance_digest(self, env, tmp_path):
        loaded = load_workflow(transform_workflow(tmp_path), env, workflow_dir=tmp_path)
        assert len(loaded.transform_digests["fit"]) == 64
        assert set(loaded.transform_digests) == {"fit"}


# --------------------------------------------------------------------------- #
# shipped-workflow digest pins
# --------------------------------------------------------------------------- #

WORKFLOW_DIR = REPO / "causalab/configs/workflows"
WORKFLOW_PINS = Path(__file__).parent / "workflow_digests.json"


class TestShippedWorkflowDigests:
    """`workflow_digests.json`, regenerated by update_workflow_digests.py.

    A diff is a loader migration (§7), never a silent re-pin — and it is what
    makes "a new step type changed no existing document" a check rather than
    a claim."""

    def test_every_shipped_workflow_is_pinned(self):
        pins = json.loads(WORKFLOW_PINS.read_text())
        assert sorted(pins) == sorted(p.name for p in WORKFLOW_DIR.glob("*.json"))

    def test_digests_match_their_pins(self, env):
        for name, digest in json.loads(WORKFLOW_PINS.read_text()).items():
            assert load_workflow(WORKFLOW_DIR / name, env).digest == digest, name
