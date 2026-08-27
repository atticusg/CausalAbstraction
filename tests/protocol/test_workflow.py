"""The workflow-protocol document model (docs/workflow_protocol.md v2 §5):
parse rules, the reference grammar, the derived schedule, digest semantics, and
the shipped weekdays-8b worked example.

One test per checklist rule, asserted **by rule number** — so a renumbering of
the spec has to be a deliberate edit here rather than a silent drift.
"""

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


def _copy_locate(tmp_path: Path) -> None:
    methods = tmp_path / "methods"
    methods.mkdir(exist_ok=True)
    shutil.copyfile(
        REPO / "causalab/configs/methods/weekdays_locate_scan.json",
        methods / "locate.json",
    )


def tiny_workflow(tmp_path: Path) -> dict[str, Any]:
    """A minimal protocol → script workflow over a copied method preset."""
    _copy_locate(tmp_path)
    return {
        "version": "1",
        "output_dir": "run",
        "steps": {
            "locate": {"type": "protocol", "document": "methods/locate.json"},
            "best": {
                "type": "script",
                "script": "causalab:select",
                "inputs": {
                    "table": {"step": "locate", "file": "iia.json"},
                    "choose": "max",
                    "emit": {"best_layer": "sites.target.layer"},
                },
                "outputs": {
                    "values": {"file": "values.json", "keys": {"best_layer": 18}}
                },
            },
        },
    }


def script_workflow(
    tmp_path: Path, body: str = "def main(inputs, outputs):\n    pass\n"
) -> dict[str, Any]:
    """A workflow over a user script in the workflow directory."""
    _copy_locate(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "reduce.py").write_text(body)
    return {
        "version": "1",
        "output_dir": "run",
        "steps": {
            "locate": {"type": "protocol", "document": "methods/locate.json"},
            "reduce": {
                "type": "script",
                "script": "scripts/reduce.py",
                "inputs": {"table": {"step": "locate", "file": "iia.json"}},
                "outputs": {
                    "out": {
                        "file": "out.json",
                        "columns": {"layer": "int64", "value": "float64"},
                    }
                },
            },
        },
    }


def expect_rule(rule: int, raw: dict[str, Any], env, tmp_path: Path) -> WorkflowError:
    with pytest.raises(WorkflowError) as err:
        load_workflow(raw, env, workflow_dir=tmp_path)
    assert err.value.rule == rule, f"expected W{rule}, got {err.value}"
    return err.value


# --------------------------------------------------------------------------- #
# rule 1 — strict keys, closed enums
# --------------------------------------------------------------------------- #


def test_is_workflow_dispatches_on_steps():
    assert is_workflow({"version": "1", "output_dir": "r", "steps": {}})
    assert not is_workflow({"version": "1", "model": {}, "save": []})


def test_rule_1_unknown_step_type_suggests():
    raw = {
        "version": "1",
        "output_dir": "run",
        "steps": {"a": {"type": "protocols", "document": "x.json"}},
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 1 and "protocol" in str(err.value)


def test_rule_1_transform_select_plot_are_gone():
    """v1's three Python-flavoured step types collapsed into `script`."""
    for retired in ("transform", "select", "plot"):
        raw = {
            "version": "1",
            "output_dir": "run",
            "steps": {"a": {"type": retired}},
        }
        with pytest.raises(WorkflowError) as err:
            parse_workflow(raw)
        assert err.value.rule == 1


def test_rule_1_unknown_top_level_key():
    raw = {"version": "1", "output_dir": "run", "steps": {}, "save": []}
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 1 and "save" in str(err.value)


def test_rule_1_script_step_needs_script_inputs_outputs():
    for missing in ("script", "inputs", "outputs"):
        step = {
            "type": "script",
            "script": "causalab:select",
            "inputs": {},
            "outputs": {"v": "v.json"},
        }
        del step[missing]
        with pytest.raises(WorkflowError) as err:
            parse_workflow({"version": "1", "output_dir": "run", "steps": {"a": step}})
        assert err.value.rule in (1, 7)


# --------------------------------------------------------------------------- #
# rule 2 — section order and output_dir
# --------------------------------------------------------------------------- #


def test_rule_2_section_order_enforced():
    raw = {
        "version": "1",
        "steps": {"a": {"type": "protocol", "document": "x.json"}},
        "output_dir": "run",
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 2


@pytest.mark.parametrize("bad", ["a/b", "/abs", "..", ".", "nested/dir"])
def test_rule_2_output_dir_is_one_segment(bad):
    raw = {
        "version": "1",
        "output_dir": bad,
        "steps": {"a": {"type": "protocol", "document": "x.json"}},
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 2


def test_rule_2_output_dir_required():
    with pytest.raises(WorkflowError) as err:
        parse_workflow(
            {"version": "1", "steps": {"a": {"type": "protocol", "document": "x"}}}
        )
    assert err.value.rule == 1


# --------------------------------------------------------------------------- #
# rule 3 — step names
# --------------------------------------------------------------------------- #


def test_rule_3_step_names_filesystem_safe():
    raw = {
        "version": "1",
        "output_dir": "run",
        "steps": {"a/b": {"type": "protocol", "document": "x.json"}},
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 3


def test_rule_3_reserved_step_names():
    """A step directory sits beside the run manifest and the sidecars."""
    for reserved in ("workflow.json", "_step"):
        raw = {
            "version": "1",
            "output_dir": "run",
            "steps": {reserved: {"type": "protocol", "document": "x.json"}},
        }
        with pytest.raises(WorkflowError) as err:
            parse_workflow(raw)
        assert err.value.rule == 3


# --------------------------------------------------------------------------- #
# rule 4 — the reference grammar
# --------------------------------------------------------------------------- #


def test_rule_4_unknown_step_in_input(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["table"]["step"] = "ghost"
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_input_names_a_file_the_producer_does_not_write(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["table"]["file"] = "ghost.json"
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_after_names_unknown_step(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["after"] = ["ghost"]
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_two_locators_refused(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["table"] = {
        "step": "locate",
        "file": "iia.json",
        "path": "x.json",
    }
    expect_rule(1, raw, env, tmp_path)


def test_rule_4_two_selectors_refused(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["table"] = {
        "step": "locate",
        "file": "acts.safetensors",
        "key": "x",
        "entry": {"k": 1},
    }
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_key_selector_needs_a_json_locator(env, tmp_path):
    """A selector must match its locator's format — decidable from the
    filename alone, which is what having only two formats buys."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["table"] = {
        "step": "locate",
        "file": "rot.safetensors",
        "key": "best_layer",
    }
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_entry_selector_needs_a_safetensors_locator(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["table"] = {
        "step": "locate",
        "file": "iia.json",
        "entry": {"k": 1},
    }
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_repo_relative_path_must_exist(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["pins"] = {"path": "configs/definitely_absent.json"}
    expect_rule(4, raw, env, tmp_path)


def test_rule_4_repo_relative_path_that_exists_loads(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["pins"] = {
        "path": "causalab/configs/methods/interchange.json"
    }
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert "best" in loaded.order


def test_rule_4_absolute_path_is_not_existence_checked(env, tmp_path):
    """Validation and execution routinely run on different hosts, so an
    absolute path naming another machine's data must not fail a load."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"]["pins"] = {"path": "/mnt/nowhere/fit.json"}
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.unchecked_paths == ("best.pins: /mnt/nowhere/fit.json",)


def test_rule_4_key_must_be_declared_by_the_producer(env, tmp_path):
    """The strengthened half of v1's rule 10: outputs are declared, so this is
    checkable against *any* step rather than only a `select` step."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["consume"] = {
        "type": "script",
        "script": "causalab:select",
        "inputs": {"layer": {"step": "best", "file": "values.json", "key": "ghost"}},
        "outputs": {"values": "out.json"},
    }
    err = expect_rule(4, raw, env, tmp_path)
    assert "best_layer" in str(err)


def test_rule_4_key_of_a_protocol_step_is_refused(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["consume"] = {
        "type": "script",
        "script": "causalab:select",
        "inputs": {"layer": {"step": "locate", "file": "iia.json", "key": "x"}},
        "outputs": {"values": "out.json"},
    }
    expect_rule(4, raw, env, tmp_path)


# --------------------------------------------------------------------------- #
# rule 5 — acyclicity and the schedule
# --------------------------------------------------------------------------- #


def test_rule_5_cycle_via_after(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["after"] = ["best"]
    err = expect_rule(5, raw, env, tmp_path)
    assert "cycle" in str(err)


def test_schedule_levels_are_derived(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.levels == (("locate",), ("best",))
    assert loaded.dependencies["best"] == ("locate",)


def test_independent_steps_share_a_level(env, tmp_path):
    """Parallelism nobody authored: two consumers of one producer."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["other"] = {
        "type": "script",
        "script": "causalab:select",
        "inputs": {
            "table": {"step": "locate", "file": "iia.json"},
            "emit": {"worst_layer": "sites.target.layer"},
            "choose": "min",
        },
        "outputs": {"values": {"file": "values.json", "keys": {"worst_layer": 0}}},
    }
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.levels[0] == ("locate",)
    assert set(loaded.levels[1]) == {"best", "other"}


# --------------------------------------------------------------------------- #
# rule 6 — script resolution, hashed and never imported
# --------------------------------------------------------------------------- #


def test_rule_6_unknown_builtin_suggests(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["script"] = "causalab:selct"
    err = expect_rule(6, raw, env, tmp_path)
    assert "select" in str(err)


def test_rule_6_missing_user_script(env, tmp_path):
    raw = script_workflow(tmp_path)
    raw["steps"]["reduce"]["script"] = "scripts/absent.py"
    expect_rule(6, raw, env, tmp_path)


def test_rule_6_script_must_not_escape_the_workflow_dir(env, tmp_path):
    raw = script_workflow(tmp_path)
    raw["steps"]["reduce"]["script"] = "../outside.py"
    expect_rule(6, raw, env, tmp_path)


def test_rule_6_script_must_parse(env, tmp_path):
    raw = script_workflow(tmp_path, body="def main(:\n")
    err = expect_rule(6, raw, env, tmp_path)
    assert "does not parse" in str(err)


def test_rule_6_script_must_declare_main(env, tmp_path):
    raw = script_workflow(tmp_path, body="def other(inputs, outputs):\n    pass\n")
    err = expect_rule(6, raw, env, tmp_path)
    assert "main" in str(err)


def test_script_hash_is_in_the_digest(env, tmp_path):
    """Why the hash is in the digest at all: `--resume` is otherwise wrong."""
    raw = script_workflow(tmp_path)
    first = load_workflow(raw, env, workflow_dir=tmp_path)
    (tmp_path / "scripts" / "reduce.py").write_text(
        "def main(inputs, outputs):\n    return 1\n"
    )
    second = load_workflow(raw, env, workflow_dir=tmp_path)
    assert first.digest != second.digest
    assert first.step_digests["reduce"] != second.step_digests["reduce"]


def test_a_nested_function_named_main_is_not_enough(env, tmp_path):
    raw = script_workflow(
        tmp_path,
        body="def wrapper():\n    def main(inputs, outputs):\n        pass\n",
    )
    expect_rule(6, raw, env, tmp_path)


# --------------------------------------------------------------------------- #
# rule 7 — outputs
# --------------------------------------------------------------------------- #


def test_rule_7_outputs_non_empty(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {}
    expect_rule(7, raw, env, tmp_path)


@pytest.mark.parametrize("bad", ["out.csv", "out.parquet", "out.png", "out"])
def test_rule_7_only_two_formats(env, tmp_path, bad):
    """JSON and safetensors, and nothing else — a third format is refused, not
    merely unknown."""
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {"values": bad}
    expect_rule(7, raw, env, tmp_path)


def test_rule_7_output_must_stay_in_the_step_dir(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {"values": "../escape.json"}
    expect_rule(7, raw, env, tmp_path)


def test_rule_7_two_slots_one_file(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {"a": "same.json", "b": "same.json"}
    expect_rule(7, raw, env, tmp_path)


def test_rule_7_columns_and_keys_are_exclusive(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {
        "values": {
            "file": "values.json",
            "columns": {"a": "int64"},
            "keys": {"best_layer": 1},
        }
    }
    expect_rule(7, raw, env, tmp_path)


def test_rule_7_unknown_column_dtype(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {
        "values": {"file": "values.json", "columns": {"a": "float32"}}
    }
    err = expect_rule(7, raw, env, tmp_path)
    assert "float64" in str(err)


def test_rule_7_safetensors_declares_no_columns(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["outputs"] = {
        "values": {"file": "w.safetensors", "columns": {"a": "int64"}}
    }
    expect_rule(7, raw, env, tmp_path)


# --------------------------------------------------------------------------- #
# rule 8 — protocol steps and their `set`
# --------------------------------------------------------------------------- #


def test_rule_8_set_override_must_target_existing_path(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["set"] = {"sites.ghost.layer": 3}
    expect_rule(8, raw, env, tmp_path)


def test_rule_8_missing_document(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["document"] = "methods/absent.json"
    expect_rule(4, raw, env, tmp_path)


def test_rule_8_inner_load_errors_surface(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["locate"]["set"] = {"model.key": "not-a-registered-model"}
    err = expect_rule(8, raw, env, tmp_path)
    assert "does not load" in str(err)


# --------------------------------------------------------------------------- #
# rule 10 / 11 — runtime and is_deterministic
# --------------------------------------------------------------------------- #


def test_rule_10_isolated_step_declares_deps(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["runtime"] = {"isolate": True}
    expect_rule(10, raw, env, tmp_path)


def test_rule_10_runtime_env_is_a_list_of_names(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["runtime"] = {"isolate": True, "deps": ["x"], "env": "TOKEN"}
    expect_rule(10, raw, env, tmp_path)


def test_runtime_is_in_the_digest(env, tmp_path):
    """A different dependency set is a different computation, so `--resume`
    must not skip across a change to it."""
    raw = tiny_workflow(tmp_path)
    plain = load_workflow(raw, env, workflow_dir=tmp_path).digest
    raw["steps"]["best"]["runtime"] = {"isolate": True, "deps": ["umap-learn"]}
    isolated = load_workflow(raw, env, workflow_dir=tmp_path).digest
    assert plain != isolated


def test_rule_11_is_deterministic_is_a_boolean(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["is_deterministic"] = "no"
    expect_rule(11, raw, env, tmp_path)


def test_nondeterministic_steps_are_reported(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["is_deterministic"] = False
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert loaded.nondeterministic == ("best",)


def test_is_deterministic_is_in_the_digest(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    a = load_workflow(raw, env, workflow_dir=tmp_path).digest
    raw["steps"]["best"]["is_deterministic"] = False
    b = load_workflow(raw, env, workflow_dir=tmp_path).digest
    assert a != b


# --------------------------------------------------------------------------- #
# gone in v2: the sink rule and the save section
# --------------------------------------------------------------------------- #


def test_a_terminal_step_no_one_consumes_is_legal(env, tmp_path):
    """v1's sink rule refused this; everything declared is now published, so a
    terminal plot or report step needs no blessing."""
    raw = tiny_workflow(tmp_path)
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    assert "best" in loaded.order  # nothing consumes `best`, and that is fine


def test_save_section_is_rejected(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["save"] = [{"step": "best", "value": "values.json", "file_path": "b.json"}]
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 1


# --------------------------------------------------------------------------- #
# canonical form and digests (§7)
# --------------------------------------------------------------------------- #


def test_output_dir_is_excluded_from_the_digest(env, tmp_path):
    """It names where the run lands, not what the run is."""
    raw = tiny_workflow(tmp_path)
    first = load_workflow(raw, env, workflow_dir=tmp_path)
    raw["output_dir"] = "somewhere_else"
    second = load_workflow(raw, env, workflow_dir=tmp_path)
    assert first.digest == second.digest


def test_inner_document_edits_move_the_workflow_digest(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    before = load_workflow(raw, env, workflow_dir=tmp_path).digest
    doc = tmp_path / "methods/locate.json"
    inner = json.loads(doc.read_text())
    inner["description"] = "edited"
    doc.write_text(json.dumps(inner))
    after = load_workflow(raw, env, workflow_dir=tmp_path).digest
    assert before != after


def test_canonical_form_carries_no_output_dir(env, tmp_path):
    loaded = load_workflow(tiny_workflow(tmp_path), env, workflow_dir=tmp_path)
    assert "output_dir" not in loaded.canonical
    assert loaded.canonical["steps"]["best"]["script_sha256"]


def test_inputs_are_sorted_in_the_canonical_form(env, tmp_path):
    raw = tiny_workflow(tmp_path)
    raw["steps"]["best"]["inputs"] = {
        "zzz": 1,
        "table": {"step": "locate", "file": "iia.json"},
        "emit": {"best_layer": "sites.target.layer"},
        "aaa": 2,
    }
    loaded = load_workflow(raw, env, workflow_dir=tmp_path)
    keys = list(loaded.canonical["steps"]["best"]["inputs"])
    assert keys == sorted(keys)


# --------------------------------------------------------------------------- #
# the shipped worked example (§10)
# --------------------------------------------------------------------------- #


def test_weekdays_example_loads_with_the_spec_schedule(env):
    loaded = load_workflow(WEEKDAYS_WF, env)
    assert [sorted(level) for level in loaded.levels] == [
        ["locate"],
        ["best", "scan_heatmap"],
        ["fit"],
        ["best_fit", "iia_by_k"],
        ["apply"],
    ]
    assert loaded.inner_digest_kind == {
        "locate": "campaign",
        "fit": "authored",
        "apply": "authored",
    }
    assert len(loaded.inner["locate"].expansion.points) == 64
    assert len(loaded.inner["fit"].expansion.points) == 9


def test_weekdays_example_digest_is_pinned(env):
    pins = json.loads((Path(__file__).parent / "workflow_digests.json").read_text())
    loaded = load_workflow(WEEKDAYS_WF, env)
    assert loaded.digest == pins["weekdays_8b.json"]
