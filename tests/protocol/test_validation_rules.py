"""One failing document per load-error checklist rule (spec §5).

Each test mutates a minimal valid document into exactly one violation and
asserts the loader refuses with that rule's code — the checklist is the
contract, the rule number is the assertion.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from causalab.protocol.errors import ParseError, ValidationError
from causalab.protocol.loader import load
from causalab.protocol.schema import parse_document
from causalab.protocol.sweep import expand
from causalab.protocol.validate import validate_document

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit


def parse_and_validate(raw: dict[str, Any], **kwargs: Any) -> None:
    validate_document(parse_document(in_order(raw)), **kwargs)


def expect_rule(rule: int, raw: dict[str, Any], **kwargs: Any) -> ValidationError:
    with pytest.raises(ValidationError) as err:
        parse_and_validate(raw, **kwargs)
    assert err.value.rule == rule, f"expected V{rule}, got {err.value}"
    return err.value


def test_base_document_is_valid():
    parse_and_validate(base_doc())


# rule 1 — strict keys, closed enums, no authored derived fields ------------- #


def test_rule_1_unknown_key_with_suggestion():
    doc = base_doc()
    doc["sites"]["tgt"]["layers"] = 3
    with pytest.raises(ParseError) as err:
        parse_document(doc)
    assert err.value.code == "P3"
    assert "layer" in str(err.value)  # the suggestion


def test_rule_1_closed_enum_with_suggestion():
    doc = base_doc()
    doc["sites"]["tgt"]["component"] = "block_out"
    with pytest.raises(ParseError) as err:
        parse_document(doc)
    assert err.value.code == "P4"
    assert "block_output" in str(err.value)


def test_rule_1_derived_field_not_authorable():
    doc = base_doc()
    doc["data"]["base"]["digest"] = "abc"  # stamped at load, never authored
    with pytest.raises(ParseError):
        parse_document(doc)


# rule 2 — section order, save last ------------------------------------------ #


def test_rule_2_section_order():
    doc = base_doc()
    reordered = {"version": doc["version"], "data": doc["data"], "model": doc["model"]}
    reordered.update({k: v for k, v in doc.items() if k not in reordered})
    with pytest.raises(ValidationError) as err:
        parse_document(reordered)
    assert err.value.rule == 2


def test_rule_2_save_not_last():
    doc = base_doc()
    save = doc.pop("save")
    metrics = doc.pop("metrics")
    doc["save"] = save
    doc["metrics"] = metrics
    with pytest.raises(ValidationError) as err:
        parse_document(doc)
    assert err.value.rule == 2


# rule 3 — one namespace, reserved names -------------------------------------- #


def test_rule_3_duplicate_name_across_sections():
    doc = base_doc()
    doc["reads"]["tgt"] = {
        "site": "tgt",
        "pos": -1,
        "model": "original",
        "input": "base",
    }
    doc["save"].append(
        {
            "value": "tgt",
            "model": "original",
            "input": "base",
            "file_path": "t.safetensors",
        }
    )
    expect_rule(3, doc)


def test_rule_3_reserved_name():
    doc = base_doc()
    doc["positions"] = {"original": {"index": -1}}
    doc["reads"]["v_cf"]["pos"] = "original"
    expect_rule(3, doc)


def test_rule_3_all_is_reserved():
    """A positions entry named ``all`` would shadow the bare-string sugar,
    so the name is reserved outright (§5.3)."""
    doc = base_doc()
    doc["positions"] = {"all": {"index": -1}}
    expect_rule(3, doc)


# rule 4 — every reference resolves ------------------------------------------- #


def test_rule_4_unknown_site():
    doc = base_doc()
    doc["writes"]["patch"]["site"] = "nowhere"
    expect_rule(4, doc)


def test_rule_4_metric_on_non_lm_head_read():
    doc = base_doc()
    doc["metrics"]["ld"]["of"] = "v_cf"
    expect_rule(4, doc)


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "class_probs", "groups": {"days": ["Monday"]}},
        {"kind": "token_logit", "token": "cf_answer"},
        {"kind": "cross_entropy", "target": "cf_answer"},
        {"kind": "match", "expected": "cf_answer"},
    ],
)
def test_rule_4_still_binds_the_token_space_kinds_to_lm_head(spec):
    """``top_k`` was loosened to any read; its siblings were not. Each of
    these resolves an authored string to a token id, which only an
    ``lm_head`` read can be indexed by."""
    doc = base_doc()
    doc["metrics"]["m"] = {"of": "v_cf", **spec}
    doc["save"].append(
        {
            "value": "m",
            "model": "original",
            "input": "counterfactual",
            "file_path": "m.json",
        }
    )
    expect_rule(4, doc)


def test_rule_4_top_k_binds_to_a_read_at_any_component():
    """The point of the change: a top-k over a wide read is the reduction that
    keeps the wide tensor off disk, so it must be expressible."""
    doc = base_doc()
    doc["metrics"]["tk"] = {"kind": "top_k", "of": "v_cf", "k": 4, "by": "abs_value"}
    doc["save"].append(
        {
            "value": "tk",
            "model": "original",
            "input": "counterfactual",
            "file_path": "tk.json",
        }
    )
    parse_and_validate(doc)


def test_rule_4_top_k_by_prob_off_lm_head_is_refused():
    """A softmax across a residual stream normalizes over an axis that is not
    an event space — the resulting numbers are probabilities of nothing."""
    doc = base_doc()
    doc["metrics"]["tk"] = {"kind": "top_k", "of": "v_cf", "k": 4, "by": "prob"}
    doc["save"].append(
        {
            "value": "tk",
            "model": "original",
            "input": "counterfactual",
            "file_path": "tk.json",
        }
    )
    err = expect_rule(4, doc)
    assert "prob" in str(err)


def test_rule_4_top_k_by_prob_on_lm_head_is_legal():
    doc = base_doc()
    doc["metrics"]["tk"] = {"kind": "top_k", "of": "logits", "k": 4, "by": "prob"}
    doc["save"].append(
        {"value": "tk", "model": "patched", "input": "base", "file_path": "tk.json"}
    )
    parse_and_validate(doc)


# rule 5 — read bindings ------------------------------------------------------ #


def test_rule_5_read_input_contradicts_im():
    doc = base_doc()
    doc["reads"]["logits"]["input"] = "counterfactual"
    expect_rule(5, doc)


def test_rule_5_read_model_undeclared():
    doc = base_doc()
    doc["reads"]["logits"]["model"] = "ghost"
    expect_rule(5, doc)


# rule 6 — operands are reads, params, or literal scalars --------------------- #


def test_rule_6_operand_names_a_metric():
    doc = base_doc()
    doc["writes"]["patch"]["do"] = {"swap": "ld"}
    expect_rule(6, doc)


# rule 7 — membership + acyclicity -------------------------------------------- #


def test_rule_7_write_in_no_im():
    doc = base_doc()
    doc["writes"]["orphan"] = {"site": "tgt", "pos": -1, "do": {"swap": "v_cf"}}
    expect_rule(7, doc)


def test_rule_7_model_graph_cycle():
    doc = base_doc()
    doc["reads"]["r_a"] = {"site": "tgt", "pos": -1, "model": "im_a", "input": "base"}
    doc["reads"]["r_b"] = {"site": "tgt", "pos": -1, "model": "im_b", "input": "base"}
    doc["writes"] = {
        "e_a": {"site": "tgt", "pos": -1, "do": {"swap": "r_b"}},
        "e_b": {"site": "tgt", "pos": -1, "do": {"swap": "r_a"}},
    }
    doc["intervened_models"] = {
        "im_a": {"input": "base", "writes": ["e_a"]},
        "im_b": {"input": "base", "writes": ["e_b"]},
    }
    doc["reads"]["logits"]["model"] = "im_a"
    expect_rule(7, doc)


# rule 8 — one absolute write per address --------------------------------------- #


def test_rule_8_two_absolute_writes_same_address():
    doc = base_doc()
    doc["writes"]["patch2"] = {"site": "tgt", "pos": -1, "do": {"swap": "v_cf"}}
    doc["intervened_models"]["patched"]["writes"].append("patch2")
    expect_rule(8, doc)


def test_rule_8_all_positions_overlaps_everything():
    """An all-positions write covers every token, so it is never provably
    disjoint from another write at the same site — including one pinned to a
    single index."""
    doc = base_doc()
    doc["writes"]["patch_all"] = {
        "site": "tgt",
        "pos": {"all": True},
        "do": {"swap": "v_cf"},
    }
    doc["intervened_models"]["patched"]["writes"].append("patch_all")
    expect_rule(8, doc)


def test_rule_8_all_positions_overlaps_itself():
    doc = base_doc()
    doc["writes"]["patch"]["pos"] = "all"
    doc["writes"]["patch_all"] = {
        "site": "tgt",
        "pos": "all",
        "do": {"swap": "v_cf"},
    }
    doc["intervened_models"]["patched"]["writes"].append("patch_all")
    expect_rule(8, doc)


def test_rule_8_all_positions_absolute_plus_additive_composes():
    """The mechanism-class order (§2.8) still applies at an all address —
    overlap only forbids a *second absolute* write."""
    doc = base_doc()
    doc["writes"]["patch"]["pos"] = "all"
    doc["writes"]["nudge"] = {
        "site": "tgt",
        "pos": "all",
        "do": {"add_scaled": {"op": "v_cf", "alpha": 0.5}},
    }
    doc["intervened_models"]["patched"]["writes"].append("nudge")
    parse_and_validate(doc)


def test_rule_9_all_positions_dims_must_be_disjoint():
    """Rule 9 composes with the all spelling exactly as it does elsewhere."""
    doc = base_doc()
    doc["writes"]["patch"]["dims"] = [0, 1]
    doc["writes"]["patch_all"] = {
        "site": "tgt",
        "pos": "all",
        "dims": [1, 2],
        "do": {"swap": "v_cf"},
    }
    doc["intervened_models"]["patched"]["writes"].append("patch_all")
    expect_rule(9, doc)


def test_rule_8_additive_write_composes():
    doc = base_doc()
    doc["writes"]["nudge"] = {
        "site": "tgt",
        "pos": -1,
        "do": {"add_scaled": {"op": "v_cf", "alpha": 0.5}},
    }
    doc["intervened_models"]["patched"]["writes"].append("nudge")
    parse_and_validate(doc)  # absolute + additive at one address is legal (§2.8)


# rule 9 — dims disjointness ---------------------------------------------------- #


def test_rule_9_intersecting_dims_absolutes():
    doc = base_doc()
    doc["writes"]["patch"]["dims"] = [0, 1]
    doc["writes"]["patch2"] = {
        "site": "tgt",
        "pos": -1,
        "dims": [1, 2],
        "do": {"swap": "v_cf"},
    }
    doc["intervened_models"]["patched"]["writes"].append("patch2")
    expect_rule(9, doc)


def test_rule_9_disjoint_dims_absolutes_are_legal():
    doc = base_doc()
    doc["writes"]["patch"]["dims"] = [0, 1]
    doc["writes"]["patch2"] = {
        "site": "tgt",
        "pos": -1,
        "dims": [2, 3],
        "do": {"swap": "v_cf"},
    }
    doc["intervened_models"]["patched"]["writes"].append("patch2")
    parse_and_validate(doc)


# rule 10 — the save manifest --------------------------------------------------- #


def test_rule_10_metric_not_saved():
    doc = base_doc()
    doc["reads"]["logits"]["dims"] = [0, 1]
    doc["save"] = [
        {
            "value": "logits",
            "model": "patched",
            "input": "base",
            "file_path": "l.safetensors",
        }
    ]
    expect_rule(10, doc)


def test_rule_10_binding_mismatch():
    doc = base_doc()
    doc["save"][0]["model"] = "original"
    expect_rule(10, doc)


def test_rule_10_untrained_featurizer_not_saveable():
    doc = base_doc()
    doc["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["reads"]["v_cf"]["featurizer"] = "rot"
    doc["writes"]["patch"]["featurizer"] = "rot"
    doc["save"].append({"value": "rot", "site": "tgt", "file_path": "rot.safetensors"})
    expect_rule(10, doc)


def test_rule_10_reduce_on_a_metric_refused():
    """§2.12: a metric is already a reduction over its read."""
    doc = base_doc()
    doc["save"][0]["reduce"] = "mean"
    expect_rule(10, doc)


def test_rule_10_reduce_on_a_featurizer_bundle_refused():
    doc = base_doc()
    doc["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["reads"]["v_cf"]["featurizer"] = "rot"
    doc["writes"]["patch"]["featurizer"] = "rot"
    doc["train"] = {
        "objective": [[1.0, "ld"]],
        "params": ["rot"],
        "optimizer": {"name": "adam", "lr": 0.001},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    doc["save"].append(
        {
            "value": "rot",
            "site": "tgt",
            "file_path": "rot.safetensors",
            "reduce": "mean",
        }
    )
    expect_rule(10, doc)


def test_a_reduced_read_is_a_valid_save():
    doc = base_doc()
    doc["save"].append(
        {
            "value": "v_cf",
            "model": "original",
            "input": "counterfactual",
            "reduce": "mean",
            "file_path": "mean.safetensors",
        }
    )
    parse_and_validate(doc)


def test_an_entry_selector_needs_a_file_path():
    """'entry' selects *inside* a loaded bundle — there is nothing to
    select in a featurizer that is fitted rather than loaded."""
    doc = base_doc()
    doc["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 4,
            "parametrization": "cayley",
            "entry": {"k": 4},
        }
    }
    doc["reads"]["v_cf"]["featurizer"] = "rot"
    with pytest.raises(ParseError):
        parse_and_validate(doc)


def test_a_featurizer_may_not_rename_its_slots():
    """A featurizer bundle's slots come from its kind; only a params
    constant may name the tensor it wants."""
    doc = base_doc()
    doc["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 4,
            "parametrization": "cayley",
            "file_path": "rot.safetensors",
            "entry": {"slot": "acts"},
        }
    }
    doc["reads"]["v_cf"]["featurizer"] = "rot"
    with pytest.raises(ParseError):
        parse_and_validate(doc)


# rule 11 — sinks ---------------------------------------------------------------- #


def test_rule_11_dead_read():
    doc = base_doc()
    doc["reads"]["extra"] = {
        "site": "tgt",
        "pos": -1,
        "model": "original",
        "input": "base",
    }
    expect_rule(11, doc)


def test_rule_11_dead_site():
    doc = base_doc()
    doc["sites"]["spare"] = {"component": "block_output", "layer": 1}
    expect_rule(11, doc)


# rule 12 — trainability ---------------------------------------------------------- #


def test_rule_12_loaded_featurizer_trained():
    doc = base_doc()
    doc["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 4,
            "parametrization": "cayley",
            "file_path": "rot.safetensors",
        }
    }
    doc["reads"]["v_cf"]["featurizer"] = "rot"
    doc["writes"]["patch"]["featurizer"] = "rot"
    doc["metrics"]["ce"] = {"kind": "cross_entropy", "of": "logits", "target": "label"}
    doc["train"] = {
        "objective": [[1.0, "ce"]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    doc["save"].append(
        {"value": "ce", "model": "patched", "input": "base", "file_path": "ce.json"}
    )
    doc["save"].append({"value": "rot", "site": "tgt", "file_path": "rot.safetensors"})
    expect_rule(12, doc)


# rule 13 — pytorch_fn is local-only ---------------------------------------------- #


def test_rule_13_pytorch_fn_on_non_local_backend():
    doc = base_doc()
    doc["writes"]["patch"]["do"] = {"pytorch_fn": {"qualname": "torch.relu"}}
    del doc["reads"]["v_cf"]  # no longer an operand; would trip the sink rule
    expect_rule(13, doc, backend_is_local=False)
    parse_and_validate(doc, backend_is_local=True)  # a local backend may run it


# rule 14 — sweep wrappers + point cap --------------------------------------------- #


def test_rule_14_malformed_sweep():
    doc = base_doc()
    doc["sites"]["tgt"]["layer"] = {"sweep": {"start": 0}}
    with pytest.raises(ValidationError) as err:
        expand(doc)
    assert err.value.rule == 14


def test_rule_14_point_cap():
    doc = base_doc()
    doc["sites"]["tgt"]["layer"] = {"sweep": {"range": [0, 100]}}
    doc["reads"]["v_cf"]["dims"] = {"sweep": {"range": [0, 100]}}
    with pytest.raises(ValidationError) as err:
        expand(doc, point_cap=4096)
    assert err.value.rule == 14


# rule 15 — artifact-valued fields resolve ------------------------------------------ #


def test_rule_15_missing_artifact(env):
    doc = base_doc()
    doc["sites"]["tgt"]["layer"] = {"artifact": "nowhere/locate", "key": "best_layer"}
    with pytest.raises(ValidationError) as err:
        load(in_order(doc), env)
    assert err.value.rule == 15


def test_rule_15_artifact_identity_mismatch(env):
    doc = base_doc()
    doc["model"] = {"key": "meta-llama/Llama-3.1-8B", "revision": "main"}
    doc["sites"]["tgt"] = {"component": "block_output", "layer": 18}
    doc["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 16,  # the fixture bundle was fitted with k=8
            "parametrization": "cayley",
            "file_path": "artifacts/weekdays/llama31_8b/subspace/rot_k8.safetensors",
        }
    }
    doc["reads"]["v_cf"]["featurizer"] = "rot"
    doc["writes"]["patch"]["featurizer"] = "rot"
    with pytest.raises(ValidationError) as err:
        load(in_order(doc), env)
    assert err.value.rule == 15
    assert "ArtifactIdentity" in str(err.value)


def test_rule_15_artifact_identity_match_passes(env):
    doc = copy.deepcopy(base_doc())
    doc["model"] = {"key": "meta-llama/Llama-3.1-8B", "revision": "main"}
    doc["sites"]["tgt"] = {"component": "block_output", "layer": 18}
    doc["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 8,
            "parametrization": "cayley",
            "file_path": "artifacts/weekdays/llama31_8b/subspace/rot_k8.safetensors",
        }
    }
    doc["reads"]["v_cf"]["featurizer"] = "rot"
    doc["writes"]["patch"]["featurizer"] = "rot"
    load(in_order(doc), env)


# §2.3 column positions / §2.10 match modes ---------------------------------- #


def test_position_needs_exactly_one_anchor_form():
    doc = base_doc()
    doc["positions"] = {"p": {"variable": "entity", "column": "entity"}}
    with pytest.raises(ParseError):
        parse_and_validate(doc)


def test_position_column_and_scope_are_exclusive():
    doc = base_doc()
    doc["positions"] = {"p": {"column": "entity", "scope": {"variable": "x"}}}
    with pytest.raises(ParseError):
        parse_and_validate(doc)


def test_anchor_ref_takes_one_of_variable_or_column():
    doc = base_doc()
    doc["positions"] = {"p": {"index": 1, "relative_to": {"nope": "x"}}}
    with pytest.raises(ParseError):
        parse_and_validate(doc)


def test_unknown_match_mode_is_a_closed_enum_error():
    doc = base_doc()
    doc["metrics"]["m"] = {
        "kind": "match",
        "of": "logits",
        "expected": "label",
        "mode": "prefix",  # the task-side spelling; the metric's is first_token
    }
    doc["save"].append(
        {
            "value": "m",
            "model": "patched",
            "input": "base",
            "file_path": "m.json",
        }
    )
    with pytest.raises(ParseError) as err:
        parse_and_validate(doc)
    assert "first_token" in str(err.value)  # the suggestion names the real mode


def test_rule_8_column_positions_are_conservatively_overlapping():
    """Two writes at one address whose positions come from *different* columns
    could hit the same token — a column holds data, not a template slot — so
    the absolute-write rule refuses rather than assuming disjointness."""
    doc = base_doc()
    doc["positions"] = {"a": {"column": "entity"}, "b": {"column": "number"}}
    doc["reads"]["v2"] = {
        "site": "tgt",
        "pos": "a",
        "model": "original",
        "input": "counterfactual",
    }
    doc["writes"] = {
        "patch": {"site": "tgt", "pos": "a", "do": {"swap": "v_cf"}},
        "patch2": {"site": "tgt", "pos": "b", "do": {"swap": "v2"}},
    }
    doc["intervened_models"]["patched"]["writes"] = ["patch", "patch2"]
    expect_rule(8, doc)


# rule 16 — generation is read-only and prefill-only ------------------------- #


def _reads_the_continuation(doc: dict[str, Any]) -> dict[str, Any]:
    """Point the saved read at the continuation, which is legal."""
    doc["positions"] = {"tail": {"generated": {"max_new_tokens": 8}, "index": -1}}
    doc["reads"]["logits"]["pos"] = "tail"
    return doc


def test_a_read_may_address_the_continuation():
    """The positive control for rule 16: reads are exactly what the frame is
    for, so nothing about a generate read is a load error."""
    parse_and_validate(_reads_the_continuation(base_doc()))


def test_rule_16_write_at_a_generated_position():
    doc = _reads_the_continuation(base_doc())
    doc["writes"]["patch"]["pos"] = "tail"
    err = expect_rule(16, doc)
    assert "prefill-only" in str(err)


def test_rule_16_write_at_an_inline_generated_position():
    """Inline specs are refused on the same footing as named ones — the rule
    reads the resolved spec, not the spelling."""
    doc = _reads_the_continuation(base_doc())
    doc["writes"]["patch"]["pos"] = {"generated": {"max_new_tokens": 4}, "index": -1}
    expect_rule(16, doc)


def test_rule_16_train_with_a_generated_position():
    """A greedy decode is an argmax chain: there is no gradient path from a
    continuation read back to a featurizer's parameters."""
    doc = _reads_the_continuation(base_doc())
    doc["featurizers"] = {"rot": {"kind": "subspace", "k": 2}}
    doc["reads"]["v_cf"]["featurizer"] = "rot"
    doc["train"] = {
        "objective": [[1.0, "ld"]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 0.001},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    doc["save"].append({"value": "rot", "site": "tgt", "file_path": "rot.safetensors"})
    err = expect_rule(16, doc)
    assert "gradient path" in str(err)


def test_the_checklist_and_the_spec_agree_on_how_many_rules_there_are():
    """`CHECKLIST_RULES` guards every ValidationError's number, so it has to
    match §5 or the guard silently drifts from the document it enforces."""
    import re
    from pathlib import Path

    from causalab.protocol.errors import CHECKLIST_RULES

    spec = Path(__file__).resolve().parents[2] / "docs" / "intervention_protocol.md"
    section = spec.read_text().split("## 5. Validation")[1].split("\n## ")[0]
    numbered = {int(m) for m in re.findall(r"^(\d+)\. ", section, flags=re.M)}
    assert numbered == set(range(1, CHECKLIST_RULES + 1))


def test_rule_4_decode_over_a_prompt_frame_read():
    """``decode`` reduces tokens a decode produced; in the prompt frame there
    are none — only tokens that were given."""
    doc = base_doc()
    doc["metrics"] = {"said": {"kind": "decode", "of": "logits"}}
    doc["save"] = [
        {
            "value": "said",
            "model": "patched",
            "input": "base",
            "file_path": "said.json",
        }
    ]
    err = expect_rule(4, doc)
    assert "generated" in str(err)


def test_decode_over_a_continuation_read_is_legal():
    doc = _reads_the_continuation(base_doc())
    doc["metrics"] = {"said": {"kind": "decode", "of": "logits"}}
    doc["save"] = [
        {
            "value": "said",
            "model": "patched",
            "input": "base",
            "file_path": "said.json",
        }
    ]
    parse_and_validate(doc)
