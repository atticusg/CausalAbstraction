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
        {"value": "ce", "model": "patched", "input": "base", "file_path": "ce.parquet"}
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
