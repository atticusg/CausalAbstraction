"""Regressions from the adversarial spec-conformance review of the
protocol layer — each test pins a hole the review demonstrated (wrong
accepts, wrong rule attributions, crash-not-refuse paths, false interning,
digest instability) so deleting the fix turns the suite red."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.protocol.engine import requires_campaign
from causalab.protocol.canonical import canonicalize, digest
from causalab.protocol.errors import ParseError, ProtocolError, ValidationError
from causalab.protocol.loader import load, load_text
from causalab.protocol.plan import plan_point
from causalab.protocol.schema import parse_document
from causalab.protocol.sweep import expand, find_axes

from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import ROT_FIXTURE_RELPATH
from tests.protocol.test_validation_rules import expect_rule, parse_and_validate

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# rule 8 — sign regimes in the overlap logic
# --------------------------------------------------------------------------- #


def _scoped(doc):
    doc["data"]["base"] = {"dataset": "weekdays/train", "field": "input"}
    return doc


def test_rule_8_scoped_span_vs_scoped_index_refuses():
    doc = base_doc()
    doc["writes"]["patch"]["pos"] = {"span": [-2, -1], "scope": {"variable": "subject"}}
    doc["writes"]["patch2"] = {
        "site": "tgt",
        "pos": {"index": 0, "scope": {"variable": "subject"}},
        "do": {"swap": "v_cf"},
    }
    doc["intervened_models"]["patched"]["writes"].append("patch2")
    expect_rule(8, doc)


def test_rule_8_disjoint_end_relative_spans_load():
    doc = base_doc()
    doc["writes"]["patch"]["pos"] = {"span": [-3, -2], "scope": {"variable": "subject"}}
    doc["writes"]["patch2"] = {
        "site": "tgt",
        "pos": {"span": [-2, -1], "scope": {"variable": "subject"}},
        "do": {"swap": "v_cf"},
    }
    doc["intervened_models"]["patched"]["writes"].append("patch2")
    parse_and_validate(doc)


def test_rule_8_provably_disjoint_indices_load():
    doc = base_doc()
    doc["writes"]["patch2"] = {"site": "tgt", "pos": 1, "do": {"swap": "v_cf"}}
    doc["writes"]["patch"]["pos"] = 0
    doc["intervened_models"]["patched"]["writes"].append("patch2")
    parse_and_validate(doc)


def test_unscoped_degenerate_span_refused_at_parse():
    doc = base_doc()
    doc["writes"]["patch"]["pos"] = {"span": [3, 3]}
    with pytest.raises(ParseError):
        parse_document(in_order(doc))
    doc["writes"]["patch"]["pos"] = {"span": [-2, -1]}  # unscoped end-relative
    with pytest.raises(ParseError):
        parse_document(in_order(doc))


# --------------------------------------------------------------------------- #
# rule 9 — literal: explicit dims selections pairwise disjoint, any class
# --------------------------------------------------------------------------- #


def test_rule_9_additive_writes_with_intersecting_dims_refuse():
    doc = base_doc()
    doc["writes"]["patch"]["dims"] = [0, 1]
    doc["writes"]["nudge"] = {
        "site": "tgt",
        "pos": -1,
        "dims": [1, 2],
        "do": {"add_scaled": {"op": "v_cf", "alpha": 0.5}},
    }
    doc["intervened_models"]["patched"]["writes"].append("nudge")
    expect_rule(9, doc)


def test_full_width_absolute_plus_dims_additive_loads():
    doc = base_doc()
    doc["writes"]["nudge"] = {
        "site": "tgt",
        "pos": -1,
        "dims": [0, 1],
        "do": {"add_scaled": {"op": "v_cf", "alpha": 0.5}},
    }
    doc["intervened_models"]["patched"]["writes"].append("nudge")
    parse_and_validate(doc)


# --------------------------------------------------------------------------- #
# rule 6 — affine A/b are params, never reads
# --------------------------------------------------------------------------- #


def test_rule_6_affine_matrix_may_not_be_a_read():
    doc = base_doc()
    doc["writes"]["patch"]["do"] = {"affine": {"A": "v_cf", "b": "v_cf"}}
    expect_rule(6, doc)


# --------------------------------------------------------------------------- #
# rule 4 — train-section references (deletability guard)
# --------------------------------------------------------------------------- #


def _train_doc():
    doc = base_doc()
    doc["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
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
    return doc


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["train"].__setitem__("objective", [[1.0, "ghost"]]),
        lambda d: d["train"].__setitem__("params", ["ghost"]),
        lambda d: d["train"].__setitem__(
            "eval",
            {"every": {"epochs": 1}, "split": "weekdays/test", "metrics": ["ghost"]},
        ),
        lambda d: d["train"].__setitem__(
            "early_stop", {"metric": "ghost", "patience": 1, "mode": "max"}
        ),
        lambda d: d["train"].__setitem__(
            "anneal", {"ghost.theta.temperature": [1, 0, 0.5]}
        ),
        lambda d: d["train"].__setitem__(
            "anneal", {"rot.theta.temperature": [1, 0, 0.5]}
        ),
    ],
    ids=[
        "objective",
        "params",
        "eval-metric",
        "early-stop",
        "anneal-featurizer",
        "anneal-slot",
    ],
)
def test_rule_4_train_reference_checks(mutate):
    doc = _train_doc()
    mutate(doc)
    expect_rule(4, doc)


def test_rule_7_im_input_must_be_a_role():
    doc = base_doc()
    doc["intervened_models"]["patched"]["input"] = "counterfactuals"
    doc["reads"]["logits"]["input"] = "counterfactuals"
    with pytest.raises(ValidationError) as err:
        parse_and_validate(doc)
    assert err.value.rule in (5, 7)


def test_rule_4_im_unknown_write():
    doc = base_doc()
    doc["intervened_models"]["patched"]["writes"] = ["patch", "ghost"]
    expect_rule(4, doc)


# --------------------------------------------------------------------------- #
# rule 10 — attribution and sub-rules
# --------------------------------------------------------------------------- #


def test_rule_10_saving_a_write_is_not_saveable_not_undeclared():
    doc = base_doc()
    doc["save"].append(
        {
            "value": "patch",
            "model": "patched",
            "input": "base",
            "file_path": "p.json",
        }
    )
    err = expect_rule(10, doc)
    assert "not saveable" in str(err)


def test_rule_10_duplicate_file_path():
    # the colliding entry saves v_cf (a block_output read) so nothing else
    # fires first — a dims slice on the lm_head read feeding `ld` would now
    # be a rule-4 error of its own, and the duplicate-path check is the first
    # thing _check_save runs, which is what this pins
    doc = base_doc()
    doc["save"].append(
        {
            "value": "v_cf",
            "model": "original",
            "input": "counterfactual",
            "file_path": "ld.json",
        }
    )
    expect_rule(10, doc)


def test_rule_10_wrong_extension():
    doc = base_doc()
    doc["save"][0]["file_path"] = "ld.safetensors"  # a metric must be .json
    expect_rule(10, doc)


def test_rule_10_trained_featurizer_must_be_saved():
    doc = _train_doc()
    doc["save"] = [e for e in doc["save"] if e["value"] != "rot"]
    expect_rule(10, doc)


def test_rule_10_featurizer_site_cross_checked():
    doc = _train_doc()
    for entry in doc["save"]:
        if entry["value"] == "rot":
            entry["site"] = "lm_head"
    expect_rule(10, doc)


# --------------------------------------------------------------------------- #
# rule 11 — dead declarations of every kind
# --------------------------------------------------------------------------- #


def test_rule_11_dead_position():
    doc = base_doc()
    doc["positions"] = {"spare": {"index": 0}}
    expect_rule(11, doc)


def test_rule_11_dead_featurizer():
    doc = base_doc()
    doc["featurizers"] = {"spare": {"kind": "gate"}}
    expect_rule(11, doc)


def test_rule_11_dead_param():
    doc = base_doc()
    doc["params"] = {"spare": {"file_path": "spare.safetensors"}}
    expect_rule(11, doc)


def test_rule_11_unread_intervened_model():
    doc = base_doc()
    doc["reads"]["v2"] = {"site": "tgt", "pos": 0, "model": "original", "input": "base"}
    doc["writes"]["patch2"] = {"site": "tgt", "pos": 0, "do": {"swap": "v2"}}
    doc["intervened_models"]["ghosted"] = {"input": "base", "writes": ["patch2"]}
    expect_rule(11, doc)


# --------------------------------------------------------------------------- #
# rule 12 — loaded params are not trainable
# --------------------------------------------------------------------------- #


def test_rule_12_loaded_param_in_train_params():
    doc = _train_doc()
    doc["params"] = {"vec": {"file_path": "vec.safetensors"}}
    doc["writes"]["steer"] = {
        "site": "tgt",
        "pos": -1,
        "do": {"add_scaled": {"op": "vec", "alpha": 1.0}},
    }
    doc["intervened_models"]["patched"]["writes"].append("steer")
    doc["train"]["params"] = ["rot", "vec"]
    expect_rule(12, doc)


# --------------------------------------------------------------------------- #
# artifact resolution — recursion, cycles, malformed shapes, placement
# --------------------------------------------------------------------------- #


def test_artifact_ref_inside_sweep_values_resolves(env):
    doc = base_doc()
    doc["sites"]["tgt"]["layer"] = {
        "sweep": [{"artifact": "weekdays/llama31_8b/locate", "key": "best_layer"}, 3]
    }
    doc["model"] = {"key": "meta-llama/Llama-3.1-8B", "revision": "main"}
    loaded = load(in_order(doc), env)
    assert [p.raw["sites"]["tgt"]["layer"] for p in loaded.expansion.points] == [18, 3]


def test_malformed_artifact_ref_refuses(env):
    doc = base_doc()
    doc["sites"]["tgt"]["layer"] = {"artifact": "weekdays/llama31_8b/locate"}
    with pytest.raises(ValidationError) as err:
        load(in_order(doc), env)
    assert err.value.rule == 15 and "malformed" in str(err.value)


def test_nested_artifact_ref_resolves(env, artifacts_root: Path):
    (artifacts_root / "indirect.json").write_text(
        json.dumps(
            {"hop": {"artifact": "weekdays/llama31_8b/locate", "key": "best_layer"}}
        )
    )
    doc = base_doc()
    doc["model"] = {"key": "meta-llama/Llama-3.1-8B", "revision": "main"}
    doc["sites"]["tgt"]["layer"] = {"artifact": "indirect", "key": "hop"}
    loaded = load(in_order(doc), env)
    assert loaded.expansion.points[0].raw["sites"]["tgt"]["layer"] == 18


def test_artifact_ref_cycle_refuses(env, artifacts_root: Path):
    (artifacts_root / "loop.json").write_text(
        json.dumps({"self": {"artifact": "loop", "key": "self"}})
    )
    doc = base_doc()
    doc["sites"]["tgt"]["layer"] = {"artifact": "loop", "key": "self"}
    with pytest.raises(ValidationError) as err:
        load(in_order(doc), env)
    assert err.value.rule == 15 and "cycle" in str(err.value)


def test_artifact_injected_nonfinite_refuses(env, artifacts_root: Path):
    (artifacts_root / "bad.json").write_text('{"alpha": Infinity}')
    doc = base_doc()
    doc["writes"]["patch"]["do"] = {
        "add_scaled": {"op": "v_cf", "alpha": {"artifact": "bad", "key": "alpha"}}
    }
    with pytest.raises(ProtocolError):
        load(in_order(doc), env)


# --------------------------------------------------------------------------- #
# digest stability and identity
# --------------------------------------------------------------------------- #


def test_integral_float_and_int_digest_identically(env):
    a = base_doc()
    b = base_doc()
    a["writes"]["patch"]["do"] = {"add_scaled": {"op": "v_cf", "alpha": 1}}
    b["writes"]["patch"]["do"] = {"add_scaled": {"op": "v_cf", "alpha": 1.0}}
    assert digest(canonicalize(in_order(a), env)) == digest(
        canonicalize(in_order(b), env)
    )


def test_pos_sugar_inside_entry_sweep_digests_identically(env):
    a = base_doc()
    b = base_doc()
    a["positions"] = {"tap": {"sweep": [-1, {"variable": "subject"}]}}
    b["positions"] = {"tap": {"sweep": [{"index": -1}, {"variable": "subject"}]}}
    for doc in (a, b):
        doc["reads"]["v_cf"]["pos"] = "tap"
        doc["writes"]["patch"]["pos"] = "tap"
    assert digest(canonicalize(in_order(a), env)) == digest(
        canonicalize(in_order(b), env)
    )


def test_im_write_order_inside_sweep_digests_identically(env):
    a = base_doc()
    b = base_doc()
    for doc in (a, b):
        doc["writes"]["nudge"] = {
            "site": "tgt",
            "pos": -1,
            "do": {"add_scaled": {"op": "v_cf", "alpha": 0.5}},
        }
    a["intervened_models"]["patched"]["writes"] = {
        "sweep": [["patch", "nudge"], ["patch"]]
    }
    b["intervened_models"]["patched"]["writes"] = {
        "sweep": [["nudge", "patch"], ["patch"]]
    }
    assert digest(canonicalize(in_order(a), env)) == digest(
        canonicalize(in_order(b), env)
    )


def test_params_content_digest_stamped(env):
    doc = base_doc()
    doc["model"] = {"key": "meta-llama/Llama-3.1-8B", "revision": "main"}
    doc["params"] = {"vec": {"file_path": ROT_FIXTURE_RELPATH}}
    doc["writes"]["patch"]["do"] = {"add_scaled": {"op": "vec", "alpha": 1.0}}
    del doc["reads"]["v_cf"]
    del doc["data"]["counterfactual"]
    doc["reads"]["logits"]["dims"] = None
    doc["reads"]["logits"].pop("dims")
    loaded = load(in_order(doc), env)
    stamped = loaded.canonical_document["params"]["vec"]
    assert len(stamped["content_digest"]) == 64


def test_params_missing_file_refuses(env):
    doc = base_doc()
    doc["params"] = {"vec": {"file_path": "nowhere.safetensors"}}
    doc["writes"]["patch"]["do"] = {"add_scaled": {"op": "vec", "alpha": 1.0}}
    del doc["reads"]["v_cf"]
    del doc["data"]["counterfactual"]
    with pytest.raises(ValidationError) as err:
        load(in_order(doc), env)
    assert err.value.rule == 15


# --------------------------------------------------------------------------- #
# plan interning identity
# --------------------------------------------------------------------------- #


def test_counterfactual_data_identity_reaches_the_patched_group():
    doc = parse_document(in_order(base_doc()))
    one = plan_point(doc, data_identity={"base": "d", "counterfactual": "s1"})
    two = plan_point(doc, data_identity={"base": "d", "counterfactual": "s2"})
    patched_one = next(g for g in one.groups if g.model == "patched")
    patched_two = next(g for g in two.groups if g.model == "patched")
    assert patched_one.digest != patched_two.digest


def test_model_identity_reaches_every_group():
    a = base_doc()
    b = base_doc()
    b["model"]["key"] = "meta-llama/Llama-3.1-8B"
    plans = [
        plan_point(
            parse_document(in_order(d)),
            data_identity={"base": "d", "counterfactual": "d"},
        )
        for d in (a, b)
    ]
    assert {g.digest for g in plans[0].groups}.isdisjoint(
        g.digest for g in plans[1].groups
    )


def test_param_operand_spec_reaches_the_group_digest():
    def with_param(path: str):
        doc = base_doc()
        doc["params"] = {"vec": {"file_path": path}}
        doc["writes"]["patch"]["do"] = {"add_scaled": {"op": "vec", "alpha": 1.0}}
        del doc["reads"]["v_cf"]
        del doc["data"]["counterfactual"]
        return parse_document(in_order(doc))

    one = plan_point(with_param("a.safetensors"), data_identity={"base": "d"})
    two = plan_point(with_param("b.safetensors"), data_identity={"base": "d"})
    patched = lambda plan: next(g for g in plan.groups if g.model == "patched")  # noqa: E731
    assert patched(one).digest != patched(two).digest


# --------------------------------------------------------------------------- #
# sweeps, YAML surface, campaign routing
# --------------------------------------------------------------------------- #


def test_enormous_range_refused_before_materializing():
    doc = base_doc()
    doc["sites"]["tgt"]["layer"] = {"sweep": {"range": [0, 10_000_000_000]}}
    with pytest.raises(ValidationError) as err:
        find_axes(doc)
    assert err.value.rule == 14


def test_yaml_duplicate_keys_refused(tmp_path: Path):
    target = tmp_path / "doc.yaml"
    target.write_text("version: '1'\nversion: '1'\n")
    with pytest.raises(ParseError):
        load_text(target)


def test_yaml_non_string_keys_refused(tmp_path: Path):
    target = tmp_path / "doc.yaml"
    target.write_text(
        "version: '1'\nsites:\n  1: {component: block_output, layer: 0}\n"
    )
    with pytest.raises(ParseError) as err:
        load_text(target)
    assert "not a string" in str(err.value)


def test_optimizer_lr_must_be_numeric():
    doc = _train_doc()
    doc["train"]["optimizer"]["lr"] = "fast"
    with pytest.raises(ParseError):
        parse_document(in_order(doc))


def test_requires_campaign_unions_over_points(env):
    doc = base_doc()
    doc["writes"]["pfn"] = {
        "site": "tgt",
        "pos": 0,
        "do": {"pytorch_fn": {"qualname": "torch.relu"}},
    }
    doc["intervened_models"]["patched"]["writes"] = {
        "sweep": [["patch"], ["patch", "pfn"]]
    }
    expansion = expand(in_order(doc))
    docs = [parse_document(p.raw) for p in expansion.points]
    needed = requires_campaign(docs)
    assert "pytorch_fn_local" in needed  # point 0 alone would miss it
