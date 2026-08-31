"""The method / application split of a protocol document (§1.1, rule 18).

One file is one experiment run; the split is a shape *inside* it. The contract
under test is one sentence — *an application may complete a method, never
contradict it* — plus the property that makes the split safe to use: composing
is **transparent**, so the same experiment digests identically whether it was
written flat or in halves.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.errors import ParseError, ValidationError
from causalab.protocol.loader import load
from causalab.protocol.method import (
    document_type,
    is_split,
    method_digest,
    parse_method,
    signature_of,
    split_document,
)

from causalab.protocol.schema import SECTION_ORDER

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SHIPPED_METHOD = REPO / "causalab/configs/methods/interchange.json"
SHIPPED_RUN = REPO / "causalab/configs/runs/weekdays_8b_interchange.json"

#: What the application half owns: the run's inputs, plus the addresses.
APPLICATION_SECTIONS = ("model", "data", "sites")


def split_doc() -> dict[str, Any]:
    """``base_doc()``, written in halves: the method keeps everything but the
    inputs and the layer."""
    doc = in_order(base_doc())
    method = {
        key: value
        for key, value in doc.items()
        if key not in ("version", *APPLICATION_SECTIONS)
    }
    method["sites"] = {
        "tgt": {"component": "block_output"},
        "lm_head": {"component": "lm_head"},
    }
    method = {key: method[key] for key in doc if key in method}
    return {
        "version": "1",
        "application": {
            "model": doc["model"],
            "data": doc["data"],
            "sites": {"tgt": {"layer": 3}},
        },
        "method": method,
    }


def method_file(raw: dict[str, Any]) -> dict[str, Any]:
    """The same method as a standalone, reusable file."""
    return {"version": "1", "type": "method", **raw}


# --------------------------------------------------------------------------- #
# document types
# --------------------------------------------------------------------------- #


def test_a_split_document_is_a_protocol_document():
    """Flat or split, it is one experiment — the type does not fork."""
    assert document_type(base_doc()) == "protocol"
    assert document_type(split_doc()) == "protocol"
    assert not is_split(base_doc())
    assert is_split(split_doc())
    assert document_type({"version": "1", "steps": {}}) == "workflow"
    assert document_type(method_file(split_doc()["method"])) == "method"


def test_a_declared_type_that_contradicts_the_structure_is_refused():
    """`type` exists to catch a mistake, so a wrong one has to be an error."""
    with pytest.raises(ParseError) as err:
        document_type({"version": "1", "type": "workflow", **base_doc()})
    assert "reads as a protocol document" in str(err.value)


def test_an_unknown_type_is_refused_with_suggestions():
    with pytest.raises(ParseError) as err:
        document_type({"version": "1", "type": "methods"})
    assert err.value.code == "P4"
    assert "method" in str(err.value)


# --------------------------------------------------------------------------- #
# what each half is
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("section", ["model", "data"])
def test_a_method_may_not_name_its_inputs(section):
    """The network and the rows are what a method leaves open — including the
    data, which is why a method transfers to another task."""
    raw = split_doc()
    method = {section: raw["application"][section], **raw["method"]}
    raw["method"] = {key: method[key] for key in SECTION_ORDER if key in method}
    with pytest.raises(ValidationError) as err:
        split_document(raw)
    assert err.value.rule == 18
    assert section in str(err.value)


@pytest.mark.parametrize("section", ["reads", "save"])
def test_a_method_declares_what_it_measures(section):
    raw = split_doc()
    del raw["method"][section]
    with pytest.raises(ValidationError) as err:
        split_document(raw)
    assert err.value.rule == 18


@pytest.mark.parametrize("section", ["model", "data"])
def test_the_application_declares_the_inputs(section):
    raw = split_doc()
    del raw["application"][section]
    with pytest.raises(ValidationError) as err:
        split_document(raw)
    assert err.value.rule == 18
    assert section in str(err.value)


def test_both_halves_are_required():
    with pytest.raises(ValidationError) as err:
        split_document({"version": "1", "application": split_doc()["application"]})
    assert err.value.rule == 18
    assert "method" in str(err.value)


def test_a_protocol_section_at_the_top_level_of_a_split_document_is_refused():
    """Every protocol section lives inside one half — the error says so."""
    raw = split_doc()
    raw["sites"] = {"tgt": {"layer": 3}}
    with pytest.raises(ParseError) as err:
        split_document(raw)
    assert err.value.code == "P3"
    assert "application" in str(err.value)


def test_method_signature_names_exactly_what_is_open():
    method = parse_method(split_doc()["method"], standalone=False)
    assert method.signature.model is True
    assert method.signature.data == ("base", "counterfactual")
    assert method.signature.sites == {"tgt": ("layer",), "lm_head": ()}
    assert not method.signature.is_closed()
    assert method.signature.lines() == (
        "model: key, revision, dtype (+ optional quantization)",
        "data.base: dataset, field",
        "data.counterfactual: dataset, field",
        "sites.tgt: layer",
    )


def test_an_unbound_site_name_is_in_the_signature():
    """A site a read references but the method leaves undeclared still has to
    be supplied — the signature is about addresses, not about declarations."""
    raw = split_doc()["method"]
    del raw["sites"]["tgt"]
    assert signature_of(raw).sites["tgt"] == ("component",)


def test_a_method_digests_by_content_not_by_bookkeeping():
    """An inlined method and the file it came from are the same method."""
    method = split_doc()["method"]
    assert method_digest(method) == method_digest(method_file(method))
    edited = copy.deepcopy(method)
    edited["metrics"]["ld"]["a"] = "other_answer"
    assert method_digest(edited) != method_digest(method)


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #


def test_composition_is_transparent(env):
    """The property the whole split rests on: halves or flat, the canonical
    bytes — and so the digest, the provenance unit — are the same."""
    composed, _, _ = split_document(split_doc())
    assert composed == in_order(base_doc())
    assert (
        load(split_doc(), env).document_digest == load(base_doc(), env).document_digest
    )


def test_an_application_may_restate_what_the_method_already_fixed(env):
    raw = split_doc()
    raw["application"]["sites"]["lm_head"] = {"component": "lm_head"}  # same value
    assert load(raw, env).document_digest == load(base_doc(), env).document_digest


def test_an_application_may_not_overrule_its_method():
    raw = split_doc()
    raw["application"]["sites"]["tgt"]["component"] = "mlp_output"
    with pytest.raises(ValidationError) as err:
        split_document(raw)
    assert err.value.rule == 18
    assert "never overrules one" in str(err.value)


def test_an_unfilled_hole_is_refused_by_name():
    raw = split_doc()
    del raw["application"]["sites"]["tgt"]["layer"]
    with pytest.raises(ValidationError) as err:
        split_document(raw)
    assert err.value.rule == 18
    assert "sites.tgt: layer" in str(err.value)


def test_the_halves_appear_in_order():
    raw = split_doc()
    reordered = {
        "version": raw["version"],
        "method": raw["method"],
        "application": raw["application"],
    }
    with pytest.raises(ValidationError) as err:
        split_document(reordered)
    assert err.value.rule == 2


def test_descriptions_join_method_first():
    raw = split_doc()
    raw["description"] = "at L3 in gpt2"
    raw["method"] = {"description": "swap the residual", **raw["method"]}
    composed, _, _ = split_document(raw)
    assert composed["description"] == "swap the residual\n\nat L3 in gpt2"


def test_the_application_halfs_description_is_not_dropped():
    """The regression: `_check_application_shape` admits `description` (it is
    in SECTION_ORDER and is not `type`), but the merge loop skips
    ("version", "type", "description") — so the words were accepted and then
    silently vanished from the composed document, with no error to explain it.

    All three describe different things — the method describes itself, the
    application describes the binding, the document describes the run — so the
    composition keeps all three, in that order.
    """
    raw = split_doc()
    raw["description"] = "at L3 in gpt2"
    raw["method"] = {"description": "swap the residual", **raw["method"]}
    raw["application"] = {
        "description": "bound to gpt2 over the fixture rows",
        **raw["application"],
    }
    composed, _, _ = split_document(raw)
    assert composed["description"] == (
        "swap the residual\n\nbound to gpt2 over the fixture rows\n\nat L3 in gpt2"
    )


def test_an_application_description_alone_still_reaches_the_document():
    """The half that was dropped is the only one present — so a bug that
    merely reordered the join would still pass the test above, but not this."""
    raw = split_doc()
    raw.pop("description", None)
    raw["application"] = {"description": "just the binding", **raw["application"]}
    composed, _, _ = split_document(raw)
    assert composed["description"] == "just the binding"


def test_a_sweep_in_the_application_expands(env):
    """A layer scan is an edit of the application half; the method is
    untouched and keeps its digest."""
    raw = split_doc()
    before = method_digest(raw["method"])
    raw["application"]["sites"]["tgt"]["layer"] = {"sweep": [1, 2, 3]}
    loaded = load(raw, env)
    assert len(loaded.expansion.points) == 3
    assert method_digest(raw["method"]) == before


def test_a_swept_component_in_the_application_digests_as_the_flat_form(env):
    """§1.1's promise, at the one shape that used to break it.

    Sweeping ``sites.<name>.component`` in the application half raised an
    unhandled ``TypeError: unhashable type: 'dict'`` from ``signature_of`` —
    while *the same experiment written flat* loaded fine. So this is both the
    regression test for that traceback and the property test for "a split
    document digests exactly as the same experiment written flat": one
    assertion, because they are the same claim.
    """
    sweep = {"sweep": ["attention_output", "mlp_output"]}
    flat = in_order(base_doc())
    flat["sites"]["tgt"] = {"component": sweep, "layer": 3}

    split = split_doc()
    split["method"]["sites"]["tgt"] = {}
    split["application"]["sites"]["tgt"] = {"component": sweep, "layer": 3}

    loaded_split = load(split, env)
    assert loaded_split.document_digest == load(flat, env).document_digest
    assert len(loaded_split.expansion.points) == 2


def test_a_swept_component_still_obliges_a_layer_when_any_value_needs_one():
    """The sweep is not a hole in the layer obligation. Every value layerless
    means no layer is owed; one layered spelling in the sweep owes one, because
    otherwise the composition names a site that cannot resolve."""
    raw = split_doc()["method"]
    raw["sites"]["tgt"] = {"component": {"sweep": ["ln_final", "lm_head"]}}
    assert signature_of(raw).sites["tgt"] == ()

    raw["sites"]["tgt"] = {"component": {"sweep": ["lm_head", "mlp_output"]}}
    assert signature_of(raw).sites["tgt"] == ("layer",)

    raw["sites"]["tgt"] = {
        "component": {"sweep": ["lm_head", "mlp_output"]},
        "layer": 3,
    }
    assert signature_of(raw).sites["tgt"] == ()


def test_overrides_address_the_composition(env):
    """``--set`` paths are the flat document's, whichever form was authored —
    one override vocabulary for both."""
    loaded = load(
        split_doc(), env, overrides={"sites.tgt.layer": 5, "model.dtype": "bf16"}
    )
    assert loaded.canonical_document["sites"]["tgt"]["layer"] == 5
    assert loaded.canonical_document["model"]["dtype"] == "bf16"


# --------------------------------------------------------------------------- #
# method files, and the loaded record
# --------------------------------------------------------------------------- #


def test_a_method_file_cannot_be_run(env):
    with pytest.raises(ValidationError) as err:
        load(method_file(split_doc()["method"]), env)
    assert err.value.rule == 18
    assert "nothing to run" in str(err.value)


def test_the_load_reports_the_method_it_composed(env):
    raw = split_doc()
    loaded = load(raw, env)
    assert loaded.method_digest == method_digest(raw["method"])
    assert loaded.method_ref is None  # inline: one file is one run
    assert load(base_doc(), env).method_digest is None


def test_a_method_path_resolves_against_the_document(env, tmp_path):
    raw = split_doc()
    (tmp_path / "methods").mkdir()
    (tmp_path / "methods/m.json").write_text(json.dumps(method_file(raw["method"])))
    inline_digest = method_digest(raw["method"])
    raw["method"] = "methods/m.json"
    path = tmp_path / "run.json"
    path.write_text(json.dumps(raw))
    loaded = load(path, env)
    assert loaded.method_ref == "methods/m.json"
    # the same method, whether it was inlined or referenced
    assert loaded.method_digest == inline_digest
    assert loaded.document_digest == load(split_doc(), env).document_digest


def test_a_missing_method_file_is_a_load_error(env, tmp_path):
    raw = split_doc()
    raw["method"] = "methods/nowhere.json"
    path = tmp_path / "run.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValidationError) as err:
        load(path, env)
    assert err.value.rule == 18


# --------------------------------------------------------------------------- #
# the shipped pair
# --------------------------------------------------------------------------- #


def test_the_shipped_run_document_composes_and_binds_everything(env):
    loaded = load(SHIPPED_RUN, env)
    assert loaded.method_ref is None
    assert loaded.canonical_document["model"]["dtype"] == "bf16"
    assert loaded.canonical_document["sites"]["target"] == {
        "component": "block_output",
        "layer": 18,
    }
    # the inlined method is the shipped method file, byte for byte
    assert loaded.method_digest == method_digest(json.loads(SHIPPED_METHOD.read_text()))


def test_the_shipped_method_is_open_exactly_where_it_should_be():
    method = parse_method(json.loads(SHIPPED_METHOD.read_text()))
    assert method.signature.model is True
    assert method.signature.data == ("base", "counterfactual")
    assert method.signature.sites == {"target": ("layer",), "lm_head": ()}


def test_a_method_file_may_be_pasted_in_verbatim(env):
    """The obvious thing to do with a method file is paste it into the half —
    its `type` and `version` come along, and both are simply checked."""
    raw = split_doc()
    inline = method_file(raw["method"])
    raw["method"] = inline
    assert load(raw, env).document_digest == load(base_doc(), env).document_digest
    assert load(raw, env).method_digest == method_digest(inline)

    wrong = dict(inline, version="2")
    with pytest.raises(ValidationError) as err:
        split_document({**raw, "method": wrong})
    assert err.value.rule == 18
