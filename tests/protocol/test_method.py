"""Methods, applications and their composition (spec §1.1, checklist rule 18).

The contract under test is one sentence — *an application may complete a
method, never contradict it* — plus the property that makes the split safe to
use: composing is **transparent**, so the same experiment digests identically
whether it was authored as one file or as two.
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
    compose,
    document_type,
    method_digest,
    parse_method,
    signature_of,
)

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SHIPPED_METHOD = REPO / "causalab/configs/methods/interchange.json"
SHIPPED_APPLICATION = REPO / "causalab/configs/applications/weekdays_8b_interchange.json"


def as_method(raw: dict[str, Any]) -> dict[str, Any]:
    """A method file: ``type`` second, sections in the §1 order."""
    ordered = in_order(raw)
    return {"version": ordered.pop("version"), "type": "method", **ordered}


def split_doc() -> tuple[dict[str, Any], dict[str, Any]]:
    """The minimal document of ``base_doc()``, authored as a method plus an
    application: the method keeps everything but the network and the layer."""
    doc = base_doc()
    method = as_method({k: v for k, v in doc.items() if k != "model"})
    method["sites"] = {
        "tgt": {"component": "block_output"},
        "lm_head": {"component": "lm_head"},
    }
    application = {
        "version": "1",
        "type": "application",
        "method": method,
        "model": doc["model"],
        "sites": {"tgt": {"layer": 3}},
    }
    return method, application


# --------------------------------------------------------------------------- #
# document types
# --------------------------------------------------------------------------- #


def test_document_type_dispatches_on_structure_and_declaration():
    method, application = split_doc()
    assert document_type(base_doc()) == "protocol"
    assert document_type(method) == "method"
    assert document_type(application) == "application"
    assert document_type({"version": "1", "steps": {}}) == "workflow"


def test_a_declared_type_that_contradicts_the_structure_is_refused():
    """`type` exists to catch a mistake, so a wrong one has to be an error."""
    with pytest.raises(ParseError) as err:
        document_type({"version": "1", "type": "workflow", **base_doc()})
    assert "reads as a protocol document" in str(err.value)


def test_an_unknown_type_is_refused_with_suggestions():
    with pytest.raises(ParseError) as err:
        document_type({"version": "1", "type": "aplication"})
    assert err.value.code == "P4"
    assert "application" in str(err.value)


# --------------------------------------------------------------------------- #
# what a method is
# --------------------------------------------------------------------------- #


def test_a_method_may_not_name_a_network():
    method, _ = split_doc()
    method["model"] = {"key": "gpt2"}
    with pytest.raises(ValidationError) as err:
        parse_method(in_order(method) | {"type": "method"})
    assert err.value.rule == 18


@pytest.mark.parametrize("section", ["reads", "save"])
def test_a_method_declares_what_it_measures(section):
    method, _ = split_doc()
    del method[section]
    with pytest.raises(ValidationError) as err:
        parse_method(method)
    assert err.value.rule == 18


def test_method_signature_names_exactly_what_is_open():
    method, _ = split_doc()
    signature = parse_method(method).signature
    assert signature.model is True
    assert signature.sites == {"tgt": ("layer",), "lm_head": ()}
    assert signature.data == ()
    assert not signature.is_closed()
    assert signature.lines() == (
        "model: key, revision, dtype (+ optional quantization)",
        "sites.tgt: layer",
    )


def test_an_unbound_site_name_is_in_the_signature():
    """A site a read references but the method leaves undeclared still has to
    be supplied — the signature is about addresses, not about declarations."""
    method, _ = split_doc()
    del method["sites"]["tgt"]
    assert signature_of(method).sites["tgt"] == ("component",)


def test_a_method_digests_by_content_not_by_declaration():
    method, _ = split_doc()
    same_words_different_order = {
        "type": "method",
        **{k: v for k, v in method.items() if k != "type"},
    }
    assert method_digest(method) == method_digest(same_words_different_order)
    edited = copy.deepcopy(method)
    edited["metrics"]["ld"]["a"] = "other_answer"
    assert method_digest(edited) != method_digest(method)


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #


def test_composition_is_transparent(env):
    """The property the whole split rests on: two files or one, the canonical
    bytes — and so the digest, the provenance unit — are the same."""
    method, application = split_doc()
    assert compose(method, application) == in_order(base_doc())
    assert load(application, env).document_digest == load(base_doc(), env).document_digest


def test_an_application_may_restate_what_the_method_already_fixed(env):
    method, application = split_doc()
    application["sites"]["lm_head"] = {"component": "lm_head"}  # the same value
    assert load(application, env).document_digest == load(base_doc(), env).document_digest


def test_an_application_may_not_overrule_its_method():
    method, application = split_doc()
    application["sites"]["tgt"]["component"] = "mlp_output"
    with pytest.raises(ValidationError) as err:
        compose(method, application)
    assert err.value.rule == 18
    assert "never overrules one" in str(err.value)


def test_an_unfilled_hole_is_refused_by_name():
    method, application = split_doc()
    del application["sites"]["tgt"]["layer"]
    with pytest.raises(ValidationError) as err:
        compose(method, application)
    assert err.value.rule == 18
    assert "sites.tgt: layer" in str(err.value)


def test_a_version_mismatch_is_refused():
    method, application = split_doc()
    application["version"] = "2"
    with pytest.raises(ValidationError) as err:
        compose(method, application)
    assert err.value.rule == 18


def test_descriptions_join_method_first():
    method, application = split_doc()
    method = as_method({**method, "description": "swap the residual"})
    application = {
        "version": "1",
        "type": "application",
        "description": "at L3 in gpt2",
        "method": method,
        **{k: v for k, v in application.items() if k in ("model", "sites")},
    }
    assert compose(method, application)["description"] == (
        "swap the residual\n\nat L3 in gpt2"
    )


def test_composed_documents_sweep_from_the_application(env):
    """A layer scan is an edit of the application; the method is untouched and
    keeps its digest."""
    method, application = split_doc()
    before = method_digest(method)
    application["sites"]["tgt"]["layer"] = {"sweep": [1, 2, 3]}
    loaded = load(application, env)
    assert len(loaded.expansion.points) == 3
    assert method_digest(method) == before


def test_a_method_cannot_be_run(env):
    method, _ = split_doc()
    with pytest.raises(ValidationError) as err:
        load(method, env)
    assert err.value.rule == 18
    assert "nothing to run" in str(err.value)


# --------------------------------------------------------------------------- #
# the loaded record
# --------------------------------------------------------------------------- #


def test_the_load_reports_the_method_it_composed(env, tmp_path):
    method, application = split_doc()
    loaded = load(application, env)
    assert loaded.method_digest == method_digest(method)
    assert loaded.method_ref is None  # inline
    assert load(base_doc(), env).method_digest is None


def test_a_method_path_resolves_against_the_application_file(env, tmp_path):
    method, application = split_doc()
    (tmp_path / "methods").mkdir()
    (tmp_path / "methods/m.json").write_text(json.dumps(method))
    application["method"] = "methods/m.json"
    path = tmp_path / "app.json"
    path.write_text(json.dumps(application))
    loaded = load(path, env)
    assert loaded.method_ref == "methods/m.json"
    assert loaded.method_digest == method_digest(method)


def test_a_missing_method_file_is_a_load_error(env, tmp_path):
    _, application = split_doc()
    application["method"] = "methods/nowhere.json"
    path = tmp_path / "app.json"
    path.write_text(json.dumps(application))
    with pytest.raises(ValidationError) as err:
        load(path, env)
    assert err.value.rule == 18


# --------------------------------------------------------------------------- #
# the shipped pair
# --------------------------------------------------------------------------- #


def test_the_shipped_application_composes_and_binds_everything(env):
    loaded = load(SHIPPED_APPLICATION, env)
    assert loaded.method_ref == "../methods/interchange.json"
    model = loaded.canonical_document["model"]
    assert model["dtype"] == "bf16"
    assert loaded.canonical_document["sites"]["target"] == {
        "component": "block_output",
        "layer": 18,
    }


def test_the_shipped_method_is_open_exactly_where_it_should_be():
    method = parse_method(json.loads(SHIPPED_METHOD.read_text()))
    assert method.signature.sites == {"target": ("layer",), "lm_head": ()}
    assert method.signature.data == ()
    assert method.signature.model is True
