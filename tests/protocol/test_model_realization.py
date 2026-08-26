"""`model.dtype` and `model.quantization` (spec §2.1, checklist rule 17).

Precision is not an execution flag: two runs of one protocol at bf16 and at
nf4 are two experiments. These tests pin that the document can say so, that
the canonical form always says so even when the author did not, that the
digest moves when the realization moves, and that a backend which cannot
realize a quantization refuses instead of running something else.
"""

from __future__ import annotations

from typing import Any

import pytest

from causalab.protocol.backend import Backend, choose_backend, requires
from causalab.protocol.canonical import canonicalize, digest
from causalab.protocol.errors import ParseError, ValidationError
from causalab.protocol.loader import load
from causalab.protocol.resolve import ARTIFACT_IDENTITY_KEYS
from causalab.protocol.schema import parse_document
from causalab.protocol.validate import validate_document

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit


def doc_with_model(**model: Any) -> dict[str, Any]:
    raw = base_doc()
    raw["model"] = {**raw["model"], **model}
    return in_order(raw)


# --------------------------------------------------------------------------- #
# dtype
# --------------------------------------------------------------------------- #


def test_dtype_materializes_even_when_unauthored(env):
    """An authored file may be silent about precision; a record may not."""
    assert canonicalize(base_doc(), env)["model"]["dtype"] == "fp32"


def test_dtype_is_part_of_the_experiment_not_of_the_run(env):
    fp32 = digest(canonicalize(base_doc(), env))
    bf16 = digest(canonicalize(doc_with_model(dtype="bf16"), env))
    assert fp32 != bf16
    # ... and an explicit fp32 is the same experiment as a silent one
    assert digest(canonicalize(doc_with_model(dtype="fp32"), env)) == fp32


def test_an_unknown_dtype_is_refused_with_suggestions():
    with pytest.raises(ParseError) as err:
        parse_document(doc_with_model(dtype="float16"))
    assert err.value.code == "P4"
    assert "fp16" in str(err.value)


def test_the_model_dtype_has_one_home(env):
    """`train.precision` used to carry a `model` entry that nothing enforced —
    it could (and in the corpus did) name a precision the run never used."""
    raw = base_doc()
    raw["train"] = {
        "objective": [[1.0, "ld"]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 4},
        "precision": {"feature": "fp32", "loss": "fp32", "model": "bf16"},
    }
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P3"
    assert "model" in str(err.value)


# --------------------------------------------------------------------------- #
# quantization
# --------------------------------------------------------------------------- #


NF4 = {"scheme": "nf4", "method": "bitsandbytes", "double_quant": True}


def test_quantization_materializes_its_scheme_defaults(env):
    canonical = canonicalize(
        doc_with_model(dtype="bf16", quantization={"scheme": "nf4"}), env
    )
    assert canonical["model"]["quantization"] == {
        "scheme": "nf4",
        "method": "bitsandbytes",
        "compute_dtype": "bf16",  # defaults to the model's own dtype
        "double_quant": False,
    }


def test_int8_materializes_its_own_knob(env):
    canonical = canonicalize(doc_with_model(quantization={"scheme": "int8"}), env)
    assert canonical["model"]["quantization"]["int8_threshold"] == 6.0
    assert "double_quant" not in canonical["model"]["quantization"]


def test_quantization_moves_the_digest(env):
    plain = digest(canonicalize(doc_with_model(dtype="bf16"), env))
    quantized = digest(
        canonicalize(doc_with_model(dtype="bf16", quantization=NF4), env)
    )
    assert plain != quantized


def test_there_is_no_bare_int4():
    """`int4` names no single realization — refusing it is the feature."""
    with pytest.raises(ParseError) as err:
        parse_document(doc_with_model(quantization={"scheme": "int4"}))
    assert err.value.code == "P4"
    assert "nf4" in str(err.value)


@pytest.mark.parametrize(
    "quantization,field",
    [
        ({"scheme": "int8", "double_quant": True}, "double_quant"),
        ({"scheme": "nf4", "int8_threshold": 6.0}, "int8_threshold"),
    ],
)
def test_a_knob_under_the_wrong_scheme_is_rule_17(quantization, field):
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(doc_with_model(quantization=quantization)))
    assert err.value.rule == 17
    assert field in str(err.value)


# --------------------------------------------------------------------------- #
# routing and stamping
# --------------------------------------------------------------------------- #


class _Plain(Backend):
    name = "plain"
    capabilities = frozenset({"paired_forward"})

    def execute(self, request):  # pragma: no cover — routing never gets here
        raise AssertionError


def test_quantization_requires_a_capability_and_refuses_without_it():
    doc = parse_document(doc_with_model(quantization=NF4))
    assert "quantized_weights" in requires(doc)
    with pytest.raises(ValidationError) as err:
        choose_backend(doc, [_Plain()])
    assert "quantized_weights" in str(err.value)


def test_the_realization_is_stampable_identity():
    """A featurizer fitted against bf16 weights is not the artifact fitted
    against fp32 ones, so the identity schema has to be able to say which."""
    assert "model_dtype" in ARTIFACT_IDENTITY_KEYS
    assert "model_quantization" in ARTIFACT_IDENTITY_KEYS


def test_a_swept_dtype_expands_like_any_other_axis(env):
    raw = doc_with_model(dtype={"sweep": ["fp32", "bf16"]})
    loaded = load(raw, env)
    assert len(loaded.expansion.points) == 2
    assert [c["model"]["dtype"] for c in loaded.canonical_points] == ["fp32", "bf16"]
    assert len(set(loaded.point_digests)) == 2


def test_a_fit_bundle_is_refused_at_a_different_realization(env, artifacts_root):
    """The other half of stamping: corpus 09 loads a rotation fitted in fp32,
    so asking to apply it at bf16 refuses rather than quietly mixing the two."""
    from tests.protocol._env import CORPUS_DIR

    assert load(CORPUS_DIR / "09_das_apply_im.json", env)  # fp32, as fitted
    with pytest.raises(ValidationError) as err:
        load(
            CORPUS_DIR / "09_das_apply_im.json", env, overrides={"model.dtype": "bf16"}
        )
    assert err.value.rule == 15
    assert "model_dtype" in str(err.value)
