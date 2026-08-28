"""Capability derivation and engine routing (spec §8)."""

from __future__ import annotations

import pytest

from causalab.protocol.engine import (
    Engine,
    ExecutionRequest,
    RunResult,
    choose_engine,
    component_capability,
    requires,
)
from causalab.protocol.errors import ValidationError
from causalab.protocol.schema import COMPONENTS, parse_document

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit


class _Stub(Engine):
    """Coarse-capability stub: serves the whole component vocabulary, so the
    capability-routing tests below vary only the §8 verbs."""

    def __init__(
        self,
        name: str,
        capabilities: frozenset[str],
        is_local: bool = False,
        components: frozenset[str] = frozenset(COMPONENTS),
        writable_components: frozenset[str] = frozenset(COMPONENTS),
    ):
        self.name = name
        self.capabilities = capabilities
        self.is_local = is_local
        self.components = components
        self.writable_components = writable_components

    def execute(self, request: ExecutionRequest) -> RunResult:  # pragma: no cover
        raise NotImplementedError


#: base_doc touches block_output (read + write) and lm_head (read).
BASE_COMPONENTS = frozenset(
    {
        component_capability("block_output"),
        component_capability("block_output", write=True),
        component_capability("lm_head"),
    }
)


def test_requires_paired_forward():
    doc = parse_document(base_doc())
    assert requires(doc) == frozenset({"paired_forward"}) | BASE_COMPONENTS


def test_requires_component_entries_split_read_from_write():
    """Every touched site contributes its component; a written site also
    contributes the :write entry — the honest routing surface once two
    engines with different site vocabularies exist (§8)."""
    doc = parse_document(base_doc())
    needed = requires(doc)
    assert component_capability("block_output", write=True) in needed
    assert component_capability("lm_head") in needed
    assert component_capability("lm_head", write=True) not in needed


def test_requires_no_coarse_verbs_for_same_input_patching():
    raw = base_doc()
    raw["reads"]["v_cf"]["input"] = "base"
    del raw["data"]["counterfactual"]
    assert requires(parse_document(raw)) == BASE_COMPONENTS


def test_requires_full_logits_when_lm_head_read_saved():
    raw = base_doc()
    raw["save"].append(
        {
            "value": "logits",
            "model": "patched",
            "input": "base",
            "file_path": "l.safetensors",
        }
    )
    assert "full_logits" in requires(parse_document(raw))


def test_requires_full_logits_for_top_k_over_an_lm_head_read():
    raw = base_doc()
    raw["metrics"]["tk"] = {"kind": "top_k", "of": "logits", "k": 5, "by": "prob"}
    raw["save"].append(
        {"value": "tk", "model": "patched", "input": "base", "file_path": "tk.json"}
    )
    assert "full_logits" in requires(parse_document(raw))


def test_top_k_over_a_non_vocabulary_read_needs_no_full_logits():
    """The saving that motivates any-read ``top_k``: ranking a residual stream
    obliges no vocabulary projection anywhere, so it must not route the
    document onto a full-vocab engine."""
    raw = base_doc()
    raw["metrics"]["tk"] = {"kind": "top_k", "of": "v_cf", "k": 5, "by": "abs_value"}
    raw["save"].append(
        {
            "value": "tk",
            "model": "original",
            "input": "counterfactual",
            "file_path": "tk.json",
        }
    )
    assert "full_logits" not in requires(parse_document(raw))


def test_top_k_over_a_featurized_lm_head_read_still_needs_full_logits():
    """Capability and axis are two different questions, split on purpose.

    A featurizer takes the read's *value* out of token-id space (so `prob`
    and token decoding are refused / withheld), but serving the read still
    means materializing the whole projection — the featurizer consumes it.
    So the document still routes onto a full-vocab engine."""
    raw = base_doc()
    raw["featurizers"] = {
        "f": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    raw["reads"]["flogits"] = {
        "site": "lm_head",
        "pos": -1,
        "model": "patched",
        "input": "base",
        "featurizer": "f",
    }
    raw["metrics"]["tk"] = {"kind": "top_k", "of": "flogits", "k": 2, "by": "value"}
    raw["save"].append(
        {"value": "tk", "model": "patched", "input": "base", "file_path": "tk.json"}
    )
    assert "full_logits" in requires(parse_document(in_order(raw)))


def test_top_k_over_a_dims_sliced_lm_head_read_needs_no_full_logits():
    """A `dims` slice needs only its named vocabulary rows — the same rule the
    saved-read derivation already applies."""
    raw = base_doc()
    raw["reads"]["flogits"] = {
        "site": "lm_head",
        "pos": -1,
        "model": "patched",
        "input": "base",
        "dims": [0, 1, 2],
    }
    raw["metrics"]["tk"] = {"kind": "top_k", "of": "flogits", "k": 2, "by": "value"}
    raw["save"].append(
        {"value": "tk", "model": "patched", "input": "base", "file_path": "tk.json"}
    )
    assert "full_logits" not in requires(parse_document(in_order(raw)))


def test_requires_writable_attention_probs():
    raw = base_doc()
    raw["sites"]["probs"] = {"component": "attention_probs", "layer": 3}
    raw["writes"]["knock"] = {
        "site": "probs",
        "pos": -1,
        "do": {"clamp": {"lo": 0, "hi": 0}},
    }
    raw["intervened_models"]["patched"]["writes"].append("knock")
    assert "writable_attention_probs" in requires(parse_document(raw))


def test_choose_engine_first_covering():
    doc = parse_document(base_doc())
    weak = _Stub("serving", frozenset({"full_logits"}))
    strong = _Stub("hooks", frozenset({"grad", "paired_forward", "full_logits"}))
    assert choose_engine(doc, [weak, strong]) is strong


def test_refusal_names_missing_capabilities():
    doc = parse_document(base_doc())
    weak = _Stub("serving", frozenset({"full_logits"}))
    with pytest.raises(ValidationError) as err:
        choose_engine(doc, [weak])
    message = str(err.value)
    assert "paired_forward" in message and "serving" in message


def test_an_engine_without_generate_refuses_with_the_capability_named():
    """Routing is how a decode-less engine declines a continuation document,
    and the refusal names what it lacks rather than failing mid-run."""
    raw = base_doc()
    raw["positions"] = {"tail": {"generated": {"max_new_tokens": 8}, "index": -1}}
    raw["reads"]["logits"]["pos"] = "tail"
    doc = parse_document(in_order(raw))
    prefill_only = _Stub("prefill_only", frozenset({"paired_forward", "full_logits"}))
    with pytest.raises(ValidationError) as err:
        choose_engine(doc, [prefill_only])
    assert "generate" in str(err.value)
    decoder = _Stub("decoder", prefill_only.capabilities | {"generate"})
    assert choose_engine(doc, [prefill_only, decoder]) is decoder


def test_an_engine_without_the_component_refuses_by_name():
    """A document touching a component outside an engine's site vocabulary
    routes past it, and the generated refusal names the component entry —
    this is how interior components an engine cannot serve route to the one
    that can, with no hand-written case anywhere."""
    doc = parse_document(base_doc())
    verbs = frozenset({"paired_forward", "full_logits"})
    no_blocks = _Stub(
        "no_blocks",
        verbs,
        components=frozenset({"lm_head"}),
        writable_components=frozenset(),
    )
    with pytest.raises(ValidationError) as err:
        choose_engine(doc, [no_blocks])
    message = str(err.value)
    assert component_capability("block_output") in message
    assert component_capability("block_output", write=True) in message
    full = _Stub("full", verbs)
    assert choose_engine(doc, [no_blocks, full]) is full


def test_a_read_only_component_declaration_refuses_the_write():
    """components without writable_components serves reads but routes a
    write away."""
    doc = parse_document(base_doc())
    read_only = _Stub(
        "read_only",
        frozenset({"paired_forward", "full_logits"}),
        writable_components=frozenset(),
    )
    with pytest.raises(ValidationError) as err:
        choose_engine(doc, [read_only])
    assert component_capability("block_output", write=True) in str(err.value)
