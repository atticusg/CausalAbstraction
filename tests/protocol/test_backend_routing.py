"""Capability derivation and backend routing (spec §8)."""

from __future__ import annotations

import pytest

from causalab.protocol.backend import (
    Backend,
    ExecutionRequest,
    RunResult,
    choose_backend,
    requires,
)
from causalab.protocol.errors import ValidationError
from causalab.protocol.schema import parse_document

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit


class _Stub(Backend):
    def __init__(self, name: str, capabilities: frozenset[str], is_local: bool = False):
        self.name = name
        self.capabilities = capabilities
        self.is_local = is_local

    def execute(self, request: ExecutionRequest) -> RunResult:  # pragma: no cover
        raise NotImplementedError


def test_requires_paired_forward():
    doc = parse_document(base_doc())
    assert requires(doc) == frozenset({"paired_forward"})


def test_requires_empty_for_same_input_patching():
    raw = base_doc()
    raw["reads"]["v_cf"]["input"] = "base"
    del raw["data"]["counterfactual"]
    assert requires(parse_document(raw)) == frozenset()


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


def test_requires_full_logits_for_top_k():
    raw = base_doc()
    raw["metrics"]["tk"] = {"kind": "top_k", "of": "logits", "k": 5}
    raw["save"].append(
        {"value": "tk", "model": "patched", "input": "base", "file_path": "tk.json"}
    )
    assert "full_logits" in requires(parse_document(raw))


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


def test_choose_backend_first_covering():
    doc = parse_document(base_doc())
    weak = _Stub("serving", frozenset({"full_logits"}))
    strong = _Stub("hooks", frozenset({"grad", "paired_forward", "full_logits"}))
    assert choose_backend(doc, [weak, strong]) is strong


def test_refusal_names_missing_capabilities():
    doc = parse_document(base_doc())
    weak = _Stub("serving", frozenset({"full_logits"}))
    with pytest.raises(ValidationError) as err:
        choose_backend(doc, [weak])
    message = str(err.value)
    assert "paired_forward" in message and "serving" in message


def test_a_backend_without_generate_refuses_with_the_capability_named():
    """Routing is how a decode-less backend declines a continuation document,
    and the refusal names what it lacks rather than failing mid-run."""
    raw = base_doc()
    raw["positions"] = {"tail": {"generated": {"max_new_tokens": 8}, "index": -1}}
    raw["reads"]["logits"]["pos"] = "tail"
    doc = parse_document(in_order(raw))
    prefill_only = _Stub("prefill_only", frozenset({"paired_forward", "full_logits"}))
    with pytest.raises(ValidationError) as err:
        choose_backend(doc, [prefill_only])
    assert "generate" in str(err.value)
    decoder = _Stub("decoder", prefill_only.capabilities | {"generate"})
    assert choose_backend(doc, [prefill_only, decoder]) is decoder
