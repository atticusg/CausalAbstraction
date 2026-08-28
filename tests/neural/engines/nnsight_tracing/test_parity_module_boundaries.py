"""The parity suite (engine plan §7.1): the same documents through both
engines, asserting the answers agree.

This is simultaneously the nnsight engine's correctness proof for the
module-boundary vocabulary (N4) and the numerical oracle the 0.8 handoff
asked for — one artifact, two uses. Reads must agree to fp32-eager-CPU
tolerance, write effects on the logits must agree, and refusals must be the
*same* refusal (code and component named), because the policy tables are
single-homed.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.engine import choose_engine, component_capability
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import parse_document
from causalab.protocol.validate import validate_document

from tests.protocol._docs import in_order

pytestmark = pytest.mark.smoke

BASE_TEXTS = ["the quick brown fox jumps", "a small red hen sits still"]
CF_TEXTS = ["a slow green turtle sleeps", "the big blue whale dives deep"]

#: Reads agree to this tolerance: fp32, eager attention, CPU on both sides —
#: the operations are the same kernels in a different order of capture, so
#: anything beyond float-noise scale is an executor bug.
ATOL = 1e-5


# --------------------------------------------------------------------------- #
# drive: the same document through either executor
# --------------------------------------------------------------------------- #


def _executor(executor_cls, doc_raw, bundle, *, with_cf: bool):
    doc = parse_document(in_order(doc_raw))
    validate_document(doc, engine_is_local=True)
    rows = [
        {"input": base, "counterfactual_inputs": [cf]}
        for base, cf in zip(BASE_TEXTS, CF_TEXTS)
    ]
    role_rows = {"base": rows}
    role_fields = {"base": "input"}
    if with_cf:
        role_rows["counterfactual"] = rows
        role_fields["counterfactual"] = "counterfactual_inputs[0]"
    return executor_cls(
        doc,
        bundle,
        role_rows=role_rows,
        role_fields=role_fields,
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )


def _data(with_cf: bool) -> dict:
    data: dict = {"base": {"dataset": "inline", "field": "input"}}
    if with_cf:
        data["counterfactual"] = {
            "dataset": "inline",
            "field": "counterfactual_inputs[0]",
        }
    return data


def _read_doc(component: str, layer: int | None, head: int | None = None) -> dict:
    site: dict = {"component": component}
    if layer is not None:
        site["layer"] = layer
    if head is not None:
        site["head"] = head
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=False),
        "sites": {"tap": site},
        "reads": {
            "r": {"site": "tap", "pos": -1, "model": "original", "input": "base"}
        },
        "save": [
            {
                "value": "r",
                "model": "original",
                "input": "base",
                "file_path": "a.safetensors",
            }
        ],
    }


def _interchange_doc(component: str, layer: int | None) -> dict:
    """base_doc's shape: read the site on the counterfactual, swap it into
    the base forward, read the patched logits."""
    site: dict = {"component": component}
    if layer is not None:
        site["layer"] = layer
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=True),
        "sites": {"tap": site, "head": {"component": "lm_head"}},
        "reads": {
            "v_cf": {
                "site": "tap",
                "pos": -1,
                "model": "original",
                "input": "counterfactual",
            },
            "logits": {
                "site": "head",
                "pos": -1,
                "model": "patched",
                "input": "base",
            },
        },
        "writes": {"patch": {"site": "tap", "pos": -1, "do": {"swap": "v_cf"}}},
        "intervened_models": {"patched": {"input": "base", "writes": ["patch"]}},
        "save": [
            {
                "value": "logits",
                "model": "patched",
                "input": "base",
                "file_path": "l.safetensors",
            }
        ],
    }


def _assert_same(a, b, what: str) -> None:
    assert a.shape == b.shape, f"{what}: {tuple(a.shape)} != {tuple(b.shape)}"
    if not a.dtype.is_floating_point:
        assert torch.equal(a, b), f"{what}: integer values differ"
        return
    diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, atol=ATOL, rtol=0), (
        f"{what}: max abs diff {diff:.3e} exceeds {ATOL}"
    )


# --------------------------------------------------------------------------- #
# reads: every module-boundary component, both families
# --------------------------------------------------------------------------- #

LLAMA_READS = [
    ("input_ids", None, None),
    ("embeddings", None, None),
    ("block_input", 1, None),
    ("attention_input_norm", 1, None),
    ("attention_output", 1, None),
    ("attention_value", 1, None),
    ("attention_value", 1, 1),  # per-head slice of the o-projection input
    ("block_mid", 1, None),
    ("mlp_input_norm", 1, None),
    ("mlp_input", 1, None),
    ("mlp_activation", 1, None),
    ("mlp_output", 1, None),
    ("block_output", 1, None),
    ("ln_final", None, None),
    ("lm_head", None, None),
]

#: The hybrid/MoE surface, on the target architecture in miniature: layer 0
#: is Gated DeltaNet, layer 3 full attention, sparse MoE in every layer.
QWEN_READS = [
    ("block_output", 0, None),  # a DeltaNet layer's boundary
    ("attention_output", 0, None),  # the DeltaNet mixer's output
    ("attention_output", 3, None),  # the full-attention mixer's output
    ("router_logits", 0, None),
    ("router_scores", 0, None),
    ("expert_idx", 0, None),
    ("routed_output", 0, None),
    ("shared_expert_gate_proj", 0, None),
    ("shared_expert_up_proj", 0, None),
    ("shared_expert_activation", 0, None),
    ("shared_expert_output", 0, None),
    ("shared_expert_gate", 0, None),
]


@pytest.mark.parametrize("component,layer,head", LLAMA_READS)
def test_llama_read_parity(hooks_llama, trace_llama, component, layer, head):
    doc = _read_doc(component, layer, head)
    hooked = _executor(PointExecutor, doc, hooks_llama, with_cf=False).read_value("r")
    traced = _executor(TracePointExecutor, doc, trace_llama, with_cf=False).read_value(
        "r"
    )
    _assert_same(hooked, traced, f"read {component!r} (layer {layer}, head {head})")


@pytest.mark.parametrize("component,layer,head", QWEN_READS)
def test_qwen_read_parity(hooks_qwen, trace_qwen, component, layer, head):
    doc = _read_doc(component, layer, head)
    hooked = _executor(PointExecutor, doc, hooks_qwen, with_cf=False).read_value("r")
    traced = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False).read_value(
        "r"
    )
    _assert_same(hooked, traced, f"read {component!r} (layer {layer})")


# --------------------------------------------------------------------------- #
# writes: the interchange's effect on the logits agrees
# --------------------------------------------------------------------------- #

LLAMA_WRITES = [
    ("embeddings", None),
    ("block_input", 1),
    ("attention_output", 1),
    ("attention_value", 1),
    ("mlp_output", 1),
    ("block_output", 1),
]

QWEN_WRITES = [
    ("block_output", 0),  # a write on a DeltaNet layer's boundary
    ("attention_output", 3),
    ("router_scores", 0),
    ("expert_idx", 0),
    ("routed_output", 0),
    ("shared_expert_output", 0),
]


@pytest.mark.parametrize("component,layer", LLAMA_WRITES)
def test_llama_write_parity(hooks_llama, trace_llama, component, layer):
    doc = _interchange_doc(component, layer)
    hooked = _executor(PointExecutor, doc, hooks_llama, with_cf=True)
    traced = _executor(TracePointExecutor, doc, trace_llama, with_cf=True)
    _assert_same(
        hooked.dense_value("logits"),
        traced.dense_value("logits"),
        f"patched logits after a swap at {component!r}",
    )


@pytest.mark.parametrize("component,layer", QWEN_WRITES)
def test_qwen_write_parity(hooks_qwen, trace_qwen, component, layer):
    doc = _interchange_doc(component, layer)
    hooked = _executor(PointExecutor, doc, hooks_qwen, with_cf=True)
    traced = _executor(TracePointExecutor, doc, trace_qwen, with_cf=True)
    _assert_same(
        hooked.dense_value("logits"),
        traced.dense_value("logits"),
        f"patched logits after a swap at {component!r}",
    )


def test_a_write_moves_the_logits_at_all(hooks_qwen, trace_qwen):
    """Anti-vacuity for the write-parity tests: 'both engines agree' must not
    be satisfiable by 'neither write landed'."""
    doc = _interchange_doc("block_output", 0)
    clean = _read_doc("lm_head", None)
    for cls, bundle in ((PointExecutor, hooks_qwen), (TracePointExecutor, trace_qwen)):
        patched = _executor(cls, doc, bundle, with_cf=True).dense_value("logits")
        unpatched = _executor(cls, clean, bundle, with_cf=False).read_value("r")
        assert not torch.allclose(patched, unpatched, atol=ATOL), (
            f"{cls.__name__}: the interchange left the logits unchanged"
        )


# --------------------------------------------------------------------------- #
# refusals: the same policy, the same words
# --------------------------------------------------------------------------- #


def _refusal(executor_cls, doc, bundle) -> str:
    with pytest.raises(ProtocolError) as excinfo:
        executor = _executor(executor_cls, doc, bundle, with_cf=True)
        executor.run_all()
    return str(excinfo.value)


def test_read_only_refusal_is_identical(hooks_qwen, trace_qwen):
    doc = _interchange_doc("router_logits", 0)
    assert _refusal(PointExecutor, doc, hooks_qwen) == _refusal(
        TracePointExecutor, doc, trace_qwen
    )


def test_swap_only_refusal_is_identical(hooks_qwen, trace_qwen):
    doc = _interchange_doc("expert_idx", 0)
    doc["writes"]["patch"]["do"] = {"add_scaled": {"value": "v_cf", "scale": 2.0}}
    assert _refusal(PointExecutor, doc, hooks_qwen) == _refusal(
        TracePointExecutor, doc, trace_qwen
    )


def test_wrong_stream_refusal_is_identical(hooks_qwen, trace_qwen):
    doc = _read_doc("attention_output", 0)
    doc["sites"]["tap"]["stream"] = "full_attention"  # layer 0 is DeltaNet
    with pytest.raises(ProtocolError) as hooks_err:
        _executor(PointExecutor, doc, hooks_qwen, with_cf=False).run_all()
    with pytest.raises(ProtocolError) as trace_err:
        _executor(TracePointExecutor, doc, trace_qwen, with_cf=False).run_all()
    assert str(hooks_err.value) == str(trace_err.value)


# --------------------------------------------------------------------------- #
# routing: what this engine does not declare routes to the one that does
# --------------------------------------------------------------------------- #


def test_attention_probs_no_longer_routes_away():
    """N5 flipped this pin: the pattern (and the whole attention interior) is
    served here through the `.source` address table, so a document naming it
    stays on this engine when it is first in the list."""
    doc = parse_document(in_order(_read_doc("attention_probs", 3)))
    engines = [NnsightEngine(), PytorchHooksEngine()]
    chosen = choose_engine(doc, engines)
    assert isinstance(chosen, NnsightEngine)
    assert component_capability("attention_probs") in (
        NnsightEngine().effective_capabilities
    )
    assert "writable_attention_probs" in NnsightEngine().capabilities


def test_a_generated_read_reaching_the_executor_refuses_by_phase(trace_llama):
    doc = _read_doc("block_output", 1)
    doc["positions"] = {"tail": {"generated": {"max_new_tokens": 4}, "index": -1}}
    doc["reads"]["r"]["pos"] = "tail"
    with pytest.raises(ProtocolError) as excinfo:
        _executor(TracePointExecutor, doc, trace_llama, with_cf=False).run_all()
    assert "N8" in str(excinfo.value)
