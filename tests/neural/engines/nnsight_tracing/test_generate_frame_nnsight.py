"""The generated frame on the nnsight engine (engine plan §10, N8).

One ``model.generate`` trace per group: prompt-frame taps and writes bind
occurrence 0 of their locations — the prefill — and the decode steps are
walked with ``tracer.iter``, occurrence ``j`` being the step that consumes
generated token ``j-1``. The reference engine hand-rolls the same decode with
hooks, which makes it the oracle here: the same documents through both, ids
and activations agreeing.

Plus what parity cannot say: the greedy self-consistency pin (the argmax of a
continuation ``lm_head`` read reproduces the decoded ids — from the
generation-frame handoff note), and the N7 bridge — the DeltaNet state read
per decode step, through the *recurrent* kernel's own address, continuous
with the prefill chunks.
"""

from __future__ import annotations

import pytest

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import parse_document
from causalab.protocol.validate import validate_document

from tests.protocol._docs import in_order

from .test_parity_module_boundaries import _assert_same, _data, _executor, _refusal

pytestmark = pytest.mark.smoke

DEPTH = 4
QWEN_ATTENTION_LAYER = 3
DELTANET_LAYER = 0


def _gen_doc(component: str, *, layer: int | None, pos: dict | None = None) -> dict:
    site: dict = {"component": component}
    if layer is not None:
        site["layer"] = layer
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=False),
        "sites": {"tap": site},
        "positions": {
            "window": pos or {"generated": {"max_new_tokens": DEPTH}, "all": True}
        },
        "reads": {
            "r": {"site": "tap", "pos": "window", "model": "original", "input": "base"}
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


# --------------------------------------------------------------------------- #
# parity: the reference engine's decode is the oracle
# --------------------------------------------------------------------------- #

LLAMA_GEN_READS = [
    ("block_output", 1),
    ("attention_output", 1),
    ("mlp_output", 1),
    ("attention_query", 1),
    ("attention_z", 1),
    ("ln_final", None),
    ("lm_head", None),
]


@pytest.mark.parametrize("component,layer", LLAMA_GEN_READS)
def test_generated_read_parity(hooks_llama, trace_llama, component, layer):
    doc = _gen_doc(component, layer=layer)
    hooked = _executor(PointExecutor, doc, hooks_llama, with_cf=False).read_value("r")
    traced = _executor(TracePointExecutor, doc, trace_llama, with_cf=False).read_value(
        "r"
    )
    _assert_same(hooked, traced, f"generated read of {component!r}")


def test_generated_parity_on_the_hybrid_fixture(hooks_qwen, trace_qwen):
    """The target architecture in miniature: a DeltaNet layer's boundary and a
    full-attention interior slot, per step."""
    for component, layer in (
        ("attention_output", DELTANET_LAYER),
        ("attention_z", QWEN_ATTENTION_LAYER),
        ("lm_head", None),
    ):
        doc = _gen_doc(component, layer=layer)
        hooked = _executor(PointExecutor, doc, hooks_qwen, with_cf=False).read_value(
            "r"
        )
        traced = _executor(
            TracePointExecutor, doc, trace_qwen, with_cf=False
        ).read_value("r")
        _assert_same(hooked, traced, f"generated read of {component!r}")


def test_the_decoded_ids_agree_with_the_reference_engine(hooks_qwen, trace_qwen):
    """Same greedy continuation on both engines — the frame itself, not just
    the activations."""
    doc = _gen_doc("block_output", layer=0)
    hooked = _executor(PointExecutor, doc, hooks_qwen, with_cf=False)
    traced = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False)
    hooked.read_value("r"), traced.read_value("r")
    assert hooked.generated_ids("r") == traced.generated_ids("r")
    assert hooked.addressed_steps("r") == traced.addressed_steps("r")


def test_a_write_reaches_the_continuation_identically(hooks_qwen, trace_qwen):
    """Writes are prefill-only on both engines (here: everything binds
    occurrence 0 — the prefill), and reach the continuation through the first
    token and the cache. The patched continuation read must agree."""
    doc = _gen_doc("block_output", layer=0)
    doc["data"] = _data(with_cf=True)
    doc["sites"]["src"] = {"component": "block_output", "layer": 0}
    doc["reads"]["v_cf"] = {
        "site": "src",
        "pos": -1,
        "model": "original",
        "input": "counterfactual",
    }
    doc["reads"]["r"]["model"] = "patched"
    doc["writes"] = {"patch": {"site": "src", "pos": -1, "do": {"swap": "v_cf"}}}
    doc["intervened_models"] = {"patched": {"input": "base", "writes": ["patch"]}}
    doc["save"][0]["model"] = "patched"
    hooked = _executor(PointExecutor, doc, hooks_qwen, with_cf=True)
    traced = _executor(TracePointExecutor, doc, trace_qwen, with_cf=True)
    _assert_same(
        hooked.read_value("r"),
        traced.read_value("r"),
        "a patched continuation read",
    )
    assert hooked.generated_ids("r") == traced.generated_ids("r")


# --------------------------------------------------------------------------- #
# the greedy self-consistency pin
# --------------------------------------------------------------------------- #


def test_the_lm_head_argmax_reproduces_the_decoded_ids(trace_qwen):
    """The distribution at generated position i is the one AFTER token i, so
    its argmax is token i+1 — the frame and the values must tell one story."""
    doc = _gen_doc("lm_head", layer=None)
    executor = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False)
    value = executor.read_value("r")  # (b, steps, vocab)
    ids = executor.generated_ids("r")
    steps = executor.addressed_steps("r")
    for row, row_steps in enumerate(steps):
        for k, step in enumerate(row_steps[:-1]):
            assert int(value[row, step].argmax()) == ids[row][k + 1]


# --------------------------------------------------------------------------- #
# the N7 bridge: the DeltaNet state per decode step
# --------------------------------------------------------------------------- #


def _single_row(trace_qwen, doc_raw: dict, text: str) -> TracePointExecutor:
    doc = parse_document(in_order(doc_raw))
    validate_document(doc, engine_is_local=True)
    return TracePointExecutor(
        doc,
        trace_qwen,
        role_rows={"base": [{"input": text}]},
        role_fields={"base": "input"},
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )


def test_the_deltanet_state_reads_per_decode_step(trace_qwen):
    """Served through the *recurrent* kernel's address — the decode path's own
    dispatch — with one state per generated position, continuous in shape
    with the prefill chunks and advancing every step."""
    info = trace_qwen.info
    doc = _gen_doc("deltanet_state", layer=DELTANET_LAYER)
    doc["sites"]["prefill"] = {"component": "deltanet_state", "layer": DELTANET_LAYER}
    doc["reads"]["last_chunk"] = {
        "site": "prefill",
        "pos": -1,
        "model": "original",
        "input": "base",
    }
    doc["save"].append(
        {
            "value": "last_chunk",
            "model": "original",
            "input": "base",
            "file_path": "c.safetensors",
        }
    )
    executor = _single_row(trace_qwen, doc, "the quick brown fox jumps")
    per_step = executor.read_value("r")
    width = (
        info.linear_num_value_heads
        * info.linear_key_head_dim
        * info.linear_value_head_dim
    )
    assert tuple(per_step.shape) == (1, DEPTH, width)
    # the state advances every step...
    for j in range(DEPTH - 1):
        assert float((per_step[:, j + 1] - per_step[:, j]).abs().max()) > 0.0
    # ...and continues from the prefill: the first decode state is one token's
    # update away from the last chunk state, not a re-initialization
    last_chunk = executor.read_value("last_chunk")
    drift_from_prefill = float((per_step[:, 0] - last_chunk[:, 0]).abs().max())
    fresh_scale = float(per_step[:, 0].abs().max())
    assert 0.0 < drift_from_prefill < fresh_scale * 10


def test_an_expert_interior_read_in_the_generated_frame_refuses_by_name(trace_qwen):
    """No generated-frame address is verified for the grouped experts kernel,
    and decode dispatches different code than prefill — refused by name, not
    served from the wrong table."""
    doc = _gen_doc("expert_gate_proj", layer=0)
    with pytest.raises(ProtocolError, match="generated-frame address"):
        _executor(TracePointExecutor, doc, trace_qwen, with_cf=False).run_all()


# --------------------------------------------------------------------------- #
# refusals stay shared: the axes rule, the same words on both engines
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component", ["attention_key", "attention_scores", "attention_probs"]
)
def test_unstackable_generated_reads_refuse_identically(
    hooks_qwen, trace_qwen, component
):
    """A key-indexed tensor grows with the cache, so its steps do not stack —
    a fact about the declared axes, shared by both engines word for word."""
    doc = _gen_doc(component, layer=QWEN_ATTENTION_LAYER)
    assert _refusal(PointExecutor, doc, hooks_qwen) == _refusal(
        TracePointExecutor, doc, trace_qwen
    )


def test_routing_no_longer_sends_generate_documents_away():
    doc = parse_document(in_order(_gen_doc("block_output", layer=1)))
    assert "generate" in NnsightEngine().capabilities
    from causalab.protocol.engine import requires

    assert "generate" in requires(doc)
    assert requires(doc) <= NnsightEngine().effective_capabilities
