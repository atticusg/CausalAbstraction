"""The per-expert MoE interior on the nnsight engine (engine plan §8, N6).

No cross-engine parity exists for these — the reference engine has no
mechanism for tensors inside a fused forward — so the correctness story rests
on three legs instead:

* **two implementations of the same math**: the grouped_mm kernel this engine
  serves from, against transformers' own eager per-expert loop, reconstructed
  through ``tracer.iter`` (the clq §2 cross-check, as a test);
* **identities**: each slot's weighted contribution sums to ``routed_output``
  exactly; the activation is ``act(gate)·up`` exactly; the permutation is a
  permutation;
* **causal writes**: a swap moves the logits, a same-value swap moves nothing,
  a written slot reads back written.

Plus the ownership seams: the reference engine refuses these by name, and
routing chooses this engine without being asked.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.engine import choose_engine, component_capability
from causalab.protocol.errors import ProtocolError, ValidationError
from causalab.protocol.schema import parse_document

from tests.protocol._docs import in_order

from .test_parity_module_boundaries import ATOL, _data, _executor

pytestmark = pytest.mark.smoke

LAYER = 0  # every fixture layer carries the sparse-MoE block

EXPERT_COMPONENTS = (
    "expert_gate_proj",
    "expert_up_proj",
    "expert_activation",
    "expert_permutation",
    "expert_output",
)


def _read_doc(component: str, *, pos: object = -1, extra: dict | None = None) -> dict:
    doc = {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=False),
        "sites": {"tap": {"component": component, "layer": LAYER}},
        "reads": {
            "r": {"site": "tap", "pos": pos, "model": "original", "input": "base"}
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
    for name, site in (extra or {}).items():
        doc["sites"][f"{name}_site"] = site
        doc["reads"][name] = {
            "site": f"{name}_site",
            "pos": pos,
            "model": "original",
            "input": "base",
        }
        doc["save"].append(
            {
                "value": name,
                "model": "original",
                "input": "base",
                "file_path": f"{name}.safetensors",
            }
        )
    return doc


def _swap_doc(component: str) -> dict:
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=True),
        "sites": {
            "tap": {"component": component, "layer": LAYER},
            "head": {"component": "lm_head"},
        },
        "reads": {
            "v_cf": {
                "site": "tap",
                "pos": -1,
                "model": "original",
                "input": "counterfactual",
            },
            "logits": {"site": "head", "pos": -1, "model": "patched", "input": "base"},
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


def _read(trace_qwen, component: str, **kw) -> torch.Tensor:
    return _executor(
        TracePointExecutor, _read_doc(component, **kw), trace_qwen, with_cf=False
    ).read_value("r")


def _single_row_read(trace_qwen, component: str, text: str) -> torch.Tensor:
    """A one-example pos:"all" read — dense, so whole-frame identities can
    reshape it (the two-row harness's uneven lengths would make it ragged)."""
    from causalab.protocol.schema import parse_document
    from causalab.protocol.validate import validate_document

    doc = parse_document(in_order(_read_doc(component, pos="all")))
    validate_document(doc, engine_is_local=True)
    executor = TracePointExecutor(
        doc,
        trace_qwen,
        role_rows={"base": [{"input": text}]},
        role_fields={"base": "input"},
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )
    return executor.read_value("r")


# --------------------------------------------------------------------------- #
# reads: shapes and identities
# --------------------------------------------------------------------------- #


def test_the_interior_reads_with_the_declared_widths(trace_qwen):
    info = trace_qwen.info
    k = info.num_experts_per_tok
    for component, width in (
        ("expert_gate_proj", k * info.moe_intermediate_size),
        ("expert_up_proj", k * info.moe_intermediate_size),
        ("expert_activation", k * info.moe_intermediate_size),
        ("expert_output", k * info.hidden_size),
        ("expert_permutation", k),
    ):
        value = _read(trace_qwen, component)
        assert tuple(value.shape) == (2, 1, width), component
        if component == "expert_permutation":
            assert not value.dtype.is_floating_point


def test_the_activation_is_act_of_gate_times_up_exactly(trace_qwen):
    """The identity that says all three taps sit inside one _apply_gate call:
    same rows, same order, no kernel in between."""
    doc = _read_doc(
        "expert_gate_proj",
        extra={
            "up": {"component": "expert_up_proj", "layer": LAYER},
            "act": {"component": "expert_activation", "layer": LAYER},
        },
    )
    executor = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False)
    gate, up, act = (
        executor.read_value("r"),
        executor.read_value("up"),
        executor.read_value("act"),
    )
    torch.testing.assert_close(
        torch.nn.functional.silu(gate) * up, act, atol=0.0, rtol=0.0
    )


def test_expert_output_sums_to_routed_output(trace_qwen):
    """Each slot's weighted contribution, summed over the top-k axis, is the
    combined expert output — pinned against the module-boundary tap, which the
    parity suite already proves against the reference engine."""
    info = trace_qwen.info
    doc = _read_doc(
        "expert_output",
        extra={"routed": {"component": "routed_output", "layer": LAYER}},
    )
    executor = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False)
    per_slot = executor.read_value("r")
    routed = executor.read_value("routed")
    summed = per_slot.reshape(
        *per_slot.shape[:-1], info.num_experts_per_tok, info.hidden_size
    ).sum(-2)
    torch.testing.assert_close(summed, routed, atol=ATOL, rtol=0)


def test_the_permutation_is_a_permutation(trace_qwen):
    """Read whole: every (token, slot) pair's sorted-row index, each row index
    used exactly once."""
    value = _single_row_read(
        trace_qwen, "expert_permutation", "the quick brown fox jumps"
    )
    flat = value.reshape(-1)
    assert torch.equal(
        torch.sort(flat).values, torch.arange(flat.numel(), dtype=flat.dtype)
    )


def test_grouped_and_eager_implementations_agree(trace_qwen):
    """The §8 cross-check: transformers' own eager per-expert loop — a wholly
    independent implementation, one Python iteration per hit expert — rebuilt
    into the same (token, slot) frame through ``tracer.iter``, against the
    grouped_mm tensor this engine serves. clq §2, as a test."""
    import nnsight

    info = trace_qwen.info
    k, hidden = info.num_experts_per_tok, info.hidden_size
    text = "the quick brown fox jumps"
    served = _single_row_read(trace_qwen, "expert_output", text)  # (1, s, k·hidden)
    rows = served.shape[0] * served.shape[1]
    served = served.reshape(rows, k, hidden)

    batch = trace_qwen.tokenizer([text], return_tensors="pt", padding=True)
    experts = trace_qwen.blocks[LAYER].mlp.experts
    model = trace_qwen.model
    model.set_experts_implementation("eager")
    try:
        with torch.no_grad():
            with model.trace(dict(batch)) as tracer:
                loop = experts.source.experts_forward_1.source
                n_hit = len(loop.nonzero_0.output)
                per_expert = nnsight.save([])
                for _ in tracer.iter[:n_hit]:
                    top_k_pos, token_idx = loop.torch_where_0.output
                    per_expert.append(
                        (top_k_pos, token_idx, loop.current_hidden_states_2.output)
                    )
    finally:
        model.set_experts_implementation("grouped_mm")

    rebuilt = torch.zeros(rows, k, hidden)
    for top_k_pos, token_idx, weighted in per_expert:
        rebuilt[token_idx, top_k_pos] = weighted.to(rebuilt.dtype)
    torch.testing.assert_close(rebuilt, served, atol=1e-5, rtol=0)


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component", ["expert_gate_proj", "expert_activation", "expert_output"]
)
def test_a_swap_moves_the_logits_and_a_self_swap_does_not(trace_qwen, component):
    def logits(doc: dict) -> torch.Tensor:
        return _executor(TracePointExecutor, doc, trace_qwen, with_cf=True).dense_value(
            "logits"
        )

    clean_doc = _read_doc("expert_output")
    clean_doc["sites"]["tap"] = {"component": "lm_head"}
    clean = _executor(
        TracePointExecutor, clean_doc, trace_qwen, with_cf=False
    ).read_value("r")

    moved = float((logits(_swap_doc(component)) - clean).abs().max())
    assert moved > 1e-5, f"{component}: the swap landed nowhere"

    self_swap = _swap_doc(component)
    self_swap["reads"]["v_cf"]["input"] = "base"
    unmoved = float((logits(self_swap) - clean).abs().max())
    assert unmoved == 0.0, f"{component}: a same-value swap must be the identity"


def test_a_written_slot_reads_back_written(trace_qwen):
    doc = _swap_doc("expert_gate_proj")
    doc["reads"]["obs"] = {
        "site": "tap",
        "pos": -1,
        "model": "patched",
        "input": "base",
    }
    doc["save"].append(
        {
            "value": "obs",
            "model": "patched",
            "input": "base",
            "file_path": "o.safetensors",
        }
    )
    executor = _executor(TracePointExecutor, doc, trace_qwen, with_cf=True)
    src, obs = executor.read_value("v_cf"), executor.read_value("obs")
    assert float((obs - src).abs().max()) == 0.0


def test_the_permutation_refuses_writes_as_kernel_bookkeeping(trace_qwen):
    doc = _swap_doc("expert_permutation")
    with pytest.raises(ProtocolError, match="row bookkeeping"):
        _executor(TracePointExecutor, doc, trace_qwen, with_cf=True).run_all()


# --------------------------------------------------------------------------- #
# ownership: the reference engine refuses by name, routing knows the owner
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", EXPERT_COMPONENTS)
def test_the_reference_engine_refuses_by_name(hooks_qwen, component):
    doc = _read_doc(component)
    with pytest.raises(ProtocolError, match="nnsight engine"):
        _executor(PointExecutor, doc, hooks_qwen, with_cf=False).run_all()


def test_routing_chooses_this_engine_even_listed_second():
    doc = parse_document(in_order(_read_doc("expert_gate_proj")))
    chosen = choose_engine(doc, [PytorchHooksEngine(), NnsightEngine()])
    assert isinstance(chosen, NnsightEngine)
    assert component_capability("expert_gate_proj") not in (
        PytorchHooksEngine().effective_capabilities
    )


def test_the_generated_refusal_names_the_missing_component():
    doc = parse_document(in_order(_read_doc("expert_output")))
    with pytest.raises(ValidationError, match="component:expert_output"):
        choose_engine(doc, [PytorchHooksEngine()])
