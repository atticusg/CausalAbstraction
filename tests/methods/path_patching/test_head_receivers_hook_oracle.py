"""Hook oracle for per-head value/query receivers + GQA addressing (GH #380).

Path patching reads internal receivers at per-head *value* and *query* vectors
(pyvene ``head_value_output`` / ``head_query_output``), the pre-attention
projections ``v_h = W_V^h·x`` / ``q_h = W_Q^h·x``. The value vector is the
documented, RoPE-free receiver, and under grouped-query attention it is addressed
in **KV-head space** (one vector per KV head, width ``head_dim``).

These tests pin that addressing against a hand-rolled capture of ``v_proj``'s
output sliced per head — no pyvene:

* on the coupled tiny Llama (``n_kv == n_heads``) each head's collected value
  equals its ``v_proj`` slice;
* on a GQA Llama (``n_kv == n_heads/2``) the value receiver indexes KV heads
  (``head ∈ {0,1}``), each matching the corresponding ``v_proj`` KV slice — so the
  receiver lives in KV-head space, not query-head space.

The query→KV-group *query-head* remap (a causalab routing concern, not a pyvene
primitive) lives in ``path_patching.targets`` and is covered by
``test_targets.py``. See ``docs/PYVENE_HOOK_COVERAGE.md``.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.LM_units import AttentionHeadQuery, AttentionHeadValue
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition, get_last_token_index
from causalab.neural.units import InterchangeTarget
from causalab.neural.activations.collect import collect_features
from causalab.neural.activations.interchange_mode import (
    batched_interchange_intervention,
)

from tests.neural.activations.hook_oracle import (
    capture_component,
    cf_example,
    component_edited_logits,
    head_slice,
    make_trace,
    next_token_logits,
)
from tests._helpers.tiny import tiny_random_model

_LAYER = 0
_POS = 1
_BASE = "the quick brown fox jumps"
_SOURCE = "a slow lazy old dog sits"


@pytest.fixture(scope="module")
def tiny_pipeline() -> LMPipeline:
    """Coupled tiny-random Llama (``n_kv == n_heads``, ``head_dim == hidden/n_head``)."""
    return LMPipeline(tiny_random_model(), max_new_tokens=1, padding_side="left")


def _head_dim(pipeline: LMPipeline) -> int:
    cfg = pipeline.model.config
    return getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


def _capture_vproj_output(pipeline: LMPipeline, layer: int, inputs) -> torch.Tensor:
    """The ``v_proj`` output (concatenated per-KV-head value vectors), grabbed via
    our own forward hook — no pyvene."""
    grabbed: dict[str, torch.Tensor] = {}
    v_proj = pipeline.model.model.layers[layer].self_attn.v_proj

    def hook(_m, _i, out):
        grabbed["v"] = out.detach().clone()

    handle = v_proj.register_forward_hook(hook)
    try:
        with torch.no_grad():
            pipeline.model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
    finally:
        handle.remove()
    return grabbed["v"]


def _collect_head_value(pipeline: LMPipeline, head: int) -> torch.Tensor:
    tp = TokenPosition(lambda _x: [_POS], pipeline, id="pos1")
    unit = AttentionHeadValue(layer=_LAYER, head=head, token_indices=tp)
    dataset = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
    return collect_features(dataset, pipeline, [unit], batch_size=1)[unit.id]


class TestHeadValueReceiverHookOracle:
    pytestmark = pytest.mark.property

    def test_value_matches_vproj_slice_coupled(self, tiny_pipeline: LMPipeline) -> None:
        """``n_kv == n_heads``: each head's value receiver equals its ``v_proj``
        slice ``[h·d_head : (h+1)·d_head]``."""
        d_head = _head_dim(tiny_pipeline)
        n_heads = tiny_pipeline.model.config.num_attention_heads
        v_out = _capture_vproj_output(
            tiny_pipeline, _LAYER, tiny_pipeline.load([make_trace(_BASE)])
        )
        for head in range(n_heads):
            expected = v_out[:, _POS, head * d_head : (head + 1) * d_head]
            collected = _collect_head_value(tiny_pipeline, head)
            torch.testing.assert_close(collected, expected, atol=1e-5, rtol=1e-4)

    def test_value_addressed_in_kv_space_under_gqa(
        self, gqa_tiny_lm: LMPipeline
    ) -> None:
        """``n_kv == n_heads/2``: the value receiver indexes KV heads (``0..n_kv-1``),
        each equal to the matching ``v_proj`` KV slice. Distinct KV heads give
        distinct vectors — the receiver lives in KV-head space."""
        cfg = gqa_tiny_lm.model.config
        d_head = _head_dim(gqa_tiny_lm)
        n_kv = cfg.num_key_value_heads
        assert n_kv < cfg.num_attention_heads  # genuinely grouped
        v_out = _capture_vproj_output(
            gqa_tiny_lm, _LAYER, gqa_tiny_lm.load([make_trace(_BASE)])
        )
        collected = [_collect_head_value(gqa_tiny_lm, kv) for kv in range(n_kv)]
        for kv in range(n_kv):
            expected = v_out[:, _POS, kv * d_head : (kv + 1) * d_head]
            torch.testing.assert_close(collected[kv], expected, atol=1e-5, rtol=1e-4)
        assert not torch.allclose(collected[0], collected[1], atol=1e-4)


def _q_proj(pipeline: LMPipeline, layer: int) -> torch.nn.Module:
    return pipeline.model.model.layers[layer].self_attn.q_proj


def _collect_head_query(pipeline: LMPipeline, head: int) -> torch.Tensor:
    tp = TokenPosition(lambda _x: [_POS], pipeline, id="pos1")
    unit = AttentionHeadQuery(layer=_LAYER, head=head, token_indices=tp)
    dataset = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
    return collect_features(dataset, pipeline, [unit], batch_size=1)[unit.id]


class TestHeadQueryReceiverHookOracle:
    """The per-head *query* receiver (``head_query_output``) — the query-path mirror
    of the value receiver, used by path patching's ``head_query_input`` edge.
    Unlike the value vector it is per-query-head even under GQA (only k/v are
    shared), and pyvene exposes it at the pre-RoPE ``q_proj`` output. This receiver
    was previously untested anywhere (GH #380)."""

    pytestmark = pytest.mark.property

    def test_query_matches_qproj_slice(self, tiny_pipeline: LMPipeline) -> None:
        """Each head's query receiver equals its (pre-RoPE) ``q_proj`` slice
        ``[h·d_head : (h+1)·d_head]``."""
        d_head = _head_dim(tiny_pipeline)
        n_heads = tiny_pipeline.model.config.num_attention_heads
        q_out = capture_component(
            tiny_pipeline,
            _q_proj(tiny_pipeline, _LAYER),
            "out",
            tiny_pipeline.load([make_trace(_BASE)]),
        )
        for head in range(n_heads):
            expected = q_out[:, _POS, head * d_head : (head + 1) * d_head]
            collected = _collect_head_query(tiny_pipeline, head)
            torch.testing.assert_close(collected, expected, atol=1e-5, rtol=1e-4)

    def test_query_interchange_matches_hook(self, tiny_pipeline: LMPipeline) -> None:
        """A full query-vector swap at the last token reproduces a hand-rolled
        overwrite of the head's pre-RoPE ``q_proj`` output slice with the source's,
        run through the rest of the model — the query-receiver edge. Intervening on
        the last token's query changes where that head attends, so the swap moves
        the next-token logits."""
        head = 1
        sl = head_slice(tiny_pipeline, head)
        q_proj = _q_proj(tiny_pipeline, _LAYER)
        tp = TokenPosition(
            lambda inp: get_last_token_index(inp, tiny_pipeline),
            tiny_pipeline,
            id="last",
        )
        unit = AttentionHeadQuery(layer=_LAYER, head=head, token_indices=tp)
        target = InterchangeTarget([[unit]])

        base_inputs = tiny_pipeline.load([make_trace(_BASE)])
        source_inputs = tiny_pipeline.load([make_trace(_SOURCE)])
        src_q = capture_component(tiny_pipeline, q_proj, "out", source_inputs)[
            :, -1, sl
        ]

        def edit(h: torch.Tensor) -> None:
            h[:, -1, sl] = src_q

        manual = component_edited_logits(
            tiny_pipeline, base_inputs, q_proj, "out", edit
        )
        clean = next_token_logits(tiny_pipeline, base_inputs)
        out = batched_interchange_intervention(
            tiny_pipeline,
            [cf_example(_BASE, _SOURCE)],
            target,
            output_scores=True,
        )
        causalab = out["scores"][0]
        # A single head's query swap moves the logits only slightly on a random
        # model (~1e-4), but the hand-rolled swap is a genuine, non-vacuous edit
        # vs clean, and pyvene must reproduce it far more tightly than that edit's
        # size — so a no-op pyvene intervention (causalab == clean, i.e. ~1e-4 off
        # `manual`) is caught by the tight `atol=1e-5, rtol=0` equivalence.
        assert not torch.allclose(manual, clean, atol=1e-5)
        torch.testing.assert_close(causalab, manual, atol=1e-5, rtol=0)
