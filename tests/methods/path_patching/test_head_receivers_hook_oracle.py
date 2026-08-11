"""Hook oracle for per-head value/query receivers + GQA addressing (GH #380).

Path patching reads internal receivers at per-head *value* and *query* vectors,
the pre-attention projections ``v_h = W_V^h·x`` / ``q_h = W_Q^h·x``. The value
vector is the documented, RoPE-free receiver, and under grouped-query attention
it is addressed in **KV-head space** (one vector per KV head, width
``head_dim``).

These tests pin that addressing through the path-patching resolver
(:func:`~causalab.methods.path_patching.targets.build_receiver_site` →
``HeadSite``) against a hand-rolled capture of ``v_proj``/``q_proj``'s output
sliced per head — no backbone imports in the oracle:

* on the coupled tiny Llama (``n_kv == n_heads``) each head's collected value
  equals its ``v_proj`` slice;
* on a GQA Llama (``n_kv == n_heads/2``) query heads remap to their KV group,
  each matching the corresponding ``v_proj`` KV slice — so the receiver lives
  in KV-head space, not query-head space;
* a full query-vector interchange through a Plan reproduces a hand-rolled
  ``q_proj`` swap — the per-head Edit the receiver-inject pass is built from.

See ``docs/PYVENE_HOOK_COVERAGE.md``.
"""

from __future__ import annotations

import pytest
import torch

from causalab.methods.path_patching.targets import ReceiverSpec, build_receiver_site
from causalab.neural.edit import Edit, ReadSource
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.head_view import HeadSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.plan import EditOp, Plan, run_plan
from causalab.neural.token_positions import TokenPosition

from tests.neural.activations.hook_oracle import (
    capture_component,
    component_edited_logits,
    head_slice,
    make_trace,
    next_token_logits,
)
from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME

_LAYER = 0
_POS = 1
_BASE = "the quick brown fox jumps"
_SOURCE = "a slow lazy old dog sits"


@pytest.fixture(scope="module")
def tiny_pipeline() -> LMPipeline:
    """Coupled tiny-random Llama (``n_kv == n_heads``, ``head_dim == hidden/n_head``).

    Fresh name-based load (not the session-cached model object): pyvene tests
    elsewhere run on that shared object and pyvene's cleanup clears ALL forward
    hooks, which would kill this module's nnterp wrapper.

    ``device="cpu"`` pins the CPU tier: a name-load resolves ``"auto" → cuda``
    whenever a GPU is merely visible (#222), but this oracle builds its ground
    truth with hand-rolled CPU tensors, so an implicit CUDA load mismatches
    (#471).
    """
    return LMPipeline(
        TINY_RANDOM_MODEL_NAME, max_new_tokens=1, padding_side="left", device="cpu"
    )


def _head_dim(pipeline: LMPipeline) -> int:
    cfg = pipeline.model.config
    return getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


def _capture_proj_output(
    pipeline: LMPipeline, proj: str, layer: int, inputs
) -> torch.Tensor:
    """A projection's output (concatenated per-head vectors), grabbed via our
    own forward hook — no backbone."""
    grabbed: dict[str, torch.Tensor] = {}
    module = getattr(pipeline.hf_model.model.layers[layer].self_attn, proj)

    def hook(_m, _i, out):
        grabbed["v"] = out.detach().clone()

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            pipeline.hf_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
    finally:
        handle.remove()
    return grabbed["v"]


def _collect_receiver(pipeline: LMPipeline, kind: str, head: int) -> torch.Tensor:
    """Resolve the receiver through the path-patching builder and collect it at
    ``_POS`` — the PASS-1 read the two-pass plan performs."""
    tp = TokenPosition(lambda _x: [_POS], pipeline, id="pos1")
    site = build_receiver_site(
        pipeline,
        ReceiverSpec(kind=kind, layer=_LAYER, head=head, token_position=tp),
    )
    assert isinstance(site, HeadSite)
    inputs = pipeline.load([make_trace(_BASE)])
    return site.collect(pipeline.model, inputs, positions=[_POS])[:, 0, :]


class TestHeadValueReceiverHookOracle:
    pytestmark = pytest.mark.property

    def test_value_matches_vproj_slice_coupled(self, tiny_pipeline: LMPipeline) -> None:
        """``n_kv == n_heads``: each head's value receiver equals its ``v_proj``
        slice ``[h·d_head : (h+1)·d_head]``."""
        d_head = _head_dim(tiny_pipeline)
        n_heads = tiny_pipeline.model.config.num_attention_heads
        v_out = _capture_proj_output(
            tiny_pipeline, "v_proj", _LAYER, tiny_pipeline.load([make_trace(_BASE)])
        )
        for head in range(n_heads):
            expected = v_out[:, _POS, head * d_head : (head + 1) * d_head]
            collected = _collect_receiver(tiny_pipeline, "head_value_input", head)
            torch.testing.assert_close(collected, expected, atol=1e-5, rtol=1e-4)

    def test_value_addressed_in_kv_space_under_gqa(
        self, gqa_tiny_lm: LMPipeline
    ) -> None:
        """``n_kv == n_heads/2``: query heads remap to their KV group, each equal
        to the matching ``v_proj`` KV slice. Distinct KV groups give distinct
        vectors — the receiver lives in KV-head space."""
        cfg = gqa_tiny_lm.model.config
        d_head = _head_dim(gqa_tiny_lm)
        n_kv = cfg.num_key_value_heads
        group = cfg.num_attention_heads // n_kv
        assert n_kv < cfg.num_attention_heads  # genuinely grouped
        v_out = _capture_proj_output(
            gqa_tiny_lm, "v_proj", _LAYER, gqa_tiny_lm.load([make_trace(_BASE)])
        )
        collected = [
            _collect_receiver(gqa_tiny_lm, "head_value_input", kv * group)
            for kv in range(n_kv)
        ]
        for kv in range(n_kv):
            expected = v_out[:, _POS, kv * d_head : (kv + 1) * d_head]
            torch.testing.assert_close(collected[kv], expected, atol=1e-5, rtol=1e-4)
        assert not torch.allclose(collected[0], collected[1], atol=1e-4)


class TestHeadQueryReceiverHookOracle:
    """The per-head *query* receiver — the query-path mirror of the value
    receiver, used by path patching's ``head_query_input`` edge. Unlike the
    value vector it is per-query-head even under GQA (only k/v are shared), and
    it reads the pre-RoPE ``q_proj`` output."""

    pytestmark = pytest.mark.property

    def test_query_matches_qproj_slice(self, tiny_pipeline: LMPipeline) -> None:
        """Each head's query receiver equals its (pre-RoPE) ``q_proj`` slice
        ``[h·d_head : (h+1)·d_head]``."""
        d_head = _head_dim(tiny_pipeline)
        n_heads = tiny_pipeline.model.config.num_attention_heads
        q_out = _capture_proj_output(
            tiny_pipeline, "q_proj", _LAYER, tiny_pipeline.load([make_trace(_BASE)])
        )
        for head in range(n_heads):
            expected = q_out[:, _POS, head * d_head : (head + 1) * d_head]
            collected = _collect_receiver(tiny_pipeline, "head_query_input", head)
            torch.testing.assert_close(collected, expected, atol=1e-5, rtol=1e-4)

    def test_query_interchange_matches_hook(self, tiny_pipeline: LMPipeline) -> None:
        """A full query-vector swap at the last token, lowered through a Plan
        (the per-head interchange Edit the receiver-inject pass is built from),
        reproduces a hand-rolled overwrite of the head's pre-RoPE ``q_proj``
        output slice with the source's. Intervening on the last token's query
        changes where that head attends, so the swap moves the next-token
        logits."""
        head = 1
        sl = head_slice(tiny_pipeline, head)
        q_proj = tiny_pipeline.hf_model.model.layers[_LAYER].self_attn.q_proj

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

        fsite = FeaturizedSite(HeadSite("query", _LAYER, head))
        plan = Plan(
            inputs={"source": source_inputs, "base": base_inputs},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        fsite,
                        g=lambda f, f_src: f_src,
                        read_sources=(
                            ReadSource(fsite, positions=[-1], input="source"),
                        ),
                        positions=[-1],
                    ),
                ),
            ),
            save_logits=("base",),
        )
        with torch.no_grad():
            result = run_plan(tiny_pipeline.model, plan)
        causalab = result.logits["base"][:, -1, :]
        # A single head's query swap moves the logits only slightly on a random
        # model (~1e-4), but the hand-rolled swap is a genuine, non-vacuous edit
        # vs clean, and the Plan must reproduce it far more tightly than that
        # edit's size — so a no-op lowering (causalab == clean, i.e. ~1e-4 off
        # `manual`) is caught by the tight `atol=1e-5, rtol=0` equivalence.
        assert not torch.allclose(manual, clean, atol=1e-5)
        torch.testing.assert_close(causalab, manual, atol=1e-5, rtol=0)
