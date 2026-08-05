"""Cross-family hook oracle for the interchange intervention (GH #380).

``test_interchange_mode.py`` pins interchange against forward-hook oracles, but
only on the Llama ``tiny_pipeline``. This file re-asserts the core interchange
contract — full residual swap and a component (MLP) swap — across model families
via the parametrised ``oracle_pipeline`` fixture (Llama / GPT-2 / grouped-query
Llama), so the gather/scatter and component→module mapping are verified on a
fused-QKV, absolute-position GPT-2 module tree as well as the separate-QKV RoPE
Llama tree. See ``docs/HOOK_ORACLES.md``.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.LM_units import MLP, AttentionOutput, ResidualStream
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget
from causalab.neural.activations.interchange_mode import (
    batched_interchange_intervention,
    run_interchange_interventions,
)

from tests.neural.activations.hook_oracle import (
    capture_component,
    capture_residual,
    cf_example,
    component_edited_logits,
    component_module,
    make_trace,
    next_token_logits,
)

_LAYER = 0
_POS = 1
_BASE = "the quick brown fox jumps"
_SOURCE = "a slow lazy old dog sits"


def _run_interchange(pipeline: LMPipeline, target: InterchangeTarget) -> torch.Tensor:
    out = batched_interchange_intervention(
        pipeline, [cf_example(_BASE, _SOURCE)], target, output_scores=True
    )
    return out["scores"][0]


def _component_unit(pipeline: LMPipeline, component: str):
    """A single-token unit for a component_type, at ``(_LAYER, _POS)``."""
    tp = TokenPosition(lambda _x: [_POS], pipeline, id="pos1")
    if component == "attention_output":
        return AttentionOutput(layer=_LAYER, token_indices=tp)
    return MLP(layer=_LAYER, token_indices=tp, location=component)


class TestInterchangeCrossFamily:
    pytestmark = pytest.mark.unit

    def test_full_residual_swap(self, oracle_pipeline: LMPipeline) -> None:
        """A full residual-stream swap must reproduce a hand-rolled overwrite of
        the base residual with the source's, at the target position — on every
        model family."""
        hidden = oracle_pipeline.model.config.hidden_size
        tp = TokenPosition(lambda _x: [_POS], oracle_pipeline, id="pos1")
        unit = ResidualStream(
            layer=_LAYER, token_indices=tp, target_output=True, shape=(hidden,)
        )
        target = InterchangeTarget([[unit]])

        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        source_inputs = oracle_pipeline.load([make_trace(_SOURCE)])
        patch = capture_residual(oracle_pipeline, _LAYER, source_inputs)[:, [_POS], :]
        manual = next_token_logits(oracle_pipeline, base_inputs, _LAYER, [_POS], patch)
        clean = next_token_logits(oracle_pipeline, base_inputs)

        causalab = _run_interchange(oracle_pipeline, target)
        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize(
        "component", ["mlp_output", "mlp_input", "mlp_activation", "attention_output"]
    )
    def test_component_interchange_swap(
        self, oracle_pipeline: LMPipeline, component: str
    ) -> None:
        """A component interchange must reproduce a hand-rolled overwrite of the
        base component activation with the source's — exercising the
        component→module mapping for ``mlp_output`` / ``mlp_input`` /
        ``mlp_activation`` (the intermediate activation: SwiGLU ``act_fn`` output on
        Llama, ``c_proj`` input on GPT-2) and the full-attention
        ``attention_output``, on each model family."""
        unit = _component_unit(oracle_pipeline, component)
        target = InterchangeTarget([[unit]])

        module, kind = component_module(oracle_pipeline, _LAYER, component)
        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        source_inputs = oracle_pipeline.load([make_trace(_SOURCE)])
        src = capture_component(oracle_pipeline, module, kind, source_inputs)[
            :, _POS, :
        ]

        def edit(h: torch.Tensor) -> None:
            h[:, _POS, :] = src

        manual = component_edited_logits(
            oracle_pipeline, base_inputs, module, kind, edit
        )
        clean = next_token_logits(oracle_pipeline, base_inputs)

        causalab = _run_interchange(oracle_pipeline, target)
        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_topk_scores_match_full_vocab(self, oracle_pipeline: LMPipeline) -> None:
        """``output_scores=k`` post-processes the full-vocab logits into top-k via
        ``convert_to_top_k``; the returned indices/logits must match ``torch.topk``
        of the independent full-vocab hook oracle for the same interchange."""
        k = 5
        hidden = oracle_pipeline.model.config.hidden_size
        tp = TokenPosition(lambda _x: [_POS], oracle_pipeline, id="pos1")
        unit = ResidualStream(
            layer=_LAYER, token_indices=tp, target_output=True, shape=(hidden,)
        )
        target = InterchangeTarget([[unit]])

        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        source_inputs = oracle_pipeline.load([make_trace(_SOURCE)])
        patch = capture_residual(oracle_pipeline, _LAYER, source_inputs)[:, [_POS], :]
        manual = next_token_logits(oracle_pipeline, base_inputs, _LAYER, [_POS], patch)
        expected = torch.topk(manual, k, dim=-1)

        result = run_interchange_interventions(
            oracle_pipeline, [cf_example(_BASE, _SOURCE)], target, output_scores=k
        )
        topk = result["scores"][0][0]
        torch.testing.assert_close(
            topk["top_k_indices"], expected.indices, atol=0, rtol=0
        )
        torch.testing.assert_close(
            topk["top_k_logits"].float(), expected.values.float(), atol=1e-4, rtol=1e-3
        )
