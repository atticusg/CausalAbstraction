"""Hook oracle for cross-model patching (``source_representations``) — GH #380.

Cross-model patching collects activations from a *source* pipeline and injects
them into a *target* pipeline's base run, instead of running the
counterfactual through the target. The contract: the value written into the
target is the *source model's* activation at the counterfactual, gathered at
the source's tokenization.

This test pins that with a hand-rolled hook ground truth. To prove the injected
value comes from the source model (not the target), the source pipeline is a
deep copy of the target with perturbed weights — so its activation at the
counterfactual differs, and the target output must follow the *source's* value.
See ``docs/PYVENE_HOOK_COVERAGE.md``.
"""

from __future__ import annotations

import copy

import pytest
import torch

from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec
from causalab.neural.token_positions import TokenPosition
from causalab.neural.activations.interchange_mode import run_interchange_interventions

from tests.neural.activations.hook_oracle import (
    capture_residual,
    cf_example,
    make_trace,
    next_token_logits,
)

_LAYER = 0
_POS = 1
_BASE = "the quick brown fox jumps"
_SOURCE = "a slow lazy old dog sits"


def _perturbed_source(pipeline: LMPipeline) -> LMPipeline:
    """A genuinely different source model that shares the tokenizer: a fresh HF
    module of the same architecture with weights perturbed by fixed-seed noise.

    Rebuilt from ``config`` + ``state_dict`` (so a grouped-query config survives)
    rather than ``deepcopy``-ing ``pipeline.hf_model`` — the latter is an
    already-dispatched nnsight module, which nnterp cannot cleanly re-wrap into a
    forward-hookable model; a clean HF module onboards normally."""
    hf = pipeline.hf_model
    src_model = type(hf)(copy.deepcopy(hf.config))
    src_model.load_state_dict(hf.state_dict())
    src_model.eval()  # constructed models default to train mode; disable dropout
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in src_model.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g))
    return LMPipeline(src_model, max_new_tokens=1, padding_side="left")


def _single_pos_groups(pipeline: LMPipeline) -> list[list[SiteSpec]]:
    tp = TokenPosition(lambda _x: [_POS], pipeline, id="pos1")
    site = SiteSpec(
        fsite=FeaturizedSite(Site("block_output", _LAYER)),
        positions=tp,
        key=f"residual.L{_LAYER}.pos1",
        width=pipeline.model.config.hidden_size,
    )
    return [[site]]


class TestCrossModelHookOracle:
    pytestmark = pytest.mark.unit

    def test_injects_source_model_activation(self, oracle_pipeline: LMPipeline) -> None:
        """The target is patched with the *source* model's residual at the
        counterfactual position. The oracle captures that source-model activation
        by hand and patches the target's base with it."""
        source_pipeline = _perturbed_source(oracle_pipeline)
        groups = _single_pos_groups(oracle_pipeline)

        # Source model's residual at the counterfactual, captured by hand.
        cf_inputs = source_pipeline.load([make_trace(_SOURCE)])
        src_resid = capture_residual(source_pipeline, _LAYER, cf_inputs)[:, [_POS], :]

        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        manual = next_token_logits(
            oracle_pipeline, base_inputs, _LAYER, [_POS], src_resid
        )

        result = run_interchange_interventions(
            oracle_pipeline,
            [cf_example(_BASE, _SOURCE)],
            groups,
            source_pipeline=source_pipeline,
            output_scores=True,
        )
        causalab = result.scores[0]  # first generated step, (1, vocab)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_differs_from_same_model_patch(self, oracle_pipeline: LMPipeline) -> None:
        """Sanity: because the source weights differ, cross-model patching gives a
        different result than patching the *target's* own activation at the
        counterfactual (ordinary same-model interchange). Confirms the injected
        value really comes from the source model."""
        source_pipeline = _perturbed_source(oracle_pipeline)
        groups = _single_pos_groups(oracle_pipeline)

        # Target model's own residual at the counterfactual (same-model patch).
        cf_inputs = oracle_pipeline.load([make_trace(_SOURCE)])
        tgt_resid = capture_residual(oracle_pipeline, _LAYER, cf_inputs)[:, [_POS], :]
        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        same_model = next_token_logits(
            oracle_pipeline, base_inputs, _LAYER, [_POS], tgt_resid
        )

        result = run_interchange_interventions(
            oracle_pipeline,
            [cf_example(_BASE, _SOURCE)],
            groups,
            source_pipeline=source_pipeline,
            output_scores=True,
        )
        cross_model = result.scores[0]  # first generated step, (1, vocab)
        assert not torch.allclose(cross_model, same_model, atol=1e-4)
