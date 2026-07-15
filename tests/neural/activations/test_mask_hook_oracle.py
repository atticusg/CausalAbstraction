"""Hook oracle for the differentiable-binary-mask intervention (GH #380).

The mask intervention (DBM / boundless-DAS masking) gates each feature between
the base and source featurized activations:
``f_out = (1-gate)·f_base + gate·f_src``, where in training mode
``gate = σ(mask/τ)`` (a soft, temperature-annealed gate) and in eval mode
``gate = 1[σ(mask) > 0.5]`` (a hard binary gate). The result is reconstructed
through the inverse featurizer.

These tests pin that blend against a hand-rolled forward-hook ground truth — no
pyvene — by setting the mask logits and temperature explicitly. The identity
featurizer makes the feature-space gate a raw-activation gate; a rotation
featurizer proves the gate is applied in feature space (and reconstructed). The
sparsity regulariser ``‖σ(mask/τ)‖₁`` is pinned as a pure function. See
``docs/PYVENE_HOOK_COVERAGE.md``.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.LM_units import ResidualStream
from causalab.neural.featurizer import Featurizer
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget
from causalab.neural.activations.intervenable_model import (
    delete_intervenable_model,
    prepare_intervenable_model,
)
from causalab.neural.activations.interchange_mode import (
    batched_interchange_intervention,
)

from tests.neural.activations.hook_oracle import (
    capture_residual,
    cf_example,
    make_trace,
    next_token_logits,
    random_rotation,
)

_LAYER = 0
_POS = 1
_BASE = "the quick brown fox jumps"
_SOURCE = "a slow lazy old dog sits"


def _mask_target(pipeline: LMPipeline, featurizer: Featurizer) -> InterchangeTarget:
    tp = TokenPosition(lambda _x: [_POS], pipeline, id="pos1")
    unit = ResidualStream(
        layer=_LAYER,
        token_indices=tp,
        target_output=True,
        shape=(pipeline.model.config.hidden_size,),
        featurizer=featurizer,
    )
    return InterchangeTarget([[unit]])


def _set_masks(model, mask_logits: torch.Tensor, temp: float, *, train: bool) -> None:
    """Pin every mask intervention's logits + temperature and train/eval mode."""
    for v in model.interventions.values():
        iv = v[0] if isinstance(v, tuple) else v
        with torch.no_grad():
            iv.mask.copy_(mask_logits.to(iv.mask.dtype))
        iv.set_temperature(temp)
        iv.train(train)


def _run_mask(
    pipeline: LMPipeline,
    target: InterchangeTarget,
    mask_logits: torch.Tensor,
    temp: float,
    *,
    train: bool,
) -> torch.Tensor:
    model = prepare_intervenable_model(pipeline, target, intervention_type="mask")
    _set_masks(model, mask_logits, temp, train=train)
    out = batched_interchange_intervention(
        pipeline, model, [cf_example(_BASE, _SOURCE)], target, output_scores=True
    )
    return out["scores"][0]


def _gate(mask_logits: torch.Tensor, temp: float, *, train: bool) -> torch.Tensor:
    if train:
        return torch.sigmoid(mask_logits / temp)
    return (torch.sigmoid(mask_logits) > 0.5).float()


class TestMaskInterventionHookOracle:
    """Mask gating at ``(_LAYER, _POS)`` vs. a forward-hook ground truth."""

    pytestmark = pytest.mark.unit

    def test_eval_hard_gate_identity(self, oracle_pipeline: LMPipeline) -> None:
        """Eval mode hard-gates each feature: first half take the source, second
        half keep the base. Identity featurizer → the oracle splices raw residual
        dims directly."""
        hidden = oracle_pipeline.model.config.hidden_size
        mask_logits = torch.cat(
            [
                torch.full((hidden // 2,), 10.0),
                torch.full((hidden - hidden // 2,), -10.0),
            ]
        )
        target = _mask_target(oracle_pipeline, Featurizer(n_features=hidden))

        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        source_inputs = oracle_pipeline.load([make_trace(_SOURCE)])
        fb = capture_residual(oracle_pipeline, _LAYER, base_inputs)[:, _POS, :]
        fs = capture_residual(oracle_pipeline, _LAYER, source_inputs)[:, _POS, :]
        gate = _gate(mask_logits, 1.0, train=False)
        patch = ((1 - gate) * fb + gate * fs).unsqueeze(1)
        manual = next_token_logits(oracle_pipeline, base_inputs, _LAYER, [_POS], patch)
        clean = next_token_logits(oracle_pipeline, base_inputs)

        causalab = _run_mask(oracle_pipeline, target, mask_logits, 1.0, train=False)
        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_train_soft_gate_identity(self, oracle_pipeline: LMPipeline) -> None:
        """Training mode uses the soft gate ``σ(mask/τ)`` — a genuine per-feature
        blend (here ~0.62 / ~0.38), reconstructed in activation space."""
        hidden = oracle_pipeline.model.config.hidden_size
        temp = 2.0
        mask_logits = torch.cat(
            [torch.full((hidden // 2,), 1.0), torch.full((hidden - hidden // 2,), -1.0)]
        )
        target = _mask_target(oracle_pipeline, Featurizer(n_features=hidden))

        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        source_inputs = oracle_pipeline.load([make_trace(_SOURCE)])
        fb = capture_residual(oracle_pipeline, _LAYER, base_inputs)[:, _POS, :]
        fs = capture_residual(oracle_pipeline, _LAYER, source_inputs)[:, _POS, :]
        gate = _gate(mask_logits, temp, train=True)
        patch = ((1 - gate) * fb + gate * fs).unsqueeze(1)
        manual = next_token_logits(oracle_pipeline, base_inputs, _LAYER, [_POS], patch)

        causalab = _run_mask(oracle_pipeline, target, mask_logits, temp, train=True)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_gate_applied_in_feature_space(self, oracle_pipeline: LMPipeline) -> None:
        """With a rotation featurizer the gate acts in the rotated basis; the
        oracle rotates base & source, gates per feature, then rotates back —
        which differs from gating raw dims, so this pins feature-space routing."""
        hidden = oracle_pipeline.model.config.hidden_size
        R = random_rotation(hidden, seed=1)
        mask_logits = torch.cat(
            [
                torch.full((hidden // 2,), 10.0),
                torch.full((hidden - hidden // 2,), -10.0),
            ]
        )
        from tests.neural.activations.hook_oracle import rotate_featurizer

        target = _mask_target(oracle_pipeline, rotate_featurizer(R))

        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        source_inputs = oracle_pipeline.load([make_trace(_SOURCE)])
        fb_raw = capture_residual(oracle_pipeline, _LAYER, base_inputs)[:, _POS, :]
        fs_raw = capture_residual(oracle_pipeline, _LAYER, source_inputs)[:, _POS, :]
        gate = _gate(mask_logits, 1.0, train=False)
        f_out = (1 - gate) * (fb_raw @ R) + gate * (fs_raw @ R)
        patch = (f_out @ R.T).unsqueeze(1)
        manual = next_token_logits(oracle_pipeline, base_inputs, _LAYER, [_POS], patch)

        causalab = _run_mask(oracle_pipeline, target, mask_logits, 1.0, train=False)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_tied_mask_is_scalar_gate(self, oracle_pipeline: LMPipeline) -> None:
        """``tie_masks=True`` uses a single scalar mask for all features. A large
        positive logit → gate 1 everywhere → a full interchange swap of the
        source at the target position."""
        hidden = oracle_pipeline.model.config.hidden_size
        target = _mask_target(
            oracle_pipeline, Featurizer(n_features=hidden, tie_masks=True)
        )

        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        source_inputs = oracle_pipeline.load([make_trace(_SOURCE)])
        fs = capture_residual(oracle_pipeline, _LAYER, source_inputs)[:, _POS, :]
        manual = next_token_logits(
            oracle_pipeline, base_inputs, _LAYER, [_POS], fs.unsqueeze(1)
        )

        causalab = _run_mask(
            oracle_pipeline, target, torch.tensor([10.0]), 1.0, train=False
        )
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)


class TestMaskSparsityLoss:
    """The L1 sparsity regulariser is a pure function of the (mask, temperature)."""

    pytestmark = pytest.mark.numerical_unit

    def test_sparsity_loss_is_l1_of_soft_gate(
        self, oracle_pipeline: LMPipeline
    ) -> None:
        hidden = oracle_pipeline.model.config.hidden_size
        temp = 0.5
        mask_logits = torch.linspace(-3.0, 3.0, hidden)
        target = _mask_target(oracle_pipeline, Featurizer(n_features=hidden))
        model = prepare_intervenable_model(
            oracle_pipeline, target, intervention_type="mask"
        )
        _set_masks(model, mask_logits, temp, train=True)
        try:
            iv = next(iter(model.interventions.values()))
            iv = iv[0] if isinstance(iv, tuple) else iv
            loss = iv.get_sparsity_loss()
        finally:
            delete_intervenable_model(model)

        expected = torch.norm(torch.sigmoid(mask_logits / temp), p=1)
        torch.testing.assert_close(loss.detach().cpu(), expected, atol=1e-6, rtol=1e-5)
