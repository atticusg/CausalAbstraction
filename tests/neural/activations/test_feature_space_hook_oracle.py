"""Hook oracle for featurizer-space replace/steer + error preservation (GH #380).

Every featurized intervention runs ``featurize(base) -> (f_base, base_err)``,
edits ``f_base``, and reconstructs ``inverse(f_out, base_err)``. The ``base_err``
term is the part of the base activation *outside* the feature subspace; a lossy
featurizer must thread it through so the orthogonal component is preserved while
the in-subspace part is edited.

The existing ``test_interchange_mode.py`` proves this for the *interchange* type
with a full rotation (``base_err = None``). This file adds:

* **replace** through a full rotation — the replacement vector lands in the
  rotated basis, not raw dims;
* **additive steering** through a full rotation;
* **error preservation** with a *lossy* subspace featurizer (rotate, keep the
  first ``k`` coords as features and the remaining ``d-k`` as the error term):
  ``replace`` overwrites only the in-subspace coords and rebuilds the orthogonal
  remainder from the base — a different activation than overwriting all ``d``
  dims, so the test pins that the error term is honoured.

All ground truths are hand-rolled forward hooks — no pyvene. See
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
from causalab.methods.steer.steer import run_steering_interventions

from tests.neural.activations.hook_oracle import (
    capture_residual,
    make_trace,
    next_token_logits,
    random_rotation,
    rotate_featurizer,
)

_LAYER = 0
_POS = 1
_BASE = "the quick brown fox jumps"


def _target(pipeline: LMPipeline, featurizer: Featurizer) -> InterchangeTarget:
    tp = TokenPosition(lambda _x: [_POS], pipeline, id="pos1")
    unit = ResidualStream(
        layer=_LAYER,
        token_indices=tp,
        target_output=True,
        shape=(pipeline.model.config.hidden_size,),
        featurizer=featurizer,
    )
    return InterchangeTarget([[unit]])


def _run_steer(
    pipeline: LMPipeline,
    target: InterchangeTarget,
    vector: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    unit_id = target.flatten()[0].id
    result = run_steering_interventions(
        pipeline,
        [{"input": make_trace(_BASE), "counterfactual_inputs": []}],
        target,
        {unit_id: vector},
        mode=mode,
        output_scores=True,
    )
    return result["scores"][0][0]


class _SplitFeaturizerModule(torch.nn.Module):
    """Lossy split: rotate into basis ``R``, expose the first ``k`` coords as
    features and the trailing ``d-k`` coords as the error term."""

    def __init__(self, R: torch.Tensor, k: int) -> None:
        super().__init__()
        self.R = torch.nn.Parameter(R, requires_grad=False)
        self.k = k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x @ self.R
        return z[..., : self.k], z[..., self.k :]


class _SplitInverseFeaturizerModule(torch.nn.Module):
    """Inverse of :class:`_SplitFeaturizerModule`: concat features + error, unrotate."""

    def __init__(self, R: torch.Tensor) -> None:
        super().__init__()
        self.R = torch.nn.Parameter(R, requires_grad=False)

    def forward(self, features: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        return torch.cat([features, error], dim=-1) @ self.R.T


class TestFeatureSpaceReplaceSteerHookOracle:
    """Replace / steer routed through a featurizer's inverse vs. a hook oracle."""

    pytestmark = pytest.mark.unit

    def test_replace_in_rotated_basis(self, oracle_pipeline: LMPipeline) -> None:
        """``replace`` overwrites the featurized activation with the replacement
        vector and reconstructs via the inverse. With a full rotation the oracle
        is ``replacement @ Rᵀ`` — different from writing the replacement into raw
        dims, so this pins that the replace runs in feature space."""
        hidden = oracle_pipeline.model.config.hidden_size
        R = random_rotation(hidden, seed=2)
        replacement = torch.linspace(-1.0, 1.0, hidden)
        target = _target(oracle_pipeline, rotate_featurizer(R))

        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        patch = (replacement @ R.T).reshape(1, 1, hidden)
        manual = next_token_logits(oracle_pipeline, base_inputs, _LAYER, [_POS], patch)
        clean = next_token_logits(oracle_pipeline, base_inputs)

        causalab = _run_steer(oracle_pipeline, target, replacement, mode="replace")
        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_additive_steer_in_rotated_basis(self, oracle_pipeline: LMPipeline) -> None:
        """``add`` adds the vector to the featurized base and reconstructs. With a
        rotation the oracle adds ``vector @ Rᵀ`` to the base residual."""
        hidden = oracle_pipeline.model.config.hidden_size
        R = random_rotation(hidden, seed=3)
        vector = torch.linspace(0.2, 0.8, hidden)
        target = _target(oracle_pipeline, rotate_featurizer(R))

        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        fb = capture_residual(oracle_pipeline, _LAYER, base_inputs)[:, _POS, :]
        patch = (fb + vector @ R.T).unsqueeze(1)
        manual = next_token_logits(oracle_pipeline, base_inputs, _LAYER, [_POS], patch)
        clean = next_token_logits(oracle_pipeline, base_inputs)

        causalab = _run_steer(oracle_pipeline, target, vector, mode="add")
        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_replace_preserves_orthogonal_error(
        self, oracle_pipeline: LMPipeline
    ) -> None:
        """With a lossy ``k``-of-``d`` subspace featurizer, ``replace`` overwrites
        only the in-subspace coords; the orthogonal ``d-k`` coords are rebuilt
        from the *base* (the error term). The oracle replaces just the first ``k``
        rotated coords and keeps the rest — proving the error term is threaded
        through, not dropped (which would zero/garble the orthogonal component)."""
        hidden = oracle_pipeline.model.config.hidden_size
        k = hidden // 2
        R = random_rotation(hidden, seed=4)
        featurizer = Featurizer(
            featurizer=_SplitFeaturizerModule(R, k),
            inverse_featurizer=_SplitInverseFeaturizerModule(R),
            n_features=k,
        )
        target = _target(oracle_pipeline, featurizer)
        replacement = torch.linspace(-2.0, 2.0, k)

        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        base_z = capture_residual(oracle_pipeline, _LAYER, base_inputs)[:, _POS, :] @ R
        patch_z = base_z.clone()
        patch_z[:, :k] = replacement  # in-subspace replaced, remainder kept (error)
        patch = (patch_z @ R.T).unsqueeze(1)
        manual = next_token_logits(oracle_pipeline, base_inputs, _LAYER, [_POS], patch)
        clean = next_token_logits(oracle_pipeline, base_inputs)

        causalab = _run_steer(oracle_pipeline, target, replacement, mode="replace")
        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)
