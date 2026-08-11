"""Hook oracle for the interpolation intervention (GH #380).

``run_interpolation_interventions`` patches an arbitrary function of the base and
source *featurized* activations into the model:
``new_act = inverse_featurizer(fn(f_base, f_src, **params), base_err)``. This file
pins that behaviour against a hand-rolled forward-hook ground truth — no pyvene —
so the contract survives the pyvene→nnsight swap. See
``tests/neural/activations/hook_oracle.py`` for the oracle primitives and
``docs/PYVENE_HOOK_COVERAGE.md`` for the full coverage map.

Linear interpolation ``(1-α)·f_base + α·f_src`` is the canonical case: at ``α=0``
it is the identity (clean run), at ``α=1`` it reduces to interchange (full swap),
and in between it is a convex blend. With the identity featurizer the feature
space *is* the activation space, so the oracle blends raw residual-stream
activations directly. A custom non-linear ``fn`` exercises the arbitrary-callable
path the same way.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec
from causalab.neural.token_positions import TokenPosition
from causalab.neural.activations.interpolate import run_interpolation_interventions

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


def _linear_interp(
    f_base: torch.Tensor, f_src: torch.Tensor, alpha: float
) -> torch.Tensor:
    return (1 - alpha) * f_base + alpha * f_src


def _single_pos_groups(pipeline: LMPipeline) -> list[list[SiteSpec]]:
    """A single-token residual-stream site at ``(_LAYER, _POS)``, identity
    featurizer — so feature space equals activation space."""
    hidden = pipeline.model.config.hidden_size
    tp = TokenPosition(lambda _x: [_POS], pipeline, id="pos1")
    site = SiteSpec(
        fsite=FeaturizedSite(Site("block_output", _LAYER)),
        positions=tp,
        key=f"residual.L{_LAYER}.pos1",
        width=hidden,
    )
    return [[site]]


def _run(pipeline: LMPipeline, fn, params: dict) -> torch.Tensor:
    """Run causalab's interpolation intervention on one example; return the
    first example's next-token logits ``(1, vocab)``."""
    out = run_interpolation_interventions(
        pipeline,
        [cf_example(_BASE, _SOURCE)],
        _single_pos_groups(pipeline),
        fn=fn,
        params=params,
        output_scores=True,
    )
    # Flat GenerationResult contract: first generated step -> (1, vocab).
    return out.scores[0]


class TestInterpolationHookOracle:
    """Interpolation at ``(_LAYER, _POS)`` vs. a forward-hook ground truth. The
    identity featurizer makes the feature-space blend a raw-activation blend, so
    the oracle patches the residual stream directly with the blended value."""

    pytestmark = pytest.mark.unit

    def test_alpha_zero_is_identity(self, oracle_pipeline: LMPipeline) -> None:
        """``α=0`` keeps the base features untouched — the run must reproduce the
        clean (un-intervened) logits exactly."""
        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        clean = next_token_logits(oracle_pipeline, base_inputs)
        causalab = self._run(oracle_pipeline, alpha=0.0)
        torch.testing.assert_close(causalab, clean, atol=1e-4, rtol=1e-3)

    def test_alpha_one_is_full_interchange(self, oracle_pipeline: LMPipeline) -> None:
        """``α=1`` discards the base features — equivalent to a full interchange
        swap of the source residual at the target position."""
        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        source_inputs = oracle_pipeline.load([make_trace(_SOURCE)])
        src_resid = capture_residual(oracle_pipeline, _LAYER, source_inputs)
        patch = src_resid[:, [_POS], :]
        manual = next_token_logits(oracle_pipeline, base_inputs, _LAYER, [_POS], patch)
        clean = next_token_logits(oracle_pipeline, base_inputs)

        causalab = self._run(oracle_pipeline, alpha=1.0)
        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_midpoint_is_convex_blend(self, oracle_pipeline: LMPipeline) -> None:
        """``0<α<1`` blends base and source residuals at the target position; the
        oracle patches that exact convex combination."""
        alpha = 0.5
        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        source_inputs = oracle_pipeline.load([make_trace(_SOURCE)])
        base_resid = capture_residual(oracle_pipeline, _LAYER, base_inputs)
        src_resid = capture_residual(oracle_pipeline, _LAYER, source_inputs)
        blended = (1 - alpha) * base_resid[:, [_POS], :] + alpha * src_resid[
            :, [_POS], :
        ]
        manual = next_token_logits(
            oracle_pipeline, base_inputs, _LAYER, [_POS], blended
        )
        clean = next_token_logits(oracle_pipeline, base_inputs)

        causalab = self._run(oracle_pipeline, alpha=alpha)
        # A genuine midpoint: distinct from both endpoints.
        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_arbitrary_callable(self, oracle_pipeline: LMPipeline) -> None:
        """An arbitrary (non-linear) ``fn`` is applied in feature space. With the
        identity featurizer the oracle applies the same elementwise ``max`` to the
        raw residuals at the target position."""

        def feature_max(f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
            return torch.maximum(f_base, f_src)

        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        source_inputs = oracle_pipeline.load([make_trace(_SOURCE)])
        base_resid = capture_residual(oracle_pipeline, _LAYER, base_inputs)
        src_resid = capture_residual(oracle_pipeline, _LAYER, source_inputs)
        patched = torch.maximum(base_resid[:, [_POS], :], src_resid[:, [_POS], :])
        manual = next_token_logits(
            oracle_pipeline, base_inputs, _LAYER, [_POS], patched
        )

        causalab = _run(oracle_pipeline, feature_max, {})
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    # -- helper ----------------------------------------------------------- #
    def _run(self, pipeline: LMPipeline, *, alpha: float) -> torch.Tensor:
        return _run(pipeline, _linear_interp, {"alpha": alpha})
