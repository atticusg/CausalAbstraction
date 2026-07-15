"""Behavioural contract for the seeded-noise intervention (GH #380).

The noise intervention is ROME-style causal-tracing corruption: it perturbs a
featurized activation toward seeded Gaussian noise scaled by a per-call
magnitude (``noised = f_base + scale·N(0,1)``), reconstructing through the
inverse featurizer. The randomness is drawn *inside* the intervention from a
per-instance generator seeded at construction.

Unlike interchange/replace, the noise draw order is an internal of the backbone
(pyvene today, nnsight tomorrow), so a byte-for-byte hook oracle would couple the
test to that internal. Instead this pins the *backbone-agnostic* contract that
any backbone must honour:

* **scale 0 is a no-op** — equals the clean (un-intervened) logits exactly;
* **non-trivial** — a positive scale moves the logits off clean;
* **reproducible** — same seed (the model is rebuilt per run, re-seeding the
  generator) gives byte-identical logits across runs;
* **seed-sensitive** — a different seed gives different logits.

See ``docs/PYVENE_HOOK_COVERAGE.md`` for the coverage map.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.LM_units import ResidualStream
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget
from causalab.methods.steer.steer import make_zero_features, run_steering_interventions

from tests.neural.activations.hook_oracle import make_trace, next_token_logits

_LAYER = 0
_POS = 1
_BASE = "the quick brown fox jumps"


def _noise_target(pipeline: LMPipeline) -> InterchangeTarget:
    hidden = pipeline.model.config.hidden_size
    tp = TokenPosition(lambda _x: [_POS], pipeline, id="pos1")
    unit = ResidualStream(
        layer=_LAYER, token_indices=tp, target_output=True, shape=(hidden,)
    )
    return InterchangeTarget([[unit]])


def _run_noise(
    pipeline: LMPipeline, target: InterchangeTarget, *, scale: float, seed: int
) -> torch.Tensor:
    """Run a pure-noise intervention; return the example's next-token logits.

    ``steering_vectors`` carries the per-unit noise *scale*; ``type_by_unit``
    flips the single unit to the ``noise`` intervention type."""
    unit_id = target.flatten()[0].id
    zeros = make_zero_features(target)
    scale_vec = {unit_id: torch.full_like(zeros[unit_id], float(scale))}
    result = run_steering_interventions(
        pipeline,
        [{"input": make_trace(_BASE), "counterfactual_inputs": []}],
        target,
        scale_vec,
        mode="replace",  # overridden per-unit by type_by_unit
        type_by_unit={unit_id: "noise"},
        noise_seed=seed,
        output_scores=True,
    )
    # run_steering_interventions nests one batch deep: scores[batch][step].
    return result["scores"][0][0]


class TestNoiseInterventionContract:
    """Seeded-noise corruption at ``(_LAYER, _POS)`` — backbone-agnostic spec."""

    pytestmark = pytest.mark.property

    def test_zero_scale_is_no_op(self, oracle_pipeline: LMPipeline) -> None:
        """``scale=0`` adds ``0·noise`` — the activation is untouched, so the run
        must reproduce the clean logits exactly."""
        target = _noise_target(oracle_pipeline)
        clean = next_token_logits(
            oracle_pipeline, oracle_pipeline.load([make_trace(_BASE)])
        )
        noised = _run_noise(oracle_pipeline, target, scale=0.0, seed=0)
        torch.testing.assert_close(noised, clean, atol=1e-5, rtol=1e-4)

    def test_positive_scale_is_non_trivial(self, oracle_pipeline: LMPipeline) -> None:
        """A positive scale must actually corrupt the activation (logits move)."""
        target = _noise_target(oracle_pipeline)
        clean = next_token_logits(
            oracle_pipeline, oracle_pipeline.load([make_trace(_BASE)])
        )
        noised = _run_noise(oracle_pipeline, target, scale=3.0, seed=0)
        assert not torch.allclose(noised, clean, atol=1e-4)

    def test_same_seed_is_reproducible(self, oracle_pipeline: LMPipeline) -> None:
        """Same seed → byte-identical corruption across independent runs (each
        rebuilds the model, re-seeding the per-instance generator)."""
        target = _noise_target(oracle_pipeline)
        a = _run_noise(oracle_pipeline, target, scale=3.0, seed=7)
        b = _run_noise(oracle_pipeline, target, scale=3.0, seed=7)
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)

    def test_different_seed_differs(self, oracle_pipeline: LMPipeline) -> None:
        """A different seed draws different noise → different logits."""
        target = _noise_target(oracle_pipeline)
        a = _run_noise(oracle_pipeline, target, scale=3.0, seed=7)
        b = _run_noise(oracle_pipeline, target, scale=3.0, seed=8)
        assert not torch.allclose(a, b, atol=1e-4)
