"""Hook oracle for collect ordering + featurized/head collection (GH #380).

``collect_features`` reads activations at a list of units in one forward and
returns ``{unit.id: (n_samples, n_features)}``. Two contracts underpin it:

* **Ordering.** the engine returns one tensor per unit in *config-add* order
  (``sorted_keys``); ``collect.py`` maps ``activations[i] → model_units[i]`` by
  position. If that order were wrong, each unit's dict entry would hold another
  unit's activation. We collect three units at *distinct* sites/positions and
  check each against an independent hook capture at that site — a misorder makes
  at least two mismatch.
* **Featurizer / component routing.** Collection runs through the unit's
  featurizer (``f_base`` is what comes back) and addresses per-head value
  outputs the same way an intervention does.

All ground truths are hand-rolled forward hooks — no engine. See
``docs/HOOK_ORACLES.md``.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.LM_units import AttentionHead, ResidualStream
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition
from causalab.neural.activations.collect import collect_features

from tests.neural.activations.hook_oracle import (
    capture_component,
    capture_head_value,
    capture_residual,
    decoder_block,
    make_trace,
    random_rotation,
    rotate_featurizer,
)

_BASE = "the quick brown fox jumps"


def _tp(pipeline: LMPipeline, pos: int) -> TokenPosition:
    return TokenPosition(lambda _x, _p=pos: [_p], pipeline, id=f"pos{pos}")


class TestCollectHookOracle:
    pytestmark = pytest.mark.unit

    def test_collect_order_matches_per_site_capture(
        self, oracle_pipeline: LMPipeline
    ) -> None:
        """Three units at distinct (component, layer, position) sites. Each unit's
        collected activation must equal the hook capture at *its own* site — which
        only holds if activations come back in config-add order."""
        hidden = oracle_pipeline.model.config.hidden_size
        u0 = ResidualStream(
            layer=0,
            token_indices=_tp(oracle_pipeline, 0),
            target_output=False,
            shape=(hidden,),
        )
        u1 = ResidualStream(
            layer=0,
            token_indices=_tp(oracle_pipeline, 1),
            target_output=True,
            shape=(hidden,),
        )
        u2 = ResidualStream(
            layer=1,
            token_indices=_tp(oracle_pipeline, 2),
            target_output=True,
            shape=(hidden,),
        )
        dataset = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
        collected = collect_features(
            dataset, oracle_pipeline, [u0, u1, u2], batch_size=1
        )

        inputs = oracle_pipeline.load([make_trace(_BASE)])
        cap0 = capture_component(
            oracle_pipeline, decoder_block(oracle_pipeline, 0), "in", inputs
        )[:, 0, :]
        cap1 = capture_residual(oracle_pipeline, 0, inputs)[:, 1, :]
        cap2 = capture_residual(oracle_pipeline, 1, inputs)[:, 2, :]

        # Distinct sites give distinct activations — a sanity check that a misorder
        # would actually be caught (the values aren't accidentally equal).
        assert not torch.allclose(cap0, cap1, atol=1e-4)
        assert not torch.allclose(cap1, cap2, atol=1e-4)

        torch.testing.assert_close(collected[u0.id], cap0, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(collected[u1.id], cap1, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(collected[u2.id], cap2, atol=1e-5, rtol=1e-4)

    def test_collect_runs_through_featurizer(self, oracle_pipeline: LMPipeline) -> None:
        """Collection returns the *featurized* activation: with a rotation
        featurizer the collected value is the captured residual rotated into the
        featurizer basis."""
        hidden = oracle_pipeline.model.config.hidden_size
        R = random_rotation(hidden, seed=5)
        unit = ResidualStream(
            layer=0,
            token_indices=_tp(oracle_pipeline, 1),
            target_output=True,
            shape=(hidden,),
            featurizer=rotate_featurizer(R),
        )
        dataset = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
        collected = collect_features(dataset, oracle_pipeline, [unit], batch_size=1)

        inputs = oracle_pipeline.load([make_trace(_BASE)])
        expected = capture_residual(oracle_pipeline, 0, inputs)[:, 1, :] @ R
        torch.testing.assert_close(collected[unit.id], expected, atol=1e-5, rtol=1e-4)

    def test_collect_head_value_matches_oproj_slice(
        self, oracle_pipeline: LMPipeline
    ) -> None:
        """A per-head value-output unit collects the head's slice of o_proj's
        input — the 4-D ``h.pos`` gather path, captured by hand via a pre-hook."""
        head = 1
        unit = AttentionHead(
            layer=0,
            head=head,
            token_indices=_tp(oracle_pipeline, 1),
            target_output=True,
        )
        dataset = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
        collected = collect_features(dataset, oracle_pipeline, [unit], batch_size=1)

        inputs = oracle_pipeline.load([make_trace(_BASE)])
        expected = capture_head_value(oracle_pipeline, 0, head, inputs)[:, 1, :]
        torch.testing.assert_close(collected[unit.id], expected, atol=1e-5, rtol=1e-4)
