"""Hook oracle for collect ordering + featurized/head collection (GH #380).

``collect_features`` reads activations at a list of sites in one forward and
returns ``{spec.key: (n_samples, n_features)}``. Two engine contracts underpin
it:

* **Routing.** Each key's tensor must hold *its own* site's activation. We
  collect three specs at *distinct* sites/positions and check each against an
  independent hook capture at that site — a misrouting makes at least two
  mismatch.
* **Featurizer / component routing.** Collection runs through the spec's
  featurizer (``f_base`` is what comes back) and addresses per-head value
  outputs the same way an intervention does.

All ground truths are hand-rolled forward hooks — no backbone involved. See
``docs/PYVENE_HOOK_COVERAGE.md``.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.head_view import HeadSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec
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
        """Three specs at distinct (component, layer, position) sites. Each key's
        collected activation must equal the hook capture at *its own* site — a
        misrouting between keys and sites makes at least two mismatch."""
        hidden = oracle_pipeline.model.config.hidden_size
        s0 = SiteSpec(
            fsite=FeaturizedSite(Site("block_input", 0)),
            positions=_tp(oracle_pipeline, 0),
            key="resid.L0.in.pos0",
            width=hidden,
        )
        s1 = SiteSpec(
            fsite=FeaturizedSite(Site("block_output", 0)),
            positions=_tp(oracle_pipeline, 1),
            key="resid.L0.out.pos1",
            width=hidden,
        )
        s2 = SiteSpec(
            fsite=FeaturizedSite(Site("block_output", 1)),
            positions=_tp(oracle_pipeline, 2),
            key="resid.L1.out.pos2",
            width=hidden,
        )
        dataset = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
        collected = collect_features(
            dataset, oracle_pipeline, [s0, s1, s2], batch_size=1
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

        torch.testing.assert_close(collected[s0.key], cap0, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(collected[s1.key], cap1, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(collected[s2.key], cap2, atol=1e-5, rtol=1e-4)

    def test_collect_runs_through_featurizer(self, oracle_pipeline: LMPipeline) -> None:
        """Collection returns the *featurized* activation: with a rotation
        featurizer the collected value is the captured residual rotated into the
        featurizer basis."""
        hidden = oracle_pipeline.model.config.hidden_size
        R = random_rotation(hidden, seed=5)
        site = SiteSpec(
            fsite=FeaturizedSite(Site("block_output", 0), rotate_featurizer(R)),
            positions=_tp(oracle_pipeline, 1),
            key="resid.L0.rotated.pos1",
            width=hidden,
        )
        dataset = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
        collected = collect_features(dataset, oracle_pipeline, [site], batch_size=1)

        inputs = oracle_pipeline.load([make_trace(_BASE)])
        expected = capture_residual(oracle_pipeline, 0, inputs)[:, 1, :] @ R
        torch.testing.assert_close(collected[site.key], expected, atol=1e-5, rtol=1e-4)

    def test_collect_head_value_matches_oproj_slice(
        self, oracle_pipeline: LMPipeline
    ) -> None:
        """A per-head value-output site collects the head's slice of o_proj's
        input — the per-head gather path, captured by hand via a pre-hook."""
        head = 1
        site = SiteSpec(
            fsite=FeaturizedSite(HeadSite(kind="attention_value", layer=0, head=head)),
            positions=_tp(oracle_pipeline, 1),
            key="attention_head.L0.H1.pos1",
        )
        dataset = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
        collected = collect_features(dataset, oracle_pipeline, [site], batch_size=1)

        inputs = oracle_pipeline.load([make_trace(_BASE)])
        expected = capture_head_value(oracle_pipeline, 0, head, inputs)[:, 1, :]
        torch.testing.assert_close(collected[site.key], expected, atol=1e-5, rtol=1e-4)
