"""End-to-end steering tests for attention-head and multi-position units.

These cover the Step-1 enabling change in ``causalab.methods.steer.steer`` that
lets ``run_steering_interventions`` (and therefore zero/mean ablation) target:

* attention-head (``h.pos``) units — the source must be the 4-D ``(b, h, s, d)``
  tensor pyvene gathers, not the 3-D ``(b, 1, d)`` the legacy code always built;
* multi-position spans on ``pos`` units (e.g. ``get_all_tokens``) — the source
  must be ``(b, n_pos, d)``.

Both previously raised inside pyvene's ``do_intervention`` / ``bhsd_to_bs_hd``.
We exercise the real tiny-random Llama pipeline (per docs/TESTS.md, our own
numerical code is never mocked) and assert that zero-*replace* ablation actually
perturbs the output logits relative to the un-intervened forward pass — using
additive zero-steering (a mathematical no-op) as the un-intervened reference so
both paths share the exact same batching/generation machinery.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.steer.steer import make_zero_features, run_steering_interventions
from causalab.neural.activations.site_grids import (
    build_attention_head_sites,
    build_mlp_sites,
)
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec
from causalab.neural.token_positions import (
    TokenPosition,
    get_all_tokens,
    get_last_token_index,
)

# Shape contracts at the steering-primitive boundary plus the behavioral
# invariant that `replace` perturbs the logits while `add`-zero is a no-op — all
# property-tier (no pinned numerical values). docs/TESTS.md: methods/ → property.
pytestmark = pytest.mark.property


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _trace(text: str) -> CausalTrace:
    """A minimal ``raw_input``-only trace, the shape ablation actually consumes."""
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _dataset() -> list[dict[str, Any]]:
    """Two equal-token-length prompts.

    Equal length keeps the all-position pyvene gather rectangular without the
    length-bucketing that the ablation method (Step 2) layers on top — these
    tests isolate the steering primitive itself.
    """
    return [{"input": _trace("hello world")}, {"input": _trace("blue green")}]


def _run(pipeline: LMPipeline, sites: list[SiteSpec], mode: str) -> torch.Tensor:
    """Run zero-vector steering in ``mode`` and return the first-step logits.

    ``mode="add"`` with zeros is a no-op (base + 0), giving the un-intervened
    reference; ``mode="replace"`` with zeros is the actual ablation.
    """
    zeros = make_zero_features(sites)
    out = run_steering_interventions(
        pipeline,
        _dataset(),
        sites,
        zeros,
        batch_size=2,
        mode=mode,  # type: ignore[arg-type]
        output_scores=True,
    )
    assert out.scores is not None  # output_scores=True above
    return out.scores[0]  # first generated step, (n_examples, vocab)


# --------------------------------------------------------------------------- #
#  Attention-head units (h.pos)                                               #
# --------------------------------------------------------------------------- #
class TestAttentionHeadSteering:
    """Replace-steering on an ``AttentionHead`` unit must run and change logits."""

    def test_single_position_head_ablation_changes_logits(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        last = TokenPosition(
            lambda inp: get_last_token_index(inp, mock_tiny_lm),
            mock_tiny_lm,
            id="last_token",
        )
        sites = [
            s
            for g in build_attention_head_sites(mock_tiny_lm, [0], [0], last)[(0, 0)]
            for s in g
        ]

        base = _run(mock_tiny_lm, sites, mode="add")
        ablated = _run(mock_tiny_lm, sites, mode="replace")

        assert base.shape == ablated.shape
        assert not torch.allclose(base, ablated)

    def test_all_tokens_head_ablation_changes_logits(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        span = get_all_tokens(_dataset()[0]["input"], mock_tiny_lm)
        sites = [
            s
            for g in build_attention_head_sites(mock_tiny_lm, [0], [0], span)[(0, 0)]
            for s in g
        ]

        base = _run(mock_tiny_lm, sites, mode="add")
        ablated = _run(mock_tiny_lm, sites, mode="replace")

        assert not torch.allclose(base, ablated)


# --------------------------------------------------------------------------- #
#  Multi-position pos units (MLP / residual over all tokens)                  #
# --------------------------------------------------------------------------- #
class TestMultiPositionSteering:
    """Replace-steering across a multi-position span must run and change logits."""

    def test_mlp_all_tokens_ablation_changes_logits(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        span = get_all_tokens(_dataset()[0]["input"], mock_tiny_lm)
        sites = [
            s
            for g in build_mlp_sites(mock_tiny_lm, [0], [span])[(0, "all_tokens")]
            for s in g
        ]

        base = _run(mock_tiny_lm, sites, mode="add")
        ablated = _run(mock_tiny_lm, sites, mode="replace")

        assert not torch.allclose(base, ablated)


# --------------------------------------------------------------------------- #
#  Regression: single-position pos steering still works                       #
# --------------------------------------------------------------------------- #
class TestSinglePositionRegression:
    """The legacy single-position ``pos`` path must keep working unchanged."""

    def test_single_position_residual_ablation_changes_logits(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        hidden = mock_tiny_lm.model.config.hidden_size
        last = TokenPosition(
            lambda inp: get_last_token_index(inp, mock_tiny_lm),
            mock_tiny_lm,
            id="last_token",
        )
        site = SiteSpec(
            fsite=FeaturizedSite(Site("block_output", 0)),
            positions=last,
            key="residual.L0.last_token",
            width=hidden,
        )

        base = _run(mock_tiny_lm, [site], mode="add")
        ablated = _run(mock_tiny_lm, [site], mode="replace")

        assert not torch.allclose(base, ablated)
