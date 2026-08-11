"""``run_interchange_interventions`` — the public interchange wrapper on the
nnsight batched-execution engine (same-model AND cross-model since SH2 #411).

The pyvene-internal tests that used to live here (prepare_intervenable_inputs,
batched_interchange_intervention, the hook-oracle capability matrix) were
deleted with the pyvene backbone at the SH2 cutover; the engine equivalents
are pinned by ``tests/neural/test_dataset.py`` (raw-hook oracle at scale),
``tests/neural/test_site.py`` (per-component read/write parity) and
``tests/neural/parity/`` (captured per-mode goldens).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.activations.interchange_mode import (
    run_interchange_interventions,
)
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.featurizer import Featurizer
from causalab.neural.pipeline import GenerationResult, LMPipeline
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec
from causalab.neural.token_positions import TokenPosition

# The pyvene-independent hook oracles and shared featurizer/trace builders now
# live in hook_oracle.py so every pyvene-coverage test file shares one ground
# truth (GH #380). Imported under their original private names to keep the call
# sites in this file unchanged.
from tests.neural.activations.hook_oracle import (
    make_trace as _make_trace,
)


# --------------------------------------------------------------------------- #
#  Local helpers — small, file-local, lifted to tests/_helpers/ only if 3+    #
#  files want the same.                                                       #
# --------------------------------------------------------------------------- #
def _make_spec(
    pipeline: LMPipeline,
    *,
    layer: int = 0,
    target_output: bool = False,
) -> SiteSpec:
    """Build a single-token residual-stream :class:`SiteSpec` anchored at the
    first token. Single position keeps interventions trivially scoped on the
    tiny 16-dim residual stream."""
    hidden = pipeline.model.config.hidden_size
    tp = TokenPosition(lambda _x: [0], pipeline, id="first_token")
    component = "block_output" if target_output else "block_input"
    return SiteSpec(
        fsite=FeaturizedSite(Site(component, layer)),
        positions=tp,
        key=f"resid_L{layer}_first_token",
        width=hidden,
    )


def _two_examples() -> list[dict[str, Any]]:
    """Two-example fixture with a single counterfactual each.

    Two examples (rather than one) is the minimum that lets us exercise the
    batched-shape contracts on ``prepare_intervenable_inputs`` — single-row
    batches would let an off-by-one bug pass silently.
    """
    return [
        {
            "input": _make_trace("hello world"),
            "counterfactual_inputs": [_make_trace("foo bar")],
        },
        {
            "input": _make_trace("blue sky"),
            "counterfactual_inputs": [_make_trace("green field")],
        },
    ]


def _long_examples() -> list[dict[str, Any]]:
    """Two long-prompt examples so positions 0,1,2 stay in-bounds on both sides.

    Used by the multi-token span tests, where the unit selects more than one
    token and we need the shape/featurizer guards — not the #176 bounds check —
    to be the only thing that can fire.
    """
    return [
        {
            "input": _make_trace("the quick brown fox"),
            "counterfactual_inputs": [_make_trace("a slow lazy old dog")],
        },
        {
            "input": _make_trace("one two three four"),
            "counterfactual_inputs": [_make_trace("five six seven eight")],
        },
    ]


def _manifold_featurizer(d: int) -> Featurizer:
    """A *rank-sensitive* (non-broadcast) featurizer: a spline manifold whose
    ``encode`` hard-codes a 2-D row layout (``argmin(dim=1)``, ``z[:, :k]``).

    Unlike ``_diag_featurizer`` / ``_RotateFeaturizerModule`` — which act on the
    last dim and broadcast over any leading axes, so they pass a ``(b, 1, d)``
    span through transparently — this featurizer is sensitive to the rank of its
    input. It is the regression surface for ``keep_last_dim=True``: before the
    leading-dim shim it raised ``IndexError`` when handed ``(b, num_pos, d)``
    (even single-token ``(b, 1, d)``). A 1-D intrinsic curve in ``d``-dim ambient
    space mirrors the path-steering setup (``intrinsic_dim=1``)."""
    from causalab.methods.spline.builders import build_spline_manifold
    from causalab.methods.spline.featurizer import ManifoldFeaturizer

    control_points = torch.linspace(0.0, 1.0, 5).unsqueeze(1)  # (5, 1)
    t = torch.linspace(0.0, 1.0, 5)
    # A non-degenerate ambient curve; only the first 4 dims vary, the rest are 0.
    cols = [t, t**2, torch.sin(3 * t), torch.cos(3 * t)]
    centroids = torch.zeros(5, d)
    for i, col in enumerate(cols[: min(4, d)]):
        centroids[:, i] = col
    manifold = build_spline_manifold(
        control_points, centroids, intrinsic_dim=1, ambient_dim=d
    )
    return ManifoldFeaturizer(manifold, n_features=d)


# --------------------------------------------------------------------------- #
#  run_interchange_interventions                                              #
# --------------------------------------------------------------------------- #
class TestRunInterchangeInterventionsUnit:
    """Top-level loop: batches the dataset through the engine, optionally
    compresses scores to top-k, and returns ONE flat
    :class:`GenerationResult` over the whole dataset (EU5b #487 — the
    internal batch split never appears in the shape; the legacy
    ``to_raw_results()`` tail is gone from the wrapper)."""

    pytestmark = pytest.mark.unit

    def test_basic_run_returns_flat_result(self, tiny_pipeline: LMPipeline) -> None:
        groups = [[_make_spec(tiny_pipeline)]]
        dataset = [
            {
                "input": _make_trace(text),
                "counterfactual_inputs": [_make_trace(cf)],
            }
            for text, cf in [
                ("alpha", "beta"),
                ("gamma", "delta"),
                ("epsilon", "zeta"),
            ]
        ]

        result = run_interchange_interventions(
            tiny_pipeline,
            dataset,
            groups,
            batch_size=2,
            output_scores=False,
        )

        # ONE flat result carrying all 3 examples (the internal
        # ceil(3/2)=2 batch split is an execution detail).
        assert isinstance(result, GenerationResult)
        assert result.sequences.shape == (3, tiny_pipeline.max_new_tokens)
        assert len(result.strings) == 3

    def test_output_scores_false_gives_none_scores(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        groups = [[_make_spec(tiny_pipeline)]]
        dataset = [
            {
                "input": _make_trace("a"),
                "counterfactual_inputs": [_make_trace("b")],
            },
        ]

        result = run_interchange_interventions(
            tiny_pipeline,
            dataset,
            groups,
            batch_size=1,
            output_scores=False,
        )
        assert result.scores is None
        assert result.scores_top_k is None

    def test_top_k_scores_uses_int_argument(self, tiny_pipeline: LMPipeline) -> None:
        """``output_scores=int`` compresses to per-step top-k structures
        (memory-efficient): ``scores_top_k`` is set over the WHOLE dataset,
        ``scores`` is dropped (the two are exclusive)."""
        groups = [[_make_spec(tiny_pipeline)]]
        dataset = [
            {
                "input": _make_trace("a"),
                "counterfactual_inputs": [_make_trace("b")],
            },
            {
                "input": _make_trace("c"),
                "counterfactual_inputs": [_make_trace("d")],
            },
        ]

        result = run_interchange_interventions(
            tiny_pipeline,
            dataset,
            groups,
            batch_size=2,
            output_scores=5,
        )
        assert result.scores is None
        assert result.scores_top_k is not None
        assert result.sequences.shape[0] == 2
        # One structure per generated step, each spanning all examples.
        for step in result.scores_top_k:
            assert step["top_k_logits"].shape == (2, 5)
            assert step["top_k_indices"].shape == (2, 5)
            assert len(step["top_k_tokens"]) == 2

    def test_small_batch_size_never_leaks_into_shape(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """batch_size=1 over 3 examples still returns ONE flat result of 3."""
        groups = [[_make_spec(tiny_pipeline)]]
        dataset = [
            {
                "input": _make_trace(t),
                "counterfactual_inputs": [_make_trace(c)],
            }
            for t, c in [("p", "q"), ("r", "s"), ("t", "u")]
        ]

        result = run_interchange_interventions(
            tiny_pipeline,
            dataset,
            groups,
            batch_size=1,
            output_scores=False,
        )
        assert result.sequences.shape[0] == 3
        assert len(result.strings) == 3

    def test_gen_kwargs_reach_the_generate_spec(
        self, tiny_pipeline: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``gen_kwargs`` threads through the wrapper into the engine's
        :class:`~causalab.neural.plan.GenerateSpec` (review #492 F3):
        ``min_new_tokens=max_new_tokens`` — the escape hatch the engine's
        ragged-scores refusal names — must be reachable from the public
        wrappers, not only from ``run_intervened_generation`` directly.
        Spy on the plan handed to ``run_plan`` to pin the forwarded kwarg,
        then confirm the run behaves accordingly (scores span the full
        budget — early EOS suppressed)."""
        import causalab.neural.dataset as dataset_mod

        seen: list[Any] = []
        real_run_plan = dataset_mod.run_plan

        def spy(model, plan, **kw):
            seen.append(plan)
            return real_run_plan(model, plan, **kw)

        monkeypatch.setattr(dataset_mod, "run_plan", spy)

        groups = [[_make_spec(tiny_pipeline)]]
        dataset = [
            {
                "input": _make_trace("a"),
                "counterfactual_inputs": [_make_trace("b")],
            },
        ]
        budget = tiny_pipeline.max_new_tokens
        result = run_interchange_interventions(
            tiny_pipeline,
            dataset,
            groups,
            batch_size=1,
            output_scores=True,
            gen_kwargs={"min_new_tokens": budget},
        )
        assert seen, "the engine's generation plan was never built"
        spec = seen[0].generate
        assert spec is not None
        assert spec.kwargs["min_new_tokens"] == budget
        assert result.scores is not None
        assert len(result.scores) == budget

    def test_empty_dataset_returns_empty_result(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """An empty dataset yields the uniform empty shape — a zero-row
        sequences tensor at the pipeline's budget width and an empty strings
        list (EU5a #486 dropped the legacy ``{"sequences": [[]], "string":
        [[]]}`` sentinel special-case; EU5b #487 exposes the flat result).

        With ``output_scores`` truthy, ``scores`` is ``[]`` (zero generated
        steps) rather than ``None``. Pinned deliberately: the zero-row
        sequences tensor keeps ``io.artifacts`` concat-able through
        ``to_raw_results()`` where the legacy ``[[]]`` sequences entry would
        have crashed ``torch.cat``."""
        groups = [[_make_spec(tiny_pipeline)]]

        result = run_interchange_interventions(
            tiny_pipeline,
            [],
            groups,
            batch_size=2,
            output_scores=False,
        )
        assert result.sequences.shape == (0, tiny_pipeline.max_new_tokens)
        assert result.strings == []
        assert result.scores is None

        with_scores = run_interchange_interventions(
            tiny_pipeline,
            [],
            groups,
            batch_size=2,
            output_scores=True,
        )
        assert with_scores.scores == []  # zero generated steps
        assert with_scores.strings == []
        assert with_scores.sequences.shape == (0, tiny_pipeline.max_new_tokens)


class TestRunInterchangeInterventionsProperty:
    """Aggregation invariants over the flat result."""

    pytestmark = pytest.mark.property

    def test_output_shape_is_batch_size_invariant(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """The flat result makes ``batch_size`` a pure execution knob — every
        split returns ONE flat result of all 5 examples, with identical
        sequences and strings."""
        groups = [[_make_spec(tiny_pipeline)]]
        dataset = [
            {
                "input": _make_trace(f"in_{i}"),
                "counterfactual_inputs": [_make_trace(f"cf_{i}")],
            }
            for i in range(5)
        ]

        reference = None
        for batch_size in (1, 2, 3, 5):
            result = run_interchange_interventions(
                tiny_pipeline,
                dataset,
                groups,
                batch_size=batch_size,
                output_scores=False,
            )
            assert result.sequences.shape[0] == 5, f"batch_size={batch_size}"
            assert len(result.strings) == 5
            if reference is None:
                reference = result
            else:
                assert torch.equal(result.sequences, reference.sequences), (
                    f"batch_size={batch_size}: sequences changed with the split"
                )
                assert result.strings == reference.strings

    def test_scores_span_all_examples_per_step(self, tiny_pipeline: LMPipeline) -> None:
        """``output_scores=True`` gives per-step ``(n_examples, vocab)``
        tensors spanning the whole dataset — the internal batch split never
        fragments the score steps."""
        groups = [[_make_spec(tiny_pipeline)]]
        dataset = [
            {
                "input": _make_trace(t),
                "counterfactual_inputs": [_make_trace(c)],
            }
            for t, c in [("aa", "bb"), ("cc", "dd"), ("ee", "ff")]
        ]
        result = run_interchange_interventions(
            tiny_pipeline,
            dataset,
            groups,
            batch_size=2,
            output_scores=True,
        )
        assert result.scores is not None
        vocab = tiny_pipeline.model.config.vocab_size
        for step in result.scores:
            assert step.shape == (3, vocab)

    def test_output_scores_false_excludes_scores(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """The contract: ``output_scores=False`` produces ``scores=None``."""
        groups = [[_make_spec(tiny_pipeline)]]
        dataset = [
            {
                "input": _make_trace("a"),
                "counterfactual_inputs": [_make_trace("b")],
            },
        ]
        result = run_interchange_interventions(
            tiny_pipeline,
            dataset,
            groups,
            batch_size=1,
            output_scores=False,
        )
        assert result.scores is None
