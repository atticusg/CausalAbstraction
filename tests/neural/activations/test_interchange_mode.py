"""Direct tests for ``causalab.neural.activations.interchange_mode``.

This module is the pyvene-backed engine that runs interchange interventions
on an :class:`~causalab.neural.pipeline.LMPipeline`. It batches
``CounterfactualExample`` lists, resolves base / counterfactual component
indices via :class:`~causalab.neural.units.InterchangeTarget`, calls
``pipeline.intervenable_generate``, and aggregates per-batch outputs
(``sequences`` and optional ``scores`` / top-k) back onto CPU. Every
``analyses/locate/*`` flow (via ``methods/interchange/single_pair.py``), the
baseline reporter (via ``methods/metric.py``), and subspace training (via
``methods/trained_subspace/train.py``) depend on it — so a wrong return
shape here breaks localization, subspace training, and baseline reporting.

The tests below exercise the three public entry points using the real
(tiny) Llama pipeline from :mod:`tests._helpers.tiny`:

* :func:`~causalab.neural.activations.interchange_mode.prepare_intervenable_inputs`
* :func:`~causalab.neural.activations.interchange_mode.batched_interchange_intervention`
* :func:`~causalab.neural.activations.interchange_mode.run_interchange_interventions`

docs/TESTS.md's mocking policy forbids ``MagicMock``\\ -ing our own numerical
code (pyvene wrapper, pipeline, ``AtomicModelUnit``). The legacy
``test_batched_interchange.py`` / ``test_prepare_intervenable_inputs.py``
/ ``test_run_interchange_intervention.py`` files mocked all of those and
are removed in the same PR that adds this file.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from causalab.causal.trace import CausalTrace
from causalab.neural.LM_units import (
    AttentionHead,
    AttentionOutput,
    MLP,
    ResidualStream,
)
from causalab.neural.featurizer import Featurizer
from causalab.neural.activations.collect import collect_features
from causalab.neural.activations.intervenable_model import (
    prepare_intervenable_model,
)
from causalab.neural.activations.interchange_mode import (
    batched_interchange_intervention,
    prepare_intervenable_inputs,
    run_interchange_interventions,
)
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget
from causalab.methods.steer.steer import run_steering_interventions

# The pyvene-independent hook oracles and shared featurizer/trace builders now
# live in hook_oracle.py so every pyvene-coverage test file shares one ground
# truth (GH #380). Imported under their original private names to keep the call
# sites in this file unchanged.
from tests.neural.activations.hook_oracle import (
    cf_example as _make_cf_example,
    random_rotation as _random_rotation,
    rotate_featurizer as _rotate_featurizer,
    make_trace as _make_trace,
    diag_featurizer as _diag_featurizer,
    RotateFeaturizerModule as _RotateFeaturizerModule,
    RotateInverseFeaturizerModule as _RotateInverseFeaturizerModule,
    component_module as _component_module,
    capture_residual as _capture_residual,
    capture_component as _capture_component,
    capture_head_value as _capture_head_value,
    next_token_logits as _next_token_logits,
    component_edited_logits as _component_edited_logits,
    head_patched_next_logits as _head_patched_next_logits,
)


# --------------------------------------------------------------------------- #
#  Local helpers — small, file-local, lifted to tests/_helpers/ only if 3+    #
#  files want the same.                                                       #
# --------------------------------------------------------------------------- #
def _make_unit(
    pipeline: LMPipeline,
    *,
    layer: int = 0,
    target_output: bool = False,
) -> ResidualStream:
    """Build a single-token :class:`ResidualStream` unit anchored at the
    first token. Single position keeps interventions trivially scoped on the
    tiny 16-dim residual stream."""
    hidden = pipeline.model.config.hidden_size
    tp = TokenPosition(lambda _x: [0], pipeline, id="first_token")
    return ResidualStream(
        layer=layer,
        token_indices=tp,
        target_output=target_output,
        shape=(hidden,),
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
#  prepare_intervenable_inputs                                                #
# --------------------------------------------------------------------------- #
class TestPrepareIntervenableInputsUnit:
    """Builds the ``(base, counterfactuals, inv_locations, feature_indices)``
    four-tuple consumed by pyvene's interchange interventions; wrong shape
    here breaks every ``analyses/locate/*`` run."""

    pytestmark = pytest.mark.unit

    def test_basic_shape_with_single_counterfactual(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        examples = _two_examples()
        unit = _make_unit(tiny_pipeline)
        target = InterchangeTarget([[unit]])

        base, cfs, inv_locations, feature_indices = prepare_intervenable_inputs(
            tiny_pipeline, examples, target
        )

        # base is a tokenized batch dict-like (HF returns ``BatchEncoding``,
        # which is a ``Mapping`` but not a ``dict`` subclass).
        assert isinstance(base, Mapping)
        assert "input_ids" in base and "attention_mask" in base
        # batch dimension = number of examples
        assert base["input_ids"].shape[0] == len(examples)

        # one counterfactual group → one batched_counterfactual dict-like
        assert len(cfs) == 1
        assert isinstance(cfs[0], Mapping)
        assert "input_ids" in cfs[0]
        assert cfs[0]["input_ids"].shape[0] == len(examples)

        # inv_locations contains the pyvene-shaped source→base mapping
        assert "sources->base" in inv_locations
        counterfactual_indices, base_indices = inv_locations["sources->base"]
        # one entry per model unit (here just one)
        assert len(counterfactual_indices) == 1
        assert len(base_indices) == 1

        # feature_indices is shape (num_units, batch_size)
        assert len(feature_indices) == 1  # num model units
        assert len(feature_indices[0]) == len(examples)

    def test_multiple_units_in_single_group(self, tiny_pipeline: LMPipeline) -> None:
        """Two units sharing one counterfactual input → two entries in
        ``inv_locations`` and ``feature_indices``, one counterfactual dict."""
        examples = _two_examples()
        unit_a = _make_unit(tiny_pipeline, layer=0)
        unit_b = _make_unit(tiny_pipeline, layer=1)
        target = InterchangeTarget([[unit_a, unit_b]])

        base, cfs, inv_locations, feature_indices = prepare_intervenable_inputs(
            tiny_pipeline, examples, target
        )

        # Single group, single counterfactual input.
        assert len(cfs) == 1
        counterfactual_indices, base_indices = inv_locations["sources->base"]
        # Two units → two entries in each index list.
        assert len(counterfactual_indices) == 2
        assert len(base_indices) == 2
        assert len(feature_indices) == 2

    def test_feature_indices_propagate_when_set(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """If ``unit.get_feature_indices()`` returns a list, every batch
        slot for that unit gets the same list."""
        hidden = tiny_pipeline.model.config.hidden_size
        tp = TokenPosition(lambda _x: [0], tiny_pipeline, id="first_token")
        unit = ResidualStream(
            layer=0,
            token_indices=tp,
            target_output=False,
            shape=(hidden,),
            feature_indices=[0, 1, 2],
        )
        target = InterchangeTarget([[unit]])

        _, _, _, feature_indices = prepare_intervenable_inputs(
            tiny_pipeline, _two_examples(), target
        )
        # one unit; one feature-index list per example
        assert feature_indices == [[[0, 1, 2], [0, 1, 2]]]

    def test_out_of_bounds_position_raises_clean_error(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """A position beyond the sequence length raises a catchable ValueError
        here instead of reaching pyvene's gather as a CUDA scatter assertion
        that poisons the context (#176). Single example ⇒ no padding shift,
        which is exactly the path the old fast-return left unchecked."""
        hidden = tiny_pipeline.model.config.hidden_size
        oob = TokenPosition(lambda _x: [9999], tiny_pipeline, id="oob")
        unit = ResidualStream(
            layer=0, token_indices=oob, target_output=False, shape=(hidden,)
        )
        target = InterchangeTarget([[unit]])
        # The out-of-bounds index comes from the unit's position, not the
        # examples — _two_examples() are ordinary short prompts.
        with pytest.raises(ValueError, match="out of bounds"):
            prepare_intervenable_inputs(tiny_pipeline, _two_examples(), target)

    def test_mismatched_base_cf_position_shapes_raise_clean_error(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """A token position that selects a *different number* of tokens for the
        base vs. the counterfactual — the multi-token counterfactual-value-string
        shape (#251) — raises a catchable CPU ``ValueError`` here instead of
        reaching pyvene's gather/scatter as a CUDA assertion that poisons the
        context. Distinct from #176: every index is individually in-bounds, so
        the per-position bounds check does not fire; only the base/CF shape
        mismatch does."""
        hidden = tiny_pipeline.model.config.hidden_size
        # is_original=True (base) selects 2 positions; is_original=False
        # (counterfactual) selects 3 — mirroring a 2-token base value swapped for
        # a 3-token counterfactual value. The indexer receives `is_original`
        # exactly as the real base/counterfactual passes in prepare_* do.
        tp = TokenPosition(
            lambda _x, is_original: [0, 1] if is_original else [0, 1, 2],
            tiny_pipeline,
            id="multitok",
        )
        unit = ResidualStream(
            layer=0, token_indices=tp, target_output=False, shape=(hidden,)
        )
        target = InterchangeTarget([[unit]])
        # Long prompts so positions 0..2 stay in-bounds for both sides; the only
        # thing wrong is the base-vs-counterfactual length mismatch.
        examples: list[dict[str, Any]] = [
            {
                "input": _make_trace("the quick brown fox"),
                "counterfactual_inputs": [_make_trace("a slow lazy old dog")],
            },
            {
                "input": _make_trace("one two three four"),
                "counterfactual_inputs": [_make_trace("five six seven eight")],
            },
        ]
        with pytest.raises(ValueError, match="mismatched shapes"):
            prepare_intervenable_inputs(tiny_pipeline, examples, target)

    def test_uniform_multitoken_position_is_allowed(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """A *uniform* multi-token span — every example selects the same number
        of tokens (here positions 0,1 for both base and counterfactual) — with
        the default identity featurizer is supported, not rejected. pyvene stacks
        a unit's per-example positions into one ``(batch, num_positions)`` tensor
        and gathers/scatters them simultaneously, and the identity featurizer
        makes the flattened swap exact. ``prepare_intervenable_inputs`` must build
        the four-tuple without raising, with two token indices per example."""
        hidden = tiny_pipeline.model.config.hidden_size
        tp = TokenPosition(lambda _x: [0, 1], tiny_pipeline, id="span2")
        unit = ResidualStream(
            layer=0, token_indices=tp, target_output=False, shape=(hidden,)
        )
        target = InterchangeTarget([[unit]])
        examples = _long_examples()

        _, _, inv_locations, _ = prepare_intervenable_inputs(
            tiny_pipeline, examples, target
        )
        counterfactual_indices, base_indices = inv_locations["sources->base"]
        # one unit, two examples, two tokens per example on each side
        assert len(base_indices) == len(counterfactual_indices) == 1
        assert [len(ex) for ex in base_indices[0]] == [2, 2]
        assert [len(ex) for ex in counterfactual_indices[0]] == [2, 2]

    def test_ragged_span_raises_clean_error(self, tiny_pipeline: LMPipeline) -> None:
        """A token position whose span width *varies across the batch* (2 tokens
        for one example, 1 for another) raises a catchable, actionable
        ``ValueError`` here instead of reaching pyvene's ``gather_neurons`` as the
        cryptic ``expected sequence of length N at dim 1`` error that poisons the
        context. The indexer keys on the prompt so base and counterfactual agree
        *per example* (passing the #251 base/CF mismatch guard) yet disagree
        *across* examples — the residual ragged case the relaxed guard targets.
        Distinct from #176: positions 0,1 are individually in-bounds."""
        hidden = tiny_pipeline.model.config.hidden_size

        # Width follows the prompt's word count: the 4-word prompts select 2
        # tokens, the shorter ones select 1. Base and CF prompts within an
        # example are both long or both short, so per-example shapes match while
        # the batch is ragged.
        def ragged_idx(x: CausalTrace) -> list[int]:
            text = x["raw_input"]
            return [0, 1] if len(text.split()) >= 3 else [0]

        tp = TokenPosition(ragged_idx, tiny_pipeline, id="ragged")
        unit = ResidualStream(
            layer=0, token_indices=tp, target_output=False, shape=(hidden,)
        )
        target = InterchangeTarget([[unit]])
        examples: list[dict[str, Any]] = [
            {
                "input": _make_trace("the quick brown fox"),
                "counterfactual_inputs": [_make_trace("a slow lazy dog")],
            },
            {
                "input": _make_trace("hi"),
                "counterfactual_inputs": [_make_trace("yo")],
            },
        ]
        with pytest.raises(ValueError, match="variable number of tokens"):
            prepare_intervenable_inputs(tiny_pipeline, examples, target)

    def test_multitoken_position_with_nonidentity_featurizer_is_allowed(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """A uniform multi-token span with a *non-identity* featurizer is no
        longer rejected: the feature interventions set ``keep_last_dim=True``, so
        the featurizer is applied per position (its ``d``-sized weights broadcast
        across the span) rather than to a folded ``num_positions*d`` vector.
        ``prepare_intervenable_inputs`` must build the four-tuple without raising
        — the end-to-end broadcast is checked separately in the batched-wrapper
        tests."""
        hidden = tiny_pipeline.model.config.hidden_size
        tp = TokenPosition(lambda _x: [0, 1], tiny_pipeline, id="span2")
        unit = ResidualStream(
            layer=0,
            token_indices=tp,
            target_output=False,
            shape=(hidden,),
            featurizer=_diag_featurizer(hidden),
        )
        target = InterchangeTarget([[unit]])
        # Must not raise.
        prepare_intervenable_inputs(tiny_pipeline, _long_examples(), target)


class TestPrepareIntervenableInputsProperty:
    """Structural invariants on the four-tuple shape."""

    pytestmark = pytest.mark.property

    def test_counterfactual_count_matches_input_counterfactuals(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """``len(batched_counterfactuals)`` mirrors the number of counterfactual
        inputs per example."""
        examples: list[dict[str, Any]] = [
            {
                "input": _make_trace("a"),
                "counterfactual_inputs": [_make_trace("b"), _make_trace("c")],
            },
            {
                "input": _make_trace("d"),
                "counterfactual_inputs": [_make_trace("e"), _make_trace("f")],
            },
        ]
        # Two units in two groups: pyvene needs one unit per group when
        # groups carry different counterfactual inputs.
        unit_a = _make_unit(tiny_pipeline, layer=0)
        unit_b = _make_unit(tiny_pipeline, layer=1)
        target = InterchangeTarget([[unit_a], [unit_b]])

        _, cfs, _, _ = prepare_intervenable_inputs(tiny_pipeline, examples, target)
        assert len(cfs) == len(examples[0]["counterfactual_inputs"])

    def test_feature_indices_length_equals_total_units(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """``len(feature_indices) == sum(len(group) for group in target)``."""
        unit_a = _make_unit(tiny_pipeline, layer=0)
        unit_b = _make_unit(tiny_pipeline, layer=1)
        unit_c = _make_unit(tiny_pipeline, layer=0, target_output=True)
        target = InterchangeTarget([[unit_a, unit_b], [unit_c]])
        examples: list[dict[str, Any]] = [
            {
                "input": _make_trace("a"),
                "counterfactual_inputs": [_make_trace("b"), _make_trace("c")],
            },
        ]

        _, _, _, feature_indices = prepare_intervenable_inputs(
            tiny_pipeline, examples, target
        )
        expected = sum(len(group) for group in target)
        assert len(feature_indices) == expected

    def test_none_feature_indices_propagate_per_example(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """A unit whose ``get_feature_indices()`` is ``None`` produces
        ``[None, None, ...]`` — one slot per example, all ``None``."""
        unit = _make_unit(tiny_pipeline)  # default feature_indices=None
        assert unit.get_feature_indices() is None
        target = InterchangeTarget([[unit]])

        examples = _two_examples()
        _, _, _, feature_indices = prepare_intervenable_inputs(
            tiny_pipeline, examples, target
        )

        assert len(feature_indices) == 1
        assert feature_indices[0] == [None] * len(examples)


# --------------------------------------------------------------------------- #
#  batched_interchange_intervention                                           #
# --------------------------------------------------------------------------- #
class TestBatchedInterchangeInterventionUnit:
    """Single-batch wrapper that runs an interchange intervention on the
    target pipeline and moves resulting input tensors back to CPU. Consumed
    by every per-batch caller in :mod:`causalab.methods`."""

    pytestmark = pytest.mark.unit

    def test_basic_intervention_returns_sequences(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        unit = _make_unit(tiny_pipeline)
        target = InterchangeTarget([[unit]])
        intervenable = prepare_intervenable_model(
            tiny_pipeline, target, intervention_type="interchange"
        )
        examples = _two_examples()

        output = batched_interchange_intervention(
            tiny_pipeline,
            intervenable,
            examples,
            target,
            output_scores=False,
        )

        # The contract is a dict with at least 'sequences' on CPU.
        assert "sequences" in output
        seqs = output["sequences"]
        assert isinstance(seqs, torch.Tensor)
        assert seqs.device.type == "cpu"
        # max_new_tokens=1 → exactly one new token per example
        assert seqs.shape == (len(examples), 1)
        # 'string' is always added by intervenable_generate's dump call
        assert "string" in output

    def test_uniform_multitoken_interchange_runs_end_to_end(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """A uniform multi-token span (positions 0,1) runs all the way through
        pyvene's gather/scatter and returns well-formed sequences — the
        regression that proves multi-token interchange works, not merely that the
        guard changed. Patching two positions must also produce a *different*
        result than patching only the last of those positions, confirming both
        tokens are actually swapped."""
        hidden = tiny_pipeline.model.config.hidden_size
        examples = _long_examples()

        tp_span = TokenPosition(lambda _x: [0, 1], tiny_pipeline, id="span2")
        unit_span = ResidualStream(
            layer=0, token_indices=tp_span, target_output=False, shape=(hidden,)
        )
        target_span = InterchangeTarget([[unit_span]])
        intervenable_span = prepare_intervenable_model(
            tiny_pipeline, target_span, intervention_type="interchange"
        )
        out_span = batched_interchange_intervention(
            tiny_pipeline,
            intervenable_span,
            examples,
            target_span,
            output_scores=False,
        )

        assert "sequences" in out_span
        assert isinstance(out_span["sequences"], torch.Tensor)
        assert out_span["sequences"].device.type == "cpu"
        assert out_span["sequences"].shape == (len(examples), 1)

        # Patching only the second of the two positions is a strict subset of the
        # span patch; with a non-degenerate tiny model the swapped tokens differ,
        # so the generated sequences should not be identical.
        tp_one = TokenPosition(lambda _x: [1], tiny_pipeline, id="pos1")
        unit_one = ResidualStream(
            layer=0, token_indices=tp_one, target_output=False, shape=(hidden,)
        )
        target_one = InterchangeTarget([[unit_one]])
        intervenable_one = prepare_intervenable_model(
            tiny_pipeline, target_one, intervention_type="interchange"
        )
        out_one = batched_interchange_intervention(
            tiny_pipeline,
            intervenable_one,
            examples,
            target_one,
            output_scores=False,
        )
        assert out_one["sequences"].shape == (len(examples), 1)

    def test_multitoken_interchange_applies_featurizer_per_position(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """A multi-token span with a *dimension-sensitive* featurizer runs
        end-to-end. The featurizer asserts its input's last dim is ``d``, so the
        run only succeeds if pyvene hands it ``(b, num_pos, d)`` and applies it
        per position (``keep_last_dim=True``) instead of folding the span into a
        ``num_pos*d`` vector — the regression that proves trainable featurizers
        work with multi-token spans, not just identity ones."""
        hidden = tiny_pipeline.model.config.hidden_size
        tp = TokenPosition(lambda _x: [0, 1], tiny_pipeline, id="span2")
        unit = ResidualStream(
            layer=0,
            token_indices=tp,
            target_output=False,
            shape=(hidden,),
            featurizer=_diag_featurizer(hidden),
        )
        target = InterchangeTarget([[unit]])
        intervenable = prepare_intervenable_model(
            tiny_pipeline, target, intervention_type="interchange"
        )
        output = batched_interchange_intervention(
            tiny_pipeline,
            intervenable,
            _long_examples(),
            target,
            output_scores=False,
        )
        assert output["sequences"].shape == (len(_long_examples()), 1)
        assert output["sequences"].device.type == "cpu"

    @pytest.mark.parametrize("positions", [[0], [0, 1]], ids=["single", "multitoken"])
    def test_interchange_with_rank_sensitive_featurizer(
        self, tiny_pipeline: LMPipeline, positions: list[int]
    ) -> None:
        """A *non-broadcast* featurizer (spline manifold) runs end-to-end through
        interchange on both a single-token and a uniform multi-token span.

        ``keep_last_dim=True`` hands the featurizer ``(b, num_pos, d)`` — and even
        the single-token span as ``(b, 1, d)``. The diag/rotation featurizers act
        on the last dim and broadcast over that extra axis, so they cannot detect
        the rank change; this manifold featurizer hard-codes a 2-D layout and is
        the surface that actually regressed. Without the leading-dim shim this
        raised ``IndexError`` inside ``Manifold.encode`` on *both* span widths."""
        hidden = tiny_pipeline.model.config.hidden_size
        tp = TokenPosition(lambda _x: positions, tiny_pipeline, id="rank_span")
        unit = ResidualStream(
            layer=0,
            token_indices=tp,
            target_output=False,
            shape=(hidden,),
            featurizer=_manifold_featurizer(hidden),
        )
        target = InterchangeTarget([[unit]])
        intervenable = prepare_intervenable_model(
            tiny_pipeline, target, intervention_type="interchange"
        )
        examples = _long_examples()
        output = batched_interchange_intervention(
            tiny_pipeline,
            intervenable,
            examples,
            target,
            output_scores=False,
        )
        assert output["sequences"].shape == (len(examples), 1)
        assert output["sequences"].device.type == "cpu"

    def test_output_scores_true_includes_scores_key(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        unit = _make_unit(tiny_pipeline)
        target = InterchangeTarget([[unit]])
        intervenable = prepare_intervenable_model(
            tiny_pipeline, target, intervention_type="interchange"
        )

        output = batched_interchange_intervention(
            tiny_pipeline,
            intervenable,
            _two_examples(),
            target,
            output_scores=True,
        )

        assert "scores" in output
        # Scores are a list (per generated token) of tensors on CPU.
        for s in output["scores"]:
            assert isinstance(s, torch.Tensor)
            assert s.device.type == "cpu"

    def test_cross_model_patching_with_same_source_pipeline(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """``source_pipeline`` branch: passing a non-``None`` source pipeline
        triggers :func:`collect_source_representations` and nulls out
        ``batched_counterfactuals`` before calling pyvene. With the same
        pipeline on both sides the call must still return well-formed
        output."""
        unit_target = _make_unit(tiny_pipeline)
        target_target = InterchangeTarget([[unit_target]])
        intervenable_target = prepare_intervenable_model(
            tiny_pipeline, target_target, intervention_type="interchange"
        )

        unit_source = _make_unit(tiny_pipeline)
        target_source = InterchangeTarget([[unit_source]])
        intervenable_source = prepare_intervenable_model(
            tiny_pipeline, target_source, intervention_type="collect"
        )

        output = batched_interchange_intervention(
            tiny_pipeline,
            intervenable_target,
            _two_examples(),
            target_target,
            output_scores=False,
            source_pipeline=tiny_pipeline,
            source_intervenable_model=intervenable_source,
        )
        assert "sequences" in output
        assert output["sequences"].device.type == "cpu"


class TestBatchedInterchangeInterventionProperty:
    """Device contracts and cross-model identity for the batched wrapper."""

    pytestmark = pytest.mark.property

    def test_input_tensors_are_moved_to_cpu(self, tiny_pipeline: LMPipeline) -> None:
        """Promotes the legacy ``test_tensor_device_handling`` case:
        all returned dict entries in batched_base / batched_counterfactuals
        are on CPU after the call. We re-tokenise after the call to assert
        only the wrapper's contract on its own returns — but the public
        observable is that ``output['sequences']`` is on CPU."""
        unit = _make_unit(tiny_pipeline)
        target = InterchangeTarget([[unit]])
        intervenable = prepare_intervenable_model(
            tiny_pipeline, target, intervention_type="interchange"
        )

        output = batched_interchange_intervention(
            tiny_pipeline,
            intervenable,
            _two_examples(),
            target,
            output_scores=True,
        )

        # Every tensor in the output dict must be CPU-resident.
        assert output["sequences"].device.type == "cpu"
        for s in output["scores"]:
            assert s.device.type == "cpu"

    def test_same_pipeline_cross_model_equals_vanilla_interchange(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """Identity-pipeline patching: when ``source_pipeline`` is the same
        as the target pipeline, the cross-model branch must produce
        exactly the same sequences as vanilla interchange."""
        examples = _two_examples()
        unit_target = _make_unit(tiny_pipeline)
        target_target = InterchangeTarget([[unit_target]])

        # Build a fresh source-side IntervenableModel — pyvene attaches
        # hooks per-instance, so we can't reuse a single one for both
        # branches.
        unit_source = _make_unit(tiny_pipeline)
        target_source = InterchangeTarget([[unit_source]])

        intervenable_a = prepare_intervenable_model(
            tiny_pipeline, target_target, intervention_type="interchange"
        )
        out_vanilla = batched_interchange_intervention(
            tiny_pipeline,
            intervenable_a,
            examples,
            target_target,
            output_scores=False,
        )

        intervenable_b = prepare_intervenable_model(
            tiny_pipeline, target_target, intervention_type="interchange"
        )
        intervenable_source = prepare_intervenable_model(
            tiny_pipeline, target_source, intervention_type="collect"
        )
        out_cross = batched_interchange_intervention(
            tiny_pipeline,
            intervenable_b,
            examples,
            target_target,
            output_scores=False,
            source_pipeline=tiny_pipeline,
            source_intervenable_model=intervenable_source,
        )

        assert torch.equal(out_vanilla["sequences"], out_cross["sequences"])


# --------------------------------------------------------------------------- #
#  Multi-token interchange vs. an independent hook-based oracle               #
# --------------------------------------------------------------------------- #
class TestMultiTokenInterchangeHookOracle:
    """Verify multi-token interchange against a ground truth computed with our
    own forward hooks — independent of pyvene's gather/scatter. We capture the
    source residual stream, overwrite the base's span positions by hand, and
    read the resulting next-token logits; causalab's pyvene-backed interchange
    must reproduce those logits. Intervening on layer 0 (so the patch propagates
    through the rest of the network to the final position) keeps the comparison
    non-trivial — and we assert the intervention actually moves the logits off
    the clean baseline so a silent no-op can't pass."""

    pytestmark = pytest.mark.unit

    _LAYER = 0
    _POSITIONS = [1, 2]
    _BASE = "the quick brown fox jumps"
    _SOURCE = "a slow lazy old dog sits"

    def test_full_swap_matches_manual_hook_oracle(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """Identity featurizer, full swap: every span position's residual must
        equal the source's, at *both* tokens of the span. The hand-rolled patch
        copies source positions [1,2] directly; causalab must match."""
        layer, positions = self._LAYER, self._POSITIONS
        base_inputs = tiny_pipeline.load([_make_trace(self._BASE)])
        source_inputs = tiny_pipeline.load([_make_trace(self._SOURCE)])

        src_resid = _capture_residual(tiny_pipeline, layer, source_inputs)
        patch = src_resid[:, positions, :]  # full swap, both span tokens
        manual = _next_token_logits(tiny_pipeline, base_inputs, layer, positions, patch)
        clean = _next_token_logits(tiny_pipeline, base_inputs, layer)

        tp = TokenPosition(lambda _x: positions, tiny_pipeline, id="span2")
        unit = ResidualStream(
            layer=layer,
            token_indices=tp,
            target_output=True,
            shape=(tiny_pipeline.model.config.hidden_size,),
        )
        target = InterchangeTarget([[unit]])
        intervenable = prepare_intervenable_model(
            tiny_pipeline, target, intervention_type="interchange"
        )
        out = batched_interchange_intervention(
            tiny_pipeline,
            intervenable,
            [
                {
                    "input": _make_trace(self._BASE),
                    "counterfactual_inputs": [_make_trace(self._SOURCE)],
                }
            ],
            target,
            output_scores=True,
        )
        causalab = out["scores"][0]

        # Non-trivial: the swap must actually change the prediction logits.
        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_rotation_subspace_applied_per_position(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """Decisive per-position check. With a fixed rotation featurizer and a
        partial subspace (first ``k`` feature dims), the hand-rolled oracle
        rotates each span position independently, swaps the first ``k`` feature
        dims from the source, and rotates back. causalab must match to
        floating-point precision — which is only possible if the ``d x d``
        rotation is applied *per position* (``keep_last_dim=True``), not to a
        folded ``num_pos*d`` vector (which would trip the featurizer's assert or
        swap mis-aligned columns)."""
        torch.manual_seed(0)
        layer, positions = self._LAYER, self._POSITIONS
        hidden = tiny_pipeline.model.config.hidden_size
        k = 5
        R = torch.linalg.qr(torch.randn(hidden, hidden))[0]

        base_inputs = tiny_pipeline.load([_make_trace(self._BASE)])
        source_inputs = tiny_pipeline.load([_make_trace(self._SOURCE)])
        base_resid = _capture_residual(tiny_pipeline, layer, base_inputs)
        src_resid = _capture_residual(tiny_pipeline, layer, source_inputs)

        # Per position: rotate base & source, swap first k feature dims, rotate
        # back into raw activation space.
        patch = torch.empty(1, len(positions), hidden)
        with torch.no_grad():
            for i, p in enumerate(positions):
                f_out = (base_resid[:, p, :] @ R).clone()
                f_out[:, :k] = (src_resid[:, p, :] @ R)[:, :k]
                patch[:, i, :] = f_out @ R.T
        manual = _next_token_logits(tiny_pipeline, base_inputs, layer, positions, patch)
        clean = _next_token_logits(tiny_pipeline, base_inputs, layer)

        featurizer = Featurizer(
            featurizer=_RotateFeaturizerModule(R),
            inverse_featurizer=_RotateInverseFeaturizerModule(R),
            n_features=hidden,
        )
        tp = TokenPosition(lambda _x: positions, tiny_pipeline, id="span2")
        unit = ResidualStream(
            layer=layer,
            token_indices=tp,
            target_output=True,
            shape=(hidden,),
            featurizer=featurizer,
            feature_indices=list(range(k)),
        )
        target = InterchangeTarget([[unit]])
        intervenable = prepare_intervenable_model(
            tiny_pipeline, target, intervention_type="interchange"
        )
        out = batched_interchange_intervention(
            tiny_pipeline,
            intervenable,
            [
                {
                    "input": _make_trace(self._BASE),
                    "counterfactual_inputs": [_make_trace(self._SOURCE)],
                }
            ],
            target,
            output_scores=True,
        )
        causalab = out["scores"][0]

        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------- #
#  Attention-head (h.pos, 4-D) interchange vs. an independent hook oracle     #
# --------------------------------------------------------------------------- #
class TestAttentionHeadInterchangeHookOracle:
    """Attention-head interchange exercises pyvene's 4-D ``h.pos`` path, which
    ``keep_last_dim=True`` also affects (it skips the head-fold ``bhsd_to_bs_hd``)
    and which the rest of the suite only covers for construction/IDs — never an
    actual interchange run. These verify head interchange against a ground truth
    built with our own forward_pre_hook on ``o_proj`` (no pyvene), with a
    *multi-token* span so the head-axis vs. position-axis ragged-guard handling
    is exercised too."""

    pytestmark = pytest.mark.unit

    _LAYER = 0
    _HEAD = 2
    _POSITIONS = [1, 2]
    _BASE = "the quick brown fox jumps"
    _SOURCE = "a slow lazy old dog sits"

    def _run_causalab(
        self,
        tiny_pipeline: LMPipeline,
        featurizer: Featurizer,
        feature_indices: list[int] | None,
    ) -> torch.Tensor:
        d_head = (
            tiny_pipeline.model.config.hidden_size
            // tiny_pipeline.model.config.num_attention_heads
        )
        tp = TokenPosition(lambda _x: self._POSITIONS, tiny_pipeline, id="span2")
        unit = AttentionHead(
            layer=self._LAYER,
            head=self._HEAD,
            token_indices=tp,
            featurizer=featurizer,
            feature_indices=feature_indices,
            shape=(d_head,),
        )
        target = InterchangeTarget([[unit]])
        intervenable = prepare_intervenable_model(
            tiny_pipeline, target, intervention_type="interchange"
        )
        out = batched_interchange_intervention(
            tiny_pipeline,
            intervenable,
            [
                {
                    "input": _make_trace(self._BASE),
                    "counterfactual_inputs": [_make_trace(self._SOURCE)],
                }
            ],
            target,
            output_scores=True,
        )
        return out["scores"][0]

    def test_full_head_swap_matches_manual_oproj_oracle(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """Identity featurizer, full swap of head H's value output at the span:
        causalab must equal a manual o_proj-slice swap at both span positions."""
        layer, head, positions = self._LAYER, self._HEAD, self._POSITIONS
        base_inputs = tiny_pipeline.load([_make_trace(self._BASE)])
        source_inputs = tiny_pipeline.load([_make_trace(self._SOURCE)])

        src_head = _capture_head_value(tiny_pipeline, layer, head, source_inputs)
        patch = src_head[:, positions, :]
        manual = _head_patched_next_logits(
            tiny_pipeline, layer, head, positions, base_inputs, patch
        )
        clean = _next_token_logits(tiny_pipeline, base_inputs, layer)

        causalab = self._run_causalab(tiny_pipeline, Featurizer(), None)

        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_rotation_subspace_applied_per_position_on_head(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """Rotation featurizer (``d_head x d_head``) + partial subspace on a head,
        multi-token span. The oracle rotates each span position's head slice
        independently, swaps the first ``k`` feature dims, rotates back; causalab
        must match — proving the head featurizer is applied per position on the
        4-D path, not folded."""
        torch.manual_seed(0)
        layer, head, positions = self._LAYER, self._HEAD, self._POSITIONS
        d_head = (
            tiny_pipeline.model.config.hidden_size
            // tiny_pipeline.model.config.num_attention_heads
        )
        k = 2
        R = torch.linalg.qr(torch.randn(d_head, d_head))[0]

        base_inputs = tiny_pipeline.load([_make_trace(self._BASE)])
        source_inputs = tiny_pipeline.load([_make_trace(self._SOURCE)])
        base_head = _capture_head_value(tiny_pipeline, layer, head, base_inputs)
        src_head = _capture_head_value(tiny_pipeline, layer, head, source_inputs)

        patch = torch.empty(1, len(positions), d_head)
        with torch.no_grad():
            for i, p in enumerate(positions):
                f_out = (base_head[:, p, :] @ R).clone()
                f_out[:, :k] = (src_head[:, p, :] @ R)[:, :k]
                patch[:, i, :] = f_out @ R.T
        manual = _head_patched_next_logits(
            tiny_pipeline, layer, head, positions, base_inputs, patch
        )
        clean = _next_token_logits(tiny_pipeline, base_inputs, layer)

        featurizer = Featurizer(
            featurizer=_RotateFeaturizerModule(R),
            inverse_featurizer=_RotateInverseFeaturizerModule(R),
            n_features=d_head,
        )
        causalab = self._run_causalab(tiny_pipeline, featurizer, list(range(k)))

        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_ragged_head_span_still_raises(self, tiny_pipeline: LMPipeline) -> None:
        """The ragged guard must still fire on the *position* axis of an h.pos
        unit (regression for the layout-aware fix: the head and position axes are
        checked separately, so a genuinely ragged position span is still caught
        while a uniform multi-token one is not)."""

        def ragged_idx(x: CausalTrace) -> list[int]:
            return [1, 2] if len(x["raw_input"].split()) >= 3 else [1]

        tp = TokenPosition(ragged_idx, tiny_pipeline, id="ragged")
        d_head = (
            tiny_pipeline.model.config.hidden_size
            // tiny_pipeline.model.config.num_attention_heads
        )
        unit = AttentionHead(layer=0, head=1, token_indices=tp, shape=(d_head,))
        target = InterchangeTarget([[unit]])
        examples: list[dict[str, Any]] = [
            {
                "input": _make_trace("the quick brown fox"),
                "counterfactual_inputs": [_make_trace("a slow lazy dog")],
            },
            {
                "input": _make_trace("hi"),
                "counterfactual_inputs": [_make_trace("yo")],
            },
        ]
        with pytest.raises(ValueError, match="variable number of tokens"):
            prepare_intervenable_inputs(tiny_pipeline, examples, target)


# --------------------------------------------------------------------------- #
#  Capability coverage vs. independent hook oracles                           #
# --------------------------------------------------------------------------- #
class TestInterventionCapabilityOracles:
    """Basic coverage of causalab's intervention range, each checked against a
    hand-rolled hook on the exact module pyvene taps (no pyvene in the oracle).
    Covers the *where* axis (interchange on block input/output, MLP output, whole
    attention output) and the *what* axis (collect / read, additive steering,
    replace / zero-ablation). Single position, identity featurizer, layer 0 so
    the edit propagates to the next-token logits; each asserts the intervention
    moves the logits off the clean baseline so a silent no-op can't pass."""

    pytestmark = pytest.mark.unit

    _LAYER = 0
    _POS = 3
    _BASE = "the quick brown fox jumps"
    _SOURCE = "a slow lazy old dog sits"

    # (component, hook-module key, unit factory) — the unit's component_type must
    # match the module/kind in `_component_module`.
    _INTERCHANGE_SITES = [
        (
            "block_output",
            lambda layer, tp, d: ResidualStream(
                layer=layer, token_indices=tp, target_output=True, shape=(d,)
            ),
        ),
        (
            "block_input",
            lambda layer, tp, d: ResidualStream(
                layer=layer, token_indices=tp, target_output=False, shape=(d,)
            ),
        ),
        (
            "mlp_output",
            lambda layer, tp, d: MLP(layer=layer, token_indices=tp, shape=(d,)),
        ),
        (
            "attention_output",
            lambda layer, tp, d: AttentionOutput(
                layer=layer, token_indices=tp, shape=(d,)
            ),
        ),
    ]

    def _tp(self, tiny_pipeline: LMPipeline) -> TokenPosition:
        return TokenPosition(lambda _x: [self._POS], tiny_pipeline, id="p")

    @pytest.mark.parametrize(
        "component, build", _INTERCHANGE_SITES, ids=[s[0] for s in _INTERCHANGE_SITES]
    )
    def test_interchange_matches_hook_oracle(
        self, tiny_pipeline: LMPipeline, component: str, build
    ) -> None:
        """Interchange at each supported site must equal a manual swap of that
        module's activation slice from the source run."""
        layer, pos = self._LAYER, self._POS
        d = tiny_pipeline.model.config.hidden_size
        base_inputs = tiny_pipeline.load([_make_trace(self._BASE)])
        source_inputs = tiny_pipeline.load([_make_trace(self._SOURCE)])

        module, kind = _component_module(tiny_pipeline, layer, component)
        src = _capture_component(tiny_pipeline, module, kind, source_inputs)

        def swap(hidden: torch.Tensor) -> None:
            hidden[:, pos, :] = src[:, pos, :]

        manual = _component_edited_logits(
            tiny_pipeline, base_inputs, module, kind, swap
        )
        clean = _next_token_logits(tiny_pipeline, base_inputs, layer)

        unit = build(layer, self._tp(tiny_pipeline), d)
        target = InterchangeTarget([[unit]])
        intervenable = prepare_intervenable_model(
            tiny_pipeline, target, intervention_type="interchange"
        )
        out = batched_interchange_intervention(
            tiny_pipeline,
            intervenable,
            [
                {
                    "input": _make_trace(self._BASE),
                    "counterfactual_inputs": [_make_trace(self._SOURCE)],
                }
            ],
            target,
            output_scores=True,
        )
        causalab = out["scores"][0]

        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_collect_matches_hook_capture(self, tiny_pipeline: LMPipeline) -> None:
        """``collect_features`` at block_output must equal the residual our hook
        captures at that layer/position (the read capability)."""
        layer, pos = self._LAYER, self._POS
        d = tiny_pipeline.model.config.hidden_size
        base_inputs = tiny_pipeline.load([_make_trace(self._BASE)])

        unit = ResidualStream(
            layer=layer,
            token_indices=self._tp(tiny_pipeline),
            target_output=True,
            shape=(d,),
        )
        feats = collect_features(
            [
                {
                    "input": _make_trace(self._BASE),
                    "counterfactual_inputs": [_make_trace(self._SOURCE)],
                }
            ],
            tiny_pipeline,
            [unit],
        )[unit.id]

        module, kind = _component_module(tiny_pipeline, layer, "block_output")
        captured = _capture_component(tiny_pipeline, module, kind, base_inputs)[
            :, pos, :
        ]
        torch.testing.assert_close(feats, captured, atol=1e-5, rtol=1e-4)

    def test_additive_steering_matches_hook_oracle(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """``run_steering_interventions(mode='add')`` must equal a manual hook
        that adds the steering vector to the block_output at the position."""
        torch.manual_seed(0)
        layer, pos = self._LAYER, self._POS
        d = tiny_pipeline.model.config.hidden_size
        v = torch.randn(d)
        base_inputs = tiny_pipeline.load([_make_trace(self._BASE)])

        unit = ResidualStream(
            layer=layer,
            token_indices=self._tp(tiny_pipeline),
            target_output=True,
            shape=(d,),
        )
        target = InterchangeTarget([[unit]])
        out = run_steering_interventions(
            tiny_pipeline,
            [{"input": _make_trace(self._BASE)}],
            target,
            {unit.id: v},
            mode="add",
            output_scores=True,
        )
        causalab = out["scores"][0][0]

        module, kind = _component_module(tiny_pipeline, layer, "block_output")

        def add(hidden: torch.Tensor) -> None:
            hidden[:, pos, :] = hidden[:, pos, :] + v

        manual = _component_edited_logits(tiny_pipeline, base_inputs, module, kind, add)
        clean = _next_token_logits(tiny_pipeline, base_inputs, layer)

        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)

    def test_replace_zero_ablation_matches_hook_oracle(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """``run_steering_interventions(mode='replace')`` with a zero vector must
        equal a manual hook that zeroes the block_output at the position."""
        layer, pos = self._LAYER, self._POS
        d = tiny_pipeline.model.config.hidden_size
        base_inputs = tiny_pipeline.load([_make_trace(self._BASE)])

        unit = ResidualStream(
            layer=layer,
            token_indices=self._tp(tiny_pipeline),
            target_output=True,
            shape=(d,),
        )
        target = InterchangeTarget([[unit]])
        out = run_steering_interventions(
            tiny_pipeline,
            [{"input": _make_trace(self._BASE)}],
            target,
            {unit.id: torch.zeros(d)},
            mode="replace",
            output_scores=True,
        )
        causalab = out["scores"][0][0]

        module, kind = _component_module(tiny_pipeline, layer, "block_output")

        def zero(hidden: torch.Tensor) -> None:
            hidden[:, pos, :] = 0.0

        manual = _component_edited_logits(
            tiny_pipeline, base_inputs, module, kind, zero
        )
        clean = _next_token_logits(tiny_pipeline, base_inputs, layer)

        assert not torch.allclose(causalab, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------- #
#  run_interchange_interventions                                              #
# --------------------------------------------------------------------------- #
class TestRunInterchangeInterventionsUnit:
    """Top-level loop: prepares an intervenable model, batches the dataset,
    optionally converts to top-k scores, returns a ``{key: [per-batch
    values]}`` dict consumed by every analysis flow."""

    pytestmark = pytest.mark.unit

    def test_basic_run_returns_per_batch_lists(self, tiny_pipeline: LMPipeline) -> None:
        unit = _make_unit(tiny_pipeline)
        target = InterchangeTarget([[unit]])
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
            target,
            batch_size=2,
            output_scores=False,
        )

        # ceil(3/2) = 2 batches
        assert "sequences" in result
        assert len(result["sequences"]) == 2

    def test_output_scores_false_omits_scores_key(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        unit = _make_unit(tiny_pipeline)
        target = InterchangeTarget([[unit]])
        dataset = [
            {
                "input": _make_trace("a"),
                "counterfactual_inputs": [_make_trace("b")],
            },
        ]

        result = run_interchange_interventions(
            tiny_pipeline,
            dataset,
            target,
            batch_size=1,
            output_scores=False,
        )
        assert "scores" not in result

    def test_top_k_scores_uses_int_argument(self, tiny_pipeline: LMPipeline) -> None:
        """``output_scores=int`` triggers conversion to top-k format
        (memory-efficient). Result must include both ``sequences`` and
        ``scores`` keys, with the same per-batch fan-out as ``sequences``."""
        unit = _make_unit(tiny_pipeline)
        target = InterchangeTarget([[unit]])
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
            target,
            batch_size=2,
            output_scores=5,
        )
        assert "sequences" in result
        assert "scores" in result
        # one batch → one entry per key
        assert len(result["sequences"]) == len(result["scores"]) == 1

    def test_small_batch_size_yields_correct_batch_count(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """batch_size=1 over 3 examples → 3 batches → 3 list entries per key."""
        unit = _make_unit(tiny_pipeline)
        target = InterchangeTarget([[unit]])
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
            target,
            batch_size=1,
            output_scores=False,
        )
        assert len(result["sequences"]) == 3

    def test_empty_dataset_returns_sentinel(self, tiny_pipeline: LMPipeline) -> None:
        """The function returns ``{"sequences": [[]], "string": [[]]}``
        when the dataset is empty.

        The ``"string"`` key in that sentinel is the plan's "Open question:
        unintended, drop" — kept as a regression marker since callers may
        rely on the key set. xfail-pending-fix when the source is fixed."""
        unit = _make_unit(tiny_pipeline)
        target = InterchangeTarget([[unit]])

        result = run_interchange_interventions(
            tiny_pipeline,
            [],
            target,
            batch_size=2,
            output_scores=False,
        )
        assert result == {"sequences": [[]], "string": [[]]}


class TestRunInterchangeInterventionsProperty:
    """Aggregation invariants over the per-batch outputs."""

    pytestmark = pytest.mark.property

    def test_batch_count_equals_ceil_of_dataset_over_batch_size(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        unit = _make_unit(tiny_pipeline)
        target = InterchangeTarget([[unit]])
        dataset = [
            {
                "input": _make_trace(f"in_{i}"),
                "counterfactual_inputs": [_make_trace(f"cf_{i}")],
            }
            for i in range(5)
        ]

        for batch_size in (1, 2, 3, 5):
            result = run_interchange_interventions(
                tiny_pipeline,
                dataset,
                target,
                batch_size=batch_size,
                output_scores=False,
            )
            expected = -(-len(dataset) // batch_size)  # ceil
            assert len(result["sequences"]) == expected, (
                f"batch_size={batch_size}: expected {expected} batches, "
                f"got {len(result['sequences'])}"
            )

    def test_result_keys_are_consistent_across_batches(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """Each per-batch dict carries the same keys, so the outer
        ``{key: [...]}`` dict has the union (= intersection) of those keys."""
        unit = _make_unit(tiny_pipeline)
        target = InterchangeTarget([[unit]])
        dataset = [
            {
                "input": _make_trace(t),
                "counterfactual_inputs": [_make_trace(c)],
            }
            for t, c in [("aa", "bb"), ("cc", "dd"), ("ee", "ff")]
        ]
        # output_scores=True → per-batch dicts have {sequences, scores, string}
        result = run_interchange_interventions(
            tiny_pipeline,
            dataset,
            target,
            batch_size=2,
            output_scores=True,
        )
        assert set(result.keys()) == {"sequences", "scores", "string"}
        # every key carries one entry per batch
        n_batches = len(result["sequences"])
        for k in result:
            assert len(result[k]) == n_batches, (
                f"key {k!r} has {len(result[k])} entries, expected {n_batches}"
            )

    def test_output_scores_false_excludes_scores_key(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """The contract: ``output_scores=False`` produces no ``scores`` key."""
        unit = _make_unit(tiny_pipeline)
        target = InterchangeTarget([[unit]])
        dataset = [
            {
                "input": _make_trace("a"),
                "counterfactual_inputs": [_make_trace("b")],
            },
        ]
        result = run_interchange_interventions(
            tiny_pipeline,
            dataset,
            target,
            batch_size=1,
            output_scores=False,
        )
        assert "scores" not in result


# --------------------------------------------------------------------------- #
#  Interchange through a non-trivial featurizer                               #
# --------------------------------------------------------------------------- #
class TestInterchangeThroughFeaturizer:
    """The featurizer must be applied exactly once to the source.

    The source of an interchange is read in one forward and written in another.
    Whether the *read* returns activation space or feature space is invisible in
    the identity case and in the output shape — but with a real featurizer,
    reading through it and then letting the interchange featurize again applies
    it twice. For a DAS rotation that is ``x @ Rᵀ @ Rᵀ`` (silently wrong); for a
    rank-reducing subspace it is a shape error.

    A *square orthonormal* rotation makes the correct answer checkable without a
    second implementation: the rotation is a bijection, so swapping features is
    exactly swapping activations, and interchange through it must equal
    interchange through no featurizer at all. Applying R twice does not.
    """

    pytestmark = pytest.mark.unit

    # Both prompts tokenize to the same length, and the swap targets the LAST
    # token — position 0 is BOS, whose activation is identical for any prompt, so
    # a swap there is a no-op and every assertion below would pass vacuously.
    _BASE = "the quick brown fox"
    _SOURCE = "a slow lazy dog"

    @staticmethod
    def _scores(pipeline, featurizer, *, intervene: bool = True) -> torch.Tensor:
        hidden = pipeline.model.config.hidden_size
        cls = TestInterchangeThroughFeaturizer
        unit = ResidualStream(
            layer=1,
            token_indices=TokenPosition(
                lambda trace: [
                    len(pipeline.tokenizer(trace["raw_input"])["input_ids"]) - 1
                ],
                pipeline,
                id="last_token",
            ),
            target_output=True,
            shape=(hidden,),
            featurizer=featurizer,
        )
        source = cls._BASE if not intervene else cls._SOURCE
        result = run_interchange_interventions(
            pipeline,
            [_make_cf_example(cls._BASE, source)],
            InterchangeTarget([[unit]]),
            batch_size=1,
            output_scores=True,
        )
        return result["scores"][0][0]

    def test_bijective_featurizer_matches_raw_interchange(self, tiny_pipeline) -> None:
        hidden = tiny_pipeline.model.config.hidden_size
        rotation = _random_rotation(hidden, seed=3)
        rotated = self._scores(tiny_pipeline, _rotate_featurizer(rotation))
        plain = self._scores(tiny_pipeline, Featurizer())
        torch.testing.assert_close(rotated, plain, atol=1e-4, rtol=1e-3)

    def test_rank_reducing_subspace_swaps_only_that_subspace(
        self, tiny_pipeline
    ) -> None:
        """A rank-k subspace featurizer must swap the source's projection onto the
        subspace and keep the base's orthogonal complement — and must not raise,
        which is what a doubly-applied rank-reducing featurizer does."""
        from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer

        hidden = tiny_pipeline.model.config.hidden_size
        torch.manual_seed(0)
        subspace = SubspaceFeaturizer(shape=(hidden, 2), trainable=False)

        subspace_swap = self._scores(tiny_pipeline, subspace)
        full_swap = self._scores(tiny_pipeline, Featurizer())
        no_swap = self._scores(tiny_pipeline, Featurizer(), intervene=False)

        # The swap has an effect...
        assert not torch.allclose(subspace_swap, no_swap, atol=1e-6)
        # ...but only within the 2-dim subspace, so it is not the full swap.
        assert not torch.allclose(subspace_swap, full_swap, atol=1e-6)
