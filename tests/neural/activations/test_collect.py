"""Direct tests for ``causalab.neural.activations.collect``.

This module hosts the dataset-wide and per-batch activation harvesting
primitives that feed every featurizer learner (``methods/pca``,
``methods/spline``, ``methods/flow``) and every interchange / path-steering
analysis. ``collect_features`` produces a ``{unit_id -> (n_samples, hidden)}``
dict on CPU; ``collect_source_representations`` / ``collect_batch_representations``
produce the per-location source tensors that ``neural/activations/interchange_mode.py``
and ``methods/metric.py`` inject as source representations during cross-model
patching; ``collect_class_centroids`` packages a ``collect_features``
call plus a per-value reduction used by path-steering visualization.

If any of these returns the wrong shape, ordering, or device, every
downstream interchange / featurizer training run silently consumes corrupted
activations — so this module is the canonical place to pin those contracts.

The *unit* classes assert envelope reshape / dict-keying / log-message
behaviour, not numerics. The *property* classes use the real ``tiny_pipeline``
fixture from ``tests/neural/conftest.py`` so the shapes are exercised end-to-end
against a real (tiny) model.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.LM_units import AttentionHead, ResidualStream
from causalab.neural.components import head_layout
from causalab.neural.activations.collect import (
    collect_batch_representations,
    collect_class_centroids,
    collect_features,
    collect_source_representations,
)
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget


# --------------------------------------------------------------------------- #
#  Local RNG helper                                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).float()


# --------------------------------------------------------------------------- #
#  Shared builders — real units and datasets on the tiny pipelines            #
# --------------------------------------------------------------------------- #
def _example(text: str) -> dict:
    """One dataset entry with no counterfactuals — all `collect_features` needs."""
    return {"input": _trace(text), "counterfactual_inputs": []}


def _dataset(n: int) -> list[dict]:
    """`n` distinct prompts, deliberately of differing token lengths so every
    collection exercises the padded-batch position shift."""
    prompts = [
        "the quick brown fox",
        "a slow lazy old dog sits",
        "hello",
        "one two three four five six",
        "alpha beta",
    ]
    return [_example(prompts[i % len(prompts)]) for i in range(n)]


def _last_tp(pipeline) -> TokenPosition:
    """The last real token of each example, in its own unpadded frame."""

    def index(trace):
        ids = pipeline.tokenizer(trace["raw_input"])["input_ids"]
        return [len(ids) - 1]

    return TokenPosition(index, pipeline, id="last_token")


# --------------------------------------------------------------------------- #
#  collect_features                                                           #
# --------------------------------------------------------------------------- #
class TestCollectFeaturesUnit:
    """Dataset-wide activation harvest feeding ``methods/`` featurizer learners.

    ``collect_features`` is the entry point for every featurizer training
    run: ``methods/pca``, ``methods/spline``, ``methods/flow``,
    ``methods/interchange/layer_scan``, and the
    ``analyses/{activation_manifold, subspace, path_steering}`` analyses all
    consume its ``{unit.id -> (n_samples, hidden)}`` output on CPU.

    These ran against a mocked ``IntervenableModel`` until the nnsight
    migration removed the object they mocked. They now run on the real tiny
    pipeline, which is both closer to docs/TESTS.md's mocking policy and
    strictly more informative — a mocked backbone could not have caught a
    wrong-width per-head read, which is the failure mode that motivated #386.
    """

    pytestmark = pytest.mark.unit

    def test_returns_dict_keyed_by_unit_id(self, tiny_pipeline) -> None:
        units = [_make_residual_unit(tiny_pipeline, layer) for layer in (0, 1)]
        result = collect_features(_dataset(3), tiny_pipeline, units, batch_size=2)
        assert isinstance(result, dict)
        assert set(result) == {u.id for u in units}
        for unit in units:
            assert isinstance(result[unit.id], torch.Tensor)

    def test_emits_debug_log_with_unit_count_and_shape(
        self, tiny_pipeline, caplog
    ) -> None:
        units = [_make_residual_unit(tiny_pipeline, 0)]
        with caplog.at_level(
            logging.DEBUG, logger="causalab.neural.activations.collect"
        ):
            collect_features(_dataset(2), tiny_pipeline, units, batch_size=2)
        assert "Collected features for" in caplog.text
        assert "Feature tensor shape:" in caplog.text

    def test_returns_tensors_on_cpu(self, tiny_pipeline) -> None:
        """Whatever device the model runs on, the harvest lands on CPU — a
        dataset-sized activation dump must not pin GPU memory."""
        units = [_make_residual_unit(tiny_pipeline, 0)]
        result = collect_features(_dataset(3), tiny_pipeline, units, batch_size=2)
        for tensor in result.values():
            assert tensor.device.type == "cpu"

    def test_result_is_2d_with_hidden_last(self, tiny_pipeline) -> None:
        hidden = tiny_pipeline.model.config.hidden_size
        units = [_make_residual_unit(tiny_pipeline, 0)]
        result = collect_features(_dataset(3), tiny_pipeline, units, batch_size=2)
        for tensor in result.values():
            assert tensor.ndim == 2
            assert tensor.shape == (3, hidden)

    def test_attention_head_collects_at_head_dim(self, tiny_pipeline) -> None:
        """A per-head unit yields ``head_dim`` columns, not ``hidden``."""
        _n_q, _n_kv, head_dim = head_layout(tiny_pipeline.model.config)
        units = [
            AttentionHead(layer=0, head=head, token_indices=_last_tp(tiny_pipeline))
            for head in (0, 1)
        ]
        result = collect_features(_dataset(3), tiny_pipeline, units, batch_size=2)
        for unit in units:
            assert result[unit.id].shape == (3, head_dim)

    def test_mixed_component_widths_are_supported(self, tiny_pipeline) -> None:
        """Residual and per-head units collect together, each at its own width."""
        hidden = tiny_pipeline.model.config.hidden_size
        _n_q, _n_kv, head_dim = head_layout(tiny_pipeline.model.config)
        residual = _make_residual_unit(tiny_pipeline, 0)
        head = AttentionHead(layer=0, head=0, token_indices=_last_tp(tiny_pipeline))
        result = collect_features(
            _dataset(3), tiny_pipeline, [residual, head], batch_size=2
        )
        assert result[residual.id].shape == (3, hidden)
        assert result[head.id].shape == (3, head_dim)

    def test_multi_token_span_contributes_one_row_per_position(
        self, tiny_pipeline
    ) -> None:
        """A unit selecting N tokens yields N samples per example — the flattening
        every featurizer fitter downstream assumes."""
        hidden = tiny_pipeline.model.config.hidden_size
        span = TokenPosition(lambda _x: [0, 1], tiny_pipeline, id="first_two")
        unit = ResidualStream(
            layer=0, token_indices=span, target_output=True, shape=(hidden,)
        )
        result = collect_features(_dataset(3), tiny_pipeline, [unit], batch_size=2)
        assert result[unit.id].shape == (6, hidden)

    def test_collect_output_logits_returns_tuple_dict_and_list(
        self, tiny_pipeline
    ) -> None:
        """``collect_output_logits=True`` returns ``(dict, list[Tensor])`` with one
        logits tensor per example, so a caller needing both activations and output
        distributions pays for one forward instead of two."""
        units = [_make_residual_unit(tiny_pipeline, 0)]
        result = collect_features(
            _dataset(3), tiny_pipeline, units, batch_size=2, collect_output_logits=True
        )
        assert isinstance(result, tuple)
        features_dict, logits_list = result
        assert isinstance(features_dict, dict)
        # 3 examples / batch_size=2 -> 2 batches -> 2 + 1 = 3 per-example logits.
        assert len(logits_list) == 3
        for logits in logits_list:
            assert isinstance(logits, torch.Tensor)
            assert logits.shape[-1] == tiny_pipeline.model.config.vocab_size

    def test_duplicate_unit_id_raises(self, tiny_pipeline) -> None:
        """Two units sharing an ``id`` would silently merge — must raise instead."""
        unit = _make_residual_unit(tiny_pipeline, 0)
        with pytest.raises(ValueError, match="Duplicate model_unit.id"):
            collect_features(_dataset(2), tiny_pipeline, [unit, unit], batch_size=2)


class TestCollectPaddingCorrectness:
    """A padded row must collect the same activation it would collect alone.

    Left padding shifts every real token's absolute index, so a forward that
    numbers positions from the pad tokens corrupts every activation in a padded
    row. RoPE models are immune (relative positions cancel a uniform shift), so
    this must run on the **absolute-position** GPT-2 stub to mean anything — the
    RoPE ``tiny_pipeline`` cannot fail it.

    This replaces a mock that asserted a ``position_ids`` tensor was handed to
    the backbone's forward. The engine now derives them itself, so the contract worth
    pinning is the observable one: batching must not change the numbers.
    """

    pytestmark = pytest.mark.unit

    def test_padded_row_matches_unpadded_collection(self, tiny_gpt2_pipeline) -> None:
        short, long = "hello", "a considerably longer prompt than the other one"
        unit_alone = _make_residual_unit(tiny_gpt2_pipeline, 1)
        alone = collect_features(
            [_example(short)], tiny_gpt2_pipeline, [unit_alone], batch_size=1
        )[unit_alone.id]

        unit_batched = _make_residual_unit(tiny_gpt2_pipeline, 1)
        batched = collect_features(
            [_example(short), _example(long)],
            tiny_gpt2_pipeline,
            [unit_batched],
            batch_size=2,
        )[unit_batched.id]

        torch.testing.assert_close(batched[0:1], alone, atol=1e-5, rtol=1e-4)


def _trace(text: str) -> CausalTrace:
    """Helper: build a minimal ``CausalTrace`` carrying just ``raw_input``."""
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _make_residual_unit(pipeline, layer: int) -> ResidualStream:
    """Helper: build a real ``ResidualStream`` unit at ``layer``, last token."""

    def last_token(trace):
        return [len(pipeline.load([trace])["input_ids"][0]) - 1]

    tp = TokenPosition(last_token, pipeline, id="last_token")
    return ResidualStream(
        layer=layer,
        token_indices=tp,
        shape=(pipeline.model.config.hidden_size,),
        target_output=True,
    )


def _make_residual_unit_at_token(
    pipeline, layer: int, token_index: int
) -> ResidualStream:
    """``ResidualStream`` reading a FIXED token index (unpadded-frame; the left-pad
    shift in ``index_component`` rebases it into the padded batch). Reading a
    *non-final* token is what surfaces the left-pad ``position_ids`` bug."""
    tp = TokenPosition(lambda trace: [token_index], pipeline, id=f"tok{token_index}")
    return ResidualStream(
        layer=layer,
        token_indices=tp,
        shape=(pipeline.model.config.hidden_size,),
        target_output=True,
    )


class TestCollectFeaturesProperty:
    """Tier-property invariants for ``collect_features`` on a real (tiny) model.

    These tests exercise the engine's
    ``((base_outputs, collected_activations), cf_outputs)`` envelope
    end-to-end against ``tests/neural/conftest.py::tiny_pipeline`` — the same
    shape contract the production runners hit. ``collect_features`` now pins
    ``model.eval()`` + ``torch.no_grad()`` for the duration of collection;
    these tests assert the resulting tensors have no autograd graph.
    """

    pytestmark = pytest.mark.property

    def test_sample_count_matches_dataset_length(self, tiny_pipeline) -> None:
        dataset = [{"input": _trace(f"hello {i}")} for i in range(5)]
        unit = _make_residual_unit(tiny_pipeline, layer=0)
        result = collect_features(dataset, tiny_pipeline, [unit], batch_size=2)
        # One scalar position per example × 5 examples → (5, hidden).
        assert result[unit.id].shape[0] == len(dataset)

    def test_tensors_are_on_cpu(self, tiny_pipeline) -> None:
        dataset = [{"input": _trace("hello world")}]
        unit = _make_residual_unit(tiny_pipeline, layer=0)
        result = collect_features(dataset, tiny_pipeline, [unit], batch_size=1)
        for t in result.values():
            assert t.device.type == "cpu"

    def test_deterministic_under_fixed_seed(self, tiny_pipeline) -> None:
        """Calling twice on the same dataset → byte-identical tensors."""
        dataset = [{"input": _trace(f"hello {i}")} for i in range(3)]
        unit_a = _make_residual_unit(tiny_pipeline, layer=0)
        out_a = collect_features(dataset, tiny_pipeline, [unit_a], batch_size=2)
        unit_b = _make_residual_unit(tiny_pipeline, layer=0)
        out_b = collect_features(dataset, tiny_pipeline, [unit_b], batch_size=2)
        # Same id-template since both units target layer 0 / last_token.
        (key_a,) = out_a.keys()
        (key_b,) = out_b.keys()
        assert torch.equal(out_a[key_a], out_b[key_b])

    def test_no_grad_is_pinned_during_collection(self, tiny_pipeline) -> None:
        """Collected tensors must not carry an autograd graph (forward-only)."""
        dataset = [{"input": _trace("hello world")}]
        unit = _make_residual_unit(tiny_pipeline, layer=0)
        result = collect_features(dataset, tiny_pipeline, [unit], batch_size=1)
        for t in result.values():
            assert t.requires_grad is False
            assert t.grad_fn is None

    def test_dict_keys_equal_unit_ids(self, tiny_pipeline) -> None:
        unit0 = _make_residual_unit(tiny_pipeline, layer=0)
        unit1 = _make_residual_unit(tiny_pipeline, layer=1)
        dataset = [{"input": _trace("hello world")}]
        result = collect_features(dataset, tiny_pipeline, [unit0, unit1], batch_size=1)
        assert set(result.keys()) == {unit0.id, unit1.id}

    @pytest.mark.parametrize(
        "pipeline_fixture", ["tiny_pipeline", "tiny_gpt2_pipeline"]
    )
    def test_left_padded_collection_matches_unpadded(
        self, pipeline_fixture, request
    ) -> None:
        """End-to-end guard for the position_ids fix on a real model.

        Collecting a non-final token must be identical whether each example is
        processed alone (``batch_size=1`` → no padding, the reference) or together
        in one left-padded batch. On an **absolute-position** model
        (``tiny_gpt2_pipeline``) this fails without ``position_ids`` on the
        collection forward — the padded row is mis-encoded (the ROME-replication
        bug). On a **RoPE** model (``tiny_pipeline``) it holds either way, since
        relative positions are invariant to a uniform left-pad shift.
        Parametrizing both pins "GPT-2 is fixed" and "Llama is unchanged" together.
        """
        pipeline = request.getfixturevalue(pipeline_fixture)
        dataset = [
            {"input": _trace("the cat sat quietly on the warm windowsill")},
            {
                "input": _trace(
                    "the quick brown fox jumps over the lazy dog again and again today"
                )
            },
        ]
        # Guards so the test actually exercises the bug: the short row must be
        # left-padded in the batch, and long enough to read a non-final token.
        short_len = int(pipeline.load([dataset[0]["input"]])["input_ids"].shape[1])
        batch_mask = pipeline.load([ex["input"] for ex in dataset])["attention_mask"]
        assert (batch_mask == 0).any(), "batch did not introduce left padding"
        assert short_len >= 3, f"short prompt too short ({short_len} tokens)"

        unit_solo = _make_residual_unit_at_token(pipeline, layer=1, token_index=1)
        unit_batched = _make_residual_unit_at_token(pipeline, layer=1, token_index=1)
        solo = collect_features(dataset, pipeline, [unit_solo], batch_size=1)
        batched = collect_features(dataset, pipeline, [unit_batched], batch_size=2)

        assert torch.allclose(
            solo[unit_solo.id], batched[unit_batched.id], atol=1e-4
        ), (
            f"[{pipeline_fixture}] left-padded collection diverged from unpadded at "
            "a non-final token — position_ids on the collection forward is what "
            "makes these equal on an absolute-position model"
        )

    def test_duplicate_unit_id_raises_on_real_pipeline(self, tiny_pipeline) -> None:
        """Duplicate-id contract must hold against the real pipeline too."""
        unit_a = _make_residual_unit(tiny_pipeline, layer=0)
        unit_b = _make_residual_unit(tiny_pipeline, layer=0)
        # Both targets are layer 0 + last_token → identical ``id``.
        assert unit_a.id == unit_b.id
        dataset = [{"input": _trace("hello")}]
        with pytest.raises(ValueError, match="Duplicate model_unit.id"):
            collect_features(dataset, tiny_pipeline, [unit_a, unit_b], batch_size=1)

    def test_multitoken_span_collects_each_position_independently(
        self, tiny_pipeline
    ) -> None:
        """A uniform multi-token span collects one d-vector PER position.

        With ``keep_last_dim=True`` (PR #334) the collect intervention gathers
        ``(b, num_pos, d)`` and ``collect_features`` flattens it to
        ``(b*num_pos, d)`` — so the span's per-position rows must equal collecting
        each position on its own. The old folded ``(b, num_pos*d)`` reshape
        interleaved positions and features and would not match. This is the one
        consumer of the collect change (review obs #2) the single-token path
        leaves byte-identical; here we lock the *multi-token* path.
        """
        hidden = tiny_pipeline.model.config.hidden_size
        dataset = [{"input": _trace(f"hello world {i}")} for i in range(3)]

        def _unit(positions: list[int], unit_id: str) -> ResidualStream:
            tp = TokenPosition(lambda _x, p=positions: p, tiny_pipeline, id=unit_id)
            return ResidualStream(
                layer=0,
                token_indices=tp,
                target_output=True,
                shape=(hidden,),
            )

        span = _unit([0, 1], "span01")
        pos0 = _unit([0], "pos0")
        pos1 = _unit([1], "pos1")

        out_span = collect_features(dataset, tiny_pipeline, [span], batch_size=2)[
            span.id
        ]
        out_p0 = collect_features(dataset, tiny_pipeline, [pos0], batch_size=2)[pos0.id]
        out_p1 = collect_features(dataset, tiny_pipeline, [pos1], batch_size=2)[pos1.id]

        # (b, num_pos, d) flattened row-major: rows are [ex0-pos0, ex0-pos1, ...].
        assert out_span.shape == (len(dataset) * 2, hidden)
        assert torch.allclose(out_span[0::2], out_p0, atol=1e-5)
        assert torch.allclose(out_span[1::2], out_p1, atol=1e-5)


# --------------------------------------------------------------------------- #
#  collect_batch_representations                                              #
# --------------------------------------------------------------------------- #
class TestCollectBatchRepresentationsUnit:
    """Per-batch primitive producing the source activations an interchange writes.

    Returns ``list[Tensor]`` ordered to match the flat-unit traversal across
    ``interchange_target`` groups. ``neural/activations/interchange_mode.py`` and
    ``methods/metric.py`` consume that list positionally against the same
    traversal, so a drift in ordering or length silently patches the wrong site.

    Each group is its own forward — the groups have different counterfactual
    inputs — which is what makes the per-group unit-cursor slicing load-bearing.
    """

    pytestmark = pytest.mark.unit

    def test_returns_one_tensor_per_unit_across_groups(self, tiny_pipeline) -> None:
        """Flat output length equals ``sum(len(group) for group in target)``."""
        target = InterchangeTarget(
            [
                [
                    _make_residual_unit(tiny_pipeline, 0),
                    _make_residual_unit(tiny_pipeline, 1),
                ],
                [_make_residual_unit_at_token(tiny_pipeline, 1, 0)],
            ]
        )
        cf_groups = [[_trace("alpha beta")], [_trace("gamma delta")]]
        batched_cf = [tiny_pipeline.load(group) for group in cf_groups]
        positions = [
            unit.resolve_positions(
                cf_groups[group_index],
                attention_mask=batched_cf[group_index]["attention_mask"],
                is_original=False,
            )
            for group_index, group in enumerate(target)
            for unit in group
        ]

        out = collect_batch_representations(
            tiny_pipeline, batched_cf, target, positions
        )
        assert len(out) == 3

    def test_each_group_reads_its_own_counterfactual(self, tiny_pipeline) -> None:
        """Two groups pointing at the same site but different inputs must return
        different activations — proof the per-group forward is not shared."""
        unit_a = _make_residual_unit(tiny_pipeline, 0)
        unit_b = _make_residual_unit(tiny_pipeline, 0)
        target = InterchangeTarget([[unit_a], [unit_b]])
        cf_groups = [[_trace("alpha beta")], [_trace("wholly different words")]]
        batched_cf = [tiny_pipeline.load(group) for group in cf_groups]
        positions = [
            unit.resolve_positions(
                cf_groups[group_index],
                attention_mask=batched_cf[group_index]["attention_mask"],
                is_original=False,
            )
            for group_index, group in enumerate(target)
            for unit in group
        ]

        first, second = collect_batch_representations(
            tiny_pipeline, batched_cf, target, positions
        )
        assert not torch.allclose(first, second)

    def test_per_group_positions_are_sliced_by_unit_cursor(self, tiny_pipeline) -> None:
        """A unit's activation must be the one at *its* position, even when the
        groups have different unit counts — i.e. the cursor advances by group size.

        Group 0 holds two units at different token positions; group 1 holds one.
        Reading the same sites directly gives the expected values.
        """
        g0_tok0 = _make_residual_unit_at_token(tiny_pipeline, 0, 0)
        g0_tok1 = _make_residual_unit_at_token(tiny_pipeline, 0, 1)
        g1_tok0 = _make_residual_unit_at_token(tiny_pipeline, 1, 0)
        target = InterchangeTarget([[g0_tok0, g0_tok1], [g1_tok0]])

        cf_groups = [[_trace("alpha beta gamma")], [_trace("delta epsilon zeta")]]
        batched_cf = [tiny_pipeline.load(group) for group in cf_groups]
        positions = [
            unit.resolve_positions(
                cf_groups[group_index],
                attention_mask=batched_cf[group_index]["attention_mask"],
                is_original=False,
            )
            for group_index, group in enumerate(target)
            for unit in group
        ]
        out = collect_batch_representations(
            tiny_pipeline, batched_cf, target, positions
        )
        assert len(out) == 3

        # Independent reads of the same three sites, one per group input.
        expected_g0 = collect_features(
            [{"input": cf_groups[0][0], "counterfactual_inputs": []}],
            tiny_pipeline,
            [g0_tok0, g0_tok1],
            batch_size=1,
        )
        expected_g1 = collect_features(
            [{"input": cf_groups[1][0], "counterfactual_inputs": []}],
            tiny_pipeline,
            [g1_tok0],
            batch_size=1,
        )
        torch.testing.assert_close(out[0].reshape(1, -1), expected_g0[g0_tok0.id])
        torch.testing.assert_close(out[1].reshape(1, -1), expected_g0[g0_tok1.id])
        torch.testing.assert_close(out[2].reshape(1, -1), expected_g1[g1_tok0.id])


class TestCollectBatchRepresentationsProperty:
    """Length-and-ordering invariants for ``collect_batch_representations``.

    The returned ``list[Tensor]`` is consumed positionally against
    the units' config-add order. The two property assertions below pin
    that contract end-to-end on the real (tiny) pipeline.
    """

    pytestmark = pytest.mark.property

    def test_output_length_equals_total_flat_units(self, tiny_pipeline) -> None:
        """Single-group case: 2 units → output list of length 2.

        Multi-group / cross-model patching cases are exercised transitively
        via ``methods/metric.py`` callers.
        """
        unit0 = _make_residual_unit(tiny_pipeline, layer=0)
        unit1 = _make_residual_unit(tiny_pipeline, layer=1)
        target = InterchangeTarget([[unit0, unit1]])

        cf_inputs = [_trace("alpha")]
        batched_cf = [tiny_pipeline.load(cf_inputs)]
        cf_indices = [
            unit0.index_component(cf_inputs, batch=True, is_original=False),
            unit1.index_component(cf_inputs, batch=True, is_original=False),
        ]
        out = collect_batch_representations(
            tiny_pipeline, batched_cf, target, cf_indices
        )
        assert len(out) == len(target.flatten())


# --------------------------------------------------------------------------- #
#  collect_source_representations                                             #
# --------------------------------------------------------------------------- #
class TestCollectSourceRepresentationsUnit:
    """Convenience wrapper: tokenize counterfactuals + delegate to batch primitive.

    Used by ``neural/activations/interchange_mode.py`` and ``methods/metric.py``
    to harvest source-pipeline activations before writing them into a target
    pipeline. The wrapper zips ``counterfactual_inputs`` into per-group batches,
    tokenizes each with the *source* pipeline, and resolves positions with
    ``is_original=False`` — the flag a paired token position uses to pick the
    counterfactual's positions rather than the base's.
    """

    pytestmark = pytest.mark.unit

    def test_zips_counterfactuals_into_per_group_batches(self, tiny_pipeline) -> None:
        """``examples[i]["counterfactual_inputs"][g]`` becomes row ``i`` of group ``g``."""
        target = InterchangeTarget(
            [
                [_make_residual_unit(tiny_pipeline, 0)],
                [_make_residual_unit(tiny_pipeline, 1)],
            ]
        )
        examples = [
            {
                "input": _trace("base one"),
                "counterfactual_inputs": [_trace("g0 e0"), _trace("g1 e0")],
            },
            {
                "input": _trace("base two"),
                "counterfactual_inputs": [_trace("g0 e1"), _trace("g1 e1")],
            },
        ]
        out = collect_source_representations(tiny_pipeline, examples, target)
        # One tensor per flat unit, each carrying both examples' rows.
        assert len(out) == 2
        for tensor in out:
            assert tensor.shape[0] == 2

    def test_resolves_counterfactual_positions_not_base_ones(
        self, tiny_pipeline
    ) -> None:
        """A paired token position must be resolved with ``is_original=False``.

        The indexer here returns position 0 for a base read and position 2 for a
        counterfactual read, so reading the wrong side is numerically visible.
        """
        hidden = tiny_pipeline.model.config.hidden_size
        paired = TokenPosition(
            lambda _x, is_original=True: [0] if is_original else [2],
            tiny_pipeline,
            id="paired",
        )
        unit = ResidualStream(
            layer=0, token_indices=paired, target_output=True, shape=(hidden,)
        )
        target = InterchangeTarget([[unit]])
        cf_text = "alpha beta gamma delta"
        examples = [
            {
                "input": _trace("base text here"),
                "counterfactual_inputs": [_trace(cf_text)],
            }
        ]

        out = collect_source_representations(tiny_pipeline, examples, target)

        at_pos_2 = _make_residual_unit_at_token(tiny_pipeline, 0, 2)
        expected = collect_features(
            [{"input": _trace(cf_text), "counterfactual_inputs": []}],
            tiny_pipeline,
            [at_pos_2],
            batch_size=1,
        )[at_pos_2.id]
        torch.testing.assert_close(out[0].reshape(1, -1), expected)


class TestCollectSourceRepresentationsProperty:
    """End-to-end invariants for the tokenize-and-collect wrapper on real pipelines."""

    pytestmark = pytest.mark.property

    def test_output_length_equals_total_flat_units(self, tiny_pipeline) -> None:
        """Single-group case: 2 units → output list of length 2.

        Multi-group counterfactual setups are exercised transitively via
        ``neural/activations/interchange_mode.py``'s cross-model patching path.
        """
        unit0 = _make_residual_unit(tiny_pipeline, layer=0)
        unit1 = _make_residual_unit(tiny_pipeline, layer=1)
        target = InterchangeTarget([[unit0, unit1]])
        examples = [
            {
                "input": _trace("base_a"),
                "counterfactual_inputs": [_trace("cf_a")],
            },
            {
                "input": _trace("base_b"),
                "counterfactual_inputs": [_trace("cf_b")],
            },
        ]
        out = collect_source_representations(tiny_pipeline, examples, target)
        assert len(out) == len(target.flatten())


# --------------------------------------------------------------------------- #
#  collect_class_centroids                                                    #
# --------------------------------------------------------------------------- #
class TestCollectClassCentroidsUnit:
    """Per-variable-value centroid reducer feeding path-steering visualisation.

    Wraps ``collect_features`` on a single unit then averages the per-sample
    feature vectors by class index (``task.intervention_values``-aligned),
    returning ``(centroids[n_classes, k], mask[n_classes])``. Consumed by
    ``analyses/path_steering/path_visualization.py``.
    """

    pytestmark = pytest.mark.unit

    def test_returns_centroids_and_mask_with_correct_shapes(self) -> None:
        unit = MagicMock(id="unit0")
        target = InterchangeTarget([[unit]])
        features = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        samples = [
            {"input": {"v": "A"}},
            {"input": {"v": "A"}},
            {"input": {"v": "B"}},
            {"input": {"v": "C"}},
        ]
        task = MagicMock()
        task.intervention_variable = "v"
        task.intervention_values = ["A", "B", "C"]

        with patch(
            "causalab.neural.activations.collect.collect_features",
            return_value={"unit0": features},
        ):
            centroids, mask = collect_class_centroids(
                samples, MagicMock(), target, task
            )

        assert centroids.shape == (3, 3)
        assert mask.shape == (3,)
        assert mask.dtype == torch.bool

    def test_centroids_average_only_matching_samples(self) -> None:
        unit = MagicMock(id="unit0")
        target = InterchangeTarget([[unit]])
        # Class A has samples 0, 1; class B has sample 2; class C absent.
        features = torch.tensor(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [0.0, 5.0],
            ]
        )
        samples = [
            {"input": {"v": "A"}},
            {"input": {"v": "A"}},
            {"input": {"v": "B"}},
        ]
        task = MagicMock()
        task.intervention_variable = "v"
        task.intervention_values = ["A", "B", "C"]

        with patch(
            "causalab.neural.activations.collect.collect_features",
            return_value={"unit0": features},
        ):
            centroids, mask = collect_class_centroids(
                samples, MagicMock(), target, task
            )

        # Class A: mean of rows 0 and 1 = [2.0, 0.0].
        assert torch.allclose(centroids[0], torch.tensor([2.0, 0.0]))
        # Class B: just row 2 = [0.0, 5.0].
        assert torch.allclose(centroids[1], torch.tensor([0.0, 5.0]))
        # Class C: no samples → mask False, zero row.
        assert not mask[2]
        assert torch.equal(centroids[2], torch.zeros(2))

    def test_mask_true_only_for_classes_with_at_least_one_sample(self) -> None:
        unit = MagicMock(id="unit0")
        target = InterchangeTarget([[unit]])
        features = torch.tensor([[1.0, 2.0]])
        samples = [{"input": {"v": "A"}}]
        task = MagicMock()
        task.intervention_variable = "v"
        task.intervention_values = ["A", "B"]

        with patch(
            "causalab.neural.activations.collect.collect_features",
            return_value={"unit0": features},
        ):
            _, mask = collect_class_centroids(samples, MagicMock(), target, task)

        assert mask.tolist() == [True, False]


class TestCollectClassCentroidsProperty:
    """Algebraic invariants of the centroid reducer."""

    pytestmark = pytest.mark.property

    def test_permutation_invariance_in_sample_order(self) -> None:
        """Reordering ``filtered_samples`` (and ``features`` to match) preserves centroids."""
        unit = MagicMock(id="unit0")
        target = InterchangeTarget([[unit]])
        features = torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 3.0],
                [0.0, 4.0],
            ]
        )
        samples = [
            {"input": {"v": "A"}},
            {"input": {"v": "A"}},
            {"input": {"v": "B"}},
            {"input": {"v": "B"}},
        ]
        task = MagicMock()
        task.intervention_variable = "v"
        task.intervention_values = ["A", "B"]

        with patch(
            "causalab.neural.activations.collect.collect_features",
            return_value={"unit0": features},
        ):
            centroids_orig, mask_orig = collect_class_centroids(
                samples, MagicMock(), target, task
            )

        # Reverse both ``samples`` and ``features`` together.
        perm = list(reversed(range(len(samples))))
        samples_perm = [samples[i] for i in perm]
        features_perm = features[perm]

        with patch(
            "causalab.neural.activations.collect.collect_features",
            return_value={"unit0": features_perm},
        ):
            centroids_perm, mask_perm = collect_class_centroids(
                samples_perm, MagicMock(), target, task
            )

        assert torch.allclose(centroids_orig, centroids_perm)
        assert torch.equal(mask_orig, mask_perm)

    def test_single_sample_centroid_equals_that_sample(self) -> None:
        """With one sample per class, each centroid row equals that sample's features."""
        unit = MagicMock(id="unit0")
        target = InterchangeTarget([[unit]])
        features = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        samples = [
            {"input": {"v": "A"}},
            {"input": {"v": "B"}},
        ]
        task = MagicMock()
        task.intervention_variable = "v"
        task.intervention_values = ["A", "B"]

        with patch(
            "causalab.neural.activations.collect.collect_features",
            return_value={"unit0": features},
        ):
            centroids, mask = collect_class_centroids(
                samples, MagicMock(), target, task
            )

        assert torch.allclose(centroids[0], features[0])
        assert torch.allclose(centroids[1], features[1])
        assert bool(mask.all().item()) is True
