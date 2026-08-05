"""Tests for the causal-tracing method (``causalab.methods.causal_tracing``).

The tiny-random Llama stub has no task behaviour, so the *behavioural* assertions
use a "match-the-base-output" metric: each intervened generation is graded
against the un-intervened generation for that example. The correctness pins:

* **clean (scale-0) noise is a no-op** — the dynamic noise intervention at
  ``scale=0`` reproduces the base output exactly, proving the mixed noise pass is
  identity when it should be.
* **noise scale is 3σ** — the corruption "vector" is the subject-embedding std
  times ``noise_scale``, not a raw multiplier.
* **noise spans multi-token subjects**, while **restoration requires a single
  token per site** (per-example clean replace).
* **windowed cells restore every unit**, and the seeded noise is reproducible.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.causal_tracing import (
    collect_clean_vectors,
    corruption_intervention_type,
    make_corruption_vectors,
    run_causal_trace,
    run_causal_trace_scan,
    run_corrupted_floor,
)
from causalab.methods.causal_tracing.vectors import _entry_activation_std
from causalab.methods.metric import InterchangeMetric, compute_base_outputs
from causalab.neural.activations.targets import build_residual_stream_targets
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import (
    TokenPosition,
    get_all_tokens,
    get_last_token_index,
)
from causalab.neural.units import InterchangeTarget

# Tiers (per docs/TESTS.md methods/ → property + numerical_unit): shape/contract
# and round-trip wiring are `property`; the seeded-noise + sigma pins are
# `numerical_unit`.


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _trace(text: str) -> CausalTrace:
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _dataset(texts: list[str]) -> list[dict[str, Any]]:
    return [{"input": _trace(t)} for t in texts]


def _equal_length_dataset() -> list[dict[str, Any]]:
    """Four two-token prompts — equal length, so no bucketing is forced."""
    return _dataset(["hello world", "blue green", "red sky", "cat dog"])


def _last_token(pipeline: LMPipeline) -> TokenPosition:
    return TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline),
        pipeline,
        id="last_token",
    )


def _all_tokens(pipeline: LMPipeline, dataset: list[dict[str, Any]]) -> TokenPosition:
    return get_all_tokens(dataset[0]["input"], pipeline)


def _match_base_metric(pipeline: LMPipeline, dataset: list[dict[str, Any]]):
    """An ``InterchangeMetric`` grading an intervened output against the base."""
    base_outputs = compute_base_outputs(dataset, pipeline, batch_size=len(dataset))

    def fn(
        intervention_output: dict[str, Any],
        expected: dict[str, Any],
        original: dict[str, Any],
    ) -> float:
        return float(
            str(intervention_output.get("string", "")).strip()
            == str(original.get("string", "")).strip()
        )

    metric = InterchangeMetric(
        fn=fn, needs_causal_expected=False, needs_original_output=True
    )
    return metric, base_outputs


def _entry_target(pipeline: LMPipeline, span: TokenPosition, layer: int = -1):
    return build_residual_stream_targets(pipeline, [layer], [span])[(layer, span.id)]


# --------------------------------------------------------------------------- #
class TestVectorShapesAndWiring:
    pytestmark = pytest.mark.property

    def test_zero_corruption_is_broadcast_vector(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        dataset = _equal_length_dataset()
        target = _entry_target(mock_tiny_lm, _last_token(mock_tiny_lm))
        vectors = make_corruption_vectors("zero", mock_tiny_lm, dataset, target)
        hidden = mock_tiny_lm.model.config.hidden_size
        for unit in target.flatten():
            assert vectors[unit.id].shape == (hidden,)
            assert torch.count_nonzero(vectors[unit.id]) == 0

    def test_mean_corruption_is_broadcast_vector(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        dataset = _equal_length_dataset()
        target = _entry_target(mock_tiny_lm, _last_token(mock_tiny_lm))
        vectors = make_corruption_vectors("mean", mock_tiny_lm, dataset, target)
        hidden = mock_tiny_lm.model.config.hidden_size
        for unit in target.flatten():
            assert vectors[unit.id].shape == (hidden,)
            assert torch.isfinite(vectors[unit.id]).all()

    def test_noise_corruption_is_sigma_scale(self, mock_tiny_lm: LMPipeline) -> None:
        dataset = _equal_length_dataset()
        target = _entry_target(mock_tiny_lm, _last_token(mock_tiny_lm))
        sigma = _entry_activation_std(mock_tiny_lm, dataset, target, batch_size=4)
        vectors = make_corruption_vectors(
            "noise", mock_tiny_lm, dataset, target, noise_scale=3.0
        )
        hidden = mock_tiny_lm.model.config.hidden_size
        assert sigma > 0
        for unit in target.flatten():
            vec = vectors[unit.id]
            # A constant (n_features,) scale equal to 3*sigma — not per-example.
            assert vec.shape == (hidden,)
            assert torch.allclose(vec, torch.full((hidden,), 3.0 * sigma))

    def test_noise_corruption_spans_multi_token(self, mock_tiny_lm: LMPipeline) -> None:
        dataset = _equal_length_dataset()  # two-token prompts
        span = _all_tokens(mock_tiny_lm, dataset)  # spans both tokens
        target = _entry_target(mock_tiny_lm, span)
        # Multi-token noise is allowed (dynamic per-token intervention).
        vectors = make_corruption_vectors("noise", mock_tiny_lm, dataset, target)
        assert set(vectors.keys()) == {u.id for u in target.flatten()}
        assert corruption_intervention_type("noise") == "noise"
        assert corruption_intervention_type("zero") == "replace"

    def test_collect_clean_vectors_shape(self, mock_tiny_lm: LMPipeline) -> None:
        dataset = _equal_length_dataset()
        targets = build_residual_stream_targets(
            mock_tiny_lm, [0, 1], [_last_token(mock_tiny_lm)]
        )
        units = [u for t in targets.values() for u in t.flatten()]
        clean = collect_clean_vectors(mock_tiny_lm, dataset, units)
        hidden = mock_tiny_lm.model.config.hidden_size
        assert set(clean.keys()) == {u.id for u in units}
        for vec in clean.values():
            assert vec.shape == (len(dataset), hidden)

    def test_restoration_requires_single_position(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        dataset = _equal_length_dataset()  # two-token prompts
        span = _all_tokens(mock_tiny_lm, dataset)
        units = build_residual_stream_targets(mock_tiny_lm, [0], [span])[
            (0, span.id)
        ].flatten()
        with pytest.raises(ValueError, match="single-position"):
            collect_clean_vectors(mock_tiny_lm, dataset, units)

    def test_scan_returns_one_value_per_cell(self, mock_tiny_lm: LMPipeline) -> None:
        dataset = _equal_length_dataset()
        last = _last_token(mock_tiny_lm)
        entry_units = _entry_target(mock_tiny_lm, last).flatten()
        corruption = make_corruption_vectors(
            "mean", mock_tiny_lm, dataset, _entry_target(mock_tiny_lm, last)
        )
        swept_targets = build_residual_stream_targets(mock_tiny_lm, [0, 1], [last])
        swept_units = [u for t in swept_targets.values() for u in t.flatten()]
        clean = collect_clean_vectors(mock_tiny_lm, dataset, swept_units)
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        scores = run_causal_trace_scan(
            swept_targets,
            dataset,
            mock_tiny_lm,
            entry_units=entry_units,
            corruption_vectors=corruption,
            clean_vectors=clean,
            metric=metric,
            original_outputs=base_outputs,
        )
        assert set(scores.keys()) == set(swept_targets.keys())
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_windowed_cell_restores_all_units(self, mock_tiny_lm: LMPipeline) -> None:
        """A cell whose target holds several units restores all of them."""
        dataset = _equal_length_dataset()
        last = _last_token(mock_tiny_lm)
        entry_units = _entry_target(mock_tiny_lm, last).flatten()
        corruption = make_corruption_vectors(
            "zero", mock_tiny_lm, dataset, _entry_target(mock_tiny_lm, last)
        )
        # One cell whose restore target spans two layers (a window of 2).
        window_units = [
            u
            for layer in (0, 1)
            for u in build_residual_stream_targets(mock_tiny_lm, [layer], [last])[
                (layer, "last_token")
            ].flatten()
        ]
        cell_target = InterchangeTarget([window_units])
        clean = collect_clean_vectors(mock_tiny_lm, dataset, window_units)
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        scores = run_causal_trace_scan(
            {(0, "window"): cell_target},
            dataset,
            mock_tiny_lm,
            entry_units=entry_units,
            corruption_vectors=corruption,
            clean_vectors=clean,
            metric=metric,
            original_outputs=base_outputs,
        )
        assert set(scores.keys()) == {(0, "window")}
        assert 0.0 <= scores[(0, "window")] <= 1.0

    def test_entry_restore_overlap_raises(self, mock_tiny_lm: LMPipeline) -> None:
        """A restored site coinciding with a corrupted entry site is rejected."""
        dataset = _equal_length_dataset()
        last = _last_token(mock_tiny_lm)
        # Corrupt residual block_output at layer 0 and try to restore the same site.
        same = build_residual_stream_targets(mock_tiny_lm, [0], [last])
        entry_units = same[(0, "last_token")].flatten()
        corruption = make_corruption_vectors(
            "zero", mock_tiny_lm, dataset, InterchangeTarget([entry_units])
        )
        clean = collect_clean_vectors(mock_tiny_lm, dataset, entry_units)
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)
        with pytest.raises(ValueError, match="overlaps the corrupted entry"):
            run_causal_trace_scan(
                same,
                dataset,
                mock_tiny_lm,
                entry_units=entry_units,
                corruption_vectors=corruption,
                clean_vectors=clean,
                metric=metric,
                original_outputs=base_outputs,
            )

    def test_clean_noise_is_noop(self, mock_tiny_lm: LMPipeline) -> None:
        """Scale-0 noise corruption (dynamic noise path) reproduces the base."""
        dataset = _equal_length_dataset()
        last = _last_token(mock_tiny_lm)
        target = _entry_target(mock_tiny_lm, last)
        entry_units = target.flatten()
        corruption = make_corruption_vectors(
            "noise", mock_tiny_lm, dataset, target, noise_scale=0.0
        )
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)
        floor = run_corrupted_floor(
            mock_tiny_lm,
            dataset,
            entry_units,
            corruption,
            metric=metric,
            entry_type="noise",
            original_outputs=base_outputs,
        )
        assert floor == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
class TestSeededNoise:
    pytestmark = pytest.mark.numerical_unit

    def test_seeded_noise_reproducible(self, mock_tiny_lm: LMPipeline) -> None:
        """Same seed → identical noised generations; different seed → differs."""
        dataset = _equal_length_dataset()
        last = _last_token(mock_tiny_lm)
        target = _entry_target(mock_tiny_lm, last)
        entry_units = target.flatten()
        # Large scale so the corruption surely moves the (argmax) generation.
        corruption = make_corruption_vectors(
            "noise", mock_tiny_lm, dataset, target, noise_scale=25.0
        )
        type_map = {u.id: "noise" for u in entry_units}
        single = InterchangeTarget([entry_units])

        def run(seed: int):
            return run_causal_trace(
                mock_tiny_lm,
                dataset,
                single,
                {u.id: corruption[u.id] for u in entry_units},
                type_by_unit=type_map,
                noise_seed=seed,
                output_scores=False,
            )["string"]

        a, b, c = run(7), run(7), run(9)
        assert a == b  # same seed → identical
        assert a != c  # different seed → at least one example differs


# --------------------------------------------------------------------------- #
class TestNoiseStreamOverMixedLengths:
    """Seeded noise over a dataset whose examples select different numbers of
    positions.

    An all-tokens entry span used to force the dataset into one bucket per
    length, each bucket its own call and so its own restart of the noise stream —
    which meant two disjoint groups of examples were handed the *same* draws.
    Spans may now be ragged, so the dataset runs as one stream.

    What causal tracing needs is that grid cells are comparable: each cell is its
    own call and replays the same stream for the same dataset. That is what these
    pin, rather than the bucket-order invariance that only existed because of the
    bucketing.
    """

    pytestmark = pytest.mark.numerical_unit

    # Length-3 and length-6 prompts under the tiny-random Llama tokenizer, so an
    # all-tokens span selects a different number of positions per example.
    _L3 = ["hello world", "red sky", "a cat"]
    _L6 = ["the quick brown fox", "one two three four five"]

    def _noise_outputs(
        self, pipeline: LMPipeline, texts: list[str], seed: int
    ) -> dict[str, str]:
        """Run noise-only causal tracing over an all-tokens span; return
        ``{prompt_text: generated_string}``."""
        dataset = _dataset(texts)
        span = _all_tokens(pipeline, dataset)
        target = _entry_target(pipeline, span)
        entry_units = target.flatten()
        corruption = make_corruption_vectors(
            "noise", pipeline, dataset, target, noise_scale=25.0
        )
        type_map = {u.id: "noise" for u in entry_units}
        out = run_causal_trace(
            pipeline,
            dataset,
            target,
            {u.id: corruption[u.id] for u in entry_units},
            type_by_unit=type_map,
            noise_seed=seed,
            output_scores=False,
        )["string"]
        flat = [gen for batch in out for gen in batch]
        return {t: s for t, s in zip(texts, flat)}

    def test_mixed_lengths_run_in_one_pass(self, mock_tiny_lm: LMPipeline) -> None:
        """Every example is generated for, in dataset order, despite the span
        selecting a different number of positions for each length group."""
        texts = self._L3 + self._L6
        out = self._noise_outputs(mock_tiny_lm, texts, seed=7)
        assert list(out) == texts
        assert all(isinstance(v, str) for v in out.values())

    def test_same_seed_and_order_reproduces(self, mock_tiny_lm: LMPipeline) -> None:
        """The property grid cells rely on: a cell re-run with the same seed over
        the same dataset replays the same noise, so two cells differ only by
        where they restore."""
        texts = self._L3 + self._L6
        first = self._noise_outputs(mock_tiny_lm, texts, seed=7)
        again = self._noise_outputs(mock_tiny_lm, texts, seed=7)
        assert first == again

    def test_a_different_seed_changes_the_noise(self, mock_tiny_lm: LMPipeline) -> None:
        """Non-vacuity for the test above: the outputs are seed-dependent, so
        reproducing them means the stream replayed rather than the noise having
        no effect at all."""
        texts = self._L3 + self._L6
        assert self._noise_outputs(mock_tiny_lm, texts, seed=7) != self._noise_outputs(
            mock_tiny_lm, texts, seed=8
        )
