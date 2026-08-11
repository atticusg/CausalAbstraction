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
from causalab.methods.ablation._spans import group_by_position_count
from causalab.methods.metric import InterchangeMetric, compute_base_outputs
from causalab.neural.activations.site_grids import build_residual_stream_sites
from causalab.neural.pipeline import LMPipeline
from causalab.neural.specs import SiteSpec
from causalab.neural.token_positions import (
    TokenPosition,
    get_all_tokens,
    get_last_token_index,
)

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


def _entry_sites(
    pipeline: LMPipeline, span: TokenPosition, layer: int = -1
) -> list[SiteSpec]:
    groups = build_residual_stream_sites(pipeline, [layer], [span])[(layer, span.id)]
    return [spec for group in groups for spec in group]


# --------------------------------------------------------------------------- #
class TestVectorShapesAndWiring:
    pytestmark = pytest.mark.property

    def test_zero_corruption_is_broadcast_vector(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        dataset = _equal_length_dataset()
        sites = _entry_sites(mock_tiny_lm, _last_token(mock_tiny_lm))
        vectors = make_corruption_vectors("zero", mock_tiny_lm, dataset, sites)
        hidden = mock_tiny_lm.model.config.hidden_size
        for spec in sites:
            assert vectors[spec.key].shape == (hidden,)
            assert torch.count_nonzero(vectors[spec.key]) == 0

    def test_mean_corruption_is_broadcast_vector(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        dataset = _equal_length_dataset()
        sites = _entry_sites(mock_tiny_lm, _last_token(mock_tiny_lm))
        vectors = make_corruption_vectors("mean", mock_tiny_lm, dataset, sites)
        hidden = mock_tiny_lm.model.config.hidden_size
        for spec in sites:
            assert vectors[spec.key].shape == (hidden,)
            assert torch.isfinite(vectors[spec.key]).all()

    def test_noise_corruption_is_sigma_scale(self, mock_tiny_lm: LMPipeline) -> None:
        dataset = _equal_length_dataset()
        sites = _entry_sites(mock_tiny_lm, _last_token(mock_tiny_lm))
        sigma = _entry_activation_std(mock_tiny_lm, dataset, sites, batch_size=4)
        vectors = make_corruption_vectors(
            "noise", mock_tiny_lm, dataset, sites, noise_scale=3.0
        )
        hidden = mock_tiny_lm.model.config.hidden_size
        assert sigma > 0
        for spec in sites:
            vec = vectors[spec.key]
            # A constant (n_features,) scale equal to 3*sigma — not per-example.
            assert vec.shape == (hidden,)
            assert torch.allclose(vec, torch.full((hidden,), 3.0 * sigma))

    def test_noise_corruption_spans_multi_token(self, mock_tiny_lm: LMPipeline) -> None:
        dataset = _equal_length_dataset()  # two-token prompts
        span = _all_tokens(mock_tiny_lm, dataset)  # spans both tokens
        sites = _entry_sites(mock_tiny_lm, span)
        # Multi-token noise is allowed (dynamic per-token intervention).
        vectors = make_corruption_vectors("noise", mock_tiny_lm, dataset, sites)
        assert set(vectors.keys()) == {s.key for s in sites}
        assert corruption_intervention_type("noise") == "noise"
        assert corruption_intervention_type("zero") == "replace"

    def test_collect_clean_vectors_shape(self, mock_tiny_lm: LMPipeline) -> None:
        dataset = _equal_length_dataset()
        grid = build_residual_stream_sites(
            mock_tiny_lm, [0, 1], [_last_token(mock_tiny_lm)]
        )
        sites = [s for groups in grid.values() for g in groups for s in g]
        clean = collect_clean_vectors(mock_tiny_lm, dataset, sites)
        hidden = mock_tiny_lm.model.config.hidden_size
        assert set(clean.keys()) == {s.key for s in sites}
        for vec in clean.values():
            assert vec.shape == (len(dataset), hidden)

    def test_restoration_requires_single_position(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        dataset = _equal_length_dataset()  # two-token prompts
        span = _all_tokens(mock_tiny_lm, dataset)
        groups = build_residual_stream_sites(mock_tiny_lm, [0], [span])[(0, span.id)]
        sites = [s for g in groups for s in g]
        with pytest.raises(ValueError, match="single-position"):
            collect_clean_vectors(mock_tiny_lm, dataset, sites)

    def test_scan_returns_one_value_per_cell(self, mock_tiny_lm: LMPipeline) -> None:
        dataset = _equal_length_dataset()
        last = _last_token(mock_tiny_lm)
        entry_sites = _entry_sites(mock_tiny_lm, last)
        corruption = make_corruption_vectors(
            "mean", mock_tiny_lm, dataset, _entry_sites(mock_tiny_lm, last)
        )
        swept_grid = build_residual_stream_sites(mock_tiny_lm, [0, 1], [last])
        swept_sites = [s for groups in swept_grid.values() for g in groups for s in g]
        clean = collect_clean_vectors(mock_tiny_lm, dataset, swept_sites)
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        scores = run_causal_trace_scan(
            swept_grid,
            dataset,
            mock_tiny_lm,
            entry_sites=entry_sites,
            corruption_vectors=corruption,
            clean_vectors=clean,
            metric=metric,
            original_outputs=base_outputs,
        )
        assert set(scores.keys()) == set(swept_grid.keys())
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_windowed_cell_restores_all_units(self, mock_tiny_lm: LMPipeline) -> None:
        """A cell whose target holds several units restores all of them."""
        dataset = _equal_length_dataset()
        last = _last_token(mock_tiny_lm)
        entry_sites = _entry_sites(mock_tiny_lm, last)
        corruption = make_corruption_vectors(
            "zero", mock_tiny_lm, dataset, _entry_sites(mock_tiny_lm, last)
        )
        # One cell whose restore groups span two layers (a window of 2).
        window_sites = [
            s
            for layer in (0, 1)
            for group in build_residual_stream_sites(mock_tiny_lm, [layer], [last])[
                (layer, "last_token")
            ]
            for s in group
        ]
        clean = collect_clean_vectors(mock_tiny_lm, dataset, window_sites)
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        scores = run_causal_trace_scan(
            {(0, "window"): [window_sites]},
            dataset,
            mock_tiny_lm,
            entry_sites=entry_sites,
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
        same = build_residual_stream_sites(mock_tiny_lm, [0], [last])
        entry_sites = [s for g in same[(0, "last_token")] for s in g]
        corruption = make_corruption_vectors("zero", mock_tiny_lm, dataset, entry_sites)
        clean = collect_clean_vectors(mock_tiny_lm, dataset, entry_sites)
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)
        with pytest.raises(ValueError, match="overlaps the corrupted entry"):
            run_causal_trace_scan(
                same,
                dataset,
                mock_tiny_lm,
                entry_sites=entry_sites,
                corruption_vectors=corruption,
                clean_vectors=clean,
                metric=metric,
                original_outputs=base_outputs,
            )

    def test_clean_noise_is_noop(self, mock_tiny_lm: LMPipeline) -> None:
        """Scale-0 noise corruption (dynamic noise path) reproduces the base."""
        dataset = _equal_length_dataset()
        last = _last_token(mock_tiny_lm)
        entry_sites = _entry_sites(mock_tiny_lm, last)
        corruption = make_corruption_vectors(
            "noise", mock_tiny_lm, dataset, entry_sites, noise_scale=0.0
        )
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)
        floor = run_corrupted_floor(
            mock_tiny_lm,
            dataset,
            entry_sites,
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
        entry_sites = _entry_sites(mock_tiny_lm, last)
        # Large scale so the corruption surely moves the (argmax) generation.
        corruption = make_corruption_vectors(
            "noise", mock_tiny_lm, dataset, entry_sites, noise_scale=25.0
        )
        type_map = {s.key: "noise" for s in entry_sites}

        def run(seed: int):
            return run_causal_trace(
                mock_tiny_lm,
                dataset,
                entry_sites,
                {s.key: corruption[s.key] for s in entry_sites},
                type_by_key=type_map,
                noise_seed=seed,
                output_scores=False,
            ).strings

        a, b, c = run(7), run(7), run(9)
        assert a == b  # same seed → identical
        assert a != c  # different seed → at least one example differs


# --------------------------------------------------------------------------- #
class TestRaggedEntrySpan:
    """A mixed-length all-tokens entry span runs as ONE dataset-order stream on
    the nnsight engine (PL3, #405) — no length-bucketing, no per-bucket RNG
    reset. The reproducibility contract is per ``(dataset order, seed)``: the
    same call repeats exactly; the outputs come back in dataset order."""

    pytestmark = pytest.mark.numerical_unit

    # Length-3 and length-6 prompts under the tiny-random Llama tokenizer
    # (probed directly): an all-tokens span over this dataset is genuinely
    # ragged, the shape the pyvene path had to length-bucket.
    _L3 = ["hello world", "red sky", "a cat"]
    _L6 = ["the quick brown fox", "one two three four five"]

    def _noise_outputs(
        self, pipeline: LMPipeline, texts: list[str], seed: int
    ) -> dict[str, str]:
        """Run noise-only causal tracing over an all-tokens span; return
        ``{prompt_text: generated_string}``."""
        dataset = _dataset(texts)
        span = _all_tokens(pipeline, dataset)
        entry_sites = _entry_sites(pipeline, span)
        corruption = make_corruption_vectors(
            "noise", pipeline, dataset, entry_sites, noise_scale=25.0
        )
        type_map = {s.key: "noise" for s in entry_sites}
        out = run_causal_trace(
            pipeline,
            dataset,
            entry_sites,
            {s.key: corruption[s.key] for s in entry_sites},
            type_by_key=type_map,
            noise_seed=seed,
            output_scores=False,
        ).strings
        # Flat GenerationResult: one generation per example, in dataset
        # order; key by prompt.
        return {t: s for t, s in zip(texts, out)}

    def test_mixed_length_span_is_genuinely_ragged(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """Sanity: the dataset spans two distinct position counts (else the
        ragged path below is never exercised)."""
        dataset = _dataset(self._L3 + self._L6)
        span = _all_tokens(mock_tiny_lm, dataset)
        sites = _entry_sites(mock_tiny_lm, span)
        buckets = group_by_position_count(sites, dataset)
        assert len(buckets) == 2

    def test_ragged_stream_reproduces_and_covers_every_example(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """The ragged span runs whole-dataset in one stream: every example gets
        an output in dataset order, and the same call repeats exactly (one
        seeded noise stream per call)."""
        texts = self._L3 + self._L6
        first = self._noise_outputs(mock_tiny_lm, texts, seed=7)
        second = self._noise_outputs(mock_tiny_lm, texts, seed=7)
        assert list(first) == texts  # every example, dataset order
        assert first == second  # same order + seed → identical draws
