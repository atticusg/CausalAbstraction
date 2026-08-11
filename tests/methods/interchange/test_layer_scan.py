"""``methods/interchange/layer_scan`` on the flat ``GenerationResult`` (EU5b, #487).

Two surfaces:

* ``run_layer_scan`` — wrapper results flow into ``score_intervention_outputs``
  as flat :class:`~causalab.neural.pipeline.GenerationResult` values (smoke on
  the real tiny-random engine), and the statically-known top-k poison
  combination refuses before any generation work.
* ``run_centroid_layer_scan`` — the retired per-batch score loop collapsed to
  flat indexing. The ragged-scores decision is pinned here: when the engine's
  internal batches early-EOS at unequal step counts, the engine's loud
  ``ValueError`` PROPAGATES (no caller-side skip) — the legacy loop silently
  dropped short batches (silent partial data), which EU5a/EU5b deliberately
  refuse.

Tiers (per docs/TESTS.md methods/ → ``numerical_unit`` + ``property``, one
class per tier): contract/refusal tests are ``unit``, the hand-computed
flat-indexing value pin is ``numerical_unit``, and the dataset-permutation
invariance of the scan runs as ``property`` on the real tiny-random engine.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.interchange.layer_scan import (
    run_centroid_layer_scan,
    run_layer_scan,
)
from causalab.methods.metric import (
    InterchangeMetric,
    _logits_to_class_probs,  # pyright: ignore[reportPrivateUsage]
)
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.pipeline import GenerationResult, LMPipeline
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec
from causalab.neural.token_positions import TokenPosition


def _trace(text: str) -> CausalTrace:
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _cell(pipeline: LMPipeline, layer: int = 0) -> list[list[SiteSpec]]:
    hidden = pipeline.model.config.hidden_size
    tp = TokenPosition(lambda _x: [0], pipeline, id="first_token")
    spec = SiteSpec(
        fsite=FeaturizedSite(Site("block_input", layer)),
        positions=tp,
        key=f"residual.L{layer}.first_token",
        width=hidden,
    )
    return [[spec]]


def _cf_dataset(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {"input": _trace(t), "counterfactual_inputs": [_trace(c)]} for t, c in pairs
    ]


# --------------------------------------------------------------------------- #
#  Centroid-scan harness — shared by the unit and numerical_unit classes       #
# --------------------------------------------------------------------------- #
_VOCAB = 11
_TOKENS = [[5], [7]]  # two single-token classes
_N_STEER = 4  # steer examples


def _run_centroid(
    pipeline: LMPipeline,
    monkeypatch: pytest.MonkeyPatch,
    engine_result: GenerationResult | Exception,
):
    """Drive ``run_centroid_layer_scan`` over a hand-fed engine result (the
    engine itself is exercised by the ``weekdays_locate_centroid`` e2e
    golden); returns ``(result, engine calls, ref_dists)``."""
    cell = _cell(pipeline)
    hidden = pipeline.model.config.hidden_size
    dataset = _cf_dataset([(f"in {i}", f"cf {i}") for i in range(_N_STEER)])
    ref_dists = torch.tensor([[0.9, 0.1], [0.2, 0.8]])

    calls: list[dict[str, Any]] = []

    def fake_engine(pipe, examples, groups, batch_size, output_scores):
        calls.append({"n_examples": len(examples), "batch_size": batch_size})
        if isinstance(engine_result, Exception):
            raise engine_result
        return engine_result

    monkeypatch.setattr(
        "causalab.neural.dataset.run_intervened_generation", fake_engine
    )
    result = run_centroid_layer_scan(
        grid={(0, "first_token"): cell},
        dataset=dataset,
        pipeline=pipeline,
        batch_size=2,
        score_token_ids=_TOKENS,
        n_classes=2,
        example_to_class=lambda ex: 0,  # all mass in class 0's centroid
        ref_dists=ref_dists,
        precomputed_features={(0, "first_token"): torch.randn(_N_STEER, hidden)},
        comparison_fn=lambda ref, probs: (ref - probs).abs().sum(dim=-1),
        return_patched_dists=True,
    )
    return result, calls, ref_dists


class TestRunLayerScan:
    pytestmark = pytest.mark.unit

    def test_scores_one_finite_value_per_key(self, mock_tiny_lm: LMPipeline) -> None:
        """End-to-end on the real engine: wrapper GenerationResults flow into
        ``score_intervention_outputs`` and come back as one float per key."""
        targets = {
            (0, "first_token"): _cell(mock_tiny_lm, layer=0),
            (1, "first_token"): _cell(mock_tiny_lm, layer=1),
        }
        dataset = _cf_dataset([("alpha beta", "gamma delta"), ("one two", "three")])

        seen: list[dict[str, Any]] = []

        def fn(intervention_output, expected, original):
            seen.append(intervention_output)
            return 1.0

        scores = run_layer_scan(
            grid=targets,
            dataset=dataset,
            pipeline=mock_tiny_lm,
            batch_size=1,  # 2 internal batches — must not leak into scoring
            metric=InterchangeMetric(fn=fn, needs_causal_expected=False),
            output_scores=True,
        )
        assert set(scores.keys()) == set(targets.keys())
        assert all(v == 1.0 for v in scores.values())
        # The metric saw flat per-step scores spanning ALL examples, plus the
        # per-example index — the metric protocol over the flat result.
        vocab = mock_tiny_lm.model.config.vocab_size
        assert len(seen) == len(targets) * len(dataset)
        for out in seen:
            assert out["scores"][0].shape == (len(dataset), vocab)
            assert "example_idx" in out

    def test_top_k_output_scores_refused_before_any_generation(
        self, mock_tiny_lm: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``output_scores=<int>`` is a statically-known poison combination:
        ``score_intervention_outputs`` refuses top-k-compressed scores
        unconditionally, so the scan refuses it up front — BEFORE the loop
        runs a single generation (review #492 F6). The monkeypatched engine
        explodes if reached, proving no generation work happened."""

        def explode(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("generation ran before the fail-fast guard")

        monkeypatch.setattr(
            "causalab.methods.interchange.layer_scan.run_interchange_interventions",
            explode,
        )
        with pytest.raises(ValueError, match="output_scores=True \\(not an int\\)"):
            run_layer_scan(
                grid={(0, "first_token"): _cell(mock_tiny_lm)},
                dataset=_cf_dataset([("a", "b")]),
                pipeline=mock_tiny_lm,
                batch_size=1,
                metric=InterchangeMetric(
                    fn=lambda *a: 1.0, needs_causal_expected=False
                ),
                output_scores=5,
            )


class TestLayerScanProperty:
    """Dataset-order properties of the scan on the real tiny-random engine."""

    pytestmark = pytest.mark.property

    def test_dataset_permutation_permutes_outputs_correspondingly(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """Permuting the dataset (with its aligned ``original_outputs``)
        permutes the per-example outputs correspondingly: each example's
        decoded output depends only on its own (input, counterfactual) pair,
        and scoring consumes the flat result in dataset order — so the
        example→output mapping and the (permutation-invariant) mean score
        both survive a reshuffle. ``batch_size=1`` keeps every example's
        forward identical across orderings (no cross-example padding)."""
        pairs = [
            ("alpha beta", "gamma delta"),
            ("one two three", "four"),
            ("hello world", "blue sky"),
        ]
        dataset = _cf_dataset(pairs)
        originals = [{"tag": i} for i in range(len(pairs))]
        cells = {(0, "first_token"): _cell(mock_tiny_lm)}

        def scan(ds, origs):
            seen: dict[int, str] = {}

            def fn(intervention_output, expected, original):
                seen[original["tag"]] = intervention_output["string"]
                return float(original["tag"])

            scores = run_layer_scan(
                grid=cells,
                dataset=ds,
                pipeline=mock_tiny_lm,
                batch_size=1,
                metric=InterchangeMetric(
                    fn=fn,
                    needs_causal_expected=False,
                    needs_original_output=True,
                ),
                output_scores=False,
                original_outputs=origs,
            )
            return scores, seen

        base_scores, base_seen = scan(dataset, originals)
        perm = [2, 0, 1]
        perm_scores, perm_seen = scan(
            [dataset[i] for i in perm], [originals[i] for i in perm]
        )
        assert perm_seen == base_seen  # example → output survives the shuffle
        assert perm_scores == base_scores  # the mean is permutation-invariant
        assert len(base_seen) == len(pairs)


class TestCentroidLayerScanFlatLoop:
    """Contract behaviour of the flat-indexed centroid loop over a hand-fed
    engine result."""

    pytestmark = pytest.mark.unit

    def test_ragged_engine_refusal_propagates(
        self, mock_tiny_lm: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate EU5b ragged-scores decision: the engine's loud
        refusal (unequal early-EOS step counts across internal batches)
        surfaces to the caller with its escape hatches — the scan does NOT
        silently skip short batches the way the legacy per-batch loop did."""
        ragged = ValueError(
            "cannot flatten per-step scores: the internal batches generated "
            "unequal step counts [1, 3] (early EOS stops a batch when all ITS "
            "rows finish). Use a single batch (batch_size >= len(dataset)) or "
            "force a fixed length (e.g. min_new_tokens=max_new_tokens)."
        )
        with pytest.raises(ValueError, match="cannot flatten per-step scores"):
            _run_centroid(mock_tiny_lm, monkeypatch, ragged)

    def test_zero_step_result_scores_nan(
        self, mock_tiny_lm: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A uniformly step-less result (no step at score_token_index for the
        WHOLE dataset) yields nan — the flat-contract analogue of the legacy
        all-batches-short skip."""
        engine_result = GenerationResult(
            sequences=torch.zeros((_N_STEER, 1), dtype=torch.long),
            strings=["x"] * _N_STEER,
            scores=[],
        )
        (scores, _dists), _calls, _refs = _run_centroid(
            mock_tiny_lm, monkeypatch, engine_result
        )
        value = scores[(0, "first_token")]
        assert value != value  # nan


class TestCentroidLayerScanNumerical:
    """The per-batch score loop collapsed to flat indexing — numerics pinned
    against a hand-fed engine result (the engine itself is exercised by the
    ``weekdays_locate_centroid`` e2e golden)."""

    pytestmark = pytest.mark.numerical_unit

    def test_flat_scores_cover_every_example(
        self, mock_tiny_lm: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch.manual_seed(0)
        step0 = torch.randn(_N_STEER, _VOCAB)
        engine_result = GenerationResult(
            sequences=torch.zeros((_N_STEER, 1), dtype=torch.long),
            strings=["x"] * _N_STEER,
            scores=[step0],
        )
        (scores, dists), calls, ref_dists = _run_centroid(
            mock_tiny_lm, monkeypatch, engine_result
        )

        # Only class 0 has examples → exactly one engine call, all N examples.
        assert calls == [{"n_examples": _N_STEER, "batch_size": 2}]

        # Hand-computed expectation over the SAME flat step: every example
        # contributes (no per-batch skip path left).
        probs = _logits_to_class_probs([step0], _TOKENS).cpu()
        expected = (
            (ref_dists[0].unsqueeze(0).expand(_N_STEER, -1) - probs)
            .abs()
            .sum(dim=-1)
            .mean()
            .item()
        )
        assert scores[(0, "first_token")] == pytest.approx(expected)

        # Patched dists average over all N examples in one pass.
        probs_fvs = _logits_to_class_probs(
            [step0], _TOKENS, full_vocab_softmax=True
        ).cpu()
        torch.testing.assert_close(dists[(0, "first_token")][0], probs_fvs.mean(dim=0))
