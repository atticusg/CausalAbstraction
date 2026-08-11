"""Tests for the component-ablation method (``causalab.methods.ablation``).

The tiny-random Llama stub has no task behaviour, so we ground the *behavioural*
assertions on a "match-the-base-output" metric: each ablated generation is
graded against the un-ablated generation for that example. Base accuracy is then
1.0 by construction and ``drop = 1 - match_fraction`` measures how much the
ablation perturbed behaviour — which is exactly what the real analysis measures,
just with the model's own output standing in for a ground-truth label. This
keeps the tests deterministic while exercising the real run/scan/combo/scoring
and length-bucketing paths.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.ablation import (
    make_mean_vectors,
    make_zero_vectors,
    run_ablation_combo,
    run_ablation_combo_multi,
    run_ablation_scan,
    run_ablation_scan_multi,
)
from causalab.methods.metric import InterchangeMetric, compute_base_outputs
from causalab.neural.activations.site_grids import (
    build_attention_head_sites,
    build_attention_output_sites,
    build_mlp_sites,
)
from causalab.neural.specs import SiteSpec
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import (
    TokenPosition,
    get_all_tokens,
    get_last_token_index,
)

# Tiers are declared per class (one class per tier, per docs/TESTS.md's
# methods/ → property + numerical_unit requirement): shape/dtype contracts and
# path-equivalence wiring are `property`; the mean-aggregation value pin is
# `numerical_unit`.


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
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


def _match_base_metric(pipeline: LMPipeline, dataset: list[dict[str, Any]]):
    """An ``InterchangeMetric`` that grades an ablated output against the base.

    Returns ``(metric, base_outputs)``. ``metric.fn`` compares the ablated
    string to the base (original) string for the same example, so the scan
    score is the fraction of examples whose behaviour the ablation left intact.
    """
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


def _flat(groups: list[list[SiteSpec]]) -> list[SiteSpec]:
    """Flatten a grid value's nested spec groups (single-group by WU2)."""
    return [spec for group in groups for spec in group]


def _last_token(pipeline: LMPipeline) -> TokenPosition:
    return TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline),
        pipeline,
        id="last_token",
    )


def _logit_diff_metric() -> InterchangeMetric:
    """Scores-based metric: drop in the base-predicted token's logit.

    Exercises the full-vocab ``output_scores=True`` path — it reads logits from
    both the ablated run (``intervention_output["scores"]``) and the base run
    (``original["scores"]``), the way the knockout analysis's ``logit_diff`` does.
    """

    def fn(
        intervention_output: dict[str, Any],
        expected: dict[str, Any],
        original: dict[str, Any],
    ) -> float:
        ablated_scores = intervention_output["scores"]
        base_scores = original["scores"]
        idx = intervention_output["example_idx"]
        base_logits = base_scores[0]
        ablated_logits = ablated_scores[0][idx]
        pred = int(base_logits.argmax())
        return float(base_logits[pred] - ablated_logits[pred])

    return InterchangeMetric(
        fn=fn,
        needs_causal_expected=False,
        needs_original_output=True,
        needs_scores=True,
    )


# --------------------------------------------------------------------------- #
#  Reference-vector shapes                                                    #
# --------------------------------------------------------------------------- #
class TestReferenceVectorShapes:
    pytestmark = pytest.mark.property

    def test_zero_vectors_head_shape(self, mock_tiny_lm: LMPipeline) -> None:
        head_dim = mock_tiny_lm.model.config.hidden_size // (
            mock_tiny_lm.model.config.num_attention_heads
        )
        sites = _flat(
            build_attention_head_sites(
                mock_tiny_lm, [0], [0], _last_token(mock_tiny_lm)
            )[(0, 0)]
        )

        zeros = make_zero_vectors(sites)

        (unit_id,) = [s.key for s in sites]
        assert zeros[unit_id].shape == (head_dim,)
        assert torch.count_nonzero(zeros[unit_id]) == 0

    def test_mean_vectors_head_shape_and_finite(self, mock_tiny_lm: LMPipeline) -> None:
        head_dim = mock_tiny_lm.model.config.hidden_size // (
            mock_tiny_lm.model.config.num_attention_heads
        )
        sites = _flat(
            build_attention_head_sites(
                mock_tiny_lm, [0], [0], _last_token(mock_tiny_lm)
            )[(0, 0)]
        )

        means = make_mean_vectors(
            mock_tiny_lm, _equal_length_dataset(), sites, batch_size=2
        )

        (unit_id,) = [s.key for s in sites]
        assert means[unit_id].shape == (head_dim,)
        assert torch.isfinite(means[unit_id]).all()

    def test_mean_vectors_mlp_shape(self, mock_tiny_lm: LMPipeline) -> None:
        hidden = mock_tiny_lm.model.config.hidden_size
        sites = _flat(
            build_mlp_sites(mock_tiny_lm, [0], [_last_token(mock_tiny_lm)])[
                (0, "last_token")
            ]
        )

        means = make_mean_vectors(
            mock_tiny_lm, _equal_length_dataset(), sites, batch_size=2
        )

        (unit_id,) = [s.key for s in sites]
        assert means[unit_id].shape == (hidden,)

    def test_mean_vectors_attention_output_shape(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """Whole-attention-sublayer output is ``hidden_size``-wide (merged across
        heads), so its mean reference vector must match — not ``head_dim``."""
        hidden = mock_tiny_lm.model.config.hidden_size
        sites = _flat(
            build_attention_output_sites(
                mock_tiny_lm, [0], [_last_token(mock_tiny_lm)]
            )[(0, "last_token")]
        )

        means = make_mean_vectors(
            mock_tiny_lm, _equal_length_dataset(), sites, batch_size=2
        )

        (unit_id,) = [s.key for s in sites]
        assert means[unit_id].shape == (hidden,)
        assert torch.isfinite(means[unit_id]).all()

    def test_mean_vectors_bucketed_over_variable_length_all_tokens(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """All-tokens span over a variable-length dataset must collect via
        length-bucketing (a single ragged batch would raise in pyvene's gather)."""
        head_dim = mock_tiny_lm.model.config.hidden_size // (
            mock_tiny_lm.model.config.num_attention_heads
        )
        dataset = _dataset(["hello world foo", "a b", "one two three"])
        span = get_all_tokens(dataset[0]["input"], mock_tiny_lm)
        sites = _flat(build_attention_head_sites(mock_tiny_lm, [0], [0], span)[(0, 0)])

        means = make_mean_vectors(mock_tiny_lm, dataset, sites, batch_size=8)

        (unit_id,) = [s.key for s in sites]
        assert means[unit_id].shape == (head_dim,)
        assert torch.isfinite(means[unit_id]).all()


# --------------------------------------------------------------------------- #
#  Reference-vector numerics (pinned aggregation math)                        #
# --------------------------------------------------------------------------- #
class TestReferenceVectorNumerics:
    pytestmark = pytest.mark.numerical_unit

    def test_mean_vectors_multi_position_mlp_matches_per_position(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """A multi-position MLP span mean must equal averaging its per-position
        means. ``pos`` units can't be collected over >1 position in one pyvene
        pass (it mangles the 3-D gather), so ``make_mean_vectors`` slices the span
        into single-position collects; this pins that the slice-and-average is
        numerically the same as the (working) single-position path — a value pin
        on the aggregation math, so it lives in the numerical_unit tier."""
        dataset = _equal_length_dataset()  # equal length -> one bucket, fixed n_pos
        hidden = mock_tiny_lm.model.config.hidden_size
        span = get_all_tokens(dataset[0]["input"], mock_tiny_lm)
        sites = _flat(build_mlp_sites(mock_tiny_lm, [0], [span])[(0, "all_tokens")])
        (unit_id,) = [s.key for s in sites]

        multi = make_mean_vectors(mock_tiny_lm, dataset, sites, batch_size=2)
        assert multi[unit_id].shape == (hidden,)
        assert torch.isfinite(multi[unit_id]).all()

        # Reconstruct the same mean from independent single-position collects.
        n_pos = len(span.index(dataset[0]["input"]))
        per_pos = []
        for k in range(n_pos):
            tp = TokenPosition(lambda inp, _k=k: [_k], mock_tiny_lm, id=f"p{k}")
            single = _flat(build_mlp_sites(mock_tiny_lm, [0], [tp])[(0, f"p{k}")])
            (sid,) = [s.key for s in single]
            per_pos.append(make_mean_vectors(mock_tiny_lm, dataset, single, 2)[sid])
        expected = torch.stack(per_pos).mean(dim=0)

        torch.testing.assert_close(multi[unit_id], expected)

    def test_mean_vectors_multi_position_attention_output_matches_per_position(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """Same multi-position slice-and-average guarantee for the new
        ``attention_output`` unit. ``AttentionOutput`` is a ``pos`` unit, so a
        >1-token span hits ``make_mean_vectors``' single-position-at-a-time
        workaround for the pyvene 3-D-gather bug; pin that the workaround is
        numerically equivalent to per-position means for this unit type too (the
        latent risk flagged for span ablations)."""
        dataset = _equal_length_dataset()  # equal length -> one bucket, fixed n_pos
        hidden = mock_tiny_lm.model.config.hidden_size
        span = get_all_tokens(dataset[0]["input"], mock_tiny_lm)
        sites = _flat(
            build_attention_output_sites(mock_tiny_lm, [0], [span])[(0, "all_tokens")]
        )
        (unit_id,) = [s.key for s in sites]

        multi = make_mean_vectors(mock_tiny_lm, dataset, sites, batch_size=2)
        assert multi[unit_id].shape == (hidden,)
        assert torch.isfinite(multi[unit_id]).all()

        n_pos = len(span.index(dataset[0]["input"]))
        per_pos = []
        for k in range(n_pos):
            tp = TokenPosition(lambda inp, _k=k: [_k], mock_tiny_lm, id=f"p{k}")
            single = _flat(
                build_attention_output_sites(mock_tiny_lm, [0], [tp])[(0, f"p{k}")]
            )
            (sid,) = [s.key for s in single]
            per_pos.append(make_mean_vectors(mock_tiny_lm, dataset, single, 2)[sid])
        expected = torch.stack(per_pos).mean(dim=0)

        torch.testing.assert_close(multi[unit_id], expected)


# --------------------------------------------------------------------------- #
#  Scan + combo behaviour                                                     #
# --------------------------------------------------------------------------- #
class TestAblationScanAndCombo:
    pytestmark = pytest.mark.property

    def test_scan_returns_one_accuracy_per_cell(self, mock_tiny_lm: LMPipeline) -> None:
        dataset = _equal_length_dataset()
        layers, heads = [0, 1], [0, 1, 2, 3]
        targets = build_attention_head_sites(
            mock_tiny_lm, layers, heads, _last_token(mock_tiny_lm)
        )
        vectors = {
            key: vec
            for groups in targets.values()
            for key, vec in make_zero_vectors(_flat(groups)).items()
        }
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        scores = run_ablation_scan(
            targets,
            dataset,
            mock_tiny_lm,
            vectors,
            metric=metric,
            batch_size=4,
            original_outputs=base_outputs,
        )

        assert set(scores.keys()) == set(targets.keys())
        assert len(scores) == len(layers) * len(heads)
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_combo_of_single_unit_matches_scan_cell(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """A one-unit combo must score identically to that cell in a scan —
        the combo and scan paths share the same intervention + scoring."""
        dataset = _equal_length_dataset()
        sites = _flat(
            build_attention_head_sites(
                mock_tiny_lm, [0], [0], _last_token(mock_tiny_lm)
            )[(0, 0)]
        )
        vectors = make_zero_vectors(sites)
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        scan_acc = run_ablation_scan(
            {(0, 0): [sites]},
            dataset,
            mock_tiny_lm,
            vectors,
            metric=metric,
            batch_size=4,
            original_outputs=base_outputs,
        )[(0, 0)]
        combo_acc = run_ablation_combo(
            sites,
            dataset,
            mock_tiny_lm,
            vectors,
            metric=metric,
            batch_size=4,
            original_outputs=base_outputs,
        )

        assert combo_acc == pytest.approx(scan_acc)

    def test_combo_of_all_cells_changes_behavior(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """Ablating every head jointly in one forward pass runs and perturbs
        behaviour (drop > 0).

        We assert behaviour *changes*, not that a joint drop dominates the single
        drops: ``FeatureReplaceIntervention`` preserves each component's
        orthogonal error term, so zero-ablation is not fully destructive and the
        drop is not monotone in the set of ablated units — especially on the
        random-weight stub, where one head can flip more outputs than all heads
        together. The single-unit↔scan consistency test below pins the path's
        correctness; this one pins that a joint ablation is wired up and active.
        """
        dataset = _equal_length_dataset()
        layers, heads = [0, 1], [0, 1, 2, 3]
        last = _last_token(mock_tiny_lm)
        targets = build_attention_head_sites(mock_tiny_lm, layers, heads, last)
        vectors = {
            key: vec
            for groups in targets.values()
            for key, vec in make_zero_vectors(_flat(groups)).items()
        }
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        # Every (layer, head) site ablated jointly.
        all_sites = [s for groups in targets.values() for s in _flat(groups)]
        combo_acc = run_ablation_combo(
            all_sites,
            dataset,
            mock_tiny_lm,
            vectors,
            metric=metric,
            batch_size=4,
            original_outputs=base_outputs,
        )

        assert 0.0 <= combo_acc < 1.0  # behaviour changed for at least one example

    def test_mlp_scan_returns_one_accuracy_per_cell(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """MLP grid scan: one finite accuracy per ``(layer, pos)`` cell.

        Mirrors ``test_scan_returns_one_accuracy_per_cell`` for ``pos``-unit MLP
        targets, so the scan path is covered for both component types — heads
        gather a 4-D ``(b, h, s, d)`` source, MLP units a 3-D ``(b, n_pos, d)``.
        """
        dataset = _equal_length_dataset()
        layers = [0, 1]
        last = _last_token(mock_tiny_lm)
        targets = build_mlp_sites(mock_tiny_lm, layers, [last])
        vectors = {
            key: vec
            for groups in targets.values()
            for key, vec in make_zero_vectors(_flat(groups)).items()
        }
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        scores = run_ablation_scan(
            targets,
            dataset,
            mock_tiny_lm,
            vectors,
            metric=metric,
            batch_size=4,
            original_outputs=base_outputs,
        )

        assert set(scores.keys()) == set(targets.keys())
        assert len(scores) == len(layers)
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_mlp_combo_of_single_unit_matches_scan_cell(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """A one-unit MLP combo scores identically to that cell in a scan —
        the combo and scan paths share the same intervention + scoring."""
        dataset = _equal_length_dataset()
        last = _last_token(mock_tiny_lm)
        sites = _flat(build_mlp_sites(mock_tiny_lm, [0], [last])[(0, "last_token")])
        vectors = make_zero_vectors(sites)
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        scan_acc = run_ablation_scan(
            {(0, "last_token"): [sites]},
            dataset,
            mock_tiny_lm,
            vectors,
            metric=metric,
            batch_size=4,
            original_outputs=base_outputs,
        )[(0, "last_token")]
        combo_acc = run_ablation_combo(
            sites,
            dataset,
            mock_tiny_lm,
            vectors,
            metric=metric,
            batch_size=4,
            original_outputs=base_outputs,
        )

        assert combo_acc == pytest.approx(scan_acc)

    def test_attention_output_scan_returns_one_accuracy_per_cell(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """Whole-attention-sublayer (``attention_output``) grid scan: one finite
        accuracy per ``(layer, pos)`` cell. Confirms the ``replace`` path runs
        through the scan for this unit (the field report's "path-patching only"
        caveat was overly conservative)."""
        dataset = _equal_length_dataset()
        layers = [0, 1]
        targets = build_attention_output_sites(
            mock_tiny_lm, layers, [_last_token(mock_tiny_lm)]
        )
        vectors = {
            key: vec
            for groups in targets.values()
            for key, vec in make_zero_vectors(_flat(groups)).items()
        }
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        scores = run_ablation_scan(
            targets,
            dataset,
            mock_tiny_lm,
            vectors,
            metric=metric,
            batch_size=4,
            original_outputs=base_outputs,
        )

        assert set(scores.keys()) == set(targets.keys())
        assert len(scores) == len(layers)
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_attention_output_combo_of_single_unit_matches_scan_cell(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """A one-unit ``attention_output`` combo scores identically to that cell
        in a scan — the combo and scan paths share the same intervention +
        scoring."""
        dataset = _equal_length_dataset()
        last = _last_token(mock_tiny_lm)
        sites = _flat(
            build_attention_output_sites(mock_tiny_lm, [0], [last])[(0, "last_token")]
        )
        vectors = make_zero_vectors(sites)
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        scan_acc = run_ablation_scan(
            {(0, "last_token"): [sites]},
            dataset,
            mock_tiny_lm,
            vectors,
            metric=metric,
            batch_size=4,
            original_outputs=base_outputs,
        )[(0, "last_token")]
        combo_acc = run_ablation_combo(
            sites,
            dataset,
            mock_tiny_lm,
            vectors,
            metric=metric,
            batch_size=4,
            original_outputs=base_outputs,
        )

        assert combo_acc == pytest.approx(scan_acc)

    def test_scan_over_all_tokens_span_with_bucketing(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """End-to-end run + bucketing + stitch + scoring over a variable-length
        dataset and an all-tokens span returns a finite per-cell accuracy."""
        dataset = _dataset(["hello world foo", "a b", "one two three", "x y z w"])
        span = get_all_tokens(dataset[0]["input"], mock_tiny_lm)
        targets = build_attention_head_sites(mock_tiny_lm, [0], [0, 1], span)
        vectors = {
            key: vec
            for groups in targets.values()
            for key, vec in make_zero_vectors(_flat(groups)).items()
        }
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        scores = run_ablation_scan(
            targets,
            dataset,
            mock_tiny_lm,
            vectors,
            metric=metric,
            batch_size=8,
            original_outputs=base_outputs,
        )

        assert set(scores.keys()) == set(targets.keys())
        assert all(0.0 <= v <= 1.0 for v in scores.values())


# --------------------------------------------------------------------------- #
#  Multi-metric scan/combo (one generation, several metrics)                  #
# --------------------------------------------------------------------------- #
class TestMultiMetric:
    pytestmark = pytest.mark.property

    def test_scan_multi_matches_single_metric(self, mock_tiny_lm: LMPipeline) -> None:
        """Single-metric ``run_ablation_scan`` is a thin wrapper over the multi
        variant, so a one-entry ``metrics`` dict must reproduce it cell-for-cell."""
        dataset = _equal_length_dataset()
        layers, heads = [0, 1], [0, 1]
        targets = build_attention_head_sites(
            mock_tiny_lm, layers, heads, _last_token(mock_tiny_lm)
        )
        vectors = {
            key: vec
            for groups in targets.values()
            for key, vec in make_zero_vectors(_flat(groups)).items()
        }
        metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)

        single = run_ablation_scan(
            targets,
            dataset,
            mock_tiny_lm,
            vectors,
            metric=metric,
            batch_size=4,
            original_outputs=base_outputs,
        )
        multi = run_ablation_scan_multi(
            targets,
            dataset,
            mock_tiny_lm,
            vectors,
            metrics={"match": metric},
            batch_size=4,
            original_outputs=base_outputs,
        )

        assert set(multi.keys()) == set(single.keys())
        for key in single:
            assert multi[key]["match"] == pytest.approx(single[key])

    def test_scan_multi_scores_both_metrics_from_one_generation(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """Each cell carries every requested metric: a behavioural match fraction
        in ``[0, 1]`` and a finite (unbounded) predicted-token logit drop."""
        dataset = _equal_length_dataset()
        targets = build_attention_head_sites(
            mock_tiny_lm, [0, 1], [0, 1], _last_token(mock_tiny_lm)
        )
        vectors = {
            key: vec
            for groups in targets.values()
            for key, vec in make_zero_vectors(_flat(groups)).items()
        }
        match_metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)
        metrics = {"match_drop": match_metric, "logit_diff": _logit_diff_metric()}

        multi = run_ablation_scan_multi(
            targets,
            dataset,
            mock_tiny_lm,
            vectors,
            metrics=metrics,
            batch_size=4,
            original_outputs=base_outputs,
        )

        assert set(multi.keys()) == set(targets.keys())
        for cell in multi.values():
            assert set(cell.keys()) == {"match_drop", "logit_diff"}
            assert 0.0 <= cell["match_drop"] <= 1.0
            assert torch.isfinite(torch.tensor(cell["logit_diff"]))

    def test_combo_multi_matches_scan_multi_cell(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """A one-unit combo scores identically to that cell in a scan, for every
        metric — combo and scan share the same generate-then-score path."""
        dataset = _equal_length_dataset()
        sites = _flat(
            build_attention_head_sites(
                mock_tiny_lm, [0], [0], _last_token(mock_tiny_lm)
            )[(0, 0)]
        )
        vectors = make_zero_vectors(sites)
        match_metric, base_outputs = _match_base_metric(mock_tiny_lm, dataset)
        metrics = {"match_drop": match_metric, "logit_diff": _logit_diff_metric()}

        scan_cell = run_ablation_scan_multi(
            {(0, 0): [sites]},
            dataset,
            mock_tiny_lm,
            vectors,
            metrics=metrics,
            batch_size=4,
            original_outputs=base_outputs,
        )[(0, 0)]
        combo = run_ablation_combo_multi(
            sites,
            dataset,
            mock_tiny_lm,
            vectors,
            metrics=metrics,
            batch_size=4,
            original_outputs=base_outputs,
        )

        for name in metrics:
            assert combo[name] == pytest.approx(scan_cell[name])
