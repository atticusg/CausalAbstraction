"""Run component ablation and score the behavioral accuracy drop.

Ablation is a ``replace`` intervention that overwrites a component's
output with a fixed reference vector (see :mod:`reference_vectors`), then
generates and grades. This reuses the head-capable steering primitive
(``run_steering_interventions(mode="replace")``) rather than a parallel
forward-hook system, so subspace/feature-index ablation comes free later.

Entry points, mirroring ``methods/interchange/layer_scan.py``:

* :func:`run_ablation` — ablate one flat site list, return raw generation outputs.
* :func:`run_ablation_scan` — ablate a grid of cells, return accuracy per cell.
* :func:`run_ablation_combo` — ablate a site set jointly, return one accuracy.
* :func:`run_ablation_scan_multi` / :func:`run_ablation_combo_multi` — as above
  but score each cell under *several* metrics from one set of generations (the
  single-metric variants are thin wrappers over these).

Sites are :class:`~causalab.neural.specs.SiteSpec` values (WU4, #506); grids
are the WU2 :data:`~causalab.neural.activations.site_grids.SiteGrid` shape
(cell key → single-group nested spec lists). The reference vectors carry the
zero-vs-mean choice; there is deliberately no ``mode`` flag here (the op is
always ``replace``).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from torch import Tensor
from tqdm import tqdm

from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.metric import InterchangeMetric, score_intervention_outputs
from causalab.methods.steer.steer import run_steering_interventions
from causalab.neural.activations.site_grids import SiteGrid
from causalab.neural.pipeline import GenerationResult, LMPipeline
from causalab.neural.specs import SiteSpec

logger = logging.getLogger(__name__)


def run_ablation(
    pipeline: LMPipeline,
    dataset: list[CounterfactualExample],
    sites: Sequence[SiteSpec],
    vectors: dict[str, Tensor],
    *,
    batch_size: int = 16,
    output_scores: bool | int = True,
    gen_kwargs: dict[str, Any] | None = None,
) -> GenerationResult:
    """Ablate ``sites`` over ``dataset`` and return the flat
    :class:`~causalab.neural.pipeline.GenerationResult` (EU5b, #487).

    ``vectors`` maps each ``spec.key`` in ``sites`` to its reference vector
    (zeros for zero-ablation, corpus mean for mean-ablation). Ragged spans (an
    all-position span over variable-length examples) batch natively on the
    nnsight engine (PL3, #405), so the outputs come back in dataset order with
    no length-bucketing or reassembly. ``gen_kwargs`` are extra HF ``generate``
    kwargs forwarded through to the engine (e.g. ``{"min_new_tokens": N}`` —
    the escape hatch its ragged-scores refusal names).
    """
    return run_steering_interventions(
        pipeline,
        dataset,
        sites,
        vectors,
        batch_size=batch_size,
        output_scores=output_scores,
        mode="replace",
        gen_kwargs=gen_kwargs,
    )


def _score_all_metrics(
    results: dict[tuple[Any, ...], GenerationResult],
    dataset: list[CounterfactualExample],
    metrics: dict[str, InterchangeMetric],
    causal_model: CausalModel | None,
    original_outputs: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """Score one cell's result (a single-key ``results``) under every metric.

    Returns ``{metric_name: score}`` for the lone key in ``results``. Scoring
    one cell at a time (rather than accumulating every cell's raw outputs and
    scoring at the end) keeps only a single cell's logits alive — the reason
    ``output_scores=True`` is affordable here even with full-vocab scores.
    """
    (key,) = results
    return {
        name: score_intervention_outputs(
            results=results,
            dataset=dataset,
            metric=metric,
            causal_model=causal_model,
            original_outputs=original_outputs,
        )[key]
        for name, metric in metrics.items()
    }


def run_ablation_scan_multi(
    grid: SiteGrid,
    dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    vectors: dict[str, Tensor],
    *,
    metrics: dict[str, InterchangeMetric],
    batch_size: int = 16,
    output_scores: bool | int = True,
    causal_model: CausalModel | None = None,
    original_outputs: list[dict[str, Any]] | None = None,
) -> dict[tuple[Any, ...], dict[str, float]]:
    """Ablate each grid cell and score it under *several* metrics at once.

    Like :func:`run_ablation_scan` but generates once per cell and scores that
    cell under every metric in ``metrics`` (e.g. a behavioral-match drop *and* a
    predicted-token logit drop), so the expensive generation is shared. Returns
    ``{key: {metric_name: score}}``.

    ``vectors`` is a single flat ``{spec.key: tensor}`` dict covering every site
    across all cells (spec keys are globally unique); the per-cell slice is
    passed through to ``run_ablation``. Each cell is scored as soon as it is
    generated (see :func:`_score_all_metrics`), so memory stays bounded to one
    cell even when ``output_scores`` keeps full-vocab logits.
    """
    results: dict[tuple[Any, ...], dict[str, float]] = {}
    for key, groups in tqdm(grid.items(), desc="Ablation scan", total=len(grid)):
        sites = [spec for group in groups for spec in group]
        cell_vectors = {spec.key: vectors[spec.key] for spec in sites}
        outputs = run_ablation(
            pipeline,
            dataset,
            sites,
            cell_vectors,
            batch_size=batch_size,
            output_scores=output_scores,
        )
        results[key] = _score_all_metrics(
            {key: outputs}, dataset, metrics, causal_model, original_outputs
        )
    return results


def run_ablation_combo_multi(
    sites: Sequence[SiteSpec],
    dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    vectors: dict[str, Tensor],
    *,
    metrics: dict[str, InterchangeMetric],
    batch_size: int = 16,
    output_scores: bool | int = True,
    causal_model: CausalModel | None = None,
    original_outputs: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Ablate a set of sites jointly and score it under several metrics at once.

    Like :func:`run_ablation_combo` but returns ``{metric_name: score}`` from a
    single joint forward pass.
    """
    # Slice to just these sites: run_steering_interventions errors on vectors for
    # sites outside the set, so a shared grid-wide dict must be narrowed here.
    combo_vectors = {spec.key: vectors[spec.key] for spec in sites}
    outputs = run_ablation(
        pipeline,
        dataset,
        sites,
        combo_vectors,
        batch_size=batch_size,
        output_scores=output_scores,
    )
    return _score_all_metrics(
        {("combo",): outputs}, dataset, metrics, causal_model, original_outputs
    )


def run_ablation_scan(
    grid: SiteGrid,
    dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    vectors: dict[str, Tensor],
    *,
    metric: InterchangeMetric,
    batch_size: int = 16,
    output_scores: bool | int = True,
    causal_model: CausalModel | None = None,
    original_outputs: list[dict[str, Any]] | None = None,
) -> dict[tuple[Any, ...], float]:
    """Ablate each grid cell and score with a single ``metric``.

    Thin single-metric wrapper over :func:`run_ablation_scan_multi`. Returns
    ``{key: ablated_accuracy}`` — the caller computes
    ``drop = base_accuracy - ablated_accuracy``.

    Structurally mirrors ``methods/interchange/layer_scan.py::run_layer_scan``.
    """
    multi = run_ablation_scan_multi(
        grid,
        dataset,
        pipeline,
        vectors,
        metrics={"_": metric},
        batch_size=batch_size,
        output_scores=output_scores,
        causal_model=causal_model,
        original_outputs=original_outputs,
    )
    return {key: scores["_"] for key, scores in multi.items()}


def run_ablation_combo(
    sites: Sequence[SiteSpec],
    dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    vectors: dict[str, Tensor],
    *,
    metric: InterchangeMetric,
    batch_size: int = 16,
    output_scores: bool | int = True,
    causal_model: CausalModel | None = None,
    original_outputs: list[dict[str, Any]] | None = None,
) -> float:
    """Ablate a set of sites jointly and return one accuracy.

    Thin single-metric wrapper over :func:`run_ablation_combo_multi`. ``sites``
    are ablated together in a single forward pass; ``vectors`` supplies a
    reference vector per ``spec.key``. Returns the joint ablated accuracy (the
    caller computes the drop).
    """
    return run_ablation_combo_multi(
        sites,
        dataset,
        pipeline,
        vectors,
        metrics={"_": metric},
        batch_size=batch_size,
        output_scores=output_scores,
        causal_model=causal_model,
        original_outputs=original_outputs,
    )["_"]
