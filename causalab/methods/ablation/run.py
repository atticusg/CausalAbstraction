"""Run component ablation and score the behavioral accuracy drop.

Ablation is a ``replace`` intervention that overwrites a component's
output with a fixed reference vector (see :mod:`reference_vectors`), then
generates and grades. This reuses the head-capable steering primitive
(``run_steering_interventions(mode="replace")``) rather than a parallel
forward-hook system, so subspace/feature-index ablation comes free later.

Entry points, mirroring ``methods/interchange/layer_scan.py``:

* :func:`run_ablation` — ablate one target, return raw generation outputs.
* :func:`run_ablation_scan` — ablate a grid of targets, return accuracy per cell.
* :func:`run_ablation_combo` — ablate a unit set jointly, return one accuracy.
* :func:`run_ablation_scan_multi` / :func:`run_ablation_combo_multi` — as above
  but score each cell under *several* metrics from one set of generations (the
  single-metric variants are thin wrappers over these).

The reference vectors carry the zero-vs-mean choice; there is deliberately no
``mode`` flag here (the op is always ``replace``).
"""

from __future__ import annotations

import logging
from typing import Any

from torch import Tensor
from tqdm import tqdm

from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.metric import InterchangeMetric, score_intervention_outputs
from causalab.methods.steer.steer import run_steering_interventions
from causalab.neural.pipeline import LMPipeline
from causalab.neural.units import AtomicModelUnit, InterchangeTarget

logger = logging.getLogger(__name__)


def _flatten_run_outputs(
    outputs: dict[str, list[Any]],
) -> tuple[list[str], list[Any], list[list[Tensor]] | None]:
    """Flatten a ``run_steering_interventions`` result to per-example lists.

    The result is nested by batch (``string`` is a list of per-batch values,
    each itself a ``list[str]`` — or a bare ``str`` when the batch held a single
    example; ``scores`` is a list of per-batch ``[tok0(B,V), tok1(B,V), ...]``).
    Returns ``(strings, sequences, scores_per_example)`` aligned with the order
    the examples were fed in; ``scores_per_example`` is ``None`` when scores
    weren't requested.
    """
    strings: list[str] = []
    for batch in outputs.get("string", []):
        if isinstance(batch, list):
            strings.extend(batch)
        else:
            strings.append(batch)

    sequences: list[Any] = []
    for batch in outputs.get("sequences", []):
        for row in batch:
            sequences.append(row)

    raw_scores = outputs.get("scores")
    scores_per_example: list[list[Tensor]] | None = None
    if raw_scores:
        scores_per_example = []
        for batch in raw_scores:  # batch = [tok0(B,V), tok1(B,V), ...]
            batch_size = batch[0].shape[0]
            for example_idx in range(batch_size):
                scores_per_example.append([tok[example_idx] for tok in batch])

    return strings, sequences, scores_per_example


def run_ablation(
    pipeline: LMPipeline,
    dataset: list[CounterfactualExample],
    target: InterchangeTarget,
    vectors: dict[str, Tensor],
    *,
    batch_size: int = 16,
    output_scores: bool | int = True,
) -> dict[str, list[Any]]:
    """Ablate ``target`` over ``dataset`` and return raw generation outputs.

    ``vectors`` maps each unit id in ``target`` to its reference vector (zeros
    for zero-ablation, corpus mean for mean-ablation). The whole dataset runs as
    one stream in its original order, so the return value scores directly against
    ``dataset`` — an all-position span over examples of differing length needs no
    regrouping, because positions are indexed per selected token rather than as a
    rectangle.
    """
    return run_steering_interventions(
        pipeline,
        dataset,
        target,
        vectors,
        batch_size=batch_size,
        output_scores=output_scores,
        mode="replace",
    )


def _score_all_metrics(
    raw_results: dict[tuple[Any, ...], dict[str, Any]],
    dataset: list[CounterfactualExample],
    metrics: dict[str, InterchangeMetric],
    causal_model: CausalModel | None,
    original_outputs: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """Score one cell's outputs (a single-key ``raw_results``) under every metric.

    Returns ``{metric_name: score}`` for the lone key in ``raw_results``. Scoring
    one cell at a time (rather than accumulating every cell's raw outputs and
    scoring at the end) keeps only a single cell's logits alive — the reason
    ``output_scores=True`` is affordable here even with full-vocab scores.
    """
    (key,) = raw_results
    return {
        name: score_intervention_outputs(
            raw_results=raw_results,
            dataset=dataset,
            metric=metric,
            causal_model=causal_model,
            original_outputs=original_outputs,
        )[key]
        for name, metric in metrics.items()
    }


def run_ablation_scan_multi(
    targets: dict[tuple[Any, ...], InterchangeTarget],
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
    """Ablate each target in a grid and score it under *several* metrics at once.

    Like :func:`run_ablation_scan` but generates once per cell and scores that
    cell under every metric in ``metrics`` (e.g. a behavioral-match drop *and* a
    predicted-token logit drop), so the expensive generation is shared. Returns
    ``{key: {metric_name: score}}``.

    ``vectors`` is a single flat ``{unit_id: tensor}`` dict covering every unit
    across all targets (unit ids are globally unique); the per-target slice is
    passed through to ``run_ablation``. Each cell is scored as soon as it is
    generated (see :func:`_score_all_metrics`), so memory stays bounded to one
    cell even when ``output_scores`` keeps full-vocab logits.
    """
    results: dict[tuple[Any, ...], dict[str, float]] = {}
    for key, target in tqdm(targets.items(), desc="Ablation scan", total=len(targets)):
        unit_ids = [unit.id for unit in target.flatten()]
        target_vectors = {unit_id: vectors[unit_id] for unit_id in unit_ids}
        outputs = run_ablation(
            pipeline,
            dataset,
            target,
            target_vectors,
            batch_size=batch_size,
            output_scores=output_scores,
        )
        results[key] = _score_all_metrics(
            {key: outputs}, dataset, metrics, causal_model, original_outputs
        )
    return results


def run_ablation_combo_multi(
    units: list[AtomicModelUnit],
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
    """Ablate a set of units jointly and score it under several metrics at once.

    Like :func:`run_ablation_combo` but returns ``{metric_name: score}`` from a
    single joint forward pass. See :func:`run_ablation_combo` for the
    one-unit-per-group rationale.
    """
    target = InterchangeTarget([[unit] for unit in units])
    # Slice to just these units: run_steering_interventions errors on vectors for
    # units outside the target, so a shared grid-wide dict must be narrowed here.
    combo_vectors = {unit.id: vectors[unit.id] for unit in units}
    outputs = run_ablation(
        pipeline,
        dataset,
        target,
        combo_vectors,
        batch_size=batch_size,
        output_scores=output_scores,
    )
    return _score_all_metrics(
        {("combo",): outputs}, dataset, metrics, causal_model, original_outputs
    )


def run_ablation_scan(
    targets: dict[tuple[Any, ...], InterchangeTarget],
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
    """Ablate each target in a grid and score with a single ``metric``.

    Thin single-metric wrapper over :func:`run_ablation_scan_multi`. Returns
    ``{key: ablated_accuracy}`` — the caller computes
    ``drop = base_accuracy - ablated_accuracy``.

    Structurally mirrors ``methods/interchange/layer_scan.py::run_layer_scan``.
    """
    multi = run_ablation_scan_multi(
        targets,
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
    units: list[AtomicModelUnit],
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
    """Ablate a set of units jointly and return one accuracy.

    Thin single-metric wrapper over :func:`run_ablation_combo_multi`. ``units``
    are ablated together in a single forward pass; ``vectors`` supplies a
    reference vector per unit id. Returns the joint ablated accuracy (the caller
    computes the drop).
    """
    return run_ablation_combo_multi(
        units,
        dataset,
        pipeline,
        vectors,
        metrics={"_": metric},
        batch_size=batch_size,
        output_scores=output_scores,
        causal_model=causal_model,
        original_outputs=original_outputs,
    )["_"]
