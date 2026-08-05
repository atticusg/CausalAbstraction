"""Reference vectors for component ablation.

Ablation overwrites a component's output with a fixed reference vector via a
``replace`` intervention. Two references are supported:

* **zero** — drop the feature contribution entirely (the orthogonal/error term
  of the featurizer is still preserved by ``ReplaceIntervention``).
* **mean** — replace with the corpus-average activation, the standard
  "mean-ablation" baseline that removes the component's *variation* while
  keeping its average effect.

These return ``{unit_id: tensor}`` dicts shaped ``(n_features,)`` per unit — the
broadcast form ``run_steering_interventions`` expands across the batch and span.
"""

from __future__ import annotations


import torch
from torch import Tensor

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.ablation._spans import (
    group_by_position_count,
)
from causalab.methods.steer.steer import make_zero_features
from causalab.neural.activations.collect import collect_features
from causalab.neural.pipeline import Pipeline
from causalab.neural.units import InterchangeTarget


def make_zero_vectors(
    target: InterchangeTarget,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    """Per-unit zero vectors for zero-ablation.

    Thin wrapper over :func:`causalab.methods.steer.steer.make_zero_features` so
    ablation callers don't reach across into the steering module.
    """
    return make_zero_features(target, device=device, dtype=dtype)


def make_mean_vectors(
    pipeline: Pipeline,
    dataset: list[CounterfactualExample],
    target: InterchangeTarget,
    batch_size: int = 32,
) -> dict[str, Tensor]:
    """Per-unit corpus-mean activation vectors for mean-ablation.

    Collects each unit's activations over ``dataset`` and averages across every
    gathered (example, position) row, yielding one ``(n_features,)`` mean vector
    per unit. The collection is bucketed by position count for the same reason
    the intervention is (see :mod:`causalab.methods.ablation._spans`):
    ``collect_features``' gather is rectangular, so an all-position span over a
    variable-length dataset must be collected one equal-length bucket at a time.
    Sums and row-counts accumulate across buckets so the result is the true
    global mean, not a mean-of-bucket-means.

    Every unit in a bucket — including a multi-position ``pos`` span — collects
    in one pass, because the mean is over the same (example, position) set
    however the rows arrive.
    """
    units = target.flatten()
    sums: dict[str, Tensor | None] = {unit.id: None for unit in units}
    counts: dict[str, int] = {unit.id: 0 for unit in units}

    def accumulate(features: dict[str, Tensor]) -> None:
        for unit_id, feats in features.items():
            # feats: (n_rows, n_features); n_rows = n_examples * (positions collected)
            row_sum = feats.sum(dim=0)
            running = sums[unit_id]
            sums[unit_id] = row_sum if running is None else running + row_sum
            counts[unit_id] += feats.shape[0]

    for _, sub_dataset in group_by_position_count(target, dataset):
        # Within a bucket every example shares each unit's position count, so a
        # single collect covers every unit and every position it selects.
        feats = collect_features(sub_dataset, pipeline, units, batch_size=batch_size)
        assert isinstance(feats, dict)  # collect_output_logits=False
        accumulate(feats)

    means: dict[str, Tensor] = {}
    for unit_id, total in sums.items():
        if total is None or counts[unit_id] == 0:
            raise ValueError(
                f"No activations collected for unit '{unit_id}'; cannot build a "
                "mean vector from an empty dataset."
            )
        means[unit_id] = total / counts[unit_id]
    return means
