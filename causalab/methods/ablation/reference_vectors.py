"""Reference vectors for component ablation.

Ablation overwrites a component's output with a fixed reference vector via a
pyvene ``replace`` intervention. Two references are supported:

* **zero** — drop the feature contribution entirely (the orthogonal/error term
  of the featurizer is still preserved by ``FeatureReplaceIntervention``).
* **mean** — replace with the corpus-average activation, the standard
  "mean-ablation" baseline that removes the component's *variation* while
  keeping its average effect.

These return ``{unit_id: tensor}`` dicts shaped ``(n_features,)`` per unit — the
broadcast form ``run_steering_interventions`` expands across the batch and span.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.ablation._spans import (
    group_by_position_count,
    unit_position_count,
)
from causalab.methods.steer.steer import make_zero_features
from causalab.neural.activations.collect import collect_features
from causalab.neural.pipeline import Pipeline
from causalab.neural.units import AtomicModelUnit, ComponentIndexer, InterchangeTarget


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


def _single_position_unit(unit: AtomicModelUnit, offset: int) -> AtomicModelUnit:
    """A shallow clone of ``unit`` whose indexer keeps only its ``offset``-th
    span position (in each example's unpadded frame).

    Used to collect a multi-position ``pos`` unit one position at a time — see
    :func:`make_mean_vectors`. The clone keeps the original ``id`` (so collected
    rows accumulate under the real unit) and shares the featurizer/shape/layer,
    overriding only ``_indices_func``; the original unit is untouched.
    """
    base = unit._indices_func  # pyright: ignore[reportPrivateUsage]

    def kth(inp: Any, is_original: bool | None = None, _base=base, _k=offset):
        positions = _base.index(inp, batch=False, is_original=is_original)
        return [positions[_k]]

    clone = copy.copy(unit)
    clone._indices_func = ComponentIndexer(kth, id=f"{base.id}#p{offset}")
    return clone


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

    **Multi-position ``pos`` units** (``ResidualStream`` / ``MLP`` over a span of
    >1 token) are collected one position at a time: pyvene's collect path mangles
    the 3-D ``(b, n_pos, d)`` gather for ``pos`` units (it works for the 4-D head
    gather), so a single multi-position collect raises in ``b_sd_to_bsd``. Slicing
    the span into single-position collects and summing the rows is equivalent —
    the mean is over the same (example, position) set either way — and sidesteps
    the bug. Single-position ``pos`` units and attention heads (any span) collect
    in one pass as before.
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
        reference = sub_dataset[0]["input"]
        # Within a bucket every example shares each unit's position count.
        simple_units: list[AtomicModelUnit] = []
        multi_units: list[tuple[AtomicModelUnit, int]] = []
        for unit in units:
            n_pos = unit_position_count(unit, reference)
            if unit.unit != "h.pos" and n_pos > 1:
                multi_units.append((unit, n_pos))
            else:
                simple_units.append(unit)

        if simple_units:
            feats = collect_features(
                sub_dataset, pipeline, simple_units, batch_size=batch_size
            )
            assert isinstance(feats, dict)  # collect_output_logits=False
            accumulate(feats)

        # Collect every multi-position pos unit at offset k together (those that
        # still have a position k); this is one collect per offset, not per unit.
        max_n_pos = max((n for _, n in multi_units), default=0)
        for k in range(max_n_pos):
            clones = [
                _single_position_unit(unit, k) for unit, n in multi_units if k < n
            ]
            feats = collect_features(
                sub_dataset, pipeline, clones, batch_size=batch_size
            )
            assert isinstance(feats, dict)
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
