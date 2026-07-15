"""Position-count bucketing for ablation spans.

pyvene gathers a *rectangular* ``(b, [h,] n_pos, d)`` tensor, so every example
in a batch must expose the same number of intervened positions. Single-position
spans (e.g. a named ``last_token``) satisfy this trivially; all-position spans
(``get_all_tokens``) do not — examples of different token length yield different
position counts and the gather (``torch.gather`` / ``torch.tensor`` on a ragged
index list) raises. These helpers group a dataset into equal-position-count
buckets so each bucket runs as one rectangular batch.

Both ``run.py`` (intervention) and ``reference_vectors.py`` (mean collection)
hit the same rectangular-gather constraint, so the bucketing lives here.
"""

from __future__ import annotations

from typing import Any

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.neural.units import AtomicModelUnit, InterchangeTarget


def unit_position_count(unit: AtomicModelUnit, example_input: Any) -> int:
    """Number of token positions ``unit`` gathers for a single example.

    ``AttentionHead.index_component`` returns ``[head_axis, position_axis]``; the
    position axis (``idx[1]``, a one-row list-of-lists for a single example) is
    what carries the position count — reading ``len(idx)`` would return the
    axis count (2) and ``len(idx[0])`` the head count. ``pos`` units return the
    position list directly. ``attention_mask=None`` keeps indices in the
    example's own unpadded frame, which is all we need for counting.
    """
    # index_component's return shape is unit-type dependent (pyvene-shaped nested
    # lists), so it's typed broadly; treat as Any for the structural indexing.
    idx: Any = unit.index_component(example_input, batch=False, attention_mask=None)
    if unit.unit == "h.pos":
        return len(idx[1][0])
    return len(idx)


def example_bucket_key(
    target: InterchangeTarget, example: CounterfactualExample
) -> tuple[int, ...]:
    """Per-unit position counts for one example, in ``flatten()`` order."""
    return tuple(
        unit_position_count(unit, example["input"]) for unit in target.flatten()
    )


def group_by_position_count(
    target: InterchangeTarget, dataset: list[CounterfactualExample]
) -> list[tuple[list[int], list[CounterfactualExample]]]:
    """Group dataset indices by equal per-unit position counts.

    Returns a list of ``(original_indices, sub_dataset)`` buckets. When every
    example shares one key — the common case for single-position spans, or any
    span over an equal-length dataset — the result is a single bucket spanning
    the whole dataset in original order, so callers can skip reassembly.
    """
    buckets: dict[tuple[int, ...], list[int]] = {}
    for i, example in enumerate(dataset):
        buckets.setdefault(example_bucket_key(target, example), []).append(i)
    return [(indices, [dataset[i] for i in indices]) for indices in buckets.values()]
