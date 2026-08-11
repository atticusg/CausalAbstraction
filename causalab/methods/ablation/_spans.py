"""Position-count bucketing for ablation spans.

The collection helpers in :mod:`causalab.methods.ablation.reference_vectors`
average one activation row per gathered (example, position); grouping a dataset
into equal-position-count buckets keeps every collect call rectangular per
site, so the per-position slicing there stays well-defined. Single-position
spans (e.g. a named ``last_token``) form one bucket trivially; all-position
spans (``get_all_tokens``) split by example token length.

Both ``run.py`` (intervention) and ``reference_vectors.py`` (mean collection)
consume the same bucketing, so it lives here. Sites are
:class:`~causalab.neural.specs.SiteSpec` values (WU4, #506); counts read each
spec's declarative ``positions``.
"""

from __future__ import annotations

from typing import Any, Sequence

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.neural.positions import resolve_positions
from causalab.neural.specs import SiteSpec


def spec_position_count(spec: SiteSpec, example_input: Any) -> int:
    """Number of token positions ``spec`` selects for a single example.

    Resolved in the example's own unpadded frame (``attention_mask=None``),
    which is all counting needs. Positions live purely on the sequence axis —
    the head axis is structural (:class:`~causalab.neural.head_view.HeadSite`),
    so the count is uniform across site kinds (the legacy two-axis
    ``AttentionHead.index_component`` return is gone).
    """
    if spec.positions is None:
        raise ValueError(
            f"spec {spec.key!r} has positions=None (unbound): position-count "
            "bucketing needs the spec to say where on the sequence axis it "
            "reads. Bind positions via spec.with_positions(...), literal rows, "
            "or load_site_specs(dir, token_positions=...)."
        )
    return len(resolve_positions(spec.positions, [example_input], None)[0])


def example_bucket_key(
    sites: Sequence[SiteSpec], example: CounterfactualExample
) -> tuple[int, ...]:
    """Per-site position counts for one example, in ``sites`` order."""
    return tuple(spec_position_count(spec, example["input"]) for spec in sites)


def group_by_position_count(
    sites: Sequence[SiteSpec], dataset: list[CounterfactualExample]
) -> list[tuple[list[int], list[CounterfactualExample]]]:
    """Group dataset indices by equal per-site position counts.

    Returns a list of ``(original_indices, sub_dataset)`` buckets. When every
    example shares one key — the common case for single-position spans, or any
    span over an equal-length dataset — the result is a single bucket spanning
    the whole dataset in original order, so callers can skip reassembly.
    """
    buckets: dict[tuple[int, ...], list[int]] = {}
    for i, example in enumerate(dataset):
        buckets.setdefault(example_bucket_key(sites, example), []).append(i)
    return [(indices, [dataset[i] for i in indices]) for indices in buckets.values()]
