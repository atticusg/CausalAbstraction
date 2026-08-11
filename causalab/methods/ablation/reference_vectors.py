"""Reference vectors for component ablation.

Ablation overwrites a component's output with a fixed reference vector via a
``replace`` intervention. Two references are supported:

* **zero** — drop the feature contribution entirely (the orthogonal/error term
  of the featurizer is still preserved by the site layer's error-term
  contract).
* **mean** — replace with the corpus-average activation, the standard
  "mean-ablation" baseline that removes the component's *variation* while
  keeping its average effect.

These return ``{spec.key: tensor}`` dicts shaped ``(n_features,)`` per site —
the broadcast form ``run_steering_interventions`` expands across the batch and
span. Sites are :class:`~causalab.neural.specs.SiteSpec` values (WU4, #506).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.ablation._spans import (
    group_by_position_count,
    spec_position_count,
)
from causalab.methods.steer.steer import make_zero_features
from causalab.neural.activations.collect import collect_features
from causalab.neural.head_view import HeadSite
from causalab.neural.pipeline import Pipeline
from causalab.neural.specs import SiteSpec

# ComponentIndexer is position machinery (TokenPosition's base class), not the
# retired unit surface — the WU6 sweep (#508) relocated it here.
from causalab.neural.token_positions import ComponentIndexer


def make_zero_vectors(
    sites: Sequence[SiteSpec],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    """Per-site zero vectors for zero-ablation.

    Thin wrapper over :func:`causalab.methods.steer.steer.make_zero_features` so
    ablation callers don't reach across into the steering module.
    """
    return make_zero_features(sites, device=device, dtype=dtype)


def _single_position_spec(spec: SiteSpec, offset: int) -> SiteSpec:
    """A shallow view of ``spec`` whose positions keep only its ``offset``-th
    span position (in each example's unpadded frame).

    Used to collect a multi-position span one position at a time — see
    :func:`make_mean_vectors`. :meth:`SiteSpec.with_positions` keeps the
    original ``key`` (so collected rows accumulate under the real spec) and
    shares the featurizer/site; only the position spec is swapped. The
    original spec is untouched.
    """
    base = spec.positions
    if base is None:
        raise ValueError(
            f"spec {spec.key!r} has positions=None (unbound); cannot slice an "
            "unbound span into single positions."
        )
    if isinstance(base, tuple):  # literal row (normalized by SiteSpec)
        return spec.with_positions((base[offset],))

    def kth(inp: Any, is_original: bool | None = None, _base=base, _k=offset):
        positions = _base.index(inp, batch=False, is_original=is_original)
        return [positions[_k]]

    return spec.with_positions(ComponentIndexer(kth, id=f"{base.id}#p{offset}"))


def make_mean_vectors(
    pipeline: Pipeline,
    dataset: list[CounterfactualExample],
    sites: Sequence[SiteSpec],
    batch_size: int = 32,
) -> dict[str, Tensor]:
    """Per-site corpus-mean activation vectors for mean-ablation.

    Collects each site's activations over ``dataset`` and averages across every
    gathered (example, position) row, yielding one ``(n_features,)`` mean vector
    per site. The collection is bucketed by position count
    (:mod:`causalab.methods.ablation._spans`), and **multi-position
    residual/MLP spans** are collected one position at a time — the collection
    scheme the legacy stack required and the mean is invariant to (it averages
    the same (example, position) set either way), kept so the collect calls and
    their accumulation order are unchanged. Sums and row-counts accumulate
    across buckets so the result is the true global mean, not a
    mean-of-bucket-means. Single-position spans and attention heads (any span)
    collect in one pass as before.
    """
    sites = list(sites)
    sums: dict[str, Tensor | None] = {spec.key: None for spec in sites}
    counts: dict[str, int] = {spec.key: 0 for spec in sites}

    def accumulate(features: dict[str, Tensor]) -> None:
        for key, feats in features.items():
            # feats: (n_rows, n_features); n_rows = n_examples * (positions collected)
            row_sum = feats.sum(dim=0)
            running = sums[key]
            sums[key] = row_sum if running is None else running + row_sum
            counts[key] += feats.shape[0]

    for _, sub_dataset in group_by_position_count(sites, dataset):
        reference = sub_dataset[0]["input"]
        # Within a bucket every example shares each site's position count.
        simple_specs: list[SiteSpec] = []
        multi_specs: list[tuple[SiteSpec, int]] = []
        for spec in sites:
            n_pos = spec_position_count(spec, reference)
            if not isinstance(spec.fsite.site, HeadSite) and n_pos > 1:
                multi_specs.append((spec, n_pos))
            else:
                simple_specs.append(spec)

        if simple_specs:
            feats = collect_features(
                sub_dataset, pipeline, simple_specs, batch_size=batch_size
            )
            assert isinstance(feats, dict)  # collect_output_logits=False
            accumulate(feats)

        # Collect every multi-position span at offset k together (those that
        # still have a position k); this is one collect per offset, not per site.
        max_n_pos = max((n for _, n in multi_specs), default=0)
        for k in range(max_n_pos):
            clones = [
                _single_position_spec(spec, k) for spec, n in multi_specs if k < n
            ]
            feats = collect_features(
                sub_dataset, pipeline, clones, batch_size=batch_size
            )
            assert isinstance(feats, dict)
            accumulate(feats)

    means: dict[str, Tensor] = {}
    for key, total in sums.items():
        if total is None or counts[key] == 0:
            raise ValueError(
                f"No activations collected for unit '{key}'; cannot build a "
                "mean vector from an empty dataset."
            )
        means[key] = total / counts[key]
    return means
