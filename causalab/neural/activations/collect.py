"""
collect.py
==========
Functions for collecting and analyzing neural network activations.

This module provides utilities for processing collected features from model
sites, including dimensionality reduction techniques like SVD/PCA.
"""

from __future__ import annotations

import logging
from typing import Sequence

from torch import Tensor

from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
)
from causalab.neural.pipeline import Pipeline
from causalab.neural.specs import SiteSpec

logger = logging.getLogger(__name__)


def collect_features(
    dataset: list[CounterfactualExample],
    pipeline: Pipeline,
    sites: Sequence[SiteSpec],
    batch_size: int = 32,
    collect_output_logits: bool = False,
) -> dict[str, Tensor] | tuple[dict[str, Tensor], list[Tensor]]:
    """
    Collect internal neural network activations (features) at specified model locations.

    This function:
    1. Creates an intervenable model configured for feature collection
    2. Processes batches from the dataset to extract activations at target locations
    3. Returns a dictionary mapping each site's ``spec.key`` to its collected features

    Args:
        dataset: List of CounterfactualExample objects
        pipeline: Neural model pipeline for processing inputs
        sites: Flat sequence of :class:`~causalab.neural.specs.SiteSpec` to
            collect features at.
        batch_size: Number of examples to process per batch (default: 32)
        collect_output_logits: If True, also capture the model's full output
            logits for each example. This avoids a redundant forward pass when
            you need both intermediate activations and output distributions
            (e.g., for reference distribution computation).

    Returns:
        If collect_output_logits is False (default):
            Dict mapping ``spec.key`` to feature tensors of shape (n_samples, n_features).
        If collect_output_logits is True:
            Tuple of (features_dict, output_logits) where output_logits is a
            list of per-example logit tensors (each of shape (seq_len, vocab_size),
            since sequence lengths may vary across batches).

    Example:
        >>> features_dict = collect_features(dataset, pipeline, sites, batch_size=32)
        >>> features_dict, logits = collect_features(
        ...     dataset, pipeline, sites, collect_output_logits=True
        ... )
    """
    # Raise on duplicate keys: the result is keyed by `spec.key` so duplicates
    # would silently merge. Downstream consumers — `methods/pca.py`,
    # `methods/spline/train.py`, `analyses/subspace/das.py` — all assume the
    # returned dict has one entry per site. The guard fires here, before the
    # engine delegation, so no forward runs on bad input.
    seen_keys: set[str] = set()
    for s in sites:
        if s.key in seen_keys:
            raise ValueError(
                f"Duplicate site key {s.key!r} in sites; keys must be unique"
            )
        seen_keys.add(s.key)

    # PL3 (#405): rerouted onto the nnsight batched-execution engine — one
    # tokenization per batch (positions resolved batch-first on the run
    # encoding), one fused early-stopped forward per batch, ragged spans
    # native. The (n_samples, n_features) example-major output contract is
    # unchanged; results key on `spec.key` (WU3 #505).
    from causalab.neural.dataset import collect_dataset_features

    return collect_dataset_features(
        pipeline,
        dataset,
        sites,
        batch_size=batch_size,
        collect_output_logits=collect_output_logits,
    )
