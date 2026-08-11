"""
steer.py
========
Core utilities for running steering interventions.

This module provides functions for applying steering vectors in feature space to
model activations. Unlike interchange interventions (which swap activations between
base and counterfactual inputs), steering interventions operate on pre-computed vectors
in a learned feature space.

Steering modes:
    - "add": Additive steering - vectors are ADDED to base features
    - "replace": Replacement steering - vectors REPLACE base features entirely

Key concepts:
    - Steering vectors are specified in feature space (e.g., 10-dimensional PCA space)
    - The component of base activations orthogonal to the feature space is preserved
    - For zero ablation, use mode="replace" with vectors from make_zero_features()
    - Sites are :class:`~causalab.neural.specs.SiteSpec` values (WU4, #506); every
      per-site dict (``steering_vectors``, ``type_by_key``) keys on ``spec.key``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.neural.pipeline import GenerationResult, Pipeline
from causalab.neural.specs import EditSpec, SiteSpec

# Configure logging
logger = logging.getLogger(__name__)


def make_zero_features(
    sites: Sequence[SiteSpec],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    """
    Create zero steering vectors for each site.

    Useful for ablation studies where you want to "zero out" the feature
    contribution (though note: with additive steering, zero vectors
    result in no change to base activations).

    Args:
        sites: Flat sequence of :class:`~causalab.neural.specs.SiteSpec`.
            Each spec must have a featurizer with n_features defined,
            or a ``width`` from which n_features can be inferred.
        device: Device for tensors (default: CPU)
        dtype: Dtype for tensors (default: float32)

    Returns:
        Dict mapping ``spec.key`` to zero tensors of shape (n_features,)

    Raises:
        ValueError: If any spec has neither n_features nor width

    Example:
        >>> zeros = make_zero_features(sites)
        >>> # {"residual_stream.L5.block_output.pos0": tensor([0., 0., ...]), ...}
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    result = {}
    for spec in sites:
        n_features = spec.fsite.featurizer.n_features
        if n_features is None:
            if spec.width is not None:
                n_features = spec.width
            else:
                raise ValueError(
                    f"Site '{spec.key}' has featurizer with n_features=None "
                    "and no width. Cannot create zero features without "
                    "knowing feature dimensionality."
                )
        result[spec.key] = torch.zeros(n_features, device=device, dtype=dtype)

    return result


def validate_steering_vectors(
    steering_vectors: dict[str, Tensor],
    sites: Sequence[SiteSpec],
    n_examples: int,
) -> None:
    """
    Validate steering vectors match expected dimensions.

    Args:
        steering_vectors: Dict mapping ``spec.key`` to steering tensors
        sites: Flat sequence of :class:`~causalab.neural.specs.SiteSpec`
        n_examples: Number of examples in the dataset

    Raises:
        ValueError: If vectors don't match expected dimensions
    """
    site_keys = {spec.key for spec in sites}

    # Check all required sites are present
    missing = site_keys - set(steering_vectors.keys())
    if missing:
        raise ValueError(
            f"Missing steering vectors for sites: {missing}. "
            f"Expected vectors for: {site_keys}"
        )

    # Check dimensions
    for spec in sites:
        vec = steering_vectors[spec.key]
        n_features = spec.fsite.featurizer.n_features

        if n_features is None:
            # Can't validate if n_features is unknown
            continue

        # Check feature dimension
        if vec.ndim == 1:
            # Broadcast mode: (n_features,)
            if vec.shape[0] != n_features:
                raise ValueError(
                    f"Steering vector for '{spec.key}' has {vec.shape[0]} features, "
                    f"but featurizer expects {n_features} features."
                )
        elif vec.ndim == 2:
            # Per-example mode: (n_examples, n_features)
            if vec.shape[0] != n_examples:
                raise ValueError(
                    f"Steering vector for '{spec.key}' has {vec.shape[0]} examples, "
                    f"but dataset has {n_examples} examples."
                )
            if vec.shape[1] != n_features:
                raise ValueError(
                    f"Steering vector for '{spec.key}' has {vec.shape[1]} features, "
                    f"but featurizer expects {n_features} features."
                )
        else:
            raise ValueError(
                f"Steering vector for '{spec.key}' has invalid shape {vec.shape}. "
                f"Expected (n_features,) for broadcast mode or "
                f"(n_examples, n_features) for per-example mode."
            )


def run_steering_interventions(
    pipeline: Pipeline,
    dataset: list[CounterfactualExample],
    sites: Sequence[SiteSpec],
    steering_vectors: dict[str, Tensor],
    batch_size: int = 32,
    output_scores: bool | int = True,
    mode: Literal["add", "replace"] = "add",
    scale: float = 1.0,
    type_by_key: dict[str, str] | None = None,
    noise_seed: int = 0,
    gen_kwargs: dict[str, Any] | None = None,
) -> GenerationResult:
    """
    Run steering interventions on a dataset.

    Applies steering vectors in feature space to base activations at the given
    sites.

    Args:
        pipeline: Pipeline containing the model
        dataset: List of CounterfactualExample with "input" field containing base inputs.
            Unlike interchange interventions, counterfactual_inputs are not used.
        sites: Flat sequence of :class:`~causalab.neural.specs.SiteSpec` —
            intervention locations and featurizers. Each spec's featurizer
            defines the feature space for steering. Grouping is meaningless
            here (no edit reads a counterfactual source), so the sites run as
            one edit group.
        steering_vectors: Dict mapping ``spec.key`` to steering tensors.
            - Shape (n_features,): Broadcast to all examples (broadcast mode)
            - Shape (n_examples, n_features): Per-example steering (per-example mode)
            Keys must cover every spec in ``sites``.
        batch_size: Batch size for processing
        output_scores: Controls score output (same as run_interchange_interventions)
        mode: Steering mode.
            - "add": Add steering vectors to base features (default, original behavior)
            - "replace": Replace base features with steering vectors entirely.
                For zero ablation, use mode="replace" with vectors from make_zero_features().
        scale: Scaling factor applied to steering vectors before intervention.
            Default is 1.0 (no scaling). Useful for controlling intervention strength.
            For mode="add": steered = base + scale * steering_vector
            For mode="replace": replaced = scale * steering_vector
        type_by_key: Optional ``spec.key -> intervention_type`` map for a *mixed*
            run — each site uses its mapped type instead of ``mode``. The
            per-site source still comes from ``steering_vectors`` and is shaped
            identically; the interventions interpret it (a ``replace`` value, a
            ``noise`` scale, etc.). Used by causal tracing to run ``noise`` on the
            corrupted entry and ``replace`` on the restored site in one pass.
        noise_seed: Seed for any ``noise``-type sites. Every noise edit carries
            this seed and the engine builds ONE shared stream per distinct seed
            per call (the #505 noise lowering rule), so a run is reproducible
            end-to-end and draws advance across edits and batches exactly like
            the retired shared-``SeededNoise`` instance.
        gen_kwargs: Extra HF ``generate`` kwargs forwarded to
            :func:`~causalab.neural.dataset.run_intervened_generation`
            (e.g. ``{"min_new_tokens": N}`` — the escape hatch its
            ragged-scores refusal names).

    Returns:
        One flat :class:`~causalab.neural.pipeline.GenerationResult` over the
        whole dataset (EU5b, #487): ``sequences`` ``(n_examples,
        max_new_tokens)``, ``strings`` one entry per example, per-step
        ``scores`` (or ``scores_top_k`` for an integer ``output_scores``).

    Example (additive steering):
        >>> # Add steering vector to base features
        >>> steering = {site.key: torch.randn(10)}
        >>> results = run_steering_interventions(
        ...     pipeline, dataset, [site], steering, mode="add"
        ... )

    Example (scaled steering):
        >>> # Add half-strength steering
        >>> results = run_steering_interventions(
        ...     pipeline, dataset, [site], steering, mode="add", scale=0.5
        ... )

    Example (zero ablation):
        >>> # Replace features with zeros
        >>> zeros = make_zero_features(sites)
        >>> results = run_steering_interventions(
        ...     pipeline, dataset, sites, zeros, mode="replace"
        ... )
    """
    n_examples = len(dataset)  # type: ignore[arg-type]

    # Validate steering vectors
    validate_steering_vectors(steering_vectors, sites, n_examples)

    # PL3 (#405): rerouted onto the nnsight batched-execution engine — one
    # generate trace per batch with prefill edits, batch-first position
    # resolution, ragged spans native (no length-bucketing upstream). The
    # engine moves values to each site's device, so the per-layer device map
    # is no longer this function's concern; `scale` is folded into the
    # vectors once, exactly as before. Clone to avoid mutating the caller's
    # dict.
    from causalab.neural.dataset import run_intervened_generation
    from causalab.neural.pipeline import compress_scores_top_k

    scaled = {key: vec.clone() * scale for key, vec in steering_vectors.items()}
    group = []
    for spec in sites:
        site_type = (type_by_key or {}).get(spec.key, mode)
        if site_type == "noise":
            # The seed — never a stream: run_intervened_generation builds ONE
            # SeededNoise per distinct seed per call, before its batch loop
            # (the #505 lowering rule), reproducing the retired shared-stream
            # draws bit-identically (pinned by tests/neural/test_dataset.py::
            # TestSpecSurfaceProperty::
            # test_noise_one_stream_per_seed_matches_legacy_shared_stream).
            group.append(
                EditSpec(spec, mode="noise", vector=scaled[spec.key], seed=noise_seed)
            )
        elif site_type in ("add", "replace"):
            group.append(EditSpec(spec, mode=site_type, vector=scaled[spec.key]))
        else:
            raise ValueError(
                f"unknown intervention type {site_type!r} for site {spec.key!r}; "
                "expected 'add', 'replace', or 'noise'"
            )

    result = run_intervened_generation(
        pipeline,
        dataset,
        [group],
        batch_size=batch_size,
        output_scores=bool(output_scores),
        **(gen_kwargs or {}),
    )
    if not isinstance(output_scores, bool) and output_scores > 0:
        result = compress_scores_top_k(result, pipeline, k=output_scores)
    return result
