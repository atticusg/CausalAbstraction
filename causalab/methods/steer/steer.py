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
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

import torch
from torch import Tensor
from tqdm import tqdm

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.neural.pipeline import Pipeline
from causalab.neural.interventions import FeatureIntervention
from causalab.neural.units import InterchangeTarget
from causalab.neural.activations.engine import (
    build_interventions,
    build_plans,
    generate_with_interventions,
)
from causalab.neural.activations.data_utils import (
    convert_to_top_k,
    move_outputs_to_cpu,
)

# Configure logging
logger = logging.getLogger(__name__)


def make_zero_features(
    interchange_target: InterchangeTarget,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    """
    Create zero steering vectors for each unit in the target.

    Useful for ablation studies where you want to "zero out" the feature
    contribution (though note: with additive steering, zero vectors
    result in no change to base activations).

    Args:
        interchange_target: Target specifying intervention locations.
            Each unit must have a featurizer with n_features defined,
            or a shape from which n_features can be inferred.
        device: Device for tensors (default: CPU)
        dtype: Dtype for tensors (default: float32)

    Returns:
        Dict mapping unit IDs to zero tensors of shape (n_features,)

    Raises:
        ValueError: If any unit has neither n_features nor shape

    Example:
        >>> zeros = make_zero_features(target)
        >>> # {"layer5_pos0": tensor([0., 0., ...]), ...}
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    result = {}
    for unit in interchange_target.flatten():
        n_features = unit.featurizer.n_features
        if n_features is None:
            if unit.shape is not None:
                n_features = unit.shape[0]
            else:
                raise ValueError(
                    f"Unit '{unit.id}' has featurizer with n_features=None "
                    "and no shape. Cannot create zero features without "
                    "knowing feature dimensionality."
                )
        result[unit.id] = torch.zeros(n_features, device=device, dtype=dtype)

    return result


def prepare_steering_inputs(
    pipeline: Pipeline,
    batch: dict[str, Any],
    interchange_target: InterchangeTarget,
) -> tuple[dict[str, Tensor], list[Any]]:
    """
    Prepare base inputs for steering intervention.

    Unlike prepare_intervenable_inputs() in interchange.py, this only
    prepares base inputs (no counterfactuals).

    Args:
        pipeline: Pipeline for tokenization
        batch: Batch with "input" key
        interchange_target: For computing intervention locations

    Returns:
        Tuple of (tokenized_base, base_indices)
            - tokenized_base: Output of pipeline.load()
            - base_indices: Shape (num_model_units, batch_size, num_component_indices)
    """
    raw_base = batch["input"]

    # Load first so the padded `attention_mask` is available for shifting
    # unpadded position indices into the padded-batch frame.
    batched_base = pipeline.load(raw_base)

    # shape: (num_model_units, batch_size, num_component_indices)
    base_indices = [
        model_unit.index_component(
            raw_base,
            batch=True,
            is_original=True,
            attention_mask=batched_base["attention_mask"],
        )
        for group in interchange_target
        for model_unit in group
    ]

    return batched_base, base_indices


def validate_steering_vectors(
    steering_vectors: dict[str, Tensor],
    interchange_target: InterchangeTarget,
    n_examples: int,
) -> None:
    """
    Validate steering vectors match expected dimensions.

    Args:
        steering_vectors: Dict mapping unit IDs to steering tensors
        interchange_target: Target specifying intervention locations
        n_examples: Number of examples in the dataset

    Raises:
        ValueError: If vectors don't match expected dimensions
    """
    flat_units = interchange_target.flatten()
    unit_ids = {unit.id for unit in flat_units}

    # Check all required units are present
    missing = unit_ids - set(steering_vectors.keys())
    if missing:
        raise ValueError(
            f"Missing steering vectors for units: {missing}. "
            f"Expected vectors for: {unit_ids}"
        )

    # Check dimensions
    for unit in flat_units:
        vec = steering_vectors[unit.id]
        n_features = unit.featurizer.n_features

        if n_features is None:
            # Can't validate if n_features is unknown
            continue

        # Check feature dimension
        if vec.ndim == 1:
            # Broadcast mode: (n_features,)
            if vec.shape[0] != n_features:
                raise ValueError(
                    f"Steering vector for '{unit.id}' has {vec.shape[0]} features, "
                    f"but featurizer expects {n_features} features."
                )
        elif vec.ndim == 2:
            # Per-example mode: (n_examples, n_features)
            if vec.shape[0] != n_examples:
                raise ValueError(
                    f"Steering vector for '{unit.id}' has {vec.shape[0]} examples, "
                    f"but dataset has {n_examples} examples."
                )
            if vec.shape[1] != n_features:
                raise ValueError(
                    f"Steering vector for '{unit.id}' has {vec.shape[1]} features, "
                    f"but featurizer expects {n_features} features."
                )
        else:
            raise ValueError(
                f"Steering vector for '{unit.id}' has invalid shape {vec.shape}. "
                f"Expected (n_features,) for broadcast mode or "
                f"(n_examples, n_features) for per-example mode."
            )


def get_batch_steering_vectors(
    steering_vectors: dict[str, Tensor],
    interchange_target: InterchangeTarget,
    batch_start: int,
    batch_size: int,
    device: torch.device | dict[str, torch.device],
) -> list[Tensor]:
    """
    Extract steering vectors for a batch, handling broadcast vs per-example modes.

    Args:
        steering_vectors: Dict mapping unit IDs to steering tensors
        interchange_target: Target specifying intervention locations
        batch_start: Start index of batch in dataset
        batch_size: Size of current batch
        device: Either a single device (single-GPU model) or a per-unit
            mapping ``unit_id -> device`` for models sharded via ``device_map``.

    Returns:
        One ``(batch_size, 1, n_features)`` tensor per unit, in ``flatten()``
        order. The engine gathers a unit's activation as
        ``(batch, n_positions, width)``, so the singleton position axis
        broadcasts the same vector across however many tokens the unit selects —
        no per-unit position count needed, and per-head units need no special
        shape because the head axis is already resolved by the time the
        intervention sees the activation.
    """
    batch_vectors = []
    for unit in interchange_target.flatten():
        vec = steering_vectors[unit.id]

        if vec.ndim == 1:
            # Broadcast mode: expand to (batch_size, n_features)
            batch_vec = vec.unsqueeze(0).expand(batch_size, -1)
        else:
            # Per-example mode: slice to (batch_size, n_features)
            batch_vec = vec[batch_start : batch_start + batch_size]

        target_device = device[unit.id] if isinstance(device, dict) else device
        batch_vectors.append(batch_vec.to(target_device).unsqueeze(1))

    return batch_vectors


def batched_steering_intervention(
    pipeline: Pipeline,
    batch: dict[str, Any],
    interchange_target: InterchangeTarget,
    steering_vectors: dict[str, Tensor],
    batch_start: int,
    batch_size: int,
    unit_devices: torch.device | dict[str, torch.device],
    interventions: Sequence[FeatureIntervention],
    output_scores: bool | int = True,
) -> dict[str, Any]:
    """
    Perform a steering intervention on a single batch.

    Args:
        pipeline: Pipeline containing the model
        batch: Batch dict with "input" key
        interchange_target: Intervention target specification
        steering_vectors: Dict mapping unit IDs to steering tensors (full dataset;
            sliced per batch by :func:`get_batch_steering_vectors`).
        batch_start: Start index of this batch in the dataset.
        batch_size: Number of examples in this batch.
        unit_devices: Single device or per-unit ``unit_id -> device`` mapping.
        interventions: The run's interventions, one per flattened unit. Built
            once by the caller and reused across batches so a ``noise`` site's
            RNG advances rather than restarting each batch.
        output_scores: Score output control

    Returns:
        Dict with generation outputs
    """
    raw_base = batch["input"]
    base_encoding = pipeline.load(raw_base)
    units = interchange_target.flatten()
    positions = [
        unit.resolve_positions(
            raw_base,
            attention_mask=base_encoding["attention_mask"],
            is_original=True,
        )
        for unit in units
    ]
    sources = get_batch_steering_vectors(
        steering_vectors, interchange_target, batch_start, batch_size, unit_devices
    )
    plans = build_plans(
        units, positions, "", sources=sources, interventions=interventions
    )
    result = generate_with_interventions(
        pipeline, base_encoding, plans, output_scores=bool(output_scores)
    )
    return pipeline.format_generation(result, base_encoding, output_scores)


def run_steering_interventions(
    pipeline: Pipeline,
    dataset: list[CounterfactualExample],
    interchange_target: InterchangeTarget,
    steering_vectors: dict[str, Tensor],
    batch_size: int = 32,
    output_scores: bool | int = True,
    mode: Literal["add", "replace"] = "add",
    scale: float = 1.0,
    type_by_unit: dict[str, str] | None = None,
    noise_seed: int = 0,
) -> dict[str, list[Any]]:
    """
    Run steering interventions on a dataset.

    Applies steering vectors in feature space to base activations at locations
    specified by interchange_target.

    Args:
        pipeline: Pipeline containing the model
        dataset: List of CounterfactualExample with "input" field containing base inputs.
            Unlike interchange interventions, counterfactual_inputs are not used.
        interchange_target: Specifies intervention locations and featurizers.
            Each unit's featurizer defines the feature space for steering.
        steering_vectors: Dict mapping unit IDs to steering tensors.
            - Shape (n_features,): Broadcast to all examples (broadcast mode)
            - Shape (n_examples, n_features): Per-example steering (per-example mode)
            Keys must match unit IDs in interchange_target.flatten()
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
        type_by_unit: Optional ``unit_id -> intervention_type`` map for a *mixed*
            model — each site uses its mapped type instead of ``mode``. The
            per-unit source still comes from ``steering_vectors`` and is shaped
            identically; the interventions interpret it (a ``replace`` value, a
            ``noise`` scale, etc.). Used by causal tracing to run ``noise`` on the
            corrupted entry and ``replace`` on the restored site in one pass.
        noise_seed: Seed for any ``noise``-type sites (reproducible per site).

    Returns:
        Dict with 'sequences', 'string', and optionally 'scores' keys

    Example (additive steering):
        >>> # Add steering vector to base features
        >>> steering = {"layer5_pos0": torch.randn(10)}
        >>> results = run_steering_interventions(
        ...     pipeline, dataset, target, steering, mode="add"
        ... )

    Example (scaled steering):
        >>> # Add half-strength steering
        >>> results = run_steering_interventions(
        ...     pipeline, dataset, target, steering, mode="add", scale=0.5
        ... )

    Example (zero ablation):
        >>> # Replace features with zeros
        >>> zeros = make_zero_features(target)
        >>> results = run_steering_interventions(
        ...     pipeline, dataset, target, zeros, mode="replace"
        ... )
    """
    n_examples = len(dataset)  # type: ignore[arg-type]

    # Validate steering vectors
    validate_steering_vectors(steering_vectors, interchange_target, n_examples)

    # Scale, and clone so the user's dict is not mutated. Placement is the
    # model's problem now: a steering vector is combined with an activation the
    # trace already produced, so it follows that activation's device rather than
    # needing a per-layer lookup on a sharded model.
    device = pipeline.model.device
    steering_vectors = {
        unit_id: vec.to(device).clone() * scale
        for unit_id, vec in steering_vectors.items()
    }

    # One set of interventions for the whole run: a `noise` site's RNG must
    # advance across batches, not restart at each one (which would make the
    # corruption depend on batch_size).
    interventions = build_interventions(
        interchange_target.flatten(),
        mode,
        type_by_unit=type_by_unit,
        noise_seed=noise_seed,
    )

    all_outputs = []

    # Process each batch with progress tracking
    for batch_start in tqdm(
        range(0, len(dataset), batch_size),
        desc="Processing batches",
        disable=not logger.isEnabledFor(logging.DEBUG),
        leave=False,
    ):
        examples = dataset[batch_start : batch_start + batch_size]
        current_batch_size = len(examples)

        # Build batch dict (what shallow_collate_fn did)
        batch = {key: [ex[key] for ex in examples] for key in examples[0].keys()}

        with torch.no_grad():
            output_dict = batched_steering_intervention(
                pipeline,
                batch,
                interchange_target,
                steering_vectors,
                batch_start,
                current_batch_size,
                device,
                interventions,
                output_scores=output_scores,
            )
            all_outputs.append(output_dict)

    # Convert to top-k format if requested
    if not isinstance(output_scores, bool) and output_scores > 0:
        all_outputs = convert_to_top_k(all_outputs, pipeline, k=output_scores)

    # Move all outputs to CPU
    all_outputs = move_outputs_to_cpu(all_outputs)

    # Remove batch structure from outputs
    if not all_outputs:
        return {"sequences": [[]], "string": [[]]}
    all_outputs = {
        k: [output[k] for output in all_outputs] for k in all_outputs[0].keys()
    }

    return all_outputs
