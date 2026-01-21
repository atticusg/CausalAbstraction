"""
collect.py
==========
Functions for collecting and analyzing neural network activations.

This module provides utilities for processing collected features from model units,
including dimensionality reduction techniques like SVD/PCA.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
from torch import Tensor
from sklearn.decomposition import TruncatedSVD  # type: ignore[import-untyped]
from tqdm import tqdm

from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from pyvene import IntervenableModel  # type: ignore[reportMissingTypeStubs]

from causalab.neural.pyvene_core.intervenable_model import (
    prepare_intervenable_model,
    delete_intervenable_model,
)
from causalab.neural.pipeline import Pipeline
from causalab.neural.model_units import AtomicModelUnit, InterchangeTarget

logger = logging.getLogger(__name__)


def _collect_activations_single_batch(
    intervenable_model: IntervenableModel,
    loaded_inputs: dict[str, Tensor],
    indices: list[Any],
) -> list[Tensor]:
    """
    Collect activations from a single batch using an intervenable model.

    This is the core primitive for activation collection, used by both
    collect_features() for dataset-wide collection and collect_batch_representations()
    for cross-model patching.

    Args:
        intervenable_model: Model configured with "collect" intervention type
        loaded_inputs: Tokenized inputs (output of pipeline.load())
        indices: Position indices for each model unit, shape (num_units, batch_size, num_positions)

    Returns:
        List of activation tensors, one per intervention location, in order matching
        intervenable_model.sorted_keys. Each tensor has shape determined by the
        intervention location (e.g., (batch_size, hidden_dim) for residual stream).
    """
    # Create location map - for collection, source and base indices are the same
    location_map = {"sources->base": (indices, indices)}

    # Run collection pass - pyvene returns ((base_outputs, collected_activations), cf_outputs)
    # For collect mode, we want collected_activations which is [0][1]
    activations = intervenable_model(loaded_inputs, unit_locations=location_map)[0][1]

    return activations


def collect_features(
    dataset: list[CounterfactualExample],
    pipeline: Pipeline,
    model_units: list[AtomicModelUnit],
    batch_size: int = 32,
) -> dict[str, Tensor]:
    """
    Collect internal neural network activations (features) at specified model locations.

    This function:
    1. Creates an intervenable model configured for feature collection
    2. Processes batches from the dataset to extract activations at target locations
    3. Returns a dictionary mapping each model unit ID to its collected features

    Args:
        dataset: List of CounterfactualExample objects
        pipeline: Neural model pipeline for processing inputs
        model_units: Flat list of model units to collect features from
        batch_size: Number of examples to process per batch (default: 32)

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping model unit IDs to feature tensors.
                                Each tensor has shape (n_samples, n_features) containing
                                the activations for all inputs in the dataset.

    Example:
        >>> from causalab.neural.collect import collect_features
        >>>
        >>> features_dict = collect_features(dataset, pipeline, model_units, batch_size=32)
    """
    # Initialize model with "collect" intervention type (extracts activations without modifying them)
    # prepare_intervenable_model auto-wraps flat lists into InterchangeTarget
    intervenable_model = prepare_intervenable_model(
        pipeline, model_units, intervention_type="collect"
    )

    # Initialize container for collected features: one list per model unit
    # Use model unit IDs as keys to handle duplicates gracefully
    data: dict[str, list[Tensor]] = {model_unit.id: [] for model_unit in model_units}

    # Process dataset in batches with progress tracking
    for start in tqdm(
        range(0, len(dataset), batch_size),
        desc="Processing batches",
        leave=False,
    ):
        batch = dataset[start : start + batch_size]
        # Get inputs from batch
        batched_inputs = [example["input"] for example in batch]

        # Compute indices for each model unit
        indices = [
            model_unit.index_component(batched_inputs, batch=True, is_original=True)
            for model_unit in model_units
        ]

        # Load inputs through pipeline
        loaded_inputs = pipeline.load(batched_inputs)

        # Use shared helper to collect activations
        activations = _collect_activations_single_batch(
            intervenable_model, loaded_inputs, indices
        )

        # Process activations: pyvene 0.1.8+ returns one tensor per unit
        if len(activations) != len(model_units):
            raise ValueError(
                f"Unexpected activations format. Got {len(activations)} tensors "
                f"but expected {len(model_units)} (one per model unit)"
            )

        for activation_idx, model_unit in enumerate(model_units):
            unit_activations = activations[activation_idx]
            hidden_size = unit_activations.shape[-1]
            reshaped_activations = unit_activations.reshape(-1, hidden_size)
            data[model_unit.id].extend(reshaped_activations.cpu())

        del loaded_inputs
        del activations

    # Clean up intervenable model
    delete_intervenable_model(intervenable_model)

    # Stack collected activations into 2D tensors with shape (n_samples, n_features)
    result = {
        unit_id: torch.stack(activations) for unit_id, activations in data.items()
    }

    logger.debug(f"Collected features for {len(result)} model units")
    sample_tensor = next(iter(result.values()))
    logger.debug(f"Feature tensor shape: {sample_tensor.shape} (samples, features)")

    # Return dictionary: {unit_id -> tensor of shape (n_samples, n_features)}
    return result


def collect_source_representations(
    source_pipeline: Pipeline,
    examples: list[CounterfactualExample] | list[LabeledCounterfactualExample],
    interchange_target: InterchangeTarget,
    source_intervenable_model: IntervenableModel | None = None,
) -> list[Tensor]:
    """
    Collect activations from source pipeline for cross-model patching.

    This is a convenience wrapper around collect_batch_representations that handles
    tokenization and index computation for the source pipeline.

    Args:
        source_pipeline: Pipeline to collect activations from
        examples: List of CounterfactualExample objects
        interchange_target: InterchangeTarget specifying which locations to collect
        source_intervenable_model: Optional pre-created intervenable model for efficiency

    Returns:
        List of activation tensors for use as pyvene's source_representations
    """
    cf_inputs_raw = list(zip(*[ex["counterfactual_inputs"] for ex in examples]))
    batched_cf_for_source = [
        source_pipeline.load(list(cf_group)) for cf_group in cf_inputs_raw
    ]
    source_cf_indices = [
        model_unit.index_component(list(cf_group), batch=True, is_original=False)
        for group, cf_group in zip(interchange_target, cf_inputs_raw)
        for model_unit in group
    ]
    return collect_batch_representations(
        source_pipeline,
        batched_cf_for_source,
        interchange_target,
        source_cf_indices,
        intervenable_model=source_intervenable_model,
    )


def collect_batch_representations(
    pipeline: Pipeline,
    batched_counterfactuals: list[dict[str, Tensor]],
    interchange_target: InterchangeTarget,
    counterfactual_indices: list[Any],
    intervenable_model: IntervenableModel | None = None,
) -> list[Tensor]:
    """
    Collect activations from a single batch for use as source_representations.

    This is the primitive for cross-model patching: collect activations from
    source_pipeline, then pass them to target_pipeline via source_representations.

    Args:
        pipeline: Source pipeline to collect activations from
        batched_counterfactuals: List of tokenized counterfactual inputs (one per group).
            Each element is the output of pipeline.load() for one counterfactual group.
        interchange_target: InterchangeTarget specifying which locations to collect from.
            Groups in the target correspond to counterfactual inputs.
        counterfactual_indices: Indices for each model unit, shape (num_units, batch_size, num_positions).
            These should be computed using the SOURCE pipeline's tokenization.
        intervenable_model: Optional pre-created intervenable model configured for collection.
            If provided, this model will be used instead of creating a new one.
            This allows hoisting model creation outside batch loops for efficiency.
            If None, a model will be created and cleaned up within this function.

    Returns:
        List of activation tensors in the format expected by pyvene's source_representations
        parameter. Each tensor corresponds to one intervention location, in order matching
        intervenable_model.sorted_keys.

    Example:
        >>> # Collect from source model
        >>> source_reps = collect_batch_representations(
        ...     source_pipeline,
        ...     batched_cf_tokenized,
        ...     interchange_target,
        ...     source_cf_indices,
        ... )
        >>> # Use in target model
        >>> target_pipeline.intervenable_generate(
        ...     intervenable_model,
        ...     base_inputs,
        ...     sources=None,  # Not needed when using source_representations
        ...     source_representations=source_reps,
        ...     ...
        ... )
    """
    # Create intervenable model in collect mode if not provided
    owns_model = intervenable_model is None
    if owns_model:
        model_units = interchange_target.flatten()
        intervenable_model = prepare_intervenable_model(
            pipeline, model_units, intervention_type="collect"
        )

    # Collect activations for each counterfactual group
    # We need to run each group separately since they have different inputs
    all_activations: list[Tensor] = []

    unit_idx = 0
    for group_idx, group in enumerate(interchange_target):
        # Get the tokenized counterfactual input for this group
        cf_input = batched_counterfactuals[group_idx]

        # Get indices for units in this group
        num_units_in_group = len(group)
        group_indices = counterfactual_indices[unit_idx : unit_idx + num_units_in_group]
        unit_idx += num_units_in_group

        # Collect activations using shared helper
        activations = _collect_activations_single_batch(
            intervenable_model, cf_input, group_indices
        )

        # Extend our list with activations for this group's units
        all_activations.extend(activations)

    # Cleanup only if we created the model
    if owns_model:
        delete_intervenable_model(intervenable_model)

    return all_activations


def compute_svd(
    features_dict: dict[str, Tensor],
    n_components: int | None = None,
    normalize: bool = False,
    algorithm: Literal["arpack", "randomized"] = "randomized",
) -> dict[str, dict[str, Any]]:
    """
    Perform SVD/PCA analysis on collected features.

    Takes a dictionary of feature tensors (output from collect_features) and computes
    SVD decomposition for each. Optionally normalizes features before SVD (making it
    equivalent to PCA).

    Args:
        features_dict: Dictionary mapping model unit IDs to feature tensors.
                      Each tensor should have shape (n_samples, n_features).
        n_components: Number of SVD components to compute. If None, uses maximum
                     possible (min(n_samples, n_features) - 1).
        normalize: If True, normalize features before SVD (equivalent to PCA).
        algorithm: SVD algorithm to use. Options:
                  - "randomized": Fast randomized SVD (default)
                  - "arpack": Memory-efficient for large matrices

    Returns:
        Dictionary mapping unit IDs to SVD results. Each result contains:
        - "components": SVD components matrix of shape (n_components, n_features)
        - "explained_variance_ratio": Variance explained by each component
        - "rotation": Transposed components as torch tensor for featurizer
        - "mean": Mean used for normalization (None if normalize=False)
        - "std": Std used for normalization (None if normalize=False)

    Example:
        >>> features_dict = collect_features(dataset, pipeline, model_units)
        >>> svd_results = compute_svd(features_dict, n_components=10, normalize=True)
        >>> # Access results for a specific unit
        >>> rotation = svd_results["layer_5_pos_0"]["rotation"]
    """
    svd_results = {}

    for unit_id, features in features_dict.items():
        # Calculate maximum possible components
        n_samples, n_features = features.shape
        max_components = min(n_samples, n_features) - 1
        n = (
            min(max_components, n_components)
            if n_components is not None
            else max_components
        )

        # Store normalization parameters
        mean = None
        std = None

        # Normalize if requested (makes this equivalent to PCA)
        if normalize:
            mean = features.mean(dim=0, keepdim=True)
            std_vals = features.var(dim=0) ** 0.5
            epsilon = 1e-6  # Prevent division by zero
            std_vals = torch.clamp(std_vals, min=epsilon)
            features = (features - mean) / std_vals
            std = std_vals

        # Perform SVD
        svd = TruncatedSVD(n_components=n, algorithm=algorithm)
        svd.fit(features)

        # Extract components and create rotation matrix
        components = svd.components_.copy()  # Shape: (n_components, n_features)
        rotation = torch.tensor(components).to(features.dtype)  # Convert to torch

        # Store results
        svd_results[unit_id] = {
            "components": components,
            "explained_variance_ratio": svd.explained_variance_ratio_,
            "rotation": rotation.T,  # Transpose for featurizer (n_features, n_components)
            "mean": mean,
            "std": std,
            "n_components": n,
        }

        variance_str = [round(float(x), 4) for x in svd.explained_variance_ratio_]
        logger.debug(f"{unit_id}: explained variance = {variance_str}")

    return svd_results
