"""
collect.py
==========
Functions for collecting and analyzing neural network activations.

This module provides utilities for processing collected features from model units,
including dimensionality reduction techniques like SVD/PCA.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from causalab.neural.pyvene_core.intervenable_model import (
    prepare_intervenable_model,
    delete_intervenable_model,
)
from causalab.neural.pipeline import Pipeline
from causalab.neural.model_units import AtomicModelUnit

logger = logging.getLogger(__name__)


def collect_features(
    dataloader: DataLoader[dict[str, Any]],
    pipeline: Pipeline,
    model_units: list[AtomicModelUnit],
) -> dict[str, Tensor]:
    """
    Collect internal neural network activations (features) at specified model locations.

    This function:
    1. Creates an intervenable model configured for feature collection
    2. Processes batches from the dataloader to extract activations at target locations
    3. Returns a dictionary mapping each model unit ID to its collected features

    Args:
        dataloader: PyTorch DataLoader providing batches of input data. Each batch should
                   have an "input" key containing the data to process.
        pipeline: Neural model pipeline for processing inputs
        model_units (List[AtomicModelUnit]): Flat list of model units to collect features from

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping model unit IDs to feature tensors.
                                Each tensor has shape (n_samples, n_features) containing
                                the activations for all inputs in the dataset.

    Example:
        >>> from causalab.neural.dataloader import create_dataloader
        >>> from causalab.neural.collect import collect_features
        >>>
        >>> dataloader = create_dataloader(dataset, batch_size=32)
        >>> features_dict = collect_features(dataloader, pipeline, model_units)
    """
    # Initialize model with "collect" intervention type (extracts activations without modifying them)
    # prepare_intervenable_model auto-wraps flat lists into InterchangeTarget
    intervenable_model = prepare_intervenable_model(
        pipeline, model_units, intervention_type="collect"
    )

    # Initialize container for collected features: one list per model unit
    # Use model unit IDs as keys to handle duplicates gracefully
    data = {model_unit.id: [] for model_unit in model_units}

    # Process dataset in batches with progress tracking
    for batch in tqdm(dataloader, desc="Processing batches", leave=False):
        # Get inputs from batch
        batched_inputs = batch["input"]

        # Compute indices for each model unit
        indices = [
            model_unit.index_component(batched_inputs, batch=True, is_original=True)
            for model_unit in model_units
        ]

        # Load inputs through pipeline
        loaded_inputs = pipeline.load(batched_inputs)

        # Create mapping for base input activations (identical source and target)
        location_map = {"sources->base": (indices, indices)}

        # Collect activations from base inputs
        # Returns a list of activation tensors, one per model unit
        # In pyvene 0.1.8+, each tensor contains all batch samples for that unit
        activations = intervenable_model(loaded_inputs, unit_locations=location_map)[0][
            1
        ]

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
    data = {unit_id: torch.stack(activations) for unit_id, activations in data.items()}

    logger.debug(f"Collected features for {len(data)} model units")
    sample_tensor = next(iter(data.values()))
    logger.debug(f"Feature tensor shape: {sample_tensor.shape} (samples, features)")

    # Return dictionary: {unit_id -> tensor of shape (n_samples, n_features)}
    return data


def compute_svd(
    features_dict: dict[str, Tensor],
    n_components: int | None = None,
    normalize: bool = False,
    algorithm: str = "randomized",
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
