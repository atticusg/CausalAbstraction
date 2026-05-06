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
from tqdm import tqdm

from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from pyvene import IntervenableModel  # type: ignore[reportMissingTypeStubs]

from causalab.neural.activations.intervenable_model import (
    prepare_intervenable_model,
    delete_intervenable_model,
)
from causalab.neural.pipeline import Pipeline
from causalab.neural.units import AtomicModelUnit, InterchangeTarget

logger = logging.getLogger(__name__)


def _collect_activations_single_batch(
    intervenable_model: IntervenableModel,
    loaded_inputs: dict[str, Tensor],
    indices: list[Any],
    return_model_output: bool = False,
) -> list[Tensor] | tuple[list[Tensor], Any]:
    """
    Collect activations from a single batch using an intervenable model.

    This is the core primitive for activation collection, used by both
    collect_features() for dataset-wide collection and collect_batch_representations()
    for cross-model patching.

    Args:
        intervenable_model: Model configured with "collect" intervention type
        loaded_inputs: Tokenized inputs (output of pipeline.load())
        indices: Position indices for each model unit, shape (num_units, batch_size, num_positions)
        return_model_output: If True, also return the model's forward pass output
            (e.g., CausalLMOutputWithPast with .logits).

    Returns:
        If return_model_output is False:
            List of activation tensors, one per intervention location.
        If return_model_output is True:
            Tuple of (activations_list, model_output).
    """
    # Create location map - for collection, source and base indices are the same
    location_map = {"sources->base": (indices, indices)}

    # Run collection pass - pyvene returns ((base_outputs, collected_activations), cf_outputs)
    result = intervenable_model(loaded_inputs, unit_locations=location_map)
    activations = result[0][1]

    if return_model_output:
        return activations, result[0][0]
    return activations


def collect_features(
    dataset: list[CounterfactualExample],
    pipeline: Pipeline,
    model_units: list[AtomicModelUnit],
    batch_size: int = 32,
    collect_output_logits: bool = False,
) -> dict[str, Tensor] | tuple[dict[str, Tensor], Tensor]:
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
        collect_output_logits: If True, also capture the model's full output
            logits for each example. This avoids a redundant forward pass when
            you need both intermediate activations and output distributions
            (e.g., for reference distribution computation).

    Returns:
        If collect_output_logits is False (default):
            Dict mapping model unit IDs to feature tensors of shape (n_samples, n_features).
        If collect_output_logits is True:
            Tuple of (features_dict, output_logits) where output_logits is a
            list of per-example logit tensors (each of shape (seq_len, vocab_size),
            since sequence lengths may vary across batches).

    Example:
        >>> features_dict = collect_features(dataset, pipeline, model_units, batch_size=32)
        >>> features_dict, logits = collect_features(
        ...     dataset, pipeline, model_units, collect_output_logits=True
        ... )
    """
    # Initialize model with "collect" intervention type (extracts activations without modifying them)
    # prepare_intervenable_model auto-wraps flat lists into InterchangeTarget
    intervenable_model = prepare_intervenable_model(
        pipeline, model_units, intervention_type="collect"
    )

    # Initialize container for collected features: one list per model unit
    # Use model unit IDs as keys to handle duplicates gracefully
    data: dict[str, list[Tensor]] = {model_unit.id: [] for model_unit in model_units}
    output_logits_list: list[Tensor] = []

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
        result = _collect_activations_single_batch(
            intervenable_model,
            loaded_inputs,
            indices,
            return_model_output=collect_output_logits,
        )

        if collect_output_logits:
            activations, model_output = result
            # Store full logits per example (seq lengths may vary across batches)
            batch_logits = model_output.logits.detach().cpu()
            for j in range(batch_logits.shape[0]):
                output_logits_list.append(batch_logits[j])
            del model_output
        else:
            activations = result

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
    features_dict = {
        unit_id: torch.stack(activations) for unit_id, activations in data.items()
    }

    logger.debug(f"Collected features for {len(features_dict)} model units")
    sample_tensor = next(iter(features_dict.values()))
    logger.debug(f"Feature tensor shape: {sample_tensor.shape} (samples, features)")

    if collect_output_logits:
        logger.debug(f"Collected output logits for {len(output_logits_list)} examples")
        return features_dict, output_logits_list

    return features_dict


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


def collect_class_centroids(
    filtered_samples: list[dict],
    pipeline,
    interchange_target,
    task,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect features and compute per-variable-value centroids.

    Features are collected through whatever featurizer is currently set on
    the interchange_target. To get raw-space centroids, set identity featurizer
    first. To get PCA-space centroids, set PCA featurizer first.

    Args:
        filtered_samples: Dataset examples.
        pipeline: Model pipeline.
        interchange_target: Target with featurizer loaded.
        task: Task object (for intervention_variable and variable_values).

    Returns:
        (centroids, valid_mask) where centroids is (n_classes, k) and
        valid_mask is (n_classes,) boolean indicating which classes had samples.
    """
    units = interchange_target.flatten()
    features_dict = collect_features(
        dataset=filtered_samples,
        pipeline=pipeline,
        model_units=units,
        batch_size=32,
    )
    features = features_dict[units[0].id].detach().float()

    steered_variable = task.intervention_variable
    values = task.intervention_values
    value_to_idx = {v: i for i, v in enumerate(values)}

    n_classes = len(values)
    k = features.shape[1]
    centroids = torch.zeros(n_classes, k)
    counts = torch.zeros(n_classes)

    for i, sample in enumerate(filtered_samples):
        val = sample["input"][steered_variable]
        if val in value_to_idx:
            idx = value_to_idx[val]
            centroids[idx] += features[i]
            counts[idx] += 1

    mask = counts > 0
    centroids[mask] = centroids[mask] / counts[mask].unsqueeze(1)

    return centroids, mask
