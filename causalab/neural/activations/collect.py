"""
collect.py
==========
Functions for collecting and analyzing neural network activations.

This module provides utilities for processing collected features from model units,
including dimensionality reduction techniques like SVD/PCA.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch import Tensor
from tqdm import tqdm

from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from causalab.neural.activations.engine import (
    build_plans,
    collect_unit_activations,
)
from causalab.neural.pipeline import Pipeline
from causalab.neural.units import AtomicModelUnit, InterchangeTarget

logger = logging.getLogger(__name__)


def collect_features(
    dataset: list[CounterfactualExample],
    pipeline: Pipeline,
    model_units: list[AtomicModelUnit],
    batch_size: int = 32,
    collect_output_logits: bool = False,
) -> dict[str, Tensor] | tuple[dict[str, Tensor], list[Tensor]]:
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
    # Raise on duplicate unit IDs: `data` below is keyed by `model_unit.id` so
    # duplicates would silently merge (the per-batch loop indexes by position
    # in `model_units`, leaving the merged dict entry with only the last
    # unit's activations). Downstream consumers — `methods/pca.py`,
    # `methods/spline/train.py`, `analyses/subspace/das.py` — all assume the
    # returned dict has one entry per unit.
    seen_ids: set[str] = set()
    for u in model_units:
        if u.id in seen_ids:
            raise ValueError(
                f"Duplicate model_unit.id={u.id!r} in model_units; ids must be unique"
            )
        seen_ids.add(u.id)

    # Pin eval mode + no_grad: collection is a forward-only operation. Without
    # these, autograd builds a graph and dropout / batchnorm-like layers stay
    # in train mode, leaking gradients and randomness into the cached
    # activations (which then get consumed by every featurizer / interchange
    # learner downstream as if they were deterministic).
    pipeline.model.eval()

    # Initialize container for collected features: one list per model unit
    data: dict[str, list[Tensor]] = {model_unit.id: [] for model_unit in model_units}
    output_logits_list: list[Tensor] = []

    # Process dataset in batches with progress tracking
    with torch.no_grad():
        for start in tqdm(
            range(0, len(dataset), batch_size),
            desc="Processing batches",
            leave=False,
        ):
            batch = dataset[start : start + batch_size]
            # Get inputs from batch
            batched_inputs = [example["input"] for example in batch]

            # Load inputs first so the padded `attention_mask` is available for
            # shifting unpadded position indices into the padded-batch frame.
            loaded_inputs = pipeline.load(batched_inputs)

            positions = [
                model_unit.resolve_positions(
                    batched_inputs,
                    attention_mask=loaded_inputs["attention_mask"],
                    is_original=True,
                )
                for model_unit in model_units
            ]

            plans = build_plans(model_units, positions, "collect")
            result = collect_unit_activations(
                pipeline, loaded_inputs, plans, return_logits=collect_output_logits
            )

            if collect_output_logits:
                assert isinstance(result, tuple)
                activations, batch_logits = result
                # Store full logits per example (seq lengths may vary across batches)
                batch_logits = batch_logits.detach().cpu()
                for j in range(batch_logits.shape[0]):
                    output_logits_list.append(batch_logits[j])
            else:
                assert isinstance(result, list)
                activations = result

            # One activation per unit — a (batch, n_positions, width) tensor.
            # Flattening the position axis makes each (example, position) pair a
            # sample, which is what every downstream featurizer fitter expects.
            for activation_idx, model_unit in enumerate(model_units):
                unit_activations = activations[activation_idx]
                hidden_size = unit_activations.shape[-1]
                reshaped_activations = unit_activations.reshape(-1, hidden_size)
                data[model_unit.id].extend(reshaped_activations.cpu())

            del loaded_inputs
            del activations

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
) -> list[Tensor]:
    """
    Collect activations from source pipeline for cross-model patching.

    This is a convenience wrapper around collect_batch_representations that handles
    tokenization and position resolution for the source pipeline.

    Args:
        source_pipeline: Pipeline to collect activations from
        examples: List of CounterfactualExample objects
        interchange_target: InterchangeTarget specifying which locations to collect

    Returns:
        One activation tensor per unit, in ``interchange_target.flatten()`` order,
        ready to be written as the sources of an interchange.
    """
    cf_inputs_raw = [
        list(group) for group in zip(*[ex["counterfactual_inputs"] for ex in examples])
    ]
    batched_cf_for_source = [
        source_pipeline.load(cf_group) for cf_group in cf_inputs_raw
    ]
    source_cf_positions = [
        model_unit.resolve_positions(
            cf_inputs_raw[group_idx],
            attention_mask=batched_cf_for_source[group_idx]["attention_mask"],
            is_original=False,
        )
        for group_idx, group in enumerate(interchange_target)
        for model_unit in group
    ]
    return collect_batch_representations(
        source_pipeline,
        batched_cf_for_source,
        interchange_target,
        source_cf_positions,
    )


def collect_batch_representations(
    pipeline: Pipeline,
    batched_counterfactuals: list[dict[str, Tensor]],
    interchange_target: InterchangeTarget,
    counterfactual_positions: list[Any],
) -> list[Tensor]:
    """
    Collect activations from a single batch for use as intervention sources.

    This is the primitive for cross-model patching: collect activations from
    source_pipeline, then write them into target_pipeline's run.

    Each counterfactual group is its own forward, because each has its own input.

    Args:
        pipeline: Source pipeline to collect activations from
        batched_counterfactuals: List of tokenized counterfactual inputs (one per group).
            Each element is the output of pipeline.load() for one counterfactual group.
        interchange_target: InterchangeTarget specifying which locations to collect from.
            Groups in the target correspond to counterfactual inputs.
        counterfactual_positions: Token positions for each model unit, shape
            (num_units, batch_size, num_positions), computed against the SOURCE
            pipeline's tokenization.

    Returns:
        One activation tensor per unit, in ``interchange_target.flatten()`` order.

    Example:
        >>> source_reps = collect_batch_representations(
        ...     source_pipeline,
        ...     batched_cf_tokenized,
        ...     interchange_target,
        ...     source_cf_positions,
        ... )
    """
    all_activations: list[Tensor] = []

    unit_idx = 0
    for group_idx, group in enumerate(interchange_target):
        num_units_in_group = len(group)
        group_positions = counterfactual_positions[
            unit_idx : unit_idx + num_units_in_group
        ]
        unit_idx += num_units_in_group

        plans = build_plans(list(group), group_positions, "collect")
        activations = collect_unit_activations(
            pipeline, batched_counterfactuals[group_idx], plans
        )
        assert isinstance(activations, list)
        all_activations.extend(activations)

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
    # filtered_samples is list[dict]; CounterfactualExample is a TypedDict with
    # the same runtime shape, so the cast is safe.
    features_dict = cast(
        "dict[str, Tensor]",
        collect_features(
            dataset=cast("list[CounterfactualExample]", filtered_samples),
            pipeline=pipeline,
            model_units=units,
            batch_size=32,
        ),
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
