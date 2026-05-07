"""
interchange.py
==============
Core utilities for running interchange intervention experiments.

This module provides functions for running interventions on neural networks using
the pyvene library. It focuses on interchange interventions where activations are
swapped between base and counterfactual inputs.

For training intervention models, see train.py.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from tqdm import tqdm
from pyvene import IntervenableModel  # type: ignore[import-untyped]

from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from causalab.neural.pipeline import Pipeline
from causalab.neural.units import InterchangeTarget
from causalab.neural.activations.intervenable_model import (
    prepare_intervenable_model,
    delete_intervenable_model,
)
from causalab.neural.activations.collect import collect_source_representations
from causalab.neural.activations.data_utils import (
    convert_to_top_k,
    move_outputs_to_cpu,
)

# Configure logging
logger = logging.getLogger(__name__)


def prepare_intervenable_inputs(
    pipeline: Pipeline,
    examples: list[CounterfactualExample] | list[LabeledCounterfactualExample],
    interchange_target: InterchangeTarget,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[list[list[int] | None]],
]:
    """
    Prepare the inputs for the intervenable model.

    This function loads the base and counterfactual inputs, and prepares the indices
    for the model units.

    Args:
        pipeline: The pipeline containing the model
        examples: List of counterfactual examples, each with "input" and
            "counterfactual_inputs" keys.
        interchange_target: InterchangeTarget containing the model units to be intervened on.
            Groups in the target correspond to counterfactual inputs.

    Returns:
        Tuple of (batched_base, batched_counterfactuals, inv_locations, feature_indices)
    """
    # Extract and batch inputs from examples
    batched_base = [ex["input"] for ex in examples]
    # Shape: (batch_size, num_counterfactuals) -> (num_counterfactuals, batch_size)
    # Convert zip tuples to lists for pipeline.load() compatibility
    batched_counterfactuals = [
        list(cf_tuple)
        for cf_tuple in zip(*[ex["counterfactual_inputs"] for ex in examples])
    ]

    # shape: (num_model_units, batch_size, num_component_indices)
    base_indices = [
        model_unit.index_component(batched_base, batch=True, is_original=True)
        for group in interchange_target
        for model_unit in group
    ]

    # shape: (num_model_units, batch_size, num_component_indices)
    counterfactual_indices = [
        model_unit.index_component(
            batched_counterfactual, batch=True, is_original=False
        )
        for group, batched_counterfactual in zip(
            interchange_target, batched_counterfactuals
        )
        for model_unit in group
    ]

    # shape: (num_model_units, batch_size, num_feature_indices)
    feature_indices = [
        [model_unit.get_feature_indices() for _ in range(len(batched_base))]
        for group in interchange_target
        for model_unit in group
    ]

    batched_base = pipeline.load(batched_base)
    batched_counterfactuals = [
        pipeline.load(batched_counterfactual)
        for batched_counterfactual in batched_counterfactuals
    ]

    inv_locations = {"sources->base": (counterfactual_indices, base_indices)}
    return batched_base, batched_counterfactuals, inv_locations, feature_indices


def batched_interchange_intervention(
    pipeline: Pipeline,
    intervenable_model: IntervenableModel,
    examples: list[CounterfactualExample],
    interchange_target: InterchangeTarget,
    output_scores: bool | int = True,
    source_pipeline: Pipeline | None = None,
    source_intervenable_model: IntervenableModel | None = None,
) -> dict[str, Any]:
    """
    Perform interchange interventions on batched inputs using an intervenable model.

    This function executes the core intervention logic by:
    1. Preparing the base and counterfactual inputs for intervention
    2. Running the model with interventions at specified locations
    3. Moving tensors back to CPU to free GPU memory

    Args:
        pipeline: Target pipeline where interventions are applied
        intervenable_model: PyVENE model with preset intervention locations
        examples: List of counterfactual examples
        interchange_target: InterchangeTarget containing model components to intervene on
        output_scores: Whether to include scores in output dictionary (default: True)
        source_pipeline: If provided, collect activations from this pipeline instead
            of the target pipeline. Enables cross-model patching.
        source_intervenable_model: Optional pre-created intervenable model for the source
            pipeline. If provided with source_pipeline, this model will be reused for
            collecting activations instead of creating a new one per batch.

    Returns:
        dict: Dictionary with 'sequences' and optionally 'scores' keys
    """
    # Prepare inputs for intervention (using target pipeline for base)
    batched_base, batched_counterfactuals, inv_locations, feature_indices = (
        prepare_intervenable_inputs(pipeline, examples, interchange_target)
    )

    # Collect source representations if using cross-model patching
    source_representations = None
    if source_pipeline is not None:
        source_representations = collect_source_representations(
            source_pipeline, examples, interchange_target, source_intervenable_model
        )
        # When using cross-model patching, we don't pass counterfactuals to pyvene
        # because we're using pre-collected source_representations instead
        batched_counterfactuals = None

    # Execute the intervention via the pipeline
    gen_kwargs = {"output_scores": output_scores}
    output = pipeline.intervenable_generate(
        intervenable_model,
        batched_base,
        batched_counterfactuals,
        inv_locations,
        feature_indices,
        source_representations=source_representations,
        **gen_kwargs,
    )

    # Move tensors to CPU to free GPU memory
    if batched_counterfactuals is not None:
        for batched in [batched_base] + batched_counterfactuals:
            for k, v in batched.items():
                batched[k] = v.cpu()
    else:
        for k, v in batched_base.items():
            batched_base[k] = v.cpu()

    return output


def run_interchange_interventions(
    pipeline: Pipeline,
    counterfactual_dataset: list[CounterfactualExample],
    interchange_target: InterchangeTarget,
    batch_size: int = 32,
    output_scores: bool | int = True,
    source_pipeline: Pipeline | None = None,
) -> dict[str, list[Any]]:
    """
    Run interchange interventions on a full counterfactual dataset in batches.

    This function:
    1. Prepares an intervenable model configured for interchange interventions
    2. Processes the dataset in batches, applying interventions to each batch
    3. Converts scores to top-k format if requested (for memory efficiency)
    4. Moves all outputs to CPU to free GPU memory
    5. Collects and returns results from all batches

    Args:
        pipeline: Target pipeline where interventions are applied
        counterfactual_dataset: List of counterfactual examples
        interchange_target: InterchangeTarget containing model components to intervene on,
                           where groups share counterfactual inputs
        batch_size: Number of examples to process in each batch
        output_scores: Controls score output format:
            - False: No scores
            - True: Full vocabulary scores (on CPU)
            - int (e.g., 10): Top-k scores (on CPU, memory efficient)
        source_pipeline: If provided, collect activations from this pipeline instead
            of the target pipeline. Enables cross-model patching where activations
            from source_pipeline are patched into pipeline (the target).

    Returns:
        List[dict]: List of dictionaries, each with 'sequences' (on CPU) and optionally
                   'scores' keys (on CPU, in top-k format if int was provided)
    """
    # Initialize intervenable model with interchange intervention type
    intervenable_model = prepare_intervenable_model(
        pipeline, interchange_target, intervention_type="interchange"
    )

    # Create source intervenable model if doing cross-model patching
    source_intervenable_model = None
    if source_pipeline is not None:
        source_intervenable_model = prepare_intervenable_model(
            source_pipeline, interchange_target, intervention_type="collect"
        )

    all_outputs = []

    # Process each batch with progress tracking
    for start in tqdm(
        range(0, len(counterfactual_dataset), batch_size),
        desc="Processing batches",
        disable=not logger.isEnabledFor(logging.DEBUG),
        leave=False,
    ):
        examples = counterfactual_dataset[start : start + batch_size]
        with torch.no_grad():  # Disable gradient tracking for inference
            # Perform interchange interventions on the batch - returns dict
            output_dict = batched_interchange_intervention(
                pipeline,
                intervenable_model,
                examples,
                interchange_target,
                output_scores=output_scores,
                source_pipeline=source_pipeline,
                source_intervenable_model=source_intervenable_model,
            )

            # Collect outputs from this batch
            all_outputs.append(output_dict)

    # Clean up the intervenable models to free GPU memory
    delete_intervenable_model(intervenable_model)
    if source_intervenable_model is not None:
        delete_intervenable_model(source_intervenable_model)

    # Convert to top-k format if requested (while still on GPU for efficiency)
    if not isinstance(output_scores, bool) and output_scores > 0:
        all_outputs = convert_to_top_k(all_outputs, pipeline, k=output_scores)

    # Move all outputs to CPU
    all_outputs = move_outputs_to_cpu(all_outputs)

    # remove batch structure from outputs
    if not all_outputs:
        return {"sequences": [[]], "string": [[]]}
    all_outputs = {
        k: [output[k] for output in all_outputs] for k in all_outputs[0].keys()
    }

    return all_outputs
