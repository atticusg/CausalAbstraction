"""
Filter counterfactual examples based on agreement between neural and causal models.

This module provides functionality to filter examples by removing those where
the neural pipeline and causal model outputs disagree.
"""

from typing import Callable, Any

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace
from causalab.neural.pipeline import LMPipeline
from causalab.causal.causal_model import CausalModel


def filter_dataset(
    dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    causal_model: CausalModel,
    metric: Callable[[Any, Any], bool],
    batch_size: int = 32,
    validate_counterfactuals: bool = True,
) -> list[CounterfactualExample]:
    """
    Filter dataset based on agreement between pipeline and causal model outputs.

    For each example in the dataset, checks if:
    1. The pipeline's prediction on the original input matches the causal model's output
    2. (Optional) The pipeline's predictions on all counterfactual inputs match the causal model's outputs

    Only examples where both conditions are met are kept in the filtered dataset.

    Args:
        dataset: List of CounterfactualExample to filter
        pipeline: Neural model pipeline that processes inputs
        causal_model: Causal model that generates expected outputs
        metric: Function that compares neural output with causal output,
                returning True if they match
        batch_size: Size of batches for processing
        validate_counterfactuals: If True, validates counterfactual outputs.
                                 If False, only validates base inputs.

    Returns:
        Filtered list of CounterfactualExample with examples that pass validation
    """
    dataset_original = len(dataset)

    # Use pipeline to compute all outputs at once (returns flattened per-example outputs)
    outputs = pipeline.compute_outputs(dataset, batch_size=batch_size)

    base_outputs_flat = outputs["base_outputs"]
    counterfactual_outputs_flat = (
        outputs["counterfactual_outputs"] if validate_counterfactuals else []
    )

    # Validate each example
    filtered_examples: list[CounterfactualExample] = []

    # Get number of counterfactuals per example (assumes all examples have same count)
    num_cf_per_example = (
        len(dataset[0]["counterfactual_inputs"]) if len(dataset) > 0 else 0
    )

    cf_idx = 0  # Track position in flattened counterfactual outputs

    for example_idx in range(dataset_original):
        example = dataset[example_idx]
        input_trace: CausalTrace = example["input"]

        # Validate base input
        base_output = base_outputs_flat[example_idx]
        # Get expected output from the trace (already computed by causal model)
        base_expected = input_trace["raw_output"]

        if not metric(base_output, base_expected):
            # Skip counterfactuals if base fails
            cf_idx += num_cf_per_example
            continue

        # Validate counterfactual inputs if required
        if validate_counterfactuals and num_cf_per_example > 0:
            cf_valid = True
            for i in range(num_cf_per_example):
                cf_trace: CausalTrace = example["counterfactual_inputs"][i]
                cf_output = counterfactual_outputs_flat[cf_idx + i]
                # Get expected output from the trace
                cf_expected = cf_trace["raw_output"]

                if not metric(cf_output, cf_expected):
                    cf_valid = False
                    break

            cf_idx += num_cf_per_example

            if not cf_valid:
                continue

        # Example passed validation
        filtered_examples.append(example)

    return filtered_examples
