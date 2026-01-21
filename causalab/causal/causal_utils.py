"""Utility functions for working with causal models."""

from __future__ import annotations

from causalab.causal.counterfactual_dataset import CounterfactualExample
from typing import TYPE_CHECKING, Any, List, Dict, Callable, Mapping, Sequence
import copy
import itertools
import logging
import random
import numpy as np
import torch
import json


if TYPE_CHECKING:
    from causalab.causal.causal_model import CausalModel

logger = logging.getLogger(__name__)


# currently unused but not dead code - usage to come
def can_distinguish_with_dataset(
    dataset,
    causal_model1,
    target_variables1,
    causal_model2=None,
    target_variables2=None,
):
    """
    Check if two causal models can be distinguished using interchange interventions
    on a counterfactual dataset.

    Compares the outputs from running interchange interventions with target_variables1
    on causal_model1 against either:
    - Interchange interventions with target_variables2 on causal_model2 (if provided)
    - The forward pass output of causal_model1 (if causal_model2 is None)

    Parameters:
    -----------
    dataset : Dataset
        Dataset containing "input" and "counterfactual_inputs" fields.
    causal_model1 : CausalModel
        The first causal model to run interchange interventions on.
    target_variables1 : list
        List of variable names to use for interchange in the first model.
    causal_model2 : CausalModel, optional
        The second causal model to compare against (default is None).
    target_variables2 : list, optional
        List of variable names to use for interchange in the second model.
        Only used if causal_model2 is provided (default is None).

    Returns:
    --------
    dict
        A dictionary containing:
            - "proportion": The proportion of examples where outputs differ
            - "count": The number of examples where outputs differ
    """
    count = 0
    for example in dataset:
        input_data = example["input"]
        counterfactual_inputs = example["counterfactual_inputs"]
        assert len(counterfactual_inputs) == 1

        # Perform interchange intervention: copy base trace and intervene with counterfactual values
        cf_trace = counterfactual_inputs[0]
        setting1 = input_data.copy()
        for var in target_variables1:
            setting1.intervene(var, cf_trace[var])

        if causal_model2 is not None and target_variables2 is not None:
            # Perform interchange intervention on second set of variables
            setting2 = input_data.copy()
            for var in target_variables2:
                setting2.intervene(var, cf_trace[var])
            if setting1["raw_output"] != setting2["raw_output"]:
                count += 1
        else:
            # Compare against baseline (input_data is already a trace)
            if setting1["raw_output"] != input_data["raw_output"]:
                count += 1

    proportion = count / len(dataset)
    logger.debug(
        f"Can distinguish between {target_variables1} and {target_variables2}: {count} out of {len(dataset)} examples"
    )
    logger.debug(f"Proportion of distinguishable examples: {proportion:.2f}")
    return {"proportion": proportion, "count": count}


def compute_interchange_scores(
    raw_results: Dict,
    causal_model,
    datasets: Mapping[str, list[CounterfactualExample]],
    target_variables_list: List[List[str]],
    checker: Callable,
) -> Dict:
    """
    Process raw intervention results by computing scores for target variables.

    This function takes the raw outputs from perform_interventions and adds
    target-variable-specific score fields to the results dictionary. It matches
    the exact data structure that perform_interventions would create if
    target_variables_list was passed directly, allowing all existing visualization
    code to work without changes.

    This separation allows you to:
    1. Run expensive interventions once
    2. Analyze results with different target_variables combinations
    3. Experiment with different causal model interpretations post-hoc

    Args:
        raw_results: Dictionary from perform_interventions containing:
            - raw_outputs: Model generation outputs (sequences, scores, strings)
            - causal_model_inputs: Base inputs and counterfactual inputs for each example
            - metadata: Model unit metadata (layer, position, etc.)
            - feature_indices: Selected features for each model unit
        causal_model: CausalModel used to generate expected outputs via label_counterfactual_data
        datasets: Dictionary mapping dataset names to list[CounterfactualExample]
        target_variables_list: List of target variable groups to evaluate.
                              Each group is a list of variable names to intervene on.
        checker: Function with signature (output_dict, expected_label) -> score
                Used to compare model outputs against causal model expectations.

    Returns:
        Dictionary with same structure as raw_results, but with added fields for each
        target variable group:
            results["dataset"][dataset_name]["model_unit"][unit_str][target_var_str] = {
                "scores": [...],           # List of scores for each example
                "average_score": 0.85      # Mean score across all examples
            }

    Example:
        >>> # Step 1: Run interventions once (expensive)
        >>> raw_results = experiment.perform_interventions(datasets, save_dir="./results")
        >>>
        >>> # Step 2: Try different target variable combinations (cheap)
        >>> results_A = compute_interchange_scores(
        ...     raw_results, causal_model, datasets,
        ...     target_variables_list=[["A"]], checker=exact_match
        ... )
        >>> results_AB = compute_interchange_scores(
        ...     raw_results, causal_model, datasets,
        ...     target_variables_list=[["A", "B"]], checker=exact_match
        ... )
        >>>
        >>> # Step 3: Visualize both (same visualization code)
        >>> experiment.plot_heatmaps(results_A, target_variables=["A"])
        >>> experiment.plot_heatmaps(results_AB, target_variables=["A", "B"])
    """
    # Create a deep copy to avoid modifying the input
    results = copy.deepcopy(raw_results)

    # Process each dataset and model unit combination
    for dataset_name in datasets.keys():
        if dataset_name not in results["dataset"]:
            continue

        for model_units_str, model_unit_data in results["dataset"][dataset_name][
            "model_unit"
        ].items():
            if model_unit_data is None:
                continue

            # Get raw outputs and causal inputs
            raw_outputs = model_unit_data.get("raw_outputs")
            causal_model_inputs = model_unit_data.get("causal_model_inputs")

            if raw_outputs is None or causal_model_inputs is None:
                continue

            # Process and decode model outputs from batch dictionaries
            dumped_outputs = []
            flattened_outputs = []
            for batch_dict in raw_outputs:
                # Use the string field that's already in batch_dict
                batch_strings = batch_dict["string"]
                # Always treat as a list for consistent processing
                if not isinstance(batch_strings, list):
                    batch_strings = [batch_strings]

                dumped_outputs.extend(batch_strings)
                # Create individual output dicts for each example in the batch
                for idx, decoded_str in enumerate(batch_strings):
                    example_dict = {"sequences": batch_dict["sequences"][idx : idx + 1]}

                    # Handle top-K formatted scores (list of dicts)
                    if "scores" in batch_dict and batch_dict["scores"]:
                        example_dict["scores"] = []
                        for score_dict in batch_dict["scores"]:
                            sliced_score = {
                                "top_k_logits": score_dict["top_k_logits"][
                                    idx : idx + 1
                                ],
                                "top_k_indices": score_dict["top_k_indices"][
                                    idx : idx + 1
                                ],
                                "top_k_tokens": [score_dict["top_k_tokens"][idx]],
                            }
                            example_dict["scores"].append(sliced_score)

                    example_dict["string"] = decoded_str
                    flattened_outputs.append(example_dict)

            # Evaluate results for each target variable group
            # This replicates the logic from intervention_experiment.py lines 219-239
            for target_variables in target_variables_list:
                target_variable_str = "-".join(target_variables)

                # Generate expected outputs from causal model
                labeled_data = causal_model.label_counterfactual_data(
                    datasets[dataset_name], target_variables
                )

                # Validate alignment
                assert len(labeled_data) == len(dumped_outputs), (
                    f"Length mismatch: {len(labeled_data)} vs {len(dumped_outputs)}"
                )
                assert len(labeled_data) == len(flattened_outputs), (
                    f"Length mismatch: {len(labeled_data)} vs {len(flattened_outputs)}"
                )

                # Compute intervention scores - pass neural dict and expected label
                scores = []
                for example, output_dict in zip(labeled_data, flattened_outputs):
                    score = checker(output_dict, example["label"])
                    if isinstance(score, torch.Tensor):
                        score = score.item()
                    scores.append(float(score))

                # Store processed results in the same structure as perform_interventions
                results["dataset"][dataset_name]["model_unit"][model_units_str][
                    target_variable_str
                ] = {"scores": scores, "average_score": np.mean(scores)}

    return results


# ============================================================================
# Helper functions for working with counterfactual examples
# ============================================================================


def generate_counterfactual_samples(
    size: int,
    sampler: Callable[[], CounterfactualExample],
    filter: Callable[[CounterfactualExample], bool] | None = None,
) -> list[CounterfactualExample]:
    """
    Generate a list of counterfactual examples.

    Args:
        size (int): Number of examples to generate.
        sampler (callable): Function that returns a CounterfactualExample.
        filter (callable, optional): Function that takes a CounterfactualExample and returns
                                    a boolean indicating whether to include it.

    Returns:
        list[CounterfactualExample]: List of counterfactual examples.
    """
    examples = []
    while len(examples) < size:
        sample = sampler()
        if filter is None or filter(sample):
            examples.append(sample)

    return examples


def display_counterfactual_examples(
    examples: Sequence[Mapping[str, Any]],
    num_examples: int = 1,
    verbose: bool = True,
    name: str = "dataset",
) -> Dict[int, Mapping[str, Any]]:
    """
    Display examples from a list of counterfactual examples.

    Args:
        examples (list): List of counterfactual example dicts.
        num_examples (int, optional): Number of examples to display. Defaults to 1.
        verbose (bool, optional): Whether to print information. Defaults to True.
        name (str, optional): Name to display for the dataset. Defaults to "dataset".

    Returns:
        dict: A dictionary mapping indices to displayed examples.
    """
    if verbose:
        print(f"Dataset '{name}':")

    displayed_examples: Dict[int, Mapping[str, Any]] = {}

    for i in range(min(num_examples, len(examples))):
        example = examples[i]

        if verbose:
            print(f"\nExample {i + 1}:")
            print(f"Input: {example['input']}")
            print(
                f"Counterfactual Inputs ({len(example['counterfactual_inputs'])} alternatives):"
            )

            for j, counterfactual_input in enumerate(example["counterfactual_inputs"]):
                print(f"  [{j + 1}] {counterfactual_input}")

        displayed_examples[i] = example

    if verbose and len(examples) > num_examples:
        print(f"\n... {len(examples) - num_examples} more examples not shown")

    return displayed_examples


def save_counterfactual_examples(
    examples: list[CounterfactualExample],
    path: str,
) -> None:
    """
    Save a list of counterfactual examples to disk as JSON.

    Args:
        examples: List of CounterfactualExample dicts to save.
        path: File path to save the JSON to (should end in .json).
    """

    def serialize_example(ex: CounterfactualExample) -> dict[str, Any]:
        return {
            "input": ex["input"].to_dict(),
            "counterfactual_inputs": [t.to_dict() for t in ex["counterfactual_inputs"]],
        }

    serialized = [serialize_example(ex) for ex in examples]
    with open(path, "w") as f:
        json.dump(serialized, f, indent=2)


def load_counterfactual_examples(
    path: str, causal_model: "CausalModel"
) -> list[CounterfactualExample]:
    """
    Load a list of counterfactual examples from disk.

    Args:
        path: File path to load the JSON from.
        causal_model: CausalModel to use for deserializing CausalTrace objects

    Returns:
        List of CounterfactualExamples
    """
    with open(path) as f:
        data = json.load(f)
    return deserialize_counterfactual_examples(data, causal_model)


def deserialize_counterfactual_examples(
    dataset: Sequence[CounterfactualExample], causal_model: "CausalModel"
) -> list[CounterfactualExample]:
    """
    Convert dicts loaded from disk back to CausalTraces.

    When a dataset is saved to disk, CausalTrace objects are serialized to dicts.
    This function converts them back to CausalTraces using the causal model's mechanisms.

    Args:
        dataset: List of CounterfactualExample dicts (from load_counterfactual_examples).
        causal_model: CausalModel to use for reconstructing traces.

    Returns:
        List of CounterfactualExample with CausalTrace objects instead of dicts.
    """
    result = []
    for example in dataset:
        input_data = example["input"]
        cf_inputs_data = example["counterfactual_inputs"]

        # Convert to CausalTrace if it's a dict
        if isinstance(input_data, dict):
            input_trace = causal_model.new_trace(input_data)
        else:
            input_trace = input_data

        # Convert counterfactual inputs
        cf_traces = []
        for cf_data in cf_inputs_data:
            if isinstance(cf_data, dict):
                cf_traces.append(causal_model.new_trace(cf_data))
            else:
                cf_traces.append(cf_data)

        result.append({"input": input_trace, "counterfactual_inputs": cf_traces})

    return result


# ============================================================================
# Functions extracted from CausalModel class
# ============================================================================


def sample_intervention(
    model: "CausalModel", filter_func: Callable[[dict[str, Any]], bool] | None = None
) -> dict[str, Any]:
    """
    Sample a random intervention that satisfies an optional filter.

    Parameters:
    -----------
    model : CausalModel
        The causal model to sample from.
    filter_func : function, optional
        A function that takes an intervention and returns a boolean indicating
        whether it satisfies the filter (default is None).

    Returns:
    --------
    dict
        A dictionary mapping variables to their sampled intervention values.
    """
    filter_func = (
        filter_func if filter_func is not None else lambda x: len(x.keys()) > 0
    )
    intervention: dict[str, Any] = {}
    while not filter_func(intervention):
        intervention = {}
        while len(intervention.keys()) == 0:
            for var in model.variables:
                if var in model.inputs or var in model.outputs:
                    continue
                if random.choice([0, 1]) == 0:
                    intervention[var] = random.choice(model.values[var])
    return intervention


def label_data_with_variables(
    model: "CausalModel",
    data: Sequence[Mapping[str, Any]],
    target_variables: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Labels a dataset based on variable settings from running the forward model.

    Takes a dataset of inputs, runs the forward model on each input, and assigns
    a unique label ID based on the values of the specified target variables.

    Parameters:
    -----------
    model : CausalModel
        The causal model to use for labeling.
    data : list
        List containing examples with "input" field.
    target_variables : list
        List of variable names to use for labeling.

    Returns:
    --------
    tuple
        A tuple containing:
            - list[dict]: A list of dicts with "input" and "label" fields.
            - dict: A mapping from concatenated target variable values to label IDs.
    """
    traces = []
    labels = []
    label_to_setting: dict[str, int] = {}

    new_id = 0
    for example in data:
        trace = example["input"]
        # Store input
        traces.append(trace)

        target_labels = [str(trace[var]) for var in target_variables]

        # Assign or create a label ID
        label_key = "".join(target_labels)
        if label_key in label_to_setting:
            id_value = label_to_setting[label_key]
        else:
            id_value = new_id
            label_to_setting[label_key] = new_id
            new_id += 1

        labels.append(id_value)

    # Return list of dicts with input and label
    labeled_data = [
        {"input": t.to_dict() if hasattr(t, "to_dict") else t, "label": label}
        for t, label in zip(traces, labels)
    ]
    return labeled_data, label_to_setting


def generate_equiv_classes(model: "CausalModel") -> dict[str, dict[Any, list[Any]]]:
    """
    Generate equivalence classes for each variable.

    This function computes, for each non-input variable, the sets of parent values
    that produce each possible value of the variable.

    Parameters:
    -----------
    model : CausalModel
        The causal model to compute equivalence classes for.

    Returns:
    --------
    dict
        A dictionary mapping variables to their equivalence classes.
    """
    equiv_classes: dict[str, dict[Any, list[Any]]] = {}
    for var in model.variables:
        if var in model.inputs:
            continue
        equiv_classes[var] = {val: [] for val in model.values[var]}
        for parent_values in itertools.product(
            *[model.values[par] for par in model.parents[var]]
        ):
            value = model.mechanisms[var](*parent_values)
            equiv_classes[var][value].append(
                {par: parent_values[i] for i, par in enumerate(model.parents[var])}
            )
    return equiv_classes


def find_live_paths(
    model: "CausalModel", intervention: dict[str, Any]
) -> dict[int, list[list[str]]]:
    """
    Find all live causal paths in the model given an intervention.

    A live path is a sequence of variables where changing the value of one
    variable can affect the value of the next variable in the sequence.

    Parameters:
    -----------
    model : CausalModel
        The causal model to analyze.
    intervention : dict or CausalTrace
        Intervention values (dict will be converted to trace).

    Returns:
    --------
    dict
        A dictionary mapping path lengths to lists of paths.
    """
    # Create actual setting trace
    actual_setting = model.new_trace(intervention)

    paths: dict[int, list[list[str]]] = {
        1: [[variable] for variable in model.variables]
    }
    step = 2
    while True:
        paths[step] = []
        for path in paths[step - 1]:
            for child in model.children[path[-1]]:
                actual_cause = False
                for value in model.values[path[-1]]:
                    # Create counterfactual trace with intervention values
                    counterfactual_setting = model.new_trace(intervention)
                    # Intervene with new value for path variable
                    counterfactual_setting.intervene(path[-1], value)

                    if counterfactual_setting[child] != actual_setting[child]:
                        actual_cause = True
                if actual_cause:
                    paths[step].append(copy.deepcopy(path) + [child])
        if len(paths[step]) == 0:
            break
        step += 1
    del paths[1]
    return paths


def sample_input_tree_balanced(
    model: "CausalModel",
    equiv_classes: dict[str, dict[Any, list[Any]]],
    output_var: str | None = None,
    output_var_value: Any | None = None,
) -> dict[str, Any]:
    """
    Sample an input that leads to a specific output value using a balanced tree approach.

    Parameters:
    -----------
    model : CausalModel
        The causal model to sample from.
    equiv_classes : dict
        Pre-computed equivalence classes (from generate_equiv_classes).
    output_var : str, optional
        The output variable to target (default is the first output variable).
    output_var_value : any, optional
        The desired value for the output variable (default is a random choice).

    Returns:
    --------
    dict
        A dictionary mapping input variables to their sampled values.
    """
    assert output_var is not None or len(model.outputs) == 1

    if output_var is None:
        output_var = model.outputs[0]
    if output_var_value is None:
        output_var_value = random.choice(model.values[output_var])

    def create_input(
        var: str, value: Any, input_dict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if input_dict is None:
            input_dict = {}
        parent_values = random.choice(equiv_classes[var][value])
        for parent in parent_values:
            if parent in model.inputs:
                input_dict[parent] = parent_values[parent]
            else:
                create_input(parent, parent_values[parent], input_dict)
        return input_dict

    input_setting = create_input(output_var, output_var_value)
    for input_var in model.inputs:
        if input_var not in input_setting:
            input_setting[input_var] = random.choice(model.values[input_var])
    return input_setting


def get_path_maxlen_filter(
    model: "CausalModel", lengths: list[int] | set[int]
) -> Callable[[dict[str, Any]], bool]:
    """
    Get a filter function that checks if the maximum length of any live path
    is in a given set of lengths.

    Parameters:
    -----------
    model : CausalModel
        The causal model to use for path finding.
    lengths : list or set
        A list or set of path lengths to check against.

    Returns:
    --------
    function
        A filter function that takes a setting and returns a boolean.
    """

    def check_path(total_setting: dict[str, Any]) -> bool:
        input_setting = {var: total_setting[var] for var in model.inputs}
        paths = find_live_paths(model, input_setting)
        max_len = max([length for length in paths.keys() if len(paths[length]) != 0])
        if max_len in lengths:
            return True
        return False

    return check_path


def get_partial_filter(
    partial_setting: dict[str, Any],
) -> Callable[[dict[str, Any]], bool]:
    """
    Get a filter function that checks if a setting matches a partial setting.

    Parameters:
    -----------
    partial_setting : dict
        A dictionary mapping variables to their desired values.

    Returns:
    --------
    function
        A filter function that takes a setting and returns a boolean.
    """

    def compare(total_setting: dict[str, Any]) -> bool:
        for var in partial_setting:
            if total_setting[var] != partial_setting[var]:
                return False
        return True

    return compare


def get_specific_path_filter(
    model: "CausalModel", start: str, end: str
) -> Callable[[dict[str, Any]], bool]:
    """
    Get a filter function that checks if there is a live path from a start
    variable to an end variable.

    Parameters:
    -----------
    model : CausalModel
        The causal model to use for path finding.
    start : str
        The start variable of the path.
    end : str
        The end variable of the path.

    Returns:
    --------
    function
        A filter function that takes a setting and returns a boolean.
    """

    def check_path(total_setting: dict[str, Any]) -> bool:
        input_setting = {var: total_setting[var] for var in model.inputs}
        paths = find_live_paths(model, input_setting)
        for k in paths:
            for path in paths[k]:
                if path[0] == start and path[-1] == end:
                    return True
        return False

    return check_path
