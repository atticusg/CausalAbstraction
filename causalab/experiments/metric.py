"""
Metric computation utilities for intervention experiments.

This module provides functions for computing scores from intervention outputs,
supporting metrics that compare intervention results against causal expectations
and original model behavior.

Key abstractions:
- InterchangeMetric: Dataclass defining a scoring function and its data requirements
- make_causal_metric: Helper to create an InterchangeMetric for causal scoring
- score_intervention_outputs: Core function to score pre-computed intervention outputs
"""

from dataclasses import dataclass
from typing import Dict, List, Callable, Tuple, Any
import copy
import torch
from causalab.neural.pyvene_core.interchange import prepare_intervenable_inputs
from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import CausalTrace, Mechanism

from causalab.neural.pipeline import LMPipeline
from causalab.neural.model_units import InterchangeTarget
from causalab.neural.pyvene_core.collect import collect_source_representations


def causal_score_intervention_outputs(
    raw_results: Dict[Tuple[Any, ...], Dict[str, Any]],
    dataset: list[CounterfactualExample],
    causal_model: CausalModel,
    target_variable_groups: List[Tuple[str, ...]],
    metric: Callable[[Any, Any], bool],
) -> Dict[str, Any]:
    """
    Score pre-computed intervention outputs using multiple variable groups.

    This function takes raw intervention results (from run_interchange_interventions)
    and scores them against causal model expectations for each variable group.

    Args:
        raw_results: Dict mapping keys to intervention outputs, as returned by
                    run_interchange_interventions(). Each value has {"string": [...], ...}
        dataset: list[CounterfactualExample] used for interventions
        causal_model: Causal model for generating expected outputs
        target_variable_groups: List of variable groups to evaluate. Each group is a tuple
                               of variable names that are evaluated jointly.
                               (e.g., [("answer",), ("answer", "position")])
        metric: Comparison function(output_dict, expected) -> bool

    Returns:
        Dictionary containing:
            - results_by_key: dict with results per target key
            - scores_by_variable: dict of average scores per variable group (across all keys)
            - avg_score: overall average score (across all keys and variables)
    """
    # Create a metric for each variable group and score
    scores_by_variable: Dict[Tuple[str, ...], Dict[Tuple[Any, ...], float]] = {}
    for var_group in target_variable_groups:
        # Create metric for this variable group
        interchange_metric = make_causal_metric(metric, var_group)

        # Score using the core scoring function
        scores = score_intervention_outputs(
            raw_results=raw_results,
            dataset=dataset,
            metric=interchange_metric,
            causal_model=causal_model,
        )
        scores_by_variable[var_group] = scores

    # Build results_by_key structure
    results_by_key = {}
    for key in raw_results.keys():
        scores_for_key = {
            str(var_group): scores_by_variable[var_group][key]
            for var_group in target_variable_groups
        }
        key_avg_score = float(sum(scores_for_key.values()) / len(scores_for_key))

        results_by_key[key] = {
            "scores_by_variable": scores_for_key,
            "avg_score": key_avg_score,
            "raw_results": raw_results[key],
        }

    # Compute overall scores per variable group
    overall_scores_by_variable: Dict[Tuple[str, ...], float] = {}
    for var_group in target_variable_groups:
        overall_scores_by_variable[var_group] = float(
            sum(scores_by_variable[var_group].values())
            / len(scores_by_variable[var_group])
        )

    avg_score = float(
        sum(overall_scores_by_variable.values()) / len(overall_scores_by_variable)
    )

    return {
        "results_by_key": results_by_key,
        "scores_by_variable": overall_scores_by_variable,
        "avg_score": avg_score,
    }


@dataclass
class InterchangeMetric:
    """
    Metric for scoring interchange interventions.

    The metric function always receives 3 arguments:
    - intervention_output: Dict with intervention result (e.g., {"string": "A"})
    - expected: Dict with causal model expected output (empty dict if needs_causal_expected=False)
    - original: Dict with original model output (empty dict if needs_original_output=False)

    Attributes:
        fn: Scoring function (intervention_output, expected, original) -> float
        needs_causal_expected: Whether metric requires causal model expected outputs
        needs_original_output: Whether metric requires original model outputs (no intervention)

    Example:
        # Causal scoring metric (compares to expected output)
        def causal_checker(intervention_output, expected, original):
            return 1.0 if intervention_output["string"] == expected["string"] else 0.0

        causal_metric = InterchangeMetric(
            fn=causal_checker,
            needs_causal_expected=True,
            needs_original_output=False,
        )

        # Faithfulness metric (compares to original behavior)
        def faithfulness_checker(intervention_output, expected, original):
            return 1.0 if intervention_output["string"] == original["string"] else 0.0

        faithfulness_metric = InterchangeMetric(
            fn=faithfulness_checker,
            needs_causal_expected=False,
            needs_original_output=True,
        )
    """

    fn: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], float]
    needs_causal_expected: bool = True
    needs_original_output: bool = False
    target_variables: Tuple[str, ...] | None = None


def make_causal_metric(
    checker: Callable[[Dict[str, Any], Any], bool],
    target_variables: Tuple[str, ...],
) -> InterchangeMetric:
    """
    Create an InterchangeMetric for causal scoring.

    This helper creates a metric that compares intervention outputs against
    expected outputs from a causal model. The checker function is wrapped
    to match the InterchangeMetric signature.

    Args:
        checker: Function(output_dict, expected) -> bool for comparison.
                 - output_dict: {"string": "A", ...} intervention result
                 - expected: causal model expected output (e.g., {"string": "A"} or "A")
        target_variables: Tuple of variable names to label with causal model
                         (e.g., ("answer",) or ("answer", "position"))

    Returns:
        InterchangeMetric configured for causal scoring.

    Example:
        ```python
        from causalab.tasks.MCQA.checker import checker

        metric = make_causal_metric(checker, target_variables=("answer",))

        # Use with custom_score_interventions or score_intervention_outputs
        scores = score_intervention_outputs(
            raw_results=raw_results,
            dataset=dataset,
            metric=metric,
            causal_model=causal_model,
        )
        ```
    """

    def causal_metric_fn(
        intervention_output: Dict[str, Any],
        expected: Dict[str, Any],
        original: Dict[str, Any],
    ) -> float:
        # The checker expects (output_dict, expected) - ignore original
        result = checker(intervention_output, expected)
        return 1.0 if result else 0.0

    return InterchangeMetric(
        fn=causal_metric_fn,
        needs_causal_expected=True,
        needs_original_output=False,
        target_variables=target_variables,
    )


def score_intervention_outputs(
    raw_results: Dict[Tuple[Any, ...], Dict[str, Any]],
    dataset: list[CounterfactualExample],
    metric: InterchangeMetric,
    causal_model: CausalModel | None = None,
    original_outputs: List[Dict[str, Any]] | None = None,
) -> Dict[Tuple[Any, ...], float]:
    """
    Score pre-computed intervention outputs using an InterchangeMetric.

    This is the core scoring function that takes the output of run_interventions()
    and computes scores for each target key. It handles:
    - Labeling dataset with causal model (if metric.needs_causal_expected)
    - Using original model outputs (if metric.needs_original_output)
    - Flattening nested output structures
    - Computing per-key average scores

    Args:
        raw_results: Dict mapping keys to intervention outputs, as returned by
                    run_interventions(). Each value has {"string": [...], ...}
        dataset: list[CounterfactualExample] used for interventions
        metric: InterchangeMetric defining the scoring function and data requirements
        causal_model: Required if metric.needs_causal_expected is True
        original_outputs: Pre-computed original model outputs. Required if
                         metric.needs_original_output is True.

    Returns:
        Dict mapping keys to average scores (0.0 to 1.0).

    Raises:
        ValueError: If metric.needs_causal_expected is True but causal_model is None
        ValueError: If metric.needs_causal_expected is True but metric.target_variables is None
        ValueError: If metric.needs_original_output is True but original_outputs is None

    Example:
        ```python
        # Run interventions first
        raw_results = run_interventions(targets, dataset, pipeline, batch_size)

        # Then score with different metrics
        causal_metric = make_causal_metric(checker, ("answer",))
        causal_scores = score_intervention_outputs(
            raw_results, dataset, causal_metric, causal_model
        )

        faithfulness_metric = InterchangeMetric(
            fn=lambda out, exp, orig: 1.0 if out["string"] == orig["string"] else 0.0,
            needs_causal_expected=False,
            needs_original_output=True,
        )
        faithfulness_scores = score_intervention_outputs(
            raw_results, dataset, faithfulness_metric, original_outputs=orig_outputs
        )
        ```
    """
    # Validate required arguments
    if metric.needs_causal_expected:
        if causal_model is None:
            raise ValueError(
                "causal_model is required when metric.needs_causal_expected is True"
            )
        if metric.target_variables is None:
            raise ValueError(
                "metric.target_variables is required when metric.needs_causal_expected is True. "
                "Use make_causal_metric() to create a metric with target_variables."
            )

    if metric.needs_original_output and original_outputs is None:
        raise ValueError(
            "original_outputs is required when metric.needs_original_output is True"
        )

    # Get expected outputs from causal model if needed
    expected_outputs: List[Dict[str, Any]] = []
    if metric.needs_causal_expected and causal_model is not None:
        assert metric.target_variables is not None  # validated above
        labeled_data = causal_model.label_counterfactual_data(
            copy.deepcopy(dataset),
            list(metric.target_variables),
        )
        expected_outputs = [example["label"] for example in labeled_data]
    else:
        expected_outputs = [{}] * len(dataset)

    # Default original outputs to empty dicts if not needed
    if original_outputs is None:
        original_outputs = [{}] * len(dataset)

    # Compute scores for each key
    scores: Dict[Tuple[Any, ...], float] = {}

    for key, outputs in raw_results.items():
        # Extract string outputs and flatten if nested
        string_outputs = outputs.get("string", [])
        flattened_outputs: List[str] = []
        for item in string_outputs:
            if isinstance(item, list):
                flattened_outputs.extend(item)
            else:
                flattened_outputs.append(item)

        # Compute score for each example
        key_scores: List[float] = []
        for idx, output_string in enumerate(flattened_outputs):
            if idx < len(expected_outputs):
                intervention_output = {"string": output_string}
                expected = expected_outputs[idx]
                original = original_outputs[idx] if idx < len(original_outputs) else {}

                score = metric.fn(intervention_output, expected, original)
                key_scores.append(float(score))

        scores[key] = sum(key_scores) / len(key_scores) if key_scores else 0.0

    return scores


def LM_loss_and_metric_fn(
    pipeline: LMPipeline,
    intervenable_model: Any,
    examples: List[LabeledCounterfactualExample],
    interchange_target: InterchangeTarget,
    checker: Callable[[Dict[str, Any], Dict[str, Any]], float],
    source_pipeline: LMPipeline | None = None,
    source_intervenable_model: Any = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
    """
    Calculate loss and evaluation metrics for language model interventions.

    Concatenates ground truth labels to inputs, runs intervention forward pass,
    then extracts logits at label positions to compute accuracy and loss.

    Args:
        pipeline: Target language model pipeline for tokenization and generation.
        intervenable_model: Model with intervention capabilities.
        examples: List of labeled counterfactual examples with inputs and labels.
        interchange_target: InterchangeTarget containing the model units to intervene on.
        checker: Function(neural_output, causal_output) -> bool/float for evaluation.
        source_pipeline: If provided, collect activations from this pipeline instead
            of the target pipeline. Enables cross-model patching during training.
        source_intervenable_model: Optional pre-created intervenable model for the source
            pipeline. If provided with source_pipeline, this model will be reused for
            collecting activations instead of creating a new one per batch.

    Returns:
        tuple: (loss, eval_metrics, logging_info)
    """
    # Prepare intervenable inputs (using target pipeline)
    batched_base, batched_counterfactuals, inv_locations, feature_indices = (
        prepare_intervenable_inputs(pipeline, examples, interchange_target)
    )

    # Collect source representations if using cross-model patching
    source_representations = None
    if source_pipeline is not None:
        source_representations = collect_source_representations(
            source_pipeline, examples, interchange_target, source_intervenable_model
        )
        batched_counterfactuals = None

    # Get ground truth labels
    batched_inv_label_strs = [ex["label"] for ex in examples]
    if isinstance(batched_inv_label_strs[0], dict):
        batched_inv_label_strs = [item["string"] for item in batched_inv_label_strs]

    # Convert strings to CausalTraces
    batched_inv_label_traces = [
        CausalTrace(
            mechanisms={
                "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
            },
            inputs={"raw_input": label_str},
        )
        for label_str in batched_inv_label_strs
    ]
    batched_inv_label = pipeline.load(
        batched_inv_label_traces,
        max_length=pipeline.max_new_tokens,
        padding_side="right",
        add_special_tokens=False,
        use_chat_template=False,
    )

    # Concatenate labels to base inputs for evaluation
    for k in batched_base:
        batched_base[k] = torch.cat([batched_base[k], batched_inv_label[k]], dim=-1)

    # Run the intervenable model with interventions
    _, counterfactual_logits = intervenable_model(
        batched_base,
        batched_counterfactuals,
        unit_locations=inv_locations,
        subspaces=feature_indices,
        source_representations=source_representations,
    )

    # Extract relevant portions of logits and labels for evaluation
    labels = batched_inv_label["input_ids"]
    logits = counterfactual_logits.logits[:, -labels.shape[-1] - 1 : -1]
    pred_ids = torch.argmax(logits, dim=-1)

    # Compute metrics using checker function
    scores = []
    for i in range(pred_ids.shape[0]):
        # Decode predictions and labels to strings
        pred_str = pipeline.dump(pred_ids[i : i + 1])

        # Create output dicts in same format as perform_interventions
        neural_output = {"string": pred_str}

        # Apply checker function
        score = checker(neural_output, examples[i]["label"])
        if isinstance(score, torch.Tensor):
            score = score.item()
        scores.append(float(score))

    accuracy = sum(scores) / len(scores) if scores else 1.0
    eval_metrics = {"accuracy": accuracy, "token_accuracy": accuracy}

    # Compute loss
    loss = compute_cross_entropy_loss(logits, labels, pipeline.tokenizer.pad_token_id)

    # Collect detailed information for logging
    logging_info: Dict[str, Any] = {
        "preds": pipeline.dump(pred_ids),
        "labels": pipeline.dump(labels),
        "base_ids": batched_base["input_ids"][0],
        "base_masks": batched_base["attention_mask"][0],
        "base_inputs": pipeline.dump(batched_base["input_ids"][0]),
        "inv_locations": inv_locations,
        "feature_indices": feature_indices,
    }
    if batched_counterfactuals is not None:
        logging_info["counterfactual_masks"] = [
            c["attention_mask"][0] for c in batched_counterfactuals
        ]
        logging_info["counterfactual_ids"] = [
            c["input_ids"][0] for c in batched_counterfactuals
        ]
        logging_info["counterfactual_inputs"] = [
            pipeline.dump(c["input_ids"][0]) for c in batched_counterfactuals
        ]

    return loss, eval_metrics, logging_info


def compute_cross_entropy_loss(
    eval_preds: torch.Tensor, eval_labels: torch.Tensor, pad_token_id: int
) -> torch.Tensor:
    """
    Compute cross-entropy loss over non-padding tokens.

    Args:
        eval_preds (torch.Tensor): Model predictions of shape (batch_size, seq_length, vocab_size)
        eval_labels (torch.Tensor): Ground truth labels of shape (batch_size, seq_length)
        pad_token_id (int): ID of the padding token to be ignored in loss calculation

    Returns:
        torch.Tensor: The computed cross-entropy loss
    """
    # Reshape predictions to (batch_size * sequence_length, vocab_size)
    _batch_size, _seq_length, vocab_size = eval_preds.shape
    preds_flat = eval_preds.reshape(-1, vocab_size)

    # Reshape labels to (batch_size * sequence_length)
    labels_flat = eval_labels.reshape(-1)

    # Compute cross entropy loss, ignoring padding tokens
    return torch.nn.functional.cross_entropy(
        preds_flat, labels_flat, ignore_index=pad_token_id
    )
