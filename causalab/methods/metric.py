"""Metric computation utilities for intervention experiments."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Callable, Tuple, Any
import copy
import math
import torch
import torch.nn.functional as F
from causalab.neural.activations.interchange_mode import prepare_intervenable_inputs
from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import CausalTrace, Mechanism

from causalab.neural.pipeline import LMPipeline
from causalab.neural.units import InterchangeTarget
from causalab.neural.activations.collect import collect_source_representations
from torch import Tensor

logger = logging.getLogger(__name__)


def tokenize_variable_values(
    tokenizer,
    values: list[str],
    token_pattern: Callable,
) -> torch.Tensor | list[list[int]]:
    """Tokenize variable values, returning token IDs per concept.

    token_pattern returns a list of string variants per value.
    Each variant is encoded; only single-token encodings are kept.
    Returns list[list[int]] where each inner list has all valid
    single-token IDs for that concept (variants like " Monday",
    "Monday", " monday").

    For multi-token outputs (e.g., graph walk generation steps),
    falls back to the first variant's token sequence.
    """
    all_concept_ids: list[list[int]] = []
    for v in values:
        variants = token_pattern(v)
        # Collect all single-token encodings across variants
        single_tok_ids = []
        first_seq = None
        for var_str in variants:
            seq = tokenizer.encode(var_str, add_special_tokens=False)
            if first_seq is None:
                first_seq = seq
            if len(seq) == 1:
                tid = seq[0]
                if tid not in single_tok_ids:
                    single_tok_ids.append(tid)
        if single_tok_ids:
            all_concept_ids.append(single_tok_ids)
        else:
            # No single-token variant: fall back to first variant's full sequence
            all_concept_ids.append(first_seq)

    return all_concept_ids


def _normalize_var_indices(var_indices) -> list[list[int]]:
    """Normalize var_indices to list[list[int]] regardless of input format.

    Accepts: torch.Tensor (1-D), list[int], or list[list[int]].
    Each inner list contains the token IDs for one concept (possibly
    multiple variants like " Monday" and "Monday").
    """
    if isinstance(var_indices, torch.Tensor):
        return [[idx.item()] for idx in var_indices]
    if var_indices and isinstance(var_indices[0], int):
        return [[idx] for idx in var_indices]
    return var_indices


def scores_to_joint_probs(
    raw_scores: list,
    var_indices: torch.Tensor | list[list[int]],
    full_vocab_softmax: bool = False,
) -> torch.Tensor | None:
    """Convert raw intervention scores to joint probability distributions.

    Args:
        raw_scores: List of batch score tensors from intervention runs
            (each element is a list of per-token-step ``(B, V)`` tensors,
            or a single ``(B, V)`` tensor).
        var_indices: Token indices for variable values — either a 1-D Tensor
            (single-token) or ``list[list[int]]`` (multi-token).
        full_vocab_softmax: If True, softmax over full vocabulary before
            extracting class token probabilities.

    Returns:
        ``(N, W)`` normalized joint probabilities, or ``None`` if no scores.
        When full_vocab_softmax=True, probabilities may not sum to 1.
    """
    var_token_seqs = _normalize_var_indices(var_indices)
    W_cats = len(var_token_seqs)

    step_batches: list[list[torch.Tensor]] = []
    for batch_scores in raw_scores:
        if isinstance(batch_scores, list):
            for k, scores_k in enumerate(batch_scores):
                if k >= len(step_batches):
                    step_batches.append([])
                step_batches[k].append(scores_k)
        elif isinstance(batch_scores, torch.Tensor):
            if len(step_batches) == 0:
                step_batches.append([])
            step_batches[0].append(batch_scores)

    if not step_batches:
        return None

    step_tensors = [torch.cat(batches, dim=0) for batches in step_batches]

    N = step_tensors[0].shape[0]
    joint_NW = torch.ones(N, W_cats)

    # Single generation step: pass the full variant lists to class_probabilities
    if len(step_tensors) == 1:
        probs = class_probabilities(
            step_tensors[0], var_token_seqs, full_vocab_softmax=full_vocab_softmax
        )
        joint_NW = probs.cpu()
    else:
        # Multi-step: each inner list is a step sequence, not variants
        for k, logits_NV in enumerate(step_tensors):
            active = [
                (w, seq[k]) for w, seq in enumerate(var_token_seqs) if k < len(seq)
            ]
            if active:
                step_ids = [t for _, t in active]
                probs = class_probabilities(
                    logits_NV, step_ids, full_vocab_softmax=full_vocab_softmax
                )
                w_idx = torch.tensor([w for w, _ in active])
                joint_NW[:, w_idx] *= probs.cpu()

    if full_vocab_softmax:
        return joint_NW  # don't renormalize — these are true P(token)
    return joint_NW / joint_NW.sum(dim=-1, keepdim=True)


# Backward-compatible alias
_scores_to_joint_probs = scores_to_joint_probs


def class_probabilities(
    logits: Tensor,
    class_token_ids: list[int] | list[list[int]],
    full_vocab_softmax: bool = False,
) -> Tensor:
    """Convert logits to per-class probabilities for a single generation step.

    Args:
        logits: (N, V) or (V,) raw logits over vocabulary.
        class_token_ids: Token IDs per class. Either a flat list (one ID per
            class) or list of lists (multiple variant IDs per class, e.g.,
            [[" Monday" id, "Monday" id], [" Tuesday" id], ...]).
            When variants are provided, their probabilities are summed.
        full_vocab_softmax: If True, softmax over the full vocabulary then
            extract/sum class tokens (probabilities won't sum to 1 across classes).
            If False (default), full-vocab softmax → sum variants → renormalize.

    Returns:
        (N, n_classes) probabilities (float32).
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # Normalize to list-of-lists format
    if class_token_ids and isinstance(class_token_ids[0], int):
        id_groups = [[tid] for tid in class_token_ids]
    else:
        id_groups = class_token_ids

    # Always compute full-vocab softmax first, then sum variants per concept
    all_probs = F.softmax(logits.float(), dim=-1)  # (N, V)
    n_classes = len(id_groups)
    result = torch.zeros(all_probs.shape[0], n_classes, device=logits.device)
    for c, variant_ids in enumerate(id_groups):
        ids = torch.tensor(variant_ids, device=logits.device)
        result[:, c] = all_probs[:, ids].sum(dim=-1)

    if not full_vocab_softmax:
        # Renormalize so concept probabilities sum to 1
        result = result / result.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    return result


def causal_score_intervention_outputs(
    raw_results: Dict[Tuple[Any, ...], Dict[str, Any]],
    dataset: list[CounterfactualExample],
    causal_model: CausalModel,
    target_variable_groups: List[Tuple[str, ...]],
    metric: Callable[[Any, Any], bool],
) -> Dict[str, Any]:
    """Score intervention outputs against causal model expectations for each variable group."""
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
    """Metric for scoring interchange interventions.

    fn receives (intervention_output, expected, original) -> float.
    """

    fn: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], float]
    needs_causal_expected: bool = True
    needs_original_output: bool = False
    target_variables: Tuple[str, ...] | None = None
    label_variable: str = "raw_output"


def string_equality_checker(intervention_output: Dict[str, Any], expected: Any) -> bool:
    """Default checker: stripped string equality between intervention output and expected."""
    output_str = intervention_output.get("string", "")
    expected_str = (
        expected.get("string", expected) if isinstance(expected, dict) else expected
    )
    return str(output_str).strip() == str(expected_str).strip()


def make_causal_metric(
    checker: Callable[[Dict[str, Any], Any], float | bool] = string_equality_checker,
    target_variables: Tuple[str, ...] = ("raw_output",),
    label_variable: str = "raw_output",
) -> InterchangeMetric:
    """Create an InterchangeMetric that compares intervention outputs to causal model labels.

    Bool checker results are coerced to 0.0/1.0.
    """

    def causal_metric_fn(
        intervention_output: Dict[str, Any],
        expected: Dict[str, Any],
        original: Dict[str, Any],
    ) -> float:
        result = checker(intervention_output, expected)
        if isinstance(result, bool):
            return 1.0 if result else 0.0
        return float(result)

    return InterchangeMetric(
        fn=causal_metric_fn,
        needs_causal_expected=True,
        needs_original_output=False,
        target_variables=target_variables,
        label_variable=label_variable,
    )


def make_kl_checker(
    ref_dists: torch.Tensor,
    score_token_ids: List[int],
    label_to_class: Callable[[Any], int],
    score_token_index: int = 1,
) -> Callable[[Dict[str, Any], Any], float]:
    """Create a checker that computes KL(ref_dists[class] || intervention_probs).

    Args:
        ref_dists: (n_classes, n_classes) reference probability distributions.
        score_token_ids: Token IDs to restrict logits to (one per class).
        label_to_class: Maps causal model label to a ref_dists row index.
        score_token_index: Which generated token's logits to use (default 1).
    """

    def checker(intervention_output: Dict[str, Any], expected: Any) -> float:
        scores = intervention_output.get("scores")
        if scores is None:
            raise ValueError("KL checker requires scores (pass output_scores=True)")
        idx = intervention_output["example_idx"]
        if len(scores) <= score_token_index:
            raise ValueError(
                f"Expected > {score_token_index} score tensors, got {len(scores)}"
            )
        logits = scores[score_token_index][idx]
        probs = class_probabilities(logits, score_token_ids).squeeze(0).cpu()
        ref = ref_dists[label_to_class(expected)].unsqueeze(0)
        return kl_divergence(ref, probs.unsqueeze(0)).item()

    return checker


def kl_divergence(reference: Tensor, predicted: Tensor) -> Tensor:
    """KL(reference || predicted), per-row.

    Args:
        reference, predicted: (N, C) probability tensors.

    Returns:
        (N,) KL values (lower = better match).
    """
    predicted_safe = predicted.clamp(min=1e-10)
    reference_safe = reference.clamp(min=1e-10)
    mask = reference > 0
    log_ratio = reference_safe.log() - predicted_safe.log()
    return (reference * log_ratio * mask.float()).sum(dim=-1)


def hellinger_distance(reference: Tensor, predicted: Tensor) -> Tensor:
    """Hellinger distance, per-row.

    Returns:
        (N,) values in [0, 1].
    """
    return (1.0 / math.sqrt(2)) * (reference.sqrt() - predicted.sqrt()).norm(dim=-1)


DISTRIBUTION_COMPARISONS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "kl": kl_divergence,
    "hellinger": hellinger_distance,
}


def _logits_to_class_probs(
    logits_per_step: list[Tensor],
    score_token_ids: list[int] | list[list[int]],
    full_vocab_softmax: bool = False,
) -> Tensor:
    """Convert per-step logits to class probabilities.

    For single-token classes (``list[int]``), uses one step.
    For multi-token classes (``list[list[int]]``), multiplies across steps.

    Returns:
        ``(N, n_classes)`` probability tensor.
    """
    token_seqs = _normalize_var_indices(score_token_ids)
    n_steps = max(len(seq) for seq in token_seqs)
    N = logits_per_step[0].shape[0]
    n_classes = len(token_seqs)
    joint = torch.ones(N, n_classes)

    for k in range(n_steps):
        if k >= len(logits_per_step):
            break
        active = [(w, seq[k]) for w, seq in enumerate(token_seqs) if k < len(seq)]
        if not active:
            continue
        step_ids = [t for _, t in active]
        probs = class_probabilities(
            logits_per_step[k], step_ids, full_vocab_softmax=full_vocab_softmax
        )
        for out_idx, (w, _) in enumerate(active):
            joint[:, w] *= probs[:, out_idx].cpu()

    if not full_vocab_softmax:
        return joint / joint.sum(dim=-1, keepdim=True)
    return joint


def compute_reference_distributions(
    dataset: list[CounterfactualExample],
    score_token_ids: list[int] | list[list[int]],
    n_classes: int,
    example_to_class: Callable[[CounterfactualExample], int],
    output_logits: list[torch.Tensor] | None = None,
    pipeline: Any = None,
    score_token_index: int = 1,
    batch_size: int = 16,
    full_vocab_softmax: bool = False,
) -> torch.Tensor:
    """Compute per-class average output distributions (no intervention).

    Returns (n_classes, n_score_tokens) tensor where n_score_tokens =
    len(score_token_ids). Uses ``output_logits`` if provided,
    otherwise runs ``pipeline.generate()`` in batches.
    """
    token_seqs = _normalize_var_indices(score_token_ids)
    n_steps = max(len(seq) for seq in token_seqs)
    n_score_tokens = len(token_seqs)
    accum = torch.zeros(n_classes, n_score_tokens)
    counts = torch.zeros(n_classes)

    if output_logits is not None:
        # Pre-computed logits — single-step only (last position)
        for i, ex in enumerate(dataset):
            logits = output_logits[i][-1]  # last position → (vocab_size,)
            # For pre-computed logits we only have one step
            step_ids = [seq[0] for seq in token_seqs]
            probs = class_probabilities(
                logits, step_ids, full_vocab_softmax=full_vocab_softmax
            ).squeeze(0)
            class_idx = example_to_class(ex)
            accum[class_idx] += probs.cpu()
            counts[class_idx] += 1
    else:
        if pipeline is None:
            raise ValueError("pipeline is required when output_logits is not provided")
        n_batches = math.ceil(len(dataset) / batch_size)
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(dataset))
            batch_examples = dataset[start:end]
            batch_inputs = [ex["input"] for ex in batch_examples]

            result = pipeline.generate(batch_inputs)
            scores = result["scores"]

            # Collect logits for each generation step we need
            logits_per_step: list[Tensor] = []
            for step in range(n_steps):
                idx = score_token_index + step
                if idx >= len(scores):
                    break
                logits_per_step.append(scores[idx])

            batch_probs = _logits_to_class_probs(
                logits_per_step, token_seqs, full_vocab_softmax=full_vocab_softmax
            )

            for bi, ex in enumerate(batch_examples):
                class_idx = example_to_class(ex)
                accum[class_idx] += batch_probs[bi].cpu()
                counts[class_idx] += 1

    for i in range(n_classes):
        if counts[i] > 0:
            accum[i] /= counts[i]
        else:
            accum[i] = 1.0 / n_classes

    return accum


def compute_base_accuracy(
    dataset: list[CounterfactualExample],
    pipeline: Any,
    batch_size: int = 16,
) -> dict:
    """Compute base model accuracy (no intervention) over a dataset.

    Checks if the model's generated output matches ``raw_output`` from each
    example.  Handles both single-answer (``str``) and multi-answer
    (``list[str]``) raw_output (e.g. graph_walk where any valid neighbor
    counts as correct).

    Also computes ``prob_accuracy``: the mean probability mass assigned to
    valid answer tokens (full-vocab softmax), which is more informative than
    binary top-1 accuracy.  Only computed for single-token outputs
    (``max_new_tokens == 1``); ``None`` otherwise.

    Returns dict with keys: accuracy, correct, total, prob_accuracy.
    """
    tokenizer = pipeline.tokenizer
    correct = 0
    total = 0
    prob_sum = 0.0
    single_token = pipeline.max_new_tokens == 1
    n_batches = math.ceil(len(dataset) / batch_size)
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(dataset))
        batch_examples = dataset[start:end]
        batch_inputs = [ex["input"] for ex in batch_examples]

        result = pipeline.generate(batch_inputs)
        strings = result["string"]
        if isinstance(strings, str):
            strings = [strings]
        scores = result.get("scores", [])

        # Full-vocab softmax for the first generated token
        if single_token and scores:
            probs = F.softmax(scores[0].float(), dim=-1)  # (B, vocab_size)

        for bi, ex in enumerate(batch_examples):
            generated = strings[bi].strip()
            raw_output = ex["input"]["raw_output"]
            if isinstance(raw_output, list):
                hit = any(generated == ans.strip() for ans in raw_output)
            else:
                hit = generated == raw_output.strip()
            if hit:
                correct += 1
            total += 1

            # P(valid answer): sum of probs for all valid answer token variants
            if single_token and scores:
                answers = raw_output if isinstance(raw_output, list) else [raw_output]
                all_token_ids = set()
                for ans in answers:
                    # Collect all single-token variants of this answer
                    ans_str = ans
                    variants = [ans_str]
                    stripped = ans_str.strip()
                    if stripped != ans_str:
                        variants.append(stripped)
                    if stripped.lower() != stripped:
                        variants.append(stripped.lower())
                        if ans_str.startswith(" "):
                            variants.append(" " + stripped.lower())
                    for var in variants:
                        ids = tokenizer.encode(var, add_special_tokens=False)
                        if len(ids) == 1:
                            all_token_ids.add(ids[0])
                if all_token_ids:
                    prob_sum += probs[bi, list(all_token_ids)].sum().item()

    accuracy = correct / total if total > 0 else 0.0
    prob_accuracy = prob_sum / total if (single_token and total > 0) else None
    logger.info("Base model accuracy: %d/%d (%.1f%%)", correct, total, accuracy * 100)
    if prob_accuracy is not None:
        logger.info("Base model prob accuracy: %.1f%%", prob_accuracy * 100)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "prob_accuracy": prob_accuracy,
    }


def score_intervention_outputs(
    raw_results: Dict[Tuple[Any, ...], Dict[str, Any]],
    dataset: list[CounterfactualExample],
    metric: InterchangeMetric,
    causal_model: CausalModel | None = None,
    original_outputs: List[Dict[str, Any]] | None = None,
) -> Dict[Tuple[Any, ...], float]:
    """Score pre-computed intervention outputs. Returns dict mapping keys to average scores."""
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
            label_variable=metric.label_variable,
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

        # Flatten score tensors across batches if present.
        # raw_scores structure: [batch0_scores, batch1_scores, ...]
        # where each batch_scores = [token0_tensor(B, V), token1_tensor(B, V)]
        # We concat per token position → [token0(N, V), token1(N, V)]
        raw_scores = outputs.get("scores")
        flat_scores: List[torch.Tensor] | None = None
        if raw_scores is not None and raw_scores:
            n_tokens = len(raw_scores[0])
            flat_scores = [
                torch.cat([batch_scores[t] for batch_scores in raw_scores], dim=0)
                for t in range(n_tokens)
            ]

        # Compute score for each example
        key_scores: List[float] = []
        for idx, output_string in enumerate(flattened_outputs):
            if idx < len(expected_outputs):
                intervention_output: Dict[str, Any] = {"string": output_string}
                if flat_scores is not None:
                    intervention_output["scores"] = flat_scores
                    intervention_output["example_idx"] = idx
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
    """Concatenate labels, run intervention forward pass, compute accuracy + cross-entropy loss."""
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
    """Cross-entropy loss over non-padding tokens."""
    _batch_size, _seq_length, vocab_size = eval_preds.shape
    return torch.nn.functional.cross_entropy(
        eval_preds.reshape(-1, vocab_size),
        eval_labels.reshape(-1),
        ignore_index=pad_token_id,
    )
