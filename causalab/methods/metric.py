"""Metric computation utilities for intervention experiments."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Callable, Sequence, Tuple, Any
import copy
import math
import torch
import torch.nn.functional as F
from causalab.neural.activations.engine import (
    build_plans,
    forward_with_interventions,
)
from causalab.neural.activations.interchange_mode import (
    collect_group_sources,
    prepare_interchange_batch,
)
from causalab.neural.interventions import FeatureIntervention
from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import CausalTrace, Mechanism

from causalab.neural.pipeline import LMPipeline
from causalab.neural.units import InterchangeTarget
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
            # No single-token variant: fall back to first variant's full sequence.
            # variants is non-empty (caller guarantees ≥1 variant per concept), so
            # first_seq must have been assigned in the loop above.
            assert first_seq is not None
            all_concept_ids.append(first_seq)

    return all_concept_ids


def answer_token_forms(answer: str) -> list[str]:
    """Candidate single-token string forms of an answer to probe.

    Returns ``{space-prefixed, bare} × {as-is, lowercased}``, de-duplicated and
    order-stable. This aligns the probability grader with the strip-tolerant
    string grader: whether the task ships ``raw_output="blue"`` or ``" blue"``,
    both the space-prefixed ``" blue"`` and bare ``blue`` tokens (and their
    lowercase forms) are tried, so the grader captures the leading-space token
    the model actually emits instead of silently scoring ~0.

    **Order matters:** the space-prefixed forms come first. Set-building callers
    (the string/prob graders) are order-insensitive, but :func:`single_token_id`
    returns the *first* single-token form, and at a word boundary the model emits
    the leading-space token (``" Mary"``, not the bare ``"Mary"`` — a different
    GPT-2 BPE id for ~half of typical name vocabularies). Listing the emitted
    form first makes that function read the vocab row the model actually scores.
    """
    stripped = answer.strip()
    forms: list[str] = []
    for prefix in (" ", ""):
        for base in (stripped, stripped.lower()):
            cand = prefix + base
            if base and cand not in forms:
                forms.append(cand)
    return forms


def single_token_id(pipeline: LMPipeline, answer: str) -> int:
    """First single-token id among the spacing/case variants of ``answer``.

    Mirrors ``compute_base_accuracy``'s grader: tries ``{" "+bare, bare}`` ×
    ``{as-is, lower}`` (see :func:`answer_token_forms`). Because that helper now
    lists the space-prefixed forms first, this returns the *emitted* token — the
    one the model places probability on after a word boundary (``" Mary"``), not
    the bare ``"Mary"``, a distinct GPT-2 BPE id for ~half of typical name
    vocabularies. Reading the bare id would score the wrong vocab row in the
    logit/prob/logit-diff metrics. Raises if no variant is single-token — the
    logit metrics need exactly one token id to read.
    """
    tokenizer = pipeline.tokenizer
    for form in answer_token_forms(answer):
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(
        f"No single-token form of answer {answer!r} under this tokenizer; the "
        f"logit metric needs single-token answers."
    )


def _normalize_var_indices(var_indices) -> list[list[int]]:
    """Normalize var_indices to list[list[int]] regardless of input format.

    Accepts: torch.Tensor (1-D), list[int], or list[list[int]].
    Each inner list contains the token IDs for one concept (possibly
    multiple variants like " Monday" and "Monday").
    """
    if isinstance(var_indices, torch.Tensor):
        return [[int(idx.item())] for idx in var_indices]
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

    ``needs_scores`` signals that ``fn`` reads ``intervention_output["scores"]``
    (raw logits) rather than only the decoded ``"string"``; callers use it to
    decide whether to request ``output_scores`` from the intervention run.
    """

    fn: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], float]
    needs_causal_expected: bool = True
    needs_original_output: bool = False
    needs_scores: bool = False
    target_variables: Tuple[str, ...] | None = None
    label_variable: str = "raw_output"


def as_label_checker(
    checker: Callable[[Dict[str, Any], str], bool],
) -> Callable[[Dict[str, Any], Any], bool]:
    """Adapt a task checker to the interchange-metric checker signature.

    A task's ``checker`` takes ``(neural_output, causal_output: str)``, but
    intervention scoring (``make_causal_metric``, ``LM_loss_and_metric_fn``)
    passes the causal *label*, which may arrive as a bare value or as a
    ``{"string": ...}`` dict.  This normalizes the label to a string before
    delegating, so ``task.checker`` becomes the single match authority for both
    base accuracy and intervention scoring (#167) — replacing the previous
    lenient-containment / string-equality defaults.
    """

    def _adapted(neural_output: Dict[str, Any], expected: Any) -> bool:
        expected_value = (
            expected.get("string", expected) if isinstance(expected, dict) else expected
        )
        return checker(neural_output, str(expected_value))

    return _adapted


def make_causal_metric(
    checker: Callable[[Dict[str, Any], Any], float | bool],
    target_variables: Tuple[str, ...] = ("raw_output",),
    label_variable: str = "raw_output",
) -> InterchangeMetric:
    """Create an InterchangeMetric that compares intervention outputs to causal model labels.

    ``checker`` is the match authority (a task's ``checker`` wrapped via
    :func:`as_label_checker`, #167) — there is no strict-equality default. Bool
    checker results are coerced to 0.0/1.0.
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


def make_distribution_shift_metric(
    score_token_ids: list[int] | list[list[int]],
    comparison_fn: Callable[[Tensor, Tensor], Tensor] = kl_divergence,
    score_token_index: int = 0,
) -> InterchangeMetric:
    """Score how far the patched output drifts from the base (pre-intervention) output.

    For each example, projects both the patched logits
    (``intervention_output["scores"]``, indexed by ``example_idx``) and the base
    logits (``original["scores"]``) onto the task's answer classes and returns
    ``comparison_fn(base_probs, patched_probs)``.

    A *larger* value means patching that cell moved the output more — i.e. the
    variable is more strongly encoded there — so this metric is **not** negated:
    ``run_interchange_scan``'s "best = highest score per layer" summary correctly
    selects the most causal cell.  Requires ``original_outputs`` (see
    ``compute_base_outputs``) and ``output_scores=True`` on the intervention run.
    """
    token_seqs = _normalize_var_indices(score_token_ids)
    n_steps = max(len(seq) for seq in token_seqs)

    def _probs_from_logits(logits_per_token: list[Tensor]) -> Tensor:
        logits_per_step = [
            logits_per_token[score_token_index + k].unsqueeze(0)
            for k in range(n_steps)
            if score_token_index + k < len(logits_per_token)
        ]
        return _logits_to_class_probs(logits_per_step, token_seqs)

    # The base distribution is constant across all (cell × example) calls for a
    # given example, but ``shift_fn`` is invoked once per cell. Memoize per base
    # output (keyed by identity of its ``scores`` list, which the caller holds
    # alive for the whole scan) so we project the base logits once, not n_cells×.
    base_probs_cache: Dict[int, Tensor] = {}

    def shift_fn(
        intervention_output: Dict[str, Any],
        expected: Dict[str, Any],
        original: Dict[str, Any],
    ) -> float:
        scores = intervention_output.get("scores")
        if scores is None:
            raise ValueError(
                "distribution-shift metric requires scores (pass output_scores=True)"
            )
        base_scores = original.get("scores")
        if base_scores is None:
            raise ValueError(
                "distribution-shift metric requires base outputs "
                "(pass original_outputs from compute_base_outputs)"
            )
        idx = intervention_output["example_idx"]
        patched_probs = _probs_from_logits([s[idx] for s in scores])
        base_key = id(base_scores)
        base_probs = base_probs_cache.get(base_key)
        if base_probs is None:
            base_probs = _probs_from_logits(list(base_scores))
            base_probs_cache[base_key] = base_probs
        return comparison_fn(base_probs, patched_probs).item()

    return InterchangeMetric(
        fn=shift_fn,
        needs_causal_expected=False,
        needs_original_output=True,
        needs_scores=True,
    )


def make_logit_metric(
    pipeline: LMPipeline,
    dataset: list[CounterfactualExample],
    answer_of: Callable[[CounterfactualExample], str],
    *,
    relative_to_base: bool = True,
    score_token_index: int = 0,
) -> InterchangeMetric:
    """Build a single-token logit / Δ-logit metric over ``dataset``.

    The continuous counterpart to :func:`make_causal_metric`'s 0/1 string match:
    instead of "did the answer change," it reads the raw logit of a fixed answer
    token, so the score tracks *how much* an intervention moved the model — the
    natural unit for self-repair, direct-effect, and ablation-impact work.

    ``answer_of`` maps each example to the answer string whose logit to read (e.g.
    CounterFact's object). Single-token ids are precomputed per example (via
    :func:`single_token_id`) and looked up by ``example_idx`` at scoring time,
    since ``InterchangeMetric.fn`` sees only ``(intervention_output, expected,
    original)``.

    Distinct from :func:`make_logit_diff_metric`, which scores the logit
    *difference between two tokens* (correct vs. distractor) on a single run; this
    scores *one* token, optionally relative to the base run.

    With ``relative_to_base`` (default) the score is the *ablation impact*
    ``base_logit − patched_logit`` — positive when the intervention pushes the
    answer logit down (the self-repair / direct-effect sign convention). With it
    ``False`` the score is the raw patched logit. Requires full-vocab
    ``output_scores=True`` on the scan; ``relative_to_base`` additionally needs
    ``original_outputs`` from :func:`compute_base_outputs`.
    """
    answer_ids = [single_token_id(pipeline, answer_of(ex)) for ex in dataset]

    def fn(
        intervention_output: Dict[str, Any],
        _expected: Dict[str, Any],
        original: Dict[str, Any],
    ) -> float:
        scores = intervention_output.get("scores")
        if scores is None:
            raise ValueError(
                "logit metric needs full-vocab scores from the patched run "
                "(output_scores=True)."
            )
        idx = intervention_output["example_idx"]
        tok = answer_ids[idx]
        patched = float(scores[score_token_index][idx][tok])
        if not relative_to_base:
            return patched
        base_scores = original.get("scores")
        if base_scores is None:
            raise ValueError(
                "logit metric with relative_to_base=True needs base outputs "
                "(pass original_outputs from compute_base_outputs)."
            )
        base = float(base_scores[score_token_index][tok])  # (vocab,) per example
        return base - patched

    return InterchangeMetric(
        fn=fn,
        needs_causal_expected=False,
        needs_original_output=relative_to_base,
        needs_scores=True,
    )


def make_prob_metric(
    pipeline: LMPipeline,
    dataset: list[CounterfactualExample],
    answer_of: Callable[[CounterfactualExample], str],
    *,
    relative_to_base: bool = True,
    score_token_index: int = 0,
) -> InterchangeMetric:
    """Build a softmax-probability metric ``P(answer)`` over ``dataset``.

    The probability counterpart to :func:`make_logit_metric`: it softmaxes the
    full-vocab logits at ``score_token_index`` and reads the answer token's
    probability, the readout ROME-style causal tracing reports (the recovered
    *probability* of the correct continuation), rather than a raw logit.

    ``answer_of`` maps each example to the answer string whose probability to
    read; single-token ids are precomputed per example and looked up by
    ``example_idx`` at scoring time.

    With ``relative_to_base`` (default) the score is ``base_prob − patched_prob``
    (positive when the intervention pushes the answer probability down); with it
    ``False`` the score is the raw patched probability — the form causal tracing
    uses, computing recovery against the corrupted floor itself. Requires
    full-vocab ``output_scores=True``; ``relative_to_base`` additionally needs
    ``original_outputs`` from :func:`compute_base_outputs`.
    """
    answer_ids = [single_token_id(pipeline, answer_of(ex)) for ex in dataset]

    def _prob_of(logits: torch.Tensor, tok: int) -> float:
        return float(torch.softmax(logits.float(), dim=-1)[tok])

    def fn(
        intervention_output: Dict[str, Any],
        _expected: Dict[str, Any],
        original: Dict[str, Any],
    ) -> float:
        scores = intervention_output.get("scores")
        if scores is None:
            raise ValueError(
                "prob metric needs full-vocab scores from the patched run "
                "(output_scores=True)."
            )
        idx = intervention_output["example_idx"]
        tok = answer_ids[idx]
        patched = _prob_of(scores[score_token_index][idx], tok)
        if not relative_to_base:
            return patched
        base_scores = original.get("scores")
        if base_scores is None:
            raise ValueError(
                "prob metric with relative_to_base=True needs base outputs "
                "(pass original_outputs from compute_base_outputs)."
            )
        base = _prob_of(base_scores[score_token_index], tok)  # (vocab,) per example
        return base - patched

    return InterchangeMetric(
        fn=fn,
        needs_causal_expected=False,
        needs_original_output=relative_to_base,
        needs_scores=True,
    )


def make_logit_diff_metric(
    pipeline: LMPipeline,
    dataset: list[CounterfactualExample],
    correct_of: Callable[[CounterfactualExample], str],
    distractor_of: Callable[[CounterfactualExample], str],
    *,
    relative_to_base: bool = True,
    score_token_index: int = 0,
) -> InterchangeMetric:
    """Build a single-run logit-*difference* metric over ``dataset``.

    The two-token sibling of :func:`make_logit_metric`: instead of one answer
    token's logit, it reads the *difference* between a correct and a distractor
    token, ``logit[correct] − logit[distractor]`` — the canonical path-patching /
    IOI readout, but equally usable for any contrastive ablation or interchange
    scan.

    ``correct_of`` / ``distractor_of`` map each example to its correct and
    distractor answer strings (e.g. IOI ``IO`` and ``name_C``). Single-token ids
    are precomputed per example (via :func:`single_token_id`) and looked up by
    ``example_idx`` at scoring time, since ``InterchangeMetric.fn`` sees only
    ``(intervention_output, expected, original)``.

    With ``relative_to_base`` (default) the score is the *direct effect*
    ``base_diff − patched_diff`` — how much of the clean logit difference flowed
    through the patched path (positive when the sender pushes the output toward
    the correct token). With it ``False`` the score is the raw patched logit
    difference. Requires full-vocab ``output_scores=True`` on the scan;
    ``relative_to_base`` additionally needs ``original_outputs`` from
    :func:`compute_base_outputs`.
    """
    correct_ids = [single_token_id(pipeline, correct_of(ex)) for ex in dataset]
    distractor_ids = [single_token_id(pipeline, distractor_of(ex)) for ex in dataset]

    def fn(
        intervention_output: Dict[str, Any],
        _expected: Dict[str, Any],
        original: Dict[str, Any],
    ) -> float:
        scores = intervention_output.get("scores")
        if scores is None:
            raise ValueError(
                "logit-difference metric needs full-vocab scores from the patched "
                "run (output_scores=True)."
            )
        idx = intervention_output["example_idx"]
        c, d = correct_ids[idx], distractor_ids[idx]
        patched = scores[score_token_index][idx]  # (vocab,)
        patched_diff = float(patched[c] - patched[d])
        if not relative_to_base:
            return patched_diff
        base_scores = original.get("scores")
        if base_scores is None:
            raise ValueError(
                "logit-difference metric with relative_to_base=True needs base "
                "outputs (pass original_outputs from compute_base_outputs)."
            )
        base = base_scores[score_token_index]  # (vocab,) per example
        base_diff = float(base[c] - base[d])
        return base_diff - patched_diff

    return InterchangeMetric(
        fn=fn,
        needs_causal_expected=False,
        needs_original_output=relative_to_base,
        needs_scores=True,
    )


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
    checker: Callable[[dict, str], bool],
    batch_size: int = 16,
    answer_fn: Callable[[CounterfactualExample], str | list[str]] | None = None,
) -> dict:
    """Compute base model accuracy (no intervention) over a dataset.

    Checks if the model's generated output matches the example's expected
    answer.  Handles both single-answer (``str``) and multi-answer
    (``list[str]``) expectations (e.g. graph_walk where any valid neighbor
    counts as correct).

    By default the expected answer is ``ex["input"]["raw_output"]``.  Pass
    ``answer_fn`` to score against a different per-example string — used by
    tasks with more than one scoring convention (e.g. MCQA scored against the
    choice *value*/colour rather than the option *letter*).  The variant
    logic below already tries both the bare and space-prefixed forms of
    whatever string is returned, so an ``answer_fn`` returning ``" orange"``
    captures the bare ``orange`` token the model actually emits.

    ``checker`` (a task's ``checker({"string": generated}, expected) -> bool``,
    i.e. ``task.checker``) is **required** and is the sole match authority — there
    is no strict-equality fallback (#167).  It lets each task decide its own
    semantics, e.g. entity_binding's ``startswith`` accepts the continuation
    tokens a ``max_new_tokens > 1`` task emits after the answer
    (``"bread\\n\\nAnn loves"`` for expected ``"bread"``), which strict equality
    would reject.

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
        probs = None
        if single_token and scores:
            probs = F.softmax(scores[0].float(), dim=-1)  # (B, vocab_size)

        for bi, ex in enumerate(batch_examples):
            generated = strings[bi]
            expected = (
                answer_fn(ex) if answer_fn is not None else ex["input"]["raw_output"]
            )
            answers = expected if isinstance(expected, list) else [expected]
            # The task's checker is the sole match authority (no strict-equality
            # fallback, #167) — e.g. entity_binding's ``startswith`` accepts the
            # continuation tokens a ``max_new_tokens > 1`` task emits after the
            # answer. A list expectation (e.g. graph_walk) counts if any matches.
            hit = any(checker({"string": generated}, ans) for ans in answers)
            if hit:
                correct += 1
            total += 1

            # P(valid answer): sum of probs for all valid answer token variants
            if single_token and scores:
                all_token_ids = set()
                for ans in answers:
                    # Collect all single-token forms of this answer, trying both
                    # spacings and cases so the prob grader matches the
                    # strip-tolerant string grader regardless of whether
                    # ``raw_output`` carries a leading space.
                    for var in answer_token_forms(ans):
                        ids = tokenizer.encode(var, add_special_tokens=False)
                        if len(ids) == 1:
                            all_token_ids.add(ids[0])
                if all_token_ids:
                    assert (
                        probs is not None
                    )  # single_token and scores branch sets probs
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


def compute_base_outputs(
    dataset: list[CounterfactualExample],
    pipeline: Any,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """Run the base (un-patched) model over a dataset, one output dict per example.

    Returns a list aligned with ``dataset`` where each entry is
    ``{"string": <generated text>, "scores": [logits_tok0 (V,), ...]}``.  This is
    the shape ``score_intervention_outputs`` expects for ``original_outputs`` when
    a metric has ``needs_original_output=True`` (e.g. the distribution-shift metric).
    """
    outputs: List[Dict[str, Any]] = []
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
        for bi in range(len(batch_examples)):
            outputs.append(
                {
                    "string": strings[bi],
                    "scores": [scores[t][bi].cpu() for t in range(len(scores))],
                }
            )
    return outputs


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
    examples: List[LabeledCounterfactualExample],
    interchange_target: InterchangeTarget,
    checker: Callable[[Dict[str, Any], Dict[str, Any]], float],
    interventions: Sequence[FeatureIntervention] | None = None,
    source_pipeline: LMPipeline | None = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
    """Concatenate labels, run an intervened forward, return loss + accuracy.

    The differentiable step of DAS / DBM training. Nothing here detaches, so the
    returned loss backpropagates through the intervention into whatever the
    featurizer or mask owns — see ``forward_with_interventions``.

    ``interventions`` are the run's intervention objects, reused across steps so
    a mask's parameters are one set for the whole run. Built per call if omitted,
    which is only correct for a stateless mode.
    """
    batch = prepare_interchange_batch(
        pipeline, examples, interchange_target, source_pipeline
    )
    # Raw: the interchange featurizes the source itself (a featurized read would
    # apply a DAS rotation twice).
    sources = collect_group_sources(source_pipeline or pipeline, batch)

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

    # Concatenate labels onto the base so one forward scores every label token.
    # Drop any load-time position_ids first: base is left-padded and the label is
    # right-padded, so segment-local ids would restart the label at 0 instead of
    # continuing the base. The backbone re-derives them from the *concatenated*
    # mask, where the base's right-aligned real tokens and the label's
    # left-aligned ones are contiguous, so cumsum numbers the whole real span
    # continuously regardless of padding side.
    base_encoding = dict(batch.base_encoding)
    base_encoding.pop("position_ids", None)
    for key in ("input_ids", "attention_mask"):
        base_encoding[key] = torch.cat(
            [base_encoding[key], batched_inv_label[key]], dim=-1
        )

    plans = build_plans(
        batch.units,
        batch.base_positions,
        "interchange",
        sources=sources,
        feature_indices=batch.feature_indices,
        interventions=interventions,
    )
    counterfactual_logits = forward_with_interventions(pipeline, base_encoding, plans)

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
        "base_ids": base_encoding["input_ids"][0],
        "base_masks": base_encoding["attention_mask"][0],
        "base_inputs": pipeline.dump(base_encoding["input_ids"][0]),
        "base_positions": batch.base_positions,
        "source_positions": batch.source_positions,
        "feature_indices": batch.feature_indices,
        "counterfactual_masks": [
            c["attention_mask"][0] for c in batch.source_encodings
        ],
        "counterfactual_ids": [c["input_ids"][0] for c in batch.source_encodings],
        "counterfactual_inputs": [
            pipeline.dump(c["input_ids"][0]) for c in batch.source_encodings
        ],
    }

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
