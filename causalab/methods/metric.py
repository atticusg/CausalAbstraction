"""Metric computation utilities for intervention experiments."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Callable, Sequence, Tuple, Any
import copy
import math
import torch
import torch.nn.functional as F
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.causal_model import CausalModel

from causalab.neural.pipeline import GenerationResult, LMPipeline
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
    step_scores: list[torch.Tensor] | None,
    var_indices: torch.Tensor | list[list[int]],
    full_vocab_softmax: bool = False,
) -> torch.Tensor | None:
    """Convert per-step intervention scores to joint probability distributions.

    Args:
        step_scores: The flat per-step scores of a generation run —
            :attr:`~causalab.neural.pipeline.GenerationResult.scores`: one
            ``(N, V)`` tensor per generated step (EU5b, #487; the pre-EU5a
            per-batch nesting is gone). ``None`` or empty when scores were
            not requested.
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

    if not step_scores:
        return None

    N = step_scores[0].shape[0]
    joint_NW = torch.ones(N, W_cats)

    # Single generation step: pass the full variant lists to class_probabilities
    if len(step_scores) == 1:
        probs = class_probabilities(
            step_scores[0], var_token_seqs, full_vocab_softmax=full_vocab_softmax
        )
        joint_NW = probs.cpu()
    else:
        # Multi-step: each inner list is a step sequence, not variants
        for k, logits_NV in enumerate(step_scores):
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
    results: Dict[Tuple[Any, ...], GenerationResult],
    dataset: list[CounterfactualExample],
    causal_model: CausalModel,
    target_variable_groups: List[Tuple[str, ...]],
    metric: Callable[[Any, Any], bool],
) -> Dict[str, Any]:
    """Score intervention outputs against causal model expectations for each variable group.

    ``results`` maps each key to its run's flat
    :class:`~causalab.neural.pipeline.GenerationResult` (EU5b, #487). Each
    ``results_by_key`` entry embeds the legacy ``raw_results`` dict view
    (:meth:`~causalab.neural.pipeline.GenerationResult.to_raw_results`) —
    the io boundary (``io.artifacts.save_intervention_results``) consumes
    that view, so the stored-artifact schema is unchanged for runs with
    ``batch_size >= n_examples``; legacy multi-batch runs stored one inner
    ``"string"`` list per batch (and single-example batches a bare str),
    where ``to_raw_results()`` always emits the one-synthetic-batch nesting.
    """
    # Create a metric for each variable group and score
    scores_by_variable: Dict[Tuple[str, ...], Dict[Tuple[Any, ...], float]] = {}
    for var_group in target_variable_groups:
        # Create metric for this variable group
        interchange_metric = make_causal_metric(metric, var_group)

        # Score using the core scoring function
        scores = score_intervention_outputs(
            results=results,
            dataset=dataset,
            metric=interchange_metric,
            causal_model=causal_model,
        )
        scores_by_variable[var_group] = scores

    # Build results_by_key structure
    results_by_key = {}
    for key in results.keys():
        scores_for_key = {
            str(var_group): scores_by_variable[var_group][key]
            for var_group in target_variable_groups
        }
        key_avg_score = float(sum(scores_for_key.values()) / len(scores_for_key))

        results_by_key[key] = {
            "scores_by_variable": scores_for_key,
            "avg_score": key_avg_score,
            # The io boundary consumes the legacy raw_results dict view
            # (EU5b, #487; schema unchanged at batch_size >= n_examples —
            # see the docstring's qualifier).
            "raw_results": results[key].to_raw_results(),
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
    intervention scoring (``make_causal_metric``, ``score_label_predictions``)
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
            scores = result.scores or []

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

    Execution only — one ``pipeline.generate`` per batch; the grading semantics
    live in :func:`score_base_outputs`, which this delegates to (and which
    Plan-saved outputs reach directly via :func:`outputs_from_logits`), so the
    answer-scoring semantics exist exactly once (MX1, #408).

    Returns dict with keys: accuracy, correct, total, prob_accuracy.
    """
    single_token = pipeline.max_new_tokens == 1
    outputs: List[Dict[str, Any]] = []
    n_batches = math.ceil(len(dataset) / batch_size)
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(dataset))
        batch_inputs = [ex["input"] for ex in dataset[start:end]]

        result = pipeline.generate(batch_inputs)
        strings = result.strings
        scores = result.scores or []

        for bi in range(len(batch_inputs)):
            output: Dict[str, Any] = {"string": strings[bi]}
            # Keep only the first step's row — the sole score consumer below is
            # the single-token prob grader; carrying every step's full-vocab
            # logits would multiply the memory footprint for nothing.
            if single_token and scores:
                output["scores"] = [scores[0][bi]]
            outputs.append(output)

    return score_base_outputs(
        outputs,
        dataset,
        checker,
        tokenizer=pipeline.tokenizer,
        answer_fn=answer_fn,
        single_token=single_token,
    )


def score_base_outputs(
    outputs: List[Dict[str, Any]],
    dataset: list[CounterfactualExample],
    checker: Callable[[dict, str], bool],
    *,
    tokenizer: Any = None,
    answer_fn: Callable[[CounterfactualExample], str | list[str]] | None = None,
    single_token: bool | None = None,
) -> dict:
    """Score per-example base outputs — :func:`compute_base_accuracy`'s grading
    semantics, decoupled from execution (MX1, #408).

    ``outputs`` is one ``{"string": ..., "scores": [...]}`` dict per example —
    :func:`compute_base_outputs`'s shape, also produced from Plan-saved logits
    by :func:`outputs_from_logits` — aligned with ``dataset``.

    ``prob_accuracy`` (mean probability mass on valid answer tokens, full-vocab
    softmax over each output's first score row) needs ``tokenizer`` to
    enumerate the answers' single-token forms, and single-token outputs.
    ``single_token`` defaults to "every output carries exactly one score row";
    ``compute_base_accuracy`` passes its ``max_new_tokens == 1`` criterion
    explicitly. ``None`` when not computed.

    Returns dict with keys: accuracy, correct, total, prob_accuracy.
    """
    if len(outputs) != len(dataset):
        raise ValueError(
            f"outputs and dataset are misaligned: {len(outputs)} outputs for "
            f"{len(dataset)} examples"
        )
    if single_token is None:
        single_token = bool(outputs) and all(
            len(o.get("scores") or []) == 1 for o in outputs
        )
    grade_probs = single_token and tokenizer is not None

    correct = 0
    total = 0
    prob_sum = 0.0
    for output, ex in zip(outputs, dataset):
        generated = output["string"]
        expected = answer_fn(ex) if answer_fn is not None else ex["input"]["raw_output"]
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
        scores = output.get("scores") or []
        if grade_probs and scores:
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
                probs = F.softmax(scores[0].float(), dim=-1)  # (vocab_size,)
                prob_sum += probs[list(all_token_ids)].sum().item()

    accuracy = correct / total if total > 0 else 0.0
    prob_accuracy = prob_sum / total if (grade_probs and total > 0) else None
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
    ``{"string": <generated text>, "sequences": (1, max_new_tokens) generated
    token ids, "scores": [logits_tok0 (V,), ...]}``.  This is the shape
    ``score_intervention_outputs`` expects for ``original_outputs`` when a
    metric has ``needs_original_output=True`` (e.g. the distribution-shift
    metric), and what :func:`as_generation_result` re-flattens for metric
    scoring.
    """
    outputs: List[Dict[str, Any]] = []
    n_batches = math.ceil(len(dataset) / batch_size)
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(dataset))
        batch_examples = dataset[start:end]
        batch_inputs = [ex["input"] for ex in batch_examples]

        result = pipeline.generate(batch_inputs)
        strings = result.strings
        scores = result.scores or []
        for bi in range(len(batch_examples)):
            outputs.append(
                {
                    "string": strings[bi],
                    "sequences": result.sequences[bi : bi + 1],
                    "scores": [scores[t][bi].cpu() for t in range(len(scores))],
                }
            )
    return outputs


def outputs_from_logits(
    pipeline: Any,
    logits: torch.Tensor | Sequence[torch.Tensor],
) -> List[Dict[str, Any]]:
    """Per-example outputs from Plan-saved logits — the scoring adapter (MX1, #408).

    ``logits`` is a ``PlanResult.logits`` value (``(batch, seq, vocab)``) or the
    per-example ``(seq, vocab)`` rows ``collect_dataset_features``'s
    ``collect_output_logits=True`` returns. A Plan forward is prefill-only, so
    there is exactly ONE generated step: the last position's next-token
    distribution. Each output is ``{"string": <argmax token decoded>,
    "sequences": (1, 1) argmax token id, "scores": [(vocab,) last-position
    logits]}`` — :func:`compute_base_outputs`'s shape — so Plan-saved logits
    feed the same scorers as generation runs: :func:`score_base_outputs`
    directly, :func:`score_intervention_outputs` through
    :func:`as_generation_result`.
    """
    last = torch.stack([row[-1].detach().cpu() for row in logits], dim=0)  # (N, V)
    token_ids = last.argmax(dim=-1)
    strings = pipeline.tokenizer.batch_decode(
        token_ids.unsqueeze(1), skip_special_tokens=True
    )
    return [
        {
            "string": s,
            "sequences": token_ids[i : i + 1].unsqueeze(1),
            "scores": [last[i]],
        }
        for i, s in enumerate(strings)
    ]


def as_generation_result(outputs: List[Dict[str, Any]]) -> GenerationResult:
    """Flatten per-example outputs into one
    :class:`~causalab.neural.pipeline.GenerationResult` for
    :func:`score_intervention_outputs` (EU5b, #487 — replaces the retired
    ``as_raw_results`` synthetic-batch adapter).

    ``outputs`` is the per-example list shape (:func:`compute_base_outputs` /
    :func:`outputs_from_logits`): each entry carries ``string``, a
    ``sequences`` row ``(1, W)``, and optionally per-step ``scores``.
    ``scores`` is set on the result only when every output carries the same,
    non-zero number of steps — otherwise the flattened value scores strings
    only (the retired adapter's contract).
    """
    step_counts = {len(o.get("scores") or []) for o in outputs}
    scores: list[torch.Tensor] | None = None
    if len(step_counts) == 1 and step_counts != {0}:
        n_tokens = next(iter(step_counts))
        scores = [
            torch.stack([o["scores"][t] for o in outputs], dim=0)
            for t in range(n_tokens)
        ]
    return GenerationResult(
        sequences=torch.cat([o["sequences"] for o in outputs], dim=0)
        if outputs
        else torch.empty((0, 0), dtype=torch.long),
        strings=[o["string"] for o in outputs],
        scores=scores,
    )


def score_intervention_outputs(
    results: Dict[Tuple[Any, ...], GenerationResult],
    dataset: list[CounterfactualExample],
    metric: InterchangeMetric,
    causal_model: CausalModel | None = None,
    original_outputs: List[Dict[str, Any]] | None = None,
) -> Dict[Tuple[Any, ...], float]:
    """Score pre-computed intervention outputs. Returns dict mapping keys to average scores.

    ``results`` maps each swept key to the flat
    :class:`~causalab.neural.pipeline.GenerationResult` its run produced
    (EU5b, #487): ``strings`` and per-step ``scores`` are consumed directly —
    the pre-EU5a batch-nested ``raw_results`` flattening is gone. A result
    holding top-k-compressed ``scores_top_k`` is refused: metrics consume
    full-vocabulary per-step tensors, so score with ``output_scores=True``
    (not an int).
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

    for key, result in results.items():
        if result.scores_top_k is not None:
            raise ValueError(
                f"cannot score key {key!r}: its result carries top-k-compressed "
                "scores (scores_top_k), but metrics consume full-vocabulary "
                "per-step tensors. Generate with output_scores=True (not an "
                "int) for metric scoring."
            )
        # Flat per-step scores, one (n_examples, vocab) tensor per generated
        # step — exactly the shape metric fns index as
        # ``scores[step][example_idx]``.
        flat_scores = result.scores

        # Compute score for each example
        key_scores: List[float] = []
        for idx, output_string in enumerate(result.strings):
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


def score_label_predictions(
    pipeline: Any,
    pred_ids: torch.Tensor,
    labels: List[Any],
    checker: Callable[[Dict[str, Any], Any], float | bool],
) -> Dict[str, Any]:
    """Score label-span predictions against expected labels — the answer-scoring
    half of the retired pyvene ``LM_loss_and_metric_fn``, off the ED3 loss
    slice (MX1, #408).

    ``pred_ids`` is the ``(batch, n_label_tokens)`` argmax that
    :func:`causalab.neural.trainable.traced_label_loss` returns. The loss-path
    forward is ED3's (``causalab.neural.trainable``) — this function only
    scores its predictions, so the answer-scoring semantics are not
    re-implemented alongside the loss. ``labels`` are per-example expected
    labels, passed to ``checker`` verbatim (an
    :func:`as_label_checker`-wrapped task checker normalizes
    ``{"string": ...}`` dicts).

    Returns ``{"accuracy": mean score, "scores": per-example floats}``.
    """
    if pred_ids.shape[0] != len(labels):
        raise ValueError(
            f"pred_ids and labels are misaligned: {pred_ids.shape[0]} rows for "
            f"{len(labels)} labels"
        )
    scores: List[float] = []
    for i in range(pred_ids.shape[0]):
        pred_str = pipeline.dump(pred_ids[i : i + 1])
        score = checker({"string": pred_str}, labels[i])
        if isinstance(score, torch.Tensor):
            score = score.item()
        scores.append(float(score))
    # Empty → 1.0 mirrors the retired ``LM_loss_and_metric_fn`` (vacuous truth on an empty
    # eval batch), so MX2's reroute is a drop-in swap.
    accuracy = sum(scores) / len(scores) if scores else 1.0
    return {"accuracy": accuracy, "scores": scores}
