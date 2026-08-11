"""Baseline analysis: unintervened model accuracy, per-class reference distributions,
and task rendering samples.

This is the generic first step to run on any task. It answers:
  1. Can the model solve this task at all?
  2. Where is the model confused across classes?
  3. What does a rendered example look like? (Sanity check on task formatting.)

No Hellinger / simplex geometry here — that lives in `output_manifold`.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, cast

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from causalab.io.plots.distance_plots import plot_matrix_heatmap
from causalab.io.plots.figure_format import resolve_figure_format_from_analysis
from causalab.io.plots.plot_utils import resolve_task_colormap
from causalab.analyses.path_steering.path_visualization import (
    plot_ground_truth_heatmaps,
)
from causalab.methods.metric import (
    _normalize_var_indices,  # pyright: ignore[reportPrivateUsage]
    answer_token_forms,
    compute_base_accuracy,
    compute_reference_distributions,
)
from causalab.runner.helpers import (
    _task_config_for_metadata,  # pyright: ignore[reportPrivateUsage]
    prepare_datasets,
    get_output_token_ids,
    resolve_task,
)
from causalab.io.pipelines import load_pipeline
from causalab.io.artifacts import (
    save_experiment_metadata,
    save_json_results,
    save_tensor_results,
)

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "baseline"


def _render_dataset(dataset: list) -> list[dict[str, str]]:
    """Extract (raw_input, raw_output) pairs from a generated dataset."""
    return [
        {
            "raw_input": str(ex["input"]["raw_input"]),
            "raw_output": str(ex["input"]["raw_output"]),
        }
        for ex in dataset
    ]


def _answer_class_index(
    ex: dict, score_token_groups: list[list[int]], tokenizer
) -> int | None:
    """Column index of an example's correct answer within the score-token space.

    The confusion / ground-truth rows should reflect the class the model is
    *supposed to emit* (its ``raw_output``), not the localization target
    variable — the two differ whenever a non-output variable (e.g. query
    polarity) also drives the answer (#259).

    The answer is resolved through the *same* token-id groups the confusion
    columns are built from (the ``output_tokens`` resolver, via
    ``get_output_token_ids``), tokenizing ``raw_output`` with
    ``answer_token_forms`` — the strip/case-tolerant probability-grader forms.
    Matching on token ids rather than a bare ``str.index`` keeps
    this on the #167 token contract and is robust to formatting gaps between
    ``raw_output`` and the score labels (e.g. ``"5"``/``"five"``,
    ``" orange"``/``"orange"``), which a string compare would silently miss.
    The boolean ``task.checker`` is the authority for scalar base accuracy, not
    for this token-probability path, so it is deliberately not used here (and
    its ``startswith`` variants would over-match across columns).

    Assumes single-token scoring: a multi-step ``raw_output`` list is reduced to
    its first step, matching the last-position read in
    ``compute_reference_distributions``'s precomputed-logits path. The polarity
    tasks this guards are all single-token, so the two coincide.

    Returns ``None`` when the answer tokenizes to nothing in the score space
    (then the caller keeps the target-variable grouping).
    """
    raw_out = ex["input"]["raw_output"]
    if isinstance(raw_out, list):  # multi-step tasks emit a per-step list
        raw_out = raw_out[0] if raw_out else ""
    answer_ids: set[int] = set()
    for form in answer_token_forms(str(raw_out)):
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            answer_ids.add(ids[0])
    if not answer_ids:
        return None
    for cls_idx, group in enumerate(score_token_groups):
        if answer_ids.intersection(group):
            return cls_idx
    return None


def _answer_rows_differ_from_target(
    dataset: list, task: Any, score_token_groups: list[list[int]], tokenizer
) -> tuple[bool, list[int] | None]:
    """Detect mixed-polarity contamination of the target-variable rows (#259).

    Returns ``(contaminated, answer_classes)``. ``contaminated`` is True only
    when some ``intervention_value_index`` group maps to more than one answer
    class — the signature of an answer that depends on a variable *other* than
    the localization target, so averaging each target row washes the diagonal
    to ~chance (e.g. ``magnitude_order`` {A,B} under mixed min/max polarity).
    ``answer_classes`` is the per-example answer-class index (reused by the
    caller to regroup without a second pass), or ``None`` when any answer is
    unresolvable in the score space (then the caller cannot relabel rows and
    keeps the target grouping).
    """
    answer_classes: list[int] = []
    for ex in dataset:
        ac = _answer_class_index(ex, score_token_groups, tokenizer)
        if ac is None:
            return False, None
        answer_classes.append(ac)
    groups: dict[int, set[int]] = {}
    for ex, ac in zip(dataset, answer_classes):
        groups.setdefault(task.intervention_value_index(ex), set()).add(ac)
    contaminated = any(len(classes) > 1 for classes in groups.values())
    return contaminated, answer_classes


def _top_logits_example(
    raw_out: Any, prediction: str, top_tokens: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build one ``top_logits.json`` example record.

    ``raw_out`` is the task's ground-truth label (a single token, or a per-step
    list for multi-step generation tasks); ``prediction`` is the model's decoded
    top-1 token (i.e. ``top_tokens[0]["token"]``). The label is stored as
    ``expected_output`` — explicitly *not* ``prediction`` — so the record can't
    be misread as the model's output. ``correct`` compares the prediction to the
    label (a list label is reduced to its first step, since the model emits one
    token here).
    """
    raw_out_str = (
        str(raw_out[0]) if isinstance(raw_out, list) and raw_out else str(raw_out or "")
    )
    return {
        "expected_output": raw_out,
        "prediction": prediction,
        "top_tokens": top_tokens,
        "correct": prediction.strip() == raw_out_str.strip(),
    }


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the baseline analysis: accuracy, reference distributions, task sanity checks.

    All artifacts are saved to ``{experiment_root}/baseline/``.
    """
    # --- Load config ---
    analysis = cfg[ANALYSIS_NAME]
    figure_fmt = resolve_figure_format_from_analysis(analysis)
    out_dir = os.path.join(cfg.experiment_root, ANALYSIS_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # --- Load task ---
    task, task_cfg_raw = resolve_task(  # pyright: ignore[reportUnusedVariable]
        task_name=cfg.task.name,
        task_config=cast(
            dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True)
        ),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )

    # --- Load dataset ---
    train_dataset, test_dataset = prepare_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        balanced=cfg.task.get("balanced", False),
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
        filter_correct=False,
    )

    # --- Save rendered train/test pairs for downstream inspection ---
    save_json_results(
        {"samples": _render_dataset(train_dataset)},
        out_dir,
        "train_samples.json",
    )
    logger.info("Saved %d rendered train samples.", len(train_dataset))
    if test_dataset:
        save_json_results(
            {"samples": _render_dataset(test_dataset)},
            out_dir,
            "test_samples.json",
        )
        logger.info("Saved %d rendered test samples.", len(test_dataset))

    # --- Load LM ---
    pipeline = load_pipeline(
        model_name=cfg.model.name,
        task=task,
        max_new_tokens=cfg.task.max_new_tokens,
        device=cfg.model.device,
        dtype=cfg.model.get("dtype"),
        eager_attn=cfg.model.get("eager_attn"),
        use_chat_template=cfg.model.get("chat_template", False),
        chat_answer_directive=cfg.model.get("chat_answer_directive"),
    )
    score_token_ids, n_score_tokens = get_output_token_ids(task, pipeline)
    n_classes = (
        len(task.intervention_values) if task.intervention_variable else n_score_tokens
    )

    # --- Base accuracy ---
    base_acc = compute_base_accuracy(
        dataset=train_dataset,
        pipeline=pipeline,
        batch_size=analysis.batch_size,
        # ``score_answer`` is set only when the task's config selects a
        # non-default scoring convention (e.g. MCQA ``score_by: value``);
        # otherwise None → score against ``raw_output`` as before.
        answer_fn=task.score_answer,
        # ``checker`` is the task's own match fn (every task ships one; e.g.
        # entity_binding's ``startswith``), the sole match authority so
        # ``max_new_tokens > 1`` continuations score correctly.
        checker=task.checker,
    )
    save_json_results(base_acc, out_dir, "accuracy.json")
    logger.info("Base accuracy: %.1f%%", base_acc["accuracy"] * 100)

    # --- Per-class reference distributions + confusion heatmap ---
    if score_token_ids is not None and task.intervention_values:
        output_logits: list[list[torch.Tensor]] = []
        n_batches = math.ceil(len(train_dataset) / analysis.batch_size)
        for batch_idx in range(n_batches):
            start = batch_idx * analysis.batch_size
            end = min(start + analysis.batch_size, len(train_dataset))
            batch_inputs = [ex["input"] for ex in train_dataset[start:end]]
            result = pipeline.generate(batch_inputs)
            scores = result.scores or []
            for bi in range(len(batch_inputs)):
                output_logits.append([s[bi].cpu() for s in scores])
        logger.info(
            "Collected per-example logits: %d examples, %d steps each",
            len(output_logits),
            len(output_logits[0]) if output_logits else 0,
        )

        # --- Diagnostic: full output distributions + decoded top logits ---
        top_k = 10
        all_probs = torch.stack(
            [F.softmax(ol[-1].float(), dim=-1) for ol in output_logits]
        )  # (n_examples, vocab_size)
        save_tensor_results(
            {"dists": all_probs}, out_dir, "full_output_dists.safetensors"
        )
        logger.info("Saved full output distributions: %s", all_probs.shape)

        top_vals, top_ids = torch.topk(all_probs, top_k, dim=-1)
        tokenizer = pipeline.tokenizer
        top_logits_examples = []
        for i, ex in enumerate(train_dataset):
            # ``raw_output`` is the task's ground-truth label; for multi-step
            # generation tasks (e.g. graph_walk) it is a per-step list.
            raw_out = ex["input"]["raw_output"]
            generated = tokenizer.decode(top_ids[i, 0].item())
            top_tokens = [
                {
                    "token": tokenizer.decode([top_ids[i, j].item()]),
                    "token_id": top_ids[i, j].item(),
                    "prob": round(top_vals[i, j].item(), 6),
                }
                for j in range(top_k)
            ]
            top_logits_examples.append(
                _top_logits_example(raw_out, generated, top_tokens)
            )
        save_json_results(
            {"top_k": top_k, "examples": top_logits_examples},
            out_dir,
            "top_logits.json",
        )
        logger.info(
            "Saved decoded top-%d logits for %d examples.",
            top_k,
            len(top_logits_examples),
        )

        if n_classes is None:
            raise ValueError(
                "n_classes is None — task lacks intervention_variable and "
                "output_tokens."
            )
        # Full-vocab softmax averages per class — consumed by locate, activation_manifold.
        ref_dists = compute_reference_distributions(
            dataset=train_dataset,
            # score_token_ids may be a Tensor; flatten/cast for the runtime
            # list-of-ids contract.
            score_token_ids=cast(list[int], score_token_ids),
            n_classes=n_classes,
            example_to_class=task.intervention_value_index,
            # output_logits is list[list[Tensor]] per multi-step generation;
            # compute_reference_distributions accepts the list-of-list shape
            # for multi-step tasks even though the type annotation is narrower.
            output_logits=cast("list[torch.Tensor]", output_logits),
            score_token_index=0,
            full_vocab_softmax=True,
        )
        save_tensor_results(
            {"dists": ref_dists}, out_dir, "per_class_output_dists.safetensors"
        )
        logger.info("Saved per-class output distributions: %s", ref_dists.shape)

        task_colormap = resolve_task_colormap(cfg.task, "rainbow")

        if task.class_token_ids is not None:
            # Dynamic output tokens (e.g. MCQA): look up per-example class
            # probabilities from the full-vocab softmax using example-specific
            # token IDs, then average per true class. The trailing column
            # accumulates residual mass on task-unrelated tokens.
            # n_classes is non-None here (validated above in the same branch).
            # No answer-space relabel here (#259): these rows are
            # ``answer_position``, which *is* the model's answer, so there is no
            # target-vs-answer indirection to correct.
            _nc: int = n_classes
            class_prob_accum = torch.zeros(_nc, _nc + 1)
            class_totals = torch.zeros(_nc)
            for i, ex in enumerate(train_dataset):
                true_cls = task.intervention_value_index(ex)
                class_totals[true_cls] += 1
                ex_token_ids = task.class_token_ids(ex, tokenizer)
                class_mass = 0.0
                for cls_idx, tid in enumerate(ex_token_ids):
                    p = all_probs[i, tid].item()
                    class_prob_accum[true_cls, cls_idx] += p
                    class_mass += p
                class_prob_accum[true_cls, -1] += max(0.0, 1.0 - class_mass)
            class_prob_dists = class_prob_accum / class_totals.clamp(min=1).unsqueeze(1)
            class_labels = [str(v) for v in task.intervention_values]

            try:
                plot_ground_truth_heatmaps(
                    dists=class_prob_dists[:, :-1],
                    variable_values=task.intervention_values,
                    output_dir=out_dir,
                    score_labels=class_labels,
                    colormap=task_colormap,
                    full_vocab_softmax=True,
                    title_prefix="Ground truth (no intervention)",
                    figure_format=figure_fmt,
                )
            except Exception as e:
                logger.warning("Ground truth path plot failed: %s", e)
            try:
                plot_matrix_heatmap(
                    class_prob_dists[:, :-1],
                    row_labels=class_labels,
                    col_labels=class_labels,
                    output_dir=out_dir,
                    filename="confusion_heatmap",
                    figure_format=figure_fmt,
                    title="Confusion (no intervention)",
                    xlabel="Predicted class",
                )
            except Exception as e:
                logger.warning("Confusion heatmap failed: %s", e)
        else:
            # Fixed score tokens: use token-probability distributions.
            # One label per score column. When the task declares ``output_tokens``
            # for the intervention variable, the columns are its distinct
            # form-groups (the score space may be the deduped union of answer
            # forms — e.g. entity_binding's 12 entity tokens — which the
            # intervention variable's own 2 positional values do not name), so
            # derive the labels from those groups (#296). Otherwise label by the
            # intervention values themselves.
            _ot = task.causal_model.output_tokens
            _var = task.intervention_variable
            if _ot and _var and _ot.get(_var):
                from causalab.methods.output_tokens import form_group_labels

                score_labels = form_group_labels(_ot[_var])
            else:
                score_labels = [str(v) for v in task.intervention_values]

            # Rows default to the localization target (``intervention_values``).
            # But the model emits ``raw_output`` (the answer), which can differ
            # from the target variable when another variable (e.g. query
            # polarity) also drives the answer — then a single target row mixes
            # answers and averages to ~chance even at high accuracy (#259).
            # Detect that and, when present, plot the confusion / ground-truth
            # against the answer space so the diagonal reflects accuracy. The
            # saved ``per_class_output_dists`` artifact above is left grouped by
            # target class — its consumers (locate, activation_manifold) expect
            # per-target-class distributions.
            # Resolve answers through the same token-id groups the columns are
            # built from (``score_token_ids`` via the ``output_tokens`` resolver).
            score_token_groups = _normalize_var_indices(score_token_ids)
            contaminated, answer_classes = _answer_rows_differ_from_target(
                train_dataset, task, score_token_groups, tokenizer
            )
            if contaminated:
                assert answer_classes is not None  # contamination ⇒ all resolved
                logger.warning(
                    "baseline confusion: the model's answer (raw_output) differs "
                    "from the target variable %r for some examples (e.g. mixed "
                    "polarity) — plotting the confusion / ground-truth against the "
                    "answer space {%s} rather than the target classes, so the "
                    "diagonal reflects accuracy (#259).",
                    task.intervention_variable,
                    ", ".join(score_labels),
                )

                # Reuse the answer classes already computed by the detector
                # (keyed by example identity, since ``compute_reference_distributions``
                # iterates this same list) rather than re-deriving them.
                answer_class_by_ex = {
                    id(ex): ac for ex, ac in zip(train_dataset, answer_classes)
                }
                plot_dists = compute_reference_distributions(
                    dataset=train_dataset,
                    score_token_ids=cast(list[int], score_token_ids),
                    n_classes=len(score_labels),
                    example_to_class=lambda ex: answer_class_by_ex[id(ex)],
                    output_logits=cast("list[torch.Tensor]", output_logits),
                    score_token_index=0,
                    full_vocab_softmax=True,
                )
                row_values: list = list(score_labels)
                row_labels = list(score_labels)
                gt_title = "Ground truth (no intervention, answer space)"
                cf_title = "Confusion (no intervention, answer space)"
                cf_xlabel = "Predicted answer"
            else:
                plot_dists = ref_dists
                row_values = task.intervention_values
                row_labels = [str(v) for v in task.intervention_values]
                gt_title = "Ground truth (no intervention)"
                cf_title = "Confusion (no intervention)"
                cf_xlabel = "Predicted class"

            try:
                plot_ground_truth_heatmaps(
                    dists=plot_dists,
                    variable_values=row_values,
                    output_dir=out_dir,
                    score_labels=score_labels,
                    colormap=task_colormap,
                    full_vocab_softmax=True,
                    title_prefix=gt_title,
                    figure_format=figure_fmt,
                )
            except Exception as e:
                logger.warning("Ground truth path plot failed: %s", e)
            try:
                plot_matrix_heatmap(
                    plot_dists,
                    row_labels=row_labels,
                    col_labels=score_labels,
                    output_dir=out_dir,
                    filename="confusion_heatmap",
                    figure_format=figure_fmt,
                    title=cf_title,
                    xlabel=cf_xlabel,
                )
            except Exception as e:
                logger.warning("Confusion heatmap failed: %s", e)

    # --- Metadata ---
    metadata = {
        "analysis": "baseline",
        "model": cfg.model.name,
        "task": cfg.task.name,
        "task_config": _task_config_for_metadata(
            cast(dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True))
        ),
        "n_train": cfg.task.n_train,
        "n_test": cfg.task.n_test,
        "seed": cfg.seed,
    }
    save_experiment_metadata(metadata, out_dir)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Baseline analysis complete. Output in %s", out_dir)
    return {"output_dir": out_dir, "accuracy": base_acc, "metadata": metadata}
