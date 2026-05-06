"""Baseline analysis: unintervened model accuracy, per-class reference distributions,
counterfactual-dataset sanity checks, and task rendering samples.

This is the generic first step to run on any task. It answers:
  1. Can the model solve this task at all?
  2. Where is the model confused across classes?
  3. Do the task's counterfactual generators actually distinguish the intervention
     variable from other variables? (Sanity check on experiment design.)
  4. What does a rendered example look like? (Sanity check on task formatting.)

No Hellinger / simplex geometry here — that lives in `output_manifold`.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
from typing import Any

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
    compute_base_accuracy,
    compute_reference_distributions,
)
from causalab.runner.helpers import (
    _task_config_for_metadata,
    generate_datasets,
    get_output_token_ids,
    resolve_task,
)
from causalab.io.pipelines import load_pipeline
from causalab.io.artifacts import (
    save_experiment_metadata,
    save_json_results,
    save_tensor_results,
)
from causalab.tasks.loader import load_task_counterfactuals

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "baseline"


def _collect_counterfactual_sanity(task, n_samples: int = 64) -> dict[str, Any]:
    """For every zero-arg counterfactual generator on the task, check that the
    generated dataset can distinguish the intervention variable from the other
    declared variables (i.e., counterfactuals deconfound the target).

    Returns a dict mapping generator name → {distinguishes, proportion, count}.
    """
    try:
        cf_module = load_task_counterfactuals(task.name)
    except ImportError:
        logger.info(
            "Task %r has no counterfactuals module; skipping sanity check.", task.name
        )
        return {}

    target = task.intervention_variable
    if target is None:
        logger.info(
            "Task has no intervention_variable; skipping counterfactual sanity."
        )
        return {}

    other_vars = [
        v
        for v in task.causal_model.mechanisms
        if v not in {"raw_input", "raw_output", target}
        and v not in task.causal_model.inputs
    ]

    results: dict[str, Any] = {}
    for name, fn in inspect.getmembers(cf_module, inspect.isfunction):
        if name.startswith("_"):
            continue
        try:
            sig = inspect.signature(fn)
            required = [
                p
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
            ]
            if required:
                continue
        except (TypeError, ValueError):
            continue

        try:
            dataset = [fn() for _ in range(n_samples)]
            if not dataset or "counterfactual_inputs" not in dataset[0]:
                continue
        except Exception as e:
            logger.debug("Skipping generator %s: %s", name, e)
            continue

        verdict = task.causal_model.can_distinguish_with_dataset(
            dataset, [target], other_vars or None
        )
        results[name] = {
            "distinguishes_target_from_others": bool(verdict["proportion"] > 0.99),
            "proportion": float(verdict["proportion"]),
            "count": int(verdict["count"]),
            "n_samples": n_samples,
        }
        logger.info(
            "Counterfactual %s: distinguishes %s from %s → %.2f (%d/%d)",
            name,
            target,
            other_vars,
            verdict["proportion"],
            verdict["count"],
            n_samples,
        )
    return results


def _render_dataset(dataset: list) -> list[dict[str, str]]:
    """Extract (raw_input, raw_output) pairs from a generated dataset."""
    return [
        {
            "raw_input": str(ex["input"]["raw_input"]),
            "raw_output": str(ex["input"]["raw_output"]),
        }
        for ex in dataset
    ]


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
    task, task_cfg_raw = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )

    # --- Counterfactual sanity (no model needed) ---
    cf_sanity = _collect_counterfactual_sanity(task)
    if cf_sanity:
        save_json_results(cf_sanity, out_dir, "counterfactual_sanity.json")

    # --- Load dataset ---
    train_dataset, test_dataset = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        balanced=cfg.task.get("balanced", False),
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
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
            scores = result["scores"]
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
            raw_out = ex["input"]["raw_output"]
            # For multi-step generation tasks (e.g. graph_walk) raw_output is a
            # list of per-step tokens; the model only emits one token here, so
            # compare against the first expected step.
            raw_out_str = (
                str(raw_out[0])
                if isinstance(raw_out, list) and raw_out
                else str(raw_out or "")
            )
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
                {
                    "raw_output": raw_out,
                    "top_tokens": top_tokens,
                    "correct": generated.strip() == raw_out_str.strip(),
                }
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

        # Full-vocab softmax averages per class — consumed by locate, activation_manifold.
        ref_dists = compute_reference_distributions(
            dataset=train_dataset,
            score_token_ids=score_token_ids,
            n_classes=n_classes,
            example_to_class=task.intervention_value_index,
            output_logits=output_logits,
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
            class_prob_accum = torch.zeros(n_classes, n_classes + 1)
            class_totals = torch.zeros(n_classes)
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
                    filename=f"confusion_heatmap.{figure_fmt}",
                    title="Confusion (no intervention)",
                    xlabel="Predicted class",
                )
            except Exception as e:
                logger.warning("Confusion heatmap failed: %s", e)
        else:
            # Fixed score tokens: use token-probability distributions.
            score_labels = (
                [str(v) for v in task.output_token_values]
                if task.output_token_values
                else [str(v) for v in task.intervention_values]
            )
            try:
                plot_ground_truth_heatmaps(
                    dists=ref_dists,
                    variable_values=task.intervention_values,
                    output_dir=out_dir,
                    score_labels=score_labels,
                    colormap=task_colormap,
                    full_vocab_softmax=True,
                    title_prefix="Ground truth (no intervention)",
                    figure_format=figure_fmt,
                )
            except Exception as e:
                logger.warning("Ground truth path plot failed: %s", e)
            try:
                plot_matrix_heatmap(
                    ref_dists,
                    row_labels=[str(v) for v in task.intervention_values],
                    col_labels=score_labels,
                    output_dir=out_dir,
                    filename=f"confusion_heatmap.{figure_fmt}",
                    title="Confusion (no intervention)",
                    xlabel="Predicted class",
                )
            except Exception as e:
                logger.warning("Confusion heatmap failed: %s", e)

    # --- Metadata ---
    metadata = {
        "analysis": "baseline",
        "model": cfg.model.name,
        "task": cfg.task.name,
        "task_config": _task_config_for_metadata(
            OmegaConf.to_container(cfg.task, resolve=True)
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
