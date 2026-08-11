"""Interchange score grid scan for the locate analysis."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Mapping

import torch

from causalab.io.artifacts import (
    load_json_results,
    load_tensor_results,
    save_tensors_with_meta,
)

logger = logging.getLogger(__name__)


def _load_baseline_artifacts(
    experiment_root: str,
) -> tuple[float, torch.Tensor | None]:
    """Load accuracy and ref_dists from the baseline analysis."""
    baseline_dir = os.path.join(experiment_root, "baseline")

    acc_path = os.path.join(baseline_dir, "accuracy.json")
    if os.path.exists(acc_path):
        base_accuracy = load_json_results(baseline_dir, "accuracy.json")["accuracy"]
        logger.info("Loaded base accuracy from baseline: %.1f%%", base_accuracy * 100)
    else:
        base_accuracy = float("nan")
        logger.warning("No baseline accuracy found at %s", acc_path)

    ref_path = os.path.join(baseline_dir, "per_class_output_dists.safetensors")
    if os.path.exists(ref_path):
        ref_dists_fvs = load_tensor_results(
            baseline_dir, "per_class_output_dists.safetensors"
        )["dists"]
        logger.info("Loaded ref_dists from baseline: %s", ref_dists_fvs.shape)
    else:
        ref_dists_fvs = None
        logger.warning("No baseline ref_dists found at %s", ref_path)

    return base_accuracy, ref_dists_fvs


def run_interchange_scan(
    pipeline,
    task,
    layers: list[int],
    train_dataset: list,
    test_dataset: list,
    mode: str,
    score_token_ids: list[int] | list[list[int]],
    n_classes: int,
    batch_size: int,
    n_steer: int,
    out_dir: str,
    position_names: list[str] | None = None,
    comparison_fn: Callable | None = None,
    intervention_metric: str = "causal_label",
    experiment_root: str | None = None,
    colormap: str | None = None,
    figure_format: str = "png",
    source_pipeline=None,
    prescan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run interchange score scan over a (layer × token_position) grid.

    Args:
        intervention_metric: For ``mode='pairwise'``, the name passed to
            ``resolve_interchange_metric`` to build the ``InterchangeMetric``
            (default ``"causal_label"`` — score the patched output against the
            causal model's expected counterfactual label).
        source_pipeline: If provided, activations are collected from this
            pipeline and patched into ``pipeline`` (cross-model patching).
            ``None`` (default) uses standard single-model patching.
        prescan: Optional attribution-patching fail-fast gate (CAP3, #456):
            ``{"enabled": bool, "top_k": int, "n_examples": int | None}``.
            When enabled (``mode: pairwise`` only), a one-backward
            gradient × Δactivation pre-scan scores every cell and exact
            interchange runs only on the ``top_k`` survivors; both scores are
            reported (``"prescan"`` result block) where both are computed.

    Returns:
        Dict with ``scores_per_cell`` (keys ``(layer, pos_id)``), summary
        ``scores_per_layer`` (best — highest — score over positions per layer),
        ``base_accuracy``, ``token_position_ids``, and — when the gate ran —
        ``prescan`` (approx scores for every cell, survivors, and
        approx-vs-exact agreement diagnostics).
    """
    from causalab.methods.interchange import (
        run_centroid_layer_scan,
        run_layer_scan,
    )
    from causalab.methods.metric import (
        compute_base_accuracy,
        compute_base_outputs,
        compute_reference_distributions,
    )
    from causalab.runner.helpers import (
        build_targets_for_grid,
        resolve_interchange_metric,
    )

    # score_token_ids is typed as list[int] | list[list[int]] (non-None);
    # upstream callers must validate non-None before this point.

    base_accuracy = float("nan")
    ref_dists_fvs = None
    if experiment_root is not None:
        base_accuracy, ref_dists_fvs = _load_baseline_artifacts(experiment_root)

    if base_accuracy != base_accuracy:  # isnan
        base_acc = compute_base_accuracy(
            dataset=test_dataset,
            pipeline=pipeline,
            batch_size=batch_size,
            # Score the pre-flight gate with the task's own match semantics so a
            # ``max_new_tokens > 1`` task isn't artifactually gated to 0%.
            checker=task.checker,
        )
        base_accuracy = base_acc["accuracy"]

    targets, token_positions = build_targets_for_grid(
        pipeline,
        task,
        layers,
        position_names=position_names,
    )
    token_position_ids = [tp.id for tp in token_positions]

    # Optional attribution-patching fail-fast gate (CAP3, #456): approximate
    # every cell with one forward+backward per batch, keep the top-k, and run
    # exact interchange only on the survivors.
    prescan_cfg = dict(prescan) if prescan is not None else {}
    prescan_block: dict[str, Any] | None = None
    if prescan_cfg.get("enabled"):
        if mode != "pairwise":
            raise ValueError(
                "locate.prescan approximates pairwise interchange effects; it "
                f"cannot gate mode={mode!r}. Use mode: pairwise or disable "
                "the prescan."
            )
        if source_pipeline is not None:
            raise ValueError(
                "locate.prescan does not support cross-model patching — the "
                "gradient × Δactivation approximation is defined on one "
                "model's forward. Disable the prescan or source_model."
            )
        from causalab.methods.interchange import (
            counterfactual_logit_diff_ids,
            run_attribution_prescan,
            select_top_k,
        )

        n_examples = prescan_cfg.get("n_examples")
        subset = test_dataset if not n_examples else test_dataset[: int(n_examples)]
        pair_ids = counterfactual_logit_diff_ids(pipeline, subset, task.causal_model)
        approx_scores = run_attribution_prescan(
            targets, subset, pipeline, batch_size, pair_ids
        )
        top_k = int(prescan_cfg.get("top_k", 10))
        # Rank by |approx|: the linearization's sign is unreliable at early
        # layers, but the magnitude separates live cells from dead ones (see
        # causalab/methods/interchange/attribution.py).
        survivors = select_top_k(approx_scores, top_k, by_abs=True)
        logger.info(
            "Attribution pre-scan: exact interchange runs on %d/%d cells "
            "(top_k=%d over %d pairs)",
            len(survivors),
            len(targets),
            top_k,
            len(subset),
        )
        targets = {key: targets[key] for key in survivors}
        prescan_block = {
            "approx_scores_per_cell": approx_scores,
            "survivors": survivors,
            "top_k": top_k,
            "n_examples": len(subset),
        }

    ref_dists = None
    if mode == "centroid":
        if not task.intervention_values:
            raise ValueError("Task must have intervention_values for centroid mode")
        if ref_dists_fvs is None or ref_dists_fvs.shape[0] != n_classes:
            if ref_dists_fvs is not None:
                logger.info(
                    "Baseline ref_dists shape %s doesn't match n_classes=%d; recomputing",
                    ref_dists_fvs.shape,
                    n_classes,
                )
            ref_dists_fvs = compute_reference_distributions(
                dataset=train_dataset,
                score_token_ids=score_token_ids,
                n_classes=n_classes,
                example_to_class=task.intervention_value_index,
                pipeline=pipeline,
                batch_size=batch_size,
                score_token_index=0,
                full_vocab_softmax=True,
            )
        ref_dists = ref_dists_fvs / ref_dists_fvs.sum(dim=-1, keepdim=True).clamp(
            min=1e-10
        )

    all_patched_dists: dict[tuple, torch.Tensor] = {}
    if mode == "pairwise":
        metric = resolve_interchange_metric(
            intervention_metric, score_token_ids=score_token_ids, checker=task.checker
        )
        original_outputs = (
            compute_base_outputs(test_dataset, pipeline, batch_size)
            if metric.needs_original_output
            else None
        )
        raw_scores = run_layer_scan(
            targets,
            dataset=test_dataset,
            pipeline=pipeline,
            batch_size=batch_size,
            metric=metric,
            output_scores=metric.needs_scores,
            causal_model=task.causal_model,
            original_outputs=original_outputs,
            source_pipeline=source_pipeline,
        )
    elif mode == "centroid":
        if ref_dists is None:
            raise ValueError(
                "ref_dists was not initialized for centroid mode — this should "
                "have been computed above."
            )
        result = run_centroid_layer_scan(
            targets,
            dataset=train_dataset,
            pipeline=pipeline,
            batch_size=batch_size,
            score_token_ids=score_token_ids,
            n_classes=n_classes,
            example_to_class=task.intervention_value_index,
            ref_dists=ref_dists,
            score_token_index=0,
            n_steer=n_steer,
            output_dir=out_dir,
            comparison_fn=comparison_fn,
            return_patched_dists=True,
            source_pipeline=source_pipeline,
        )
        # return_patched_dists=True branch returns a 2-tuple; pyright sees the union.
        assert isinstance(result, tuple)
        raw_scores, all_patched_dists = result
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Full (layer, pos_id) scores
    scores_per_cell: dict[tuple[int, Any], float] = dict(raw_scores)
    # Summary: best (highest) score at each layer across positions
    scores_per_layer: dict[int, float] = {}
    for (layer, _pos_id), score in scores_per_cell.items():
        if layer not in scores_per_layer or score > scores_per_layer[layer]:
            scores_per_layer[layer] = score

    # Per-(layer, position) patched-distribution heatmaps (centroid mode only)
    if ref_dists is not None and mode == "centroid" and all_patched_dists:
        from causalab.analyses.path_steering.path_visualization import (
            plot_ground_truth_heatmaps,
        )

        variable_values = task.intervention_values
        _ot = task.causal_model.output_tokens
        _var = task.intervention_variable
        if _ot and _var and _ot.get(_var):
            from causalab.methods.output_tokens import form_group_labels

            score_labels = form_group_labels(_ot[_var])
        else:
            score_labels = None

        for key, patched in all_patched_dists.items():
            layer, pos_id = key
            cell_dir = os.path.join(out_dir, f"L{layer}", f"P{pos_id}")
            os.makedirs(cell_dir, exist_ok=True)
            try:
                save_tensors_with_meta(
                    {"value": patched}, {}, cell_dir, "patched_dists"
                )
                plot_ground_truth_heatmaps(
                    dists=patched,
                    variable_values=variable_values,
                    output_dir=cell_dir,
                    score_labels=score_labels,
                    colormap=colormap or "seismic",
                    full_vocab_softmax=True,
                    title_prefix=f"Centroid patching (L{layer}, {pos_id})",
                    figure_format=figure_format,
                    filename_prefix="patched",
                )
            except Exception as e:
                logger.warning(
                    "Patched heatmap for (L%d, %s) failed: %s", layer, pos_id, e
                )

    result: dict[str, Any] = {
        "scores_per_cell": scores_per_cell,
        "scores_per_layer": scores_per_layer,
        "base_accuracy": base_accuracy,
        "token_position_ids": token_position_ids,
    }
    if prescan_block is not None:
        # Report approx and exact together wherever both were computed, so
        # the approximation quality is visible (the #456 contract).
        from causalab.methods.interchange import (
            spearman_rank_correlation,
            top_k_agreement,
        )

        approx_scores = prescan_block["approx_scores_per_cell"]
        prescan_block["exact_and_approx"] = {
            key: {"approx": approx_scores[key], "exact": exact}
            for key, exact in scores_per_cell.items()
        }
        prescan_block["agreement_at_k"] = top_k_agreement(
            approx_scores, scores_per_cell, prescan_block["top_k"], by_abs=True
        )
        prescan_block["rank_correlation"] = spearman_rank_correlation(
            approx_scores, scores_per_cell
        )
        prescan_block["abs_rank_correlation"] = spearman_rank_correlation(
            {key: abs(value) for key, value in approx_scores.items()},
            scores_per_cell,
        )
        result["prescan"] = prescan_block
    return result
