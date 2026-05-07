"""Interchange score grid scan for the locate analysis."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

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
    score_token_ids: list[int],
    n_classes: int,
    batch_size: int,
    n_steer: int,
    out_dir: str,
    position_names: list[str] | None = None,
    comparison_fn: Callable | None = None,
    experiment_root: str | None = None,
    colormap: str | None = None,
    figure_format: str = "pdf",
    source_pipeline=None,
) -> dict[str, Any]:
    """Run interchange score scan over a (layer × token_position) grid.

    Args:
        source_pipeline: If provided, activations are collected from this
            pipeline and patched into ``pipeline`` (cross-model patching).
            ``None`` (default) uses standard single-model patching.
            Cross-model + pairwise mode is not supported and raises
            ``ValueError``.

    Returns:
        Dict with ``scores_per_cell`` (keys ``(layer, pos_id)``), summary
        ``scores_per_layer`` (min over positions per layer), ``base_accuracy``,
        and ``token_position_ids``.
    """
    if source_pipeline is not None and mode == "pairwise":
        raise ValueError(
            "Cross-model patching (source_pipeline != None) is not supported "
            "with mode='pairwise'. Use mode='centroid' instead."
        )
    from causalab.methods.interchange import (
        run_centroid_layer_scan,
        run_pairwise_layer_scan,
    )
    from causalab.methods.metric import (
        compute_base_accuracy,
        compute_reference_distributions,
    )
    from causalab.runner.helpers import build_targets_for_grid

    if score_token_ids is None:
        raise ValueError("Task must provide score_token_ids for interchange locate")

    base_accuracy = float("nan")
    ref_dists_fvs = None
    if experiment_root is not None:
        base_accuracy, ref_dists_fvs = _load_baseline_artifacts(experiment_root)

    if base_accuracy != base_accuracy:  # isnan
        base_acc = compute_base_accuracy(
            dataset=test_dataset,
            pipeline=pipeline,
            batch_size=batch_size,
        )
        base_accuracy = base_acc["accuracy"]

    targets, token_positions = build_targets_for_grid(
        pipeline,
        task,
        layers,
        position_names=position_names,
    )
    token_position_ids = [tp.id for tp in token_positions]

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
        raw_scores = run_pairwise_layer_scan(
            interchange_targets=targets,
            dataset=test_dataset,
            pipeline=pipeline,
            batch_size=batch_size,
            score_token_ids=score_token_ids,
            score_token_index=0,
            output_dir=out_dir,
            comparison_fn=comparison_fn,
        )
    elif mode == "centroid":
        result = run_centroid_layer_scan(
            interchange_targets=targets,
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
        score_labels = (
            [str(v) for v in task.output_token_values]
            if task.output_token_values
            else None
        )

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

    return {
        "scores_per_cell": scores_per_cell,
        "scores_per_layer": scores_per_layer,
        "base_accuracy": base_accuracy,
        "token_position_ids": token_position_ids,
    }
