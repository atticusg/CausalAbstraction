"""DBM binary masking grid scan for the locate analysis.

Trains DBM with ``tie_masks=True`` over a (layer × token_position) grid in a
single batched call to ``train_interventions``, then plots the resulting
selected-cells heatmap. Mirrors the analysis-layer pattern used by
:func:`~causalab.analyses.subspace.grid.run_dbm_grid`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


def run_dbm_binary_scan(
    pipeline,
    task,
    train_dataset: list,
    test_dataset: list,
    layers: list[int],
    batch_size: int,
    out_dir: str,
    metric: Callable,
    position_names: list[str] | None = None,
    dbm_cfg: Mapping[str, Any] | None = None,
    figure_format: str = "pdf",
) -> dict[str, Any]:
    """Run DBM binary masking over a (layer × token_position) grid.

    All cells are trained in a single ``train_interventions`` call by passing
    the per-cell targets as a ``Dict[(layer, pos_id), InterchangeTarget]``.

    Args:
        pipeline: LM pipeline.
        task: Task object (provides ``causal_model`` and token-position lookup).
        train_dataset: In-memory training counterfactual examples.
        test_dataset: In-memory test counterfactual examples.
        layers: Layers to scan.
        batch_size: Training/evaluation batch size.
        out_dir: Output directory for artifacts.
        metric: Intervention success metric (string match).
        position_names: Restrict to these positions; ``None`` uses all task positions.
        dbm_cfg: DBM hyperparameters with keys ``training_epoch``, ``lr``,
            ``regularization_coefficient``. Defaults match the legacy locate
            hardcodes (20, 0.001, 100).
        figure_format: Format for the binary-mask heatmap.

    Returns:
        Dict with:
        - ``scores_per_cell``: ``{(layer, pos_id): test_score}``
        - ``scores_per_layer``: ``{layer: min_score_over_positions}``
        - ``token_position_ids``: ordered list of position IDs in the grid.
    """
    from causalab.methods.trained_subspace.train import train_interventions
    from causalab.configs.train_config import merge_with_defaults
    from causalab.runner.helpers import build_targets_for_grid
    from causalab.io.plots.binary_mask import plot_binary_mask

    cfg = dict(dbm_cfg) if dbm_cfg else {}
    training_epoch = int(cfg.get("training_epoch", 20))
    init_lr = float(cfg.get("lr", 0.001))
    regularization_coefficient = float(cfg.get("regularization_coefficient", 100))

    os.makedirs(out_dir, exist_ok=True)

    targets, token_positions = build_targets_for_grid(
        pipeline,
        task,
        layers,
        position_names=position_names,
    )
    token_position_ids = [tp.id for tp in token_positions]

    mask_config = merge_with_defaults(
        {
            "intervention_type": "mask",
            "featurizer_kwargs": {"tie_masks": True},
            "train_batch_size": batch_size,
            "evaluation_batch_size": batch_size,
            "training_epoch": training_epoch,
            "init_lr": init_lr,
            "masking": {"regularization_coefficient": regularization_coefficient},
        }
    )
    mask_config["log_dir"] = os.path.join(out_dir, "logs")
    os.makedirs(mask_config["log_dir"], exist_ok=True)

    result = train_interventions(
        causal_model=task.causal_model,
        interchange_targets=targets,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        pipeline=pipeline,
        target_variable_group=("raw_output",),
        metric=metric,
        config=mask_config,
    )

    scores_per_cell: dict[tuple[int, Any], float] = {
        key: res["test_score"] for key, res in result["results_by_key"].items()
    }
    scores_per_layer: dict[int, float] = {}
    for (layer, _pos_id), score in scores_per_cell.items():
        if layer not in scores_per_layer or score < scores_per_layer[layer]:
            scores_per_layer[layer] = score

    aggregated_feature_indices: dict[str, Any] = {}
    for res in result["results_by_key"].values():
        aggregated_feature_indices.update(res.get("feature_indices", {}))
    if aggregated_feature_indices:
        heatmap_dir = os.path.join(out_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        plot_binary_mask(
            feature_indices=aggregated_feature_indices,
            title="DBM Selected Units (binary)",
            save_path=os.path.join(heatmap_dir, f"raw_output_mask.{figure_format}"),
            figure_format=figure_format,
        )

    logger.info(
        "DBM binary locate scan complete. cells=%d, avg test=%.3f",
        len(scores_per_cell),
        result["avg_test_score"],
    )

    return {
        "scores_per_cell": scores_per_cell,
        "scores_per_layer": scores_per_layer,
        "token_position_ids": token_position_ids,
    }
