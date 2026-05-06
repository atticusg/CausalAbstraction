"""Grid-mode subspace analysis: scan a (layer x token_position) grid.

Each function accepts a pre-built target dict and returns in-memory results
(scores per cell). Disk I/O (heatmaps, metadata) is handled by the caller
in main.py.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from causalab.causal.causal_model import CausalModel
from causalab.neural.pipeline import LMPipeline
from causalab.neural.units import InterchangeTarget

logger = logging.getLogger(__name__)


def run_das_grid(
    targets: dict[tuple, InterchangeTarget],
    train_dataset: list,
    test_dataset: list,
    pipeline: LMPipeline,
    causal_model: CausalModel,
    k_features: int,
    batch_size: int,
    metric: Callable,
    loss_config: dict | None = None,
    target_variable_group: tuple[str, ...] = ("raw_output",),
    log_dir: str | None = None,
) -> dict[str, Any]:
    """Train DAS across all grid cells and return per-cell scores.

    Delegates to ``train_interventions`` which handles the dict of targets
    natively, training each cell sequentially.

    Returns:
        Dict with keys:
        - ``train_scores``: Dict[(layer, pos_id) -> float]
        - ``test_scores``:  Dict[(layer, pos_id) -> float]
        - ``train_result``: raw result from ``train_interventions``
    """
    from causalab.methods.trained_subspace.train import train_interventions
    from causalab.configs.train_config import merge_with_defaults

    loss_config = loss_config or {}
    das_config = merge_with_defaults(
        {
            "intervention_type": "interchange",
            "DAS": {"n_features": k_features},
            "train_batch_size": batch_size,
            "evaluation_batch_size": batch_size,
            **loss_config,
        }
    )
    das_config["log_dir"] = log_dir or "logs"
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    result = train_interventions(
        causal_model=causal_model,
        interchange_targets=targets,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        pipeline=pipeline,
        target_variable_group=target_variable_group,
        metric=metric,
        config=das_config,
    )

    train_scores = {
        key: res["train_score"] for key, res in result["results_by_key"].items()
    }
    test_scores = {
        key: res["test_score"] for key, res in result["results_by_key"].items()
    }

    logger.info(
        "DAS grid complete. Avg train=%.3f, avg test=%.3f",
        result["avg_train_score"],
        result["avg_test_score"],
    )
    return {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "train_result": result,
    }


def run_pca_grid(
    targets: dict[tuple, InterchangeTarget],
    train_dataset: list,
    pipeline: LMPipeline,
    k_features: int,
    batch_size: int,
) -> dict[str, Any]:
    """Run PCA at each grid cell and return explained variance per cell.

    The score for each cell is the total explained variance captured by the
    top-k principal components — i.e. how much structure a k-dimensional
    linear subspace can capture at that location.

    Returns:
        Dict with keys:
        - ``scores``: Dict[(layer, pos_id) -> float]  (total variance of top-k)
        - ``svd_results_by_cell``: Dict[(layer, pos_id) -> svd_result_dict]
        - ``features_by_key``: Dict[(layer, pos_id) -> Tensor (N, k)]
          projected features for each grid position (for per-layer_x_pos
          scatter plots).
    """
    from causalab.neural.activations.collect import collect_features
    from causalab.methods.pca import compute_svd

    scores: dict[tuple, float] = {}
    svd_results_by_cell: dict[tuple, dict] = {}
    features_by_key: dict[tuple, Any] = {}

    for key, target in targets.items():
        unit = target.flatten()[0]
        raw = collect_features(
            dataset=train_dataset,
            pipeline=pipeline,
            model_units=[unit],
            batch_size=batch_size,
        )
        svd = compute_svd(raw, n_components=k_features, preprocess="center")
        var_ratios = svd[unit.id]["explained_variance_ratio"]
        scores[key] = sum(var_ratios)
        svd_results_by_cell[key] = svd[unit.id]

        rotation = svd[unit.id]["rotation"]
        features_by_key[key] = (
            raw[unit.id].detach().float() @ rotation.float()
        ).detach()

        logger.debug("PCA cell %s: top-%d variance=%.3f", key, k_features, scores[key])

    logger.info(
        "PCA grid complete. Best cell=%s (%.3f)",
        max(scores, key=scores.get),
        max(scores.values()),
    )
    return {
        "scores": scores,
        "svd_results_by_cell": svd_results_by_cell,
        "features_by_key": features_by_key,
    }


def run_dbm_grid(
    targets: dict[tuple, InterchangeTarget],
    train_dataset: list,
    test_dataset: list,
    pipeline: LMPipeline,
    causal_model: CausalModel,
    k_features: int,
    batch_size: int,
    metric: Callable,
    dbm_config: dict | None = None,
    target_variable_group: tuple[str, ...] = ("raw_output",),
    log_dir: str | None = None,
) -> dict[str, Any]:
    """Train DBM feature masks across all grid cells and return per-cell scores.

    Uses ``intervention_type="mask"`` with ``tie_masks=False`` (per-feature
    binary masks, a.k.a. Boundless DAS).

    Returns:
        Dict with keys:
        - ``train_scores``: Dict[(layer, pos_id) -> float]
        - ``test_scores``:  Dict[(layer, pos_id) -> float]
        - ``train_result``: raw result from ``train_interventions``
    """
    from causalab.methods.trained_subspace.train import train_interventions
    from causalab.configs.train_config import merge_with_defaults

    dbm_config = dbm_config or {}
    tie_masks = bool(dbm_config.pop("tie_masks", False))
    mask_config = merge_with_defaults(
        {
            "intervention_type": "mask",
            "DAS": {"n_features": k_features},
            "featurizer_kwargs": {"tie_masks": tie_masks},
            "train_batch_size": batch_size,
            "evaluation_batch_size": batch_size,
            **dbm_config,
        }
    )
    mask_config["log_dir"] = log_dir or "logs"
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    result = train_interventions(
        causal_model=causal_model,
        interchange_targets=targets,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        pipeline=pipeline,
        target_variable_group=target_variable_group,
        metric=metric,
        config=mask_config,
    )

    train_scores = {
        key: res["train_score"] for key, res in result["results_by_key"].items()
    }
    test_scores = {
        key: res["test_score"] for key, res in result["results_by_key"].items()
    }

    logger.info(
        "DBM grid complete. Avg train=%.3f, avg test=%.3f",
        result["avg_train_score"],
        result["avg_test_score"],
    )
    return {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "train_result": result,
    }


def run_boundless_grid(
    targets: dict[tuple, InterchangeTarget],
    train_dataset: list,
    test_dataset: list,
    pipeline: LMPipeline,
    causal_model: CausalModel,
    k_features: int,
    batch_size: int,
    metric: Callable,
    boundless_config: dict | None = None,
    target_variable_group: tuple[str, ...] = ("raw_output",),
    log_dir: str | None = None,
) -> dict[str, Any]:
    """Train Boundless DAS (mask intervention with tie_masks=False) across all grid cells.

    Boundless DAS learns binary masks over individual feature dimensions within each
    unit, providing fine-grained feature-level selection. This differs from standard
    DBM (tie_masks=True) which selects/deselects entire units.

    Uses ``intervention_type="mask"`` with ``tie_masks=False`` (per-feature-dimension
    masks). This is distinct from :func:`run_dbm_grid`, which also currently uses
    ``tie_masks=False`` internally but is exposed as the legacy ``dbm`` method.
    The ``boundless`` method is the explicitly-named, first-class path for this behavior.

    Args:
        targets: Dict mapping (layer, position) tuples to InterchangeTargets.
        train_dataset: Training counterfactual examples.
        test_dataset: Test counterfactual examples.
        pipeline: LM pipeline.
        causal_model: Causal model for intervention training.
        k_features: Number of features in the DAS subspace.
        batch_size: Training/evaluation batch size.
        metric: Intervention success metric.
        boundless_config: Config dict with optional keys:
            - training_epoch: Number of training epochs (default: 20).
            - init_lr: Initial learning rate (default: 0.001).
            - masking.regularization_coefficient: Sparsity weight (default: 0.1).
        target_variable_group: Variable group for training.
        log_dir: Directory for training logs.

    Returns:
        Dict with keys:
        - ``train_scores``: Dict[(layer, pos_id) -> float] training scores.
        - ``test_scores``:  Dict[(layer, pos_id) -> float] test scores.
        - ``train_result``: Raw result dict from ``train_interventions``.
    """
    from causalab.methods.trained_subspace.train import train_interventions
    from causalab.configs.train_config import merge_with_defaults

    boundless_config = boundless_config or {}
    mask_config = merge_with_defaults(
        {
            "intervention_type": "mask",
            "DAS": {"n_features": k_features},
            "featurizer_kwargs": {
                "tie_masks": False
            },  # Boundless DAS: per-dimension masks
            "train_batch_size": batch_size,
            "evaluation_batch_size": batch_size,
            **boundless_config,
        }
    )
    mask_config["log_dir"] = log_dir or "logs"
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    result = train_interventions(
        causal_model=causal_model,
        interchange_targets=targets,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        pipeline=pipeline,
        target_variable_group=target_variable_group,
        metric=metric,
        config=mask_config,
    )

    train_scores = {
        key: res["train_score"] for key, res in result["results_by_key"].items()
    }
    test_scores = {
        key: res["test_score"] for key, res in result["results_by_key"].items()
    }

    logger.info(
        "Boundless DAS grid complete. Avg train=%.3f, avg test=%.3f",
        result["avg_train_score"],
        result["avg_test_score"],
    )
    return {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "train_result": result,
    }
