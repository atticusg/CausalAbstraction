"""DBM (Differential Binary Masking) subspace discovery — single-cell mode.

Trains masks over feature dimensions or whole units to identify which
features encode a causal variable. The ``tie_masks`` knob selects between:

- ``tie_masks=True``: one mask per unit — unit selected/deselected entirely.
- ``tie_masks=False``: one mask per feature dimension — fine-grained selection
  within each unit (equivalent to Boundless DAS).

Mirrors :func:`~causalab.analyses.subspace.grid.run_dbm_grid` for the
single-cell case.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Sequence, cast

from causalab.causal.causal_model import CausalModel
from causalab.neural.specs import SiteSpec
from causalab.neural.pipeline import LMPipeline

logger = logging.getLogger(__name__)


def find_dbm_subspace(
    sites: Sequence[Sequence[SiteSpec]],
    train_dataset: list,
    test_dataset: list,
    pipeline: LMPipeline,
    causal_model: CausalModel,
    k_features: int,
    batch_size: int = 32,
    output_dir: str = "",
    metric: Callable | None = None,
    dbm_config: dict | None = None,
    figure_format: str = "png",
) -> dict[str, Any]:
    """Train DBM masks to identify which features (or units) encode the variable.

    Args:
        sites: Nested spec groups (a grid cell value).
        train_dataset: Training counterfactual examples.
        test_dataset: Test counterfactual examples.
        pipeline: LM pipeline.
        causal_model: Causal model for intervention training.
        k_features: Number of features in the DAS subspace.
        batch_size: Training/evaluation batch size.
        output_dir: Where to save DBM artifacts.
        metric: Intervention success metric.
        dbm_config: DBM hyperparameters (``training_epoch``, ``init_lr``,
            ``masking.regularization_coefficient``, ``tie_masks``). Pulled out
            and merged into the training config; ``tie_masks`` is consumed
            here and forwarded as ``featurizer_kwargs``.
        figure_format: Format for the feature-count heatmap.

    Returns:
        Dict with key ``dbm_result`` containing:
        - ``train_score``: Training accuracy.
        - ``test_score``: Test accuracy.
        - ``n_features_by_unit``: Number of selected feature dimensions per unit.
    """
    from causalab.methods.trained_subspace.train import train_interventions
    from causalab.configs.train_config import (
        merge_with_defaults,
        PartialExperimentConfig,
    )
    from causalab.io.plots.feature_masks import plot_feature_counts

    os.makedirs(output_dir or ".", exist_ok=True)

    dbm_config = dict(dbm_config) if dbm_config else {}
    tie_masks = bool(dbm_config.pop("tie_masks", False))

    # Cast: dict literal conforms to PartialExperimentConfig TypedDict at runtime.
    mask_config = merge_with_defaults(
        cast(
            PartialExperimentConfig,
            {
                "intervention_type": "mask",
                "DAS": {"n_features": k_features},
                "featurizer_kwargs": {"tie_masks": tie_masks},
                "train_batch_size": batch_size,
                "evaluation_batch_size": batch_size,
                **dbm_config,
            },
        )
    )
    mask_config["log_dir"] = (
        os.path.join(output_dir or ".", "logs") if output_dir else "logs"
    )

    if metric is None:
        raise ValueError("`metric` is required for find_dbm_subspace")
    result = train_interventions(
        causal_model=causal_model,
        grid={("single",): sites},
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        pipeline=pipeline,
        target_variable_group=("raw_output",),
        metric=metric,
        # ExperimentConfig (TypedDict) is structurally compatible with dict[str, Any]
        config=cast(dict[str, Any], mask_config),
    )

    single_result = result["results_by_key"][("single",)]
    train_score = single_result["train_score"]
    test_score = single_result["test_score"]
    feature_indices = single_result.get("feature_indices", {})
    n_features_by_unit = {
        key: (len(indices) if indices is not None else 0)
        for key, indices in feature_indices.items()
    }

    if output_dir:
        # Plot axes come from the trained specs' structural records (never
        # from parsing the opaque spec keys); n_features from each spec's
        # trained featurizer, mirroring the legacy per-unit gate.
        from causalab.analyses.subspace._trained import trained_specs
        from causalab.io.plots.grid_cells import cells_from_site_grid

        trained = trained_specs(single_result)
        n_features_for_plot = {
            spec.key: spec.fsite.featurizer.n_features
            for spec in trained
            if spec.fsite.featurizer.n_features is not None
        }
        if n_features_for_plot:
            component_type, cells = cells_from_site_grid(
                {("single",): [trained]}, feature_indices, n_features_for_plot
            )
            heatmap_dir = os.path.join(output_dir, "heatmaps")
            os.makedirs(heatmap_dir, exist_ok=True)
            plot_feature_counts(
                component_type,
                cells,
                scores=float(test_score),
                title=f"DBM feature counts (tie_masks={tie_masks})",
                save_path=os.path.join(
                    heatmap_dir, f"raw_output_features.{figure_format}"
                ),
                figure_format=figure_format,
            )

    logger.info(
        "DBM single-cell complete (tie_masks=%s). train=%.3f, test=%.3f",
        tie_masks,
        train_score,
        test_score,
    )

    return {
        "dbm_result": {
            "train_score": train_score,
            "test_score": test_score,
            "n_features_by_unit": n_features_by_unit,
        },
    }
