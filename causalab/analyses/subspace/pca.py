"""PCA subspace discovery.

Finds a k-dimensional PCA subspace, projects features, and optionally
generates the features_3d visualization.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch
from safetensors.torch import save_file

from causalab.analyses.subspace._visualization import save_features_visualization
from causalab.methods.trained_subspace.subspace import build_SVD_featurizers
from causalab.neural.units import InterchangeTarget
from causalab.neural.pipeline import LMPipeline
from causalab.neural.activations.collect import collect_features
from causalab.methods.pca import compute_svd

logger = logging.getLogger(__name__)


def find_pca_subspace(
    target: InterchangeTarget,
    train_dataset: list,
    pipeline: LMPipeline,
    k_features: int,
    batch_size: int = 32,
    output_dir: str = "",
    intervention_variable: str | None = None,
    embeddings: dict[str, Callable] | None = None,
    colormap: str | None = None,
    vis_dims: list[int] | None = None,
    variable_values: list[str] | None = None,
    detailed_hover: bool = False,
    max_hover_chars: int = 50,
    figure_format: str = "pdf",
) -> dict[str, Any]:
    """Find a k-dimensional PCA subspace and project features through it.

    Collects raw activations, computes SVD, sets the PCA featurizer on the
    target, and projects the raw features through the rotation matrix (single
    forward pass).

    Args:
        target: Interchange target (must contain a single unit).
        train_dataset: Training counterfactual examples.
        pipeline: LM pipeline for collecting activations.
        k_features: Number of principal components.
        batch_size: Batch size for feature collection.
        output_dir: Where to save rotation + features artifacts.
        intervention_variable: Variable name for feature visualization coloring.
        embeddings: Embedding functions for visualization.

    Returns:
        Dict with keys:
        - ``rotation``: (d_model, k) rotation matrix
        - ``explained_variance_ratio``: list of per-component ratios
        - ``features``: (N, k) projected features tensor
    """
    unit = target.flatten()[0]

    logger.info("Computing PCA with k=%d...", k_features)
    raw_features_dict = collect_features(
        dataset=train_dataset,
        pipeline=pipeline,
        model_units=[unit],
        batch_size=batch_size,
    )

    svd_results = compute_svd(
        raw_features_dict,
        n_components=k_features,
        preprocess="center",
    )
    build_SVD_featurizers(
        [unit],
        svd_results,
        trainable=False,
        featurizer_id="PCA",
    )

    raw_features = raw_features_dict[unit.id].detach()
    rotation = svd_results[unit.id]["rotation"]  # (d_model, k)
    features = (raw_features.float() @ rotation.float()).detach()

    var_ratios = svd_results[unit.id]["explained_variance_ratio"]
    logger.info(
        "PCA variance explained: %s, total: %.1f%%",
        [f"{v:.1%}" for v in var_ratios],
        sum(var_ratios) * 100,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        from causalab.io.counterfactuals import save_counterfactual_examples

        save_counterfactual_examples(
            train_dataset,
            os.path.join(output_dir, "train_dataset.json"),
        )

        save_file(
            {
                "rotation_matrix": rotation.contiguous(),
                "explained_variance_ratio": torch.tensor(
                    var_ratios, dtype=torch.float32
                ),
            },
            os.path.join(output_dir, "rotation.safetensors"),
        )

        features_dir = os.path.join(output_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        save_file(
            {"features": features.contiguous()},
            os.path.join(features_dir, "training_features.safetensors"),
        )
        save_file(
            {"features": raw_features.contiguous()},
            os.path.join(features_dir, "raw_features.safetensors"),
        )

        vis_features = features[:, vis_dims] if vis_dims is not None else features
        save_features_visualization(
            vis_features,
            train_dataset,
            output_dir,
            intervention_variable,
            embeddings,
            colormap=colormap,
            variable_values=variable_values,
            detailed_hover=detailed_hover,
            max_hover_chars=max_hover_chars,
            figure_format=figure_format,
            explained_variance_ratio=var_ratios,
        )

    return {
        "rotation": rotation,
        "explained_variance_ratio": var_ratios,
        "features": features,
    }
