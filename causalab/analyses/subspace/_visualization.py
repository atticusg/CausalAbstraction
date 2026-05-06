"""Shared visualization for subspace discovery methods."""

from __future__ import annotations

import logging
import os
from typing import Callable

from torch import Tensor

from causalab.io.plots.figure_format import path_with_figure_format

logger = logging.getLogger(__name__)


def save_features_visualization(
    features: Tensor,
    train_dataset: list,
    output_dir: str,
    intervention_variable: str | None,
    embeddings: dict[str, Callable] | None,
    colormap: str | None = None,
    variable_values: list[str] | None = None,
    detailed_hover: bool = False,
    max_hover_chars: int = 50,
    figure_format: str = "pdf",
    skip_3d: bool = False,
    explained_variance_ratio: list[float] | None = None,
) -> None:
    """Save feature scatter plots and optional variance histogram."""
    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    # 3D interactive (HTML)
    if not skip_3d:
        try:
            from causalab.io.plots.plot_3d_interactive import plot_3d

            plot_3d(
                features=features,
                output_path=os.path.join(vis_dir, "features_3d.html"),
                train_dataset=train_dataset,
                intervention_variable=intervention_variable,
                embeddings=embeddings,
                colormap=colormap,
                variable_values=variable_values,
                detailed_hover=detailed_hover,
                max_hover_chars=max_hover_chars,
            )
            logger.info("Saved features_3d.html to %s", vis_dir)
        except Exception as e:
            logger.warning("3D feature visualization failed: %s", e, exc_info=True)

    # 2D static (PNG/PDF)
    try:
        from causalab.io.plots.pca_scatter import plot_features_2d

        plot_features_2d(
            features=features,
            output_path=path_with_figure_format(
                os.path.join(vis_dir, "features_2d.pdf"),
                figure_format,
            ),
            train_dataset=train_dataset,
            intervention_variable=intervention_variable,
            embeddings=embeddings,
            colormap=colormap,
            variable_values=variable_values,
            figure_format=figure_format,
        )
    except Exception as e:
        logger.warning("2D feature visualization failed: %s", e, exc_info=True)

    # Variance histogram (PCA only)
    if explained_variance_ratio is not None:
        try:
            from causalab.io.plots.pca_scatter import plot_variance_histogram

            plot_variance_histogram(
                explained_variance_ratio,
                save_path=path_with_figure_format(
                    os.path.join(vis_dir, "features_variance.png"),
                    figure_format,
                ),
                figure_format=figure_format,
            )
        except Exception as e:
            logger.warning("Variance histogram failed: %s", e, exc_info=True)

    # 3D static matplotlib scatter
    try:
        from causalab.io.plots.pca_scatter import plot_features_3d_static

        plot_features_3d_static(
            features=features,
            output_path=path_with_figure_format(
                os.path.join(vis_dir, "features_3d.png"),
                figure_format,
            ),
            train_dataset=train_dataset,
            intervention_variable=intervention_variable,
            embeddings=embeddings,
            colormap=colormap,
            variable_values=variable_values,
            figure_format=figure_format,
        )
    except Exception as e:
        logger.warning("Static 3D visualization failed: %s", e, exc_info=True)
