"""
Score-based heatmap visualization functions.

These functions display numeric scores (e.g., accuracy) in heatmap cells.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .utils import create_heatmap
from causalab.neural.activations.targets import (
    detect_component_type_from_targets,
    extract_grid_dimensions_from_targets,
)
from causalab.neural.units import InterchangeTarget


def plot_attention_head_heatmap(
    scores: Dict[Tuple[int, int], float],
    layers: List[int],
    heads: List[int],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figure_format: str = "pdf",
) -> None:
    """
    Plot heatmap of attention head intervention scores.

    Args:
        scores: Dict mapping (layer, head) tuples to scores (0.0 to 1.0).
        layers: List of layer indices (y-axis).
        heads: List of head indices (x-axis).
        title: Optional custom title.
        save_path: Optional path to save figure.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    # Build score matrix with NaN for missing values
    score_matrix = np.full((len(heads), len(layers)), np.nan)

    for (layer, head), score in scores.items():
        if layer in layers and head in heads:
            layer_idx = layers.index(layer)
            head_idx = heads.index(head)
            score_matrix[head_idx, layer_idx] = score

    # Transpose so layers are on y-axis
    score_matrix = score_matrix.T

    # Format labels with prefixes
    x_labels = [f"H{head}" for head in heads]
    y_labels = [f"L{layer}" for layer in layers]

    # Auto-generate title if not provided
    if title is None:
        title = "Attention Head Intervention Accuracy"

    # Create the heatmap using existing visualization function
    create_heatmap(
        score_matrix=score_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        save_path=save_path,
        x_label="Head",
        y_label="Layer",
        use_custom_bounds=False,  # Use 0-1 bounds for accuracy
        cbar_label="Accuracy (%)",
        figsize=(max(12, len(heads) * 0.6), max(6, len(layers) * 0.8)),
        figure_format=figure_format,
    )


def plot_residual_stream_heatmap(
    scores: Dict[Tuple[int, Any], float],
    layers: List[int],
    token_position_ids: List[Any],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figure_format: str = "pdf",
) -> None:
    """
    Plot heatmap of residual stream intervention scores.

    Args:
        scores: Dict mapping (layer, position_id) tuples to scores (0.0 to 1.0).
        layers: List of layer indices (y-axis). Can include -1 for embeddings.
        token_position_ids: List of position IDs (x-axis).
        title: Optional custom title.
        save_path: Optional path to save figure.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    # Build score matrix with NaN for missing values
    # Rows = layers (reversed), Columns = positions
    score_matrix = np.full((len(layers), len(token_position_ids)), np.nan)

    for (layer, pos_id), score in scores.items():
        if layer in layers and pos_id in token_position_ids:
            layer_idx = layers.index(layer)
            pos_idx = token_position_ids.index(pos_id)
            score_matrix[layer_idx, pos_idx] = score

    # Format labels
    y_labels = [f"L{layer}" if layer >= 0 else "Embed" for layer in layers]
    x_labels = [str(pos_id) for pos_id in token_position_ids]

    # Auto-generate title if not provided
    if title is None:
        title = "Residual Stream Intervention Accuracy"

    # Flip the score matrix vertically so layer -1 is at bottom
    score_matrix = np.flipud(score_matrix)
    y_labels = list(reversed(y_labels))

    # Create the heatmap
    create_heatmap(
        score_matrix=score_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        save_path=save_path,
        xlabel="Position",
        ylabel="Layer",
        figsize=(max(10, len(token_position_ids) * 1.5), max(6, len(layers) * 0.5)),
        figure_format=figure_format,
    )


def plot_variable_localization_heatmap(
    scores_by_variable: Dict[str, Dict[str, float]],
    title: str = "Variable Localization Heatmap",
    save_path: Optional[str] = None,
    figure_format: str = "pdf",
) -> None:
    """
    Plot a layer x variable heatmap from locate results.

    Args:
        scores_by_variable: Dict mapping variable name to
            {"layer_str": score, ...} (scores_per_layer from results.json).
        title: Plot title.
        save_path: Path to save figure.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    # Collect all layers across all variables, sorted numerically
    all_layers = sorted(
        {int(k) for d in scores_by_variable.values() for k in d},
    )
    variables = list(scores_by_variable.keys())

    # Build matrix: rows=layers, cols=variables
    score_matrix = np.full((len(all_layers), len(variables)), np.nan)
    for j, var in enumerate(variables):
        for layer_str, score in scores_by_variable[var].items():
            i = all_layers.index(int(layer_str))
            score_matrix[i, j] = score

    x_labels = [v.replace("_", " ") for v in variables]
    y_labels = [f"L{layer}" for layer in all_layers]

    create_heatmap(
        score_matrix=score_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        xlabel="Causal Variable",
        ylabel="Layer",
        save_path=save_path,
        figsize=(max(6, len(variables) * 2.5), max(6, len(all_layers) * 0.6)),
        figure_format=figure_format,
    )


def plot_score_heatmap(
    scores: Dict[Tuple[Any, ...], float],
    interchange_targets: Dict[tuple[Any, ...], InterchangeTarget],
    title: str,
    save_path: str,
    figure_format: str = "pdf",
) -> None:
    """
    Plot score heatmap, auto-detecting component type from targets.

    This unified dispatcher automatically detects whether the targets represent
    attention heads, residual stream positions, or MLPs, and calls the appropriate
    plotting function.

    Args:
        scores: Dict mapping (layer, x) tuples to scores (0.0 to 1.0).
                For attention heads: (layer, head) -> score
                For residual stream/MLP: (layer, token_position_id) -> score
        interchange_targets: Dict mapping keys to InterchangeTarget objects.
                            Used to detect component type and extract grid dimensions.
        title: Plot title.
        save_path: Path to save the figure.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    # Detect component type
    component_type = detect_component_type_from_targets(interchange_targets)

    # Extract grid dimensions
    grid_dims = extract_grid_dimensions_from_targets(
        component_type, interchange_targets
    )

    # Dispatch to appropriate plotting function
    if component_type == "attention_head":
        plot_attention_head_heatmap(
            scores=scores,
            layers=grid_dims["layers"],
            heads=grid_dims["heads"],
            title=title,
            save_path=save_path,
            figure_format=figure_format,
        )
    else:
        # residual_stream and mlp both use (layer, token_position) grid
        plot_residual_stream_heatmap(
            scores=scores,
            layers=grid_dims["layers"],
            token_position_ids=grid_dims["token_position_ids"],
            title=title,
            save_path=save_path,
            figure_format=figure_format,
        )
