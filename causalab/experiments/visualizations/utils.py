"""
Utility functions for visualizations.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def create_heatmap(
    score_matrix: np.ndarray[Any, np.dtype[Any]],
    x_labels: list[str],
    y_labels: list[str],
    title: str = "Heatmap",
    xlabel: str = "",
    ylabel: str = "",
    save_path: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple[float, float] = (10, 8),
    **kwargs: Any,
) -> None:
    """
    Create a heatmap visualization.

    Args:
        score_matrix: 2D numpy array of scores
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        title: Title of the heatmap
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        save_path: Path to save the figure
        cmap: Colormap to use
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        figsize: Figure size as (width, height)
        **kwargs: Additional arguments passed to imshow
    """
    _fig, ax = plt.subplots(figsize=figsize)

    # Create the heatmap
    im = ax.imshow(
        score_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", **kwargs
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Score", rotation=270, labelpad=15)

    # Add text annotations
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            if not np.isnan(score_matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{score_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.show()
    plt.close()


def create_binary_mask_heatmap(
    mask_matrix: np.ndarray[Any, np.dtype[Any]],
    x_labels: list[str],
    y_labels: list[str],
    title: str = "Binary Mask Heatmap",
    xlabel: str = "",
    ylabel: str = "",
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (12, 8),
    **kwargs: Any,
) -> None:
    """
    Create a binary mask heatmap visualization for attention heads.

    Args:
        mask_matrix: 2D numpy array of binary values (0 or 1)
        x_labels: Labels for x-axis (head indices)
        y_labels: Labels for y-axis (layer indices)
        title: Title of the heatmap
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        save_path: Path to save the figure
        figsize: Figure size as (width, height)
        **kwargs: Additional arguments passed to imshow
    """
    _fig, ax = plt.subplots(figsize=figsize)

    # Create the binary heatmap
    im = ax.imshow(mask_matrix, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto", **kwargs)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_label("Selected", rotation=270, labelpad=15)
    cbar.ax.set_yticklabels(["Not Selected", "Selected"])

    # Add grid for better visibility
    ax.set_xticks(np.arange(len(x_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(y_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


# =============================================================================
# Feature Count Heatmap (shared by feature_masks.py plotting functions)
# =============================================================================


def create_feature_count_heatmap(
    count_matrix: np.ndarray[Any, np.dtype[Any]],
    x_labels: List[str],
    y_labels: List[str],
    scores: Union[float, Dict[int, float]],
    layers: List[int],
    title: str = "Feature Counts",
    xlabel: str = "",
    ylabel: str = "Layer",
    score_label: str = "Acc",
    colorbar_label: str = "Feature Count",
    save_path: Optional[str] = None,
    flip_vertical: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    show_accuracy_column: bool = True,
) -> None:
    """
    Create a feature count heatmap with optional accuracy sidebar.

    This is the shared rendering function for all feature count visualizations
    (attention heads, residual stream, MLPs).

    Args:
        count_matrix: 2D numpy array of feature counts (rows=layers, cols=x-axis units).
        x_labels: Labels for x-axis (heads, token positions, etc.).
        y_labels: Labels for y-axis (layers).
        scores: Either:
            - float for single overall accuracy
            - Dict[int, float] for per-layer accuracies
        layers: List of layer indices (needed for per-layer score mapping).
        title: Title of the heatmap.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        score_label: Label for accuracy column (default: "Acc").
        colorbar_label: Label for the colorbar (default: "Feature Count").
        save_path: Path to save the figure.
        flip_vertical: Whether to flip the matrix vertically (for residual stream/MLP).
        figsize: Figure size as (width, height). Auto-calculated if None.
        show_accuracy_column: Whether to show accuracy sidebar (default: True).
    """
    has_per_layer_scores = isinstance(scores, dict)
    num_layers = len(y_labels)
    num_cols = len(x_labels)

    # Apply vertical flip if requested
    if flip_vertical:
        count_matrix = np.flipud(count_matrix)
        y_labels = list(reversed(y_labels))
        layers_for_scores = list(reversed(layers))
    else:
        layers_for_scores = layers

    # Build accuracy column
    if has_per_layer_scores:
        accuracy_col = np.zeros((num_layers, 1))
        for idx, layer in enumerate(layers_for_scores):
            if layer in scores:
                accuracy_col[idx, 0] = scores[layer]
    else:
        accuracy_col = np.full((num_layers, 1), scores)

    # Calculate figure size if not provided
    if figsize is None:
        figsize = (max(10, num_cols * 0.6 + 2), max(6, num_layers * 0.4))

    # Create figure
    if show_accuracy_column:
        fig, (ax_acc, ax_main) = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={
                "width_ratios": [1, max(6, num_cols * 0.5)],
                "wspace": 0.05,
            },
        )
        _render_accuracy_column(
            ax_acc,
            accuracy_col,
            y_labels,
            num_layers,
            has_per_layer_scores,
            scores,
            score_label,
            ylabel,
        )
        show_y_labels_on_main = False
    else:
        fig, ax_main = plt.subplots(figsize=figsize)
        show_y_labels_on_main = True

    # Plot main heatmap
    im_main = ax_main.imshow(count_matrix, cmap="viridis", aspect="auto")

    # Set ticks and labels
    ax_main.set_xticks(np.arange(num_cols))
    ax_main.set_yticks(np.arange(num_layers))
    ax_main.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

    if show_y_labels_on_main:
        ax_main.set_yticklabels(y_labels, fontsize=9)
        ax_main.set_ylabel(ylabel, fontsize=11)
    else:
        ax_main.set_yticklabels([])

    ax_main.set_xlabel(xlabel, fontsize=11)

    # Add feature count text in each cell
    _add_cell_annotations(ax_main, count_matrix, num_layers, num_cols)

    # Add grid
    ax_main.set_xticks(np.arange(num_cols + 1) - 0.5, minor=True)
    ax_main.set_yticks(np.arange(num_layers + 1) - 0.5, minor=True)
    ax_main.grid(which="minor", color="gray", linestyle="-", linewidth=0.3)
    ax_main.tick_params(which="minor", size=0)

    # Add colorbar
    cbar = plt.colorbar(im_main, ax=ax_main)
    cbar.set_label(colorbar_label, rotation=270, labelpad=15)

    # Set title
    if show_accuracy_column:
        fig.suptitle(title, fontsize=12)
    else:
        if has_per_layer_scores:
            ax_main.set_title(title, fontsize=12, pad=10)
        else:
            ax_main.set_title(
                f"{title}\nOverall Accuracy: {scores:.3f}", fontsize=12, pad=10
            )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def _render_accuracy_column(
    ax: Any,
    accuracy_col: np.ndarray[Any, np.dtype[Any]],
    y_labels: List[str],
    num_layers: int,
    has_per_layer_scores: bool,
    scores: Union[float, Dict[int, float]],
    score_label: str,
    ylabel: str,
) -> None:
    """Render the accuracy sidebar column."""
    ax.imshow(accuracy_col, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks([0])
    ax.set_xticklabels([score_label], fontsize=10)
    ax.set_yticks(np.arange(num_layers))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)

    # Add accuracy text in cells
    if has_per_layer_scores:
        for i in range(num_layers):
            acc = accuracy_col[i, 0]
            text_color = "white" if acc < 0.5 else "black"
            ax.text(
                0,
                i,
                f"{acc:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )
    else:
        # Single score: show in middle row
        mid_row = num_layers // 2
        acc = float(scores) if not isinstance(scores, dict) else 0.0
        text_color = "white" if acc < 0.5 else "black"
        ax.text(
            0,
            mid_row,
            f"{acc:.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=11,
            fontweight="bold",
        )


def _add_cell_annotations(
    ax: Any,
    count_matrix: np.ndarray[Any, np.dtype[Any]],
    num_rows: int,
    num_cols: int,
) -> None:
    """Add feature count annotations to each cell with adaptive text color."""
    vmin, vmax = count_matrix.min(), count_matrix.max()
    for i in range(num_rows):
        for j in range(num_cols):
            count = int(count_matrix[i, j])
            normalized = (count - vmin) / (vmax - vmin + 1e-8)
            text_color = "white" if normalized < 0.5 else "black"
            ax.text(
                j,
                i,
                str(count),
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
                fontweight="bold",
            )
