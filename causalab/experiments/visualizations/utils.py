"""
Utility functions for visualizations.
"""

import os
from typing import Any, Optional

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
