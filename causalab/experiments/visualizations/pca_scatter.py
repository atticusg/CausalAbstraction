"""
PCA scatter plot visualization functions.

This module provides functions to visualize PCA-reduced features as scatter plots,
showing how samples cluster in principal component space. Supports both 2D and 3D
plotting, with optional dual-label encoding using color and marker shape.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from torch import Tensor

# Type alias for numpy arrays
NDArray = np.ndarray[Any, np.dtype[Any]]

# Type for RGBA color tuples (from matplotlib colormaps)
ColorTuple = Tuple[float, float, float, float]

# Distinguishable markers for shape_labels
MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h"]


def _get_color_map(unique_labels: NDArray) -> Dict[Any, ColorTuple]:
    """Create a mapping from labels to colors."""
    n_labels = len(unique_labels)
    if n_labels <= 10:
        cmap = plt.colormaps["tab10"]
    elif n_labels <= 20:
        cmap = plt.colormaps["tab20"]
    else:
        cmap = plt.colormaps["hsv"]

    return {
        label: cmap(i / max(n_labels - 1, 1)) for i, label in enumerate(unique_labels)
    }


def _get_marker_map(unique_shape_labels: NDArray) -> Dict[Any, str]:
    """Create a mapping from shape labels to markers."""
    if len(unique_shape_labels) > len(MARKERS):
        raise ValueError(
            f"Too many unique shape labels ({len(unique_shape_labels)}). "
            f"Maximum supported is {len(MARKERS)}."
        )
    return {label: MARKERS[i] for i, label in enumerate(unique_shape_labels)}


def _scatter_by_groups(
    ax: Axes,
    df: pd.DataFrame,
    pc_columns: List[str],
    color_map: Dict[Any, ColorTuple],
    marker_map: Optional[Dict[Any, str]],
) -> None:
    """
    Plot scatter points grouped by color_label (and optionally shape_label).

    Works for both 2D and 3D axes - matplotlib's scatter accepts variable
    number of coordinate arguments via unpacking.
    """
    if "shape_label" in df.columns and marker_map is not None:
        # Group by both color and shape
        for key, group in df.groupby(["color_label", "shape_label"]):
            if group.empty:
                continue
            color_label, shape_label = key  # type: ignore[misc]
            coords = [group[pc].to_numpy() for pc in pc_columns]
            ax.scatter(
                *coords,
                c=[color_map[color_label]],
                marker=marker_map[shape_label],
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidths=0.5,
            )
    else:
        # Group by color only
        for color_label, group in df.groupby("color_label"):
            coords = [group[pc].to_numpy() for pc in pc_columns]
            ax.scatter(
                *coords,
                c=[color_map[color_label]],
                label=str(color_label),
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidths=0.5,
            )


def _build_legend_handles(
    unique_colors: NDArray,
    color_map: Dict[Any, ColorTuple],
    unique_shapes: Optional[NDArray],
    marker_map: Optional[Dict[Any, str]],
) -> Tuple[List[Line2D], List[str]]:
    """Build legend handles and labels for color and optionally shape encoding."""
    handles: List[Line2D] = []
    labels: List[str] = []

    # Color legend
    for label in unique_colors:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[label],
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
        )
        labels.append(str(label))

    # Shape legend (if applicable)
    if unique_shapes is not None and marker_map is not None:
        # Separator
        handles.append(Line2D([0], [0], color="w", marker="", linestyle=""))
        labels.append("")

        for shape_label in unique_shapes:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[shape_label],
                    color="w",
                    markerfacecolor="gray",
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            )
            labels.append(str(shape_label))

    return handles, labels


def plot_pca_scatter(
    features: Tensor,
    svd_result: Dict[str, Any],
    labels: Sequence[Any],
    component_tuples: List[Tuple[int, ...]],
    shape_labels: Optional[Sequence[Any]] = None,
    title: Optional[str] = None,
    save_dir: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> None:
    """
    Create scatter plots of samples in PCA space.

    Projects features onto principal components and creates scatter plots for
    specified component tuples, with points colored by labels and optionally
    shaped by a second label set.

    Args:
        features: Feature tensor of shape (n_points, n_features).
        svd_result: Dictionary containing SVD/PCA results with keys:
                   - "rotation": Rotation matrix of shape (n_features, n_components)
                   - "explained_variance_ratio": Variance explained by each component
                   - "mean": Mean used for normalization (None if not normalized)
                   - "std": Standard deviation used for normalization (None if not normalized)
        labels: Sequence of labels for coloring points. If len(labels) < len(features)
                and len(features) is evenly divisible by len(labels), labels are
                automatically replicated. This supports stack mode where features
                from multiple units are stacked.
        component_tuples: List of component tuples specifying which components to plot.
                         2-tuples like (0, 1) create 2D plots.
                         3-tuples like (0, 1, 2) create 3D plots.
        shape_labels: Optional second label sequence for marker shapes. Same
                     auto-replication rules as labels. When provided, labels control
                     color and shape_labels control marker shape.
        title: Optional custom title for plots. If None, auto-generated.
        save_dir: Optional directory to save figures. If provided, saves each plot
                 as "{save_dir}/pc{x}_vs_pc{y}.png" or "{save_dir}/pc{x}_vs_pc{y}_vs_pc{z}.png".
        figsize: Figure size as (width, height) tuple.

    Raises:
        ValueError: If component_tuples contains tuples with length != 2 or 3.
        ValueError: If component indices exceed available components.
        ValueError: If shape_labels has more unique values than available markers (10).
    """
    # Validate component_tuples
    for ct in component_tuples:
        if len(ct) not in (2, 3):
            raise ValueError(
                f"component_tuples must have length 2 or 3, got {len(ct)}: {ct}"
            )

    # Auto-replicate labels if needed (for stack mode)
    n_points = len(features)
    n_labels = len(labels)
    labels_list: List[Any]
    if n_labels != n_points:
        if n_points % n_labels == 0:
            n_repeats = n_points // n_labels
            labels_list = list(labels) * n_repeats
        else:
            raise ValueError(
                f"Labels length ({n_labels}) must equal features length ({n_points}) "
                f"or evenly divide it for auto-replication"
            )
    else:
        labels_list = list(labels)

    # Auto-replicate shape_labels if provided
    shape_labels_list: Optional[List[Any]] = None
    if shape_labels is not None:
        n_shape_labels = len(shape_labels)
        if n_shape_labels != n_points:
            if n_points % n_shape_labels == 0:
                n_repeats = n_points // n_shape_labels
                shape_labels_list = list(shape_labels) * n_repeats
            else:
                raise ValueError(
                    f"shape_labels length ({n_shape_labels}) must equal features length "
                    f"({n_points}) or evenly divide it for auto-replication"
                )
        else:
            shape_labels_list = list(shape_labels)

    # Extract SVD components
    rotation = svd_result["rotation"]
    explained_variance_ratio = svd_result["explained_variance_ratio"]
    mean = svd_result.get("mean")
    std = svd_result.get("std")

    # Normalize features if mean/std were used during SVD
    if mean is not None and std is not None:
        if not isinstance(mean, Tensor):
            mean = torch.tensor(mean, dtype=features.dtype, device=features.device)
        if not isinstance(std, Tensor):
            std = torch.tensor(std, dtype=features.dtype, device=features.device)

        std_safe = std.clone()
        std_safe[std_safe == 0] = 1.0
        features_normalized = (features - mean) / std_safe
    else:
        features_normalized = features

    # Project features onto principal components
    projected = features_normalized @ rotation
    projected_np = projected.detach().cpu().numpy()

    # Validate component indices
    n_components = projected_np.shape[1]
    for ct in component_tuples:
        for idx in ct:
            if idx >= n_components:
                raise ValueError(
                    f"Component index {idx} exceeds available components ({n_components})"
                )

    # Build DataFrame with PC projections and labels
    df_data: Dict[str, Any] = {
        f"PC{i}": projected_np[:, i] for i in range(n_components)
    }
    df_data["color_label"] = labels_list
    if shape_labels_list is not None:
        df_data["shape_label"] = shape_labels_list
    df = pd.DataFrame(df_data)

    # Get unique labels and create mappings
    unique_colors = np.array(sorted(set(labels_list), key=str))
    color_map = _get_color_map(unique_colors)

    unique_shapes: Optional[NDArray] = None
    marker_map: Optional[Dict[Any, str]] = None
    if shape_labels_list is not None:
        unique_shapes = np.array(sorted(set(shape_labels_list), key=str))
        marker_map = _get_marker_map(unique_shapes)

    # Create scatter plot for each component tuple
    for component_tuple in component_tuples:
        pc_columns = [f"PC{i}" for i in component_tuple]
        is_3d = len(component_tuple) == 3

        # Create figure
        if is_3d:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=figsize)

        # Plot scatter points
        _scatter_by_groups(ax, df, pc_columns, color_map, marker_map)

        # Set axis labels with explained variance
        for i, pc_idx in enumerate(component_tuple):
            variance = explained_variance_ratio[pc_idx]
            label_text = f"PC{pc_idx} ({variance:.1%})"
            if i == 0:
                ax.set_xlabel(label_text)
            elif i == 1:
                ax.set_ylabel(label_text)
            elif i == 2 and is_3d:
                ax.set_zlabel(label_text)  # type: ignore[attr-defined]

        # Set title
        if len(component_tuple) == 2:
            pc_x, pc_y = component_tuple
            plot_title = (
                f"{title}: PC{pc_x} vs PC{pc_y}"
                if title
                else f"PCA Scatter: PC{pc_x} vs PC{pc_y}"
            )
            filename = f"pc{pc_x}_vs_pc{pc_y}.png"
        else:
            pc_x, pc_y, pc_z = component_tuple
            plot_title = (
                f"{title}: PC{pc_x} vs PC{pc_y} vs PC{pc_z}"
                if title
                else f"PCA Scatter: PC{pc_x} vs PC{pc_y} vs PC{pc_z}"
            )
            filename = f"pc{pc_x}_vs_pc{pc_y}_vs_pc{pc_z}.png"

        ax.set_title(plot_title)

        # Add legend
        handles, legend_labels = _build_legend_handles(
            unique_colors, color_map, unique_shapes, marker_map
        )
        if is_3d:
            ax.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(1.0, 1))
        else:
            ax.legend(
                handles, legend_labels, bbox_to_anchor=(1.05, 1), loc="upper left"
            )
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()
        plt.close()
