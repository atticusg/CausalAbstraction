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

from causalab.io.plots.figure_format import path_with_figure_format
from causalab.io.plots.utils import show_current_figure

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
    figure_format: str = "png",
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
                   - "mean": Mean used for centering (None if not centered)
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
                 as ``pc{{x}}_vs_pc{{y}}`` (extension from ``figure_format``).
        figsize: Figure size as (width, height) tuple.
        figure_format: ``png`` or ``pdf`` for static output (default ``png``).

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

    # Center features if mean was used during SVD (vanilla PCA)
    if mean is not None:
        if not isinstance(mean, Tensor):
            mean = torch.tensor(mean, dtype=features.dtype, device=features.device)
        features_centered = features - mean
    else:
        features_centered = features

    # Cast to float32 before projection — features may be bfloat16, which numpy can't handle
    features_centered = features_centered.to(dtype=torch.float32)
    rotation = rotation.to(dtype=torch.float32, device=features_centered.device)
    projected = features_centered @ rotation
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
            filename = f"pc{pc_x}_vs_pc{pc_y}"
        else:
            pc_x, pc_y, pc_z = component_tuple
            plot_title = (
                f"{title}: PC{pc_x} vs PC{pc_y} vs PC{pc_z}"
                if title
                else f"PCA Scatter: PC{pc_x} vs PC{pc_y} vs PC{pc_z}"
            )
            filename = f"pc{pc_x}_vs_pc{pc_y}_vs_pc{pc_z}"

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
            save_path = path_with_figure_format(
                os.path.join(save_dir, filename + ".png"),
                figure_format,
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        show_current_figure()
        plt.close()


def plot_features_2d(
    features: Tensor,
    output_path: str,
    train_dataset: list | None = None,
    title: str = "Features 2D",
    intervention_variable: str | None = None,
    edges: list[tuple[int, int]] | None = None,
    param_dict: Dict[str, NDArray] | None = None,
    edge_node_coords: Dict[int, Dict[str, float]] | None = None,
    embeddings: dict | None = None,
    colormap: str | None = None,
    variable_values: Optional[List[str]] = None,
    distance_matrix: Optional[Tensor] = None,
    figure_format: str = "pdf",
    pre_computed_centroids_2d: NDArray | None = None,
    # 'raw' (default): features are taken as-is. 'hellinger': features are
    # per-example PROBABILITIES (n, W+1); the function √-transforms them
    # internally and uses √(mean(p)) per class as centroids — the proper
    # Hellinger centroid (no Jensen drift). Use 'hellinger' whenever you have
    # probability data and want a Jensen-drift-free 2D scatter.
    feature_kind: str = "raw",
) -> None:
    """Plot training features + centroids colored by causal parameter (2D static image).

    Projects features to 2D via PCA (default) or MDS (when distance_matrix is
    provided), computes centroids, and draws a static scatter with categorical
    coloring and optional edge lines between centroids.

    Args:
        features: (n, k) features, already in subspace.
        output_path: Path for the output image; extension set by ``figure_format``.
        train_dataset: Counterfactual examples for extracting param values.
            Not needed if param_dict is provided directly.
        title: Plot title.
        intervention_variable: If set, only use this variable for coloring/centroids.
        edges: (i, j) index pairs for edge lines between centroids.
        param_dict: Pre-computed parameter values mapping param_name ->
            array of shape (n_features,). Alternative to train_dataset.
        edge_node_coords: Node ID -> {param_name: value} for edge remapping.
        embeddings: Optional dict mapping variable names to embedding functions.
        colormap: Matplotlib colormap name for coloring points by class.
        variable_values: String labels for each unique class value.
        distance_matrix: (n, n) precomputed distance matrix. When provided,
            uses MDS instead of PCA for the 2D projection.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    from causalab.io.plots.plot_3d_interactive import (
        _extract_param_values,
        _fit_mds,
    )
    from causalab.methods.spline.builders import compute_centroids

    import logging

    logger = logging.getLogger(__name__)

    # --- Resolve param_dict ---
    if param_dict is not None:
        param_names = sorted(param_dict.keys())
    elif train_dataset is not None:
        param_dict, param_names = _extract_param_values(
            train_dataset,
            intervention_variable=intervention_variable,
            embeddings=embeddings,
        )
    else:
        logger.warning("No train_dataset or param_dict provided; skipping plot.")
        return
    if not param_names:
        logger.warning("No causal parameters found; skipping plot.")
        return

    if feature_kind not in ("raw", "hellinger"):
        raise ValueError(
            f"feature_kind must be 'raw' or 'hellinger', got {feature_kind!r}",
        )
    hellinger_mode = feature_kind == "hellinger"
    raw_probs: Tensor | None = None
    if hellinger_mode:
        # Caller passed per-example probabilities. √-transform for projection
        # and display; keep originals for proper √(mean(p)) centroid math.
        raw_probs = features.detach().clone()
        features = torch.sqrt(raw_probs.clamp(min=0.0)).float()

    # --- 2D projection ---
    use_mds = distance_matrix is not None
    if use_mds:
        dm_np = (
            distance_matrix.numpy()
            if isinstance(distance_matrix, Tensor)
            else np.asarray(distance_matrix)
        )
        features_2d = _fit_mds(dm_np, n_components=2)
        projector_fn = None
    else:
        from causalab.io.plots.plot_3d_interactive import (
            _fit_projector,
        )

        features_2d, projector_fn = _fit_projector(features, n_components=2)
    n_features = features_2d.shape[0]

    # --- Centroids ---
    # In 'hellinger' mode aggregate per-class in probability space first then
    # √ (correct, lies on unit sphere). In 'raw' mode the centroid is just
    # mean(features) in 2D-projected space. pre_computed_centroids_2d wins
    # over both when supplied.
    param_tensors = {k: torch.tensor(v) for k, v in param_dict.items()}
    features_2d_t = torch.from_numpy(features_2d).float()
    if hellinger_mode and not use_mds:
        raw_cpu = raw_probs[:n_features].detach().cpu().float()
        control_points, mean_p, metadata = compute_centroids(raw_cpu, param_tensors)
        sqrt_mean_p = torch.sqrt(mean_p.clamp(min=0.0)).numpy()
        centroids_2d = projector_fn(sqrt_mean_p)
    else:
        control_points, centroids_2d_t, metadata = compute_centroids(
            features_2d_t, param_tensors
        )
        centroids_2d = centroids_2d_t.numpy()
    if pre_computed_centroids_2d is not None:
        centroids_2d = np.asarray(pre_computed_centroids_2d)
    centroid_param_names = metadata["parameter_names"]
    centroid_param_values = control_points.detach().cpu().numpy()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    ax.set_facecolor("white")

    # Draw edges
    if edges is not None:
        draw_edges = edges
        if edge_node_coords is not None:
            coord_to_ci: Dict[tuple, int] = {}
            for ci in range(centroids_2d.shape[0]):
                key = tuple(
                    round(float(centroid_param_values[ci, j]), 6)
                    for j in range(len(centroid_param_names))
                )
                coord_to_ci[key] = ci
            remapped: list[tuple[int, int]] = []
            for i, j in edges:
                ci = coord_to_ci.get(
                    tuple(
                        round(float(edge_node_coords[i][p]), 6)
                        for p in centroid_param_names
                    )
                )
                cj = coord_to_ci.get(
                    tuple(
                        round(float(edge_node_coords[j][p]), 6)
                        for p in centroid_param_names
                    )
                )
                if ci is not None and cj is not None:
                    remapped.append((ci, cj))
            draw_edges = remapped
        for i, j in draw_edges:
            if i >= len(centroids_2d) or j >= len(centroids_2d):
                continue
            ax.plot(
                [centroids_2d[i, 0], centroids_2d[j, 0]],
                [centroids_2d[i, 1], centroids_2d[j, 1]],
                color="gray",
                alpha=0.4,
                linewidth=1,
                zorder=1,
            )

    # Coloring
    pname = param_names[0]
    vals = param_dict[pname][:n_features]
    unique_vals = np.unique(vals[~np.isnan(vals)])
    n_unique = len(unique_vals)
    cmap_name = colormap or ("tab20" if n_unique > 10 else "tab10")
    cmap_obj = plt.colormaps.get_cmap(cmap_name).resampled(max(n_unique, 1))
    colors = [cmap_obj(i / max(n_unique - 1, 1)) for i in range(n_unique)]

    # Scatter points (low opacity)
    for i, uval in enumerate(unique_vals):
        mask = vals == uval
        label = (
            variable_values[i]
            if variable_values and i < len(variable_values)
            else str(int(uval) if uval == int(uval) else uval)
        )
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            s=10,
            alpha=0.25,
            color=colors[i],
            zorder=2,
            label=label,
        )

    # Centroids (solid, diamond markers)
    if pname in centroid_param_names:
        cp_idx = centroid_param_names.index(pname)
        centroid_vals = centroid_param_values[:, cp_idx]
    else:
        centroid_vals = np.arange(centroids_2d.shape[0], dtype=float)

    for ci in range(centroids_2d.shape[0]):
        color_idx = int(np.searchsorted(unique_vals, centroid_vals[ci]))
        color = colors[color_idx % n_unique]
        ax.scatter(
            centroids_2d[ci, 0],
            centroids_2d[ci, 1],
            s=100,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            marker="D",
            zorder=4,
        )

    dim_label = "MDS" if use_mds else "PC"
    ax.set_xlabel(f"{dim_label} 0")
    ax.set_ylabel(f"{dim_label} 1")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, ncol=max(1, n_unique // 10), markerscale=3, framealpha=0.8)
    fig.tight_layout()

    out = path_with_figure_format(output_path, figure_format)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {out}")


def plot_variance_histogram(
    explained_variance_ratio: list[float] | NDArray,
    save_path: str | None = None,
    figure_format: str = "png",
) -> None:
    """Cumulative explained variance curve relative to total embedding-space variance."""
    import logging

    logger = logging.getLogger(__name__)

    ratios = np.asarray(explained_variance_ratio)
    cumulative = np.cumsum(ratios) * 100
    k = len(ratios)

    fig, ax = plt.subplots(figsize=(max(6, min(k * 0.5 + 2, 16)), 4), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(range(1, k + 1), cumulative, color="steelblue")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (% of total)")
    ax.set_title("Cumulative PCA Variance — relative to full embedding space")
    ax.set_xticks(range(1, k + 1))
    ax.set_ylim(0, 100)
    ax.axhline(
        cumulative[-1],
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"k={k}: {cumulative[-1]:.1f}%",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        out = path_with_figure_format(save_path, figure_format)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, dpi=150, facecolor="white")
        logger.info(f"Saved {out}")
    plt.close(fig)


def plot_features_3d_static(
    features: Tensor,
    output_path: str,
    train_dataset: list | None = None,
    title: str = "Features 3D",
    intervention_variable: str | None = None,
    embeddings: dict | None = None,
    colormap: str | None = None,
    variable_values: Optional[List[str]] = None,
    figure_format: str = "png",
) -> None:
    """Static matplotlib 3D scatter of features colored by causal parameter.

    Projects features to 3D via PCA, computes centroids, and draws a static
    3D scatter with categorical coloring.

    Args:
        features: (n, k) features, already in subspace.
        output_path: Path for the output image; extension set by ``figure_format``.
        train_dataset: Counterfactual examples for extracting param values.
        title: Plot title.
        intervention_variable: If set, only use this variable for coloring.
        embeddings: Optional dict mapping variable names to embedding functions.
        colormap: Matplotlib colormap name for coloring points by class.
        variable_values: String labels for each unique class value.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    from causalab.io.plots.plot_3d_interactive import (
        _extract_param_values,
        _fit_projector,
    )
    from causalab.methods.spline.builders import compute_centroids

    import logging

    logger = logging.getLogger(__name__)

    if train_dataset is None:
        logger.warning("No train_dataset provided; skipping 3D static plot.")
        return

    param_dict, param_names = _extract_param_values(
        train_dataset,
        intervention_variable=intervention_variable,
        embeddings=embeddings,
    )
    if not param_names:
        logger.warning("No causal parameters found; skipping 3D static plot.")
        return

    features_3d, _ = _fit_projector(features, n_components=3)
    n_features = features_3d.shape[0]

    param_tensors = {k: torch.tensor(v) for k, v in param_dict.items()}
    features_3d_t = torch.from_numpy(features_3d).float()
    control_points, centroids_3d_t, metadata = compute_centroids(
        features_3d_t, param_tensors
    )
    centroids_3d = centroids_3d_t.numpy()
    centroid_param_names = metadata["parameter_names"]
    centroid_param_values = control_points.detach().cpu().numpy()

    pname = param_names[0]
    vals = param_dict[pname][:n_features]
    unique_vals = np.unique(vals[~np.isnan(vals)])
    n_unique = len(unique_vals)
    cmap_name = colormap or ("tab20" if n_unique > 10 else "tab10")
    cmap_obj = plt.colormaps.get_cmap(cmap_name).resampled(max(n_unique, 1))
    colors = [cmap_obj(i / max(n_unique - 1, 1)) for i in range(n_unique)]

    fig = plt.figure(figsize=(9, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    for i, uval in enumerate(unique_vals):
        mask = vals == uval
        label = (
            variable_values[i]
            if variable_values and i < len(variable_values)
            else str(int(uval) if uval == int(uval) else uval)
        )
        ax.scatter(
            features_3d[mask, 0],
            features_3d[mask, 1],
            features_3d[mask, 2],
            s=10,
            alpha=0.25,
            color=colors[i],
            label=label,
        )

    if pname in centroid_param_names:
        cp_idx = centroid_param_names.index(pname)
        centroid_vals = centroid_param_values[:, cp_idx]
    else:
        centroid_vals = np.arange(centroids_3d.shape[0], dtype=float)

    for ci in range(centroids_3d.shape[0]):
        color_idx = int(np.searchsorted(unique_vals, centroid_vals[ci]))
        color = colors[color_idx % n_unique]
        ax.scatter(
            centroids_3d[ci, 0],
            centroids_3d[ci, 1],
            centroids_3d[ci, 2],
            s=80,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            marker="D",
            zorder=4,
        )

    ax.set_xlabel("PC 0")
    ax.set_ylabel("PC 1")
    ax.set_zlabel("PC 2")
    ax.set_title(title)
    ax.legend(
        fontsize=7,
        ncol=max(1, n_unique // 10),
        markerscale=3,
        framealpha=0.8,
        loc="upper left",
    )
    fig.tight_layout()

    out = path_with_figure_format(output_path, figure_format)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {out}")


def plot_manifold_3d_static(
    features: Tensor,
    manifold_obj: Any,
    mean: Tensor,
    std: Tensor,
    ranges: tuple,
    output_path: str,
    train_dataset: list | None = None,
    title: str = "Manifold 3D",
    intervention_variable: str | None = None,
    embeddings: dict | None = None,
    colormap: str | None = None,
    variable_values: Optional[List[str]] = None,
    grid_res: int = 200,
    figure_format: str = "png",
    paths: list | None = None,
    param_dict: dict | None = None,
) -> None:
    """Static matplotlib 3D scatter + manifold curve (1D) or surface (2D).

    Mirrors the Plotly manifold_3d.html output as a saveable PNG. Fits the same
    PCA projector on the features, decodes the manifold via ``manifold_obj.decode``,
    and overlays the curve/surface on the scatter.

    Args:
        features: (n, k) features in subspace coordinates (pre-standardization).
        manifold_obj: Fitted SplineManifold (or any object with ``.decode(u, r)``).
        mean: (k,) standardization mean applied before manifold.
        std: (k,) standardization std applied before manifold.
        ranges: Intrinsic coordinate ranges, one (lo, hi) per intrinsic dim.
        output_path: Output file path (extension set by ``figure_format``).
        train_dataset: Counterfactual examples for parameter coloring.
        title: Plot title.
        intervention_variable: Variable to use for coloring.
        embeddings: Optional embedding functions.
        colormap: Matplotlib colormap name.
        variable_values: String labels for unique class values.
        grid_res: Number of sample points along the manifold curve.
        figure_format: ``png`` or ``pdf``.
    """
    import logging
    import torch
    from causalab.io.plots.plot_3d_interactive import (
        _extract_param_values,
        _fit_projector,
    )
    from causalab.methods.spline.builders import compute_centroids

    logger = logging.getLogger(__name__)

    # --- Fit PCA projector on features ---
    features_3d, project_fn = _fit_projector(features, n_components=3)
    n_pts = features_3d.shape[0]

    # --- Param coloring ---
    if param_dict is not None:
        param_names = list(param_dict.keys())
    elif train_dataset is not None:
        param_dict, param_names = _extract_param_values(
            train_dataset,
            intervention_variable=intervention_variable,
            embeddings=embeddings,
        )
    else:
        param_dict, param_names = {}, []

    # --- Figure ---
    fig = plt.figure(figsize=(9, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    if param_names:
        pname = param_names[0]
        vals = param_dict[pname][:n_pts]
        unique_vals = np.unique(vals[~np.isnan(vals)])
        n_unique = len(unique_vals)
        cmap_name = colormap or ("tab20" if n_unique > 10 else "tab10")
        cmap_obj = plt.colormaps.get_cmap(cmap_name).resampled(max(n_unique, 1))
        colors = [cmap_obj(i / max(n_unique - 1, 1)) for i in range(n_unique)]

        # Scatter points
        for i, uval in enumerate(unique_vals):
            mask = vals == uval
            label = (
                variable_values[i]
                if variable_values and i < len(variable_values)
                else str(int(uval) if uval == int(uval) else uval)
            )
            ax.scatter(
                features_3d[mask, 0],
                features_3d[mask, 1],
                features_3d[mask, 2],
                s=10,
                alpha=0.25,
                color=colors[i],
                label=label,
            )

        # Centroids
        param_tensors = {k: torch.tensor(v) for k, v in param_dict.items()}
        features_3d_t = torch.from_numpy(features_3d).float()
        control_points, centroids_3d_t, metadata = compute_centroids(
            features_3d_t, param_tensors
        )
        centroids_3d = centroids_3d_t.numpy()
        centroid_param_names = metadata["parameter_names"]
        centroid_param_values = control_points.detach().cpu().numpy()

        if pname in centroid_param_names:
            cp_idx = centroid_param_names.index(pname)
            centroid_vals = centroid_param_values[:, cp_idx]
        else:
            centroid_vals = np.arange(centroids_3d.shape[0], dtype=float)

        for ci in range(centroids_3d.shape[0]):
            color_idx = int(np.searchsorted(unique_vals, centroid_vals[ci]))
            ax.scatter(
                centroids_3d[ci, 0],
                centroids_3d[ci, 1],
                centroids_3d[ci, 2],
                s=80,
                color=colors[color_idx % n_unique],
                edgecolors="black",
                linewidths=0.8,
                marker="D",
                zorder=4,
            )

    else:
        ax.scatter(
            features_3d[:, 0],
            features_3d[:, 1],
            features_3d[:, 2],
            s=10,
            alpha=0.25,
            color="steelblue",
        )

    # --- Manifold curve / surface ---
    try:
        d = len(ranges)
        try:
            device = next(manifold_obj.buffers()).device
        except StopIteration:
            device = mean.device
        mean_d = mean.to(device)
        std_d = std.to(device)
        periodic_dims: set[int] = set(getattr(manifold_obj, "periodic_dims", []))

        if d == 1:
            periodic = 0 in periodic_dims
            u = np.linspace(ranges[0][0], ranges[0][1], grid_res, endpoint=not periodic)
            u_tensor = torch.tensor(u[:, None], dtype=torch.float32, device=device)
            with torch.no_grad():
                decoded = manifold_obj.decode(u_tensor, r=None)
                decoded = decoded * (std_d + 1e-6) + mean_d
            curve_3d = project_fn(decoded.cpu().numpy())
            if periodic:
                curve_3d = np.concatenate([curve_3d, curve_3d[:1]], axis=0)
            ax.plot(
                curve_3d[:, 0],
                curve_3d[:, 1],
                curve_3d[:, 2],
                color="dimgray",
                linewidth=2,
                zorder=5,
                label="Manifold",
            )

        elif d == 2:
            g = int(grid_res**0.5)
            u0 = np.linspace(ranges[0][0], ranges[0][1], g)
            u1 = np.linspace(ranges[1][0], ranges[1][1], g)
            U0, U1 = np.meshgrid(u0, u1)
            grid = np.stack([U0.ravel(), U1.ravel()], axis=1)
            u_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
            with torch.no_grad():
                decoded = manifold_obj.decode(u_tensor, r=None)
                decoded = decoded * (std_d + 1e-6) + mean_d
            surf_3d = project_fn(decoded.cpu().numpy()).reshape(g, g, 3)
            ax.plot_surface(
                surf_3d[:, :, 0],
                surf_3d[:, :, 1],
                surf_3d[:, :, 2],
                alpha=0.25,
                color="dimgray",
            )
    except Exception as e:
        logger.warning("Manifold rendering failed: %s", e, exc_info=True)

    # --- Path traces ---
    if paths:
        for pt in paths:
            pts = pt.points
            pts_np = pts.numpy() if hasattr(pts, "numpy") else np.array(pts)
            pts_3d = project_fn(pts_np)
            ax.plot(
                pts_3d[:, 0],
                pts_3d[:, 1],
                pts_3d[:, 2],
                color=getattr(pt, "color", "black"),
                linewidth=2.5,
                zorder=6,
                label=getattr(pt, "name", "path"),
            )

    # Legend — drawn after paths so path entries are included
    n_unique = (
        len(
            np.unique(param_dict[param_names[0]][~np.isnan(param_dict[param_names[0]])])
        )
        if param_names
        else 0
    )
    ax.legend(
        fontsize=7,
        ncol=max(1, n_unique // 10),
        markerscale=3,
        framealpha=0.8,
        loc="upper left",
    )

    ax.set_xlabel("PC 0")
    ax.set_ylabel("PC 1")
    ax.set_zlabel("PC 2")
    ax.set_title(title)
    fig.tight_layout()

    out = path_with_figure_format(output_path, figure_format)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {out}")
