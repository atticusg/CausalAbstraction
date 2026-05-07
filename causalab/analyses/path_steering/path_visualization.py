"""Path visualization — geodesic vs linear steering between two centroids.

Produces:
1. path_vis_3d.html — Interactive 3D: manifold mesh + geodesic + linear paths
2. path_vis_distributions.pdf — P(token) vs alpha, linear vs geodesic, mean ± std
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from torch import Tensor

from causalab.methods.metric import tokenize_variable_values
from causalab.methods.steer.collect import collect_grid_distributions
from causalab.analyses.activation_manifold.utils import _get_manifold_device
from causalab.io.plots.plot_3d_interactive import (
    PathTrace,
    plot_3d,
)
from causalab.io.plots.figure_format import normalize_figure_format
from causalab.methods.spline.builders import (
    compute_centroids,
    extract_parameters_from_dataset,
)

logger = logging.getLogger(__name__)


def _extract_manifold_components(featurizer) -> dict[str, Any]:
    """Extract manifold_obj, mean, std from a composed featurizer.

    Expects stages = [..., standardize, manifold].
    """
    manifold = featurizer.stages[-1].featurizer.manifold
    std_stage = featurizer.stages[-2].featurizer
    return {
        "manifold_obj": manifold,
        "mean": std_stage._mean,
        "std": std_stage._std,
    }


def _get_centroid_intrinsic_coords(
    features: Tensor,
    train_dataset: list,
    manifold_obj: Any,
    mean: Tensor,
    std: Tensor,
    intervention_variable: str,
    embeddings: dict | None,
) -> tuple[dict, Tensor, Tensor, dict]:
    """Compute per-value centroid intrinsic coordinates.

    Returns:
        value_to_intrinsic: Maps float/tuple key -> (d,) intrinsic coords.
        centroid_features: (n_centroids, k) features in ambient space.
        control_points: (n_centroids, n_params) parameter values.
        metadata: Dict from compute_centroids.
    """
    # Extract parameter values for the target variable only
    excluded = None
    if train_dataset:
        first_input = train_dataset[0].get("input", {})
        if hasattr(first_input, "_values"):
            all_vars = set(first_input._values.keys())
        elif isinstance(first_input, dict):
            all_vars = set(first_input.keys())
        else:
            all_vars = set()
        if all_vars:
            excluded = all_vars - {intervention_variable}

    param_tensors = extract_parameters_from_dataset(
        train_dataset,
        excluded_vars=excluded,
        embeddings=embeddings,
    )

    # Compute centroids in standardized space
    features_cpu = features.detach().cpu().float()
    mean_cpu = mean.detach().cpu()
    std_cpu = std.detach().cpu()
    features_std = (features_cpu - mean_cpu) / (std_cpu + 1e-6)

    control_points, centroids_std, metadata = compute_centroids(
        features_std,
        param_tensors,
    )

    # Encode centroids to intrinsic space using nearest-centroid
    device = _get_manifold_device(manifold_obj)
    centroids_d = centroids_std.to(device)
    with torch.no_grad():
        intrinsic, _ = manifold_obj.encode(centroids_d)
    intrinsic = intrinsic.cpu()

    # Map parameter values to intrinsic coordinates
    param_names = metadata["parameter_names"]
    value_to_intrinsic = {}
    for ci in range(control_points.shape[0]):
        if len(param_names) == 1:
            val = control_points[ci, 0].item()
            value_to_intrinsic[val] = intrinsic[ci]
        else:
            key = tuple(control_points[ci, j].item() for j in range(len(param_names)))
            value_to_intrinsic[key] = intrinsic[ci]

    # Denormalize centroids back to ambient space
    centroid_features = centroids_std * (std_cpu + 1e-6) + mean_cpu

    return value_to_intrinsic, centroid_features, control_points, metadata


def _parse_value(value: str):
    """Parse a value string into a Python object (float, tuple, or str)."""
    import ast

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (int, float)):
            return float(parsed)
        if isinstance(parsed, (tuple, list)):
            # Try all-float tuple first; fall back to mixed (str, int, ...) tuple
            try:
                return tuple(float(x) for x in parsed)
            except (ValueError, TypeError):
                return tuple(parsed)
        return value
    except (ValueError, SyntaxError):
        return value


def _resolve_value_key(
    value: str,
    value_to_intrinsic: dict,
    variable_values: list,
    embeddings: dict | None,
    intervention_variable: str,
) -> Tensor | None:
    """Look up intrinsic coords for a value string."""
    parsed = _parse_value(value)

    # Direct tuple/float match against centroid keys (all-numeric tuples only)
    if isinstance(parsed, tuple) and all(isinstance(x, (int, float)) for x in parsed):
        for k, v in value_to_intrinsic.items():
            if isinstance(k, tuple) and len(k) == len(parsed):
                if all(abs(a - b) < 1e-4 for a, b in zip(k, parsed)):
                    return v
    elif isinstance(parsed, float):
        for k, v in value_to_intrinsic.items():
            if isinstance(k, (int, float)) and abs(k - parsed) < 1e-4:
                return v

    # Try embedding-based lookup
    if embeddings and intervention_variable in embeddings:
        embed_fn = embeddings[intervention_variable]
        # Pass the parsed value (tuple/float), not the raw string
        try:
            coords = embed_fn(parsed if not isinstance(parsed, str) else value)
        except (TypeError, ValueError, KeyError):
            coords = None
        if coords is not None:
            if len(coords) == 1:
                for k, v in value_to_intrinsic.items():
                    if isinstance(k, (int, float)) and abs(k - coords[0]) < 1e-4:
                        return v
            else:
                key = tuple(coords)
                for k, v in value_to_intrinsic.items():
                    if isinstance(k, tuple) and len(k) == len(key):
                        if all(abs(a - b) < 1e-4 for a, b in zip(k, key)):
                            return v

    # Try index in variable_values
    if isinstance(parsed, str) and parsed in variable_values:
        idx = float(variable_values.index(parsed))
        for k, v in value_to_intrinsic.items():
            if isinstance(k, (int, float)) and abs(k - idx) < 1e-4:
                return v

    return None


def _format_as_pi_fraction(value: float, tol: float = 1e-4) -> str | None:
    """Try to express a float as a nice fraction of π. Returns None if not close."""
    from fractions import Fraction

    if abs(value) < tol:
        return "0"
    ratio = value / math.pi
    frac = Fraction(ratio).limit_denominator(36)
    if abs(float(frac) - ratio) > tol:
        return None
    num, den = frac.numerator, frac.denominator
    if den == 1:
        return f"{num}π" if num != 1 else "π"
    if num == 1:
        return f"π/{den}"
    return f"{num}π/{den}"


def _format_value_label(value: float) -> str:
    """Format a numeric value as a π fraction if possible, otherwise as a number."""
    pi_label = _format_as_pi_fraction(value)
    if pi_label is not None:
        return pi_label
    # Clean integer-like floats
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def _format_endpoint(value_str: str) -> str:
    """Format a start/end value string with π fractions where applicable.

    Handles both scalar strings ("Monday", "3.14159") and tuple strings
    ("(0.0, 4.0)", "(2.7925, 4.0)").
    """
    parsed = _parse_value(value_str)
    if isinstance(parsed, tuple):
        parts = [
            _format_value_label(x) if isinstance(x, (int, float)) else str(x)
            for x in parsed
        ]
        return f"({', '.join(parts)})"
    elif isinstance(parsed, float):
        return _format_value_label(parsed)
    return value_str  # string like "Monday"


def _plot_path_distributions(
    ax,
    means: np.ndarray,
    x: np.ndarray,
    labels: list[str],
    colors: list[str],
    stds: np.ndarray | None = None,
    full_vocab_softmax: bool = False,
    other_color: str = "red",
    show_labels: bool = True,
    show_markers: bool = False,
    x_labels: list[str] | None = None,
    colored_indices: list[int] | None = None,
    dimmed_alpha: float = 0.2,
) -> None:
    """Plot P(token) lines on a single axes.

    Shared core for ground truth, single-mode steering, and multi-panel
    steering plots.

    Args:
        ax: Matplotlib axes to plot on.
        means: (P, W) mean probabilities per position/alpha.
        x: (P,) x-axis values (alpha or discrete positions).
        labels: Length-W labels for each line.
        colors: Length-W colors for each line.
        stds: (P, W) std deviations for fill_between bands. None to skip.
        full_vocab_softmax: If True, add dashed "other" line (1 - sum).
        other_color: Color for the "other" line.
        show_labels: Whether to add labels to lines (for legend).
        show_markers: Whether to add markers to lines.
        x_labels: Discrete tick labels for x-axis. If provided, sets ticks
            at integer positions with rotated labels.
        colored_indices: If given, only these line indices are drawn at full
            alpha and shown in the legend; the rest keep their colormap color
            but are dimmed to ``dimmed_alpha`` and excluded from the legend.
            ``None`` means all lines are fully colored and labeled.
        dimmed_alpha: Alpha to apply to non-highlighted lines.
    """
    from causalab.io.plots.plot_utils import FigureGenerator

    fg = FigureGenerator()

    W = means.shape[1]
    marker_kw = dict(marker="o", markersize=5) if show_markers else {}
    highlight_set = set(colored_indices) if colored_indices is not None else None
    for w in range(W):
        is_highlighted = highlight_set is None or w in highlight_set
        label = (
            labels[w] if (show_labels and is_highlighted and w < len(labels)) else None
        )
        line_alpha = 1.0 if is_highlighted else dimmed_alpha
        band_alpha = 0.2 if is_highlighted else dimmed_alpha * 0.25
        ax.plot(
            x,
            means[:, w],
            label=label,
            color=colors[w],
            linewidth=2,
            alpha=line_alpha,
            **marker_kw,
        )
        if stds is not None:
            ax.fill_between(
                x,
                (means[:, w] - stds[:, w]).clip(0),
                (means[:, w] + stds[:, w]).clip(0, 1),
                alpha=band_alpha,
                color=colors[w],
            )

    if full_vocab_softmax:
        other_mean = (1.0 - means.sum(axis=1)).clip(0)
        ax.plot(
            x,
            other_mean,
            label="other" if show_labels else None,
            color=other_color,
            linewidth=2,
            linestyle="--",
            **marker_kw,
        )
        if stds is not None:
            other_std = np.sqrt((stds**2).sum(axis=1))
            ax.fill_between(
                x,
                (other_mean - other_std).clip(0),
                (other_mean + other_std).clip(0, 1),
                alpha=0.2,
                color=other_color,
            )

    if x_labels is not None:
        if len(x_labels) > 15:
            # With many points, only label positions that parse as integers (centroids).
            tick_pos, tick_lbl = [], []
            for xi, lbl in zip(x, x_labels):
                try:
                    v = float(lbl)
                    if v == int(v):
                        tick_pos.append(xi)
                        tick_lbl.append(lbl)
                except (ValueError, TypeError):
                    tick_pos.append(xi)
                    tick_lbl.append(lbl)
        else:
            tick_pos, tick_lbl = list(x), x_labels
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=fg.font_sizes["tick"], rotation=90)

    ax.set_ylabel("P(token)", fontsize=fg.font_sizes["axis_label"])
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.15)
    fg.style_axes(ax, tick_size=fg.font_sizes["tick"])


def _build_grid_layout(
    variable_values: list,
) -> tuple[int, int, list[str], list[str], dict[int, tuple[int, int]]]:
    """Derive grid dimensions and labels from 2D variable values.

    Args:
        variable_values: List of (dim0_val, dim1_val) tuples.

    Returns:
        (n_rows, n_cols, row_labels, col_labels, class_to_rc)
        where class_to_rc maps class index to (row, col) in the grid.
    """
    dim0_vals = list(dict.fromkeys(v[0] for v in variable_values))
    dim1_vals = list(dict.fromkeys(v[1] for v in variable_values))
    dim0_to_row = {v: i for i, v in enumerate(dim0_vals)}
    dim1_to_col = {v: i for i, v in enumerate(dim1_vals)}
    class_to_rc = {
        i: (dim0_to_row[v[0]], dim1_to_col[v[1]]) for i, v in enumerate(variable_values)
    }
    return (
        len(dim0_vals),
        len(dim1_vals),
        [
            _format_value_label(v) if isinstance(v, (int, float)) else str(v)
            for v in dim0_vals
        ],
        [
            _format_value_label(v) if isinstance(v, (int, float)) else str(v)
            for v in dim1_vals
        ],
        class_to_rc,
    )


def _build_rc_to_w(
    variable_values: list,
    class_to_rc: dict[int, tuple[int, int]],
    score_values: list | None = None,
) -> dict[tuple[int, int], int]:
    """Map (row, col) grid positions to probability-vector indices.

    When ``score_values`` is provided, multiple grid cells may share the same
    token index. Otherwise the mapping is 1:1 with class indices.
    """
    if score_values is not None:
        score_list = list(score_values)
        dim1_vals = [v[1] for v in variable_values]
        dim0_vals = [v[0] for v in variable_values]
        score_str_set = {str(s) for s in score_list}
        if all(str(d) in score_str_set for d in dict.fromkeys(dim1_vals)):
            score_dim = 1
        elif all(str(d) in score_str_set for d in dict.fromkeys(dim0_vals)):
            score_dim = 0
        else:
            score_dim = 1
        score_str_list = [str(s) for s in score_list]
        rc_to_w = {}
        for cls_idx, (r, c) in class_to_rc.items():
            val = variable_values[cls_idx][score_dim]
            rc_to_w[(r, c)] = score_str_list.index(str(val))
        return rc_to_w
    return {(r, c): cls_idx for cls_idx, (r, c) in class_to_rc.items()}


def _render_grid_snapshots(
    snapshot_probs: np.ndarray,
    snapshot_labels: list[str],
    n_rows: int,
    n_cols: int,
    row_labels: list[str],
    col_labels: list[str],
    rc_to_w: dict[tuple[int, int], int],
    cell_colors: list[tuple[float, float, float]],
    color_by_dim: int,
    title: str,
    output_path: str,
    min_opacity: float = 0.0,
    gamma: float = 1.0,
    normalize_across_snapshots: bool = True,
    figsize: tuple[float, float] | None = None,
    font_scale: float = 1.0,
) -> None:
    """Render a row of grid snapshots as a single figure.

    Args:
        snapshot_probs: (S, W) array — one probability vector per snapshot.
        snapshot_labels: Length-S list of labels for each snapshot (e.g. "alpha=0.2").
        n_rows, n_cols: Grid dimensions.
        row_labels, col_labels: Tick labels for rows/columns.
        rc_to_w: Mapping from (row, col) to index into the W dimension.
        cell_colors: RGB tuple per row (color_by_dim=0) or per column (color_by_dim=1).
        color_by_dim: 0 to color by row, 1 to color by column.
        title: Figure suptitle.
        output_path: Output PDF path.
        min_opacity: Floor opacity so grid structure is always visible.
        gamma: Power scaling exponent for probability → opacity mapping.
            Values < 1 (e.g. 0.3) amplify low probabilities to reveal spread.
        normalize_across_snapshots: If True, divide all probabilities by the
            global max across all snapshots so the peak cell hits full opacity.
    """
    from matplotlib.patches import Rectangle
    from causalab.io.plots.plot_utils import FigureGenerator

    fg = FigureGenerator()
    n_snap = len(snapshot_labels)

    # Normalize across all snapshots so the peak cell hits full opacity
    global_max = float(snapshot_probs.max()) if normalize_across_snapshots else 1.0
    if global_max < 1e-10:
        global_max = 1.0

    fig, axes = plt.subplots(
        1,
        n_snap,
        figsize=figsize or (3.2 * n_snap, 3.5),
        constrained_layout=True,
        sharey=True,
    )
    if n_snap == 1:
        axes = [axes]

    for snap_i in range(n_snap):
        probs = snapshot_probs[snap_i]  # (W,)
        ax = axes[snap_i]
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_aspect("equal")

        for r in range(n_rows):
            for c in range(n_cols):
                w_idx = rc_to_w.get((r, c))
                prob = float(probs[w_idx]) / global_max if w_idx is not None else 0.0
                color = cell_colors[r if color_by_dim == 0 else c]
                scaled = prob**gamma if prob > 0 else 0.0
                opacity = min_opacity + (1.0 - min_opacity) * scaled
                rect = Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor=(*color, opacity),
                    edgecolor="gray",
                    linewidth=0.5,
                )
                ax.add_patch(rect)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, fontsize=7 * font_scale, rotation=45)
        ax.set_title(
            snapshot_labels[snap_i], fontsize=fg.font_sizes["title"] * font_scale
        )

        if snap_i == 0:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(row_labels, fontsize=9 * font_scale, fontweight="bold")

    fig.suptitle(title, fontsize=fg.font_sizes["title"] * font_scale, fontweight="bold")
    fg.save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved grid flow plot: %s", output_path)


def _get_grid_row_colors(
    n_rows: int,
    colormap: str = "managua",
) -> list[tuple[float, float, float]]:
    """Get row colors from a colormap."""
    try:
        row_cmap = plt.get_cmap(colormap)
    except ValueError:
        row_cmap = plt.get_cmap("tab10")
    return [mcolors.to_rgb(row_cmap(i / max(n_rows - 1, 1))) for i in range(n_rows)]


def _periodic_cdist(
    a: np.ndarray,
    b: np.ndarray,
    periodic_dims: list[int] | None = None,
    periods: list[float] | None = None,
) -> np.ndarray:
    """Compute pairwise L2 distances with periodic wrapping.

    Args:
        a: (N, d) array.
        b: (M, d) array.
        periodic_dims: Which dimensions are periodic.
        periods: Period length for each periodic dimension.

    Returns:
        (N, M) distance matrix.
    """
    diff = a[:, None, :] - b[None, :, :]  # (N, M, d)
    if periodic_dims and periods:
        for pd, per in zip(periodic_dims, periods):
            diff[:, :, pd] = diff[:, :, pd] - per * np.round(diff[:, :, pd] / per)
    return np.sqrt((diff**2).sum(axis=2))


def select_centroid_snapshots(
    pair_grid_points: np.ndarray,
    centroids: np.ndarray,
    n_snapshots: int = 5,
    periodic_dims: list[int] | None = None,
    periods: list[float] | None = None,
) -> list[np.ndarray]:
    """Select snapshot indices per pair at timesteps closest to centroids.

    For each pair's path, finds timesteps where the path passes closest to
    any centroid in embedding space (local minima of distance-to-nearest-
    centroid). Always includes endpoints. Falls back to uniform spacing
    when there aren't enough distinct close approaches.

    Args:
        pair_grid_points: (n_pairs, num_steps, d) path coordinates.
        centroids: (n_centroids, d) centroid positions.
        n_snapshots: Number of snapshots to select per pair.
        periodic_dims: Which dimensions are periodic (from manifold).
        periods: Period for each periodic dimension.

    Returns:
        List of (n_snapshots,) index arrays, one per pair.
    """
    n_pairs, n_steps, _ = pair_grid_points.shape
    result = []

    for pi in range(n_pairs):
        path = pair_grid_points[pi]  # (num_steps, d)
        # Distance from each timestep to nearest centroid
        dists = _periodic_cdist(
            path,
            centroids,
            periodic_dims,
            periods,
        )  # (num_steps, n_centroids)
        min_dist = dists.min(axis=1)  # (num_steps,)

        # Find local minima (close approaches to centroids)
        candidates = {0, n_steps - 1}
        for t in range(1, n_steps - 1):
            if min_dist[t] <= min_dist[t - 1] and min_dist[t] <= min_dist[t + 1]:
                candidates.add(t)

        candidates = sorted(candidates)

        if len(candidates) < n_snapshots:
            # Fill with uniform spacing
            uniform = np.linspace(0, n_steps - 1, n_snapshots, dtype=int).tolist()
            candidates = sorted(set(candidates) | set(uniform))

        if len(candidates) > n_snapshots:
            # Keep endpoints, subsample inner points evenly
            inner = candidates[1:-1]
            n_inner = n_snapshots - 2
            if len(inner) <= n_inner:
                selected = [candidates[0]] + inner + [candidates[-1]]
            else:
                idx = np.linspace(0, len(inner) - 1, n_inner, dtype=int)
                selected = [candidates[0]] + [inner[i] for i in idx] + [candidates[-1]]
            candidates = selected

        result.append(np.array(candidates[:n_snapshots], dtype=int))

    return result


def _render_single_mode_grid_flow(
    probs: Tensor,
    variable_values: list,
    start_value: str,
    end_value: str,
    output_path: str,
    label: str,
    colormap: str = "managua",
    n_snapshots: int = 5,
    min_opacity: float = 0.0,
    coordinate_names: list[str] | None = None,
    snap_indices: np.ndarray | None = None,
    color_by_dim: int = 0,
    figsize: tuple[float, float] | None = None,
    font_scale: float = 1.0,
) -> None:
    """Grid flow for a single path mode. Wraps _render_grid_snapshots."""
    n_rows, n_cols, row_labels, col_labels, class_to_rc = _build_grid_layout(
        variable_values,
    )
    rc_to_w = _build_rc_to_w(variable_values, class_to_rc)
    n_colors = n_rows if color_by_dim == 0 else n_cols
    cell_colors = _get_grid_row_colors(n_colors, colormap)

    mean_probs = probs.mean(dim=1).numpy()  # (A, W)
    n_alpha = mean_probs.shape[0]
    if snap_indices is None:
        snap_indices = np.linspace(0, n_alpha - 1, n_snapshots, dtype=int)
    snap_labels = [f"\u03b1 = {idx / max(n_alpha - 1, 1):.2f}" for idx in snap_indices]

    _render_grid_snapshots(
        snapshot_probs=mean_probs[snap_indices],
        snapshot_labels=snap_labels,
        n_rows=n_rows,
        n_cols=n_cols,
        row_labels=row_labels,
        col_labels=col_labels,
        rc_to_w=rc_to_w,
        cell_colors=cell_colors,
        color_by_dim=color_by_dim,
        title=f"{label}: {_format_endpoint(start_value)} to {_format_endpoint(end_value)}",
        output_path=output_path,
        min_opacity=min_opacity,
        figsize=figsize,
        font_scale=font_scale,
    )


def _render_single_mode_distributions(
    probs: Tensor,
    variable_values: list,
    start_value: str,
    end_value: str,
    output_path: str,
    label: str,
    colormap: str = "tab10",
    full_vocab_softmax: bool = False,
) -> None:
    """Line plot of P(token) vs alpha for a single path mode."""
    from causalab.io.plots.plot_utils import FigureGenerator

    fg = FigureGenerator()
    n_alpha = probs.shape[0]
    alphas = np.linspace(0, 1, n_alpha)
    labels = [str(v) for v in variable_values]

    mean = probs.mean(dim=1).numpy()  # (A, W)
    std_dev = probs.std(dim=1).numpy()
    W = mean.shape[1]

    cmap = plt.get_cmap(colormap)
    colors = [mcolors.to_hex(cmap(i / max(W - 1, 1))) for i in range(W)]

    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_path_distributions(
        ax,
        means=mean,
        x=alphas,
        labels=labels,
        colors=colors,
        stds=std_dev,
        full_vocab_softmax=full_vocab_softmax,
    )
    ax.set_xlabel(
        f"{_format_endpoint(start_value)} to {_format_endpoint(end_value)}",
        fontsize=fg.font_sizes["axis_label"],
    )
    ax.set_title(label, fontsize=fg.font_sizes["title"])
    ax.set_xlim(0, 1)
    leg = ax.legend(
        fontsize=fg.font_sizes["legend"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    leg.set_in_layout(False)

    plt.tight_layout()
    fg.save_figure(fig, output_path, extra_artists=[leg])
    plt.close(fig)
    logger.info("Saved %s distribution plot: %s", label, output_path)


def _plot_grid_flow(
    geodesic_probs: Tensor,
    linear_probs: Tensor,
    variable_values: list,
    start_value: str,
    end_value: str,
    output_path: str,
    colormap: str = "managua",
    n_snapshots: int = 5,
    min_opacity: float = 0.0,
    coordinate_names: list[str] | None = None,
) -> None:
    """Grid flow visualization for path interpolation (geodesic + linear).

    Produces two PDFs (one per path type), each showing small-multiples of the
    2D grid at evenly spaced alpha steps. Only used when W == n_classes (each
    grid cell has its own distinct probability).

    Args:
        geodesic_probs: (A, N, W) per-sample distributions along geodesic path.
        linear_probs: (A, N, W) per-sample distributions along linear path.
        variable_values: Full list of (dim0, dim1) tuples. W must equal len.
        start_value: Start endpoint label.
        end_value: End endpoint label.
        output_path: Output PDF path (suffixed with _geodesic / _linear).
        colormap: Colormap for row colors.
        n_snapshots: Number of alpha snapshots to show.
        min_opacity: Floor opacity so grid structure is always visible.
        coordinate_names: Names for (dim0, dim1) axes.
    """
    n_rows, n_cols, row_labels, col_labels, class_to_rc = _build_grid_layout(
        variable_values,
    )
    rc_to_w = _build_rc_to_w(variable_values, class_to_rc)
    row_colors = _get_grid_row_colors(n_rows, colormap)

    n_alpha = geodesic_probs.shape[0]
    snap_indices = np.linspace(0, n_alpha - 1, n_snapshots, dtype=int)
    snap_labels = [f"\u03b1 = {idx / max(n_alpha - 1, 1):.2f}" for idx in snap_indices]

    for path_probs, path_label, suffix in [
        (geodesic_probs, "Manifold Steering", "geodesic"),
        (linear_probs, "Linear Steering", "linear"),
    ]:
        mean_probs = path_probs.mean(dim=1).numpy()  # (A, W)
        snapshot_probs = mean_probs[snap_indices]  # (n_snapshots, W)

        base, ext = os.path.splitext(output_path)
        path_out = f"{base}_{suffix}{ext}"

        _render_grid_snapshots(
            snapshot_probs=snapshot_probs,
            snapshot_labels=snap_labels,
            n_rows=n_rows,
            n_cols=n_cols,
            row_labels=row_labels,
            col_labels=col_labels,
            rc_to_w=rc_to_w,
            cell_colors=row_colors,
            color_by_dim=0,
            title=f"{path_label}: {_format_endpoint(start_value)} to {_format_endpoint(end_value)}",
            output_path=path_out,
            min_opacity=min_opacity,
        )


def _find_path_nodes(
    node_coordinates: list[tuple[float, ...]],
    start_node: int,
    end_node: int,
) -> tuple[list[int], int, list[int]]:
    """Find nodes along an axis-aligned path between two endpoints.

    Returns:
        (path_nodes, vary_dim, fixed_dims) — sorted node IDs, the varying
        dimension index, and the fixed dimension indices.
    """
    coords = node_coordinates
    start_coord = coords[start_node]
    end_coord = coords[end_node]
    n_dims = len(start_coord)

    varying_dims = [d for d in range(n_dims) if start_coord[d] != end_coord[d]]
    fixed_dims = [d for d in range(n_dims) if start_coord[d] == end_coord[d]]

    if len(varying_dims) != 1:
        logger.warning(
            "Ground truth path plot requires exactly one varying dimension, "
            f"got {len(varying_dims)} between {start_coord} and {end_coord}. "
            "Skipping."
        )
        return [], -1, fixed_dims

    vary_dim = varying_dims[0]
    fixed_vals = {d: start_coord[d] for d in fixed_dims}

    path_nodes = []
    for node_id in range(len(coords)):
        c = coords[node_id]
        if all(abs(c[d] - fixed_vals[d]) < 1e-6 for d in fixed_dims):
            vary_val = c[vary_dim]
            lo = min(start_coord[vary_dim], end_coord[vary_dim])
            hi = max(start_coord[vary_dim], end_coord[vary_dim])
            if lo - 1e-6 <= vary_val <= hi + 1e-6:
                path_nodes.append(node_id)

    path_nodes.sort(key=lambda nid: coords[nid][vary_dim])
    if start_coord[vary_dim] > end_coord[vary_dim]:
        path_nodes.reverse()

    return path_nodes, vary_dim, fixed_dims


def _plot_ground_truth_1d(
    path_dists: Tensor,
    node_labels: list[str],
    score_labels: list[str] | None,
    dim_name: str,
    output_path: str,
    colormap: str = "tab10",
    full_vocab_softmax: bool = False,
    title: str | None = None,
) -> None:
    """1D ground truth: line plot of P(token) at each node along a path."""
    from causalab.io.plots.plot_utils import FigureGenerator

    fg = FigureGenerator()
    data = path_dists.numpy()  # (P, W)
    W = data.shape[1]
    labels = score_labels or [str(i) for i in range(W)]

    cmap = plt.get_cmap(colormap)
    colors = [mcolors.to_hex(cmap(i / max(W - 1, 1))) for i in range(W)]

    plot_title = f"{title} ({dim_name})" if title else f"Ground truth ({dim_name})"
    x_pos = np.arange(len(node_labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_path_distributions(
        ax,
        means=data,
        x=x_pos,
        labels=labels,
        colors=colors,
        full_vocab_softmax=full_vocab_softmax,
        show_markers=True,
        x_labels=node_labels,
    )
    ax.set_title(plot_title, fontsize=fg.font_sizes["title"])
    ax.set_xlabel(
        f"{node_labels[0]} to {node_labels[-1]}",
        fontsize=fg.font_sizes["axis_label"],
    )
    leg = ax.legend(
        fontsize=fg.font_sizes["legend"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    leg.set_in_layout(False)

    plt.tight_layout()
    fg.save_figure(fig, output_path, extra_artists=[leg])
    plt.close(fig)
    logger.info("Saved ground truth line plot: %s", output_path)


def plot_ground_truth_path(
    avg_node_dists: Tensor,
    node_coordinates: list[tuple[float, ...]],
    start_node: int,
    end_node: int,
    output_path: str,
    coordinate_names: list[str] | None = None,
    score_labels: list[str] | None = None,
    colormap: str = "seismic",
    full_vocab_softmax: bool = False,
    title: str | None = None,
    node_value_labels: list[str] | None = None,
    variable_values: list | None = None,
) -> None:
    """Plot ground truth distributions along an axis-aligned path.

    For 2D variables where each grid cell has a distinct probability (W ==
    n_classes), renders a grid flow. Otherwise renders a line plot.

    Args:
        avg_node_dists: (n_nodes, W) tensor — W is n_score_tokens or n_nodes.
        node_coordinates: Length n_nodes list of coordinate tuples.
        start_node: Node ID at alpha=0.
        end_node: Node ID at alpha=1.
        output_path: Output PDF path.
        coordinate_names: Names for coordinate dimensions.
        score_labels: Labels for the W score token columns.
        colormap: Colormap for grid row colors or line plot.
        full_vocab_softmax: If True, add "other" line for non-concept probability.
        title: Optional title prefix for the plot.
        node_value_labels: Human-readable labels per node.
        variable_values: Full list of variable values (tuples for 2D). Required
            for grid flow rendering.
    """
    coords = node_coordinates
    n_dims = len(coords[0])
    n_nodes = avg_node_dists.shape[0]
    W = avg_node_dists.shape[1]

    path_nodes, vary_dim, _ = _find_path_nodes(coords, start_node, end_node)
    if len(path_nodes) < 2:
        logger.warning(f"Only {len(path_nodes)} nodes found along path, skipping.")
        return

    path_dists = avg_node_dists[path_nodes]  # (P, W)

    dim_name = (
        coordinate_names[vary_dim]
        if coordinate_names and vary_dim < len(coordinate_names)
        else f"dim{vary_dim}"
    )
    if node_value_labels is not None:
        node_labels = [node_value_labels[nid] for nid in path_nodes]
    else:
        node_labels = [f"{coords[nid][vary_dim]:.0f}" for nid in path_nodes]

    # Use grid flow only when 2D and each grid cell has its own probability
    # (W == n_classes). When score_labels exist (deduplicated tokens, e.g.
    # weekdays_2d where groups share tokens), use line plot instead.
    use_grid = (
        n_dims == 2
        and variable_values is not None
        and score_labels is None
        and W == n_nodes
    )

    if use_grid:
        n_rows, n_cols, row_labels, col_labels, class_to_rc = _build_grid_layout(
            variable_values,
        )
        rc_to_w = _build_rc_to_w(variable_values, class_to_rc)
        row_colors = _get_grid_row_colors(n_rows, colormap)

        plot_title = f"{title} ({dim_name})" if title else f"Ground truth ({dim_name})"
        _render_grid_snapshots(
            snapshot_probs=path_dists.numpy(),
            snapshot_labels=node_labels,
            n_rows=n_rows,
            n_cols=n_cols,
            row_labels=row_labels,
            col_labels=col_labels,
            rc_to_w=rc_to_w,
            cell_colors=row_colors,
            color_by_dim=0,
            title=plot_title,
            output_path=output_path,
        )
    else:
        _plot_ground_truth_1d(
            path_dists=path_dists,
            node_labels=node_labels,
            score_labels=score_labels,
            dim_name=dim_name,
            output_path=output_path,
            colormap=colormap,
            full_vocab_softmax=full_vocab_softmax,
            title=title,
        )


def plot_ground_truth_heatmaps(
    dists: Tensor,
    variable_values: list,
    output_dir: str,
    score_labels: list[str] | None = None,
    coordinate_names: list[str] | None = None,
    colormap: str = "seismic",
    full_vocab_softmax: bool = False,
    title_prefix: str | None = None,
    figure_format: str = "pdf",
    filename_prefix: str = "ground_truth",
) -> list[str]:
    """Auto-generate heatmaps for each coordinate dimension.

    For each dimension, holds other dimensions at their minimum value and
    sweeps from min to max. Produces one heatmap per dimension by calling
    :func:`plot_ground_truth_path`.

    Args:
        dists: (n_classes, n_score_tokens) distribution per class.
        variable_values: Length n_classes list of variable values (tuples or scalars).
        output_dir: Directory to save plots.
        score_labels: Labels for score token columns.
        coordinate_names: Names for coordinate dimensions.
        colormap: Colormap for heatmaps.
        full_vocab_softmax: If True, add "other" row and include in entropy.
        title_prefix: Optional prefix for plot titles.
        figure_format: File extension for saved figures (``png`` or ``pdf``).
        filename_prefix: Prefix for output filenames (e.g. "ground_truth", "manifold_steering", "patched").

    Returns:
        List of output file paths.
    """
    fmt = normalize_figure_format(figure_format, default="pdf")
    os.makedirs(output_dir, exist_ok=True)

    # Build numeric node coordinates from variable values.
    # For tuples: each position is a dimension. Strings are mapped to their
    # index among unique values in that position (e.g. "Monday" -> 0).
    if variable_values and isinstance(variable_values[0], (tuple, list)):
        n_dims = len(variable_values[0])
        dim_mappings: list[dict] = []
        dim_value_names: list[dict[float, str]] = []
        for d in range(n_dims):
            unique_vals = list(dict.fromkeys(v[d] for v in variable_values))
            if all(isinstance(u, (int, float)) for u in unique_vals):
                mapping = {u: float(u) for u in unique_vals}
            else:
                mapping = {u: float(i) for i, u in enumerate(unique_vals)}
            dim_mappings.append(mapping)
            dim_value_names.append({mapping[u]: str(u) for u in unique_vals})
        coords = [
            tuple(dim_mappings[d][v[d]] for d in range(n_dims)) for v in variable_values
        ]
        if coordinate_names is None:
            coordinate_names = [f"dim{d}" for d in range(n_dims)]
    else:
        coords = [(float(i),) for i in range(len(variable_values))]
        dim_value_names = [{float(i): str(v) for i, v in enumerate(variable_values)}]
        n_dims = 1

    output_paths = []
    for dim in range(n_dims):
        # Hold other dims at their minimum, sweep this dim
        fixed_vals = {d: min(c[d] for c in coords) for d in range(n_dims) if d != dim}

        matching_nodes = [
            nid
            for nid, c in enumerate(coords)
            if all(abs(c[d] - fixed_vals[d]) < 1e-6 for d in fixed_vals)
        ]

        if len(matching_nodes) < 2:
            logger.warning(f"Dim {dim}: only {len(matching_nodes)} nodes, skipping.")
            continue

        matching_nodes.sort(key=lambda nid: coords[nid][dim])

        dim_label = (
            coordinate_names[dim]
            if coordinate_names and dim < len(coordinate_names)
            else f"dim{dim}"
        )
        out_path = os.path.join(output_dir, f"{filename_prefix}_{dim_label}.{fmt}")

        # Build human-readable labels for each node
        names = dim_value_names[dim]
        node_labels = [names.get(c[dim], f"{c[dim]:.0f}") for c in coords]

        plot_ground_truth_path(
            avg_node_dists=dists,
            node_coordinates=coords,
            start_node=matching_nodes[0],
            end_node=matching_nodes[-1],
            output_path=out_path,
            coordinate_names=coordinate_names,
            score_labels=score_labels,
            colormap=colormap,
            full_vocab_softmax=full_vocab_softmax,
            title=title_prefix,
            node_value_labels=node_labels,
            variable_values=variable_values,
        )
        output_paths.append(out_path)

    return output_paths


def _resolve_centroid_endpoints(
    variable_values: list,
    start_value: str,
    end_value: str,
    centroids: Tensor,
    embeddings: dict | None,
    steered_variable: str,
) -> tuple[int, int]:
    """Find centroid indices matching start_value and end_value.

    Returns (start_idx, end_idx) into the centroids tensor.
    """
    parsed_start = _parse_value(start_value)
    parsed_end = _parse_value(end_value)

    def _match(parsed, values):
        # Direct match
        if parsed in values:
            return values.index(parsed)
        # String match
        for i, v in enumerate(values):
            if str(v) == str(parsed):
                return i
        # Embedding match for tuples
        if embeddings and steered_variable in embeddings:
            embed_fn = embeddings[steered_variable]
            try:
                coords = embed_fn(
                    parsed if not isinstance(parsed, str) else start_value
                )
                if coords is not None:
                    for i, v in enumerate(values):
                        v_coords = embed_fn(v)
                        if v_coords is not None and len(coords) == len(v_coords):
                            if all(abs(a - b) < 1e-4 for a, b in zip(coords, v_coords)):
                                return i
            except (TypeError, ValueError, KeyError):
                pass
        return None

    start_idx = _match(parsed_start, variable_values)
    end_idx = _match(parsed_end, variable_values)

    if start_idx is None:
        raise ValueError(
            f"Could not find centroid for start_value={start_value!r}. "
            f"Available values: {variable_values[:10]}..."
        )
    if end_idx is None:
        raise ValueError(
            f"Could not find centroid for end_value={end_value!r}. "
            f"Available values: {variable_values[:10]}..."
        )

    return start_idx, end_idx


def visualize_paths(
    pipeline,
    interchange_target,
    featurizer,
    filtered_samples: list[dict],
    task,
    start_value: str,
    end_value: str,
    num_steps_along_path: int,
    output_dir: str,
    features: Tensor,
    train_dataset: list,
    path_modes: list | None = None,
    n_prompts: int = 50,
    batch_size: int = 32,
    colormaps: str | list[str] = "tab10",
    path_colors: dict[str, str] | None = None,
    full_vocab_softmax: bool = False,
    figure_format: str = "pdf",
) -> dict[str, str]:
    """Generate path visualizations for each path mode.

    For each path mode, computes centroids in the appropriate space,
    builds an interpolation path, collects output distributions along it,
    and generates distribution + 3D plots.

    Args:
        pipeline: LMPipeline for steering.
        interchange_target: InterchangeTarget for interventions.
        featurizer: Composed featurizer (PCA >> standardize >> manifold).
        filtered_samples: Counterfactual samples for steering.
        task: Task object with variable_values, intervention_variable, etc.
        start_value: Start centroid value (e.g., "Monday").
        end_value: End centroid value (e.g., "Friday").
        num_steps_along_path: Number of interpolation steps.
        output_dir: Directory to save outputs.
        features: (N, k) training features in ambient space.
        train_dataset: Training dataset for centroid computation.
        path_modes: List of PathMode objects. If None, defaults to geometric only.
        n_prompts: Number of prompts for distribution collection.
        batch_size: Batch size for interventions.
        colormaps: Matplotlib colormap name(s) for distribution plots.
        path_colors: Per-mode colors for 3D plot (e.g. {"geometric": "black"}).
        full_vocab_softmax: If True, use full-vocabulary softmax for P(token).
        figure_format: ``png`` or ``pdf`` for distribution plots.

    Returns:
        Dict with output file paths.
    """
    from causalab.neural.activations.collect import collect_class_centroids

    os.makedirs(output_dir, exist_ok=True)
    _fig_fmt = normalize_figure_format(figure_format, default="pdf")

    if path_modes is None:
        # Caller didn't pass path_modes — shouldn't happen with proper orchestration,
        # but provide a safe default
        logger.warning("No path_modes provided, defaulting to geometric only")
        path_modes = []

    steered_variable = task.intervention_variable
    variable_values = task.intervention_values
    embeddings = getattr(task, "embeddings", None) or getattr(
        task.causal_model, "embeddings", None
    )

    # For 2D variables, use deduplicated output tokens for probability collection
    score_values = task.output_token_values
    token_values = score_values if score_values is not None else variable_values

    var_indices = tokenize_variable_values(
        pipeline.tokenizer,
        token_values,
        task.result_token_pattern,
    )

    # Normalize colormaps to a list
    if isinstance(colormaps, str):
        cmap_list = [colormaps]
    else:
        cmap_list = list(colormaps)

    # Build filename suffix from endpoints
    def _sanitize(v: str) -> str:
        return v.replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")

    path_suffix = f"{_sanitize(start_value)}_to_{_sanitize(end_value)}"

    eval_samples = filtered_samples[:n_prompts]
    expected_W = len(token_values)

    # Default path colors
    default_colors = {
        "geometric": "black",
        "linear_subspace": "darkgray",
        "linear": "red",
    }
    if path_colors:
        default_colors.update(path_colors)

    # Extract manifold components (needed for geometric modes and 3D plot)
    manifold_obj = None
    mean = None
    std = None
    ranges = None
    try:
        components = _extract_manifold_components(featurizer)
        manifold_obj = components["manifold_obj"]
        mean = components["mean"]
        std = components["std"]
        from causalab.analyses.activation_manifold.utils import (
            _compute_intrinsic_ranges,
        )

        ranges = _compute_intrinsic_ranges(features, manifold_obj, mean, std)
    except (AttributeError, IndexError):
        logger.info("No manifold found in featurizer — geometric modes will be skipped")

    # Collect distributions for each path mode
    path_results: dict[str, dict] = {}  # label -> {probs, path, is_intrinsic}

    for pm in path_modes:
        label = pm.label
        cache_path = os.path.join(output_dir, f"path_vis_data_{path_suffix}_{label}.pt")

        # Check cache
        cached_probs = None
        if os.path.exists(cache_path):
            logger.info("Loading cached %s data from %s", label, cache_path)
            cached = torch.load(cache_path, map_location="cpu")
            if (
                cached.get("probs") is not None
                and cached["probs"].shape[-1] == expected_W
            ):
                cached_probs = cached["probs"]

        if cached_probs is not None:
            path_results[label] = {
                "probs": cached_probs,
                "path": cached.get("grid_points"),
                "is_intrinsic": cached.get("is_intrinsic", False),
            }
            continue

        # Compute centroids and path for this mode
        if pm.centroid_space == "intrinsic":
            if manifold_obj is None:
                logger.warning("Skipping %s — no manifold available", label)
                continue

            value_to_intrinsic, centroid_features, control_points, _ = (
                _get_centroid_intrinsic_coords(
                    features,
                    train_dataset,
                    manifold_obj,
                    mean,
                    std,
                    steered_variable,
                    embeddings,
                )
            )
            start_pt = _resolve_value_key(
                start_value,
                value_to_intrinsic,
                variable_values,
                embeddings,
                steered_variable,
            )
            end_pt = _resolve_value_key(
                end_value,
                value_to_intrinsic,
                variable_values,
                embeddings,
                steered_variable,
            )
            if start_pt is None or end_pt is None:
                logger.warning("Could not resolve endpoints for %s, skipping", label)
                continue
            is_intrinsic = True
        else:
            # Centroid-based path. Swap featurizer to collect centroids in the
            # right space for this mode.
            active_featurizer = pm.featurizer_override or featurizer
            for u in interchange_target.flatten():
                u.set_featurizer(active_featurizer)

            try:
                centroids, mask = collect_class_centroids(
                    filtered_samples,
                    pipeline,
                    interchange_target,
                    task,
                )
            finally:
                for u in interchange_target.flatten():
                    u.set_featurizer(featurizer)

            start_idx, end_idx = _resolve_centroid_endpoints(
                variable_values,
                start_value,
                end_value,
                centroids,
                embeddings,
                steered_variable,
            )

            if not mask[start_idx] or not mask[end_idx]:
                logger.warning(
                    "Centroid missing for %s endpoints (start=%s, end=%s), skipping",
                    label,
                    mask[start_idx].item(),
                    mask[end_idx].item(),
                )
                continue

            start_pt = centroids[start_idx]
            end_pt = centroids[end_idx]
            is_intrinsic = False

        grid_points = pm.build_path(
            start_pt,
            end_pt,
            num_steps_along_path,
            manifold_obj=manifold_obj,
        )

        # Swap featurizer for distribution collection
        active_featurizer = pm.featurizer_override or featurizer
        for u in interchange_target.flatten():
            u.set_featurizer(active_featurizer)

        try:
            logger.info(
                "Collecting %s distributions (%d steps, %d prompts)...",
                label,
                num_steps_along_path,
                len(eval_samples),
            )
            probs = collect_grid_distributions(
                pipeline=pipeline,
                grid_points=grid_points,
                interchange_target=interchange_target,
                filtered_samples=eval_samples,
                var_indices=var_indices,
                batch_size=batch_size,
                n_base_samples=len(eval_samples),
                average=False,
                full_vocab_softmax=full_vocab_softmax,
            )  # (A, N, W)
        finally:
            for u in interchange_target.flatten():
                u.set_featurizer(featurizer)

        # Cache
        torch.save(
            {
                "probs": probs,
                "grid_points": grid_points,
                "is_intrinsic": is_intrinsic,
                "start_value": start_value,
                "end_value": end_value,
                "num_steps_along_path": num_steps_along_path,
            },
            cache_path,
        )

        path_results[label] = {
            "probs": probs,
            "path": grid_points,
            "is_intrinsic": is_intrinsic,
        }

    if not path_results:
        logger.warning("No path modes produced results, skipping visualization")
        return {}

    # Generate distribution plots — one per path mode
    n_dims = (
        len(variable_values[0])
        if variable_values and isinstance(variable_values[0], (tuple, list))
        else 1
    )
    output_files = {}

    for label, result in path_results.items():
        probs = result["probs"]  # (A, N, W)
        mode_dir = os.path.join(output_dir, label)
        os.makedirs(mode_dir, exist_ok=True)

        if n_dims == 2 and score_values is None:
            # 2D grid flow
            coordinate_names = getattr(task, "coordinate_names", None)
            dist_path = os.path.join(
                mode_dir,
                f"path_vis_grid_flow_{path_suffix}.{_fig_fmt}",
            )
            # _plot_grid_flow expects geodesic + linear; use single-mode wrapper
            _render_single_mode_grid_flow(
                probs=probs,
                variable_values=variable_values,
                start_value=start_value,
                end_value=end_value,
                output_path=dist_path,
                label=label,
                colormap=cmap_list[0] if cmap_list else "managua",
                coordinate_names=coordinate_names,
            )
        else:
            # Line plot
            plot_values = score_values if score_values is not None else variable_values
            dist_path = os.path.join(
                mode_dir,
                f"path_vis_distributions_{path_suffix}.{_fig_fmt}",
            )
            _render_single_mode_distributions(
                probs=probs,
                variable_values=plot_values,
                start_value=start_value,
                end_value=end_value,
                output_path=dist_path,
                label=label,
                colormap=cmap_list[0] if cmap_list else "tab10",
                full_vocab_softmax=full_vocab_softmax,
            )
        output_files[f"path_vis_{label}"] = dist_path

    # Generate 3D path plot with all path modes
    html_path = os.path.join(output_dir, f"path_vis_3d_{path_suffix}.html")
    path_traces = []
    for label, result in path_results.items():
        if result["path"] is not None:
            color = default_colors.get(label, "gray")
            # Noisy modes: derive color from base
            if label.startswith("noisy_"):
                color = default_colors.get("geometric", "black")
            path_traces.append(
                PathTrace(
                    points=result["path"],
                    name=label,
                    color=color,
                    is_intrinsic=result["is_intrinsic"],
                )
            )

    if manifold_obj is not None and path_traces:
        plot_3d(
            features=features,
            output_path=html_path,
            train_dataset=train_dataset,
            title=f"Steering: {_format_endpoint(start_value)} to {_format_endpoint(end_value)}",
            intervention_variable=steered_variable,
            embeddings=embeddings,
            colormap=cmap_list,
            manifold_obj=manifold_obj,
            mean=mean,
            std=std,
            ranges=ranges,
            paths=path_traces,
            variable_values=variable_values,
        )
        output_files["path_vis_3d"] = html_path

    return output_files


_PATH_MODE_TITLE = {
    "geometric": "Manifold Steering",
    "linear": "Linear Steering",
    "linear_subspace": "Linear Steering",
}


def _format_path_mode_title(label: str) -> str:
    """Pretty title for a path mode (e.g. ``geometric`` → ``Manifold Steering``)."""
    if label in _PATH_MODE_TITLE:
        return _PATH_MODE_TITLE[label]
    if label.startswith("noisy_"):
        base = label.split("_", 1)[1]
        base_title = _PATH_MODE_TITLE.get(base, base.replace("_", " ").title())
        return f"Noisy {base_title}"
    return label.replace("_", " ").title()


def _resolve_colored_indices(
    value_labels: list[str],
    colored_concepts: list[str] | None,
) -> list[int] | None:
    """Map a list of concept-label strings to indices into ``value_labels``.

    Returns ``None`` (meaning "color everything") when ``colored_concepts`` is
    ``None`` or empty. Unknown labels are silently dropped.
    """
    if not colored_concepts:
        return None
    label_to_idx = {lbl: i for i, lbl in enumerate(value_labels)}
    return [label_to_idx[c] for c in colored_concepts if c in label_to_idx]


def _is_2d_spatial(token_values: list) -> bool:
    """Check if output token values have a 2D spatial layout (e.g., grid coords)."""
    if not token_values:
        return False
    v = token_values[0]
    return (
        isinstance(v, (tuple, list))
        and len(v) == 2
        and all(isinstance(x, (int, float)) for x in v)
    )


def plot_saved_pair_distributions(
    pair_distributions: Tensor,
    pairs: list[tuple[int, int]],
    value_labels: list[str],
    output_dir: str,
    path_mode_label: str,
    score_labels: list[str] | None = None,
    colormap: str = "rainbow",
    full_vocab_softmax: bool = True,
    output_token_values: list | None = None,
    per_pair_snap_indices: list[np.ndarray] | None = None,
    color_by_dim: int = 0,
    figure_format: str = "pdf",
    colored_concepts_in_legend: list[str] | None = None,
    figsize: tuple[float, float] | list[float] | None = None,
    font_scale: float = 1.0,
) -> None:
    """Plot path distributions for all saved centroid pairs.

    Uses grid flow visualization when output tokens have a 2D spatial layout
    (e.g., graph walk grid coordinates), otherwise uses line plots.

    Args:
        pair_distributions: (n_pairs, num_steps, n_prompts, W) tensor.
        pairs: List of (start_idx, end_idx) pairs.
        value_labels: Label for each variable value (length n_values).
        output_dir: Directory to save plots.
        path_mode_label: Name of the path mode (for plot titles).
        score_labels: Labels for output token columns. Defaults to value_labels.
        colormap: Matplotlib colormap for lines.
        full_vocab_softmax: Whether to show "other" line.
        output_token_values: Raw output token values. If these are 2D numeric
            tuples (e.g., grid coordinates), uses grid flow visualization.
        per_pair_snap_indices: Pre-computed snapshot indices per pair (for
            consistent snapshots across path modes). If None, uses uniform spacing.
        color_by_dim: 0 to color grid cells by row (dim0), 1 by column (dim1).
        figure_format: ``png`` or ``pdf`` for saved figures.
        figsize: Optional ``(width, height)`` in inches. ``None`` uses each
            branch's default ((10, 5) for line plot; ``(3.2 * n_snap, 3.5)``
            for grid flow).
        font_scale: Multiplier on title/axis-label/tick/legend font sizes.
            ``1.0`` = unchanged.
    """
    from causalab.io.plots.plot_utils import FigureGenerator

    os.makedirs(output_dir, exist_ok=True)
    _fmt = normalize_figure_format(figure_format, default="pdf")

    use_grid = output_token_values is not None and _is_2d_spatial(output_token_values)
    _figsize = tuple(figsize) if figsize is not None else None

    if use_grid:
        for pi, (si, ei) in enumerate(pairs):
            probs = pair_distributions[pi]  # (num_steps, n_prompts, W)
            snap_idx = (
                per_pair_snap_indices[pi] if per_pair_snap_indices is not None else None
            )
            _render_single_mode_grid_flow(
                probs=probs,
                variable_values=output_token_values,
                start_value=value_labels[si],
                end_value=value_labels[ei],
                output_path=os.path.join(
                    output_dir,
                    f"pair_{value_labels[si]}_{value_labels[ei]}.{_fmt}",
                ),
                label=path_mode_label,
                colormap=colormap,
                snap_indices=snap_idx,
                color_by_dim=color_by_dim,
                figsize=_figsize,
                font_scale=font_scale,
            )
    else:
        labels = score_labels or value_labels
        W = pair_distributions.shape[-1]
        num_steps = pair_distributions.shape[1]
        cmap = plt.get_cmap(colormap)
        colors = [mcolors.to_hex(cmap(i / max(W - 1, 1))) for i in range(W)]
        alphas_x = np.linspace(0, 1, num_steps)
        colored_indices = _resolve_colored_indices(labels, colored_concepts_in_legend)

        for pi, (si, ei) in enumerate(pairs):
            fg = FigureGenerator()
            probs = pair_distributions[pi]  # (num_steps, n_prompts, W)
            means = probs.mean(dim=1).numpy()
            stds = probs.std(dim=1).numpy()

            fig, ax = plt.subplots(figsize=_figsize or (10, 5))
            _plot_path_distributions(
                ax,
                means=means,
                x=alphas_x,
                labels=labels,
                colors=colors,
                stds=stds,
                full_vocab_softmax=full_vocab_softmax,
                colored_indices=colored_indices,
            )
            ax.set_title(
                _format_path_mode_title(path_mode_label),
                fontsize=fg.font_sizes["title"] * font_scale,
            )
            ax.set_xlabel(
                f"{value_labels[si]} to {value_labels[ei]}",
                fontsize=fg.font_sizes["axis_label"] * font_scale,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks(np.arange(0.0, 1.01, 0.2))
            ax.set_yticks([0.0, 0.25, 0.5, 0.75])
            ax.tick_params(labelsize=fg.font_sizes["tick"] * font_scale)
            leg = ax.legend(
                fontsize=fg.font_sizes["legend"] * font_scale,
                loc="center left",
                bbox_to_anchor=(1.08, 0.5),
            )
            leg.set_in_layout(False)
            plt.tight_layout()
            fg.save_figure(
                fig,
                os.path.join(
                    output_dir,
                    f"pair_{value_labels[si]}_{value_labels[ei]}.{_fmt}",
                ),
                extra_artists=[leg],
            )
            plt.close(fig)

    logger.info("Saved %d path visualizations to %s", len(pairs), output_dir)


# ---------------------------------------------------------------------------
# Belief-space path visualization (MDS + Nyström)
# ---------------------------------------------------------------------------


def plot_paths_in_belief_space(
    natural_dists: Tensor,
    hellinger_pca: Any,
    all_pair_distributions: dict[str, Tensor],
    pairs: list[tuple[int, int]],
    value_labels: list[str],
    output_dir: str,
    path_colors: dict[str, str] | None = None,
    variable_values: list[str] | None = None,
    colormap: str | None = None,
    edges: list[tuple[int, int]] | None = None,
    edge_node_coords: dict[int, dict[str, float]] | None = None,
    intervention_variable: str | None = None,
    train_dataset: list = None,  # required: row-aligned with natural_dists, supplies TRUE class labels
    belief_manifold: Any = None,
) -> None:
    """Overlay steering paths on the belief-manifold √p embedding (3D).

    Sends ``√p`` (Hellinger embedding) of natural outputs, paths, and the
    belief manifold itself to ``plot_3d``, which fits its own internal PCA
    on the W+1 sphere features and renders the manifold mesh (curve for 1D
    output manifolds, surface for 2D) plus the path overlays uniformly.
    The ``hellinger_pca`` arg is accepted for backward compatibility but
    no longer used — projection is done by ``plot_3d``.
    """
    from causalab.io.plots.plot_3d_interactive import plot_3d, PathTrace
    from causalab.analyses.activation_manifold.utils import _compute_intrinsic_ranges

    if path_colors is None:
        path_colors = {"geometric": "black", "linear": "darkgray"}

    os.makedirs(output_dir, exist_ok=True)

    # √p sphere embedding for the natural outputs and per-pair paths.
    def _to_sqrt_simplex(p: torch.Tensor, target_dim: int) -> torch.Tensor:
        p = p.double()
        if p.shape[-1] < target_dim:
            other = (1.0 - p.sum(dim=-1, keepdim=True)).clamp(min=0.0)
            p = torch.cat([p, other], dim=-1)
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        return torch.sqrt(p.clamp(min=0)).float()

    pca_dim = natural_dists.shape[-1]  # W+1
    if belief_manifold is not None:
        pca_dim = belief_manifold.centroids.shape[-1]
    # Pad probabilities to pca_dim for path traces; raw_nat is per-example p
    # passed to plot_3d with feature_kind='hellinger' so it does the √ +
    # √(mean(p)) centroids itself.
    raw_nat = natural_dists.float()
    if raw_nat.shape[-1] < pca_dim:
        other = (1.0 - raw_nat.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        raw_nat = torch.cat([raw_nat, other], dim=-1)
    raw_nat = raw_nat / raw_nat.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    iv_name = intervention_variable or "class"
    if train_dataset is None:
        raise ValueError(
            "train_dataset is required: row-aligned with natural_dists, used to "
            "derive TRUE class labels."
        )
    if len(train_dataset) < raw_nat.shape[0]:
        raise ValueError(
            f"train_dataset length ({len(train_dataset)}) is shorter than "
            f"natural_dists rows ({raw_nat.shape[0]})."
        )
    label_to_idx = {str(v): i for i, v in enumerate(value_labels)}

    def _iv_val(ex):
        inp = ex["input"]
        return inp[iv_name] if isinstance(inp, dict) else inp._values.get(iv_name)

    classes = np.array(
        [
            label_to_idx.get(str(_iv_val(ex)), -1)
            for ex in train_dataset[: raw_nat.shape[0]]
        ]
    )
    if (classes < 0).any():
        bad = {
            str(_iv_val(ex))
            for ex in train_dataset[: raw_nat.shape[0]]
            if label_to_idx.get(str(_iv_val(ex)), -1) < 0
        }
        raise ValueError(
            f"train_dataset has IV={iv_name} values not present in value_labels: "
            f"{sorted(bad)}. Either pass the right value_labels or fix the dataset."
        )
    param_dict = {iv_name: classes.astype(float)}

    # Belief manifold has sphere_project=True → no standardization, so
    # mean=0, std=1 makes plot_3d's standardize step a no-op. Intrinsic
    # ranges are derived from the √p features (computed inside plot_3d).
    manifold_kwargs: dict[str, Any] = {}
    if belief_manifold is not None:
        zero_mean = torch.zeros(pca_dim)
        unit_std = torch.ones(pca_dim)
        sqrt_nat_for_ranges = torch.sqrt(raw_nat.clamp(min=0)).float()
        ranges = _compute_intrinsic_ranges(
            sqrt_nat_for_ranges, belief_manifold, zero_mean, unit_std
        )
        manifold_kwargs = dict(
            manifold_obj=belief_manifold,
            mean=zero_mean,
            std=unit_std,
            ranges=ranges,
        )

    for pi, (si, ei) in enumerate(pairs):
        path_traces_3d = []
        for mode_label, pair_dists in all_pair_distributions.items():
            if mode_label == "linear_subspace":
                continue
            path_dists = pair_dists[pi].mean(dim=1)
            path_traces_3d.append(
                PathTrace(
                    points=_to_sqrt_simplex(path_dists, pca_dim),
                    name=mode_label,
                    color=path_colors.get(mode_label, "gray"),
                    width=4,
                    is_intrinsic=False,
                )
            )
        pair_label = f"{value_labels[si]}_{value_labels[ei]}"
        plot_3d(
            features=raw_nat,
            output_path=os.path.join(output_dir, f"belief_space_{pair_label}.html"),
            param_dict=param_dict,
            intervention_variable=iv_name,
            variable_values=value_labels,
            colormap=colormap,
            paths=path_traces_3d,
            edges=edges,
            edge_node_coords=edge_node_coords,
            feature_kind="hellinger",
            **manifold_kwargs,
        )

    logger.info("Saved %d belief-space path plots to %s", len(pairs), output_dir)
