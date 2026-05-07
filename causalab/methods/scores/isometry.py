"""
Isometry Metric — Compare the geometry of the activation manifold to the
geometry of the output (belief) manifold.

Both modes operate on the **same vertex set** — manifold-derived points
sampled at equispaced fractions along each centroid pair's u-space geodesic.
Modes differ only in the distance metric on those vertices:

  geometric: D_X[a, b] = sum_k ||act_decode(u_k+1) − act_decode(u_k)||  (arc length)
  linear:    D_X[a, b] = ||act_decode(u_a) − act_decode(u_b)||           (chord)

D_Y is always the belief-manifold arc length between u_a and u_b (Hellinger).

Pearson correlation between D_X and D_Y measures isometry — purely a property
of the two learned manifolds, with no dependence on intervened model outputs.
The off-PCA-subspace component of the raw residual stream is intentionally
excluded: the activation manifold lives in the PCA subspace by construction,
so off-subspace variation isn't part of its intrinsic geometry.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from omegaconf import DictConfig

import numpy as np
import torch
from torch import Tensor

from causalab.methods.distances import (
    DISTANCE_LABELS as _DISTANCE_LABELS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------


def compute_isometry_metrics(
    D_X: np.ndarray,
    D_Y: np.ndarray,
) -> dict[str, Any]:
    """Pearson correlation between two distance arrays.

    Accepts either 1D vectors of pair distances or square distance matrices
    (in which case the strict upper triangle is used).
    """
    if D_X.ndim == 2:
        idx = np.triu_indices_from(D_X, k=1)
        dx = D_X[idx]
        dy = D_Y[idx]
    else:
        dx = np.asarray(D_X).flatten()
        dy = np.asarray(D_Y).flatten()

    n_pairs = int(len(dx))
    if n_pairs == 0 or np.std(dx) == 0 or np.std(dy) == 0:
        return {"pearson_r": float("nan"), "n_pairs": n_pairs}

    pearson_r = float(np.corrcoef(dx, dy)[0, 1])
    return {"pearson_r": pearson_r, "n_pairs": n_pairs}


# ---------------------------------------------------------------------------
# Path-length helpers
# ---------------------------------------------------------------------------


def _shortest_arc_delta(
    u_a: Tensor,
    u_b: Tensor,
    periodic_dims: list[int] | None,
    periods: list[float] | None,
) -> Tensor:
    """Return u_b - u_a, wrapping each periodic dim to the shorter direction."""
    delta = u_b - u_a
    if periodic_dims and periods:
        for pd, per in zip(periodic_dims, periods):
            if abs(float(delta[pd])) > per / 2:
                delta[pd] = delta[pd] - torch.sign(delta[pd]) * per
    return delta


def _shortest_arc_delta_batched(
    u_a: Tensor,
    u_b: Tensor,
    periodic_dims: list[int] | None,
    periods: list[float] | None,
) -> Tensor:
    """Batched ``_shortest_arc_delta``. ``u_a``, ``u_b`` of shape ``(B, d)``."""
    delta = u_b - u_a
    if periodic_dims and periods:
        for pd, per in zip(periodic_dims, periods):
            col = delta[:, pd]
            wrap = col.abs() > (per / 2)
            delta[:, pd] = torch.where(wrap, col - torch.sign(col) * per, col)
    return delta


def _decoded_path_length_batched(
    u_a: Tensor,
    u_b: Tensor,
    decode_fn: Callable[[Tensor], Tensor],
    n_steps: int,
    periodic_dims: list[int] | None = None,
    periods: list[float] | None = None,
    chunk_size: int = 1000,
) -> Tensor:
    """Batched arc-length sum over many ``(u_a, u_b)`` pairs.

    Numerically equivalent to calling :func:`_decoded_path_length` per pair
    (modulo float associativity in the sum), but issues a single batched
    ``decode_fn`` call per chunk instead of one per pair.

    Args:
        u_a, u_b: ``(B, d)`` u-space endpoints.
        chunk_size: Cap on pairs per batched decode (bounds peak memory at
            roughly ``chunk_size * (n_steps + 1) * d_out`` floats).

    Returns:
        ``(B,)`` tensor of path lengths.
    """
    B = u_a.shape[0]
    if B == 0:
        return torch.zeros(0, device=u_a.device, dtype=u_a.dtype)
    out = torch.empty(B, device=u_a.device, dtype=u_a.dtype)
    t = torch.linspace(0, 1, n_steps + 1, device=u_a.device, dtype=u_a.dtype)
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        ua = u_a[start:end]  # (b, d)
        ub = u_b[start:end]  # (b, d)
        delta = _shortest_arc_delta_batched(ua, ub, periodic_dims, periods)  # (b, d)
        # u_path: (b, n_steps+1, d) = ua[:, None, :] + t[None, :, None] * delta[:, None, :]
        u_path = ua.unsqueeze(1) + t.view(1, -1, 1) * delta.unsqueeze(1)
        b, n_pts, d = u_path.shape
        with torch.no_grad():
            decoded = decode_fn(u_path.reshape(b * n_pts, d))
        decoded = decoded.reshape(b, n_pts, -1)
        diffs = decoded[:, 1:, :] - decoded[:, :-1, :]
        out[start:end] = diffs.norm(dim=-1).sum(dim=-1)
    return out


def _decoded_path_length(
    u_a: Tensor,
    u_b: Tensor,
    decode_fn: Callable[[Tensor], Tensor],
    n_steps: int,
    periodic_dims: list[int] | None = None,
    periods: list[float] | None = None,
) -> float:
    """Sum of consecutive Euclidean distances between decoded path points.

    Builds an n_steps+1 piecewise-linear path in u-space (with periodic
    shortest-arc handling) and sums ||decode(u_{k+1}) - decode(u_k)||₂.
    """
    delta = _shortest_arc_delta(u_a, u_b, periodic_dims, periods)
    t = torch.linspace(0, 1, n_steps + 1, device=u_a.device, dtype=u_a.dtype)
    u_path = u_a.unsqueeze(0) + t.unsqueeze(1) * delta.unsqueeze(0)
    with torch.no_grad():
        decoded = decode_fn(u_path)
    diffs = decoded[1:] - decoded[:-1]
    return float(diffs.norm(dim=-1).sum())


def _activation_decode_fn(manifold: Any, mean: Tensor, std: Tensor):
    """Build a decode callable that un-standardises to ambient activation space."""

    def _decode(u: Tensor) -> Tensor:
        return manifold.decode(u) * (std + 1e-6) + mean

    return _decode


def _belief_decode_fn(belief_manifold: Any):
    """Decode callable for the belief manifold (output is in Hellinger space)."""

    def _decode(u: Tensor) -> Tensor:
        return belief_manifold.decode(u)

    return _decode


def _get_manifold_device(manifold: Any) -> torch.device:
    """Get device from manifold parameters or buffers, fallback to CPU."""
    return next(
        (p.device for p in getattr(manifold, "parameters", lambda: [])()),
        next(
            (b.device for b in getattr(manifold, "buffers", lambda: [])()),
            torch.device("cpu"),
        ),
    )


# ---------------------------------------------------------------------------
# Manifold-vs-manifold isometry
# ---------------------------------------------------------------------------


def compute_isometry_from_manifolds(
    activation_manifold: Any,
    activation_mean: Tensor,
    activation_std: Tensor,
    belief_manifold: Any,
    n_arc_steps: int = 150,
    path_mode: str = "geometric",
    n_interior_per_pair: int = 0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, Tensor, Tensor]:
    """Compare the geometry of the activation side to the belief manifold.

    Both manifolds are assumed to have been fit on centroids for the same
    set of class values, so their ``control_points`` align by row index.

    Both ``path_mode``s share the **same vertex set** — manifold-derived
    points in PCA space sampled at equispaced fractions along each centroid
    pair's u-space geodesic. The modes differ only in the *distance metric*
    used to fill ``D_X``:

      - ``"geometric"``: arc length along the activation-manifold spline,
        accumulated by summing Euclidean increments between consecutive
        decoded points along the path.
      - ``"linear"``: straight-line Euclidean distance between the two
        manifold-decoded vertices (i.e. chord in the PCA subspace where the
        manifold lives).

    D_Y is always the belief manifold geodesic arc length (Hellinger).

    The off-PCA-subspace component of the raw residual stream is intentionally
    excluded. The activation manifold lives entirely in the PCA subspace by
    construction; off-subspace variation is data outside the manifold and
    isn't part of its intrinsic geometry. The PCA → residual lift is an
    isometry, so the linear chord measured in PCA equals the chord measured
    in residual stream between the lifted points; there's nothing to gain by
    computing in residual stream.

    When ``n_interior_per_pair > 0``, the vertex set is augmented with
    interior points sampled at K equispaced fractions along each centroid
    pair's geodesic (in u-space on each manifold). Each interior point
    ``(i, j, f)`` has a paired representation in activation and belief
    spaces via the same fraction f along the respective geodesic. Pairs of
    vertices whose support sets are subset-related (i.e., they lie on the
    same i↔j geodesic) are excluded — those distances are forced by
    construction and would inflate the correlation tautologically.

    Returns:
        metrics: ``{"pearson_r", "n_pairs"}``.
        D_X: (V, V) symmetric matrix of activation-side distances over the
            full vertex set (NaN on excluded entries / diagonal).
        D_Y: (V, V) symmetric matrix of belief arc lengths (Hellinger),
            same indexing.
        intrinsic_coords: (W, d_act) activation-manifold control points,
            useful as colour/hover anchors for visualisation.
    """
    act_cps = activation_manifold.control_points  # (W, d_act)
    bel_cps_raw = belief_manifold.control_points  # (W, d_bel)
    W = act_cps.shape[0]
    if W != bel_cps_raw.shape[0]:
        raise ValueError(
            f"Activation manifold has {W} centroids but belief manifold has "
            f"{bel_cps_raw.shape[0]}; cannot align by class index."
        )
    if act_cps.shape[1] != bel_cps_raw.shape[1]:
        raise ValueError(
            f"Activation control_points have dim {act_cps.shape[1]} but belief "
            f"control_points have dim {bel_cps_raw.shape[1]}; expected matching "
            "intrinsic coords (e.g. both (height, angle) in cylinder)."
        )

    # Both manifolds are constructed in class-index order (the activation side
    # via compute_centroids on the canonical task parameter; the belief side
    # via fit_belief_tps_* iterating over intervention_values). Row i in each
    # therefore refers to the same class, regardless of whether the intrinsic
    # coordinate system is "parameter" or "pca" (PCA-derived coords differ in
    # value between the two manifolds but preserve row order by construction).
    bel_cps = bel_cps_raw
    # Invariant: control_points align by row index across the two manifolds
    # by construction of the spline fit. The runtime alignment guard
    # (_align_control_points) was deleted in 2c4a997 because it does not
    # hold under intrinsic_mode: pca; a future re-introduction must gate
    # on intrinsic_mode != "pca".

    act_periodic = list(getattr(activation_manifold, "periodic_dims", []) or []) or None
    act_periods = (
        list(activation_manifold.periods)
        if hasattr(activation_manifold, "periods")
        else []
    ) or None
    bel_periodic = list(getattr(belief_manifold, "periodic_dims", []) or []) or None
    bel_periods = (
        list(belief_manifold.periods) if hasattr(belief_manifold, "periods") else []
    ) or None

    act_device = _get_manifold_device(activation_manifold)
    bel_device = _get_manifold_device(belief_manifold)
    act_mean = activation_mean.to(act_device)
    act_std = activation_std.to(act_device)

    act_decode = _activation_decode_fn(activation_manifold, act_mean, act_std)
    bel_decode = _belief_decode_fn(belief_manifold)

    act_cps_dev = act_cps.to(act_device)
    bel_cps_dev = bel_cps.to(bel_device)

    if path_mode not in ("geometric", "linear"):
        raise ValueError(
            f"Unknown path_mode: {path_mode!r}. Expected 'geometric' or 'linear'."
        )

    # Build the vertex set: W centroids first, then K interior points per
    # ordered (i < j) pair. Each vertex carries (u_act, u_bel, support), where
    # support is a frozenset of centroid indices on whose geodesic this vertex
    # lies (used to exclude tautologically-correlated same-geodesic pairs).
    K = int(n_interior_per_pair)
    # Equispaced interior fractions, exclusive of endpoints: (1/(K+1), …, K/(K+1))
    fractions = [(k + 1) / (K + 1) for k in range(K)] if K > 0 else []

    u_act_vertices: list[Tensor] = []
    u_bel_vertices: list[Tensor] = []
    supports: list[frozenset[int]] = []

    for i in range(W):
        u_act_vertices.append(act_cps_dev[i])
        u_bel_vertices.append(bel_cps_dev[i])
        supports.append(frozenset({i}))

    if K > 0:
        for i in range(W):
            for j in range(i + 1, W):
                d_act_uv = _shortest_arc_delta(
                    act_cps_dev[i],
                    act_cps_dev[j],
                    act_periodic,
                    act_periods,
                )
                d_bel_uv = _shortest_arc_delta(
                    bel_cps_dev[i],
                    bel_cps_dev[j],
                    bel_periodic,
                    bel_periods,
                )
                for f in fractions:
                    u_act_vertices.append(act_cps_dev[i] + f * d_act_uv)
                    u_bel_vertices.append(bel_cps_dev[i] + f * d_bel_uv)
                    supports.append(frozenset({i, j}))

    V = len(supports)

    # PCA-space coords for every vertex — only needed for the linear chord metric.
    pca_pts: Tensor | None = None
    if path_mode == "linear":
        with torch.no_grad():
            u_stack = torch.stack(u_act_vertices, dim=0)
            pca_pts = act_decode(u_stack)  # (V, k_pca)

    D_X = np.zeros((V, V), dtype=np.float64)
    D_Y = np.zeros((V, V), dtype=np.float64)
    same_geo_mask = np.zeros((V, V), dtype=bool)

    # Build the (V*(V-1)/2,) flat list of upper-triangle pairs once, then
    # batch-decode rather than looping per-pair (was O(V^2) decode calls;
    # now a single batched call per side, optionally chunked for memory).
    pair_a, pair_b = np.triu_indices(V, k=1)
    n_pairs = pair_a.size

    # same_geo mask: support-subset relation (cheap; do in Python).
    for a, b in zip(pair_a.tolist(), pair_b.tolist()):
        sa, sb = supports[a], supports[b]
        if (sa <= sb) or (sb <= sa):
            same_geo_mask[a, b] = same_geo_mask[b, a] = True

    if n_pairs > 0:
        u_act_stack = torch.stack(u_act_vertices, dim=0)  # (V, d_act_u)
        u_bel_stack = torch.stack(u_bel_vertices, dim=0)  # (V, d_bel_u)
        idx_a = torch.from_numpy(pair_a).to(act_device)
        idx_b = torch.from_numpy(pair_b).to(act_device)
        idx_a_bel = torch.from_numpy(pair_a).to(bel_device)
        idx_b_bel = torch.from_numpy(pair_b).to(bel_device)

        if path_mode == "geometric":
            d_act_pairs = (
                _decoded_path_length_batched(
                    u_act_stack.index_select(0, idx_a),
                    u_act_stack.index_select(0, idx_b),
                    act_decode,
                    n_steps=n_arc_steps,
                    periodic_dims=act_periodic,
                    periods=act_periods,
                )
                .detach()
                .cpu()
                .numpy()
            )
        else:  # linear
            assert pca_pts is not None
            d_act_pairs = (
                (pca_pts[idx_a] - pca_pts[idx_b]).norm(dim=-1).detach().cpu().numpy()
            )

        d_bel_pairs = _decoded_path_length_batched(
            u_bel_stack.index_select(0, idx_a_bel),
            u_bel_stack.index_select(0, idx_b_bel),
            bel_decode,
            n_steps=n_arc_steps,
            periodic_dims=bel_periodic,
            periods=bel_periods,
        ).detach().cpu().numpy() / (2.0**0.5)

        D_X[pair_a, pair_b] = d_act_pairs
        D_X[pair_b, pair_a] = d_act_pairs
        D_Y[pair_a, pair_b] = d_bel_pairs
        D_Y[pair_b, pair_a] = d_bel_pairs

    # Correlation excludes same-geodesic pairs (their distances are forced
    # by construction). Saved D matrices stay complete so downstream MDS /
    # visualisation has a valid distance matrix.
    iu = np.triu_indices(V, k=1)
    keep = ~same_geo_mask[iu]
    metrics = compute_isometry_metrics(D_X[iu][keep], D_Y[iu][keep])
    metrics["n_centroids"] = W
    metrics["n_interior_per_pair"] = K
    metrics["n_vertices"] = V
    metrics["n_excluded_same_geodesic"] = int((~keep).sum())
    logger.info(
        "Isometry [%s]: r=%.4f over %d pairs "
        "(V=%d vertices, K=%d interior/pair, excluded=%d, n_arc_steps=%d)",
        path_mode,
        metrics["pearson_r"],
        metrics["n_pairs"],
        V,
        K,
        metrics["n_excluded_same_geodesic"],
        n_arc_steps,
    )
    # Stack all V vertex coords (both spaces) for downstream visualisation.
    # First W rows are centroids; the remainder are interior points (i, j, f)
    # ordered by (i, j) lexicographically with f varying fastest. Returning
    # both lets the MDS plot color by belief-manifold u-coords (the canonical
    # ordering of the conceptual domain) on both panels.
    vertex_coords = torch.stack([v.detach().cpu() for v in u_act_vertices], dim=0)
    vertex_coords_belief = torch.stack(
        [v.detach().cpu() for v in u_bel_vertices], dim=0
    )
    return metrics, D_X, D_Y, vertex_coords, vertex_coords_belief


# ---------------------------------------------------------------------------
# Saving artifacts
# ---------------------------------------------------------------------------


def _save_isometry_artifacts(
    metrics: dict[str, Any],
    D_manifold: np.ndarray,
    D_output: np.ndarray,
    grid_points: torch.Tensor | None,
    output_dir: str,
    metadata: dict[str, Any] | None = None,
    grid_points_belief: torch.Tensor | None = None,
) -> None:
    """Save isometry computation artifacts for downstream visualization."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if metadata is not None:
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    from safetensors.torch import save_file

    tensors = {
        "D_manifold": torch.from_numpy(np.nan_to_num(D_manifold, nan=0.0)).float(),
        "D_output": torch.from_numpy(np.nan_to_num(D_output, nan=0.0)).float(),
    }
    if grid_points is not None:
        tensors["grid_points_valid"] = grid_points.cpu().float()
    if grid_points_belief is not None:
        tensors["grid_points_valid_belief"] = grid_points_belief.cpu().float()
    save_file(tensors, os.path.join(output_dir, "tensors.safetensors"))
    logger.info("Saved isometry artifacts to %s", output_dir)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _default_grid_hover(grid_points: np.ndarray) -> list[str]:
    d = grid_points.shape[1]
    return [
        "centroid " + ", ".join(f"u{k}={grid_points[i, k]:.3f}" for k in range(d))
        for i in range(len(grid_points))
    ]


def _isometry_subtitle(metrics: dict[str, Any]) -> str:
    return f"r = {metrics['pearson_r']:.3f} | n = {metrics['n_pairs']}"


def _plot_isometry_scatter(
    D_X: np.ndarray,
    D_Y: np.ndarray,
    metrics: dict[str, Any],
    output_dir: str,
    distance_function: str = "hellinger",
    figure_format: str = "pdf",
) -> None:
    from causalab.io.plots.distance_plots import plot_distance_scatter

    metric_label = _DISTANCE_LABELS.get(distance_function, distance_function)
    plot_distance_scatter(
        D_X,
        D_Y,
        output_path=os.path.join(output_dir, "isometry_scatter.png"),
        figure_format=figure_format,
        x_label="Activation manifold path length",
        y_label=f"Output manifold path length ({metric_label})",
        title="Isometry: activation vs output manifold",
        annotations={
            "Pearson r": metrics["pearson_r"],
            "N pairs": metrics["n_pairs"],
        },
    )


def _mpl_to_plotly_colorscale(
    name: str, n_samples: int = 11
) -> list[list[float | str]]:
    """Sample a matplotlib colormap to a Plotly-compatible colorscale.

    Plotly accepts a list of ``[position, "rgb(r,g,b)"]`` pairs, which lets
    us pass matplotlib-only colormaps (e.g. ``managua``, ``twilight_shifted``)
    through unchanged.
    """
    import matplotlib

    cmap = matplotlib.colormaps[name]
    positions = np.linspace(0.0, 1.0, n_samples)
    colorscale: list[list[float | str]] = []
    for p in positions:
        r, g, b, _ = cmap(p)
        colorscale.append(
            [float(p), f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"]
        )
    return colorscale


def _plot_isometry_mds(
    D_manifold: np.ndarray,
    D_output: np.ndarray,
    grid_points: np.ndarray,
    metrics: dict[str, Any],
    output_dir: str,
    hover_labels: list[str] | None = None,
    distance_function: str = "hellinger",
    n_components: int = 3,
    colormap: str = "Viridis",
    color_grid_points: np.ndarray | None = None,
    edges: list[tuple[int, int]] | None = None,
) -> None:
    from causalab.io.plots.distance_plots import plot_dual_mds

    metric_label = _DISTANCE_LABELS.get(distance_function, distance_function)
    kwargs = {"width": 1100, "height": 500} if n_components == 2 else {}
    try:
        colorscale = _mpl_to_plotly_colorscale(colormap)
    except (KeyError, ValueError) as exc:
        logger.debug(
            "Colormap %r not in matplotlib, falling back to plotly built-in: %s",
            colormap,
            exc,
        )
        colorscale = colormap  # fall back to Plotly built-in name
    n_centroids = int(metrics.get("n_centroids", grid_points.shape[0]))
    # Color BOTH MDS panels by the output (belief) manifold's u-coords when
    # available. Falls back to activation u-coords for legacy artifacts that
    # only saved one grid.
    color_source = color_grid_points if color_grid_points is not None else grid_points
    colorbar_title = "u0 (output)" if color_grid_points is not None else "u0"
    plot_dual_mds(
        D_manifold,
        D_output,
        output_path=os.path.join(output_dir, "isometry_mds.html"),
        color_values=color_source[:, 0],
        left_title="Activation manifold (MDS)",
        right_title=f"Output manifold — {metric_label} (MDS)",
        title=f"Isometry: activation vs output manifold ({metric_label})",
        subtitle=_isometry_subtitle(metrics),
        hover_labels=hover_labels or _default_grid_hover(grid_points),
        colorbar_title=colorbar_title,
        colorscale=colorscale,
        n_mds_components=n_components,
        n_centroids=n_centroids,
        edges=edges,
        **kwargs,
    )


def _build_variable_value_hover(
    grid_points: np.ndarray,
    variable_values: list[str],
    grid_range: list[float],
) -> list[str]:
    """Map grid u0 coordinates to nearest variable value names for hover labels."""
    n_values = len(variable_values)
    lo, hi = grid_range
    labels = []
    for i in range(len(grid_points)):
        u0 = grid_points[i, 0]
        frac = (u0 - lo) / (hi - lo) if hi > lo else 0.0
        idx = max(0, min(n_values - 1, int(round(frac * (n_values - 1)))))
        nearest = variable_values[idx]
        d = grid_points.shape[1]
        coords = ", ".join(f"u{k}={grid_points[i, k]:.3f}" for k in range(d))
        labels.append(f"{nearest} ({coords})")
    return labels


def visualize_isometry(
    artifact_dir: str,
    viz_cfg: DictConfig,
    distance_function: str,
    variable_values: list[str] | None = None,
    grid_range: list[float] | None = None,
    output_dir: str | None = None,
    colormap: str = "Viridis",
    edges: list[tuple[int, int]] | None = None,
) -> dict[str, str]:
    """Load isometry artifacts from disk and produce plots."""
    from safetensors.torch import load_file

    tensors = load_file(os.path.join(artifact_dir, "tensors.safetensors"))
    with open(os.path.join(artifact_dir, "metrics.json")) as f:
        metrics = json.load(f)
    metadata: dict[str, Any] = {}
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    D_manifold = tensors["D_manifold"].numpy()
    D_output = tensors["D_output"].numpy()
    grid_points = tensors["grid_points_valid"].numpy()
    grid_points_belief = (
        tensors["grid_points_valid_belief"].numpy()
        if "grid_points_valid_belief" in tensors
        else None
    )

    # For periodic intrinsic dims, wrap u-coords into [0, period) so cyclic
    # order is preserved when mapped to a colorscale (otherwise interior
    # points whose shortest-arc interpolation went the negative direction
    # appear as outliers below cmin). Activation and belief manifolds may
    # have different periodic dims/periods, so wrap each by its own metadata.
    periodic_dims = metadata.get("periodic_dims") or []
    periods = metadata.get("periods") or []
    for dim, period in zip(periodic_dims, periods):
        if 0 <= dim < grid_points.shape[1] and period > 0:
            grid_points[:, dim] = np.mod(grid_points[:, dim], period)
    if grid_points_belief is not None:
        bel_periodic_dims = metadata.get("belief_periodic_dims") or []
        bel_periods = metadata.get("belief_periods") or []
        for dim, period in zip(bel_periodic_dims, bel_periods):
            if 0 <= dim < grid_points_belief.shape[1] and period > 0:
                grid_points_belief[:, dim] = np.mod(grid_points_belief[:, dim], period)

    hover_format = viz_cfg.get("hover_label_format", "grid_coords")
    if (
        hover_format == "variable_values"
        and variable_values is not None
        and grid_range is not None
    ):
        hover_labels = _build_variable_value_hover(
            grid_points, variable_values, grid_range
        )
    else:
        hover_labels = _default_grid_hover(grid_points)

    plot_dir = output_dir or artifact_dir
    os.makedirs(plot_dir, exist_ok=True)

    _plot_isometry_scatter(
        D_manifold,
        D_output,
        metrics,
        plot_dir,
        distance_function=distance_function,
        figure_format=viz_cfg.get("figure_format", "pdf"),
    )
    n_mds = viz_cfg.get("n_mds_components", 3)
    _plot_isometry_mds(
        D_manifold,
        D_output,
        grid_points,
        metrics,
        plot_dir,
        hover_labels=hover_labels,
        distance_function=distance_function,
        n_components=n_mds,
        colormap=colormap,
        color_grid_points=grid_points_belief,
        edges=edges,
    )

    return {
        "scatter": os.path.join(artifact_dir, "isometry_scatter.png"),
        "mds": os.path.join(artifact_dir, "isometry_mds.html"),
    }
