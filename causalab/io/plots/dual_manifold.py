"""Dual Manifold Viewer — Activation ↔ Belief Correspondence.

Generates a self-contained HTML file with two 3D plots side-by-side:
- Activation space: PCA-projected neural activations with manifold paths
- Belief space: Hellinger PCA-projected output distributions

Interactive: select a centroid pair, then drag a slider to move along that
path in both spaces simultaneously.

Generated as part of the evaluate stage visualization pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from torch import Tensor

from causalab.io.plots.plot_3d_interactive import (
    _fit_projector,
    _resolve_categorical_colors,
    _build_manifold_mesh_traces,
    _make_edge_traces,
)
from causalab.analyses.path_steering.path_visualization import (
    _is_2d_spatial,
    _build_grid_layout,
    _build_rc_to_w,
    _get_grid_row_colors,
)
from causalab.io.plots.plot_utils import FigureGenerator
from causalab.methods.spline.manifold import SplineManifold

logger = logging.getLogger(__name__)

_CENTROID_SIZE = 8
_POINT_SIZE = 10
_BG_OPACITY = 0.15
_MANIFOLD_COLOR = "darkgray"
_FONT_FAMILY = "Avenir, Avenir Next, Helvetica Neue, sans-serif"

# ─────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────


@dataclass
class DualManifoldData:
    """All data needed to render both plots."""

    # Per-pair path data (geometric = follows manifold, linear = straight in PCA)
    pairs: list[tuple[int, int]]
    geo_act_paths_3d: list[np.ndarray]  # per pair: (n_steps, 3)
    geo_bel_paths_3d: list[np.ndarray]  # per pair: (n_steps, 3)
    geo_dists: list[np.ndarray]  # per pair: (n_steps, W) mean distributions

    # Centroids
    act_centroids_3d: np.ndarray  # (n_centroids, 3)
    bel_centroids_3d: np.ndarray  # (n_centroids, 3)

    # Feature scatter (activation space background)
    act_features_3d: np.ndarray  # (n_train, 3)
    act_feature_classes: np.ndarray  # (n_train,) class assignments

    # Belief background scatter
    bel_background_3d: np.ndarray  # (n_examples, 3)
    bel_class_assignments: np.ndarray  # (n_examples,)

    # Fields with defaults (must come after non-default fields)
    lin_act_paths_3d: list[np.ndarray] | None = None
    lin_bel_paths_3d: list[np.ndarray] | None = None
    lin_dists: list[np.ndarray] | None = None
    act_mesh_traces: list = field(default_factory=list)
    act_edges: list | None = None  # (i, j) centroid index pairs for graph edges
    bel_tps_curve_3d: np.ndarray | None = None
    n_classes: int = 0
    class_labels: list[str] = field(default_factory=list)
    output_token_values: list | None = None
    n_normal_steps: int | None = (
        None  # steps before oversteering begins; None = no oversteer
    )

    @classmethod
    def from_evaluate_artifacts(
        cls,
        geo_grid_points: Tensor,  # (n_pairs, num_steps, d) intrinsic
        geo_distributions: Tensor,  # (n_pairs, num_steps, n_prompts, W)
        lin_grid_points: Tensor | None,  # (n_pairs, num_steps, D) raw activation space
        lin_distributions: Tensor | None,  # (n_pairs, num_steps, n_prompts, W)
        pairs: list[tuple[int, int]],
        manifold_obj,  # SplineManifold
        feat_mean: Tensor,
        feat_std: Tensor,
        pca_features: Tensor,  # (N, k) for PCA 3D fitting
        subspace_featurizer: torch.nn.Module | None,  # D → k projection
        feature_classes: np.ndarray,  # (N,) class index per training example
        belief_manifold,  # belief manifold TPS (or None)
        hellinger_pca,  # sklearn PCA on sqrt-probabilities
        natural_dists: Tensor,  # (N, W+1) for background scatter
        class_labels: list[str],
        output_token_values: list | None = None,
        edges: list | None = None,
        n_normal_steps: int | None = None,
        bel_class_assignments_true: np.ndarray = None,  # TRUE labels for natural_dists rows (required)
    ) -> "DualManifoldData":
        """Build from evaluate stage artifacts (no disk I/O)."""
        if bel_class_assignments_true is None:
            raise ValueError(
                "bel_class_assignments_true is required: pass an array of true "
                "class indices row-aligned with natural_dists."
            )
        if len(bel_class_assignments_true) < natural_dists.shape[0]:
            raise ValueError(
                f"bel_class_assignments_true length ({len(bel_class_assignments_true)}) "
                f"is shorter than natural_dists rows ({natural_dists.shape[0]})."
            )
        n_classes = len(class_labels)
        n_pairs = geo_grid_points.shape[0]
        n_steps = geo_grid_points.shape[1]

        device = manifold_obj.control_points.device
        feat_mean_d = feat_mean.to(device)
        feat_std_d = feat_std.to(device)

        # PCA 3D projection (fit once on training features)
        _, act_project_fn = _fit_projector(pca_features, n_components=3)

        _pca_dim = hellinger_pca.n_features_in_

        def _to_hellinger_3d(dists_2d: np.ndarray) -> np.ndarray:
            W = dists_2d.shape[-1]
            if _pca_dim == W + 1:
                other = np.clip(1.0 - dists_2d.sum(axis=-1, keepdims=True), 0, None)
                full = np.concatenate([dists_2d, other], axis=-1)
            else:
                full = dists_2d
            return hellinger_pca.transform(
                np.sqrt(np.clip(full, 0, None)).astype(np.float32)
            )

        # --- Geometric paths (intrinsic → manifold decode → PCA 3D) ---
        all_intrinsic = geo_grid_points.reshape(-1, geo_grid_points.shape[-1]).to(
            device
        )
        with torch.no_grad():
            decoded = manifold_obj.decode(all_intrinsic, r=None)
            decoded_k = decoded * (feat_std_d + 1e-6) + feat_mean_d
        all_geo_act_3d = act_project_fn(decoded_k.detach().cpu().float().numpy())
        geo_act_paths_3d = [
            all_geo_act_3d[i * n_steps : (i + 1) * n_steps] for i in range(n_pairs)
        ]

        geo_mean = geo_distributions.mean(dim=2)  # (n_pairs, n_steps, W)
        geo_bel_paths_3d = [
            _to_hellinger_3d(geo_mean[i].numpy()) for i in range(n_pairs)
        ]
        geo_dists = [geo_mean[i].numpy() for i in range(n_pairs)]

        # --- Linear paths (raw D-dim → subspace featurizer → k-dim → 3D) ---
        lin_act_paths_3d = None
        lin_bel_paths_3d = None
        lin_dists = None
        if (
            lin_grid_points is not None
            and lin_distributions is not None
            and subspace_featurizer is not None
        ):
            _lin_device = next(
                (
                    p.device
                    for p in [
                        *subspace_featurizer.parameters(),
                        *subspace_featurizer.buffers(),
                    ]
                ),
                torch.device("cpu"),
            )
            all_lin_raw = lin_grid_points.reshape(-1, lin_grid_points.shape[-1]).to(
                _lin_device
            )
            with torch.no_grad():
                lin_k, _ = subspace_featurizer(all_lin_raw)  # (P, k)
            all_lin_act_3d = act_project_fn(lin_k.detach().cpu().float().numpy())
            lin_n_steps = lin_grid_points.shape[1]
            lin_act_paths_3d = [
                all_lin_act_3d[i * lin_n_steps : (i + 1) * lin_n_steps]
                for i in range(n_pairs)
            ]

            lin_mean = lin_distributions.mean(dim=2)
            lin_bel_paths_3d = [
                _to_hellinger_3d(lin_mean[i].numpy()) for i in range(n_pairs)
            ]
            lin_dists = [lin_mean[i].numpy() for i in range(n_pairs)]

        # --- Centroids (compute from features, not manifold.centroids, to match class ordering) ---
        act_centroids_k = torch.zeros(n_classes, pca_features.shape[1])
        _counts = torch.zeros(n_classes)
        for i in range(len(feature_classes)):
            ci = feature_classes[i]
            act_centroids_k[ci] += pca_features[i]
            _counts[ci] += 1
        for ci in range(n_classes):
            if _counts[ci] > 0:
                act_centroids_k[ci] /= _counts[ci]
        act_centroids_3d = act_project_fn(
            act_centroids_k.detach().cpu().float().numpy()
        )

        # Activation manifold mesh/curve traces + wireframe
        from causalab.analyses.activation_manifold.utils import (
            _compute_intrinsic_ranges,
        )

        ranges = _compute_intrinsic_ranges(
            pca_features, manifold_obj, feat_mean, feat_std
        )
        act_mesh_traces = _build_manifold_mesh_traces(
            manifold_obj,
            feat_mean,
            feat_std,
            ranges,
            act_project_fn,
            grid_res=30,
        )

        # Training features projected to 3D
        act_features_3d = act_project_fn(pca_features.detach().cpu().float().numpy())

        # Belief centroids from geometric path endpoints
        endpoint_dists: dict[int, list[np.ndarray]] = {}
        for pi, (si, ei) in enumerate(pairs):
            endpoint_dists.setdefault(si, []).append(geo_mean[pi, 0].numpy())
            endpoint_dists.setdefault(ei, []).append(geo_mean[pi, -1].numpy())
        bel_centroids_raw = []
        for ci in range(n_classes):
            if ci in endpoint_dists:
                bel_centroids_raw.append(np.mean(endpoint_dists[ci], axis=0))
            else:
                bel_centroids_raw.append(
                    np.ones(geo_mean.shape[-1]) / geo_mean.shape[-1]
                )
        bel_centroids_3d = _to_hellinger_3d(np.array(bel_centroids_raw))

        # Belief manifold TPS reference curve
        bel_tps_curve_3d = None
        if belief_manifold is not None:
            bel_tps_curve = _decode_manifold_curve(
                belief_manifold,
                torch.zeros(belief_manifold._ambient_dim),
                torch.ones(belief_manifold._ambient_dim),
                n_points=200,
                start_at=belief_manifold.control_points[0, 0].item(),
            )
            bel_tps_curve_3d = hellinger_pca.transform(bel_tps_curve.astype(np.float32))

        # Background scatter — match PCA expected dimensionality
        nat_np = natural_dists.clamp(min=0).float().numpy()
        if nat_np.shape[-1] != _pca_dim:
            nat_np = nat_np[:, :_pca_dim]
        bel_background_3d = hellinger_pca.transform(np.sqrt(nat_np).astype(np.float32))
        bel_class_assignments = np.asarray(
            bel_class_assignments_true[: natural_dists.shape[0]], dtype=int
        )

        return cls(
            pairs=pairs,
            geo_act_paths_3d=geo_act_paths_3d,
            geo_bel_paths_3d=geo_bel_paths_3d,
            geo_dists=geo_dists,
            lin_act_paths_3d=lin_act_paths_3d,
            lin_bel_paths_3d=lin_bel_paths_3d,
            lin_dists=lin_dists,
            act_mesh_traces=act_mesh_traces,
            act_edges=edges,
            bel_tps_curve_3d=bel_tps_curve_3d,
            act_centroids_3d=act_centroids_3d,
            bel_centroids_3d=bel_centroids_3d,
            act_features_3d=act_features_3d,
            act_feature_classes=feature_classes,
            bel_background_3d=bel_background_3d,
            bel_class_assignments=bel_class_assignments,
            n_classes=n_classes,
            class_labels=class_labels,
            output_token_values=output_token_values,
            n_normal_steps=n_normal_steps,
        )

    @classmethod
    def from_pullback_artifacts(
        cls,
        pullback_results: dict,  # {(ci, cj): result_dict} from alpha_inf
        manifold_obj,
        feat_mean: Tensor,
        feat_std: Tensor,
        pca_features: Tensor,
        feature_classes: np.ndarray,
        belief_manifold,
        hellinger_pca,
        natural_dists: Tensor,
        class_labels: list[str],
        output_token_values: list | None = None,
        edges: list | None = None,
        bel_class_assignments_true: np.ndarray = None,  # TRUE labels for natural_dists rows (required)
    ) -> "DualManifoldData":
        """Build from pullback stage alpha=inf optimization results.

        In pullback, the primary motion is in belief space (geodesic target),
        and we observe the corresponding activation-space embedding paths.
        We map:
          - geo paths = optimized embedding paths (the pullback result)
          - lin paths = geometric embedding paths (manifold shortest-arc)
        """
        if bel_class_assignments_true is None:
            raise ValueError(
                "bel_class_assignments_true is required: pass an array of true "
                "class indices row-aligned with natural_dists."
            )
        if len(bel_class_assignments_true) < natural_dists.shape[0]:
            raise ValueError(
                f"bel_class_assignments_true length ({len(bel_class_assignments_true)}) "
                f"is shorter than natural_dists rows ({natural_dists.shape[0]})."
            )
        n_classes = len(class_labels)
        device = manifold_obj.control_points.device
        feat_mean_d = feat_mean.to(device)
        feat_std_d = feat_std.to(device)

        _, act_project_fn = _fit_projector(pca_features, n_components=3)
        _pca_dim = hellinger_pca.n_features_in_

        def _to_hellinger_3d(dists_2d: np.ndarray) -> np.ndarray:
            d = dists_2d
            if d.shape[-1] != _pca_dim:
                if _pca_dim == d.shape[-1] + 1:
                    other = np.clip(1.0 - d.sum(axis=-1, keepdims=True), 0, None)
                    d = np.concatenate([d, other], axis=-1)
                else:
                    d = d[:, :_pca_dim]
            return hellinger_pca.transform(
                np.sqrt(np.clip(d, 0, None)).astype(np.float32)
            )

        pairs = []
        geo_act_paths_3d = []  # optimized embedding paths
        geo_bel_paths_3d = []  # optimized belief paths
        geo_dists = []
        lin_act_paths_3d = []  # geometric embedding paths
        lin_bel_paths_3d = []  # geometric belief paths
        lin_dists = []

        for (ci, cj), result in sorted(pullback_results.items()):
            if ci >= cj:
                continue  # skip self-pairs and reverse duplicates
            pairs.append((ci, cj))

            # Optimized path → "geo" slot (primary)
            v_opt = result["v_optimized_k"]  # (n_steps, k)
            opt_act_3d = act_project_fn(v_opt.cpu().float().numpy())
            geo_act_paths_3d.append(opt_act_3d)

            opt_bel = result["opt_probs_AW1"].cpu().float().numpy()  # (n_steps, W+1)
            geo_bel_paths_3d.append(_to_hellinger_3d(opt_bel))
            geo_dists.append(opt_bel[:, :n_classes])  # strip "other" for bar/line chart

            # Geometric path → "lin" slot (comparison)
            v_geo = result["v_geometric_k"]
            geo_act_3d = act_project_fn(v_geo.cpu().float().numpy())
            lin_act_paths_3d.append(geo_act_3d)

            geo_bel = result["geo_probs_AW1"].cpu().float().numpy()
            lin_bel_paths_3d.append(_to_hellinger_3d(geo_bel))
            lin_dists.append(geo_bel[:, :n_classes])

        # Centroids
        act_centroids_k = manifold_obj.centroids * (feat_std_d + 1e-6) + feat_mean_d
        act_centroids_3d = act_project_fn(
            act_centroids_k.detach().cpu().float().numpy()
        )

        # Belief centroids from endpoint distributions
        bel_centroids_raw = []
        endpoint_dists: dict[int, list[np.ndarray]] = {}
        for pi, (si, ei) in enumerate(pairs):
            endpoint_dists.setdefault(si, []).append(geo_dists[pi][0])
            endpoint_dists.setdefault(ei, []).append(geo_dists[pi][-1])
        for ci in range(n_classes):
            if ci in endpoint_dists:
                bel_centroids_raw.append(np.mean(endpoint_dists[ci], axis=0))
            else:
                bel_centroids_raw.append(np.ones(n_classes) / n_classes)
        bel_centroids_3d = _to_hellinger_3d(np.array(bel_centroids_raw))

        # Manifold mesh
        from causalab.analyses.activation_manifold.utils import (
            _compute_intrinsic_ranges,
        )

        ranges = _compute_intrinsic_ranges(
            pca_features, manifold_obj, feat_mean, feat_std
        )
        act_mesh_traces = _build_manifold_mesh_traces(
            manifold_obj,
            feat_mean,
            feat_std,
            ranges,
            act_project_fn,
            grid_res=30,
        )

        # Features
        act_features_3d = act_project_fn(pca_features.detach().cpu().float().numpy())

        # Belief TPS curve
        bel_tps_curve_3d = None
        if belief_manifold is not None:
            bel_tps_curve = _decode_manifold_curve(
                belief_manifold,
                torch.zeros(belief_manifold._ambient_dim),
                torch.ones(belief_manifold._ambient_dim),
                n_points=200,
                start_at=belief_manifold.control_points[0, 0].item(),
            )
            bel_tps_curve_3d = hellinger_pca.transform(bel_tps_curve.astype(np.float32))

        # Background scatter
        nat_np = natural_dists.clamp(min=0).float().numpy()
        if nat_np.shape[-1] != _pca_dim:
            nat_np = nat_np[:, :_pca_dim]
        bel_background_3d = hellinger_pca.transform(np.sqrt(nat_np).astype(np.float32))
        bel_class_assignments = np.asarray(
            bel_class_assignments_true[: natural_dists.shape[0]], dtype=int
        )

        return cls(
            pairs=pairs,
            geo_act_paths_3d=geo_act_paths_3d,
            geo_bel_paths_3d=geo_bel_paths_3d,
            geo_dists=geo_dists,
            act_centroids_3d=act_centroids_3d,
            bel_centroids_3d=bel_centroids_3d,
            act_features_3d=act_features_3d,
            act_feature_classes=feature_classes,
            bel_background_3d=bel_background_3d,
            bel_class_assignments=bel_class_assignments,
            lin_act_paths_3d=lin_act_paths_3d,
            lin_bel_paths_3d=lin_bel_paths_3d,
            lin_dists=lin_dists,
            act_mesh_traces=act_mesh_traces,
            act_edges=edges,
            bel_tps_curve_3d=bel_tps_curve_3d,
            n_classes=n_classes,
            class_labels=class_labels,
            output_token_values=output_token_values,
        )


# ─────────────────────────────────────────────────────────────────────
# Manifold curve decoding (for TPS reference)
# ─────────────────────────────────────────────────────────────────────


def _decode_manifold_curve(
    manifold: SplineManifold,
    mean: Tensor,
    std: Tensor,
    n_points: int = 200,
    start_at: float | None = None,
) -> np.ndarray:
    """Decode a dense curve along the manifold, with periodic closure."""
    d = manifold.intrinsic_dim
    periodic_dims = set(manifold.periodic_dims) if manifold.periodic_dims else set()

    if d == 1:
        periodic = 0 in periodic_dims
        if periodic and manifold.periods:
            period = manifold.periods[0]
            lo = start_at if start_at is not None else 0.0
            hi = lo + period
        else:
            cp = manifold.control_points
            lo = cp[:, 0].min().item()
            hi = cp[:, 0].max().item()

        u = np.linspace(lo, hi, n_points, endpoint=not periodic)
        t = torch.tensor(u[:, None], dtype=torch.float32)
    else:
        t = manifold.control_points

    device = manifold.control_points.device
    t = t.to(device)
    mean = mean.to(device)
    std = std.to(device)
    with torch.no_grad():
        decoded = manifold.decode(t)
        decoded = decoded * (std + 1e-6) + mean

    points = decoded.cpu().float().numpy()

    if d == 1 and 0 in periodic_dims:
        points = np.concatenate([points, points[:1]], axis=0)

    return points


# ─────────────────────────────────────────────────────────────────────
# Figure building
# ─────────────────────────────────────────────────────────────────────


def _add_scatter_and_manifold(
    fig: go.Figure,
    scatter_3d: np.ndarray,
    scatter_classes: np.ndarray,
    mesh_traces: list | None,
    centroids_3d: np.ndarray,
    data: DualManifoldData,
    colors: list[str],
    first_centroid_trace_idx: list[int],
    edges: list | None = None,
) -> None:
    """Add background scatter, manifold mesh/curve, centroids, and edges."""
    # Background scatter (low opacity)
    for c in range(data.n_classes):
        mask = scatter_classes == c
        if not mask.any():
            continue
        pts = scatter_3d[mask]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=2, color=colors[c], opacity=_BG_OPACITY),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Manifold mesh/curve traces
    if mesh_traces:
        for trace in mesh_traces:
            fig.add_trace(trace)

    # Centroids — each is its own trace so click events identify which one
    first_centroid_trace_idx.append(len(fig.data))
    show_labels = data.n_classes <= 30
    for i in range(data.n_classes):
        fig.add_trace(
            go.Scatter3d(
                x=[centroids_3d[i, 0]],
                y=[centroids_3d[i, 1]],
                z=[centroids_3d[i, 2]],
                mode="markers+text" if show_labels else "markers",
                marker=dict(size=_CENTROID_SIZE, color=colors[i], symbol="diamond"),
                text=[data.class_labels[i]] if show_labels else None,
                textposition="top center" if show_labels else None,
                textfont=dict(size=10, color=colors[i], family=_FONT_FAMILY)
                if show_labels
                else None,
                showlegend=False,
                hovertemplate=f"<b>{data.class_labels[i]}</b><extra></extra>",
                customdata=[i],
            )
        )

    # Graph edges between centroids
    if edges:
        for trace in _make_edge_traces(centroids_3d, edges):
            fig.add_trace(trace)


def _build_figure(
    data: DualManifoldData,
    scatter_3d: np.ndarray,
    scatter_classes: np.ndarray,
    mesh_traces: list | None,
    centroids_3d: np.ndarray,
    geo_first: np.ndarray,
    lin_first: np.ndarray | None,
    colors: list[str],
    has_linear: bool,
    edges: list | None = None,
) -> tuple[go.Figure, int, int]:
    """Build a styled 3D figure.

    Returns (fig, geo_point_trace_idx, lin_point_trace_idx).
    lin_point_trace_idx == -1 if no linear paths.
    """
    fig = go.Figure()
    first_centroid_idx_holder: list[int] = []

    _add_scatter_and_manifold(
        fig,
        scatter_3d,
        scatter_classes,
        mesh_traces,
        centroids_3d,
        data,
        colors,
        first_centroid_idx_holder,
        edges=edges,
    )

    # Style static traces + add shadows
    fg = FigureGenerator()
    fg.style_plotly_3d(fig, floor_shadow=False)
    z_floor = fg._get_z_floor(fig)
    if z_floor is not None:
        fg._add_floor_shadows(fig, z_floor=z_floor)

    # Geometric path line (black, dashed)
    fig.add_trace(
        go.Scatter3d(
            x=geo_first[:, 0],
            y=geo_first[:, 1],
            z=geo_first[:, 2],
            mode="lines",
            line=dict(color="black", width=5, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    geo_line_idx = len(fig.data) - 1

    # Linear path line (gray, dashed)
    lin_line_idx = -1
    if has_linear and lin_first is not None:
        fig.add_trace(
            go.Scatter3d(
                x=lin_first[:, 0],
                y=lin_first[:, 1],
                z=lin_first[:, 2],
                mode="lines",
                line=dict(color="#999", width=5, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        lin_line_idx = len(fig.data) - 1

    # Geometric marker (square, color interpolated by JS)
    fig.add_trace(
        go.Scatter3d(
            x=[geo_first[0, 0]],
            y=[geo_first[0, 1]],
            z=[geo_first[0, 2]],
            mode="markers",
            marker=dict(
                size=_POINT_SIZE,
                color=colors[0],
                symbol="square",
                line=dict(color="black", width=1),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    geo_pt_idx = len(fig.data) - 1

    # Linear marker (circle, smaller, color interpolated by JS)
    lin_pt_idx = -1
    if has_linear and lin_first is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[lin_first[0, 0]],
                y=[lin_first[0, 1]],
                z=[lin_first[0, 2]],
                mode="markers",
                marker=dict(
                    size=_POINT_SIZE - 2,
                    color=colors[0],
                    symbol="circle",
                    line=dict(color="#666", width=1),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        lin_pt_idx = len(fig.data) - 1

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family=_FONT_FAMILY),
    )

    return fig, geo_line_idx, lin_line_idx, geo_pt_idx, lin_pt_idx


# ─────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────


def _build_grid_flow_html(
    output_token_values: list,
    prefix: str,
    title: str,
    colormap: str = "managua",
    color_by_dim: int = 1,
) -> tuple[str, dict]:
    """Build HTML for an interactive grid flow panel.

    Returns (html_str, grid_info) where grid_info has keys needed by JS:
    n_rows, n_cols, rc_to_w (as "r,c" -> w string keys).
    """
    n_rows, n_cols, row_labels, col_labels, class_to_rc = _build_grid_layout(
        output_token_values,
    )
    rc_to_w = _build_rc_to_w(output_token_values, class_to_rc)
    n_colors = n_cols if color_by_dim == 1 else n_rows
    cell_colors_rgb = _get_grid_row_colors(n_colors, colormap)

    # Convert RGB tuples to hex
    def _rgb_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )

    cell_size = max(18, min(36, 280 // max(n_rows, n_cols)))
    lbl_style = (
        f"font-size:9px;color:#999;font-family:{_FONT_FAMILY};text-align:center;"
    )

    # Column labels header row
    col_hdr = f'<div style="display:flex;margin-left:{cell_size}px;">'
    for c in range(n_cols):
        col_hdr += f'<div style="width:{cell_size}px;{lbl_style}">{col_labels[c]}</div>'
    col_hdr += "</div>"

    rows_html = []
    for r in range(n_rows):
        # Row label + cells
        row_lbl = (
            f'<div style="width:{cell_size}px;height:{cell_size}px;'
            f'display:flex;align-items:center;justify-content:center;{lbl_style}">'
            f"{row_labels[r]}</div>"
        )
        cells = []
        for c in range(n_cols):
            color = _rgb_hex(cell_colors_rgb[c if color_by_dim == 1 else r])
            cid = f"{prefix}-{r}-{c}"
            cells.append(
                f'<div style="width:{cell_size}px;height:{cell_size}px;'
                f'border:0.5px solid #999;background:white;">'
                f'<div id="{cid}" style="width:100%;height:100%;'
                f'background:{color};opacity:0;"></div></div>'
            )
        rows_html.append(
            '<div style="display:flex;">' + row_lbl + "".join(cells) + "</div>"
        )

    html = (
        f'<div style="text-align:center;margin:4px 0;">'
        f'<div style="font-size:13px;color:#777;font-family:{_FONT_FAMILY};'
        f'margin-bottom:2px;font-weight:500;">{title}</div>'
        f'<div style="display:inline-block;">'
        + col_hdr
        + "".join(rows_html)
        + "</div></div>"
    )

    # Build JS-friendly rc_to_w mapping
    rc_map = {f"{r},{c}": w for (r, c), w in rc_to_w.items()}

    return html, {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "rc_to_w": rc_map,
    }


def save_dual_manifold_html(
    data: DualManifoldData,
    output_path: str,
    colormap: str | None = None,
    dist_mode: str = "lines",  # "lines", "bars", "grid_flow", "grid_flow_dual"
    color_by_dim: int = 1,
) -> None:
    """Render dual manifold viewer as a self-contained HTML file."""
    is_grid_mode = dist_mode in ("grid_flow", "grid_flow_dual")

    # Per-class colors: for 2D spatial tasks, color by the chosen dimension
    if (
        is_grid_mode
        and data.output_token_values
        and _is_2d_spatial(data.output_token_values)
    ):
        n_rows, n_cols, _, _, class_to_rc = _build_grid_layout(data.output_token_values)
        n_color_bins = n_cols if color_by_dim == 1 else n_rows
        bin_colors = _resolve_categorical_colors(n_color_bins, colormap)
        colors = []
        for ci in range(data.n_classes):
            r, c = class_to_rc.get(ci, (0, 0))
            colors.append(bin_colors[c if color_by_dim == 1 else r])
    else:
        colors = _resolve_categorical_colors(data.n_classes, colormap)
    has_linear = data.lin_act_paths_3d is not None

    geo_first_act = (
        data.geo_act_paths_3d[0] if data.geo_act_paths_3d else np.zeros((1, 3))
    )
    lin_first_act = data.lin_act_paths_3d[0] if has_linear else None
    geo_first_bel = (
        data.geo_bel_paths_3d[0] if data.geo_bel_paths_3d else np.zeros((1, 3))
    )
    lin_first_bel = data.lin_bel_paths_3d[0] if has_linear else None

    act_fig, act_geo_line, act_lin_line, act_geo_pt, act_lin_pt = _build_figure(
        data,
        data.act_features_3d,
        data.act_feature_classes,
        data.act_mesh_traces,
        data.act_centroids_3d,
        geo_first_act,
        lin_first_act,
        colors,
        has_linear,
        edges=data.act_edges,
    )

    # Belief TPS curve as a trace list (matching mesh_traces interface)
    bel_mesh = []
    if data.bel_tps_curve_3d is not None:
        bel_mesh.append(
            go.Scatter3d(
                x=data.bel_tps_curve_3d[:, 0],
                y=data.bel_tps_curve_3d[:, 1],
                z=data.bel_tps_curve_3d[:, 2],
                mode="lines",
                line=dict(color=_MANIFOLD_COLOR, width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    bel_fig, bel_geo_line, bel_lin_line, bel_geo_pt, bel_lin_pt = _build_figure(
        data,
        data.bel_background_3d,
        data.bel_class_assignments,
        bel_mesh,
        data.bel_centroids_3d,
        geo_first_bel,
        lin_first_bel,
        colors,
        has_linear,
    )

    act_html = pio.to_html(
        act_fig, full_html=False, include_plotlyjs="cdn", div_id="act-plot"
    )
    bel_html = pio.to_html(
        bel_fig, full_html=False, include_plotlyjs=False, div_id="bel-plot"
    )

    n_steps = len(data.geo_act_paths_3d[0]) if data.geo_act_paths_3d else 1
    n_normal_steps = (
        data.n_normal_steps
        if data.n_normal_steps and data.n_normal_steps < n_steps
        else None
    )

    # Oversteer tick mark HTML (empty string when oversteering is disabled)
    if n_normal_steps is not None and n_steps > 1:
        _pct = (n_normal_steps - 1) / (n_steps - 1) * 100
        _after_pct = min(_pct + 3, 97)
        _oversteer_ticks_html = (
            f'<div style="position:relative;width:50%;max-width:500px;margin:0 auto;'
            f'height:20px;font-size:10px;color:#888;">'
            f'<span style="position:absolute;left:{_pct:.1f}%;transform:translateX(-50%);'
            f'white-space:nowrap;line-height:1.2;">&#9650;<br>Target</span>'
            f'<span style="position:absolute;left:{_after_pct:.1f}%;color:#c44;'
            f'white-space:nowrap;">Overshoot &#8594;</span>'
            f"</div>"
        )
    else:
        _oversteer_ticks_html = ""

    # Build line-plot panels for output distributions
    W = data.geo_dists[0].shape[-1] if data.geo_dists else 0
    dist_labels = (
        data.class_labels[:W] if W <= data.n_classes else [str(i) for i in range(W)]
    )
    # x-axis normalized to [0, 1]
    x_norm = np.linspace(0, 1, n_steps).tolist()

    def _make_dist_line_fig(dists: np.ndarray, title: str, div_id: str) -> str:
        fig = go.Figure()
        means = dists.mean(axis=1) if dists.ndim == 3 else dists
        for w in range(W):
            fig.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=means[:, w].tolist(),
                    mode="lines",
                    line=dict(color=colors[w] if w < len(colors) else "#999", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        other_vals = (1.0 - means.sum(axis=-1)).clip(0, 1).tolist()
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=other_vals,
                mode="lines",
                line=dict(color="#cc0000", width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_vline(x=0, line=dict(color="#333", width=1, dash="dot"))
        fig.update_layout(
            height=140,
            margin=dict(l=35, r=10, t=30, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family=_FONT_FAMILY, size=10),
            yaxis=dict(range=[0, 1], title=None, gridcolor="#eee"),
            xaxis=dict(range=[0, 1], title=None, dtick=0.25),
            title=dict(
                text=title,
                font=dict(size=14, color="#777", family=_FONT_FAMILY),
                x=0.5,
                xanchor="center",
                y=0.97,
            ),
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id)

    def _make_dist_bar_fig(div_id: str) -> str:
        """Bar chart showing geometric (solid) and linear (faded) at current step."""
        fig = go.Figure()
        bar_labels = list(dist_labels) + ["other"]
        bar_colors = list(colors[:W]) + ["#cc0000"]
        init_geo = data.geo_dists[0][0].tolist() if data.geo_dists else [0] * max(W, 1)
        init_geo = init_geo + [max(0.0, 1.0 - sum(init_geo))]
        fig.add_trace(
            go.Bar(
                x=bar_labels,
                y=init_geo,
                name="Geometric",
                marker_color=bar_colors,
                opacity=0.9,
            )
        )
        if has_linear and data.lin_dists:
            init_lin = data.lin_dists[0][0].tolist()
            init_lin = init_lin + [max(0.0, 1.0 - sum(init_lin))]
            fig.add_trace(
                go.Bar(
                    x=bar_labels,
                    y=init_lin,
                    name="Linear",
                    marker_color=bar_colors,
                    opacity=0.35,
                )
            )
        fig.update_layout(
            barmode="group",
            height=180,
            margin=dict(l=35, r=10, t=10, b=25),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family=_FONT_FAMILY, size=11),
            yaxis=dict(range=[0, 1], title=None, gridcolor="#eee"),
            xaxis=dict(title=None),
            showlegend=False,
            bargap=0.15,
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id)

    grid_info_geo = grid_info_lin = None

    if dist_mode == "bars":
        dist_html_block = _make_dist_bar_fig("bar-dist")
    elif is_grid_mode and data.output_token_values:
        gf_cmap = colormap or "managua"
        geo_gf_html, grid_info_geo = _build_grid_flow_html(
            data.output_token_values, "gf-geo", "Geometric", gf_cmap, color_by_dim
        )
        dist_html_block = geo_gf_html
        if has_linear and data.lin_dists:
            lin_gf_html, grid_info_lin = _build_grid_flow_html(
                data.output_token_values, "gf-lin", "Linear", gf_cmap, color_by_dim
            )
            dist_html_block += lin_gf_html
    else:
        geo_dist_html = _make_dist_line_fig(data.geo_dists[0], "Geometric", "geo-dist")
        lin_dist_html = ""
        if has_linear and data.lin_dists:
            lin_dist_html = _make_dist_line_fig(data.lin_dists[0], "Linear", "lin-dist")
        dist_html_block = geo_dist_html + lin_dist_html

    # Embed data as JSON
    geo_act_json = json.dumps([p.tolist() for p in data.geo_act_paths_3d])
    geo_bel_json = json.dumps([p.tolist() for p in data.geo_bel_paths_3d])
    geo_dists_json = json.dumps([d.tolist() for d in data.geo_dists])
    lin_act_json = (
        json.dumps([p.tolist() for p in data.lin_act_paths_3d])
        if has_linear
        else "null"
    )
    lin_bel_json = (
        json.dumps([p.tolist() for p in data.lin_bel_paths_3d])
        if has_linear
        else "null"
    )
    lin_dists_json = (
        json.dumps([d.tolist() for d in data.lin_dists])
        if has_linear and data.lin_dists
        else "null"
    )
    pairs_json = json.dumps(data.pairs)
    colors_json = json.dumps(colors)
    grid_info_geo_json = json.dumps(grid_info_geo) if grid_info_geo else "null"
    grid_info_lin_json = json.dumps(grid_info_lin) if grid_info_lin else "null"

    pair_options = []
    for si, ei in data.pairs:
        sl = data.class_labels[si] if si < len(data.class_labels) else str(si)
        el = data.class_labels[ei] if ei < len(data.class_labels) else str(ei)
        pair_options.append(f"{sl}  \u2192  {el}")
    pair_options_json = json.dumps(pair_options)

    page = f"""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Activation ↔ Belief</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: white; font-family: {_FONT_FAMILY};
         padding: 4px 6px; }}
  .plots {{ display: flex; flex-wrap: wrap; gap: 2px; }}
  .plot-container {{ min-width: 400px; flex: 1 1 400px; height: 600px; }}
  .plot-container h3 {{
    text-align: center; margin: 0; padding: 0; font-weight: 400;
    font-size: 12px; color: #999; letter-spacing: 0.05em;
    text-transform: uppercase; line-height: 1;
  }}
  .dist-stack {{
    max-width: 550px; margin: 0 auto 8px; padding: 0 16px;
  }}
  .dist-row {{
    display: flex; flex-wrap: wrap; gap: 8px; justify-content: center;
    margin: 0 auto 8px; padding: 0 16px;
  }}
  .controls {{
    display: flex; align-items: center; justify-content: center;
    gap: 20px; padding: 8px 20px;
    position: relative; z-index: 10;
    background: white;
  }}
  select {{
    font-family: {_FONT_FAMILY}; font-size: 14px;
    padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px;
    color: #333; background: white; cursor: pointer; outline: none;
  }}
  select:focus {{ border-color: #999; }}
  input[type=range] {{
    -webkit-appearance: none; height: 4px; background: #ddd;
    border-radius: 2px; outline: none; width: 50%; max-width: 500px;
    cursor: pointer;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none; width: 14px; height: 14px;
    border-radius: 50%; background: #444; cursor: pointer;
  }}
  input[type=range]::-moz-range-thumb {{
    width: 14px; height: 14px; border-radius: 50%;
    background: #444; border: none; cursor: pointer;
  }}
</style>
</head>
<body>
  {
        ""
        if dist_mode == "grid_flow_dual"
        else '''
  <div class="plots">
    <div class="plot-container">
      <h3>Activation Space</h3>
      '''
        + act_html
        + '''
    </div>
    <div class="plot-container">
      <h3>Belief Space</h3>
      '''
        + bel_html
        + '''
    </div>
  </div>
  <div class="'''
        + ("dist-row" if is_grid_mode else "dist-stack")
        + '''">
    '''
        + dist_html_block
        + '''
  </div>
  '''
    }
  {
        ""
        if dist_mode != "grid_flow_dual"
        else '''
  <div class="plots">
    <div class="plot-container">
      <h3>Activation Space</h3>
      '''
        + act_html
        + '''
    </div>
    <div class="plot-container" style="display:flex;flex-direction:column;justify-content:flex-start;align-items:center;padding-top:20px;gap:8px;">
      '''
        + dist_html_block
        + '''
    </div>
  </div>
  '''
    }
  <div class="controls">
    <select id="pair-select"></select>
    <input type="range" id="step-slider" min="0" max="{n_steps - 1}" value="0">
    {_oversteer_ticks_html}
  </div>
  <script>
  (function() {{
    var geoAct = {geo_act_json};
    var geoBel = {geo_bel_json};
    var geoDists = {geo_dists_json};
    var linAct = {lin_act_json};
    var linBel = {lin_bel_json};
    var linDists = {lin_dists_json};
    var pairs = {pairs_json};
    var pairLabels = {pair_options_json};
    var nNormalSteps = {n_normal_steps if n_normal_steps else n_steps};
    var colors = {colors_json};
    var hasLinear = {str(has_linear).lower()};

    var distMode = '{dist_mode}';
    var gridInfoGeo = {grid_info_geo_json};
    var gridInfoLin = {grid_info_lin_json};
    var actDiv = document.getElementById('act-plot');
    var belDiv = document.getElementById('bel-plot');
    var barDistDiv = document.getElementById('bar-dist');
    var geoDistDiv = document.getElementById('geo-dist');
    var linDistDiv = document.getElementById('lin-dist');
    var select = document.getElementById('pair-select');
    var slider = document.getElementById('step-slider');
    var W = geoDists[0][0].length;

    var aGL={act_geo_line}, aLL={act_lin_line}, aGP={act_geo_pt}, aLP={act_lin_pt};
    var bGL={bel_geo_line}, bLL={bel_lin_line}, bGP={bel_geo_pt}, bLP={bel_lin_pt};

    function hexToRgb(h) {{
      h=h.replace('#','');
      return [parseInt(h.substring(0,2),16),parseInt(h.substring(2,4),16),parseInt(h.substring(4,6),16)];
    }}
    function rgbToHex(r,g,b) {{
      return '#'+[r,g,b].map(function(v){{var s=Math.round(Math.max(0,Math.min(255,v))).toString(16);return s.length<2?'0'+s:s;}}).join('');
    }}
    function lerp(c1,c2,t) {{
      var a=hexToRgb(c1),b=hexToRgb(c2);
      return rgbToHex(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t);
    }}

    for (var i=0; i<pairLabels.length; i++) {{
      var o=document.createElement('option'); o.value=i; o.text=pairLabels[i];
      select.appendChild(o);
    }}

    function updateGridFlow(prefix, dists, step, info) {{
      var p = dists[step];
      var mx = 0;
      for (var i=0; i<p.length; i++) if (p[i]>mx) mx=p[i];
      var rcToW = info.rc_to_w;
      for (var r=0; r<info.n_rows; r++) {{
        for (var c=0; c<info.n_cols; c++) {{
          var el = document.getElementById(prefix+'-'+r+'-'+c);
          if (!el) continue;
          var w = rcToW[r+','+c];
          el.style.opacity = (w!==undefined && mx>0) ? (p[w]/mx) : 0;
        }}
      }}
    }}

    function restylePt(div,idx,path,step,col) {{
      Plotly.restyle(div,{{x:[[path[step][0]]],y:[[path[step][1]]],z:[[path[step][2]]],'marker.color':[col]}},[idx]);
    }}
    function restyleLine(div,idx,path) {{
      if (idx<0||!div) return;
      var xs=[],ys=[],zs=[];
      for(var i=0;i<path.length;i++){{ xs.push(path[i][0]); ys.push(path[i][1]); zs.push(path[i][2]); }}
      Plotly.restyle(div,{{x:[xs],y:[ys],z:[zs]}},[idx]);
    }}

    function update(pi, step) {{
      var si=pairs[pi][0], ei=pairs[pi][1];
      var t=geoAct[pi].length>1 ? step/(geoAct[pi].length-1) : 0;
      var col=lerp(colors[si], colors[ei], t);

      restylePt(actDiv, aGP, geoAct[pi], step, col);
      if (belDiv) restylePt(belDiv, bGP, geoBel[pi], step, col);

      if (hasLinear && linAct && aLP>=0) {{
        var ls=Math.min(step, linAct[pi].length-1);
        var lt=linAct[pi].length>1 ? ls/(linAct[pi].length-1) : 0;
        var lc=lerp(colors[si], colors[ei], lt);
        restylePt(actDiv, aLP, linAct[pi], ls, lc);
        if (belDiv) restylePt(belDiv, bLP, linBel[pi], ls, lc);
      }}

      // Update distribution plots
      if ((distMode === 'grid_flow' || distMode === 'grid_flow_dual') && gridInfoGeo) {{
        updateGridFlow('gf-geo', geoDists[pi], step, gridInfoGeo);
        if (hasLinear && linDists && gridInfoLin) {{
          var ls4 = Math.min(step, linDists[pi].length - 1);
          updateGridFlow('gf-lin', linDists[pi], ls4, gridInfoLin);
        }}
      }} else if (distMode === 'bars' && barDistDiv) {{
        var gyd = geoDists[pi][step].slice();
        var gOther = Math.max(0, 1 - gyd.reduce(function(a,b){{return a+b;}}, 0));
        gyd.push(gOther);
        Plotly.restyle(barDistDiv, {{y:[gyd]}}, [0]);
        if (hasLinear && linDists) {{
          var ls2=Math.min(step, linDists[pi].length-1);
          var lyd = linDists[pi][ls2].slice();
          var lOther = Math.max(0, 1 - lyd.reduce(function(a,b){{return a+b;}}, 0));
          lyd.push(lOther);
          Plotly.restyle(barDistDiv, {{y:[lyd]}}, [1]);
        }}
      }} else {{
        var xNorm = geoAct[pi].length > 1 ? step / (geoAct[pi].length - 1) : 0;
        if (geoDistDiv) Plotly.relayout(geoDistDiv, {{'shapes[0].x0':xNorm, 'shapes[0].x1':xNorm}});
        if (hasLinear && linDistDiv) {{
          var ls3 = Math.min(step, linAct[pi].length - 1);
          var lxNorm = linAct[pi].length > 1 ? ls3 / (linAct[pi].length - 1) : 0;
          Plotly.relayout(linDistDiv, {{'shapes[0].x0':lxNorm, 'shapes[0].x1':lxNorm}});
        }}
      }}
    }}

    function changePair(pi) {{
      // Update path lines
      restyleLine(actDiv, aGL, geoAct[pi]);
      restyleLine(belDiv, bGL, geoBel[pi]);
      if (hasLinear && linAct) {{
        restyleLine(actDiv, aLL, linAct[pi]);
        restyleLine(belDiv, bLL, linBel[pi]);
      }}
      if (distMode === 'lines') {{
        var gm = geoDists[pi];
        var nS = gm.length;
        for (var w=0; w<W; w++) {{
          var ys = [];
          for (var s=0; s<nS; s++) ys.push(gm[s][w]);
          if (geoDistDiv) Plotly.restyle(geoDistDiv, {{y:[ys]}}, [w]);
        }}
        var gOtherYs = [];
        for (var s=0; s<nS; s++) {{
          var gSm=0; for (var ww=0; ww<W; ww++) gSm+=gm[s][ww];
          gOtherYs.push(Math.max(0, 1-gSm));
        }}
        if (geoDistDiv) Plotly.restyle(geoDistDiv, {{y:[gOtherYs]}}, [W]);
        if (hasLinear && linDists && linDistDiv) {{
          var lm = linDists[pi];
          var lnS = lm.length;
          for (var w2=0; w2<W; w2++) {{
            var lys = [];
            for (var s2=0; s2<lnS; s2++) lys.push(lm[s2][w2]);
            Plotly.restyle(linDistDiv, {{y:[lys]}}, [w2]);
          }}
          var lOtherYs = [];
          for (var s3=0; s3<lnS; s3++) {{
            var lSm=0; for (var w3=0; w3<W; w3++) lSm+=lm[s3][w3];
            lOtherYs.push(Math.max(0, 1-lSm));
          }}
          Plotly.restyle(linDistDiv, {{y:[lOtherYs]}}, [W]);
        }}
      }}
    }}

    select.addEventListener('change', function() {{
      var pi=parseInt(this.value);
      slider.value=0;
      changePair(pi);
      update(pi, 0);
    }});
    slider.addEventListener('input', function() {{
      update(parseInt(select.value), parseInt(this.value));
    }});
    changePair(0);
    update(0, 0);
  }})();
  </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(page)
    logger.info("Saved dual manifold HTML to %s", output_path)
