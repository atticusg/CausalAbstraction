"""Builder functions for constructing spline manifolds from data."""

from __future__ import annotations

import torch
from torch import Tensor

from .manifold import SplineManifold


def build_spline_manifold(
    control_points: Tensor,
    centroids: Tensor,
    intrinsic_dim: int | None = None,
    ambient_dim: int | None = None,
    smoothness: float = 0.0,
    device: str | torch.device = "cpu",
    periodic_dims: list[int] | tuple[bool, ...] | None = None,
    periods: list[float] | None = None,
    spline_method: str = "auto",
    sphere_project: bool = False,
) -> SplineManifold:
    """Build a SplineManifold from control points and centroids.

    Args:
        control_points: Parameter combinations (n_centroids, n_params).
        centroids: Mean features per group (n_centroids, ambient_dim).
        intrinsic_dim: Intrinsic dimension. Defaults to control_points.shape[1].
        ambient_dim: Ambient dimension. Defaults to centroids.shape[1].
        smoothness: Smoothness parameter (TPS regularizer or cubic Reinsch λ).
        device: Device to place the manifold on.
        periodic_dims: Which dimensions of control_points are periodic.
        periods: Period for each periodic dimension.
        spline_method: Backend selector. ``"auto"`` picks a natural cubic
            spline for 1D non-cyclic data and TPS otherwise. ``"tps"`` and
            ``"cubic"`` force the corresponding backend.
        sphere_project: When True, decode() projects the ambient spline
            value onto the unit L2 sphere. Use for the belief manifold
            (Hellinger space) so every decoded point is a valid sqrt(p).

    Returns:
        SplineManifold instance.
    """
    if intrinsic_dim is None:
        intrinsic_dim = control_points.shape[1]
    if ambient_dim is None:
        ambient_dim = centroids.shape[1]

    device = torch.device(device)
    control_points = control_points.to(device)
    centroids = centroids.to(device)

    manifold = SplineManifold(
        control_points=control_points,
        target_points=centroids,
        intrinsic_dim=intrinsic_dim,
        ambient_dim=ambient_dim,
        smoothness=smoothness,
        periodic_dims=periodic_dims,
        periods=periods,
        spline_method=spline_method,
        sphere_project=sphere_project,
    )

    return manifold


# ─────────────────────────────────────────────────────────────────────
# Periodic dimension detection
# ─────────────────────────────────────────────────────────────────────


def detect_periodic_dims(
    control_points: Tensor,
    eigenvalues: Tensor,
    eigenvalue_tol: float = 0.45,
    min_variance_fraction: float = 0.1,
) -> list[tuple[int, int]]:
    """Detect periodic dimension pairs from near-degenerate eigenvalues.

    A pair (i, j) is periodic if:
    1. Both eigenvalues are significant (each ≥ min_variance_fraction of total)
    2. Eigenvalues are near-degenerate: |λ_i - λ_j| / max(λ_i, λ_j) < eigenvalue_tol

    Near-degenerate eigenvalues signal a closed loop (circle/ellipse).

    Args:
        control_points: (n_centroids, n_components) coordinates.
        eigenvalues: (n_components,) variance per dimension.
        eigenvalue_tol: Max relative eigenvalue difference for pairing.
            0.5 allows aspect ratios up to 2:1 (elliptical loops).
        min_variance_fraction: Minimum fraction of total eigenvalue sum
            that each dimension must explain to be considered.

    Returns:
        List of (i, j) periodic dimension pairs.
    """
    n_comp = control_points.shape[1]
    total_var = eigenvalues.sum().item()
    min_eigenvalue = min_variance_fraction * total_var

    used = set()
    pairs = []

    for i in range(n_comp):
        if i in used:
            continue
        li = eigenvalues[i].item()
        if li < min_eigenvalue:
            continue
        for j in range(i + 1, n_comp):
            if j in used:
                continue
            lj = eigenvalues[j].item()
            if lj < min_eigenvalue:
                continue

            ratio = abs(li - lj) / max(li, lj)
            if ratio < eigenvalue_tol:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break  # Move to next i

    return pairs


def remap_periodic_to_angle(
    control_points: Tensor,
    periodic_pairs: list[tuple[int, int]],
    eigenvalues: Tensor | None = None,
) -> tuple[Tensor, list[int], list[float]]:
    """Collapse periodic dimension pairs into angular columns.

    Each periodic pair (i, j) -> one column θ with period 2π. When eigenvalues
    are provided, coordinates are normalized by sqrt(eigenvalue) before atan2,
    which maps an elliptical embedding back to a circle for uniform angular
    spacing (equivalent to arc-length parameterization).

    Non-periodic dimensions pass through unchanged.

    Args:
        control_points: (n_centroids, n_components) in ISOMAP coordinates.
        periodic_pairs: List of (i, j) pairs from detect_periodic_dims.
        eigenvalues: (n_components,) ISOMAP eigenvalues for normalization.

    Returns:
        new_points: (n_centroids, new_n_components) with collapsed dims.
        periodic_dim_indices: Indices of periodic columns in new_points.
        periods: Period for each periodic column (always 2π).
    """
    import math

    n_comp = control_points.shape[1]

    paired_dims = set()
    for i, j in periodic_pairs:
        paired_dims.add(i)
        paired_dims.add(j)

    columns = []
    periodic_dim_indices = []
    periods = []

    # Add angular columns for each periodic pair
    for i, j in periodic_pairs:
        col_i = control_points[:, i] - control_points[:, i].mean()
        col_j = control_points[:, j] - control_points[:, j].mean()
        # Normalize by sqrt(eigenvalue) to map ellipse -> circle
        if eigenvalues is not None:
            col_i = col_i / eigenvalues[i].sqrt()
            col_j = col_j / eigenvalues[j].sqrt()
        angle = torch.atan2(col_j, col_i)
        periodic_dim_indices.append(len(columns))
        periods.append(2 * math.pi)
        columns.append(angle)

    # Add non-periodic columns
    for d in range(n_comp):
        if d not in paired_dims:
            columns.append(control_points[:, d])

    new_points = torch.stack(columns, dim=1)
    return new_points, periodic_dim_indices, periods
