"""Tests for PCA parameterization of spline manifolds.

PCA parameterization uses the first d columns of centroids (already in PCA space
from the subspace stage) as control points, with periodicity detection on the
variance of those columns.
"""

import math

import torch

from causalab.methods.spline.builders import (
    compute_centroids,
    detect_periodic_dims,
    remap_periodic_to_angle,
    build_spline_manifold,
)


def test_pca_periodicity_detection_circle():
    """Circle already in PCA space: top 2 dims have near-equal variance → periodic."""
    torch.manual_seed(0)
    n = 24
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    # Simulate features already in PCA space: cos/sin in first 2 dims, noise in rest
    features = torch.zeros(n, 10)
    features[:, 0] = torch.cos(t)
    features[:, 1] = torch.sin(t)
    features[:, 2:] = torch.randn(n, 8) * 0.1

    param_tensors = {"angle_idx": torch.arange(n).float()}
    _, centroids, _ = compute_centroids(features, param_tensors)

    control_points = centroids[:, :2]
    eigenvalues = control_points.var(dim=0)

    pairs = detect_periodic_dims(control_points, eigenvalues)
    assert len(pairs) >= 1, "Should detect periodic pair for circle data"


def test_pca_periodicity_detection_line():
    """Line already in PCA space: first PC dominates → no periodic pairs."""
    n = 20
    t = torch.linspace(0, 1, n)
    # Simulate features in PCA space: linear in dim 0, noise in rest
    features = torch.zeros(n, 10)
    features[:, 0] = t
    features[:, 1:] = torch.randn(n, 9) * 0.1

    param_tensors = {"pos": torch.arange(n).float()}
    _, centroids, _ = compute_centroids(features, param_tensors)

    control_points = centroids[:, :2]
    eigenvalues = control_points.var(dim=0)

    pairs = detect_periodic_dims(control_points, eigenvalues)
    assert len(pairs) == 0, "Should not detect periodicity for line data"


def test_pca_line_tps_interpolation():
    """Line in PCA space → use PC1 as control points → TPS interpolates centroids."""
    torch.manual_seed(1)
    n = 12
    ambient_dim = 20

    t = torch.linspace(0, 1, n)
    # Features in PCA space: linear in dim 0, noise in rest
    features = torch.randn(n, ambient_dim) * 0.1
    features[:, 0] = t

    param_tensors = {"pos": torch.arange(n).float()}
    _, centroids, _ = compute_centroids(features, param_tensors)

    cp_1d = centroids[:, 0:1]

    manifold = build_spline_manifold(
        control_points=cp_1d,
        centroids=centroids,
        intrinsic_dim=1,
        ambient_dim=ambient_dim,
    )

    recon = manifold.decode(cp_1d)
    assert torch.allclose(recon, centroids, atol=1e-3), (
        f"TPS should interpolate centroids: max error = {(recon - centroids).abs().max()}"
    )


def test_pca_circle_wrapping():
    """Circle in PCA space → detect periodicity → TPS wraps: decode(t) ≈ decode(t+2π)."""
    torch.manual_seed(0)
    n = 24
    ambient_dim = 10

    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    # Features in PCA space: cos/sin in first 2 dims, noise in rest
    features = torch.randn(n, ambient_dim) * 0.1
    features[:, 0] = torch.cos(t)
    features[:, 1] = torch.sin(t)

    param_tensors = {"angle_idx": torch.arange(n).float()}
    _, centroids, _ = compute_centroids(features, param_tensors)

    control_points = centroids[:, :2]
    eigenvalues = control_points.var(dim=0)

    pairs = detect_periodic_dims(control_points, eigenvalues)
    assert len(pairs) >= 1

    new_points, periodic_dim_indices, periods = remap_periodic_to_angle(
        control_points,
        pairs,
        eigenvalues,
    )

    angle_col = new_points[:, periodic_dim_indices[0]].unsqueeze(1)

    manifold = build_spline_manifold(
        control_points=angle_col,
        centroids=centroids,
        intrinsic_dim=1,
        ambient_dim=ambient_dim,
        periodic_dims=[True],
        periods=periods,
    )

    test_t = torch.tensor([[0.5]])
    val_t = manifold.decode(test_t)
    val_t_wrap = manifold.decode(test_t + 2 * math.pi)
    assert torch.allclose(val_t, val_t_wrap, atol=1e-3), (
        f"Manifold should wrap: max diff = {(val_t - val_t_wrap).abs().max()}"
    )
