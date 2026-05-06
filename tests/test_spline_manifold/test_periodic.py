"""Tests for ThinPlateSpline periodic modes and SplineManifold."""

import pytest
import torch

from causalab.methods.spline.tps import ThinPlateSpline
from causalab.methods.spline.manifold import SplineManifold


# ---------------------------------------------------------------------------
# ThinPlateSpline periodic tests
# ---------------------------------------------------------------------------


@pytest.fixture
def weekday_tps():
    """7-point ThinPlateSpline for a single periodic dimension."""
    n = 7
    period = 7.0
    cp = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    angles = 2 * torch.pi * torch.arange(n).float() / n
    values = torch.stack(
        [torch.cos(angles), torch.sin(angles), torch.arange(n).float() / n], dim=1
    )
    tps = ThinPlateSpline(
        cp, values, periodic_dims=[0], periods=[period], smoothness=0.0
    )
    return tps, cp, values, period


def test_tps_exact_interpolation(weekday_tps):
    """ThinPlateSpline interpolates exactly through control points."""
    tps, cp, values, _ = weekday_tps
    result = tps.evaluate(cp)
    assert torch.allclose(result, values, atol=1e-4), (
        f"Max error at control points: {(result - values).abs().max().item()}"
    )


def test_tps_wraparound(weekday_tps):
    """evaluate(u) == evaluate(u + period) for periodic dims."""
    tps, _, _, period = weekday_tps
    u = torch.tensor([[2.5]])
    u_shifted = torch.tensor([[2.5 + period]])
    v1 = tps.evaluate(u)
    v2 = tps.evaluate(u_shifted)
    assert torch.allclose(v1, v2, atol=1e-4), f"Wrap-around mismatch: {v1} vs {v2}"


@pytest.fixture
def mixed_periodic_tps():
    """2D control points: dim 0 is periodic (period=7), dim 1 is linear."""
    torch.manual_seed(42)
    n = 14
    cp = torch.zeros(n, 2)
    cp[:, 0] = torch.arange(7).float().repeat(2)  # periodic dim
    cp[:, 1] = torch.cat([torch.zeros(7), torch.ones(7)])  # linear dim
    # Make unique combinations
    values = torch.randn(n, 3)
    tps = ThinPlateSpline(cp, values, periodic_dims=[0], periods=[7.0], smoothness=0.0)
    return tps, cp, values


def test_mixed_periodic_tps_interpolation(mixed_periodic_tps):
    """Mixed periodic+linear TPS interpolates at control points."""
    tps, cp, values = mixed_periodic_tps
    result = tps.evaluate(cp)
    assert torch.allclose(result, values, atol=1e-3), (
        f"Max error: {(result - values).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# SplineManifold periodic tests
# ---------------------------------------------------------------------------


@pytest.fixture
def periodic_manifold():
    """SplineManifold with periodic dim."""
    n = 7
    period = 7.0
    cp = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    angles = 2 * torch.pi * torch.arange(n).float() / n
    tp = torch.stack(
        [torch.cos(angles), torch.sin(angles), torch.arange(n).float() / n], dim=1
    )
    manifold = SplineManifold(
        control_points=cp,
        target_points=tp,
        intrinsic_dim=1,
        ambient_dim=3,
        periodic_dims=[0],
        periods=[period],
    )
    return manifold, cp, tp, period


def test_manifold_decode(periodic_manifold):
    """decode at control points matches target points."""
    manifold, cp, tp, _ = periodic_manifold
    result = manifold.decode(cp)
    assert torch.allclose(result, tp, atol=1e-4)


def test_manifold_state_dict_roundtrip(periodic_manifold):
    """state_dict roundtrip."""
    manifold, _, _, _ = periodic_manifold
    state = manifold.state_dict_to_save()

    # spline_method is persisted so reloads pick the same backend
    assert state["spline_method"] == manifold.spline_method

    loaded = SplineManifold.from_state_dict(state)
    test_u = torch.tensor([[2.0]])
    assert torch.allclose(manifold.decode(test_u), loaded.decode(test_u), atol=1e-4)


def test_manifold_steering_grid(periodic_manifold):
    """make_steering_grid uses [0, period) for periodic dims."""
    manifold, _, _, period = periodic_manifold
    grid = manifold.make_steering_grid(n_points_per_dim=8)
    assert grid.shape == (8, 1)
    assert grid.min().item() >= 0.0
    assert grid.max().item() <= period


def test_backward_compat_tps():
    """Standard TPS manifold still works with new signature (backward compat)."""
    cp = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    tp = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.6],
        ]
    )
    manifold = SplineManifold(
        control_points=cp, target_points=tp, intrinsic_dim=2, ambient_dim=3
    )
    result = manifold.decode(cp)
    assert torch.allclose(result, tp, atol=1e-4)

    # state_dict roundtrip
    state = manifold.state_dict_to_save()
    loaded = SplineManifold.from_state_dict(state)
    assert torch.allclose(manifold.decode(cp), loaded.decode(cp), atol=1e-5)


def test_from_state_dict_ignores_old_spline_method():
    """from_state_dict ignores spline_method from old checkpoints."""
    cp = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    tp = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.6],
        ]
    )

    # Old checkpoint with spline_method field
    for old_method in ["tps", "euclidean_tps", "mixed_tps", "periodic_tps"]:
        state = {
            "control_points": cp,
            "centroids": tp,
            "intrinsic_dim": 2,
            "ambient_dim": 3,
            "smoothness": 0.0,
            "periodic_dims": None,
            "periods": None,
            "spline_method": old_method,
        }
        loaded = SplineManifold.from_state_dict(state)
        result = loaded.decode(cp)
        assert torch.allclose(result, tp, atol=1e-4)
