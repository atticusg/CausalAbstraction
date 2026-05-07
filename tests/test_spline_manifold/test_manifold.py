"""Tests for SplineManifold and ThinPlateSpline."""

import math

import pytest
import torch

from causalab.methods.spline.tps import ThinPlateSpline
from causalab.methods.spline.manifold import SplineManifold
from causalab.methods.spline.builders import build_spline_manifold


@pytest.fixture
def simple_2d_to_3d_data():
    """Create 5 control points (2D) and target points (3D)."""
    control_points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )
    target_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.6],
        ]
    )
    return control_points, target_points


def test_tps_interpolation(simple_2d_to_3d_data):
    """TPS(smoothness=0) interpolates exactly through control points."""
    control_points, target_points = simple_2d_to_3d_data
    tps = ThinPlateSpline(control_points, target_points, smoothness=0.0)
    result = tps.evaluate(control_points)
    assert torch.allclose(result, target_points, atol=1e-4)


def test_tps_smoothness(simple_2d_to_3d_data):
    """smoothness>0 makes result differ from exact interpolation."""
    control_points, target_points = simple_2d_to_3d_data
    tps_exact = ThinPlateSpline(control_points, target_points, smoothness=0.0)
    tps_smooth = ThinPlateSpline(control_points, target_points, smoothness=1.0)
    result_exact = tps_exact.evaluate(control_points)
    result_smooth = tps_smooth.evaluate(control_points)
    assert not torch.allclose(result_exact, result_smooth, atol=1e-4)


def test_spline_manifold_creation(simple_2d_to_3d_data):
    """Checks intrinsic_dim==2, ambient_dim==3, n_centroids==5."""
    control_points, target_points = simple_2d_to_3d_data
    manifold = SplineManifold(
        control_points=control_points,
        target_points=target_points,
        intrinsic_dim=2,
        ambient_dim=3,
    )
    assert manifold.intrinsic_dim == 2
    assert manifold.ambient_dim == 3
    assert manifold.n_centroids == 5


def test_spline_manifold_decode(simple_2d_to_3d_data):
    """decode(control_points) matches target_points."""
    control_points, target_points = simple_2d_to_3d_data
    manifold = SplineManifold(
        control_points=control_points,
        target_points=target_points,
        intrinsic_dim=2,
        ambient_dim=3,
    )
    result = manifold.decode(control_points)
    assert torch.allclose(result, target_points, atol=1e-4)


def test_spline_manifold_encode(simple_2d_to_3d_data):
    """encode returns shapes (5,2) and (5,1), residual < 0.0001."""
    control_points, target_points = simple_2d_to_3d_data
    manifold = SplineManifold(
        control_points=control_points,
        target_points=target_points,
        intrinsic_dim=2,
        ambient_dim=3,
    )
    u, residual = manifold.encode(target_points)
    assert u.shape == (5, 2)
    assert residual.shape == (5, 1)
    assert residual.max().item() < 0.0001


def test_spline_manifold_roundtrip(simple_2d_to_3d_data):
    """encode->decode roundtrip."""
    control_points, target_points = simple_2d_to_3d_data
    manifold = SplineManifold(
        control_points=control_points,
        target_points=target_points,
        intrinsic_dim=2,
        ambient_dim=3,
    )
    u, _ = manifold.encode(target_points)
    reconstructed = manifold.decode(u)
    assert torch.allclose(reconstructed, target_points, atol=1e-4)


def test_spline_manifold_project(simple_2d_to_3d_data):
    """Projection reduces distance to target_points."""
    control_points, target_points = simple_2d_to_3d_data
    manifold = SplineManifold(
        control_points=control_points,
        target_points=target_points,
        intrinsic_dim=2,
        ambient_dim=3,
    )
    noisy = target_points + torch.randn_like(target_points) * 0.1
    projected = manifold.project(noisy)
    dist_before = torch.cdist(noisy, target_points, p=2).min(dim=1).values.mean()
    dist_after = torch.cdist(projected, target_points, p=2).min(dim=1).values.mean()
    assert dist_after <= dist_before


def test_spline_manifold_batching(simple_2d_to_3d_data):
    """Batch of 10 random points, check shapes."""
    control_points, target_points = simple_2d_to_3d_data
    manifold = SplineManifold(
        control_points=control_points,
        target_points=target_points,
        intrinsic_dim=2,
        ambient_dim=3,
    )
    batch = torch.randn(10, 2)
    result = manifold.decode(batch)
    assert result.shape == (10, 3)


def test_spline_manifold_state_dict(simple_2d_to_3d_data):
    """Save/load roundtrip."""
    control_points, target_points = simple_2d_to_3d_data
    manifold = SplineManifold(
        control_points=control_points,
        target_points=target_points,
        intrinsic_dim=2,
        ambient_dim=3,
    )
    state = manifold.state_dict_to_save()
    loaded = SplineManifold.from_state_dict(state)
    test_input = torch.tensor([[0.5, 0.5]])
    assert torch.allclose(
        manifold.decode(test_input),
        loaded.decode(test_input),
        atol=1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spline_manifold_gpu(simple_2d_to_3d_data):
    """CUDA test."""
    control_points, target_points = simple_2d_to_3d_data
    manifold = SplineManifold(
        control_points=control_points,
        target_points=target_points,
        intrinsic_dim=2,
        ambient_dim=3,
    )
    manifold = manifold.to("cuda")
    test_input = torch.tensor([[0.5, 0.5]], device="cuda")
    result = manifold.decode(test_input)
    assert result.device.type == "cuda"
    assert result.shape == (1, 3)


def test_spline_manifold_warnings():
    """Warns for large ambient dims."""
    control_points = torch.randn(5, 2)
    target_points = torch.randn(5, 40)
    with pytest.warns(UserWarning):
        SplineManifold(
            control_points=control_points,
            target_points=target_points,
            intrinsic_dim=2,
            ambient_dim=40,
        )


def test_dimension_mismatch_errors():
    """Raises on mismatched control/target points count."""
    control_points = torch.randn(5, 2)
    target_points = torch.randn(3, 3)
    with pytest.raises(RuntimeError):
        SplineManifold(
            control_points=control_points,
            target_points=target_points,
            intrinsic_dim=2,
            ambient_dim=3,
        )


# ---- Periodic TPS tests ----


def test_periodic_tps_exact_periodicity():
    """A 1D periodic TPS mapping [0,1) -> circle in 2D must satisfy f(0) == f(1)."""
    n = 8
    t = torch.linspace(0, 1, n + 1)[:-1]  # [0, 1/8, ..., 7/8] — no duplicate endpoint
    cp = t.unsqueeze(1)
    targets = torch.stack(
        [torch.cos(2 * math.pi * t), torch.sin(2 * math.pi * t)], dim=1
    )

    tps = ThinPlateSpline(cp, targets, smoothness=0.0, periodic_dims=(True,))

    # f(0) and f(1) must be identical
    val_0 = tps.evaluate(torch.tensor([[0.0]]))
    val_1 = tps.evaluate(torch.tensor([[1.0]]))
    assert torch.allclose(val_0, val_1, atol=1e-5)


def test_periodic_tps_interpolation():
    """Periodic TPS with smoothness=0 must still pass through control points."""
    n = 8
    t = torch.linspace(0, 1, n + 1)[:-1]
    cp = t.unsqueeze(1)
    targets = torch.stack(
        [torch.cos(2 * math.pi * t), torch.sin(2 * math.pi * t)], dim=1
    )

    tps = ThinPlateSpline(cp, targets, smoothness=0.0, periodic_dims=(True,))
    result = tps.evaluate(cp)
    assert torch.allclose(result, targets, atol=1e-4)


def test_periodic_tps_midpoint_sanity():
    """Midpoints between control points on a circle should have ~unit norm."""
    n = 12
    t = torch.linspace(0, 1, n + 1)[:-1]
    cp = t.unsqueeze(1)
    targets = torch.stack(
        [torch.cos(2 * math.pi * t), torch.sin(2 * math.pi * t)], dim=1
    )

    tps = ThinPlateSpline(cp, targets, smoothness=0.0, periodic_dims=(True,))

    # Evaluate at midpoints
    t_mid = t + 0.5 / n
    result = tps.evaluate(t_mid.unsqueeze(1))
    norms = result.norm(dim=1)
    # On a well-fit circle, midpoint norms should be close to 1.0
    assert (norms - 1.0).abs().max() < 0.05


def test_periodic_tps_natural_periodicity_beyond_domain():
    """f(u) == f(u+1) for arbitrary u values, not just at 0/1."""
    n = 8
    t = torch.linspace(0, 1, n + 1)[:-1]
    cp = t.unsqueeze(1)
    targets = torch.stack(
        [torch.cos(2 * math.pi * t), torch.sin(2 * math.pi * t)], dim=1
    )

    tps = ThinPlateSpline(cp, targets, smoothness=0.0, periodic_dims=(True,))

    test_u = torch.tensor([[0.13], [0.37], [0.71], [0.99]])
    val_u = tps.evaluate(test_u)
    val_u_plus_1 = tps.evaluate(test_u + 1.0)
    assert torch.allclose(val_u, val_u_plus_1, atol=1e-5)


def test_periodic_tps_derivative_continuity_at_boundary():
    """Derivative at u=0+ should match derivative at u=1- (smooth wraparound)."""
    n = 12
    t = torch.linspace(0, 1, n + 1)[:-1]
    cp = t.unsqueeze(1)
    targets = torch.stack(
        [torch.cos(2 * math.pi * t), torch.sin(2 * math.pi * t)], dim=1
    )

    tps = ThinPlateSpline(cp, targets, smoothness=0.0, periodic_dims=(True,))

    eps = 1e-4
    # Forward difference at 0
    deriv_at_0 = (
        tps.evaluate(torch.tensor([[eps]])) - tps.evaluate(torch.tensor([[0.0]]))
    ) / eps
    # Backward difference at 1
    deriv_at_1 = (
        tps.evaluate(torch.tensor([[1.0]])) - tps.evaluate(torch.tensor([[1.0 - eps]]))
    ) / eps
    assert torch.allclose(deriv_at_0, deriv_at_1, atol=1e-2)


def test_periodic_tps_backward_compat(simple_2d_to_3d_data):
    """Default periodic_dims=None must behave identically to explicit all-False."""
    control_points, target_points = simple_2d_to_3d_data
    tps_default = ThinPlateSpline(control_points, target_points, smoothness=0.0)
    tps_explicit = ThinPlateSpline(
        control_points, target_points, smoothness=0.0, periodic_dims=(False, False)
    )
    test_u = torch.tensor([[0.3, 0.7], [0.1, 0.9]])
    assert torch.allclose(
        tps_default.evaluate(test_u), tps_explicit.evaluate(test_u), atol=1e-6
    )


def test_periodic_tps_smoothness():
    """smoothness > 0 should change output for periodic TPS."""
    n = 8
    t = torch.linspace(0, 1, n + 1)[:-1]
    cp = t.unsqueeze(1)
    targets = torch.stack(
        [torch.cos(2 * math.pi * t), torch.sin(2 * math.pi * t)], dim=1
    )

    tps_exact = ThinPlateSpline(cp, targets, smoothness=0.0, periodic_dims=(True,))
    tps_smooth = ThinPlateSpline(cp, targets, smoothness=0.1, periodic_dims=(True,))
    result_exact = tps_exact.evaluate(cp)
    result_smooth = tps_smooth.evaluate(cp)
    assert not torch.allclose(result_exact, result_smooth, atol=1e-4)


def test_periodic_manifold_decode_periodic():
    """SplineManifold with periodic dim: decode wraps correctly."""
    n = 8
    # Raw control points in [0, 7] (like weekday indices)
    t_raw = torch.arange(n).float()
    cp = t_raw.unsqueeze(1)
    targets = torch.stack(
        [
            torch.cos(2 * math.pi * t_raw / n),
            torch.sin(2 * math.pi * t_raw / n),
        ],
        dim=1,
    )

    manifold = SplineManifold(
        control_points=cp,
        target_points=targets,
        intrinsic_dim=1,
        ambient_dim=2,
        periodic_dims=(True,),
    )

    # The period in raw coordinates spans the control point range
    # After normalization to [0,1), f(min) == f(min + range)
    val_0 = manifold.decode(torch.tensor([[0.0]]))
    val_n = manifold.decode(torch.tensor([[float(n)]]))
    assert torch.allclose(val_0, val_n, atol=1e-4)


def test_periodic_manifold_mixed_dims():
    """SplineManifold with mixed periodic/linear dims (cylinder via ghost points)."""
    n_h, n_a = 3, 8
    heights = torch.linspace(0, 1, n_h)
    angles = torch.linspace(0, 1, n_a + 1)[:-1]
    h_grid, a_grid = torch.meshgrid(heights, angles, indexing="ij")
    cp = torch.stack([h_grid.flatten(), a_grid.flatten()], dim=1)

    h_flat = h_grid.flatten()
    a_flat = a_grid.flatten()
    targets = torch.stack(
        [
            h_flat,
            torch.cos(2 * math.pi * a_flat),
            torch.sin(2 * math.pi * a_flat),
        ],
        dim=1,
    )

    manifold = SplineManifold(
        control_points=cp,
        target_points=targets,
        intrinsic_dim=2,
        ambient_dim=3,
        periodic_dims=(False, True),
    )

    # Periodic along angle dim: f(h, 0) == f(h, 1)
    val_0 = manifold.decode(torch.tensor([[0.5, 0.0]]))
    val_1 = manifold.decode(torch.tensor([[0.5, 1.0]]))
    assert torch.allclose(val_0, val_1, atol=1e-4)

    # Not periodic along height dim: f(0, a) != f(1, a)
    val_bot = manifold.decode(torch.tensor([[0.0, 0.3]]))
    val_top = manifold.decode(torch.tensor([[1.0, 0.3]]))
    assert not torch.allclose(val_bot, val_top, atol=0.1)

    # Interpolates through control points
    decoded = manifold.decode(cp)
    assert torch.allclose(decoded, targets, atol=1e-3)


def test_periodic_manifold_state_dict_roundtrip():
    """Save/load roundtrip preserves periodic_dims and produces same output."""
    n = 8
    t = torch.linspace(0, 1, n + 1)[:-1]
    cp = t.unsqueeze(1)
    targets = torch.stack(
        [torch.cos(2 * math.pi * t), torch.sin(2 * math.pi * t)], dim=1
    )

    manifold = SplineManifold(
        control_points=cp,
        target_points=targets,
        intrinsic_dim=1,
        ambient_dim=2,
        periodic_dims=(True,),
    )
    state = manifold.state_dict_to_save()
    loaded = SplineManifold.from_state_dict(state)

    assert loaded.periodic_dims == [0]
    test_u = torch.tensor([[0.3], [0.7]])
    assert torch.allclose(manifold.decode(test_u), loaded.decode(test_u), atol=1e-5)


def test_periodic_builder_passthrough():
    """build_spline_manifold forwards periodic_dims to SplineManifold."""
    n = 8
    t = torch.linspace(0, 1, n + 1)[:-1]
    cp = t.unsqueeze(1)
    targets = torch.stack(
        [torch.cos(2 * math.pi * t), torch.sin(2 * math.pi * t)], dim=1
    )

    manifold = build_spline_manifold(cp, targets, periodic_dims=(True,))
    val_0 = manifold.decode(torch.tensor([[0.0]]))
    val_1 = manifold.decode(torch.tensor([[1.0]]))
    assert torch.allclose(val_0, val_1, atol=1e-5)
