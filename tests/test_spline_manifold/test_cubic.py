"""Tests for CubicSpline1D and the cubic-backend dispatch in SplineManifold."""

import math

import pytest
import torch

from causalab.methods.spline import (
    CubicSpline1D,
    NaturalCubicSpline1D,
    SplineManifold,
    ThinPlateSpline,
)


def _quadratic_data(n: int = 8, ambient: int = 4):
    torch.manual_seed(0)
    cp = torch.linspace(-2.0, 3.0, n).unsqueeze(-1)
    # smooth multi-output target so cubic should reproduce it well
    coefs = torch.randn(3, ambient)
    y = cp**2 @ coefs[0:1] + cp @ coefs[1:2] + coefs[2:3]
    return cp, y


def test_cubic_interpolates_controls_exactly():
    cp, y = _quadratic_data()
    spline = CubicSpline1D(cp, y, smoothness=0.0)
    out = spline.evaluate(cp)
    assert torch.allclose(out, y, atol=1e-5), (out - y).abs().max()


def test_cubic_matches_scipy():
    pytest.importorskip("scipy")
    from scipy.interpolate import CubicSpline

    cp, y = _quadratic_data(n=10, ambient=3)
    spline = CubicSpline1D(cp, y, smoothness=0.0)

    sci = CubicSpline(cp[:, 0].numpy(), y.numpy(), bc_type="natural")
    u = torch.linspace(-1.5, 2.5, 50).unsqueeze(-1)
    ours = spline.evaluate(u).numpy()
    ref = sci(u[:, 0].numpy())
    assert ours.shape == ref.shape
    assert abs(ours - ref).max() < 1e-5


def test_cubic_linear_extrapolation():
    cp, y = _quadratic_data(n=6, ambient=2)
    spline = CubicSpline1D(cp, y, smoothness=0.0)

    # Evaluate three colinear points outside the left boundary; the deltas
    # between consecutive outputs must be equal (linear extrapolation).
    x_min = cp[:, 0].min().item()
    u_outside = torch.tensor([[x_min - 1.0], [x_min - 0.5], [x_min - 0.1]])
    out = spline.evaluate(u_outside)
    d1 = out[1] - out[0]
    d2 = out[2] - out[1]
    # Slopes are proportional to step sizes (0.5 and 0.4) — compare per-step.
    assert torch.allclose(d1 / 0.5, d2 / 0.4, atol=1e-5)


def test_cubic_continuity_at_interior_knots():
    cp, y = _quadratic_data(n=8, ambient=3)
    spline = CubicSpline1D(cp, y, smoothness=0.0)

    # No value jump at any interior knot: the limit from the left and the
    # limit from the right must match. We probe with a tiny offset.
    eps = 1e-6
    for k in range(1, cp.shape[0] - 1):
        x_k = cp[k, 0].item()
        v_left = spline.evaluate(torch.tensor([[x_k - eps]]))
        v_right = spline.evaluate(torch.tensor([[x_k + eps]]))
        v_at = spline.evaluate(torch.tensor([[x_k]]))
        assert torch.allclose(v_left, v_at, atol=1e-4)
        assert torch.allclose(v_right, v_at, atol=1e-4)


def test_cubic_first_derivative_continuous_at_knots():
    cp, y = _quadratic_data(n=8, ambient=3)
    spline = CubicSpline1D(cp, y, smoothness=0.0)

    # Compare one-sided derivatives at each interior knot. The secant slope
    # over a tiny interval ε on each side of the knot should agree to O(ε).
    eps = 1e-4
    for k in range(1, cp.shape[0] - 1):
        x_k = cp[k, 0].item()
        # Slope from inside the left segment, ending at x_k
        d_left = (
            spline.evaluate(torch.tensor([[x_k]]))
            - spline.evaluate(torch.tensor([[x_k - eps]]))
        ) / eps
        # Slope from inside the right segment, starting at x_k
        d_right = (
            spline.evaluate(torch.tensor([[x_k + eps]]))
            - spline.evaluate(torch.tensor([[x_k]]))
        ) / eps
        # Difference is O(ε · |f''|) which is at most ~ε·second_derivative_magnitude.
        assert torch.allclose(d_left, d_right, atol=1e-2)


def test_cubic_smoothing_reduces_roughness():
    # With smoothness > 0 the fit should not interpolate exactly, and the
    # resulting curve should be smoother (lower second-derivative norm) than
    # the interpolating one.
    torch.manual_seed(1)
    n = 20
    cp = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
    clean = torch.sin(2 * math.pi * cp)
    noise = 0.1 * torch.randn_like(clean)
    y = clean + noise

    interp = CubicSpline1D(cp, y, smoothness=0.0)
    smooth = CubicSpline1D(cp, y, smoothness=1e-4)

    # Interpolating spline reproduces the noisy data exactly.
    assert torch.allclose(interp.evaluate(cp), y, atol=1e-5)
    # Smoothing spline does not.
    assert (smooth.evaluate(cp) - y).abs().max() > 1e-4

    # Roughness (sum of squared second derivatives at interior knots) drops.
    rough_interp = (interp.gamma**2).sum()
    rough_smooth = (smooth.gamma**2).sum()
    assert rough_smooth < rough_interp


def test_cubic_rejects_non_1d_input():
    cp = torch.randn(5, 2)
    y = torch.randn(5, 3)
    with pytest.raises(ValueError, match="shape"):
        CubicSpline1D(cp, y)


def test_cubic_rejects_duplicate_knots():
    cp = torch.tensor([[0.0], [0.0], [1.0]])
    y = torch.randn(3, 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        CubicSpline1D(cp, y)


def test_auto_dispatch_picks_cubic_for_1d_noncyclic():
    cp = torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
    y = torch.randn(6, 4)
    m = SplineManifold(
        control_points=cp,
        target_points=y,
        intrinsic_dim=1,
        ambient_dim=4,
    )
    assert isinstance(m.spline, NaturalCubicSpline1D)
    # Default should still interpolate centroids exactly.
    assert torch.allclose(m.decode(cp), y, atol=1e-4)


def test_auto_dispatch_picks_tps_for_2d():
    cp = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    y = torch.randn(5, 3)
    m = SplineManifold(
        control_points=cp,
        target_points=y,
        intrinsic_dim=2,
        ambient_dim=3,
    )
    assert isinstance(m.spline, ThinPlateSpline)


def test_auto_dispatch_picks_cubic_for_1d_periodic():
    # 1D periodic now goes to the cubic backend in auto mode.
    n = 7
    period = float(n)
    cp = torch.arange(n).float().unsqueeze(-1)
    angles = 2 * math.pi * cp[:, 0] / period
    y = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    m = SplineManifold(
        control_points=cp,
        target_points=y,
        intrinsic_dim=1,
        ambient_dim=2,
        periodic_dims=[0],
        periods=[period],
    )
    assert isinstance(m.spline, CubicSpline1D)
    assert m.spline.bc == "periodic"
    # Interpolates centroids exactly.
    assert torch.allclose(m.decode(cp), y, atol=1e-4)


def test_explicit_cubic_rejects_2d():
    cp = torch.randn(5, 2)
    y = torch.randn(5, 3)
    with pytest.raises(ValueError, match="cubic"):
        SplineManifold(
            control_points=cp,
            target_points=y,
            intrinsic_dim=2,
            ambient_dim=3,
            spline_method="cubic",
        )


def test_explicit_cubic_accepts_periodic():
    n = 7
    cp = torch.arange(n).float().unsqueeze(-1)
    angles = 2 * math.pi * cp[:, 0] / n
    y = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    m = SplineManifold(
        control_points=cp,
        target_points=y,
        intrinsic_dim=1,
        ambient_dim=2,
        periodic_dims=[0],
        periods=[float(n)],
        spline_method="cubic",
    )
    assert isinstance(m.spline, CubicSpline1D)
    assert m.spline.bc == "periodic"


def test_explicit_tps_for_1d_periodic_still_works():
    # Users can still opt into TPS+B4 explicitly for 1D periodic data.
    n = 7
    cp = torch.arange(n).float().unsqueeze(-1)
    y = torch.randn(n, 2)
    m = SplineManifold(
        control_points=cp,
        target_points=y,
        intrinsic_dim=1,
        ambient_dim=2,
        periodic_dims=[0],
        periods=[float(n)],
        spline_method="tps",
    )
    assert isinstance(m.spline, ThinPlateSpline)


def test_explicit_tps_on_1d_noncyclic():
    cp = torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
    y = torch.randn(6, 3)
    m = SplineManifold(
        control_points=cp,
        target_points=y,
        intrinsic_dim=1,
        ambient_dim=3,
        spline_method="tps",
    )
    assert isinstance(m.spline, ThinPlateSpline)


def test_state_dict_roundtrip_preserves_backend():
    cp = torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
    y = torch.randn(6, 4)
    m = SplineManifold(
        control_points=cp,
        target_points=y,
        intrinsic_dim=1,
        ambient_dim=4,
    )
    state = m.state_dict_to_save()
    assert state["spline_method"] == "auto"
    loaded = SplineManifold.from_state_dict(state)
    assert isinstance(loaded.spline, NaturalCubicSpline1D)
    assert torch.allclose(m.decode(cp), loaded.decode(cp), atol=1e-5)


def test_legacy_state_dict_without_spline_method_loads_as_tps():
    # Simulates a checkpoint produced before the spline_method field existed.
    cp = torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
    y = torch.randn(6, 3)
    state = {
        "control_points": cp,
        "centroids": y,
        "intrinsic_dim": 1,
        "ambient_dim": 3,
        "smoothness": 0.0,
        "periodic_dims": None,
        "periods": None,
    }
    loaded = SplineManifold.from_state_dict(state)
    assert isinstance(loaded.spline, ThinPlateSpline)


def test_unrecognized_legacy_method_falls_back_to_tps():
    cp = torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
    y = torch.randn(6, 3)
    state = {
        "control_points": cp,
        "centroids": y,
        "intrinsic_dim": 1,
        "ambient_dim": 3,
        "smoothness": 0.0,
        "periodic_dims": None,
        "periods": None,
        "spline_method": "euclidean_tps",  # legacy, no longer recognized
    }
    loaded = SplineManifold.from_state_dict(state)
    assert isinstance(loaded.spline, ThinPlateSpline)


def test_encode_to_nearest_point_works_with_cubic():
    # The Gauss-Newton projection in SplineManifold uses finite-difference
    # Jacobians on .decode, so it should not depend on the backend.
    torch.manual_seed(0)
    cp = torch.linspace(-1.0, 1.0, 9).unsqueeze(-1)
    y = torch.stack([cp[:, 0], cp[:, 0] ** 2], dim=-1)
    m = SplineManifold(
        control_points=cp,
        target_points=y,
        intrinsic_dim=1,
        ambient_dim=2,
    )
    # A point exactly on the manifold should project back to its preimage.
    u_truth = torch.tensor([[0.3]])
    x_on = m.decode(u_truth)
    u_proj, residual = m.encode_to_nearest_point(x_on, n_iters=20, tol=1e-9)
    assert torch.allclose(u_proj, u_truth, atol=1e-3)
    assert residual.abs().max() < 1e-4


# ---------------------------------------------------------------------------
# Periodic CubicSpline1D tests
# ---------------------------------------------------------------------------


def _periodic_circle_data(n: int = 8, period: float | None = None):
    """Knots evenly placed on a cycle with no duplicated wrap-around."""
    if period is None:
        period = float(n)
    cp = torch.arange(n, dtype=torch.float32).unsqueeze(-1) * (period / n)
    angles = 2 * math.pi * cp[:, 0] / period
    y = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    return cp, y, period


def test_periodic_cubic_interpolates_controls_exactly():
    cp, y, period = _periodic_circle_data(n=8)
    spline = CubicSpline1D(cp, y, bc="periodic", period=period, smoothness=0.0)
    out = spline.evaluate(cp)
    assert torch.allclose(out, y, atol=1e-5), (out - y).abs().max()


def test_periodic_cubic_matches_scipy():
    pytest.importorskip("scipy")
    from scipy.interpolate import CubicSpline as SciCS

    n = 10
    period = 2 * math.pi
    cp = torch.linspace(0.0, period, n + 1)[:-1].unsqueeze(-1)  # no duplicate seam
    y = torch.stack([torch.cos(cp[:, 0]), torch.sin(cp[:, 0])], dim=-1)
    spline = CubicSpline1D(cp, y, bc="periodic", period=period, smoothness=0.0)

    # scipy requires y[0] == y[-1] for bc_type="periodic"; build that closed form.
    cp_closed = torch.cat([cp[:, 0], torch.tensor([period])])
    y_closed = torch.cat([y, y[:1]], dim=0)
    sci = SciCS(cp_closed.numpy(), y_closed.numpy(), bc_type="periodic")

    u = torch.linspace(0.0, period, 100).unsqueeze(-1)
    ours = spline.evaluate(u).numpy()
    ref = sci(u[:, 0].numpy())
    assert ours.shape == ref.shape
    assert abs(ours - ref).max() < 1e-5


def test_periodic_cubic_wraps():
    cp, y, period = _periodic_circle_data(n=12)
    spline = CubicSpline1D(cp, y, bc="periodic", period=period, smoothness=0.0)

    u = torch.linspace(0.0, period, 50).unsqueeze(-1)
    v_in = spline.evaluate(u)
    v_shift = spline.evaluate(u + period)
    v_neg = spline.evaluate(u - period)
    assert torch.allclose(v_in, v_shift, atol=1e-5)
    assert torch.allclose(v_in, v_neg, atol=1e-5)


def test_periodic_cubic_c2_at_seam():
    cp, y, period = _periodic_circle_data(n=12)
    # Use double precision so FD-estimated second derivative isn't dominated
    # by float32 roundoff at small step sizes.
    sp = CubicSpline1D(
        cp.double(),
        y.double(),
        bc="periodic",
        period=period,
        smoothness=0.0,
    )

    seam = 0.0
    h = (period / cp.shape[0]) / 16  # well inside the segments on both sides

    def at(u_val):
        return sp.evaluate(torch.tensor([[u_val]], dtype=torch.float64))

    # Value continuity across seam.
    v_l = at(seam - 1e-9)
    v_r = at(seam + 1e-9)
    assert torch.allclose(v_l, v_r, atol=1e-7)

    # Approach f'(0) and f''(0) from each side using one-sided second-order
    # FD schemes (sample points only on the corresponding segment).
    # Backward 2nd-order: f'(x) ≈ [3 f(x) - 4 f(x-h) + f(x-2h)] / (2h)
    # Forward  2nd-order: f'(x) ≈ [-3 f(x) + 4 f(x+h) - f(x+2h)] / (2h)
    d_l = (3 * at(seam) - 4 * at(seam - h) + at(seam - 2 * h)) / (2 * h)
    d_r = (-3 * at(seam) + 4 * at(seam + h) - at(seam + 2 * h)) / (2 * h)
    assert torch.allclose(d_l, d_r, atol=1e-3), (d_l - d_r).abs().max()

    # 3-point FD for f'' on each side.
    sd_l = (at(seam) - 2 * at(seam - h) + at(seam - 2 * h)) / (h**2)
    sd_r = (at(seam + 2 * h) - 2 * at(seam + h) + at(seam)) / (h**2)
    assert torch.allclose(sd_l, sd_r, atol=5e-2), (sd_l - sd_r).abs().max()


def test_periodic_cubic_smoothing_reduces_roughness():
    torch.manual_seed(2)
    n = 24
    period = 2 * math.pi
    cp = torch.linspace(0.0, period, n + 1)[:-1].unsqueeze(-1)
    clean = torch.stack([torch.cos(cp[:, 0]), torch.sin(cp[:, 0])], dim=-1)
    noise = 0.05 * torch.randn_like(clean)
    y = clean + noise

    interp = CubicSpline1D(cp, y, bc="periodic", period=period, smoothness=0.0)
    smooth = CubicSpline1D(cp, y, bc="periodic", period=period, smoothness=1e-2)

    assert torch.allclose(interp.evaluate(cp), y, atol=1e-5)
    assert (smooth.evaluate(cp) - y).abs().max() > 1e-4
    assert (smooth.gamma**2).sum() < (interp.gamma**2).sum()


def test_periodic_cubic_rejects_missing_period():
    cp, y, _ = _periodic_circle_data(n=6)
    with pytest.raises(ValueError, match="period"):
        CubicSpline1D(cp, y, bc="periodic")


def test_periodic_cubic_rejects_period_le_span():
    cp, y, _ = _periodic_circle_data(n=6, period=6.0)
    span = (cp.max() - cp.min()).item()
    with pytest.raises(ValueError, match="period"):
        CubicSpline1D(cp, y, bc="periodic", period=span)  # period must exceed span


def test_natural_cubic_rejects_period():
    cp, y = _quadratic_data(n=6)
    with pytest.raises(ValueError, match="period"):
        CubicSpline1D(cp, y, bc="natural", period=10.0)


def test_cubic_rejects_invalid_bc():
    cp, y = _quadratic_data(n=6)
    with pytest.raises(ValueError, match="bc"):
        CubicSpline1D(cp, y, bc="clamped")


def test_state_dict_roundtrip_periodic_cubic():
    # Ensure auto-dispatch + persisted spline_method give cubic on reload.
    n = 7
    period = float(n)
    cp = torch.arange(n).float().unsqueeze(-1)
    angles = 2 * math.pi * cp[:, 0] / period
    y = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    m = SplineManifold(
        control_points=cp,
        target_points=y,
        intrinsic_dim=1,
        ambient_dim=2,
        periodic_dims=[0],
        periods=[period],
    )
    state = m.state_dict_to_save()
    loaded = SplineManifold.from_state_dict(state)
    assert isinstance(loaded.spline, CubicSpline1D)
    assert loaded.spline.bc == "periodic"
    u = torch.linspace(0.0, period, 25).unsqueeze(-1)
    assert torch.allclose(m.decode(u), loaded.decode(u), atol=1e-5)


def test_natural_cubic_spline_alias():
    # Backward-compat: NaturalCubicSpline1D should still construct a
    # natural-BC cubic spline.
    cp, y = _quadratic_data(n=6)
    a = NaturalCubicSpline1D(cp, y)
    assert isinstance(a, CubicSpline1D)
    assert a.bc == "natural"
