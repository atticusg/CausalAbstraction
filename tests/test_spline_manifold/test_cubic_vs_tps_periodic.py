"""Comparison of cubic vs TPS+B4 backends on 1D periodic data.

Not a regression test — runs on demand to compare the two backends'
reconstruction accuracy, mid-segment accuracy, roughness proxy, wrap
continuity, and fit/eval time. Marked ``@pytest.mark.benchmark``; run with

    uv run pytest tests/test_spline_manifold/test_cubic_vs_tps_periodic.py -s -m benchmark
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from causalab.methods.spline import SplineManifold


def _circle_data(n: int, noise: float = 0.0, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    period = 2 * math.pi
    cp = torch.linspace(0.0, period, n + 1)[:-1].unsqueeze(-1)
    y = torch.stack([torch.cos(cp[:, 0]), torch.sin(cp[:, 0])], dim=-1)
    if noise > 0:
        y = y + noise * torch.randn(y.shape, generator=g)
    return cp, y, period


def _eval_dense(manifold: SplineManifold, period: float, n_pts: int = 1024):
    u = torch.linspace(0.0, period, n_pts + 1)[:-1].unsqueeze(-1)
    return u, manifold.decode(u)


def _midpoint_error(manifold: SplineManifold, cp: torch.Tensor, period: float):
    """Mid-segment reconstruction error vs the ground-truth circle."""
    x_aug = torch.cat([cp[:, 0], torch.tensor([cp[0, 0].item() + period])])
    mids = (x_aug[:-1] + x_aug[1:]) / 2.0
    truth = torch.stack([torch.cos(mids), torch.sin(mids)], dim=-1)
    pred = manifold.decode(mids.unsqueeze(-1))
    err = (pred - truth).norm(dim=-1)
    return err.mean().item(), err.max().item()


def _roughness(manifold: SplineManifold, period: float, n_pts: int = 2048):
    """Numerical estimate of ∫ ||f''(x)||² dx via 3-point FD on dense grid."""
    u, fu = _eval_dense(manifold, period, n_pts)
    h = period / n_pts
    f_prev = torch.roll(fu, shifts=1, dims=0)
    f_next = torch.roll(fu, shifts=-1, dims=0)
    f_dd = (f_next - 2 * fu + f_prev) / (h**2)
    return (f_dd.norm(dim=-1) ** 2).mean().item() * period


def _wrap_continuity(manifold: SplineManifold, period: float):
    eps = 1e-3
    a = manifold.decode(torch.tensor([[0.0]]))
    b = manifold.decode(torch.tensor([[period]]))
    val_jump = (a - b).norm().item()
    d_l = (
        manifold.decode(torch.tensor([[0.0]])) - manifold.decode(torch.tensor([[-eps]]))
    ) / eps
    d_r = (
        manifold.decode(torch.tensor([[eps]])) - manifold.decode(torch.tensor([[0.0]]))
    ) / eps
    slope_jump = (d_l - d_r).norm().item()
    return val_jump, slope_jump


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [6, 12, 24])
def test_cubic_vs_tps_periodic(n: int):
    cp, y, period = _circle_data(n=n)

    rows = []
    for method in ("cubic", "tps"):
        t0 = time.perf_counter()
        m = SplineManifold(
            control_points=cp,
            target_points=y,
            intrinsic_dim=1,
            ambient_dim=2,
            periodic_dims=[0],
            periods=[period],
            spline_method=method,
        )
        fit_time = time.perf_counter() - t0

        # Reconstruction at controls.
        recon_err = (m.decode(cp) - y).norm(dim=-1).max().item()

        mid_mean, mid_max = _midpoint_error(m, cp, period)
        rough = _roughness(m, period)
        val_jump, slope_jump = _wrap_continuity(m, period)

        t0 = time.perf_counter()
        u_dense = torch.linspace(0.0, period, 10_000).unsqueeze(-1)
        _ = m.decode(u_dense)
        eval_time = time.perf_counter() - t0

        rows.append(
            (
                method,
                recon_err,
                mid_mean,
                mid_max,
                rough,
                val_jump,
                slope_jump,
                fit_time,
                eval_time,
            )
        )

    print(f"\n=== n={n} knots on unit circle ===")
    header = (
        f"{'method':<6} {'recon_err':>10} {'mid_mean':>10} {'mid_max':>10} "
        f"{'rough':>10} {'val_jump':>10} {'slope_jump':>10} "
        f"{'fit_s':>8} {'eval_s':>8}"
    )
    print(header)
    for r in rows:
        print(
            f"{r[0]:<6} {r[1]:>10.2e} {r[2]:>10.2e} {r[3]:>10.2e} "
            f"{r[4]:>10.2e} {r[5]:>10.2e} {r[6]:>10.2e} "
            f"{r[7]:>8.4f} {r[8]:>8.4f}"
        )

    # Sanity: both backends should interpolate controls to high accuracy.
    by_method = {r[0]: r for r in rows}
    assert by_method["cubic"][1] < 1e-4, by_method["cubic"][1]
    assert by_method["tps"][1] < 1e-3, by_method["tps"][1]

    # Both backends should be C¹ across the seam (slope jump small).
    assert by_method["cubic"][5] < 1e-3
    assert by_method["cubic"][6] < 1e-1
