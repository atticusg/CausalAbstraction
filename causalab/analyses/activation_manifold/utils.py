"""Manifold utilities — featurizer loading and intrinsic range computation."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _get_manifold_device(manifold: Any) -> torch.device:
    """Get device from a manifold (works for flow and spline)."""
    p = next(manifold.parameters(), None)
    if p is not None:
        return p.device
    b = next(manifold.buffers(), None)
    if b is not None:
        return b.device
    return torch.device("cpu")


def _compute_intrinsic_ranges(
    features: Tensor,
    manifold: Any,
    mean: Tensor,
    std: Tensor,
    coverage: float = 0.997,
) -> tuple[tuple[float, float], ...]:
    """Compute intrinsic coordinate ranges from encoded features.

    Encodes all features through the manifold and takes quantiles.
    Works for both flows (unbounded latent) and splines (snaps to
    nearest centroid, giving control point min/max).
    """
    device = _get_manifold_device(manifold)
    features_d = features.to(device)
    mean_d = mean.to(device)
    std_d = std.to(device)

    features_norm = (features_d - mean_d) / (std_d + 1e-6)
    with torch.no_grad():
        u, _ = manifold.encode(features_norm)

    d = u.shape[1]
    lower_pct = (1 - coverage) / 2
    upper_pct = (1 + coverage) / 2

    # Read periodic info from manifold
    periodic_dims: set[int] = set()
    periods: dict[int, float] = {}
    if hasattr(manifold, "periodic_dims") and hasattr(manifold, "periods"):
        for pd, per in zip(manifold.periodic_dims, manifold.periods):
            periodic_dims.add(pd)
            periods[pd] = per

    ranges: list[tuple[float, float]] = []
    for dim in range(d):
        if dim in periodic_dims:
            # Use full period so mesh/curve wraps smoothly
            ranges.append((0.0, periods[dim]))
        else:
            u_dim = u[:, dim]
            dim_min = float(torch.quantile(u_dim, lower_pct).item())
            dim_max = float(torch.quantile(u_dim, upper_pct).item())
            ranges.append((dim_min, dim_max))

    logger.info(
        f"Intrinsic ranges ({coverage * 100:.1f}% coverage): "
        + ", ".join(f"dim{i}=[{lo:.3f}, {hi:.3f}]" for i, (lo, hi) in enumerate(ranges))
    )
    return tuple(ranges)
