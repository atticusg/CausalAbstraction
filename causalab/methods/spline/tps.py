"""Thin-Plate Spline implementation in PyTorch with polynomial term."""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
import warnings


def thin_plate_kernel(r: Tensor, dim: int) -> Tensor:
    r_safe = torch.clamp(r, min=1e-10)
    if dim == 2:
        kernel = r_safe**2 * torch.log(r_safe)
    else:
        kernel = r_safe ** (dim - 2) * torch.log(r_safe)
    kernel = torch.where(r < 1e-10, torch.zeros_like(kernel), kernel)
    return kernel


def _bernoulli4_kernel(d: Tensor, period: float) -> Tensor:
    """B4 Bernoulli polynomial kernel for periodic dimension.

    k(d) = B_4(d/T) where B_4(x) = x^4 - 2x^3 + x^2 - 1/30
    and d is the circular distance wrapped to [0, T/2].
    """
    # Normalize to [0, 1] using circular distance
    x = (d % period) / period
    # Fold to [0, 0.5] for symmetry
    x = torch.min(x, 1.0 - x)
    # B_4(x) = x^4 - 2x^3 + x^2 - 1/30
    return x**4 - 2.0 * x**3 + x**2 - 1.0 / 30.0


class ThinPlateSpline(nn.Module):
    """Kernel interpolation with periodic dimensions.

    Uses Bernoulli polynomial kernel for periodic dimensions and
    standard TPS kernel for linear dimensions, combined as a product kernel.

    Args:
        control_points: (n, d) control point coordinates.
        values: (n, ambient_dim) target values.
        periodic_dims: Which dimensions are periodic. Accepts:
            - None or [] → no periodic dims
            - list[int] → indices of periodic dims (e.g. [0, 2])
            - tuple[bool, ...] → boolean mask (e.g. (True, False))
        periods: Period for each periodic dimension. Defaults to [1.0] per
            periodic dim if not provided.
        smoothness: Regularization parameter (0 = exact interpolation).
    """

    def __init__(
        self,
        control_points: Tensor,
        values: Tensor,
        smoothness: float = 0.0,
        periodic_dims: list[int] | Tuple[bool, ...] | None = None,
        periods: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("control_points", control_points)
        self.register_buffer("values", values)
        self.smoothness = smoothness
        self.intrinsic_dim = control_points.shape[1]
        self.ambient_dim = values.shape[1]

        # Normalize periodic_dims to list[int]
        if periodic_dims is None:
            self.periodic_dims: list[int] = []
        elif periodic_dims and isinstance(periodic_dims[0], bool):
            self.periodic_dims = [i for i, p in enumerate(periodic_dims) if p]
        else:
            self.periodic_dims = list(periodic_dims)

        # Default periods to 1.0 per periodic dim
        if periods is None:
            self.periods = [1.0] * len(self.periodic_dims)
        else:
            self.periods = list(periods)

        # Identify linear dims
        all_dims = set(range(self.intrinsic_dim))
        self.linear_dims = sorted(all_dims - set(self.periodic_dims))

        self._fit()

    def _compute_kernel_matrix(self, A: Tensor, B: Tensor) -> Tensor:
        """Compute kernel between point sets A and B.

        When periodic dims are present, computes a topology-aware distance
        (circular for periodic dims, Euclidean for linear dims) and applies
        the standard TPS kernel on the combined distance. This avoids the
        product-kernel zero problem where the linear kernel vanishes at
        equal-coordinate pairs.

        When all dims are linear, uses the standard TPS kernel.
        When all dims are periodic, uses the product B4 kernel.
        """
        if not self.periodic_dims:
            # Pure linear: standard TPS
            dists = torch.cdist(A, B, p=2)
            return thin_plate_kernel(dists, dim=self.intrinsic_dim)

        if not self.linear_dims:
            # Pure periodic: product B4 kernel
            n_a, n_b = A.shape[0], B.shape[0]
            K = torch.ones(n_a, n_b, device=A.device, dtype=A.dtype)
            for dim_idx, period in zip(self.periodic_dims, self.periods):
                diff = A[:, dim_idx : dim_idx + 1] - B[:, dim_idx : dim_idx + 1].T
                K = K * _bernoulli4_kernel(diff.abs(), period)
            return K

        # Mixed: fall through to standard TPS on full dim
        # (Ghost points handle periodicity at the SplineManifold level)
        dists = torch.cdist(A, B, p=2)
        return thin_plate_kernel(dists, dim=self.intrinsic_dim)

    def _build_polynomial(self, points: Tensor) -> Tensor:
        """Build polynomial matrix: [1, x_linear_dims].

        Constant column always included. Linear terms only for non-periodic dims.
        """
        n = points.shape[0]
        device = points.device
        dtype = points.dtype

        cols = [torch.ones(n, 1, device=device, dtype=dtype)]
        for dim in self.linear_dims:
            cols.append(points[:, dim : dim + 1])

        return torch.cat(cols, dim=1)

    def _fit(self) -> None:
        n = self.control_points.shape[0]
        device = self.control_points.device
        dtype = self.control_points.dtype

        if n > 10000:
            warnings.warn(f"Fitting TPS with {n} control points. This may be slow.")

        K = self._compute_kernel_matrix(self.control_points, self.control_points)
        if self.smoothness > 0:
            K = K + self.smoothness * torch.eye(n, device=device, dtype=dtype)

        P = self._build_polynomial(self.control_points)
        m = P.shape[1]

        A = torch.zeros(n + m, n + m, device=device, dtype=dtype)
        A[:n, :n] = K
        A[:n, n:] = P
        A[n:, :n] = P.T

        rhs = torch.zeros(n + m, self.ambient_dim, device=device, dtype=dtype)
        rhs[:n] = self.values

        coeffs = torch.linalg.solve(A, rhs)
        self.register_buffer("rbf_weights", coeffs[:n])
        self.register_buffer("poly_coeffs", coeffs[n:])

    def evaluate(self, u: Tensor) -> Tensor:
        """Evaluate the TPS at query points u.

        Args:
            u: (batch, intrinsic_dim) query coordinates.

        Returns:
            (batch, ambient_dim) interpolated values.
        """
        K_query = self._compute_kernel_matrix(u, self.control_points)
        P_query = self._build_polynomial(u)
        return (K_query @ self.rbf_weights) + (P_query @ self.poly_coeffs)

    def forward(self, u: Tensor) -> Tensor:
        return self.evaluate(u)

    @property
    def n_control_points(self) -> int:
        return self.control_points.shape[0]
