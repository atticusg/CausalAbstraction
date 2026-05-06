"""Cubic spline for 1D interpolation/smoothing with natural or periodic BCs.

The 1D analog of the thin-plate spline: minimizes the bending energy
``∫ f''(x)² dx`` over functions interpolating (or, with ``smoothness > 0``,
penalty-fitting) the control points.

- ``bc="natural"`` enforces ``f''(x_0) = f''(x_{n-1}) = 0`` and extrapolates
  linearly outside the knot range.
- ``bc="periodic"`` treats the knots as a cycle of length ``period`` so all
  derivatives match across the seam ``x_{n-1} → x_0 + period``. Queries are
  wrapped modulo ``period``; no extrapolation correction.

This module provides ``CubicSpline1D`` (with ``NaturalCubicSpline1D`` kept as
a backward-compat alias) sharing the eval contract of ``ThinPlateSpline``:
``.evaluate(u) -> Tensor`` taking ``(batch, 1)`` queries and returning
``(batch, ambient_dim)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


_VALID_BC = ("natural", "periodic")


class CubicSpline1D(nn.Module):
    """Cubic spline over a single intrinsic dimension.

    Args:
        control_points: ``(n, 1)`` knot coordinates. Need not be sorted.
        values: ``(n, ambient_dim)`` target values at the knots. For periodic
            mode the knots must NOT include a duplicated wrap-around copy of
            ``x_0`` — the cycle is closed implicitly.
        smoothness: Reinsch smoothing parameter ``λ ≥ 0``. ``0`` performs
            exact interpolation; positive values trade off fit vs roughness
            via the penalty ``λ ∫ f''²``.
        bc: ``"natural"`` (default) or ``"periodic"``.
        period: Required when ``bc="periodic"``; must satisfy
            ``period > x_max - x_min``. Must be ``None`` for natural mode.
    """

    def __init__(
        self,
        control_points: Tensor,
        values: Tensor,
        smoothness: float = 0.0,
        bc: str = "natural",
        period: float | None = None,
    ) -> None:
        super().__init__()
        if control_points.ndim != 2 or control_points.shape[1] != 1:
            raise ValueError(
                f"CubicSpline1D requires control_points of shape (n, 1), "
                f"got {tuple(control_points.shape)}"
            )
        n = control_points.shape[0]
        if n < 2:
            raise ValueError(f"need at least 2 control points, got {n}")
        if values.shape[0] != n:
            raise ValueError(
                f"values has {values.shape[0]} rows but control_points has {n}"
            )
        if smoothness < 0:
            raise ValueError(f"smoothness must be non-negative, got {smoothness}")
        if bc not in _VALID_BC:
            raise ValueError(f"bc must be one of {_VALID_BC}, got {bc!r}")

        if bc == "periodic":
            if period is None:
                raise ValueError("bc='periodic' requires a period")
            if n < 3:
                raise ValueError(
                    f"periodic mode needs at least 3 control points, got {n}"
                )
        else:
            if period is not None:
                raise ValueError(
                    f"period must be None for bc='natural' (got period={period})"
                )

        self.smoothness = float(smoothness)
        self.ambient_dim = values.shape[1]
        self.bc = bc
        self.period = float(period) if period is not None else None

        # Sort knots ascending; carry values along.
        x_raw = control_points[:, 0]
        sort_idx = torch.argsort(x_raw)
        x_sorted = x_raw[sort_idx].contiguous()
        y_sorted = values[sort_idx].contiguous()

        # Reject duplicate knots — the system would be singular.
        h_natural = x_sorted[1:] - x_sorted[:-1]
        if torch.any(h_natural <= 0):
            raise ValueError(
                "control_points must be strictly increasing after sorting "
                "(no duplicates)"
            )

        if bc == "periodic":
            span = (x_sorted[-1] - x_sorted[0]).item()
            if self.period <= span:
                raise ValueError(
                    f"period={self.period} must be strictly greater than the "
                    f"knot span x_max - x_min = {span}"
                )
            # Spacing vector for periodic mode is length n with wrap-around gap.
            wrap_gap = self.period - span
            h = torch.cat(
                [
                    h_natural,
                    torch.tensor(
                        [wrap_gap], device=h_natural.device, dtype=h_natural.dtype
                    ),
                ]
            )
        else:
            h = h_natural

        self.register_buffer("control_points", control_points)
        self.register_buffer("values", values)
        self.register_buffer("x_sorted", x_sorted)
        self.register_buffer("h", h)

        if bc == "periodic":
            gamma, y_hat = self._fit_periodic(y_sorted, h, self.smoothness)
        else:
            gamma, y_hat = self._fit_natural(x_sorted, y_sorted, h, self.smoothness)
        self.register_buffer("gamma", gamma)
        self.register_buffer("y_hat", y_hat)

    @staticmethod
    def _fit_natural(
        x: Tensor, y: Tensor, h: Tensor, lam: float
    ) -> tuple[Tensor, Tensor]:
        """Solve for second derivatives γ at all knots and fitted values ŷ.

        Uses the Reinsch (1967) formulation. With γ_0 = γ_{n-1} = 0 (natural
        BC), the interior γ_1..γ_{n-2} satisfy

            (R + λ Qᵀ Q) γ_int = Qᵀ y,

        and the smoothed knot values are ŷ = y − λ Q γ_int. When λ = 0 the
        system collapses to plain natural-cubic-spline interpolation: ŷ = y
        and R γ_int = Qᵀ y.
        """
        n = x.shape[0]
        device, dtype = y.device, y.dtype
        ambient = y.shape[1]

        if n == 2:
            # Degenerate: only the two endpoints, both have γ = 0 (linear fit).
            gamma = torch.zeros(n, ambient, device=device, dtype=dtype)
            return gamma, y.clone()

        m = n - 2  # interior knot count

        # Build R (m×m, tridiagonal, symmetric).
        diag_R = (h[:-1] + h[1:]) / 3.0  # (m,)
        off_R = h[1:-1] / 6.0  # (m-1,)
        R = torch.diag(diag_R) + torch.diag(off_R, 1) + torch.diag(off_R, -1)

        # Compute Qᵀ y directly without materializing Q.
        # Q has columns indexed by interior knot j (j = 1..n-2):
        #   Q[j-1, j-1] = 1/h[j-1]
        #   Q[j,   j-1] = -(1/h[j-1] + 1/h[j])
        #   Q[j+1, j-1] = 1/h[j]
        # So (Qᵀ y)[j-1] = y[j-1]/h[j-1] - (1/h[j-1] + 1/h[j]) y[j] + y[j+1]/h[j].
        inv_h = 1.0 / h  # (n-1,)
        Qt_y = (
            y[:-2] * inv_h[:-1].unsqueeze(-1)
            - y[1:-1] * (inv_h[:-1] + inv_h[1:]).unsqueeze(-1)
            + y[2:] * inv_h[1:].unsqueeze(-1)
        )  # (m, ambient)

        if lam == 0.0:
            A = R
        else:
            Q = torch.zeros(n, m, device=device, dtype=dtype)
            for j in range(m):
                Q[j, j] = inv_h[j]
                Q[j + 1, j] = -(inv_h[j] + inv_h[j + 1])
                Q[j + 2, j] = inv_h[j + 1]
            A = R + lam * (Q.T @ Q)

        gamma_int = torch.linalg.solve(A, Qt_y)  # (m, ambient)

        gamma = torch.zeros(n, ambient, device=device, dtype=dtype)
        gamma[1:-1] = gamma_int

        if lam == 0.0:
            y_hat = y.clone()
        else:
            Qg = torch.zeros_like(y)
            for j in range(m):
                Qg[j] += inv_h[j] * gamma_int[j]
                Qg[j + 1] += -(inv_h[j] + inv_h[j + 1]) * gamma_int[j]
                Qg[j + 2] += inv_h[j + 1] * gamma_int[j]
            y_hat = y - lam * Qg

        return gamma, y_hat

    @staticmethod
    def _fit_periodic(y: Tensor, h: Tensor, lam: float) -> tuple[Tensor, Tensor]:
        """Cyclic Reinsch system. Same structure as natural but with wrap-around.

        For γ ∈ ℝⁿ (no boundary reduction), the cyclic n×n second-difference
        operator Q and bending matrix R satisfy R γ = Qᵀ y for interpolation
        (λ = 0) and (R + λ Qᵀ Q) γ = Qᵀ y, ŷ = y − λ Q γ for smoothing.

        Indices wrap modulo n; ``h`` has length n where ``h[n-1]`` is the
        wrap-around gap ``period - (x_{n-1} - x_0)``.
        """
        n = y.shape[0]
        device, dtype = y.device, y.dtype
        ambient = y.shape[1]
        inv_h = 1.0 / h  # (n,)

        # h_prev[i] = h[(i-1) mod n], h_next[i] = h[i]
        h_prev = torch.roll(h, shifts=1, dims=0)
        inv_h_prev = torch.roll(inv_h, shifts=1, dims=0)
        inv_h_next = inv_h

        # R (n×n cyclic tridiagonal): diag = (h_prev + h)/3, off = h/6 between
        # i and i+1 (mod n).
        diag_R = (h_prev + h) / 3.0
        R = torch.diag(diag_R)
        for i in range(n):
            j = (i + 1) % n
            R[i, j] += h[i] / 6.0
            R[j, i] += h[i] / 6.0

        # Q (n×n cyclic): column j has entries
        #   Q[(j-1) mod n, j] = 1/h[(j-1) mod n]
        #   Q[j, j]           = -(1/h[(j-1) mod n] + 1/h[j])
        #   Q[(j+1) mod n, j] = 1/h[j]
        # Build Qᵀ y in closed form:
        y_prev = torch.roll(y, shifts=1, dims=0)
        y_next = torch.roll(y, shifts=-1, dims=0)
        Qt_y = (
            y_prev * inv_h_prev.unsqueeze(-1)
            - y * (inv_h_prev + inv_h_next).unsqueeze(-1)
            + y_next * inv_h_next.unsqueeze(-1)
        )

        if lam == 0.0:
            A = R
        else:
            Q = torch.zeros(n, n, device=device, dtype=dtype)
            for j in range(n):
                Q[(j - 1) % n, j] += inv_h[(j - 1) % n]
                Q[j, j] += -(inv_h[(j - 1) % n] + inv_h[j])
                Q[(j + 1) % n, j] += inv_h[j]
            A = R + lam * (Q.T @ Q)

        # The cyclic system is rank n-1 (constants are in the null space of
        # the second-difference operator). Solve in least-squares sense; for
        # interpolation that recovers the unique zero-mean γ.
        try:
            gamma = torch.linalg.solve(A, Qt_y)
        except torch._C._LinAlgError:
            gamma = torch.linalg.lstsq(A, Qt_y).solution

        if lam == 0.0:
            y_hat = y.clone()
        else:
            gamma_prev = torch.roll(gamma, shifts=1, dims=0)
            gamma_next = torch.roll(gamma, shifts=-1, dims=0)
            Qg = (
                inv_h_prev.unsqueeze(-1) * gamma_prev
                - (inv_h_prev + inv_h_next).unsqueeze(-1) * gamma
                + inv_h_next.unsqueeze(-1) * gamma_next
            )
            y_hat = y - lam * Qg

        return gamma, y_hat

    @staticmethod
    def _eval_segment(
        u: Tensor,
        x_left: Tensor,
        x_right: Tensor,
        y_l: Tensor,
        y_r: Tensor,
        g_l: Tensor,
        g_r: Tensor,
    ) -> Tensor:
        """Standard cubic-spline segment formula in (a, b) form. ``u`` is 1D."""
        h_seg = (x_right - x_left).unsqueeze(-1)
        a = (x_right - u).unsqueeze(-1)
        b = (u - x_left).unsqueeze(-1)
        return (
            g_l * (a**3) / (6.0 * h_seg)
            + g_r * (b**3) / (6.0 * h_seg)
            + (y_l / h_seg - g_l * h_seg / 6.0) * a
            + (y_r / h_seg - g_r * h_seg / 6.0) * b
        )

    def evaluate(self, u: Tensor) -> Tensor:
        """Evaluate the spline at query points.

        Args:
            u: ``(batch, 1)`` query coordinates.

        Returns:
            ``(batch, ambient_dim)`` interpolated/smoothed values.
        """
        if u.ndim != 2 or u.shape[1] != 1:
            raise ValueError(
                f"CubicSpline1D.evaluate expects (batch, 1), got {tuple(u.shape)}"
            )

        if self.bc == "periodic":
            return self._evaluate_periodic(u[:, 0])
        return self._evaluate_natural(u[:, 0])

    def _evaluate_natural(self, u_flat: Tensor) -> Tensor:
        x = self.x_sorted
        h = self.h
        gamma = self.gamma
        y_hat = self.y_hat
        n = x.shape[0]

        u_clamped = u_flat.clamp(x[0], x[-1])
        idx = torch.searchsorted(x, u_clamped, right=True) - 1
        idx = idx.clamp(0, n - 2)

        out = self._eval_segment(
            u_clamped,
            x[idx],
            x[idx + 1],
            y_hat[idx],
            y_hat[idx + 1],
            gamma[idx],
            gamma[idx + 1],
        )

        # Linear extrapolation outside the knot range. γ_0 = γ_{n-1} = 0 so the
        # boundary slopes simplify accordingly.
        h0 = h[0]
        hn = h[-1]
        slope_left = (y_hat[1] - y_hat[0]) / h0 - h0 * gamma[1] / 6.0
        slope_right = (y_hat[-1] - y_hat[-2]) / hn + hn * gamma[-2] / 6.0

        delta_left = (u_flat - x[0]).clamp(max=0.0).unsqueeze(-1)
        delta_right = (u_flat - x[-1]).clamp(min=0.0).unsqueeze(-1)
        out = (
            out
            + slope_left.unsqueeze(0) * delta_left
            + slope_right.unsqueeze(0) * delta_right
        )

        return out

    def _evaluate_periodic(self, u_flat: Tensor) -> Tensor:
        x = self.x_sorted
        h = self.h
        gamma = self.gamma
        y_hat = self.y_hat
        n = x.shape[0]
        period = self.period
        x0 = x[0]

        # Wrap query into [x_0, x_0 + period).
        u_wrapped = ((u_flat - x0) % period) + x0

        # Augmented knot array of length n+1 with the seam at x_0 + period.
        x_aug = torch.cat([x, (x0 + period).unsqueeze(0)])

        idx = torch.searchsorted(x_aug, u_wrapped, right=True) - 1
        idx = idx.clamp(0, n - 1)

        idx_next = (idx + 1) % n  # wrap for y/gamma fetch
        x_left = x_aug[idx]
        x_right = x_aug[idx + 1]

        return self._eval_segment(
            u_wrapped,
            x_left,
            x_right,
            y_hat[idx],
            y_hat[idx_next],
            gamma[idx],
            gamma[idx_next],
        )

    def forward(self, u: Tensor) -> Tensor:
        return self.evaluate(u)

    @property
    def n_control_points(self) -> int:
        return self.control_points.shape[0]


# Backward-compat alias. "Natural" no longer fits when bc="periodic", but
# downstream code that imported the old name keeps working as long as it
# only constructs natural-mode splines (the default).
NaturalCubicSpline1D = CubicSpline1D
