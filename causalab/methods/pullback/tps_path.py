"""TPS (cubic-spline) path parameterization for pullback optimization.

Replaces the per-step "free points" parameterization with a smooth natural
cubic spline whose control points are the learnable variables. Reduces the
optimization dimensionality and acts as an implicit smoothness prior on the
activation-space trajectory.

The forward pass solves the natural-cubic-spline system through the current
control values (autograd-friendly via ``torch.linalg.solve``) and evaluates
the resulting spline at a fixed grid of eval-t fractions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from causalab.methods.spline.cubic import CubicSpline1D


class TPSPathModule(nn.Module):
    """1D natural cubic spline parameterizing a path in PCA k-space.

    Args:
        n_control: Number of control points (≥ 2). Optimization variables
            live at uniform t-fractions linspace(0, 1, n_control).
        n_eval: Number of path evaluation points (typically the existing
            ``n_steps_interior`` of the optimizer).
        k_features: Ambient dimensionality (e.g. PCA k_features for pullback).
        smoothness: Reinsch λ. ``0`` = exact interpolation through controls.
            ``> 0`` adds a smoothing penalty.
    """

    def __init__(
        self,
        n_control: int,
        n_eval: int,
        k_features: int,
        smoothness: float = 0.0,
    ) -> None:
        super().__init__()
        if n_control < 2:
            raise ValueError(f"need at least 2 control points, got {n_control}")
        if n_eval < 2:
            raise ValueError(f"need at least 2 eval points, got {n_eval}")

        self.n_control = n_control
        self.n_eval = n_eval
        self.k_features = k_features
        self.smoothness = float(smoothness)

        # Fixed knot positions (control t) and eval positions (eval t) — all
        # uniform on [0, 1]. Knot/eval positions never change during
        # optimization, so we can precompute the segment-index lookup.
        x_knots = torch.linspace(0.0, 1.0, n_control)
        eval_t = torch.linspace(0.0, 1.0, n_eval)
        h = x_knots[1:] - x_knots[:-1]
        idx = torch.searchsorted(x_knots, eval_t, right=True) - 1
        idx = idx.clamp(0, n_control - 2)
        self.register_buffer("x_knots", x_knots)
        self.register_buffer("eval_t", eval_t)
        self.register_buffer("h", h)
        self.register_buffer("eval_seg_idx", idx)

        # Learnable control values.
        self.values = nn.Parameter(torch.zeros(n_control, k_features))

    @torch.no_grad()
    def initialize_from_path(self, path: Tensor) -> None:
        """Linearly interpolate a (K, k_features) init path at the control t-positions.

        ``path`` is assumed to live on uniform t-positions in [0, 1] (which is
        what the existing optimizers produce). We sample at exactly the
        control-t positions so the spline can interpolate them faithfully —
        otherwise the assumed-uniform knot spacing would not match the actual
        sample positions and the spline would deviate from the init path.
        """
        if path.shape[-1] != self.k_features:
            raise ValueError(
                f"path k-dim {path.shape[-1]} != k_features {self.k_features}"
            )
        K = path.shape[0]
        if K == self.n_control:
            self.values.data.copy_(path.to(self.values.device, self.values.dtype))
            return
        path_dev = path.to(self.values.device, self.values.dtype)
        t_path = torch.linspace(
            0.0, 1.0, K, device=path_dev.device, dtype=path_dev.dtype
        )
        t_ctrl = torch.linspace(
            0.0,
            1.0,
            self.n_control,
            device=path_dev.device,
            dtype=path_dev.dtype,
        )
        idx = torch.searchsorted(t_path, t_ctrl, right=True) - 1
        idx = idx.clamp(0, K - 2)
        t_lo = t_path[idx]
        t_hi = t_path[idx + 1]
        alpha = ((t_ctrl - t_lo) / (t_hi - t_lo + 1e-12)).unsqueeze(-1)
        interp = (1 - alpha) * path_dev[idx] + alpha * path_dev[idx + 1]
        self.values.data.copy_(interp)

    def evaluate_at_values(self, values: Tensor) -> Tensor:
        """Evaluate the spline at ``eval_t`` using the given control values.

        Differentiable through ``values``. Used both by ``forward`` (via
        ``self.values``) and to decode raw control-value snapshots into K-step
        path tensors after optimization.
        """
        if values.shape != (self.n_control, self.k_features):
            raise ValueError(
                f"values shape {tuple(values.shape)} != "
                f"({self.n_control}, {self.k_features})"
            )
        gamma, y_hat = CubicSpline1D._fit_natural(
            self.x_knots,
            values,
            self.h,
            self.smoothness,
        )
        idx = self.eval_seg_idx
        return CubicSpline1D._eval_segment(
            self.eval_t,
            self.x_knots[idx],
            self.x_knots[idx + 1],
            y_hat[idx],
            y_hat[idx + 1],
            gamma[idx],
            gamma[idx + 1],
        )

    def forward(self) -> Tensor:
        """Return ``(n_eval, k_features)`` interpolated path under current values."""
        return self.evaluate_at_values(self.values)
