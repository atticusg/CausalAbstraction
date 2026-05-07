"""Spline-based manifold using Thin-Plate Spline interpolation."""

from __future__ import annotations

import warnings
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .cubic import CubicSpline1D
from .tps import ThinPlateSpline


_VALID_SPLINE_METHODS = ("auto", "tps", "cubic")


class SplineManifold(nn.Module):
    def __init__(
        self,
        control_points: Tensor,
        target_points: Tensor,
        intrinsic_dim: int,
        ambient_dim: int,
        smoothness: float = 0.0,
        periodic_dims: list[int] | tuple[bool, ...] | None = None,
        periods: list[float] | None = None,
        spline_method: str = "auto",
        sphere_project: bool = False,
    ) -> None:
        super().__init__()
        self._intrinsic_dim = intrinsic_dim
        self._ambient_dim = ambient_dim
        self.smoothness = smoothness
        self.encode_mode = "nearest_centroid"
        # When True, decode() returns points on the unit L2 sphere by fitting
        # the spline in the *tangent space* of the sphere at a base point and
        # mapping back via exp. This is the geometrically-correct way to
        # interpolate sphere-constrained data (Hellinger √(p)): a Euclidean
        # cubic in tangent space → smooth curve on the sphere, no overshoot
        # or "wavy" lateral motion you'd get from interpolating directly in
        # ambient space and then re-projecting onto the sphere.
        self._sphere_project = bool(sphere_project)

        if spline_method not in _VALID_SPLINE_METHODS:
            raise ValueError(
                f"spline_method must be one of {_VALID_SPLINE_METHODS}, got {spline_method!r}"
            )

        # Normalize periodic_dims to list[int] or empty list
        if periodic_dims is None:
            self._periodic_dims: list[int] = []
        elif periodic_dims and isinstance(periodic_dims[0], bool):
            # tuple[bool, ...] format for backward compat
            self._periodic_dims = [i for i, p in enumerate(periodic_dims) if p]
        else:
            self._periodic_dims = list(periodic_dims)

        # Default periods: infer from control point range for each periodic dim
        if periods is not None:
            self._periods = list(periods)
        elif self._periodic_dims:
            self._periods = []
            for i in self._periodic_dims:
                n_unique = len(torch.unique(control_points[:, i]))
                cp_range = (
                    control_points[:, i].max() - control_points[:, i].min()
                ).item()
                if n_unique > 1:
                    # With n equally-spaced points, period = range * n / (n-1)
                    self._periods.append(cp_range * n_unique / (n_unique - 1))
                else:
                    self._periods.append(1.0)
        else:
            self._periods = []

        self.register_buffer("control_points", control_points)
        self.register_buffer("target_points", target_points)

        # For periodic dims, normalize as u/period; for linear dims, use min/range
        cp_min = control_points.min(dim=0).values.clone()
        cp_range = (control_points.max(dim=0).values - cp_min).clamp(min=1e-8).clone()
        # Override periodic dims: no shift, scale by period
        for pd, per in zip(self._periodic_dims, self._periods):
            cp_min[pd] = 0.0
            cp_range[pd] = per
        self.register_buffer("_cp_min", cp_min)
        self.register_buffer("_cp_range", cp_range)
        normalized_cp = (control_points - cp_min) / cp_range

        # Resolve "auto" to a concrete backend. The 1D cubic spline is the
        # 1D analog of TPS (both minimize ∫ f''²) and now covers both
        # natural-BC and periodic-BC 1D cases. Higher-dim or mixed
        # periodic+linear stays on TPS.
        is_1d = intrinsic_dim == 1
        if spline_method == "auto":
            resolved = "cubic" if is_1d else "tps"
        else:
            resolved = spline_method
            if resolved == "cubic" and not is_1d:
                raise ValueError(
                    "spline_method='cubic' requires intrinsic_dim=1; got "
                    f"intrinsic_dim={intrinsic_dim}"
                )
        self._resolved_spline_method = resolved
        self.spline_method = spline_method

        # Sphere-aware fit: pick a base point (extrinsic mean of target points,
        # projected back onto the sphere) and fit the spline on the LOG-mapped
        # targets (in the tangent plane at base) instead of the targets
        # themselves. decode() inverts this with the exp map. For Hellinger
        # √(p) all centroids live in the positive orthant of the sphere, so
        # the base point lies in their convex hull and every target is at most
        # π/2 away from it → log-map is well-defined and well-conditioned.
        if self._sphere_project:
            base = target_points.mean(dim=0)
            base = base / base.norm().clamp(min=1e-8)
            self.register_buffer("_sphere_base", base)
            spline_targets = self._sphere_log(base, target_points)
        else:
            self.register_buffer("_sphere_base", torch.zeros(0))  # sentinel
            spline_targets = target_points

        if resolved == "cubic":
            # Periodic dim 0 in normalized coords has period 1.0 since
            # _cp_range[0] equals the user-supplied period.
            if self._periodic_dims:
                bc = "periodic"
                period: float | None = 1.0
            else:
                bc = "natural"
                period = None
            self.spline = CubicSpline1D(
                normalized_cp,
                spline_targets,
                smoothness=smoothness,
                bc=bc,
                period=period,
            )
        else:
            if ambient_dim > 20:
                warnings.warn(
                    f"SplineManifold with large ambient_dim={ambient_dim}. "
                    f"TPS fitting may be slow or numerically unstable."
                )
            self.spline = self._build_tps(
                normalized_cp,
                spline_targets,
                intrinsic_dim,
                smoothness,
            )

        # `centroids` always holds the user-facing sphere/ambient coords so
        # downstream consumers (encode, pullback, viz) see the same interface
        # whether or not sphere_project is on.
        self.register_buffer("centroids", target_points)

    def _build_tps(
        self,
        normalized_cp: Tensor,
        target_points: Tensor,
        intrinsic_dim: int,
        smoothness: float,
    ) -> ThinPlateSpline:
        # Add ghost points for mixed periodic+linear dims: duplicate control
        # points shifted by ±period so the standard TPS interpolates smoothly
        # across the wrap boundary. Not needed for pure periodic (B4 kernel
        # handles wrapping natively).
        fit_cp = normalized_cp
        fit_targets = target_points
        has_linear = len(self._periodic_dims) < intrinsic_dim
        if self._periodic_dims and has_linear:
            ghost_cps = []
            ghost_targets = []
            for pd in self._periodic_dims:
                # Shift all points by +1 and -1 in normalized coords (period=1)
                cp_plus = normalized_cp.clone()
                cp_plus[:, pd] = cp_plus[:, pd] + 1.0
                cp_minus = normalized_cp.clone()
                cp_minus[:, pd] = cp_minus[:, pd] - 1.0
                ghost_cps.extend([cp_plus, cp_minus])
                ghost_targets.extend([target_points, target_points])
            fit_cp = torch.cat([normalized_cp] + ghost_cps, dim=0)
            fit_targets = torch.cat([target_points] + ghost_targets, dim=0)

        # Always pass periodic info so TPS excludes periodic dims from the
        # polynomial term.  For pure periodic the B4 kernel handles wrapping;
        # for mixed, ghost points handle the kernel side while excluding
        # periodic dims from the polynomial prevents a linear trend in θ
        # that would break closure.
        if self._periodic_dims:
            spline_periodic_dims = self._periodic_dims
            spline_periods = [1.0] * len(self._periods)  # normalized
        else:
            spline_periodic_dims = None
            spline_periods = None

        return ThinPlateSpline(
            fit_cp,
            fit_targets,
            smoothness=smoothness,
            periodic_dims=spline_periodic_dims,
            periods=spline_periods,
        )

    def get_config(self) -> dict:
        """Return constructor args needed to rebuild this manifold from a state dict."""
        return {
            "type": "spline",
            "intrinsic_dim": self._intrinsic_dim,
            "ambient_dim": self._ambient_dim,
            "smoothness": self.smoothness,
            "periodic_dims": self._periodic_dims,
            "periods": self._periods,
            "spline_method": self.spline_method,
        }

    @property
    def periodic_dims(self) -> list[int]:
        return self._periodic_dims

    @property
    def periods(self) -> list[float]:
        """Public read access to per-dimension periods (for periodic dims)."""
        return list(self._periods)

    @property
    def intrinsic_dim(self) -> int:
        return self._intrinsic_dim

    @property
    def k(self) -> int:
        return self._intrinsic_dim

    @property
    def ambient_dim(self) -> int:
        return self._ambient_dim

    @property
    def n(self) -> int:
        return self._ambient_dim

    @property
    def n_centroids(self) -> int:
        return self.centroids.shape[0]

    def _normalize_intrinsic(self, u: Tensor) -> Tensor:
        return (u - self._cp_min) / self._cp_range

    def decode(self, u: Tensor, r: Tensor | None = None) -> Tensor:
        u = u.to(self._cp_min.device)
        u_norm = self._normalize_intrinsic(u)
        h = self.spline.evaluate(u_norm)
        if self._sphere_project:
            # The spline output lives in the tangent plane at _sphere_base
            # (linear combinations of log-mapped targets, all of which are in
            # that plane). Exp-map maps it back to the sphere smoothly.
            h = self._sphere_exp(self._sphere_base, h)
        return h

    @staticmethod
    def _sphere_log(base: Tensor, points: Tensor) -> Tensor:
        """Log map on the unit sphere at `base`. Returns tangent vectors of
        length θ in the direction of each input point. Output rows lie in the
        hyperplane perpendicular to `base`."""
        cos_theta = (points @ base).clamp(-1.0 + 1e-8, 1.0 - 1e-8)
        theta = cos_theta.acos()
        diff = points - cos_theta.unsqueeze(-1) * base
        diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return diff / diff_norm * theta.unsqueeze(-1)

    @staticmethod
    def _sphere_exp(base: Tensor, tangent: Tensor) -> Tensor:
        """Exp map on the unit sphere at `base`. Maps a tangent vector back
        to a point on the sphere. exp(log(p)) = p for any sphere point p."""
        norm = tangent.norm(dim=-1, keepdim=True)
        safe = norm.clamp(min=1e-8)
        return torch.cos(norm) * base + torch.sin(norm) * tangent / safe

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.to(self.centroids.device)
        dists = torch.cdist(x, self.centroids, p=2)
        nearest_idx = dists.argmin(dim=1)
        u = self.control_points[nearest_idx]
        residual = dists[torch.arange(x.shape[0]), nearest_idx].unsqueeze(-1)
        return u, residual

    def encode_to_nearest_point(
        self,
        x: Tensor,
        n_iters: int = 5,
        tol: float = 1e-6,
        damping: float = 1e-6,
    ) -> Tuple[Tensor, Tensor]:
        """Project x onto the manifold via Gauss-Newton.

        Solves argmin_u ||decode(u) - x||^2 starting from the nearest-centroid
        initialization.

        Args:
            x: Points in ambient space (batch, ambient_dim).
            n_iters: Maximum Gauss-Newton iterations.
            tol: Convergence tolerance on max ||delta_u||.
            damping: Tikhonov regularization added to J^T J for stability.

        Returns:
            u: Continuous intrinsic coordinates (batch, intrinsic_dim).
            residual: Off-manifold displacement vector (batch, ambient_dim),
                      satisfying decode(u) + residual ~ x.
        """
        x = x.to(self.centroids.device)
        d = self._intrinsic_dim
        u, _ = self.encode(x)  # warm start
        u = u.clone().detach()

        fd_eps = 1e-5

        for _ in range(n_iters):
            x_hat = self.decode(u)
            r = x_hat - x  # (batch, ambient_dim)

            # Jacobian via central finite differences
            J = torch.zeros(*x.shape, d, device=x.device, dtype=x.dtype)
            for j in range(d):
                u_fwd = u.clone()
                u_fwd[:, j] += fd_eps
                u_bwd = u.clone()
                u_bwd[:, j] -= fd_eps
                J[:, :, j] = (self.decode(u_fwd) - self.decode(u_bwd)) / (2 * fd_eps)

            # Gauss-Newton step
            JtJ = torch.einsum("...ni,...nj->...ij", J, J)
            JtJ += damping * torch.eye(d, device=x.device, dtype=x.dtype)
            Jtr = torch.einsum("...ni,...n->...i", J, r)
            delta = torch.linalg.solve(JtJ, Jtr)
            u = u - delta

            # Wrap periodic dims
            for pd, per in zip(self._periodic_dims, self._periods):
                u[:, pd] = u[:, pd] % per

            if delta.norm(dim=-1).max() < tol:
                break

        residual = x - self.decode(u)
        return u, residual

    def project(self, x: Tensor) -> Tensor:
        u, _ = self.encode(x)
        return self.decode(u)

    def make_steering_grid(
        self,
        n_points_per_dim: int = 11,
        range_min: float = -3.0,
        range_max: float = 3.0,
        ranges: Tuple[Tuple[float, float], ...] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        d = self._intrinsic_dim

        if ranges is not None:
            if len(ranges) != d:
                raise ValueError(
                    f"ranges has {len(ranges)} entries but intrinsic_dim is {d}"
                )
            dim_ranges = list(ranges)
        else:
            dim_ranges = [(range_min, range_max)] * d

        # Override ranges for periodic dims: [0, period)
        for pd, per in zip(self._periodic_dims, self._periods):
            if ranges is None:
                dim_ranges[pd] = (0.0, per)

        if d == 1:
            coords = torch.linspace(
                dim_ranges[0][0], dim_ranges[0][1], n_points_per_dim
            )
            return coords.unsqueeze(-1)

        elif d == 2:
            coords0 = torch.linspace(
                dim_ranges[0][0], dim_ranges[0][1], n_points_per_dim
            )
            coords1 = torch.linspace(
                dim_ranges[1][0], dim_ranges[1][1], n_points_per_dim
            )
            u1, u2 = torch.meshgrid(coords0, coords1, indexing="ij")
            return torch.stack([u1.flatten(), u2.flatten()], dim=-1)

        else:
            grids = []
            for dim in range(d):
                coords = torch.linspace(
                    dim_ranges[dim][0], dim_ranges[dim][1], n_points_per_dim
                )
                sparse = torch.zeros(n_points_per_dim, d)
                sparse[:, dim] = coords
                grids.append(sparse)
            return torch.cat(grids, dim=0)

    def fwd(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Flow-compatible forward: x (n, k) -> (z, logdet).

        Encodes ambient-dim x into k-dim latent [u, 0...0].
        """
        encode_fn = (
            self.encode_to_nearest_point
            if self.encode_mode == "nearest_point"
            else self.encode
        )
        u, _residual = encode_fn(x)
        z = torch.zeros_like(x)
        z[:, : self._intrinsic_dim] = u
        logdet = torch.zeros(x.shape[0], device=x.device)
        return z, logdet

    def inv(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Flow-compatible inverse: z (n, k) -> (x, logdet).

        Decodes intrinsic coordinates from z[:, :d] back to ambient space.
        """
        u = z[:, : self._intrinsic_dim]
        x = self.decode(u)
        logdet = torch.zeros(z.shape[0], device=z.device)
        return x, logdet

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(x)

    def state_dict_to_save(self) -> dict[str, Any]:
        return {
            "control_points": self.control_points.cpu(),
            "centroids": self.centroids.cpu(),
            "intrinsic_dim": self._intrinsic_dim,
            "ambient_dim": self._ambient_dim,
            "smoothness": self.smoothness,
            "periodic_dims": self._periodic_dims,
            "periods": self._periods,
            "spline_method": self.spline_method,
            "sphere_project": self._sphere_project,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SplineManifold":
        # Default to "tps" for legacy checkpoints (missing key or unrecognized
        # legacy values like "euclidean_tps"/"mixed_tps") so they replay
        # bit-exact on the original backend.
        method = state.get("spline_method", "tps")
        if method not in _VALID_SPLINE_METHODS:
            method = "tps"
        return cls(
            control_points=state["control_points"],
            target_points=state["centroids"],
            intrinsic_dim=state["intrinsic_dim"],
            ambient_dim=state["ambient_dim"],
            smoothness=state["smoothness"],
            periodic_dims=state.get("periodic_dims"),
            periods=state.get("periods"),
            spline_method=method,
            sphere_project=state.get("sphere_project", False),
        )
