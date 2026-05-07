"""Manifold flow: learning k-dim manifolds in n-dim space."""

from __future__ import annotations
from typing import Any, Dict, Tuple

import torch
from torch import nn, Tensor
from typing_extensions import Self

from .model import Flow


class ManifoldFlow(nn.Module):
    """
    Manifold flow for learning k-dimensional manifolds in n-dimensional space.

    Wraps an n-to-n invertible flow and partitions the latent space z = (u, r):
    - u: k-dimensional intrinsic coordinates (on-manifold)
    - r: (n-k)-dimensional residual (off-manifold)

    Training uses reconstruction loss + residual regularization to push r toward zero.
    At test time, decode(u, r=0) gives deterministic on-manifold reconstruction.

    Args:
        flow: An n-to-n normalizing flow (bijection)
        intrinsic_dim: Dimensionality k of the manifold (k < n)
    """

    def __init__(self, flow: Flow, intrinsic_dim: int):
        super().__init__()
        self.flow = flow
        self.k = intrinsic_dim
        self.n = flow.base_dist.D

        if self.k >= self.n:
            raise ValueError(
                f"intrinsic_dim ({self.k}) must be less than ambient dim ({self.n})"
            )
        if self.k < 1:
            raise ValueError(f"intrinsic_dim must be >= 1, got {self.k}")

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode data x to latent (u, r) via the flow's forward pass.

        Args:
            x: Data samples (B, n)

        Returns:
            u: Intrinsic coordinates (B, k)
            r: Residual coordinates (B, n-k)
        """
        z, _ = self.flow.fwd(x)
        u = z[:, : self.k]
        r = z[:, self.k :]
        return u, r

    def decode(self, u: Tensor, r: Tensor | None = None) -> Tensor:
        """
        Decode latent (u, r) to data x via the flow's inverse.

        Args:
            u: Intrinsic coordinates (B, k)
            r: Residual coordinates (B, n-k). If None, uses zeros (on-manifold).

        Returns:
            x: Reconstructed data (B, n)
        """
        if r is None:
            r = u.new_zeros(u.shape[0], self.n - self.k)
        z = torch.cat([u, r], dim=-1)
        x, _ = self.flow.inv(z)
        return x

    def project(self, x: Tensor) -> Tensor:
        """
        Project x onto the learned manifold.

        Encodes x to (u, r), then decodes with r=0 to get on-manifold reconstruction.

        Args:
            x: Data samples (B, n)

        Returns:
            x_proj: On-manifold projection (B, n)
        """
        u, _ = self.encode(x)
        return self.decode(u, r=None)

    def loss(
        self,
        x: Tensor,
        residual_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute manifold loss: reconstruction + residual regularization.

        Loss = reconstruction_weight * MSE(decode(u, 0), x) + residual_weight * mean(||r||^2)

        Args:
            x: Data samples (B, n)
            residual_weight: Weight for residual regularization term
            reconstruction_weight: Weight for reconstruction loss

        Returns:
            total_loss: Combined loss (scalar tensor)
            metrics: Dictionary with component losses {"recon": float, "residual": float}
        """
        u, r = self.encode(x)
        x_hat = self.decode(u, r=None)

        # MSE reconstruction loss (sum over dims, mean over batch)
        recon_loss = ((x_hat - x) ** 2).sum(dim=-1).mean()

        # Residual regularization (push r toward zero)
        residual_loss = (r**2).sum(dim=-1).mean()

        total = reconstruction_weight * recon_loss + residual_weight * residual_loss

        metrics = {
            "recon": recon_loss.item(),
            "residual": residual_loss.item(),
        }

        return total, metrics

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returns projection (for compatibility with nn.Module)."""
        return self.project(x)

    def to(self, *args, **kwargs) -> Self:  # type: ignore[override]
        """Move manifold flow and inner flow to device."""
        super().to(*args, **kwargs)
        self.flow = self.flow.to(*args, **kwargs)
        return self


class FlowManifold(nn.Module):
    """
    Manifold protocol adapter wrapping ManifoldFlow with standardization.

    This adapter implements the Manifold protocol for use with ManifoldFeaturizer,
    handling standardization internally. Unlike the raw ManifoldFlow which operates
    on standardized data, FlowManifold handles the standardization/unstandardization
    so users can work directly in original feature space.

    The flow's latent space is partitioned as z = (u, r) where:
    - u: intrinsic_dim-dimensional on-manifold coordinates
    - r: (ambient_dim - intrinsic_dim)-dimensional residual

    Args:
        manifold_flow: Trained ManifoldFlow instance
        mean: Mean for standardization (ambient_dim,)
        std: Std for standardization (ambient_dim,)
        eps: Numerical stability epsilon
    """

    def __init__(
        self,
        manifold_flow: ManifoldFlow,
        mean: Tensor,
        std: Tensor,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.manifold_flow = manifold_flow
        self.register_buffer("_mean", mean)
        self._mean: Tensor
        self.register_buffer("_std", std)
        self._std: Tensor
        self._eps = eps

    @property
    def intrinsic_dim(self) -> int:
        """Dimensionality of on-manifold intrinsic coordinates."""
        return self.manifold_flow.k

    @property
    def ambient_dim(self) -> int:
        """Dimensionality of ambient space (flow latent dim)."""
        return self.manifold_flow.n

    @property
    def residual_dim(self) -> int:
        """Dimensionality of off-manifold residual."""
        return self.ambient_dim - self.intrinsic_dim

    def _standardize(self, x: Tensor) -> Tensor:
        """Standardize input."""
        return (x - self._mean) / (self._std + self._eps)

    def _unstandardize(self, x: Tensor) -> Tensor:
        """Unstandardize output."""
        return x * (self._std + self._eps) + self._mean

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode ambient-space points to (intrinsic, residual).

        Args:
            x: Points in ambient space (batch, ambient_dim)

        Returns:
            intrinsic: On-manifold latent coordinates (batch, intrinsic_dim)
            residual: Off-manifold latent coordinates (batch, residual_dim)
        """
        x_norm = self._standardize(x)
        return self.manifold_flow.encode(x_norm)

    def decode(self, intrinsic: Tensor, residual: Tensor | None = None) -> Tensor:
        """
        Decode (intrinsic, residual) to ambient space.

        Args:
            intrinsic: On-manifold latent coordinates (batch, intrinsic_dim)
            residual: Off-manifold latent coordinates (batch, residual_dim).
                     If None, projects to manifold (r=0).

        Returns:
            x: Points in ambient space (batch, ambient_dim)
        """
        x_norm = self.manifold_flow.decode(intrinsic, residual)
        return self._unstandardize(x_norm)

    def project(self, x: Tensor) -> Tensor:
        """Project to manifold surface (r=0)."""
        intrinsic, _ = self.encode(x)
        return self.decode(intrinsic, None)

    def make_steering_grid(
        self,
        n_points_per_dim: int = 11,
        range_min: float = -3.0,
        range_max: float = 3.0,
        ranges: Tuple[Tuple[float, float], ...] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate Cartesian grid in intrinsic latent space.

        For flow manifolds, intrinsic coordinates are Gaussian-like latents,
        so a regular Cartesian grid with range ~[-3, 3] covers most of the
        probability mass.

        Args:
            n_points_per_dim: Points per dimension (for d<=2)
            range_min, range_max: Global range for all coordinates
            ranges: Per-dimension ranges as ((min0, max0), (min1, max1), ...)

        Returns:
            Grid of intrinsic coordinates (n_points, intrinsic_dim)
        """
        d = self.intrinsic_dim

        # Build per-dimension ranges
        if ranges is not None:
            if len(ranges) != d:
                raise ValueError(
                    f"ranges has {len(ranges)} entries but intrinsic_dim is {d}"
                )
            dim_ranges = ranges
        else:
            dim_ranges = tuple((range_min, range_max) for _ in range(d))

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
            # Sparse grid: vary one dimension at a time
            grids = []
            for dim in range(d):
                coords = torch.linspace(
                    dim_ranges[dim][0], dim_ranges[dim][1], n_points_per_dim
                )
                sparse = torch.zeros(n_points_per_dim, d)
                sparse[:, dim] = coords
                grids.append(sparse)
            return torch.cat(grids, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returns projection."""
        return self.project(x)
