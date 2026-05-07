"""Base distribution for normalizing flows."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.distributions import Normal, Independent
from torch import Size
from typing_extensions import Self


class StandardNormal:
    """
    Standard multivariate normal distribution (diagonal covariance).

    Uses torch.distributions.Independent to treat each dimension independently.

    Args:
        D: Dimension of the distribution
        device: Device for tensors (default: None uses CPU)
    """

    def __init__(self, D: int, device: torch.device | str | None = None) -> None:
        self.D = int(D)
        self._device = device
        loc = torch.zeros(D, device=device)
        scale = torch.ones(D, device=device)
        self.dist = Independent(Normal(loc=loc, scale=scale), 1)

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Compute log probability of samples.

        Args:
            z: Samples of shape (B, D)

        Returns:
            Log probabilities of shape (B,)
        """
        return self.dist.log_prob(z)

    def sample(self, shape: tuple[int, ...] | Size) -> Tensor:
        """
        Sample from the distribution.

        Args:
            shape: Sample shape (typically (n,))

        Returns:
            Samples of shape (*shape, D)
        """
        return self.dist.sample(shape)

    def to(self, device: Any) -> Self:
        """Move distribution to specified device."""
        self._device = device
        loc = torch.zeros(self.D, device=device)
        scale = torch.ones(self.D, device=device)
        self.dist = Independent(Normal(loc=loc, scale=scale), 1)
        return self
