"""Flow model composing bijectors with a base distribution."""

from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING, cast

import torch
from torch import nn, Tensor
from typing_extensions import Self

from .base_dist import StandardNormal
from .bijectors.base import Bijector

if TYPE_CHECKING:
    from .builders import FlowConfig


class Flow(nn.Module):
    """
    Normalizing flow model.

    Composes a sequence of bijectors with a base distribution to model
    complex distributions via change of variables:
        log p_X(x) = log p_Z(f(x)) + log|det(df/dx)|

    Args:
        layers: List of bijector modules
        base_dist: Base distribution (e.g., StandardNormal)
        config: Optional FlowConfig storing architecture hyperparameters
    """

    def __init__(
        self,
        layers: List[nn.Module],
        base_dist: StandardNormal,
        config: "FlowConfig | None" = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.base_dist = base_dist
        self._config = config

    @property
    def config(self) -> "FlowConfig | None":
        """Return the flow configuration, if available."""
        return self._config

    def fwd(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass: transform x to latent z.

        Args:
            x: Data samples (B, D)

        Returns:
            z: Latent samples (B, D)
            total_logdet: Sum of log-determinants (B,)
        """
        z = x
        total = x.new_zeros(x.shape[0])
        for layer in self.layers:
            z, ld = layer(z)
            total = total + ld
        return z, total

    def inv(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Inverse pass: transform latent z to data x.

        Args:
            z: Latent samples (B, D)

        Returns:
            x: Data samples (B, D)
            total_logdet: Sum of log-determinants of inverses (B,)
        """
        x = z
        total = z.new_zeros(z.shape[0])
        for layer in reversed(self.layers):
            bijector = cast(Bijector, layer)
            x, ld = bijector.inverse(x)
            total = total + ld
        return x, total

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute log probability of data.

        Args:
            x: Data samples (B, D)

        Returns:
            Log probabilities (B,)
        """
        z, logdet = self.fwd(x)
        return self.base_dist.log_prob(z) + logdet

    @torch.no_grad()  # type: ignore[reportUntypedFunctionDecorator]
    def sample(self, n: int, device: torch.device | str | None = None) -> Tensor:
        """
        Sample from the flow.

        Args:
            n: Number of samples
            device: Device for samples

        Returns:
            Samples (n, D)
        """
        z = self.base_dist.sample((n,))
        if device is not None:
            z = z.to(device)
        x, _ = self.inv(z)
        return x

    def to(self, *args, **kwargs) -> Self:  # type: ignore[override]
        """Move flow and base distribution to device."""
        super().to(*args, **kwargs)
        self.base_dist = self.base_dist.to(*args, **kwargs)
        return self
