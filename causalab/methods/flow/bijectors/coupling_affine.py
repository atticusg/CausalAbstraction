"""RealNVP-style affine coupling layer."""

from __future__ import annotations
from typing import Tuple

import torch
from torch import nn, Tensor

from .base import Bijector


class MLP(nn.Module):
    """Simple feedforward network for computing scale and shift."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AffineCoupling(Bijector):
    """
    Affine coupling layer (RealNVP style).

    Splits input into conditioning dims (a) and transformed dims (b).
    Computes scale (s) and shift (t) from x_a, then:
        y_b = x_b * exp(s) + t
        y_a = x_a

    Uses tanh squashing on scale for numerical stability:
        s = s_scale * tanh(raw_s)

    Args:
        D: Total dimension of input
        idx_a: Indices of conditioning dimensions
        idx_b: Indices of transformed dimensions
        hidden: Hidden layer size in MLP
        depth: Number of hidden layers in MLP
        s_scale: Scale factor for tanh squashing (default 2.0)
    """

    def __init__(
        self,
        D: int,
        idx_a: Tensor,
        idx_b: Tensor,
        hidden: int = 256,
        depth: int = 2,
        s_scale: float = 2.0,
    ):
        super().__init__()
        idx_a = idx_a.to(dtype=torch.long)
        idx_b = idx_b.to(dtype=torch.long)
        self.register_buffer("idx_a", idx_a)
        self.idx_a: Tensor
        self.register_buffer("idx_b", idx_b)
        self.idx_b: Tensor
        self.nn = MLP(
            in_dim=idx_a.numel(),
            out_dim=2 * idx_b.numel(),
            hidden=hidden,
            depth=depth,
        )
        self.s_scale = float(s_scale)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward transformation.

        Args:
            x: Input tensor (B, D)

        Returns:
            y: Output tensor (B, D)
            logdet: Log determinant (B,), equals sum of s over transformed dims
        """
        xa = x[:, self.idx_a]
        xb = x[:, self.idx_b]
        st = self.nn(xa)
        raw_s, t = st.chunk(2, dim=-1)
        s = self.s_scale * torch.tanh(raw_s)

        y = x.clone()
        yb = xb * torch.exp(s) + t
        y[:, self.idx_b] = yb

        logdet = s.sum(dim=-1)
        return y, logdet

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Inverse transformation.

        Args:
            y: Input tensor (B, D)

        Returns:
            x: Output tensor (B, D)
            logdet: Log determinant of inverse (B,), equals -sum of s
        """
        ya = y[:, self.idx_a]
        yb = y[:, self.idx_b]
        st = self.nn(ya)  # Same conditioning as forward (ya == xa)
        raw_s, t = st.chunk(2, dim=-1)
        s = self.s_scale * torch.tanh(raw_s)

        x = y.clone()
        xb = (yb - t) * torch.exp(-s)
        x[:, self.idx_b] = xb

        logdet = -s.sum(dim=-1)
        return x, logdet
