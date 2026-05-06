"""Fixed permutation bijector with zero log-determinant."""

from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor

from .base import Bijector


class Permutation(Bijector):
    """
    Fixed permutation of feature dimensions.

    Since permutation matrices have determinant +/-1, the log-determinant is 0.

    Args:
        D: Dimension of input vectors
        seed: Random seed for generating permutation (if perm not provided)
        perm: Explicit permutation tensor of indices 0..D-1
    """

    def __init__(self, D: int, seed: int | None = None, perm: Tensor | None = None):
        super().__init__()
        if perm is None:
            g = torch.Generator()
            if seed is not None:
                g.manual_seed(seed)
            perm = torch.randperm(D, generator=g)
        else:
            perm = perm.to(dtype=torch.long)
            assert perm.numel() == D, "perm must have length D"
            assert set(perm.tolist()) == set(range(D)), (
                "perm must be a permutation of 0..D-1"
            )

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(D, device=perm.device)

        self.register_buffer("perm", perm)
        self.perm: Tensor
        self.register_buffer("inv_perm", inv_perm)
        self.inv_perm: Tensor

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply permutation: y[i] = x[perm[i]]"""
        y = x[:, self.perm]
        logdet = x.new_zeros(x.shape[0])
        return y, logdet

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply inverse permutation."""
        x = y[:, self.inv_perm]
        logdet = y.new_zeros(y.shape[0])
        return x, logdet
