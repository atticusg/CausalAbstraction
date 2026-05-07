"""Abstract base class for bijectors."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple

from torch import nn, Tensor


class Bijector(nn.Module, ABC):
    """
    Abstract base class for bijective transformations.

    All bijectors must implement forward and inverse methods that return
    (output, logdet) tuples where logdet has shape (B,).

    Consistency requirement: for paired x -> y,
        logdet_inv(y) == -logdet_fwd(x)
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward transformation: x -> y

        Args:
            x: Input tensor of shape (B, D)

        Returns:
            y: Transformed tensor of shape (B, D)
            logdet: Log determinant of Jacobian, shape (B,)
        """
        raise NotImplementedError

    @abstractmethod
    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Inverse transformation: y -> x

        Args:
            y: Input tensor of shape (B, D)

        Returns:
            x: Transformed tensor of shape (B, D)
            logdet: Log determinant of Jacobian of inverse, shape (B,)
                   For paired x->y: logdet_inv(y) == -logdet_fwd(x)
        """
        raise NotImplementedError
