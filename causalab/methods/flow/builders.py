"""Builder functions for constructing normalizing flows."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

import torch

from .model import Flow
from .manifold import ManifoldFlow
from .base_dist import StandardNormal
from .bijectors import AffineCoupling, Permutation


@dataclass
class FlowConfig:
    """Configuration for reconstructing a Flow.

    Stores all hyperparameters needed to rebuild a flow architecture.
    Used for serialization/deserialization of trained flows.
    """

    dim: int
    num_layers: int = 8
    hidden: int = 256
    depth: int = 2
    s_scale: float = 2.0
    seed: int = 0

    def to_dict(self) -> dict[str, int | float]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlowConfig":
        """Reconstruct from dictionary."""
        return cls(**d)


def build_realNVP_flow(
    dim: int,
    num_layers: int = 8,
    hidden: int = 256,
    depth: int = 2,
    s_scale: float = 2.0,
    seed: int = 0,
) -> Flow:
    """Build standard RealNVP-style flow with alternating coupling layers.

    Architecture:
    - Alternates even/odd dimension splits for coupling layers
    - Each coupling layer followed by a fixed random permutation
    - Standard normal base distribution

    Args:
        dim: Dimension of the flow (D)
        num_layers: Number of coupling layers
        hidden: Hidden layer size in coupling MLPs
        depth: Number of hidden layers in coupling MLPs
        s_scale: Scale factor for tanh squashing in coupling
        seed: Random seed for permutations

    Returns:
        Configured Flow with attached FlowConfig
    """
    layers: list[torch.nn.Module] = []

    for i in range(num_layers):
        if i % 2 == 0:
            idx_a = torch.arange(0, dim, 2)
            idx_b = torch.arange(1, dim, 2)
        else:
            idx_a = torch.arange(1, dim, 2)
            idx_b = torch.arange(0, dim, 2)

        layers.append(
            AffineCoupling(
                dim,
                idx_a=idx_a,
                idx_b=idx_b,
                hidden=hidden,
                depth=depth,
                s_scale=s_scale,
            )
        )
        layers.append(Permutation(dim, seed=seed + i))

    base = StandardNormal(dim)
    config = FlowConfig(
        dim=dim,
        num_layers=num_layers,
        hidden=hidden,
        depth=depth,
        s_scale=s_scale,
        seed=seed,
    )

    return Flow(layers=layers, base_dist=base, config=config)


def build_realNVP_flow_from_state_dict(
    state_dict: dict[str, Any],
    config: FlowConfig | dict[str, Any],
) -> Flow:
    """Build flow from state_dict using provided config.

    Args:
        state_dict: Flow state dict (from flow.state_dict())
        config: FlowConfig or dict with architecture parameters.

    Returns:
        Flow with loaded weights
    """
    if isinstance(config, dict):
        config = FlowConfig.from_dict(config)

    flow = build_realNVP_flow(
        dim=config.dim,
        num_layers=config.num_layers,
        hidden=config.hidden,
        depth=config.depth,
        s_scale=config.s_scale,
        seed=config.seed,
    )
    flow.load_state_dict(state_dict)
    return flow


def build_manifold_flow(
    dim: int,
    intrinsic_dim: int,
    num_layers: int = 8,
    hidden: int = 256,
    depth: int = 2,
    s_scale: float = 2.0,
    seed: int = 0,
) -> ManifoldFlow:
    """Build ManifoldFlow wrapping a RealNVP flow.

    Creates a full-dimensional flow and wraps it in ManifoldFlow
    which partitions the latent space into intrinsic (on-manifold)
    and residual (off-manifold) components.

    Args:
        dim: Ambient dimension (n) of the data
        intrinsic_dim: Dimension of the manifold (k < n)
        num_layers: Number of coupling layers in inner flow
        hidden: Hidden layer size in coupling MLPs
        depth: Number of hidden layers in coupling MLPs
        s_scale: Scale factor for tanh squashing
        seed: Random seed for permutations

    Returns:
        ManifoldFlow wrapping a configured RealNVP flow
    """
    flow = build_realNVP_flow(
        dim=dim,
        num_layers=num_layers,
        hidden=hidden,
        depth=depth,
        s_scale=s_scale,
        seed=seed,
    )
    return ManifoldFlow(flow, intrinsic_dim=intrinsic_dim)
