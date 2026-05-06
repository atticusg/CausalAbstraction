"""Standardize (affine normalization) featurizer."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from causalab.neural.featurizer import Featurizer


class StandardizeFeaturizerModule(torch.nn.Module):
    """Bijective affine standardization: (x - mean) / (std + eps).

    Returns (standardized, None) - lossless transform, no error term.
    """

    def __init__(self, mean: Tensor, std: Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("_mean", mean)
        self._mean: Tensor
        self.register_buffer("_std", std)
        self._std: Tensor
        self._eps = eps

    def forward(self, x: Tensor) -> tuple[Tensor, None]:
        mean = self._mean.to(x.device)
        std = self._std.to(x.device)
        return (x - mean) / (std + self._eps), None


class StandardizeInverseFeaturizerModule(torch.nn.Module):
    """Inverse of StandardizeFeaturizerModule."""

    def __init__(self, mean: Tensor, std: Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("_mean", mean)
        self._mean: Tensor
        self.register_buffer("_std", std)
        self._std: Tensor
        self._eps = eps

    def forward(self, x: Tensor, error: None) -> Tensor:
        mean = self._mean.to(x.device)
        std = self._std.to(x.device)
        return x * (std + self._eps) + mean


class StandardizeFeaturizer(Featurizer):
    FEATURIZER_MODULE_CLASS_NAME = "StandardizeFeaturizerModule"

    """Bijective affine standardization featurizer.

    Maps x -> ((x - mean) / (std + eps), None). Fully invertible.
    Useful for composing with other featurizers that expect standardized input.

    Example:
        # Compose for DAS -> standardize -> manifold pipeline
        das = SubspaceFeaturizer(rotation_subspace=rot)
        standardize = StandardizeFeaturizer(mean, std)
        manifold_feat = ManifoldFeaturizer(manifold)
        composed = das >> standardize >> manifold_feat
    """

    def __init__(
        self,
        mean: Tensor,
        std: Tensor,
        eps: float = 1e-6,
        *,
        id: str = "standardize",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            StandardizeFeaturizerModule(mean, std, eps),
            StandardizeInverseFeaturizerModule(mean, std, eps),
            n_features=mean.shape[0],
            id=id,
            **kwargs,
        )
        self._eps = eps

    def to_dict(self) -> dict[str, Any]:
        """Serialize standardize featurizer."""
        return {
            "model_info": {
                "featurizer_class": "StandardizeFeaturizerModule",
                "inverse_featurizer_class": "StandardizeInverseFeaturizerModule",
                "n_features": self.n_features,
                "featurizer_id": self.id,
                "additional_config": {
                    "mean": self.featurizer._mean.detach().clone(),  # pyright: ignore[reportPrivateUsage,reportCallIssue]
                    "std": self.featurizer._std.detach().clone(),  # pyright: ignore[reportPrivateUsage,reportCallIssue]
                    "eps": self._eps,
                },
            },
            "featurizer_state_dict": self.featurizer.state_dict(),
            "inverse_state_dict": self.inverse_featurizer.state_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StandardizeFeaturizer":
        model_info = data["model_info"]
        additional = model_info["additional_config"]
        return cls(
            additional["mean"],
            additional["std"],
            additional.get("eps", 1e-6),
            id=model_info.get("featurizer_id", "standardize"),
        )
