"""SAE (Sparse Autoencoder) featurizer."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from causalab.neural.featurizer import Featurizer


class SAEFeaturizerModule(torch.nn.Module):
    """Wrapper around a *Sparse Autoencoder*'s encode() / decode() pair."""

    def __init__(self, sae: Any) -> None:
        super().__init__()
        self.sae = sae

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.sae.encode(x.to(self.sae.dtype))
        error = x - self.sae.decode(features).to(x.dtype)
        return features.to(x.dtype), error


class SAEInverseFeaturizerModule(torch.nn.Module):
    """Inverse for :class:`SAEFeaturizerModule`."""

    def __init__(self, sae: Any) -> None:
        super().__init__()
        self.sae = sae

    def forward(self, features: Tensor, error: Tensor) -> Tensor:
        return self.sae.decode(features.to(self.sae.dtype)).to(
            features.dtype
        ) + error.to(features.dtype)


# currently unused but not dead code - usage to come
class SAEFeaturizer(Featurizer):
    """Featurizer backed by a pre-trained sparse auto-encoder.

    Notes
    -----
    Serialisation is *disabled* for SAE featurizers -- saving will raise
    ``NotImplementedError``.
    """

    FEATURIZER_MODULE_CLASS_NAME = "SAEFeaturizerModule"

    def __init__(self, sae: Any, *, trainable: bool = False, **kwargs: Any) -> None:
        sae.requires_grad_(trainable)
        super().__init__(
            SAEFeaturizerModule(sae),
            SAEInverseFeaturizerModule(sae),
            n_features=sae.cfg.to_dict()["d_sae"],
            id="sae",
            **kwargs,
        )

    def to_dict(self) -> None:  # type: ignore[override]
        return None

    def save_modules(self, path: str) -> tuple[None, None]:  # type: ignore[override]
        return None, None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SAEFeaturizer":
        raise NotImplementedError(
            "SAEFeaturizer cannot be reconstructed from a dict — load via sae_lens."
        )

    @classmethod
    def load_modules(cls, path: str) -> "SAEFeaturizer":
        raise NotImplementedError(
            "SAEFeaturizer cannot be loaded from disk — load via sae_lens."
        )
