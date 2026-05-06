"""Parametric UMAP featurizer.

Fits sklearn UMAP to get target coordinates, then trains MLP encoder/decoder
to approximate the mapping. This gives differentiable inference and standard
torch serialization without TF/Keras dependency.

The composed featurizer chain is:
    UMAPFeaturizer >> StandardizeFeaturizer >> ManifoldFeaturizer(spline)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from causalab.neural.featurizer import Featurizer


class UMAPFeaturizerModule(nn.Module):
    """MLP encoder that maps activations to UMAP coordinates.

    Forward returns ``(features, error)`` where error is the reconstruction
    residual from the paired decoder -- matching the Featurizer protocol.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        x = x.float()
        features = self.encoder(x)
        x_recon = self.decoder(features)
        error = x - x_recon
        return features.to(input_dtype), error.to(input_dtype)


class UMAPInverseFeaturizerModule(nn.Module):
    """MLP decoder that reconstructs activations from UMAP coordinates + error."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, features: Tensor, error: Tensor) -> Tensor:
        input_dtype = features.dtype
        return (self.decoder(features.float()) + error.float()).to(input_dtype)


def _build_mlp(in_dim: int, out_dim: int, hidden_dim: int = 256) -> nn.Sequential:
    """Build a 3-layer MLP with ReLU activations."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class UMAPFeaturizer(Featurizer):
    """Parametric UMAP featurizer with encoder/decoder MLPs."""

    FEATURIZER_MODULE_CLASS_NAME = "UMAPFeaturizerModule"

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *,
        n_components: int,
        in_dim: int,
        hidden_dim: int,
        id: str = "umap",
    ) -> None:
        super().__init__(
            UMAPFeaturizerModule(encoder, decoder),
            UMAPInverseFeaturizerModule(decoder),
            n_features=n_components,
            id=id,
        )
        self._in_dim = in_dim
        self._hidden_dim = hidden_dim
        self._n_components = n_components

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_info": {
                "featurizer_class": "UMAPFeaturizerModule",
                "inverse_featurizer_class": "UMAPInverseFeaturizerModule",
                "n_features": self.n_features,
                "featurizer_id": self.id,
                "additional_config": {
                    "n_components": self._n_components,
                    "in_dim": self._in_dim,
                    "hidden_dim": self._hidden_dim,
                },
            },
            "featurizer_state_dict": self.featurizer.state_dict(),
            "inverse_state_dict": self.inverse_featurizer.state_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UMAPFeaturizer":
        model_info = data["model_info"]
        additional = model_info["additional_config"]
        feat = build_umap_featurizer(
            in_dim=additional["in_dim"],
            n_components=additional["n_components"],
            hidden_dim=additional["hidden_dim"],
        )
        feat.featurizer.load_state_dict(data["featurizer_state_dict"])
        feat.inverse_featurizer.load_state_dict(data["inverse_state_dict"])
        feat.id = model_info.get("featurizer_id", "umap")
        return feat


def build_umap_featurizer(
    in_dim: int,
    n_components: int = 3,
    hidden_dim: int = 256,
) -> UMAPFeaturizer:
    """Build an untrained UMAPFeaturizer with encoder/decoder MLPs."""
    encoder = _build_mlp(in_dim, n_components, hidden_dim)
    decoder = _build_mlp(n_components, in_dim, hidden_dim)
    return UMAPFeaturizer(
        encoder,
        decoder,
        n_components=n_components,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
    )


def train_umap_featurizer(
    raw_features: Tensor,
    umap_coords: Tensor,
    featurizer: UMAPFeaturizer,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
) -> UMAPFeaturizer:
    """Train the encoder/decoder MLPs to approximate the UMAP mapping.

    Args:
        raw_features: Raw activations, shape ``(n, in_dim)``.
        umap_coords: Target UMAP coordinates, shape ``(n, n_components)``.
        featurizer: Untrained UMAPFeaturizer from ``build_umap_featurizer``.
        epochs: Number of training epochs for each of encoder and decoder.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam optimizer.
        device: Device to train on. Defaults to features device.

    Returns:
        The trained featurizer (modified in-place).
    """
    if device is None:
        device = raw_features.device

    umap_module = featurizer.featurizer
    encoder = umap_module.encoder.to(device)
    decoder = umap_module.decoder.to(device)

    raw_features = raw_features.to(device).float()
    umap_coords = umap_coords.to(device).float()

    n = raw_features.shape[0]

    # Train encoder: activations -> umap_coords
    print(f"[UMAP] Training encoder ({epochs} epochs, {n} samples)...")
    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=lr)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = encoder(raw_features[idx])
            loss = torch.nn.functional.mse_loss(pred, umap_coords[idx])
            optimizer_enc.zero_grad()
            loss.backward()
            optimizer_enc.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"  Encoder epoch {epoch + 1}/{epochs}, loss={total_loss / n_batches:.6f}"
            )

    # Train decoder: umap_coords -> activations
    print(f"[UMAP] Training decoder ({epochs} epochs, {n} samples)...")
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=lr)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = decoder(umap_coords[idx])
            loss = torch.nn.functional.mse_loss(pred, raw_features[idx])
            optimizer_dec.zero_grad()
            loss.backward()
            optimizer_dec.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"  Decoder epoch {epoch + 1}/{epochs}, loss={total_loss / n_batches:.6f}"
            )

    # Move back to CPU for saving
    encoder.cpu()
    decoder.cpu()

    return featurizer
