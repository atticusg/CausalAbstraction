"""SAE (Sparse Autoencoder) featurizer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from causalab.neural.featurizer import Featurizer


def decoder_subspace(
    decoder: Tensor,
    feature_ids: Sequence[int],
    *,
    d_model: int | None = None,
    orthonormalize: bool = True,
) -> Tensor:
    """Build a ``(d_model, k)`` subspace basis from SAE decoder directions.

    ``decoder`` is an SAE decoder weight in either ``(d_sae, d_model)`` or
    ``(d_model, d_sae)`` orientation; ``feature_ids`` index the ``d_sae``
    (feature) axis. The selected decoder directions are stacked column-wise
    into a ``(d_model, k)`` matrix.

    When ``orthonormalize`` (the default), the columns are replaced by an
    orthonormal basis of their span via QR. The span — i.e. the actual
    subspace — is preserved exactly; the leading column is no longer any
    particular decoder direction, so callers that care about a specific
    ordered axis should pass ``orthonormalize=False``.

    Orientation is resolved like :func:`loading._orient` on the consumer
    side: if ``d_model`` is given, the matching axis is ``d_model`` and the
    other is the feature axis; otherwise the *larger* axis is taken as the
    feature axis (SAEs are over-complete, so ``d_sae > d_model``).

    Returns a contiguous ``float32`` tensor (matching the dtype the subspace
    loader normalises to).
    """
    if decoder.ndim != 2:
        raise ValueError(
            f"Expected a 2-D decoder weight; got shape {tuple(decoder.shape)}."
        )
    ids = list(feature_ids)
    if not ids:
        raise ValueError("feature_ids must be non-empty.")

    a, b = int(decoder.shape[0]), int(decoder.shape[1])
    if d_model is not None:
        if d_model not in (a, b):
            raise ValueError(
                f"d_model={d_model} matches neither decoder axis (shape={a}x{b})."
            )
        if a == b:
            raise ValueError(
                f"Decoder is square ({a}x{b}); d_model alone cannot disambiguate "
                "which axis is the feature axis."
            )
        feature_axis = 1 if a == d_model else 0
    else:
        if a == b:
            raise ValueError(
                f"Decoder is square ({a}x{b}); pass d_model to identify the "
                "feature axis."
            )
        feature_axis = 0 if a > b else 1

    n_features = decoder.shape[feature_axis]
    out_of_range = [i for i in ids if i < 0 or i >= n_features]
    if out_of_range:
        raise ValueError(
            f"feature_ids out of range for feature axis of size {n_features}: "
            f"{out_of_range}."
        )

    # Gather selected directions into (d_model, k).
    selected = decoder.index_select(feature_axis, torch.tensor(ids, dtype=torch.long))
    directions = selected if feature_axis == 1 else selected.t()
    directions = directions.detach().to(torch.float32).contiguous()

    if not orthonormalize:
        return directions
    basis, _ = torch.linalg.qr(directions)
    return basis.contiguous()


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
