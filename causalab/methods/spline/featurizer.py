"""Manifold (spline / flow) featurizer."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from causalab.neural.featurizer import Featurizer


def _over_leading_dims(x: Tensor, fn: Callable[[Tensor], Tensor]) -> Tensor:
    """Apply ``fn`` — which assumes a 2-D ``(rows, d)`` input — over arbitrary
    leading dims by flattening them into one row axis, then restoring them.

    The spline manifold's ``encode``/``fwd``/``inv`` hard-code a 2-D batch layout
    (``argmin(dim=1)``, ``z[:, :k]``, ``torch.arange(x.shape[0])``). Feature
    interventions set ``keep_last_dim=True``, so a featurizer is handed
    ``(b, num_pos, d)`` (per-position application) rather than ``(b, d)`` — and
    even the single-token path arrives as ``(b, 1, d)``. Flatten the leading
    ``(b, num_pos)`` into rows so each position is projected independently, then
    reshape back. For a genuine 2-D input this is a no-op (``x.dim() <= 2``), so
    the offline callers that pass ``(n, d)`` directly are unaffected.
    """
    if x.dim() <= 2:
        return fn(x)
    lead = x.shape[:-1]
    out = fn(x.reshape(-1, x.shape[-1]))
    return out.reshape(*lead, out.shape[-1])


class ManifoldFeaturizerModule(torch.nn.Module):
    """Featurizer wrapping a manifold (spline or flow).

    Maps x -> (z, None) where z = manifold.fwd(x)[0].
    Lossless (bijective), so error term is always None.

    Use this when composing with StandardizeFeaturizer.
    """

    def __init__(self, manifold: torch.nn.Module) -> None:
        super().__init__()
        self.manifold = manifold

    def _manifold_dtype(self) -> torch.dtype:
        """Get manifold dtype from parameters or buffers."""
        for p in self.manifold.parameters():
            return p.dtype
        for b in self.manifold.buffers():
            return b.dtype
        return torch.float32

    def forward(self, x: Tensor) -> tuple[Tensor, None]:
        dtype = self._manifold_dtype()
        self.manifold.to(x.device)
        z = _over_leading_dims(
            x.to(dtype),
            lambda t: self.manifold.fwd(t)[0],  # type: ignore[attr-defined]
        )
        return z.to(x.dtype), None


class ManifoldInverseFeaturizerModule(torch.nn.Module):
    """Inverse of ManifoldFeaturizerModule."""

    def __init__(self, manifold: torch.nn.Module) -> None:
        super().__init__()
        self.manifold = manifold

    def _manifold_dtype(self) -> torch.dtype:
        """Get manifold dtype from parameters or buffers."""
        for p in self.manifold.parameters():
            return p.dtype
        for b in self.manifold.buffers():
            return b.dtype
        return torch.float32

    def forward(self, z: Tensor, error: None) -> Tensor:
        dtype = self._manifold_dtype()
        self.manifold.to(z.device)
        x = _over_leading_dims(
            z.to(dtype),
            lambda t: self.manifold.inv(t)[0],  # type: ignore[attr-defined]
        )
        return x.to(z.dtype)


class ManifoldProjectFeaturizerModule(torch.nn.Module):
    """Projects points onto the manifold surface via encode→decode round-trip.

    forward: standardize → encode → decode → unstandardize → (x_projected, None)

    Lossy: the off-manifold residual is discarded (same as ManifoldFeaturizer,
    which also silently drops the residual in its fwd call). This ensures that
    interpolation happens purely between on-manifold projected points.
    """

    def __init__(
        self,
        manifold: torch.nn.Module,
        mean: Tensor,
        std: Tensor,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.register_buffer("_mean", mean)
        self._mean: Tensor
        self.register_buffer("_std", std)
        self._std: Tensor
        self._eps = eps

    def _manifold_dtype(self) -> torch.dtype:
        for p in self.manifold.parameters():
            return p.dtype
        for b in self.manifold.buffers():
            return b.dtype
        return torch.float32

    def forward(self, x: Tensor) -> tuple[Tensor, None]:
        dtype = self._manifold_dtype()
        self.manifold.to(x.device)
        mean = self._mean.to(x.device)
        std = self._std.to(x.device)
        # standardize (broadcasts over any leading dims)
        x_std = (x - mean) / (std + self._eps)

        # encode → slice intrinsic coords → decode, projected back to standardized
        # ambient space. The manifold ops below assume a 2-D row layout (the
        # `z[:, :k]` slice in particular), so flatten the leading dims first.
        def _project(t: Tensor) -> Tensor:
            z, _ = self.manifold.fwd(t)  # type: ignore[attr-defined]
            u = z[:, : self.manifold.intrinsic_dim]  # type: ignore[attr-defined]
            return self.manifold.decode(u, r=None)  # type: ignore[attr-defined]

        x_proj_std = _over_leading_dims(x_std.to(dtype), _project).to(x.dtype)
        # unstandardize
        x_proj = x_proj_std * (std + self._eps) + mean
        return x_proj, None


class ManifoldProjectInverseFeaturizerModule(torch.nn.Module):
    """Inverse of ManifoldProjectFeaturizerModule: identity (no error to restore)."""

    def forward(self, x_proj: Tensor, error: None) -> Tensor:
        return x_proj


class ManifoldProjectFeaturizer(Featurizer):
    """Projects points onto the manifold surface via encode→decode round-trip.

    Feature space is ambient (same dimensionality as input), but constrained
    to the manifold surface. The error term captures the off-manifold residual.

    Args:
        manifold: The manifold module (SplineManifold, etc.)
        mean: Standardization mean
        std: Standardization std
        n_features: Dimension of the ambient space
        eps: Numerical stability epsilon
    """

    def __init__(
        self,
        manifold: torch.nn.Module,
        mean: Tensor,
        std: Tensor,
        n_features: int,
        *,
        eps: float = 1e-6,
        id: str = "manifold_project",
        **kwargs: Any,
    ) -> None:
        manifold.requires_grad_(False)
        super().__init__(
            ManifoldProjectFeaturizerModule(manifold, mean, std, eps),
            ManifoldProjectInverseFeaturizerModule(),
            n_features=n_features,
            id=id,
            **kwargs,
        )
        self._manifold = manifold


class ManifoldFeaturizer(Featurizer):
    FEATURIZER_MODULE_CLASS_NAME = "ManifoldFeaturizerModule"

    """Featurizer backed by a manifold (spline or flow).

    Provides bijective (lossless) mapping. Use with StandardizeFeaturizer
    for the full standardize -> manifold pipeline:

        standardize = StandardizeFeaturizer(mean, std)
        manifold_feat = ManifoldFeaturizer(manifold)
        composed = standardize >> manifold_feat

    Args:
        manifold: The manifold module (SplineManifold, RealNVP flow, etc.)
        n_features: Dimension of the manifold's ambient space
        trainable: Whether manifold gradients are enabled
    """

    def __init__(
        self,
        manifold: torch.nn.Module,
        n_features: int,
        *,
        trainable: bool = False,
        id: str = "manifold",
        **kwargs: Any,
    ) -> None:
        manifold.requires_grad_(trainable)

        super().__init__(
            ManifoldFeaturizerModule(manifold),
            ManifoldInverseFeaturizerModule(manifold),
            n_features=n_features,
            id=id,
            **kwargs,
        )
        self._trainable = trainable
        self._manifold = manifold

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifold featurizer."""
        additional_config: dict[str, Any] = {
            "requires_grad": self._trainable,
        }
        # Save manifold config for reliable reconstruction
        additional_config["manifold_config"] = self._manifold.get_config()

        return {
            "model_info": {
                "featurizer_class": "ManifoldFeaturizerModule",
                "inverse_featurizer_class": "ManifoldInverseFeaturizerModule",
                "n_features": self.n_features,
                "featurizer_id": self.id,
                "additional_config": additional_config,
            },
            "featurizer_state_dict": self.featurizer.state_dict(),
            "inverse_state_dict": self.inverse_featurizer.state_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifoldFeaturizer":
        model_info = data["model_info"]
        additional = model_info["additional_config"]
        requires_grad = additional.get("requires_grad", False)
        n_features = model_info["n_features"]
        featurizer_sd = data["featurizer_state_dict"]
        manifold_config = additional.get("manifold_config")

        manifold_sd = {
            k.removeprefix("manifold."): v
            for k, v in featurizer_sd.items()
            if k.startswith("manifold.")
        }

        if manifold_config is None or manifold_config.get("type") != "spline":
            raise ValueError(
                "ManifoldFeaturizer.from_dict only supports spline manifolds; "
                f"got manifold_config={manifold_config!r}. Normalizing-flow "
                "manifolds are no longer supported."
            )

        from causalab.methods.spline.manifold import SplineManifold

        mc = manifold_config
        manifold = SplineManifold(
            control_points=manifold_sd["control_points"],
            target_points=manifold_sd["target_points"],
            intrinsic_dim=mc["intrinsic_dim"],
            ambient_dim=mc["ambient_dim"],
            smoothness=mc["smoothness"],
            periodic_dims=mc.get("periodic_dims"),
            periods=mc.get("periods"),
        )

        feat = cls(
            manifold,
            n_features,
            trainable=requires_grad,
            id=model_info.get("featurizer_id", "manifold"),
        )

        manifold.requires_grad_(requires_grad)
        return feat
