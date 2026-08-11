"""Subspace (DAS / SVD) featurizer."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from causalab.neural.featurizer import Featurizer


class LowRankRotateLayer(torch.nn.Module):
    """An ``(n, m)`` rotation weight consumed via ``.weight`` — pyvene's
    ``models.layers.LowRankRotateLayer``, inlined at the MX2 reroute (#409) so
    the DAS stack is backbone-free. Same parameter name, shape, and
    ``init_orth`` initialization, so the orthogonal-parametrized state dict
    (``rotate.parametrizations.weight.{0.base, original}``) that the subspace
    goldens pin is unchanged."""

    def __init__(self, n: int, m: int, init_orth: bool = True) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)


class SubspaceFeaturizerModule(torch.nn.Module):
    """Linear projector onto an orthogonal *rotation* sub-space."""

    def __init__(
        self, rotate_layer: torch.nn.Module
    ) -> None:  # LowRankRotateLayer (orthogonal-parametrized)
        super().__init__()
        self.rotate = rotate_layer

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r = cast(Tensor, self.rotate.weight).to(x.device).T  # (out, in)^T
        f = x.to(r.dtype) @ r.T
        error = x - (f @ r).to(x.dtype)
        return f, error


class SubspaceInverseFeaturizerModule(torch.nn.Module):
    """Inverse of :class:`SubspaceFeaturizerModule`."""

    def __init__(
        self, rotate_layer: torch.nn.Module
    ) -> None:  # LowRankRotateLayer (orthogonal-parametrized)
        super().__init__()
        self.rotate = rotate_layer

    def forward(self, f: Tensor, error: Tensor | None = None) -> Tensor:
        r = cast(Tensor, self.rotate.weight).to(f.device).T
        result = (f.to(r.dtype) @ r).to(f.dtype)
        if error is not None:
            result = result + error.to(f.dtype)
        return result


class SubspaceFeaturizer(Featurizer):
    """Orthogonal linear sub-space featurizer."""

    FEATURIZER_MODULE_CLASS_NAME = "SubspaceFeaturizerModule"

    def __init__(
        self,
        *,
        shape: tuple[int, int] | None = None,
        rotation_subspace: Tensor | None = None,
        trainable: bool = True,
        id: str = "subspace",
        **kwargs: Any,
    ) -> None:
        assert shape is not None or rotation_subspace is not None, (
            "Provide either `shape` or `rotation_subspace`."
        )

        if shape is not None:
            rotate = LowRankRotateLayer(*shape, init_orth=True)
        else:
            assert rotation_subspace is not None  # validated by assert at top
            shape = cast("tuple[int, int]", tuple(rotation_subspace.shape))
            rotate = LowRankRotateLayer(*shape, init_orth=False)
            rotate.weight.data.copy_(rotation_subspace)

        rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
        rotate.requires_grad_(trainable)

        weight = cast(Tensor, rotate.weight)
        super().__init__(
            SubspaceFeaturizerModule(rotate),
            SubspaceInverseFeaturizerModule(rotate),
            n_features=int(weight.shape[1]),
            id=id,
            **kwargs,
        )

    def _rotation_config(self) -> dict[str, Any]:
        weight = cast(Tensor, self.featurizer.rotate.weight)  # type: ignore[attr-defined]
        return {
            "rotation_matrix": weight.detach().clone(),
            "requires_grad": weight.requires_grad,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_info": {
                "featurizer_class": "SubspaceFeaturizerModule",
                "inverse_featurizer_class": "SubspaceInverseFeaturizerModule",
                "n_features": self.n_features,
                "featurizer_id": self.id,
                "additional_config": self._rotation_config(),
            },
            "featurizer_state_dict": self.featurizer.state_dict(),
            "inverse_state_dict": self.inverse_featurizer.state_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubspaceFeaturizer":
        model_info = data["model_info"]
        rot = model_info["additional_config"]["rotation_matrix"]
        requires_grad = model_info["additional_config"]["requires_grad"]
        feat = cls(
            rotation_subspace=rot,
            trainable=requires_grad,
            id=model_info.get("featurizer_id", "subspace"),
        )
        feat.featurizer.load_state_dict(data["featurizer_state_dict"])
        feat.inverse_featurizer.load_state_dict(data["inverse_state_dict"])
        return feat
