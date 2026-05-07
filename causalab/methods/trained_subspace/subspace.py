"""Subspace (DAS / SVD) featurizer."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor
import pyvene as pv  # type: ignore[import-untyped]

from causalab.neural.featurizer import Featurizer


class SubspaceFeaturizerModule(torch.nn.Module):
    """Linear projector onto an orthogonal *rotation* sub-space."""

    def __init__(
        self, rotate_layer: torch.nn.Module
    ) -> None:  # pv.models.layers.LowRankRotateLayer
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
    ) -> None:  # pv.models.layers.LowRankRotateLayer
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
            rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=True)
        else:
            shape = rotation_subspace.shape
            rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=False)
            rotate.weight.data.copy_(rotation_subspace)

        rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
        rotate.requires_grad_(trainable)

        super().__init__(
            SubspaceFeaturizerModule(rotate),
            SubspaceInverseFeaturizerModule(rotate),
            n_features=rotate.weight.shape[1],
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

    def save_modules(self, path: str) -> tuple[str, str]:
        model_info = {
            "featurizer_class": "SubspaceFeaturizerModule",
            "inverse_featurizer_class": "SubspaceInverseFeaturizerModule",
            "n_features": self.n_features,
            "featurizer_id": self.id,
            "additional_config": self._rotation_config(),
        }
        torch.save(
            {"model_info": model_info, "state_dict": self.featurizer.state_dict()},
            f"{path}_featurizer",
        )
        torch.save(
            {
                "model_info": model_info,
                "state_dict": self.inverse_featurizer.state_dict(),
            },
            f"{path}_inverse_featurizer",
        )
        return f"{path}_featurizer", f"{path}_inverse_featurizer"

    @classmethod
    def load_modules(cls, path: str) -> "SubspaceFeaturizer":
        featurizer_data = torch.load(f"{path}_featurizer")
        inverse_data = torch.load(f"{path}_inverse_featurizer")
        model_info = featurizer_data["model_info"]
        rot = model_info["additional_config"]["rotation_matrix"]
        requires_grad = model_info["additional_config"]["requires_grad"]
        feat = cls(
            rotation_subspace=rot,
            trainable=requires_grad,
            id=model_info.get("featurizer_id", "subspace"),
        )
        feat.featurizer.load_state_dict(featurizer_data["state_dict"])
        feat.inverse_featurizer.load_state_dict(inverse_data["state_dict"])
        assert feat.featurizer.rotate.weight.shape == rot.shape, (  # type: ignore[attr-defined]
            "Rotation-matrix shape mismatch after deserialisation."
        )
        return feat


def build_SVD_featurizers(
    model_units: list[Any],
    svd_results: dict[str, dict[str, Any]],
    trainable: bool = False,
    featurizer_id: str = "SVD",
) -> None:
    """Attach SubspaceFeaturizers to model units from pre-computed SVD results.

    Each unit's id must appear in ``svd_results``; ``svd_results[id]["rotation"]``
    is used as the rotation matrix.
    """
    for model_unit in model_units:
        if model_unit.id not in svd_results:
            raise ValueError(
                f"Model unit '{model_unit.id}' not found in svd_results. "
                f"Available IDs: {list(svd_results.keys())}"
            )
        rotation = svd_results[model_unit.id]["rotation"]
        featurizer = SubspaceFeaturizer(
            rotation_subspace=rotation,
            trainable=trainable,
            id=featurizer_id,
        )
        model_unit.set_featurizer(featurizer)
        model_unit.set_feature_indices(None)
