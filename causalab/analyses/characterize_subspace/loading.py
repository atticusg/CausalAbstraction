"""Adaptive subspace loader.

Phase-1 hands over a ``.safetensors`` artifact whose tensor layout is not
standardised — different upstream pipelines emit different key names and
either ``(d_model, k)`` or ``(k, d_model)`` orientations. This module
inspects the file and produces a uniform :class:`SubspaceProjector` the
rest of the analysis can use.

Square rotation matrices (``d_model == k``) are ambiguous in orientation;
in that case the caller must supply ``k_features_hint`` as an int so we
know which axis is which.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor

logger = logging.getLogger(__name__)


_CANDIDATE_KEYS = (
    "rotation_matrix",
    "rotation",
    "directions",
    "subspace",
    "basis",
    "U",
    "components",
)


@dataclass(frozen=True)
class SubspaceProjector:
    """A subspace projector with rotation matrix in ``(d_model, k)`` orientation.

    Use :func:`project` for batched projection. The rotation is kept on CPU
    by default; callers move it to the activation device at projection time.
    """

    rotation: Tensor
    k: int
    d_model: int
    source: dict[str, Any]


def _pick_tensor(data: dict[str, Tensor]) -> tuple[str, Tensor]:
    """Choose the rotation tensor from a multi-key safetensors payload."""
    for name in _CANDIDATE_KEYS:
        if name in data:
            return name, data[name]
    # Fall back to the single tensor if there is exactly one.
    if len(data) == 1:
        only_key = next(iter(data))
        return only_key, data[only_key]
    # Otherwise pick the largest 2-D tensor and warn.
    twod = [(k, t) for k, t in data.items() if t.ndim == 2]
    if not twod:
        raise ValueError(
            "Subspace artifact contains no 2-D tensor and no recognised key. "
            f"Tensors present: {sorted(data)}"
        )
    twod.sort(key=lambda kt: kt[1].numel(), reverse=True)
    name, tensor = twod[0]
    logger.warning(
        "Subspace artifact has no recognised key; falling back to largest 2-D "
        "tensor %r (shape=%s).",
        name,
        tuple(tensor.shape),
    )
    return name, tensor


def _orient(
    tensor: Tensor,
    *,
    k_features_hint: int | str,
    d_model_hint: int | None,
) -> tuple[Tensor, int, int]:
    """Return ``(rotation_d_k, k, d_model)`` with the tensor in ``(d_model, k)`` orientation.

    Resolution rules:
    - Both axes equal: ``k_features_hint`` must be an int. Treat it as ``k``.
    - One axis matches ``d_model_hint`` (when supplied): the other axis is ``k``.
    - ``k_features_hint`` is an int and exactly one axis matches: that axis is ``k``.
    - Default heuristic: the smaller axis is ``k`` (subspaces are usually low-rank).
    """
    if tensor.ndim == 1:
        # Single direction; treat as (d_model, 1).
        return tensor.unsqueeze(1).contiguous(), 1, int(tensor.shape[0])
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected a 1-D or 2-D rotation tensor; got shape {tuple(tensor.shape)}."
        )

    a, b = int(tensor.shape[0]), int(tensor.shape[1])

    if a == b:
        if not isinstance(k_features_hint, int):
            raise ValueError(
                f"Subspace rotation is square ({a}x{b}); orientation is ambiguous. "
                "Pass an integer k_features_hint to disambiguate (the value of k)."
            )
        k = k_features_hint
        if k not in (a, b):
            raise ValueError(
                f"k_features_hint={k} does not match either axis of the square "
                f"rotation tensor (shape={a}x{b})."
            )
        return tensor.contiguous(), k, a

    # Non-square: resolve which axis is d_model.
    if d_model_hint is not None and d_model_hint in (a, b):
        if a == d_model_hint:
            return tensor.contiguous(), b, a
        return tensor.t().contiguous(), a, b

    if isinstance(k_features_hint, int) and k_features_hint in (a, b):
        if b == k_features_hint:
            return tensor.contiguous(), b, a
        return tensor.t().contiguous(), a, b

    # Heuristic: smaller axis is k.
    if a < b:
        return tensor.t().contiguous(), a, b
    return tensor.contiguous(), b, a


def load_subspace(
    artifact_path: str,
    *,
    k_features_hint: int | str = "auto",
    d_model_hint: int | None = None,
) -> SubspaceProjector:
    """Load a subspace projector from a ``.safetensors`` artifact.

    Args:
        artifact_path: Path to the ``.safetensors`` file.
        k_features_hint: Either ``"auto"`` or an explicit ``k`` value. Required
            (as int) when the rotation tensor is square.
        d_model_hint: Optional hint for the model's hidden size. When known,
            it removes orientation ambiguity for non-square tensors.

    Returns:
        A :class:`SubspaceProjector` with the rotation in ``(d_model, k)``
        orientation, ready for projection.
    """
    data = load_file(artifact_path)
    chosen_key, raw = _pick_tensor(data)
    rotation, k, d_model = _orient(
        raw, k_features_hint=k_features_hint, d_model_hint=d_model_hint
    )
    other_keys = sorted(k for k in data if k != chosen_key)
    logger.info(
        "Loaded subspace from %s: key=%r, shape=(%d, %d) (d_model x k). "
        "Auxiliary tensors in artifact: %s",
        artifact_path,
        chosen_key,
        d_model,
        k,
        other_keys or "(none)",
    )
    return SubspaceProjector(
        rotation=rotation.detach().to(torch.float32),
        k=k,
        d_model=d_model,
        source={
            "artifact_path": artifact_path,
            "chosen_key": chosen_key,
            "other_keys": other_keys,
        },
    )


def project(activations: Tensor, projector: SubspaceProjector) -> Tensor:
    """Project ``(..., d_model)`` activations onto the subspace.

    Returns a tensor with shape ``(..., k)``. Rotation is moved to the
    activation device and dtype on the fly so callers don't need to manage
    device placement.
    """
    if activations.shape[-1] != projector.d_model:
        raise ValueError(
            f"Activation last dim ({activations.shape[-1]}) does not match "
            f"projector d_model ({projector.d_model})."
        )
    rot = projector.rotation.to(device=activations.device, dtype=activations.dtype)
    return activations @ rot
