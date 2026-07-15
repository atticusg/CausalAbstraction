"""Shared on-disk bundle writer for subspace producers.

Every subspace ``method`` (``pca`` fits a rotation; ``fixed`` loads a given
one) ultimately writes the **same** artifact bundle that the manifold pipeline
(``activation_manifold`` → ``path_steering``) reads back:

- ``train_dataset.json`` — row-aligned to the features.
- ``rotation.safetensors`` — tensor key ``rotation_matrix`` ``(d_model, k)``
  (plus ``explained_variance_ratio`` when the producer has one).
- ``features/training_features.safetensors`` — key ``features`` ``(N, k)``,
  the projected features the manifold is fit on.
- ``features/raw_features.safetensors`` — key ``features`` ``(N, d_model)``,
  the un-projected activations. ``path_steering`` needs these for the **linear**
  path mode; without them it silently drops the geodesic-vs-linear comparison
  (``path_steering/main.py``).

Factoring the writer here keeps the producers from drifting: the PCA path and
the fixed-rotation path can't disagree on layout, keys, or which files exist.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import torch
from torch import Tensor
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def save_subspace_artifacts(
    output_dir: str,
    train_dataset: list,
    rotation: Tensor,
    raw_features: Tensor,
    projected_features: Tensor,
    *,
    explained_variance_ratio: Sequence[float] | None = None,
) -> None:
    """Write the canonical subspace bundle under ``output_dir``.

    Args:
        output_dir: Subspace cell directory (e.g. ``…/subspace/pca_k8/<tv>``).
        train_dataset: Counterfactual examples, row-aligned to the features.
        rotation: ``(d_model, k)`` rotation matrix.
        raw_features: ``(N, d_model)`` un-projected activations.
        projected_features: ``(N, k)`` features projected through ``rotation``.
        explained_variance_ratio: Optional per-component ratios; written into
            ``rotation.safetensors`` only when provided (PCA has it; a given
            fixed rotation does not).
    """
    os.makedirs(output_dir, exist_ok=True)

    from causalab.io.counterfactuals import save_counterfactual_examples

    save_counterfactual_examples(
        train_dataset,
        os.path.join(output_dir, "train_dataset.json"),
    )

    rotation_payload: dict[str, Tensor] = {"rotation_matrix": rotation.contiguous()}
    if explained_variance_ratio is not None:
        rotation_payload["explained_variance_ratio"] = torch.tensor(
            list(explained_variance_ratio), dtype=torch.float32
        )
    save_file(rotation_payload, os.path.join(output_dir, "rotation.safetensors"))

    features_dir = os.path.join(output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    save_file(
        {"features": projected_features.contiguous()},
        os.path.join(features_dir, "training_features.safetensors"),
    )
    save_file(
        {"features": raw_features.contiguous()},
        os.path.join(features_dir, "raw_features.safetensors"),
    )
