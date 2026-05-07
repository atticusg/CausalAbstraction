"""Safetensors+meta serialization for sklearn PCA instances.

A fitted sklearn.decomposition.PCA carries portable array state plus a
handful of scalar hyperparameters. We save the arrays as safetensors and
the scalars as JSON, then rebuild a usable PCA on load by assigning the
arrays back onto a fresh instance — no re-fit, no pickle.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
from sklearn.decomposition import PCA

from causalab.io.artifacts import (
    save_tensors_with_meta,
    load_tensors_with_meta,
)


_TENSOR_FIELDS = (
    "components_",
    "mean_",
    "explained_variance_",
    "explained_variance_ratio_",
    "singular_values_",
    "noise_variance_",
)


def save_pca(pca: PCA, output_dir: str, stem: str = "hellinger_pca") -> Tuple[str, str]:
    """Save a fitted sklearn PCA via safetensors + meta JSON.

    Tensors: components_, mean_, explained_variance_, explained_variance_ratio_,
    singular_values_, noise_variance_ (each a 1D or 2D np.ndarray).
    Meta: scalar fit attributes needed to rebuild a usable PCA.
    """
    tensors = {}
    for field in _TENSOR_FIELDS:
        arr = getattr(pca, field, None)
        if arr is None:
            continue
        # noise_variance_ may be a python float on some sklearn versions
        tensors[field] = torch.from_numpy(
            np.atleast_1d(np.asarray(arr, dtype=np.float64)).copy()
        )
    meta = {
        "n_components_": int(pca.n_components_),
        "n_features_in_": int(pca.n_features_in_),
        "n_samples_": int(getattr(pca, "n_samples_", 0)),
        "whiten": bool(pca.whiten),
        "svd_solver": str(pca.svd_solver),
    }
    return save_tensors_with_meta(tensors, meta, output_dir, stem)


def load_pca(output_dir: str, stem: str = "hellinger_pca") -> PCA:
    """Load a PCA previously written by `save_pca`."""
    tensors, meta = load_tensors_with_meta(output_dir, stem)
    pca = PCA(
        n_components=meta["n_components_"],
        whiten=meta["whiten"],
        svd_solver=meta["svd_solver"],
    )
    # Restore fitted state. sklearn expects ndarrays.
    pca.components_ = tensors["components_"].numpy()
    pca.mean_ = tensors["mean_"].numpy().squeeze()
    pca.explained_variance_ = tensors["explained_variance_"].numpy()
    pca.explained_variance_ratio_ = tensors["explained_variance_ratio_"].numpy()
    if "singular_values_" in tensors:
        pca.singular_values_ = tensors["singular_values_"].numpy()
    if "noise_variance_" in tensors:
        nv = tensors["noise_variance_"].numpy()
        pca.noise_variance_ = float(nv) if nv.size == 1 else nv
    pca.n_components_ = meta["n_components_"]
    pca.n_features_in_ = meta["n_features_in_"]
    if meta.get("n_samples_"):
        pca.n_samples_ = meta["n_samples_"]
    return pca
