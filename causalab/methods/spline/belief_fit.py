"""Fit a TPS manifold in Hellinger belief space.

Centroids are computed in probability space (per-class averages on the simplex),
then mapped to Hellinger space via the sqrt transform.  The TPS interpolates
through these sqrt-probability centroids.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from torch import Tensor

from causalab.io.artifacts import load_tensors_with_meta, save_tensors_with_meta
from causalab.methods.spline.builders import (
    build_spline_manifold,
    detect_periodic_dims,
    remap_periodic_to_angle,
)
from causalab.methods.spline.manifold import SplineManifold

logger = logging.getLogger(__name__)

EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────


CKPT_STEM = "ckpt_final"


def _save_spline_checkpoint(
    spline_dir: str,
    manifold: SplineManifold,
    config_dict: dict[str, Any],
    mean: Tensor,
    std: Tensor,
    eps: float,
) -> None:
    """Write a spline-manifold checkpoint as safetensors + meta JSON.

    Schema (under ``spline_dir/ckpt_final.{safetensors,meta.json}``):
      tensors:
        - ``manifold.control_points``  (n_centroids, intrinsic_dim)
        - ``manifold.centroids``       (n_centroids, ambient_dim)
        - ``preprocess.mean``          (ambient_dim,)
        - ``preprocess.std``           (ambient_dim,)
      meta:
        - ``manifold_type``: "spline"
        - ``manifold_state``: scalar fields needed by ``SplineManifold.from_state_dict``
        - ``config``: per-call hyperparams (mirrors the legacy payload)
        - ``preprocess.eps``
    """
    sd = manifold.state_dict_to_save()
    tensors = {
        "manifold.control_points": sd["control_points"],
        "manifold.centroids": sd["centroids"],
        "preprocess.mean": mean.cpu(),
        "preprocess.std": std.cpu(),
    }
    meta: dict[str, Any] = {
        "manifold_type": "spline",
        "manifold_state": {
            "intrinsic_dim": int(sd["intrinsic_dim"]),
            "ambient_dim": int(sd["ambient_dim"]),
            "smoothness": float(sd["smoothness"]),
            "periodic_dims": list(sd["periodic_dims"]),
            "periods": [float(p) for p in sd["periods"]],
            "spline_method": sd["spline_method"],
            "sphere_project": bool(sd["sphere_project"]),
        },
        "config": dict(config_dict),
        "preprocess": {"eps": float(eps)},
    }
    save_tensors_with_meta(tensors, meta, spline_dir, CKPT_STEM)


def _load_spline_checkpoint(
    spline_dir: str,
) -> tuple[SplineManifold, dict[str, Any], dict[str, Tensor]]:
    """Inverse of :func:`_save_spline_checkpoint`.

    Returns the reconstructed manifold, the meta dict, and a dict of the
    preprocess tensors keyed by their leaf name (``mean`` / ``std``).
    """
    tensors, meta = load_tensors_with_meta(spline_dir, CKPT_STEM)
    ms = meta["manifold_state"]
    manifold = SplineManifold.from_state_dict(
        {
            "control_points": tensors["manifold.control_points"],
            "centroids": tensors["manifold.centroids"],
            "intrinsic_dim": ms["intrinsic_dim"],
            "ambient_dim": ms["ambient_dim"],
            "smoothness": ms["smoothness"],
            "periodic_dims": ms.get("periodic_dims") or [],
            "periods": ms.get("periods") or [],
            "spline_method": ms.get("spline_method", "auto"),
            "sphere_project": ms.get("sphere_project", False),
        }
    )
    preprocess = {
        "mean": tensors["preprocess.mean"],
        "std": tensors["preprocess.std"],
    }
    return manifold, meta, preprocess


def load_output_manifold(
    experiment_root: str,
    output_manifold_sub: str,
) -> tuple[SplineManifold, Tensor]:
    """Load a pre-fitted output manifold from an output_manifold analysis checkpoint.

    Args:
        experiment_root: Root experiment directory.
        output_manifold_sub: Subdirectory name (e.g. "spline_s0.0").

    Returns:
        belief_spline: Loaded SplineManifold in Hellinger space.
        centroid_hellinger: (W, W+1) Hellinger centroids (sqrt-probability).
    """
    spline_dir = os.path.join(
        experiment_root,
        "output_manifold",
        output_manifold_sub,
        "manifold_spline",
    )
    expected = os.path.join(spline_dir, f"{CKPT_STEM}.safetensors")
    if not os.path.exists(expected):
        raise FileNotFoundError(
            f"Output manifold checkpoint not found at {expected}. "
            f"Run the output_manifold analysis first."
        )
    belief_spline, _meta, _preprocess = _load_spline_checkpoint(spline_dir)
    centroid_hellinger = belief_spline.centroids
    logger.info(
        "Loaded output manifold from %s (intrinsic_dim=%d, n_centroids=%d)",
        expected,
        belief_spline.intrinsic_dim,
        belief_spline.n_centroids,
    )
    return belief_spline, centroid_hellinger


def hellinger_to_simplex(h: Tensor, eps: float = 1e-8) -> Tensor:
    """Map sqrt-probability vectors back to the probability simplex.

    Projects onto the unit sphere first so that squaring produces a valid
    distribution.  This avoids the geometric distortion from renormalizing
    *after* squaring (radial projection on the simplex ≠ closest point on
    the sphere in Fisher-Rao geometry).
    """
    h = h.double().clamp(min=0)
    h = h / h.norm(dim=-1, keepdim=True).clamp(min=eps)  # project onto unit sphere
    return (h**2).float()


# ─────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────


def _prob_to_hellinger(p: Tensor) -> Tensor:
    """Map probability vectors to Hellinger space: p -> sqrt(p)."""
    return torch.sqrt(p.double().clamp(min=EPS))


def _eval_belief_reconstruction(
    manifold: SplineManifold,
    centroid_hellinger: Tensor,
) -> dict[str, Any]:
    """Encode centroids, decode back, measure reconstruction quality."""
    device = manifold.centroids.device
    h = centroid_hellinger.float().to(device)

    u, _ = manifold.encode(h)
    h_recon = manifold.decode(u)

    diff = h.double() - h_recon.double()
    mse = diff.pow(2).mean().item()

    # Hellinger distance: H(p, q) = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||
    # Since we're already in sqrt-space, this is (1/sqrt(2)) * ||h - h_recon||
    per_class_hellinger = diff.pow(2).sum(dim=-1).clamp(min=0).sqrt() / (2.0**0.5)
    mean_hellinger = per_class_hellinger.mean().item()
    max_hellinger = per_class_hellinger.max().item()

    return {
        "recon_mse": mse,
        "mean_hellinger": mean_hellinger,
        "max_hellinger": max_hellinger,
        "per_class_hellinger": per_class_hellinger.tolist(),
    }


def _save_belief_checkpoint(
    spline_dir: str,
    manifold: SplineManifold,
    config_dict: dict[str, Any],
) -> None:
    """Save belief manifold checkpoint in the same format as the manifold analysis."""
    W_plus_1 = manifold.ambient_dim
    _save_spline_checkpoint(
        spline_dir,
        manifold,
        config_dict,
        mean=torch.zeros(W_plus_1),
        std=torch.ones(W_plus_1),
        eps=EPS,
    )
    logger.info(
        "Saved belief manifold checkpoint to %s",
        os.path.join(spline_dir, f"{CKPT_STEM}.safetensors"),
    )


def _compute_intrinsic_ranges(
    manifold: SplineManifold,
) -> tuple[tuple[float, float], ...]:
    """Compute intrinsic coordinate ranges from control points."""
    cp = manifold.control_points.cpu()
    ranges = []
    for dim_i in range(cp.shape[1]):
        col = cp[:, dim_i]
        if manifold.periodic_dims and dim_i in manifold.periodic_dims:
            idx = manifold.periodic_dims.index(dim_i)
            ranges.append((0.0, manifold.periods[idx]))
        else:
            margin = 0.05 * (col.max() - col.min()).item()
            ranges.append((col.min().item() - margin, col.max().item() + margin))
    return tuple(ranges)


# ─────────────────────────────────────────────────────────────────────
# PCA intrinsic mode
# ─────────────────────────────────────────────────────────────────────


def fit_belief_tps_pca(
    per_class_dists: Tensor,
    hellinger_pca,
    intrinsic_dim: int,
    smoothness: float,
    output_dir: str,
    max_control_points: str | int = "all",
) -> dict[str, Any]:
    """Fit a belief TPS using PCA of Hellinger centroids as intrinsic coordinates.

    Args:
        per_class_dists: (W, W+1) probability centroids (computed in prob space).
        hellinger_pca: Fitted sklearn PCA from belief_manifold (in sqrt-p space).
        intrinsic_dim: Number of intrinsic dimensions.
        smoothness: TPS smoothness parameter.
        output_dir: Directory to save checkpoint and metadata.

    Returns:
        Dict with manifold, centroid_hellinger, control_points, metadata.
    """
    W, D = per_class_dists.shape  # D = W+1
    centroid_hellinger = _prob_to_hellinger(per_class_dists)  # (W, D)

    # Project centroids through the belief_manifold PCA
    n_check = min(2 * intrinsic_dim, D)
    centroid_pca_all = hellinger_pca.transform(
        centroid_hellinger.numpy()
    )  # (W, n_components)
    centroid_pca = torch.from_numpy(
        centroid_pca_all[:, :n_check]
    ).double()  # (W, n_check)

    # Periodicity detection
    eigenvalues = centroid_pca.var(dim=0)  # (n_check,)
    periodic_pairs = detect_periodic_dims(centroid_pca, eigenvalues)

    if periodic_pairs:
        logger.info("Belief TPS: detected periodic pairs %s", periodic_pairs)
        new_points, periodic_dim_indices, periods = remap_periodic_to_angle(
            centroid_pca,
            periodic_pairs,
            eigenvalues,
        )
        non_periodic_cols = [
            i for i in range(new_points.shape[1]) if i not in periodic_dim_indices
        ]
        selected = periodic_dim_indices[:intrinsic_dim]
        remaining = intrinsic_dim - len(selected)
        if remaining > 0:
            selected.extend(non_periodic_cols[:remaining])

        control_points = new_points[:, selected]
        periodic_dims_list = list(range(len(periodic_dim_indices[:intrinsic_dim])))
        periods_list = periods[: len(periodic_dims_list)]
    else:
        logger.info(
            "Belief TPS: no periodic pairs — using top %d PCA components", intrinsic_dim
        )
        control_points = centroid_pca[:, :intrinsic_dim]
        periodic_dims_list = None
        periods_list = None

    # Optionally subsample control points (e.g. for large domains)
    if isinstance(max_control_points, int):
        from causalab.methods.spline.train import _subsample_control_points

        control_points, centroid_hellinger = _subsample_control_points(
            control_points.float(),
            centroid_hellinger.float(),
            max_control_points,
        )
        control_points = control_points.double()
        centroid_hellinger = centroid_hellinger.double()
        logger.info(
            "Subsampled to %d control points (max=%d)",
            control_points.shape[0],
            max_control_points,
        )

    # Fit TPS: control_points (W, d) -> centroid_hellinger (W, D)
    belief_spline = build_spline_manifold(
        control_points=control_points.float(),
        centroids=centroid_hellinger.float(),
        intrinsic_dim=intrinsic_dim,
        ambient_dim=D,
        smoothness=smoothness,
        device="cpu",
        periodic_dims=periodic_dims_list,
        periods=periods_list,
        sphere_project=True,  # belief manifold: decode → unit L2 sphere
    )

    logger.info(
        "Fitted belief TPS (pca): %d control points, intrinsic_dim=%d, ambient_dim=%d",
        control_points.shape[0],
        intrinsic_dim,
        D,
    )

    # Evaluate reconstruction
    recon_metrics = _eval_belief_reconstruction(belief_spline, centroid_hellinger)

    # Save
    spline_dir = os.path.join(output_dir, "manifold_spline")
    os.makedirs(spline_dir, exist_ok=True)

    config_dict = {
        "smoothness": smoothness,
        "intrinsic_dim": intrinsic_dim,
        "ambient_dim": D,
        "n_centroids": W,
        "intrinsic_mode": "pca",
    }
    _save_belief_checkpoint(spline_dir, belief_spline, config_dict)

    spline_meta = {
        "intrinsic_mode": "pca",
        "intrinsic_dim": intrinsic_dim,
        "ambient_dim": D,
        "n_centroids": W,
        "smoothness": smoothness,
        **recon_metrics,
    }
    with open(os.path.join(spline_dir, "metadata.json"), "w") as f:
        json.dump(spline_meta, f, indent=2)

    metrics_line = json.dumps({"step": 0, **recon_metrics})
    with open(os.path.join(output_dir, "metrics.jsonl"), "w") as f:
        f.write(metrics_line + "\n")

    return {
        "manifold": belief_spline,
        "centroid_hellinger": centroid_hellinger,
        "control_points": control_points,
        "metadata": recon_metrics,
    }


# ─────────────────────────────────────────────────────────────────────
# Parameter intrinsic mode
# ─────────────────────────────────────────────────────────────────────


def fit_belief_tps_parameter(
    per_class_dists: Tensor,
    causal_model,
    intervention_variable: str,
    intervention_values: list,
    intrinsic_dim: int,
    smoothness: float,
    output_dir: str,
    max_control_points: str | int = "all",
) -> dict[str, Any]:
    """Fit a belief TPS using causal parameter values as intrinsic coordinates.

    Args:
        per_class_dists: (W, W+1) probability centroids (computed in prob space).
        causal_model: CausalModel with embeddings and periods.
        intervention_variable: Name of the intervention variable.
        intervention_values: List of intervention values (one per class).
        intrinsic_dim: Number of intrinsic dimensions.
        smoothness: TPS smoothness parameter.
        output_dir: Directory to save checkpoint and metadata.

    Returns:
        Dict with manifold, centroid_hellinger, control_points, metadata.
    """
    W, D = per_class_dists.shape  # D = W+1
    centroid_hellinger = _prob_to_hellinger(per_class_dists)  # (W, D)

    embeddings = causal_model.embeddings or {}
    embedding_fn = embeddings.get(intervention_variable)

    # Build control points from parameter values
    coords_list = []
    for val in intervention_values:
        if embedding_fn is not None:
            c = embedding_fn(val)
            coords_list.append(c)
        elif isinstance(val, (tuple, list)):
            coords_list.append([float(v) for v in val])
        else:
            coords_list.append([float(val)])

    control_points = torch.tensor(coords_list, dtype=torch.float64)  # (W, d)

    if control_points.shape[1] != intrinsic_dim:
        logger.warning(
            "Parameter embedding dim (%d) != intrinsic_dim (%d). Using embedding dim.",
            control_points.shape[1],
            intrinsic_dim,
        )
        intrinsic_dim = control_points.shape[1]

    # Periodic dims from causal model: in parameter mode the period is declared
    # explicitly on the variable, so trust it directly. For 1D variables the
    # key is the bare variable name; for multi-dim variables (e.g. graph_walk
    # node_coordinates) the keys are dim-suffixed ("node_coordinates_0",
    # "node_coordinates_1", ...) — same convention as the activation-side
    # spline trainer.
    periodic_dims_list = None
    periods_list = None
    periods_dict = causal_model.periods or {}
    if intervention_variable in periods_dict:
        periodic_dims_list = [0]
        periods_list = [float(periods_dict[intervention_variable])]
    elif intrinsic_dim > 1:
        pds, ps = [], []
        for d in range(intrinsic_dim):
            key = f"{intervention_variable}_{d}"
            if key in periods_dict:
                pds.append(d)
                ps.append(float(periods_dict[key]))
        if pds:
            periodic_dims_list = pds
            periods_list = ps

    # Optionally subsample control points (e.g. for large domains)
    if isinstance(max_control_points, int):
        from causalab.methods.spline.train import _subsample_control_points

        control_points, centroid_hellinger = _subsample_control_points(
            control_points.float(),
            centroid_hellinger.float(),
            max_control_points,
        )
        control_points = control_points.double()
        centroid_hellinger = centroid_hellinger.double()
        logger.info(
            "Subsampled to %d control points (max=%d)",
            control_points.shape[0],
            max_control_points,
        )

    # Fit TPS: control_points (W, d) -> centroid_hellinger (W, D)
    belief_spline = build_spline_manifold(
        control_points=control_points.float(),
        centroids=centroid_hellinger.float(),
        intrinsic_dim=intrinsic_dim,
        ambient_dim=D,
        smoothness=smoothness,
        device="cpu",
        periodic_dims=periodic_dims_list,
        periods=periods_list,
        sphere_project=True,  # belief manifold: decode → unit L2 sphere
    )

    logger.info(
        "Fitted belief TPS (parameter): %d control points, intrinsic_dim=%d, ambient_dim=%d",
        control_points.shape[0],
        intrinsic_dim,
        D,
    )

    # Evaluate reconstruction
    recon_metrics = _eval_belief_reconstruction(belief_spline, centroid_hellinger)

    # Save
    spline_dir = os.path.join(output_dir, "manifold_spline")
    os.makedirs(spline_dir, exist_ok=True)

    config_dict = {
        "smoothness": smoothness,
        "intrinsic_dim": intrinsic_dim,
        "ambient_dim": D,
        "n_centroids": W,
        "intrinsic_mode": "parameter",
    }
    _save_belief_checkpoint(spline_dir, belief_spline, config_dict)

    spline_meta = {
        "intrinsic_mode": "parameter",
        "intrinsic_dim": intrinsic_dim,
        "ambient_dim": D,
        "n_centroids": W,
        "smoothness": smoothness,
        **recon_metrics,
    }
    with open(os.path.join(spline_dir, "metadata.json"), "w") as f:
        json.dump(spline_meta, f, indent=2)

    metrics_line = json.dumps({"step": 0, **recon_metrics})
    with open(os.path.join(output_dir, "metrics.jsonl"), "w") as f:
        f.write(metrics_line + "\n")

    return {
        "manifold": belief_spline,
        "centroid_hellinger": centroid_hellinger,
        "control_points": control_points,
        "metadata": recon_metrics,
    }
