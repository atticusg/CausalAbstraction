"""
Train Spline Manifold - Learn a d-dimensional manifold using spline interpolation.

This module trains a SplineManifold on collected DAS/PCA features by fitting
a Thin-Plate Spline through parameter centroids. Unlike neural flows, this is
a one-shot fitting process that directly interpolates through observed centroids.

The output is designed for use with steer_manifold.py:
- Returns the SplineManifold for composition into featurizers
- Returns preprocessing params (mean, std) for standardization
- Checkpoint format compatible with manifold loading

Output Structure:
================
output_dir/
├── metadata.json          # Config + centroid stats + final metrics
├── ckpt_final.pt          # Final checkpoint with spline weights
└── metrics.jsonl          # Reconstruction metrics (single entry)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Callable, cast

import torch
from torch import Tensor

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.spline import SplineManifold
from causalab.methods.spline.builders import (
    extract_parameters_from_dataset,
    compute_centroids,
    detect_periodic_dims,
    remap_periodic_to_angle,
    build_spline_manifold,
)
from causalab.neural.units import InterchangeTarget, AtomicModelUnit
from causalab.neural.pipeline import Pipeline
from causalab.neural.activations.collect import collect_features
from causalab.causal.causal_model import CausalModel

logger = logging.getLogger(__name__)


@dataclass
class SplineManifoldConfig:
    """Configuration for spline manifold fitting."""

    smoothness: float = 0.0
    min_samples_per_centroid: int = 1
    max_control_points: str | int = "all"
    chord_length: bool = False
    device: str = "auto"
    batch_size: int = 512
    intrinsic_mode: str = "parameter"
    spline_method: str = "auto"

    @property
    def standardize_coords(self) -> bool:
        """Standardize coords for 'parameter' mode, not for 'pca'."""
        return self.intrinsic_mode != "pca"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _has_subspace_rotation(featurizer: Any) -> bool:
    """Check if featurizer has a dimensionality-reducing module (SubspaceFeaturizer rotation or UMAP encoder)."""
    if hasattr(featurizer, "featurizer"):
        mod = featurizer.featurizer
        if hasattr(mod, "rotate") or hasattr(mod, "encoder"):
            return True

    if hasattr(featurizer, "stages"):
        for stage in featurizer.stages:
            if hasattr(stage, "featurizer"):
                mod = stage.featurizer
                if hasattr(mod, "rotate") or hasattr(mod, "encoder"):
                    return True

    return False


def _validate_target(target: InterchangeTarget) -> AtomicModelUnit:
    """Validate target has single unit with SubspaceFeaturizer, return it."""
    units = target.flatten()
    if len(units) != 1:
        raise ValueError(f"Expected single unit, got {len(units)}")

    unit = units[0]
    featurizer = unit.featurizer

    # Check it's a SubspaceFeaturizer (or compatible with DAS rotation)
    if not _has_subspace_rotation(featurizer):
        raise ValueError(
            f"Expected SubspaceFeaturizer with rotation, got {type(featurizer)}"
        )

    return unit


def _save_checkpoint(
    output_dir: str,
    manifold: SplineManifold,
    mean: Tensor,
    std: Tensor,
    config: SplineManifoldConfig,
) -> None:
    """Save checkpoint compatible with manifold loading.

    Writes ``<output_dir>/ckpt_final.{safetensors,meta.json}`` via the shared
    spline-checkpoint helper in :mod:`causalab.methods.spline.belief_fit`.
    """
    from causalab.methods.spline.belief_fit import _save_spline_checkpoint

    os.makedirs(output_dir, exist_ok=True)
    config_dict = {
        "smoothness": config.smoothness,
        "intrinsic_dim": manifold.intrinsic_dim,
        "ambient_dim": manifold.ambient_dim,
        "n_centroids": manifold.n_centroids,
    }
    _save_spline_checkpoint(output_dir, manifold, config_dict, mean, std, eps=1e-6)


@torch.no_grad()  # type: ignore[reportUntypedFunctionDecorator]
def _eval_reconstruction(
    manifold: SplineManifold,
    features: Tensor,
    mean: Tensor,
    std: Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    """
    Evaluate reconstruction error on features.

    Args:
        manifold: SplineManifold to evaluate
        features: (n, k) feature tensor
        mean: (k,) standardization mean
        std: (k,) standardization std
        device: Device for computation
        batch_size: Batch size for evaluation

    Returns:
        Dict with reconstruction metrics
    """
    manifold.eval()
    total_recon_mse = 0.0
    total_residual = 0.0
    count = 0

    n = features.shape[0]
    for i in range(0, n, batch_size):
        x = features[i : i + batch_size].to(device)
        x_std = (x - mean) / std

        # Encode then decode
        u, residual = manifold.encode(x_std)
        x_recon = manifold.decode(u)

        # Reconstruction MSE: mean over ambient dims, then mean over batch
        recon_mse = (x_std - x_recon) ** 2
        recon_mse = recon_mse.mean(dim=1).mean()
        residual_mean = residual.mean()

        total_recon_mse += recon_mse.item() * x.shape[0]
        total_residual += residual_mean.item() * x.shape[0]
        count += x.shape[0]

    return {
        "recon_mse": total_recon_mse / max(count, 1),
        "residual": total_residual / max(count, 1),
    }


def _subsample_control_points(
    control_points: Tensor,
    centroids: Tensor,
    max_n: int,
) -> tuple[Tensor, Tensor]:
    """Subsample control points to at most *max_n*, preserving first and last.

    For 1-D control points (single column): sort by parameter value and select
    *max_n* evenly-spaced indices.  For 2-D+: farthest-point sampling in
    parameter space.
    """
    n = control_points.shape[0]
    if n <= max_n:
        return control_points, centroids

    if control_points.shape[1] == 1:
        # 1-D: sort, pick evenly-spaced indices (always include first & last)
        order = control_points[:, 0].argsort()
        control_points = control_points[order]
        centroids = centroids[order]
        indices = torch.linspace(0, n - 1, max_n).round().long()
        return control_points[indices], centroids[indices]

    # Multi-dim: farthest-point sampling in parameter space
    selected = [0]
    min_dists = torch.full((n,), float("inf"))
    for _ in range(max_n - 1):
        last = control_points[selected[-1]]
        dists = ((control_points - last) ** 2).sum(dim=1)
        min_dists = torch.minimum(min_dists, dists)
        # Exclude already selected
        min_dists[selected[-1]] = -1.0
        selected.append(int(min_dists.argmax().item()))
    idx = torch.tensor(sorted(selected))
    return control_points[idx], centroids[idx]


def _chord_length_parameterize(
    control_points: Tensor,
    centroids: Tensor,
) -> tuple[Tensor, Tensor]:
    """Replace 1-D control-point values with cumulative chord lengths.

    Sorts by the original parameter value, then computes cumulative Euclidean
    distances between consecutive centroids in ambient (feature) space.
    """
    order = control_points[:, 0].argsort()
    control_points = control_points[order]
    centroids = centroids[order]

    diffs = centroids[1:] - centroids[:-1]
    seg_lengths = diffs.norm(dim=1)
    cum = torch.zeros(centroids.shape[0])
    cum[1:] = seg_lengths.cumsum(dim=0)

    return cum.unsqueeze(1), centroids


def train_spline_manifold(
    interchange_target: InterchangeTarget,
    dataset_path: str | list[CounterfactualExample],
    pipeline: Pipeline,
    intrinsic_dim: int,
    output_dir: str,
    config: SplineManifoldConfig | None = None,
    causal_model: CausalModel | None = None,
    verbose: bool = True,
    features: Tensor | None = None,
    embeddings: dict[str, Callable[[Any], list[float]]] | None = None,
    intervention_variable: str | None = None,
    periodic_info: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Train a SplineManifold by fitting through parameter centroids.

    The target must have a single unit with a trained SubspaceFeaturizer (DAS/PCA rotation).
    Features are collected from the dataset, then grouped by parameter values to compute
    centroids. A Thin-Plate Spline is fitted through these centroids to create a smooth
    manifold interpolation.

    Args:
        interchange_target: Single target with trained DAS/PCA featurizer.
            Must have exactly one group with one unit.
        dataset_path: Path to dataset JSON file or list of CounterfactualExample objects.
        pipeline: Model pipeline for processing inputs.
        intrinsic_dim: Dimensionality d of manifold (d < k where k is DAS/PCA dimension).
        output_dir: Directory for outputs.
        config: Spline fitting configuration. Uses defaults if None.
        causal_model: CausalModel for extracting parameters. Required for spline method.
        verbose: Whether to show progress.

    Returns:
        {
            "manifold": SplineManifold,     # Trained spline manifold
            "mean": Tensor,                 # (k,) standardization mean
            "std": Tensor,                  # (k,) standardization std
            "checkpoint_path": str,         # Path to ckpt_final.pt
            "features": Tensor,             # (n, k) collected features
            "metadata": {
                "ambient_dim": int,         # k (DAS/PCA features)
                "intrinsic_dim": int,       # d (manifold dim)
                "n_samples": int,           # Total samples
                "n_centroids": int,         # Number of centroids
                "recon_mse": float,         # Reconstruction error
                "residual": float,          # Mean residual distance
            },
        }

    Raises:
        ValueError: If causal_model is None (required for parameter extraction)
    """
    if config is None:
        config = SplineManifoldConfig()

    if causal_model is None:
        raise ValueError(
            "causal_model is required for spline method (to extract parameters)"
        )

    os.makedirs(output_dir, exist_ok=True)

    # Validate target
    unit = _validate_target(interchange_target)
    k = unit.featurizer.n_features
    if k is None:
        raise ValueError(
            "Featurizer n_features is None; cannot determine subspace dimension"
        )

    if intrinsic_dim >= k:
        raise ValueError(f"intrinsic_dim ({intrinsic_dim}) must be < n_features ({k})")

    logger.info(
        f"Training spline manifold: ambient_dim={k}, intrinsic_dim={intrinsic_dim}"
    )

    # Load dataset
    if isinstance(dataset_path, list):
        dataset: list[CounterfactualExample] = dataset_path
    else:
        with open(dataset_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "input" in data:
            dataset = [
                {"input": inp, "counterfactual_inputs": cf}
                for inp, cf in zip(data["input"], data["counterfactual_inputs"])
            ]
        else:
            dataset = cast(list[CounterfactualExample], data)

    logger.info(f"Loaded {len(dataset)} examples from dataset")

    # Collect k-dim features
    if features is None:
        if verbose:
            logger.info("Collecting features...")

        features_dict = collect_features(
            dataset=dataset,
            pipeline=pipeline,
            model_units=[unit],
            batch_size=config.batch_size,
        )
        features = features_dict[unit.id].detach().float()
        logger.info(f"Collected features: {features.shape}")
    else:
        if verbose:
            logger.info(f"Using pre-collected features: {features.shape}")
        features = features.detach().float()

    # Compute standardization on full dataset
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp(min=1e-6)

    # Extract parameters from dataset
    if verbose:
        logger.info("Extracting parameters from dataset...")

    excluded_vars = None
    if intervention_variable is not None and causal_model is not None:
        excluded_vars = {
            v for v in causal_model.variables if v != intervention_variable
        }

    param_tensors = extract_parameters_from_dataset(
        dataset,
        excluded_vars=excluded_vars,
        embeddings=embeddings,
        causal_model=causal_model,
    )
    logger.info(f"Extracted parameters: {list(param_tensors.keys())}")

    # Validate intrinsic_dim matches number of parameters
    if len(param_tensors) != intrinsic_dim:
        raise ValueError(
            f"intrinsic_dim={intrinsic_dim} but got {len(param_tensors)}"
            f" parameters. For spline method, intrinsic_dim must match number of causal parameters."
        )

    # Standardize features for spline fitting
    features_std = (features - mean) / (std + 1e-6)
    coord_features = features_std if config.standardize_coords else features

    # Fit Thin-Plate Spline through centroids
    if verbose:
        logger.info(
            "Fitting Thin-Plate Spline through centroids (standardize_coords=%s)...",
            config.standardize_coords,
        )

    from causalab.neural.pipeline import resolve_device

    device = torch.device(resolve_device(config.device))

    if config.intrinsic_mode == "pca":
        # ── PCA intrinsic coords ──
        # Features are already in PCA space from the subspace analysis.
        # Use the first few PC dimensions directly as control point coordinates.
        coord_features = features_std if config.standardize_coords else features
        logger.info(
            "Using PCA intrinsic coords (standardize_coords=%s)",
            config.standardize_coords,
        )

        # Compute centroids in PCA space — first columns are top PCs
        _, centroids, centroid_metadata = compute_centroids(
            coord_features,
            param_tensors,
        )

        # Use first 2*intrinsic_dim columns for periodicity detection
        n_check = min(2 * intrinsic_dim, centroids.shape[1])
        control_points = centroids[:, :n_check].clone()

        # Eigenvalues = variance of each PC across centroids (for periodicity detection)
        eigenvalues = control_points.var(dim=0)

        # Filter centroids with too few samples
        counts = torch.tensor(centroid_metadata["counts"])
        mask = counts >= config.min_samples_per_centroid
        n_filtered = int((~mask).sum().item())
        surviving_centroid_indices = torch.where(mask)[0].tolist()
        control_points = control_points[mask]
        centroids = centroids[mask]
        centroid_metadata["n_filtered"] = n_filtered

        logger.info(
            f"PCA centroids: {centroids.shape[0]} groups, checking {n_check} dims "
            f"(variances: {eigenvalues[:4].tolist()})"
        )

        # Subsample control points
        if isinstance(config.max_control_points, int):
            control_points, centroids = _subsample_control_points(
                control_points, centroids, config.max_control_points
            )
            logger.info(
                f"Subsampled to {control_points.shape[0]} control points (max={config.max_control_points})"
            )

        # Detect periodic dimensions from PCA eigenvalues
        periodic_pairs = detect_periodic_dims(control_points, eigenvalues)

        if periodic_pairs:
            logger.info("Detected periodic pairs: %s", periodic_pairs)
            new_points, periodic_dim_indices, periods = remap_periodic_to_angle(
                control_points,
                periodic_pairs,
                eigenvalues,
            )

            # Select intrinsic_dim columns: periodic dims first, then top non-periodic
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
                "No periodic pairs detected — using top %d PCA components",
                intrinsic_dim,
            )
            control_points = control_points[:, :intrinsic_dim]
            periodic_dims_list = None
            periods_list = None

        manifold = build_spline_manifold(
            control_points=control_points,
            centroids=centroids,
            intrinsic_dim=intrinsic_dim,
            ambient_dim=k,
            smoothness=config.smoothness,
            device=device,
            periodic_dims=periodic_dims_list,
            periods=periods_list,
            spline_method=config.spline_method,
        )

    else:
        # ── Default parameter intrinsic coords ──
        control_points, centroids, centroid_metadata = compute_centroids(
            coord_features,
            param_tensors,
        )

        # Filter centroids with too few samples
        counts = torch.tensor(centroid_metadata["counts"])
        mask = counts >= config.min_samples_per_centroid
        n_filtered = int((~mask).sum().item())
        surviving_centroid_indices = torch.where(mask)[0].tolist()
        control_points = control_points[mask]
        centroids = centroids[mask]
        centroid_metadata["n_filtered"] = n_filtered

        # Subsample control points
        if isinstance(config.max_control_points, int):
            control_points, centroids = _subsample_control_points(
                control_points, centroids, config.max_control_points
            )
            logger.info(
                f"Subsampled to {control_points.shape[0]} control points (max={config.max_control_points})"
            )

        # Chord-length parameterization (1D only)
        if config.chord_length and intrinsic_dim == 1:
            control_points, centroids = _chord_length_parameterize(
                control_points, centroids
            )
            logger.info("Applied chord-length parameterization")

        # Determine periodic dims/periods from periodic_info + param names
        periodic_dims_list = None
        periods_list = None
        if periodic_info:
            param_names = centroid_metadata["parameter_names"]
            periodic_dims_list = []
            periods_list = []
            for i, pname in enumerate(param_names):
                if pname in periodic_info:
                    periodic_dims_list.append(i)
                    periods_list.append(float(periodic_info[pname]))

        manifold = build_spline_manifold(
            control_points=control_points,
            centroids=centroids,
            intrinsic_dim=intrinsic_dim,
            ambient_dim=k,
            smoothness=config.smoothness,
            device=device,
            periodic_dims=periodic_dims_list,
            periods=periods_list,
            spline_method=config.spline_method,
        )

    logger.info(
        f"Fitted spline with {centroid_metadata['n_centroids']} centroids ("
        f"{n_filtered} filtered due to min_samples)"
    )

    # Evaluate reconstruction
    if verbose:
        logger.info("Evaluating reconstruction...")

    if config.standardize_coords:
        eval_mean = mean.to(device)
        eval_std = std.to(device)
    else:
        eval_mean = torch.zeros_like(mean, device=device)
        eval_std = torch.ones_like(std, device=device)
    eval_metrics = _eval_reconstruction(
        manifold, features, eval_mean, eval_std, device, config.batch_size
    )

    logger.info(
        f"Reconstruction metrics: recon_mse={eval_metrics['recon_mse']:.4f}"
        f", residual={eval_metrics['residual']:.4f}"
    )

    # Save checkpoint (writes <output_dir>/ckpt_final.{safetensors,meta.json}).
    _save_checkpoint(output_dir, manifold, mean, std, config)
    checkpoint_path = os.path.join(output_dir, "ckpt_final.safetensors")

    # Save metadata
    metadata = {
        "manifold_method": "spline",
        "ambient_dim": k,
        "intrinsic_dim": intrinsic_dim,
        "n_samples": len(dataset),
        "n_centroids": centroid_metadata["n_centroids"],
        "n_filtered": centroid_metadata["n_filtered"],
        "min_samples_per_centroid": min(centroid_metadata["counts"]),
        "max_samples_per_centroid": max(centroid_metadata["counts"]),
        "mean_samples_per_centroid": sum(centroid_metadata["counts"])
        / len(centroid_metadata["counts"]),
        "parameter_names": centroid_metadata["parameter_names"],
        "recon_mse": eval_metrics["recon_mse"],
        "residual": eval_metrics["residual"],
        "max_control_points": config.max_control_points,
        "chord_length": config.chord_length,
        "config": config.to_dict(),
        "unit_id": unit.id,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Write metrics
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    with open(metrics_path, "w") as f:
        record = {
            "step": 0,
            "recon_mse": eval_metrics["recon_mse"],
            "residual": eval_metrics["residual"],
        }
        f.write(json.dumps(record) + "\n")

    logger.info(f"Training complete. Checkpoint saved to {checkpoint_path}")

    return {
        "manifold": manifold,
        "mean": mean.cpu(),
        "std": std.cpu(),
        "checkpoint_path": checkpoint_path,
        "features": features,
        "surviving_centroid_indices": surviving_centroid_indices,
        "metadata": {
            "ambient_dim": k,
            "intrinsic_dim": intrinsic_dim,
            "n_samples": len(dataset),
            "n_centroids": centroid_metadata["n_centroids"],
            "recon_mse": eval_metrics["recon_mse"],
            "residual": eval_metrics["residual"],
        },
    }
