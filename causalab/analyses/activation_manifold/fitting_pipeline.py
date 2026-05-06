"""Manifold Fitting Pipeline.

Fits a manifold in a pre-computed k-dimensional feature space.  Subspace
discovery (PCA / DAS / DBM) lives in ``causalab.analyses.subspace`` and must
be run first.

Pipeline steps:
1. Fit manifold (normalizing flow or thin-plate spline) in k-dim space
2. Compose featurizer chain: subspace >> standardize >> manifold
3. Compute intrinsic ranges + 3D visualization
3b. Decoding reconstruction test (optional, off by default)

Manifold-traversal steering (interpolating between centroids and plotting the
resulting output distributions) is intentionally not run here; it is the
``path_steering`` analysis's concern. This keeps activation_manifold focused
on geometry fitting and lets it run on cached artifacts without loading
model weights.

Output Structure:
================
output_dir/
├── manifold_spline/ or manifold_flow/
│   ├── ckpt_final.pt
│   └── metadata.json
└── visualization/
    └── manifold_3d.html
"""

from __future__ import annotations

import logging
import os

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from causalab.causal.causal_model import CausalModel
from causalab.methods.steer.collect import (
    collect_grid_distributions,
    make_intrinsic_steering_grid,
)
from causalab.methods.spline.train import (
    SplineManifoldConfig,
    train_spline_manifold,
)
from causalab.methods.spline.featurizer import ManifoldFeaturizer
from causalab.methods.standardize import StandardizeFeaturizer
from causalab.neural.units import InterchangeTarget
from causalab.neural.pipeline import LMPipeline

from causalab.analyses.activation_manifold.utils import (
    _compute_intrinsic_ranges,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# New config — manifold fitting only (no subspace fields)
# ─────────────────────────────────────────────────────────────────────


@dataclass
class ManifoldFittingConfig:
    """Configuration for the manifold fitting pipeline (post-subspace)."""

    # Required
    pipeline: LMPipeline = field(repr=False)
    interchange_target: InterchangeTarget = field(repr=False)
    features: Tensor = field(repr=False)
    train_dataset: list | str = field(repr=False)
    causal_model: CausalModel = field(repr=False)
    output_dir: str = ""

    # Manifold
    k_features: int = 3
    intrinsic_dim: int | None = None
    manifold_method: str = "spline"
    smoothness: float = 0.0
    spline_method: str = "auto"
    intervention_variable: str | None = None
    periodic_info: dict[str, int] | None = None
    embeddings: dict[str, Callable] | None = None

    # Parameterization
    intrinsic_mode: str = "parameter"

    @property
    def standardize_coords(self) -> bool:
        """Standardize coords for 'parameter' mode, not for 'pca'."""
        return self.intrinsic_mode != "pca"

    # Reconstruction test (optional)
    n_grid: int = 21
    score_token_ids: list[int] | list[list[int]] | None = None
    batch_size: int = 32
    ref_dists: Tensor | None = None
    n_classes: int | None = None
    comparison_fn: Callable | None = None
    seed: int = 42
    skip_decoding_eval: bool = True
    embedding_shuffle_seed: int | None = None
    colormap: str | None = None
    score_variable_values: dict[str, list] | None = None
    figure_format: str = "pdf"
    max_control_points: str | int = "all"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _auto_detect_intrinsic_dim(causal_model: CausalModel) -> int:
    """Detect intrinsic dimension from the number of causal parameters."""
    excluded = {
        "probs",
        "sequence",
        "raw_input",
        "raw_output",
        "true_probs",
        "observations",
        "context_length",
    }
    n_params = 0
    for var in causal_model.variables:
        if var in excluded:
            continue
        vals = causal_model.values.get(var)
        if vals is None:
            continue
        sample = vals[0]
        if isinstance(sample, (tuple, list)):
            n_params += len(sample)
        else:
            n_params += 1
    return n_params


def _shuffle_embeddings(
    embeddings: dict[str, Callable],
    causal_model: CausalModel,
    seed: int,
) -> dict[str, Callable]:
    """Permute embedding coordinate assignments per variable.

    The centroids (mean activations) are unchanged — only the scalar
    coordinates assigned to each categorical value are permuted.
    """
    rng = np.random.RandomState(seed)
    shuffled: dict[str, Callable] = {}
    for var, embed_fn in embeddings.items():
        values = causal_model.values.get(var)
        if values is None:
            shuffled[var] = embed_fn
            continue
        original_embeds = [embed_fn(v) for v in values]
        perm = rng.permutation(len(values))
        shuffled_map = {v: original_embeds[perm[i]] for i, v in enumerate(values)}
        shuffled[var] = lambda v, m=shuffled_map: m[v]
    return shuffled


# ─────────────────────────────────────────────────────────────────────
# Main pipeline (manifold only — starts from pre-computed features)
# ─────────────────────────────────────────────────────────────────────


def run_manifold_fitting_pipeline(
    config: ManifoldFittingConfig,
) -> dict[str, Any]:
    """Run the manifold fitting pipeline on pre-computed features.

    Steps:
        1. Fit manifold (flow or spline)
        2. Compose featurizer chain
        3. Compute intrinsic ranges + visualization
        3b. Manifold reconstruction test (optional)

    Args:
        config: Pipeline configuration with pre-computed features.

    Returns:
        Dict with manifold, mean, std, ranges, and other step outputs.
    """
    import time as _time

    _t0 = _time.time()

    os.makedirs(config.output_dir, exist_ok=True)

    features = config.features
    k = config.k_features
    manifold_method = config.manifold_method.lower()

    # Auto-detect intrinsic dim if not specified
    if config.intrinsic_dim is not None:
        d = config.intrinsic_dim
    elif manifold_method == "spline":
        if config.intervention_variable is not None:
            vals = config.causal_model.values.get(config.intervention_variable)
            if vals is not None:
                sample = vals[0]
                d = len(sample) if isinstance(sample, (tuple, list)) else 1
            else:
                d = _auto_detect_intrinsic_dim(config.causal_model)
        else:
            d = _auto_detect_intrinsic_dim(config.causal_model)
        logger.info("Auto-detected intrinsic_dim=%d", d)
    else:
        d = 2  # default for flows
        logger.info("Using default intrinsic_dim=%d", d)

    if d >= k:
        new_k = d + 1
        logger.info(
            "Auto-increasing k_features=%d (must be > intrinsic_dim=%d)", new_k, d
        )
        k = new_k

    # Validate target
    units = config.interchange_target.flatten()
    if len(units) != 1:
        raise ValueError(f"Expected single unit in target, got {len(units)}")
    unit = units[0]

    logger.info(
        "Starting manifold fitting pipeline: k=%d, d=%d, method=%s",
        k,
        d,
        manifold_method,
    )

    # Load training dataset if path
    if isinstance(config.train_dataset, str):
        from causalab.io.counterfactuals import load_counterfactual_examples

        train_dataset = load_counterfactual_examples(
            config.train_dataset,
            config.causal_model,
        )
    else:
        train_dataset = config.train_dataset

    result: dict[str, Any] = {"features": features}

    # =====================================================================
    # Step 1: Fit manifold (flow or spline)
    # =====================================================================
    if manifold_method == "spline":
        manifold_dir = os.path.join(config.output_dir, "manifold_spline")
    else:
        manifold_dir = os.path.join(config.output_dir, "manifold_flow")

    logger.info("Step 1: Fitting %s manifold (d=%d in k=%d)...", manifold_method, d, k)

    if manifold_method == "spline":
        from causalab.neural.pipeline import resolve_device

        # Spline fitting runs on the cached features tensor; we don't need the
        # model's device (or the model itself) to be loaded. resolve_device()
        # picks a sensible default whether or not pipeline weights are present.
        device = resolve_device()
        spline_config = SplineManifoldConfig(
            smoothness=config.smoothness,
            batch_size=config.batch_size,
            device=device,
            intrinsic_mode=config.intrinsic_mode,
            max_control_points=config.max_control_points,
            spline_method=config.spline_method,
        )
        embeddings = config.embeddings
        if embeddings is not None and config.embedding_shuffle_seed is not None:
            logger.info(
                "Shuffling embedding order with seed=%d",
                config.embedding_shuffle_seed,
            )
            embeddings = _shuffle_embeddings(
                embeddings,
                config.causal_model,
                config.embedding_shuffle_seed,
            )

        manifold_result = train_spline_manifold(
            interchange_target=config.interchange_target,
            dataset_path=train_dataset,
            pipeline=config.pipeline,
            intrinsic_dim=d,
            output_dir=manifold_dir,
            config=spline_config,
            causal_model=config.causal_model,
            features=features,
            intervention_variable=config.intervention_variable,
            periodic_info=config.periodic_info,
            embeddings=embeddings,
        )
    else:
        from causalab.methods.flow.train import (
            train_manifold,
        )

        manifold_result = train_manifold(
            interchange_target=config.interchange_target,
            dataset_path=train_dataset,
            pipeline=config.pipeline,
            intrinsic_dim=d,
            output_dir=manifold_dir,
            features=features,
        )

    _t1 = _time.time()
    logger.info("Step 1 complete (%.1fs)", _t1 - _t0)

    manifold_obj = manifold_result["manifold"]
    raw_mean = manifold_result["mean"]
    raw_std = manifold_result["std"]

    # When standardize_coords=false, the manifold operates in raw feature space
    # so we use identity standardization throughout the pipeline
    if config.standardize_coords:
        mean, std = raw_mean, raw_std
    else:
        mean = torch.zeros_like(raw_mean)
        std = torch.ones_like(raw_std)

    surviving_centroid_indices = manifold_result.get("surviving_centroid_indices")
    result["manifold"] = manifold_obj
    result["mean"] = mean
    result["std"] = std

    # =====================================================================
    # Step 2: Compose featurizer chain (subspace >> standardize >> manifold)
    # =====================================================================
    logger.info("Step 2: Composing featurizer chain...")

    standardize = StandardizeFeaturizer(mean, std)
    manifold_feat = ManifoldFeaturizer(manifold_obj, n_features=k)
    composed = unit.featurizer >> standardize >> manifold_feat
    unit.set_featurizer(composed)

    logger.info("Composed featurizer: %s", type(unit.featurizer).__name__)

    # Persist the composed featurizer so evaluate can reload it
    models_dir = os.path.join(config.output_dir, "models")
    key_str = f"{unit.layer}__{unit.get_index_id()}"
    save_dir = os.path.join(models_dir, key_str)
    config.interchange_target.save(save_dir)
    logger.info("Step 2 complete (%.1fs)", _time.time() - _t1)

    # =====================================================================
    # Step 3: Compute intrinsic ranges + visualization
    # =====================================================================
    _t3 = _time.time()
    logger.info("Step 3: Computing intrinsic ranges...")

    ranges = _compute_intrinsic_ranges(features, manifold_obj, mean, std)
    result["ranges"] = ranges

    logger.info("Building steering grid...")
    grid = make_intrinsic_steering_grid(
        ranges=ranges,
        n_points_per_dim=config.n_grid,
        intrinsic_dim=d,
    )
    result["steering_grid"] = grid
    logger.info("Steering grid: %d points", grid.shape[0])

    # Manifold visualization (features_3d belongs to subspace analysis)
    vis_dir = os.path.join(config.output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    try:
        from causalab.io.plots.plot_3d_interactive import plot_3d

        plot_3d(
            features=features,
            train_dataset=train_dataset,
            manifold_obj=manifold_obj,
            mean=mean,
            std=std,
            ranges=ranges,
            output_path=os.path.join(vis_dir, "manifold_3d.html"),
            intervention_variable=config.intervention_variable,
            embeddings=config.embeddings,
            colormap=config.colormap,
        )
        logger.info("Saved manifold_3d.html to %s (%.1fs)", vis_dir, _time.time() - _t3)
    except Exception as e:
        logger.warning(
            "3D visualization failed (%.1fs): %s", _time.time() - _t3, e, exc_info=True
        )

    try:
        from causalab.io.plots.pca_scatter import plot_manifold_3d_static

        plot_manifold_3d_static(
            features=features,
            manifold_obj=manifold_obj,
            mean=mean,
            std=std,
            ranges=ranges,
            output_path=os.path.join(vis_dir, "manifold_3d.png"),
            train_dataset=train_dataset,
            intervention_variable=config.intervention_variable,
            embeddings=config.embeddings,
            colormap=config.colormap,
        )
        logger.info("Saved manifold_3d.png to %s", vis_dir)
    except Exception as e:
        logger.warning("Static 3D manifold visualization failed: %s", e, exc_info=True)

    # =====================================================================
    # Step 3b: Decoding reconstruction test (optional)
    #
    # Steers to each centroid and measures how closely the resulting output
    # distribution matches the per-class reference distribution. Off by
    # default — this is the only step that requires loaded model weights.
    # For richer manifold-traversal evaluations (interpolation between
    # centroids, heatmaps, distortion metrics), use the path_steering
    # analysis.
    # =====================================================================
    if (
        not config.skip_decoding_eval
        and config.n_classes is not None
        and config.ref_dists is not None
        and config.score_token_ids is not None
        and hasattr(manifold_obj, "control_points")
    ):
        if config.pipeline is None:
            raise ValueError(
                "skip_decoding_eval=False but no pipeline was provided. "
                "Either pass a full pipeline (with model weights) or set "
                "skip_decoding_eval=True."
            )
        if config.comparison_fn is None:
            raise ValueError(
                "comparison_fn is required for the decoding reconstruction "
                "test. Ensure intervention_metric resolves to a distribution "
                "comparison."
            )

        logger.info("Step 3b: Decoding reconstruction test...")
        ref_dists = config.ref_dists
        recon_cmp = config.comparison_fn

        control_points = manifold_obj.control_points.cpu()
        logger.info(
            "Steering to %d control points (%d intrinsic dims)...",
            control_points.shape[0],
            control_points.shape[1],
        )

        steered_probs_fvs = collect_grid_distributions(
            pipeline=config.pipeline,
            grid_points=control_points,
            interchange_target=config.interchange_target,
            filtered_samples=train_dataset,
            var_indices=config.score_token_ids,
            batch_size=config.batch_size,
            full_vocab_softmax=True,
        )
        # Normalize for KL comparison (concept-only softmax)
        steered_probs = steered_probs_fvs / steered_probs_fvs.sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=1e-10)

        # When centroids are filtered, surviving control points may not correspond
        # to consecutive class indices. Use surviving_centroid_indices to align.
        if (
            surviving_centroid_indices is not None
            and len(surviving_centroid_indices) == steered_probs.shape[0]
        ):
            valid_indices = [
                i for i in surviving_centroid_indices if i < ref_dists.shape[0]
            ]
            ref_dists_aligned = ref_dists[valid_indices]
            steered_aligned = steered_probs[: len(valid_indices)]
        else:
            n_points = min(steered_probs.shape[0], ref_dists.shape[0])
            ref_dists_aligned = ref_dists[:n_points]
            steered_aligned = steered_probs[:n_points]
        scores_per_point = recon_cmp(ref_dists_aligned, steered_aligned)
        n_compared = scores_per_point.shape[0]
        recon_scores = {cls: scores_per_point[cls].item() for cls in range(n_compared)}

        mean_score = scores_per_point.mean().item()
        logger.info(
            "Reconstruction test: mean score = %.4f across %d control points",
            mean_score,
            len(recon_scores),
        )
        result["reconstruction_kl"] = mean_score
        result["reconstruction_score"] = mean_score
        result["reconstruction_kls_per_class"] = recon_scores

    logger.info("Pipeline complete. Results in %s", config.output_dir)
    return result
