"""
Collect Features and Compute PCA - Feature extraction and dimensionality reduction.

This module provides functionality to collect neural network activations from specified
model units and perform PCA analysis on them. It follows the standard job pattern for
saving results and generating visualizations.

Key Features:
1. Accepts pre-built Dict[tuple, InterchangeTarget] specifying where to collect features
2. Merges all targets for efficient single-pass feature collection
3. Supports two combine modes: "concatenate" or "stack" features within each target
4. Generates 2D and 3D scatter plots for specified component tuples
5. Supports dual-label encoding with color and shape
6. Saves features, SVD results, and visualizations

Output Structure:
================
output_dir/
├── metadata.json                   # Experiment configuration
├── features/                       # Collected feature tensors
│   └── {target_key}.safetensors
├── svd/                           # SVD/PCA results
│   └── {target_key}.json
└── plots/                         # PCA scatter plots
    └── {target_key}/
        ├── pc0_vs_pc1.png
        ├── pc0_vs_pc1_vs_pc2.png  # 3D plots
        └── ...

Usage Example:
==============
```python
from causalab.methods.targets import build_residual_stream_targets
from causalab.methods.collect_PCA import collect_and_compute_PCA

# Build targets
targets = build_residual_stream_targets(
    pipeline, layers=[0, 1], token_positions=[token_pos], mode="one_target_per_unit"
)

# Collect features and compute PCA (concatenate mode - default)
result = collect_and_compute_PCA(
    interchange_targets=targets,
    data=[{"input": x} for x in inputs],
    pipeline=pipeline,
    labels=labels,
    component_tuples=[(0, 1), (1, 2), (0, 1, 2)],  # 2D and 3D plots
    n_components=10,
    output_dir="outputs/pca",
)

# Stack mode - each unit becomes separate points, useful for comparing units
result = collect_and_compute_PCA(
    interchange_targets=targets,
    data=[{"input": x} for x in inputs],
    pipeline=pipeline,
    labels=labels,
    component_tuples=[(0, 1), (0, 1, 2)],
    combine_mode="stack",  # Each unit's features stacked as separate samples
    output_dir="outputs/pca_stacked",
)
# To color by unit, create labels from targets (auto-replicated by plot_pca_scatter):
# unit_labels = [f"Layer {u.layer}" for u in targets[key].flatten()]

# Dual-label encoding: color by one variable, shape by another
result = collect_and_compute_PCA(
    interchange_targets=targets,
    data=[{"input": x} for x in inputs],
    pipeline=pipeline,
    labels=answer_positions,       # Color by position
    shape_labels=answer_letters,   # Shape by letter
    component_tuples=[(0, 1), (0, 1, 2)],
    output_dir="outputs/pca_dual",
)
```
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.neural.units import InterchangeTarget
from causalab.neural.pipeline import Pipeline
from causalab.neural.activations.collect import collect_features

logger = logging.getLogger(__name__)


def compute_svd(
    features_dict: dict[str, Tensor],
    n_components: int | None = None,
    preprocess: Literal["none", "center", "normalize"] = "none",
    *,
    normalize: bool | None = None,
) -> dict[str, dict[str, Any]]:
    """Perform SVD/PCA analysis on collected features.

    Uses numpy's full SVD (via LAPACK) which is numerically stable across
    all platforms, unlike sklearn's randomized SVD which triggers overflow
    warnings on Apple Silicon with Accelerate BLAS.

    Returns per-unit dict with keys: components, explained_variance_ratio,
    rotation (shape (n_features, n_components)), mean, n_components.
    """
    if normalize is not None:
        preprocess = "normalize" if normalize else "none"

    svd_results = {}
    for unit_id, features in features_dict.items():
        features = features.float()
        n_samples, n_features = features.shape
        max_components = min(n_samples, n_features) - 1
        n = (
            min(max_components, n_components)
            if n_components is not None
            else max_components
        )

        mean = None
        if preprocess in ("center", "normalize"):
            mean = features.mean(dim=0, keepdim=True)
            features = features - mean
            if preprocess == "normalize":
                std_vals = features.var(dim=0) ** 0.5
                std_vals = torch.clamp(std_vals, min=1e-6)
                features = features / std_vals

        X = features.numpy().astype(np.float64)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        components = Vt[:n]
        explained_variance = (S[:n] ** 2) / (n_samples - 1)
        total_variance = np.sum(S**2) / (n_samples - 1)
        explained_variance_ratio = explained_variance / total_variance

        rotation = torch.tensor(components, dtype=features.dtype)

        svd_results[unit_id] = {
            "components": components,
            "explained_variance_ratio": explained_variance_ratio,
            "rotation": rotation.T,
            "mean": mean,
            "n_components": n,
        }
        variance_str = [round(float(x), 4) for x in explained_variance_ratio]
        logger.debug(f"{unit_id}: explained variance = {variance_str}")

    return svd_results


def _key_to_str(key: Tuple[Any, ...]) -> str:
    """Convert a tuple key to a string representation for file naming."""
    if len(key) == 2:
        return f"{key[0]}__{key[1]}"
    elif len(key) == 1:
        return str(key[0])
    return "__".join(str(k) for k in key)


def collect_and_compute_PCA(
    interchange_targets: Dict[Tuple[Any, ...], InterchangeTarget],
    data: list[CounterfactualExample],
    pipeline: Pipeline,
    labels: Sequence[Any],
    component_tuples: List[Tuple[int, ...]],
    shape_labels: Optional[Sequence[Any]] = None,
    combine_mode: str = "concatenate",
    n_components: Optional[int] = None,
    normalize: bool = True,
    batch_size: int = 32,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Collect features from model units and compute PCA for visualization.

    This function efficiently processes all InterchangeTargets by:
    1. Merging all units from all targets for a single collection pass
    2. Splitting features back by original target
    3. Combining features within each target (concatenate or stack)
    4. Computing SVD/PCA on the combined features
    5. Generating scatter plots for specified principal component tuples

    Args:
        interchange_targets: Pre-built Dict[tuple, InterchangeTarget] from any builder.
                            Each target's units will have their features collected.
        data: List of CounterfactualExample objects.
        pipeline: Pipeline object with loaded model for feature extraction
        labels: Sequence of labels (one per sample) for coloring scatter plots.
                With stack mode, labels are auto-replicated to match stacked features.
        component_tuples: List of component tuples to plot. 2-tuples like (0, 1)
                         create 2D plots, 3-tuples like (0, 1, 2) create 3D plots.
        shape_labels: Optional second label sequence for marker shapes. Same
                     auto-replication rules as labels. When provided, labels control
                     color and shape_labels control marker shape.
        combine_mode: How to combine features from multiple units within a target.
                     - "concatenate": (n_samples, n_units * hidden_size) - joint feature space
                     - "stack": (n_samples * n_units, hidden_size) - shared feature space
        n_components: Number of principal components to compute. If None, computes
                     max(max(ct) for ct in component_tuples) + 1.
        normalize: If True (default), center features before SVD (vanilla PCA).
                  If False, compute pure SVD without centering.
        batch_size: Batch size for feature collection (default: 32).
        verbose: Whether to print progress information and enable debug logging.

    Returns:
        Dictionary containing:
            - features_by_target: Dict mapping target keys to combined feature tensors
            - svd_results_by_target: Dict mapping target keys to SVD result dicts (for visualization)
            - svd_results_by_unit: Dict mapping model_unit.id to SVD result dicts (for featurizer init).
                Each SVD result contains "rotation" tensor of shape (hidden_dim, n_components).
            - metadata: Experiment configuration and summary statistics

    Raises:
        ValueError: If combine_mode is not "concatenate" or "stack"
        ValueError: If component_tuples contains tuples with length != 2 or 3

    Example:
        >>> result = collect_and_compute_PCA(
        ...     interchange_targets=targets,
        ...     data=[{"input": x} for x in inputs],
        ...     pipeline=pipeline,
        ...     labels=["A", "B", "A", "B"],
        ...     component_tuples=[(0, 1), (0, 2), (0, 1, 2)],
        ...     combine_mode="stack",
        ... )
        >>> # Color by unit instead of sample label (labels auto-replicated)
        >>> key = list(targets.keys())[0]
        >>> unit_labels = [f"Layer {u.layer}" for u in targets[key].flatten()]
        >>> plot_pca_scatter(
        ...     features=result["features_by_target"][_key_to_str(key)],
        ...     svd_result=result["svd_results_by_target"][_key_to_str(key)],
        ...     labels=unit_labels,
        ...     component_tuples=[(0, 1)],
        ... )
    """
    # Configure logging level based on verbose flag
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Validate arguments
    if combine_mode not in ("concatenate", "stack"):
        raise ValueError(
            f"combine_mode must be 'concatenate' or 'stack', got '{combine_mode}'"
        )

    # Validate component_tuples
    for ct in component_tuples:
        if len(ct) not in (2, 3):
            raise ValueError(
                f"component_tuples must have length 2 or 3, got {len(ct)}: {ct}"
            )

    # Determine n_components if not specified
    if n_components is None:
        n_components = max(max(ct) for ct in component_tuples) + 1
        logger.debug(f"Auto-computed n_components={n_components} from component_tuples")

    n_samples = len(data)

    # =========================================================================
    # Step 1: Merge all units from all targets for efficient single-pass collection
    # =========================================================================
    all_units = []
    unit_to_target_key: Dict[str, Tuple[Any, ...]] = {}

    for key, target in interchange_targets.items():
        for unit in target.flatten():
            all_units.append(unit)
            unit_to_target_key[unit.id] = key

    logger.info(
        f"Collecting features from {len(all_units)} total units across {len(interchange_targets)} targets"
    )

    # Single collection pass for all units
    all_features_dict = collect_features(
        data,
        pipeline,
        all_units,
        batch_size=batch_size,
    )

    # =========================================================================
    # Step 2: Compute per-unit SVD (for featurizer initialization)
    # =========================================================================
    # Convert features to float32 for sklearn compatibility
    features_float32 = {
        unit_id: features.float() for unit_id, features in all_features_dict.items()
    }
    svd_results_by_unit = compute_svd(
        features_float32, n_components=n_components, normalize=normalize
    )
    logger.info(f"Computed per-unit SVD for {len(svd_results_by_unit)} units")

    # =========================================================================
    # Step 3: Split features back by original target
    # =========================================================================
    features_by_target_key: Dict[Tuple[Any, ...], Dict[str, torch.Tensor]] = (
        defaultdict(dict)
    )
    for unit_id, features in all_features_dict.items():
        target_key = unit_to_target_key[unit_id]
        features_by_target_key[target_key][unit_id] = features

    # =========================================================================
    # Step 4: Combine features within each target and compute PCA (for visualization)
    # =========================================================================
    features_by_target: Dict[str, torch.Tensor] = {}
    svd_results_by_target: Dict[str, Dict[str, Any]] = {}

    for key, target in interchange_targets.items():
        key_str = _key_to_str(key)
        model_units = target.flatten()
        unit_features = features_by_target_key[key]

        # Get ordered feature list matching unit order
        feature_list = [unit_features[unit.id] for unit in model_units]

        if combine_mode == "concatenate":
            # (n_samples, n_units * hidden_size) - joint feature space
            combined_features = torch.cat(feature_list, dim=1)
            logger.debug(
                f"{key_str}: Concatenated {len(model_units)} units -> "
                f"shape {combined_features.shape}"
            )
        else:  # stack
            # (n_samples * n_units, hidden_size) - unit-major order
            combined_features = torch.cat(feature_list, dim=0)
            logger.debug(
                f"{key_str}: Stacked {len(model_units)} units -> "
                f"shape {combined_features.shape}"
            )

        # Convert to float32 for sklearn compatibility (BFloat16 not supported)
        combined_features = combined_features.float()

        # Store combined features
        features_by_target[key_str] = combined_features

        # Compute SVD/PCA
        combined_dict = {"combined": combined_features}
        svd_results = compute_svd(
            combined_dict, n_components=n_components, normalize=normalize
        )

        svd_result = svd_results["combined"]
        svd_results_by_target[key_str] = svd_result

        logger.debug(
            f"{key_str}: Computed {svd_result['n_components']} PCs, "
            f"variance explained: {sum(svd_result['explained_variance_ratio']):.4f}"
        )

    # =========================================================================
    # Step 5: Create metadata and save results
    # =========================================================================
    metadata: Dict[str, Any] = {
        "experiment_type": "pca_analysis",
        "num_samples": n_samples,
        "num_targets": len(interchange_targets),
        "num_total_units": len(all_units),
        "combine_mode": combine_mode,
        "n_components": n_components,
        "normalize": normalize,
        "component_tuples": [list(ct) for ct in component_tuples],
        "num_color_labels": len(set(labels)),
        "num_shape_labels": len(set(shape_labels)) if shape_labels else None,
        "batch_size": batch_size,
    }

    return {
        "features_by_target": features_by_target,
        "svd_results_by_target": svd_results_by_target,
        "svd_results_by_unit": svd_results_by_unit,
        "metadata": metadata,
    }


# --------------------------------------------------------------------------- #
# Save helpers                                                                #
# --------------------------------------------------------------------------- #

import os as _os  # noqa: E402
from causalab.io.artifacts import (  # noqa: E402
    save_experiment_metadata as _save_experiment_metadata,
    save_tensor_results as _save_tensor_results,
    save_json_results as _save_json_results,
)
from causalab.io.plots.pca_scatter import plot_pca_scatter as _plot_pca_scatter  # noqa: E402


def save_pca_results(
    result: Dict[str, Any],
    labels: Sequence[Any],
    component_tuples: List[Tuple[int, ...]],
    output_dir: str,
    shape_labels: Optional[Sequence[Any]] = None,
) -> Dict[str, str]:
    """Save collect_and_compute_PCA results: features, SVD, scatter plots."""
    features_by_target = result["features_by_target"]
    svd_results_by_target = result["svd_results_by_target"]
    metadata = result["metadata"]

    _os.makedirs(output_dir, exist_ok=True)
    output_paths: Dict[str, str] = {
        "metadata_path": _save_experiment_metadata(metadata, output_dir),
    }

    features_dir = _os.path.join(output_dir, "features")
    _os.makedirs(features_dir, exist_ok=True)
    for key_str, features in features_by_target.items():
        _save_tensor_results(
            {"features": features}, features_dir, f"{key_str}.safetensors"
        )
    output_paths["features_dir"] = features_dir

    svd_dir = _os.path.join(output_dir, "svd")
    _os.makedirs(svd_dir, exist_ok=True)
    for key_str, svd_result in svd_results_by_target.items():
        svd_json = {
            "n_components": int(svd_result["n_components"]),
            "explained_variance_ratio": [
                float(x) for x in svd_result["explained_variance_ratio"]
            ],
            "components_shape": list(svd_result["components"].shape),
        }
        _save_json_results(svd_json, svd_dir, f"{key_str}.json")
    output_paths["svd_dir"] = svd_dir

    plots_dir = _os.path.join(output_dir, "plots")
    _os.makedirs(plots_dir, exist_ok=True)
    for key_str, svd_result in svd_results_by_target.items():
        target_plot_dir = _os.path.join(plots_dir, key_str)
        _os.makedirs(target_plot_dir, exist_ok=True)
        _plot_pca_scatter(
            features=features_by_target[key_str],
            svd_result=svd_result,
            labels=labels,
            component_tuples=component_tuples,
            shape_labels=shape_labels,
            title=key_str,
            save_dir=target_plot_dir,
        )
    output_paths["plots_dir"] = plots_dir
    return output_paths
