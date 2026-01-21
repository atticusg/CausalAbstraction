"""
Train DBM with per-feature masks (tie_masks=False) on any component type.

This module provides a unified function to train DBM with per-feature masks on
attention heads, residual stream positions, or MLPs. Unlike DBM with tie_masks=True
which selects/deselects entire units, this learns which feature dimensions within
each unit are important.

This function respects pre-initialized featurizers (e.g., from PCA), allowing mask
learning over frozen feature spaces like principal components.

Key Features:
1. Accepts pre-built InterchangeTarget (single or Dict[int, InterchangeTarget] for per-layer)
2. Auto-detects component type from unit IDs
3. Trains with intervention_type="mask" and tie_masks=False
4. Respects pre-initialized featurizers (doesn't overwrite them)
5. Generates appropriate feature count heatmap visualization
6. Saves trained models and evaluation results

Output Structure:
================
output_dir/
├── metadata.json               # Experiment configuration and summary
├── models/                     # Trained models
│   └── {ComponentType}(...)/
├── training/                   # Training artifacts
│   └── feature_indices.json
├── train_eval/                 # Training set evaluation
│   ├── scores.json
│   └── raw_results.json
├── test_eval/                  # Test set evaluation
│   ├── scores.json
│   └── raw_results.json
└── heatmaps/                   # Visualization images
    └── {var}_features.png

Usage Example:
==============
```python
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.experiments.jobs.DBM_feature_masks_grid import train_DBM_feature_masks
from causalab.neural.featurizers import SubspaceFeaturizer

# Build targets
targets = build_residual_stream_targets(
    pipeline, layers, token_positions, mode="one_target_all_units"
)
target = targets[("all",)]

# Option 1: Use with identity featurizer (masks over raw activations)
result = train_DBM_feature_masks(
    causal_model=causal_model,
    interchange_target=target,
    train_dataset_path=train_path,
    test_dataset_path=test_path,
    pipeline=pipeline,
    target_variable_group=("answer",),
    output_dir="outputs/dbm_feature_masks",
    metric=metric,
)

# Option 2: Pre-initialize with PCA featurizers (masks over PCA components)
for unit in target.flatten():
    rotation = pca_results[unit.id]["rotation"]
    unit.set_featurizer(SubspaceFeaturizer(
        rotation_subspace=rotation,
        trainable=False,  # Keep PCA frozen
        id="PCA",
    ))

result = train_DBM_feature_masks(
    causal_model=causal_model,
    interchange_target=target,
    ...
)
```
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from causalab.causal.causal_model import CausalModel
from causalab.experiments.configs.train_config import (
    ExperimentConfigInput,
    merge_with_defaults,
)
from causalab.experiments.interchange_targets import detect_component_type_from_targets
from causalab.experiments.io import save_experiment_metadata
from causalab.experiments.train import train_interventions
from causalab.experiments.visualizations.feature_masks import plot_feature_counts
from causalab.neural.model_units import InterchangeTarget
from causalab.neural.pipeline import LMPipeline

logger = logging.getLogger(__name__)


def _get_n_features_by_unit(
    interchange_target: Union[InterchangeTarget, Dict[int, InterchangeTarget]],
) -> Dict[str, int]:
    """
    Extract n_features from each unit's featurizer.

    Args:
        interchange_target: Single target or dict of targets.

    Returns:
        Dict mapping unit_id -> n_features

    Raises:
        ValueError: If any unit's featurizer has n_features=None
    """
    n_features_by_unit: Dict[str, int] = {}

    # Get all units
    if isinstance(interchange_target, dict):
        all_units = [u for t in interchange_target.values() for u in t.flatten()]
    else:
        all_units = interchange_target.flatten()

    for unit in all_units:
        if unit.featurizer.n_features is None:
            raise ValueError(
                f"Unit {unit.id} has featurizer with n_features=None. "
                f"All units must have featurizers with n_features set for "
                f"per-feature mask training. Either set n_features on the "
                f"featurizer or use a SubspaceFeaturizer."
            )
        n_features_by_unit[unit.id] = unit.featurizer.n_features

    return n_features_by_unit


def train_DBM_feature_masks(
    causal_model: CausalModel,
    interchange_target: Union[InterchangeTarget, Dict[int, InterchangeTarget]],
    train_dataset_path: str,
    test_dataset_path: str,
    pipeline: LMPipeline,
    target_variable_group: Tuple[str, ...],
    output_dir: str,
    metric: Callable[[Any, Any], bool],
    config: ExperimentConfigInput = None,
    save_results: bool = True,
    verbose: bool = True,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train DBM with per-feature masks (tie_masks=False) on any component type.

    This function accepts either:
    - A single InterchangeTarget (trains one model, returns single scores)
    - A Dict[int, InterchangeTarget] mapping layers to targets (trains per-layer,
      returns per-layer scores)

    The component type (attention_head, residual_stream, or mlp) is auto-detected.

    Pre-initialized featurizers (with id != "null") are preserved, allowing mask
    learning over custom feature spaces like PCA components.

    Args:
        causal_model: Causal model for generating expected outputs
        interchange_target: Either:
            - Single InterchangeTarget (e.g., from mode="one_target_all_units")
            - Dict[int, InterchangeTarget] mapping layer -> target
              (e.g., from mode="one_target_per_layer")
        train_dataset_path: Path to filtered training dataset directory
        test_dataset_path: Path to filtered test dataset directory
        pipeline: LMPipeline object with loaded model
        target_variable_group: Tuple of target variable names (e.g., ("answer",))
        output_dir: Output directory for results and models
        metric: Function to compare neural output with expected output
        config: Training configuration dict. Will be configured for mask intervention.
                (default: DEFAULT_CONFIG with mask settings)
        save_results: Whether to save metadata and results to disk (default: True)
        verbose: Whether to print progress information (default: True)
        title: Custom title for the visualization. If None, a default title is
               generated based on component type and variable name.

    Returns:
        Dictionary containing:
            For single target mode:
                - train_score: float
                - test_score: float
                - feature_indices: Dict[str, Optional[List[int]]]
            For per-layer mode:
                - train_scores: Dict[int, float] (layer -> score)
                - test_scores: Dict[int, float] (layer -> score)
                - feature_indices: Dict[int, Dict[str, Optional[List[int]]]]
                  (layer -> indices)
            Common fields:
                - n_features_by_unit: Dict[str, int] (unit_id -> n_features)
                - component_type: detected component type
                - mode: "single" or "per_layer"
                - metadata: experiment configuration and summary
                - output_paths: paths to saved files and directories

    Raises:
        ValueError: If component type cannot be detected or featurizers lack n_features
    """
    # Configure logging level based on verbose flag
    if verbose:
        logging.getLogger("causalab").setLevel(logging.DEBUG)

    # Determine mode: single target or per-layer
    is_per_layer = isinstance(interchange_target, dict)
    mode = "per_layer" if is_per_layer else "single"

    # Detect component type
    if is_per_layer:
        targets_for_detection = {
            (layer,): target for layer, target in interchange_target.items()
        }
        component_type = detect_component_type_from_targets(targets_for_detection)
    else:
        component_type = detect_component_type_from_targets(
            {("single",): interchange_target}
        )

    # Get n_features for each unit (validates that all featurizers have n_features)
    n_features_by_unit = _get_n_features_by_unit(interchange_target)

    # Setup configuration - merge with defaults
    if config is None:
        config = merge_with_defaults(
            {
                "intervention_type": "mask",
                "train_batch_size": 32,
                "evaluation_batch_size": 64,
                "training_epoch": 20,
                "init_lr": 0.001,
                "id": f"{component_type}_DBM_feature_masks",
                "featurizer_kwargs": {"tie_masks": False},
            }
        )
    else:
        config = merge_with_defaults(config)
        # Ensure mask intervention with per-feature masks
        config["intervention_type"] = "mask"
        config["featurizer_kwargs"] = {"tie_masks": False}

    # Convert input to format expected by train_interventions
    if is_per_layer:
        targets_dict = {
            (layer,): target for layer, target in interchange_target.items()
        }
    else:
        targets_dict = {("single",): interchange_target}

    # Train using train_interventions
    result = train_interventions(
        causal_model=causal_model,
        interchange_targets=targets_dict,
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        pipeline=pipeline,
        target_variable_group=target_variable_group,
        output_dir=output_dir,
        metric=metric,
        config=config,
        save_results=save_results,
    )

    # Build return structure based on mode
    return_result: Dict[str, Any]
    if is_per_layer:
        # Per-layer mode: extract scores and indices per layer
        train_scores: Dict[int, float] = {}
        test_scores: Dict[int, float] = {}
        feature_indices: Dict[int, Dict[str, Optional[List[int]]]] = {}

        for key, res in result["results_by_key"].items():
            layer = key[0]
            train_scores[layer] = res["train_score"]
            test_scores[layer] = res["test_score"]
            feature_indices[layer] = res["feature_indices"]

        return_result = {
            "train_scores": train_scores,
            "test_scores": test_scores,
            "feature_indices": feature_indices,
            "n_features_by_unit": n_features_by_unit,
            "component_type": component_type,
            "mode": mode,
            "metadata": result["metadata"],
            "output_paths": result.get("output_paths", {}),
        }

        # Find best layer
        best_layer = max(test_scores, key=lambda k: test_scores[k])
        return_result["metadata"]["best_layer"] = best_layer
        return_result["metadata"]["best_test_score"] = test_scores[best_layer]
        return_result["metadata"]["avg_test_score"] = float(
            np.mean(list(test_scores.values()))
        )
        return_result["metadata"]["layers"] = sorted(interchange_target.keys())

    else:
        # Single target mode: extract single scores
        single_result = result["results_by_key"][("single",)]

        return_result = {
            "train_score": single_result["train_score"],
            "test_score": single_result["test_score"],
            "feature_indices": single_result["feature_indices"],
            "n_features_by_unit": n_features_by_unit,
            "component_type": component_type,
            "mode": mode,
            "metadata": result["metadata"],
            "output_paths": result.get("output_paths", {}),
        }

    # Update metadata with common fields
    return_result["metadata"]["component_type"] = component_type
    return_result["metadata"]["mode"] = mode
    return_result["metadata"]["tie_masks"] = False
    return_result["metadata"]["n_features_by_unit"] = n_features_by_unit

    # Generate visualizations
    if save_results:
        heatmap_dir = os.path.join(output_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)

        # Generate display name for title and filename
        var_name = "_".join(target_variable_group)
        heatmap_path = os.path.join(heatmap_dir, f"{var_name}_features.png")

        # Create title if not provided
        if title is None:
            component_display_names = {
                "attention_head": "Attention Heads",
                "residual_stream": "Residual Stream",
                "mlp": "MLPs",
            }
            component_display = component_display_names.get(
                component_type, component_type
            )
            title = f"DBM Feature Masks ({component_display}): {var_name.replace('_', ' ').title()}"

        plot_feature_counts(
            feature_indices=return_result["feature_indices"],
            scores=return_result["test_scores"]
            if is_per_layer
            else return_result["test_score"],
            n_features=n_features_by_unit,
            title=title,
            save_path=heatmap_path,
        )

        return_result["output_paths"]["heatmap_dir"] = heatmap_dir
        return_result["output_paths"]["heatmap"] = heatmap_path

        # Save enhanced metadata
        save_experiment_metadata(return_result["metadata"], output_dir)

    return return_result
