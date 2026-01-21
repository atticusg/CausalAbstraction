"""
Train Boundless DAS (mask intervention with tie_masks=False) on any component type.

This module provides a unified function to train Boundless DAS on attention heads,
residual stream positions, or MLPs. Unlike DBM (tie_masks=True) which selects/deselects
entire units, Boundless DAS learns which feature dimensions within each unit are important.

Boundless DAS initializes a trainable SubspaceFeaturizer on each unit, then learns
per-feature masks over the learned subspace. This is a lightweight wrapper around
train_DBM_feature_masks that handles the SubspaceFeaturizer initialization.

Key Features:
1. Accepts pre-built InterchangeTarget (single or Dict[int, InterchangeTarget] for per-layer)
2. Auto-detects component type from unit IDs
3. Initializes SubspaceFeaturizer with trainable=True on each unit
4. Trains with intervention_type="mask" and tie_masks=False
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
from causalab.experiments.interchange_targets import (
    build_attention_head_targets,
    build_residual_stream_targets,
    build_mlp_targets,
)
from causalab.experiments.jobs.boundless_DAS_feature_mask_grid import train_boundless_DAS

# Single target mode (all units at once)
targets = build_attention_head_targets(
    pipeline, layers, heads, token_position, mode="one_target_all_units"
)
config = {"DAS": {"n_features": 32}}  # n_features is read from config
result = train_boundless_DAS(
    causal_model=causal_model,
    interchange_target=targets[("all",)],  # Single InterchangeTarget
    train_dataset_path=train_path,
    test_dataset_path=test_path,
    pipeline=pipeline,
    target_variable_group=("answer",),
    output_dir="outputs/attention_boundless",
    metric=metric,
    config=config,
)

# Per-layer mode
targets = build_residual_stream_targets(
    pipeline, layers, token_positions, mode="one_target_per_layer"
)
# Convert {(layer,): target} to {layer: target}
per_layer_targets = {key[0]: target for key, target in targets.items()}
result = train_boundless_DAS(
    causal_model=causal_model,
    interchange_target=per_layer_targets,  # Dict[int, InterchangeTarget]
    ...
)
```
"""

import copy
import logging
from typing import Any, Callable, Dict, Tuple, Union

from causalab.causal.causal_model import CausalModel
from causalab.experiments.configs.train_config import (
    ExperimentConfigInput,
    merge_with_defaults,
)
from causalab.experiments.interchange_targets import detect_component_type_from_targets
from causalab.experiments.jobs.DBM_feature_masks_grid import train_DBM_feature_masks
from causalab.neural import SubspaceFeaturizer
from causalab.neural.model_units import InterchangeTarget
from causalab.neural.pipeline import LMPipeline

logger = logging.getLogger(__name__)


def train_boundless_DAS(
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
) -> Dict[str, Any]:
    """
    Train Boundless DAS (mask with tie_masks=False) on any component type.

    This function initializes a trainable SubspaceFeaturizer on each unit, then
    delegates to train_DBM_feature_masks for training. The SubspaceFeaturizer
    learns a rotation matrix that projects activations into a lower-dimensional
    space, and masks are learned over this projected space.

    This function accepts either:
    - A single InterchangeTarget (trains one model, returns single scores)
    - A Dict[int, InterchangeTarget] mapping layers to targets (trains per-layer,
      returns per-layer scores)

    The component type (attention_head, residual_stream, or mlp) is auto-detected.

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
        config: Training configuration dict. n_features is read from config["DAS"]["n_features"]
                (default: 32). Will be configured for mask intervention.
                (default: DEFAULT_CONFIG with mask settings)
        save_results: Whether to save metadata and results to disk (default: True)
        verbose: Whether to print progress information (default: True)

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
        ValueError: If component type cannot be detected or if a unit has no shape defined
    """
    # Setup configuration - merge with defaults
    config = merge_with_defaults(config)

    # Deep copy interchange_target to avoid mutating caller's target
    interchange_target = copy.deepcopy(interchange_target)

    # Extract n_features from config (guaranteed present after merge)
    n_features = config["DAS"]["n_features"]

    # Get all units from the target(s)
    if isinstance(interchange_target, dict):
        all_units = [u for t in interchange_target.values() for u in t.flatten()]
    else:
        all_units = interchange_target.flatten()

    # Initialize SubspaceFeaturizer on each unit that doesn't already have one
    for unit in all_units:
        if unit.featurizer.id == "null":
            if unit.shape is None:
                raise ValueError(f"Unit {unit.id} has no shape defined")
            if n_features > unit.shape[0]:
                logger.warning(
                    f"n_features ({n_features}) > unit dimension ({unit.shape[0]}) "
                    f"for {unit.id}. This projects to a higher-dimensional space."
                )
            unit.set_featurizer(
                SubspaceFeaturizer(
                    shape=(unit.shape[0], n_features),
                    trainable=True,
                    id="boundless_DAS",
                )
            )

    # Detect component type for title
    if isinstance(interchange_target, dict):
        targets_for_detection = {
            (layer,): target for layer, target in interchange_target.items()
        }
        component_type = detect_component_type_from_targets(targets_for_detection)
    else:
        component_type = detect_component_type_from_targets(
            {("single",): interchange_target}
        )

    # Build title
    var_name = "_".join(target_variable_group)
    component_display_names = {
        "attention_head": "Attention Heads",
        "residual_stream": "Residual Stream",
        "mlp": "MLPs",
    }
    component_display = component_display_names.get(component_type, component_type)
    title = f"Boundless DAS ({component_display}): {var_name.replace('_', ' ').title()}"

    # Delegate to train_DBM_feature_masks
    result = train_DBM_feature_masks(
        causal_model=causal_model,
        interchange_target=interchange_target,
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        pipeline=pipeline,
        target_variable_group=target_variable_group,
        output_dir=output_dir,
        metric=metric,
        config=config,
        save_results=save_results,
        verbose=verbose,
        title=title,
    )

    # Add boundless-DAS-specific metadata
    result["metadata"]["experiment_type"] = "boundless_DAS"
    result["metadata"]["n_features"] = n_features

    return result
