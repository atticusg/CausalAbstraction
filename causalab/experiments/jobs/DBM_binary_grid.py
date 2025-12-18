"""
Train Desiderata-Based Masking (DBM) with binary masks on any component type.

This module provides a unified function to train DBM with tie_masks=True (binary
selection) on attention heads, residual stream positions, or MLPs. The component
type is automatically detected from the provided InterchangeTarget.

Key Features:
1. Accepts pre-built InterchangeTarget from any builder function
2. Auto-detects component type from unit IDs
3. Trains DBM with binary masks (tie_masks=True)
4. Generates appropriate binary mask heatmap visualization
5. Saves trained model and evaluation results

Output Structure:
================
output_dir/
├── metadata.json               # Experiment configuration and summary
├── model/                      # Trained DBM model
│   └── {ComponentType}(...)/
├── results/                    # Evaluation results
│   ├── scores.json             # Train and test scores
│   └── feature_indices.json    # Selected feature indices
└── heatmaps/                   # Visualization images
    └── {var}_mask.png

Usage Example:
==============
```python
from causalab.experiments.interchange_targets import (
    build_attention_head_targets,
    build_residual_stream_targets,
    build_mlp_targets,
)
from causalab.experiments.scripts.DBM_binary_grid import train_DBM_binary_heatmaps

# Attention heads
targets = build_attention_head_targets(
    pipeline, layers, heads, token_position, mode="one_target_all_units"
)
result = train_DBM_binary_heatmaps(
    causal_model=causal_model,
    interchange_target=targets[("all",)],
    train_dataset_path=train_path,
    test_dataset_path=test_path,
    pipeline=pipeline,
    target_variable_group=("answer",),
    output_dir="outputs/attention_dbm",
    metric=metric,
)

# Residual stream
targets = build_residual_stream_targets(
    pipeline, layers, token_positions, mode="one_target_all_units"
)
result = train_DBM_binary_heatmaps(
    causal_model=causal_model,
    interchange_target=targets[("all",)],
    ...
)

# MLPs
targets = build_mlp_targets(
    pipeline, layers, token_positions, mode="one_target_all_units"
)
result = train_DBM_binary_heatmaps(
    causal_model=causal_model,
    interchange_target=targets[("all",)],
    ...
)
```
"""

import logging
import os
from typing import Dict, Any, Callable, Optional, Tuple

from causalab.neural.pipeline import LMPipeline
from causalab.neural.model_units import InterchangeTarget
from causalab.causal.causal_model import CausalModel
from causalab.experiments.train import train_interventions
from causalab.experiments.visualizations.binary_mask import (
    plot_binary_mask,
    get_selected_units,
    extract_grid_dimensions,
)
from causalab.experiments.io import save_experiment_metadata

logger = logging.getLogger(__name__)


def train_DBM_binary_heatmaps(
    causal_model: CausalModel,
    interchange_target: InterchangeTarget,
    train_dataset_path: str,
    test_dataset_path: str,
    pipeline: LMPipeline,
    target_variable_group: Tuple[str, ...],
    output_dir: str,
    metric: Callable[[Any, Any], bool],
    tie_masks: bool = True,
    config: Optional[Dict[str, Any]] = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train DBM (Desiderata-Based Masking) with binary masks on any component type.

    This function accepts a pre-built InterchangeTarget and automatically:
    1. Detects the component type (attention_head, residual_stream, or mlp)
    2. Trains DBM with tie_masks=True for binary selection
    3. Generates the appropriate binary mask heatmap
    4. Returns selected units based on component type

    Args:
        causal_model: Causal model for generating expected outputs
        interchange_target: Pre-built InterchangeTarget from any builder function.
                           Should use mode="one_target_all_units" for DBM.
        train_dataset_path: Path to filtered training dataset directory
        test_dataset_path: Path to filtered test dataset directory
        pipeline: LMPipeline object with loaded model
        target_variable_group: Tuple of target variable names (e.g., ("answer",) or ("answer", "position"))
        output_dir: Output directory for results and models
        metric: Function to compare neural output with expected output
        tie_masks: If True, use one mask per unit (unit selected/deselected as a whole).
                   If False, use separate masks per feature dimension. (default: True)
        config: Training configuration dict. Should contain train_batch_size.
                (default: DEFAULT_CONFIG with masking settings)
        save_results: Whether to save metadata and results to disk (default: True)
        verbose: Whether to print progress information

    Returns:
        Dictionary containing:
            - train_score: training accuracy
            - test_score: test accuracy
            - selected_units: list of selected unit tuples (format depends on component type)
            - feature_indices: raw feature indices dict from InterchangeTarget
            - component_type: detected component type
            - metadata: experiment configuration and summary
            - output_paths: paths to saved files and directories

    Raises:
        FileNotFoundError: If dataset paths do not exist
        ValueError: If component type cannot be detected from the target
    """
    # Configure logging level based on verbose flag
    if verbose:
        logging.getLogger("causalab").setLevel(logging.DEBUG)

    # Detect component type from the target
    units = interchange_target.flatten()
    if not units:
        raise ValueError("InterchangeTarget has no units")

    # Use the first unit ID to detect component type
    sample_unit_id = units[0].id
    if "AttentionHead" in sample_unit_id:
        component_type = "attention_head"
    elif "ResidualStream" in sample_unit_id:
        component_type = "residual_stream"
    elif "MLP" in sample_unit_id:
        component_type = "mlp"
    else:
        raise ValueError(f"Unknown component type in unit_id: {sample_unit_id}")

    logger.debug(f"Detected component type: {component_type}")
    logger.debug(f"Number of units in target: {len(units)}")

    # Setup config with DBM defaults
    if config is None:
        config = {
            "train_batch_size": 32,
            "evaluation_batch_size": 64,
            "training_epoch": 20,
            "init_lr": 0.001,
            "masking": {
                "regularization_coefficient": 0.1,
            },
        }

    # Ensure featurizer_kwargs is set for tie_masks
    config["featurizer_kwargs"] = {"tie_masks": tie_masks}

    # Train DBM using the train_interventions function
    result = train_interventions(
        causal_model=causal_model,
        interchange_targets=interchange_target,  # Single target, will be wrapped
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        pipeline=pipeline,
        target_variable_group=target_variable_group,
        output_dir=output_dir,
        metric=metric,
        config=config,
        save_results=save_results,
    )

    # Extract results for single target
    single_result = result["results_by_key"][("single",)]

    # Get feature indices and selected units
    feature_indices = single_result["feature_indices"]
    selected_units = get_selected_units(feature_indices)

    # Build return structure
    metadata: Dict[str, Any] = result["metadata"]
    return_result: Dict[str, Any] = {
        "train_score": single_result["train_score"],
        "test_score": single_result["test_score"],
        "feature_indices": feature_indices,
        "selected_units": selected_units,
        "component_type": component_type,
        "metadata": metadata,
        "output_paths": {},
    }

    # Extract grid dimensions for metadata
    grid_dims = extract_grid_dimensions(component_type, feature_indices)

    # Update metadata with component-specific info
    return_result["metadata"]["component_type"] = component_type
    return_result["metadata"]["num_units"] = len(units)
    return_result["metadata"]["num_selected_units"] = len(selected_units)
    return_result["metadata"]["selected_units"] = [
        tuple(int(x) if isinstance(x, int) else x for x in unit)
        for unit in selected_units
    ]

    if component_type == "attention_head":
        return_result["metadata"]["layers"] = grid_dims["layers"]
        return_result["metadata"]["heads"] = grid_dims["heads"]
    else:
        return_result["metadata"]["layers"] = grid_dims["layers"]
        return_result["metadata"]["token_position_ids"] = grid_dims[
            "token_position_ids"
        ]

    # Generate visualizations (conditional)
    if save_results:
        heatmap_dir = os.path.join(output_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)

        # Generate display name for title and filename
        var_name = "_".join(target_variable_group)

        # Generate binary mask heatmap using unified dispatcher
        mask_path = os.path.join(heatmap_dir, f"{var_name}_mask.png")
        plot_binary_mask(
            feature_indices=feature_indices,
            title=f"DBM Selected Units: {var_name.replace('_', ' ').title()}",
            save_path=mask_path,
        )

        return_result["output_paths"]["heatmap_dir"] = heatmap_dir
        return_result["output_paths"]["mask_heatmap"] = mask_path

        # Save enhanced metadata
        save_experiment_metadata(
            metadata=return_result["metadata"],
            output_dir=output_dir,
        )

    return return_result
