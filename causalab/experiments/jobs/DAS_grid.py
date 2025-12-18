"""
Train Distributed Alignment Search (DAS) on any component type.

This module provides a unified function to train DAS on attention heads,
residual stream positions, or MLPs. The component type is automatically
detected from the provided InterchangeTarget dict.

Key Features:
1. Accepts pre-built Dict[tuple, InterchangeTarget] from any builder function
2. Auto-detects component type from unit IDs
3. Trains DAS models on each target
4. Generates appropriate score heatmap visualization
5. Saves trained models and evaluation results

Output Structure:
================
output_dir/
├── metadata.json               # Experiment configuration and summary
├── models/                     # Trained DAS models
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
    ├── train.png
    └── test.png

Usage Example:
==============
```python
from causalab.experiments.interchange_targets import (
    build_attention_head_targets,
    build_residual_stream_targets,
    build_mlp_targets,
)
from causalab.experiments.scripts.train_DAS_grid import train_DAS

# Attention heads
targets = build_attention_head_targets(
    pipeline, layers, heads, token_position, mode="one_target_per_unit"
)
result = train_DAS(
    causal_model=causal_model,
    interchange_targets=targets,
    train_dataset_path=train_path,
    test_dataset_path=test_path,
    pipeline=pipeline,
    target_variable_group=("answer",),
    output_dir="outputs/attention_das",
    metric=metric,
)

# Residual stream
targets = build_residual_stream_targets(
    pipeline, layers, token_positions, mode="one_target_per_unit"
)
result = train_DAS(
    causal_model=causal_model,
    interchange_targets=targets,
    ...
)

# MLPs
targets = build_mlp_targets(
    pipeline, layers, token_positions, mode="one_target_per_unit"
)
result = train_DAS(
    causal_model=causal_model,
    interchange_targets=targets,
    ...
)
```
"""

import logging
import os
import copy
from typing import Dict, Any, Callable, Optional, Tuple

import numpy as np

from causalab.neural.pipeline import LMPipeline
from causalab.neural.model_units import InterchangeTarget
from causalab.causal.causal_model import CausalModel
from causalab.experiments.train import train_interventions
from causalab.experiments.configs.train_config import DEFAULT_CONFIG
from causalab.experiments.visualizations import (
    detect_component_type_from_targets,
    extract_grid_dimensions_from_targets,
    plot_score_heatmap,
)
from causalab.experiments.io import save_experiment_metadata

logger = logging.getLogger(__name__)


def train_DAS(
    causal_model: CausalModel,
    interchange_targets: Dict[tuple[Any, ...], InterchangeTarget],
    train_dataset_path: str,
    test_dataset_path: str,
    pipeline: LMPipeline,
    target_variable_group: Tuple[str, ...],
    output_dir: str,
    metric: Callable[[Any, Any], bool],
    config: Optional[Dict[str, Any]] = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train DAS models on any component type.

    This function accepts a pre-built Dict[tuple, InterchangeTarget] and automatically:
    1. Detects the component type (attention_head, residual_stream, or mlp)
    2. Trains DAS on each target
    3. Generates the appropriate score heatmap
    4. Returns scores and metadata

    Args:
        causal_model: Causal model for generating expected outputs
        interchange_targets: Pre-built Dict[tuple, InterchangeTarget] from any builder.
                            Should use mode="one_target_per_unit".
        train_dataset_path: Path to filtered training dataset directory
        test_dataset_path: Path to filtered test dataset directory
        pipeline: LMPipeline object with loaded model
        target_variable_group: Tuple of target variable names (e.g., ("answer",) or ("answer", "position"))
        output_dir: Output directory for results and models
        metric: Function to compare neural output with expected output
        config: Training configuration dict. Should contain train_batch_size.
                (default: DEFAULT_CONFIG with DAS settings)
        save_results: Whether to save metadata and results to disk (default: True)
        verbose: Whether to print progress information

    Returns:
        Dictionary containing:
            - train_scores: training scores per target key
            - test_scores: test scores per target key
            - component_type: detected component type
            - metadata: experiment configuration and summary
            - output_paths: paths to saved files and directories

    Raises:
        FileNotFoundError: If dataset paths do not exist
        ValueError: If component type cannot be detected
    """
    # Configure logging level based on verbose flag
    if verbose:
        logging.getLogger("causalab").setLevel(logging.DEBUG)

    # Detect component type
    component_type = detect_component_type_from_targets(interchange_targets)

    # Extract grid dimensions for visualization
    grid_dims = extract_grid_dimensions_from_targets(
        component_type, interchange_targets
    )

    # Setup configuration
    if config is None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update(
            {
                "intervention_type": "interchange",
                "train_batch_size": 32,
                "id": f"{component_type}_DAS",
                "evaluation_batch_size": 64,
                "training_epoch": 4,
                "init_lr": 0.001,
                "log_dir": os.path.join(output_dir, "logs"),
            }
        )
        config["DAS"] = {"n_features": 32}
    else:
        # Ensure intervention_type is set for DAS
        config["intervention_type"] = "interchange"

    os.makedirs(config.get("log_dir", os.path.join(output_dir, "logs")), exist_ok=True)

    # Train using train_interventions
    result = train_interventions(
        causal_model=causal_model,
        interchange_targets=interchange_targets,
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        pipeline=pipeline,
        target_variable_group=target_variable_group,
        output_dir=output_dir,
        metric=metric,
        config=config,
        save_results=save_results,
    )

    # Extract scores from result
    train_scores = {
        key: res["train_score"] for key, res in result["results_by_key"].items()
    }
    test_scores = {
        key: res["test_score"] for key, res in result["results_by_key"].items()
    }

    # Enhance metadata
    metadata = result["metadata"]
    metadata["component_type"] = component_type

    # Add score statistics
    train_best_loc = max(train_scores.items(), key=lambda x: x[1])[0]
    test_best_loc = max(test_scores.items(), key=lambda x: x[1])[0]

    if component_type == "attention_head":
        train_best_str = f"Layer {train_best_loc[0]}, Head {train_best_loc[1]}"
        test_best_str = f"Layer {test_best_loc[0]}, Head {test_best_loc[1]}"
        metadata["layers"] = grid_dims["layers"]
        metadata["heads"] = grid_dims["heads"]
    else:
        train_best_str = f"Layer {train_best_loc[0]}, {train_best_loc[1]}"
        test_best_str = f"Layer {test_best_loc[0]}, {test_best_loc[1]}"
        metadata["layers"] = grid_dims["layers"]
        metadata["token_position_ids"] = grid_dims["token_position_ids"]

    metadata.update(
        {
            "train_avg_score": float(np.mean(list(train_scores.values()))),
            "train_max_score": float(max(train_scores.values())),
            "train_best_location": train_best_str,
            "test_avg_score": float(np.mean(list(test_scores.values()))),
            "test_max_score": float(max(test_scores.values())),
            "test_best_location": test_best_str,
        }
    )

    # Generate visualizations
    output_paths = result.get("output_paths", {})

    if save_results:
        heatmap_dir = os.path.join(output_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)

        # Generate display name for title
        var_name = "_".join(target_variable_group)

        # Training heatmap
        train_heatmap_path = os.path.join(heatmap_dir, "train.png")
        plot_score_heatmap(
            scores=train_scores,
            interchange_targets=interchange_targets,
            title=f"DAS Training: {var_name.replace('_', ' ').title()}",
            save_path=train_heatmap_path,
        )

        # Test heatmap
        test_heatmap_path = os.path.join(heatmap_dir, "test.png")
        plot_score_heatmap(
            scores=test_scores,
            interchange_targets=interchange_targets,
            title=f"DAS Test: {var_name.replace('_', ' ').title()}",
            save_path=test_heatmap_path,
        )

        output_paths["heatmap_dir"] = heatmap_dir

        # Save enhanced metadata
        save_experiment_metadata(metadata, output_dir)

    return {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "component_type": component_type,
        "metadata": metadata,
        "output_paths": output_paths,
    }
