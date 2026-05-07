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
from causalab.methods.targets import (
    build_attention_head_targets,
    build_residual_stream_targets,
    build_mlp_targets,
)
from causalab.methods.DAS_grid import train_DAS

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
from typing import Dict, Any, Callable, Tuple

import numpy as np

from causalab.neural.pipeline import LMPipeline
from causalab.neural.units import InterchangeTarget
from causalab.causal.causal_model import CausalModel
from causalab.methods.trained_subspace.train import train_interventions
from causalab.io.counterfactuals import load_counterfactual_examples
from causalab.configs.train_config import (
    ExperimentConfigInput,
    merge_with_defaults,
)
from causalab.io.plots import (
    detect_component_type_from_targets,
    extract_grid_dimensions_from_targets,
)

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
    config: ExperimentConfigInput = None,
    verbose: bool = True,
    source_pipeline: LMPipeline | None = None,
) -> Dict[str, Any]:
    """
    Train DAS models on any component type.

    This function accepts a pre-built Dict[tuple, InterchangeTarget] and automatically:
    1. Detects the component type (attention_head, residual_stream, or mlp)
    2. Trains DAS on each target
    3. Returns scores and metadata

    Args:
        causal_model: Causal model for generating expected outputs
        interchange_targets: Pre-built Dict[tuple, InterchangeTarget] from any builder.
                            Should use mode="one_target_per_unit".
        train_dataset_path: Path to filtered training dataset directory
        test_dataset_path: Path to filtered test dataset directory
        pipeline: Target LMPipeline where interventions are applied
        target_variable_group: Tuple of target variable names (e.g., ("answer",) or ("answer", "position"))
        output_dir: Output directory for results and models
        metric: Function to compare neural output with expected output
        config: Training configuration dict. Should contain train_batch_size.
                (default: DEFAULT_CONFIG with DAS settings)
        verbose: Whether to print progress information
        source_pipeline: If provided, collect activations from this pipeline instead
            of the target pipeline. Enables cross-model patching.

    Returns:
        Dictionary containing:
            - train_scores: training scores per target key
            - test_scores: test scores per target key
            - component_type: detected component type
            - metadata: experiment configuration and summary

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

    # Setup configuration - merge with defaults
    if config is None:
        config = merge_with_defaults(
            {
                "intervention_type": "interchange",
                "train_batch_size": 32,
                "id": f"{component_type}_DAS",
                "evaluation_batch_size": 64,
                "training_epoch": 4,
                "init_lr": 0.001,
                "log_dir": os.path.join(output_dir, "logs"),
                "DAS": {"n_features": 32},
            }
        )
    else:
        config = merge_with_defaults(config)
        # Ensure intervention_type is set for DAS
        config["intervention_type"] = "interchange"

    config["log_dir"] = config.get("log_dir") or os.path.join(output_dir, "logs")
    os.makedirs(config["log_dir"], exist_ok=True)

    # Load datasets at the caller boundary
    train_dataset = load_counterfactual_examples(train_dataset_path, causal_model)
    test_dataset = load_counterfactual_examples(test_dataset_path, causal_model)

    # Train using train_interventions
    result = train_interventions(
        causal_model=causal_model,
        interchange_targets=interchange_targets,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        pipeline=pipeline,
        target_variable_group=target_variable_group,
        metric=metric,
        config=config,
        source_pipeline=source_pipeline,
    )

    # Extract scores from result
    train_scores = {
        key: res["train_score"] for key, res in result["results_by_key"].items()
    }
    test_scores = {
        key: res["test_score"] for key, res in result["results_by_key"].items()
    }

    # Enhance metadata — re-stamp research-question fields stripped by methods/
    metadata = result["metadata"]
    metadata["experiment_type"] = "DAS"
    metadata["model"] = getattr(pipeline, "model_or_name", None)
    metadata["train_dataset_path"] = train_dataset_path
    metadata["test_dataset_path"] = test_dataset_path
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

    return {
        "results_by_key": result["results_by_key"],
        "train_scores": train_scores,
        "test_scores": test_scores,
        "component_type": component_type,
        "metadata": metadata,
    }


# --------------------------------------------------------------------------- #
# Save helpers                                                                #
# --------------------------------------------------------------------------- #

import os as _os  # noqa: E402
from causalab.io.artifacts import save_experiment_metadata as _save_experiment_metadata  # noqa: E402
from causalab.io.plots import plot_score_heatmap as _plot_score_heatmap  # noqa: E402
from causalab.methods.trained_subspace.train import save_train_results  # noqa: E402


def save_DAS_results(
    result: Dict[str, Any],
    interchange_targets: Dict[Tuple[Any, ...], InterchangeTarget],
    target_variable_group: Tuple[str, ...],
    output_dir: str,
) -> Dict[str, str]:
    """Save train_DAS results: train artifacts + train/test heatmaps."""
    output_paths = save_train_results(result, output_dir)

    heatmap_dir = _os.path.join(output_dir, "heatmaps")
    _os.makedirs(heatmap_dir, exist_ok=True)
    var_name = "_".join(target_variable_group)

    _plot_score_heatmap(
        scores=result["train_scores"],
        interchange_targets=interchange_targets,
        title=f"DAS Training: {var_name.replace('_', ' ').title()}",
        save_path=_os.path.join(heatmap_dir, "train.png"),
    )
    _plot_score_heatmap(
        scores=result["test_scores"],
        interchange_targets=interchange_targets,
        title=f"DAS Test: {var_name.replace('_', ' ').title()}",
        save_path=_os.path.join(heatmap_dir, "test.png"),
    )

    output_paths["heatmap_dir"] = heatmap_dir
    _save_experiment_metadata(result["metadata"], output_dir)
    return output_paths


# Backwards-compatible alias
save_das_results = save_DAS_results
