"""
Interchange Score Heatmap - Generic interchange intervention scoring with visualization.

This module provides a unified function to run interchange interventions and generate
score heatmaps for attention heads, residual stream positions, or MLPs. The component
type is automatically detected from the provided InterchangeTarget dict.

Key Features:
1. Accepts pre-built Dict[tuple, InterchangeTarget] from any builder function
2. Auto-detects component type from unit IDs
3. Runs interchange interventions on each target
4. Computes causal scores using a provided metric
5. Generates appropriate score heatmap visualization
6. Saves evaluation results

Output Structure:
================
output_dir/
├── metadata.json                   # Experiment configuration
├── results/                        # Evaluation results
│   ├── raw_results.json            # Raw intervention string outputs
│   ├── raw_results.safetensors     # Raw intervention tensor outputs
│   └── scores.json                 # Results with causal scores
└── heatmaps/                       # Visualization images
    └── {dataset}_{var}.png

Usage Example:
==============
```python
from causalab.experiments.interchange_targets import (
    build_attention_head_targets,
    build_residual_stream_targets,
    build_mlp_targets,
)
from causalab.experiments.scripts.interchange_score_heatmap import run_interchange_score_heatmap

# Attention heads
targets = build_attention_head_targets(
    pipeline, layers, heads, token_position, mode="one_target_per_unit"
)
result = run_interchange_score_heatmap(
    causal_model=causal_model,
    interchange_targets=targets,
    dataset_path=dataset_path,
    pipeline=pipeline,
    target_variable_groups=[("answer",)],
    batch_size=32,
    output_dir="outputs/attention_patching",
    metric=metric,
)

# Residual stream
targets = build_residual_stream_targets(
    pipeline, layers, token_positions, mode="one_target_per_unit"
)
result = run_interchange_score_heatmap(
    causal_model=causal_model,
    interchange_targets=targets,
    ...
)

# MLPs
targets = build_mlp_targets(
    pipeline, layers, token_positions, mode="one_target_per_unit"
)
result = run_interchange_score_heatmap(
    causal_model=causal_model,
    interchange_targets=targets,
    ...
)
```
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Callable, Tuple

from datasets import load_from_disk, Dataset

from tqdm import tqdm

from causalab.neural.pipeline import LMPipeline
from causalab.neural.model_units import InterchangeTarget
from causalab.neural.pyvene_core.interchange import run_interchange_interventions
from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import CounterfactualDataset
from causalab.experiments.metric import (
    InterchangeMetric,
    score_intervention_outputs,
    causal_score_intervention_outputs,
)
from causalab.experiments.visualizations import (
    detect_component_type_from_targets,
    extract_grid_dimensions_from_targets,
    plot_score_heatmap,
)

logger = logging.getLogger(__name__)


def run_interchange_score_heatmap(
    causal_model: CausalModel,
    interchange_targets: Dict[tuple[Any, ...], InterchangeTarget],
    dataset_path: str,
    pipeline: LMPipeline,
    target_variable_groups: List[Tuple[str, ...]],
    batch_size: int,
    output_dir: str,
    metric: Callable[[Any, Any], bool],
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run interchange interventions and generate score heatmaps for any component type.

    This function accepts a pre-built Dict[tuple, InterchangeTarget] and automatically:
    1. Detects the component type (attention_head, residual_stream, or mlp)
    2. Runs interchange interventions on each target
    3. Computes causal scores using the provided metric
    4. Generates the appropriate score heatmap visualization
    5. Returns scores and metadata

    Args:
        causal_model: Causal model for generating expected outputs
        interchange_targets: Pre-built Dict[tuple, InterchangeTarget] from any builder.
                            Should use mode="one_target_per_unit".
        dataset_path: Path to filtered dataset directory
        pipeline: LMPipeline object with loaded model
        target_variable_groups: List of variable groups to evaluate. Each group is a tuple
                               of variable names that are evaluated jointly.
                               (e.g., [("answer",), ("answer", "position")])
        batch_size: Batch size for evaluation
        output_dir: Output directory for results
        metric: Function to compare neural output with expected output
        save_results: Whether to save metadata and results to disk (default: True)
        verbose: Whether to print progress information

    Returns:
        Dictionary containing:
            - scores: Dict mapping variable group (as tuple) to {key: score} dict
            - component_type: detected component type
            - metadata: experiment configuration
            - output_paths: paths to saved files

    Raises:
        FileNotFoundError: If dataset_path does not exist
        ValueError: If component type cannot be detected
    """
    # Configure logging level based on verbose flag
    if verbose:
        logging.getLogger("causalab").setLevel(logging.DEBUG)

    # Detect component type
    component_type = detect_component_type_from_targets(interchange_targets)

    logger.debug(f"Detected component type: {component_type}")
    logger.debug(f"Number of targets: {len(interchange_targets)}")

    # Extract grid dimensions for visualization
    grid_dims = extract_grid_dimensions_from_targets(
        component_type, interchange_targets
    )

    # Load dataset
    dataset_name = Path(dataset_path).parent.name
    hf_dataset = load_from_disk(dataset_path)
    if not isinstance(hf_dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(hf_dataset).__name__}")
    dataset = CounterfactualDataset(dataset=hf_dataset, id=dataset_name)

    # Run interventions
    raw_results: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    pbar = tqdm(
        interchange_targets.items(),
        desc="Running interventions",
        disable=not verbose,
        total=len(interchange_targets),
    )
    for key, target in pbar:
        raw_results[key] = run_interchange_interventions(
            pipeline=pipeline,
            counterfactual_dataset=dataset,
            interchange_target=target,
            batch_size=batch_size,
            output_scores=False,
        )
    pbar.close()

    # Score intervention outputs
    result = causal_score_intervention_outputs(
        raw_results=raw_results,
        dataset=dataset,
        causal_model=causal_model,
        target_variable_groups=target_variable_groups,
        metric=metric,
    )

    # Extract per-variable-group scores
    all_scores = {}
    for var_group in target_variable_groups:
        all_scores[var_group] = {
            res_key: res["scores_by_variable"][str(var_group)]
            for res_key, res in result["results_by_key"].items()
        }

    # Create metadata
    model_name = getattr(pipeline, "model_or_name", None)
    metadata: Dict[str, Any] = {
        "experiment_type": "evaluation",
        "model": model_name,
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "target_variable_groups": [list(vg) for vg in target_variable_groups],
        "num_examples": len(dataset),
        "num_targets": len(result["results_by_key"]),
        "avg_score": float(result["avg_score"]),
        "scores_by_variable": {
            str(k): float(v) for k, v in result["scores_by_variable"].items()
        },
        "component_type": component_type,
    }

    if component_type == "attention_head":
        metadata["layers"] = grid_dims["layers"]
        metadata["heads"] = grid_dims["heads"]
    else:
        metadata["layers"] = grid_dims["layers"]
        metadata["token_position_ids"] = grid_dims["token_position_ids"]

    # Generate visualizations
    output_paths: Dict[str, Any] = {}

    if save_results:
        heatmap_dir = os.path.join(output_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)

        for var_group in target_variable_groups:
            scores = all_scores[var_group]

            # Generate filename-safe version of the variable group
            var_name = "_".join(var_group)

            # Generate heatmap
            heatmap_path = os.path.join(heatmap_dir, f"{dataset_name}_{var_name}.png")
            plot_score_heatmap(
                scores=scores,
                interchange_targets=interchange_targets,
                title=f"{var_name.replace('_', ' ').title()} - {dataset_name}",
                save_path=heatmap_path,
            )

        output_paths["heatmap_dir"] = heatmap_dir

    return {
        "scores": all_scores,
        "component_type": component_type,
        "metadata": metadata,
        "output_paths": output_paths,
    }


def run_interchange_custom_score_heatmap(
    interchange_targets: Dict[tuple[Any, ...], InterchangeTarget],
    dataset_path: str,
    pipeline: LMPipeline,
    batch_size: int,
    output_dir: str,
    metric: InterchangeMetric,
    causal_model: CausalModel | None = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run interchange interventions with custom metric and generate score heatmaps.

    Similar to run_interchange_score_heatmap but uses InterchangeMetric for flexible
    scoring. The metric declares what data it needs (causal expected outputs,
    original model outputs) and the function handles computing those automatically.

    This function accepts a pre-built Dict[tuple, InterchangeTarget] and automatically:
    1. Detects the component type (attention_head, residual_stream, or mlp)
    2. Runs interchange interventions on each target
    3. Computes scores using the provided InterchangeMetric
    4. Generates the appropriate score heatmap visualization
    5. Returns scores and metadata

    Args:
        interchange_targets: Pre-built Dict[tuple, InterchangeTarget] from any builder.
                            Should use mode="one_target_per_unit".
        dataset_path: Path to filtered dataset directory
        pipeline: LMPipeline object with loaded model
        batch_size: Batch size for evaluation
        output_dir: Output directory for results
        metric: InterchangeMetric defining the scoring function and data requirements
        causal_model: Required if metric.needs_causal_expected is True
        save_results: Whether to save metadata and results to disk (default: True)
        verbose: Whether to print progress information

    Returns:
        Dictionary containing:
            - scores: Dict mapping keys to scores
            - component_type: detected component type
            - metadata: experiment configuration
            - output_paths: paths to saved files

    Raises:
        FileNotFoundError: If dataset_path does not exist
        ValueError: If component type cannot be detected
        ValueError: If metric.needs_causal_expected is True but causal_model is None

    Example:
        ```python
        from causalab.experiments.metric import InterchangeMetric
        from causalab.experiments.scripts.interchange_score_heatmap import run_interchange_custom_score_heatmap

        # Define a faithfulness metric (compares to original model output)
        def faithfulness_checker(intervention_output, expected, original):
            return 1.0 if intervention_output["string"] == original["string"] else 0.0

        metric = InterchangeMetric(
            fn=faithfulness_checker,
            needs_causal_expected=False,
            needs_original_output=True,
        )

        result = run_interchange_custom_score_heatmap(
            interchange_targets=targets,
            dataset_path=dataset_path,
            pipeline=pipeline,
            batch_size=32,
            output_dir="outputs/faithfulness",
            metric=metric,
        )
        ```
    """
    # Configure logging level based on verbose flag
    if verbose:
        logging.getLogger("causalab").setLevel(logging.DEBUG)

    # Validate required arguments
    if metric.needs_causal_expected:
        if causal_model is None:
            raise ValueError(
                "causal_model is required when metric.needs_causal_expected is True"
            )
        if metric.target_variables is None:
            raise ValueError(
                "metric.target_variables is required when metric.needs_causal_expected is True. "
                "Use make_causal_metric() to create a metric with target_variables."
            )

    # Detect component type
    component_type = detect_component_type_from_targets(interchange_targets)

    logger.debug(f"Detected component type: {component_type}")
    logger.debug(f"Number of targets: {len(interchange_targets)}")

    # Extract grid dimensions for visualization
    grid_dims = extract_grid_dimensions_from_targets(
        component_type, interchange_targets
    )

    # Load dataset
    dataset_name = Path(dataset_path).parent.name
    hf_dataset = load_from_disk(dataset_path)
    if not isinstance(hf_dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(hf_dataset).__name__}")
    dataset = CounterfactualDataset(dataset=hf_dataset, id=dataset_name)

    # Compute original outputs if needed
    original_outputs = None
    if metric.needs_original_output:
        outputs = pipeline.compute_outputs(dataset, batch_size=batch_size)
        original_outputs = outputs["base_outputs"]

    # Run interventions
    raw_results: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    pbar = tqdm(
        interchange_targets.items(),
        desc="Running interventions",
        disable=not verbose,
        total=len(interchange_targets),
    )
    for key, target in pbar:
        raw_results[key] = run_interchange_interventions(
            pipeline=pipeline,
            counterfactual_dataset=dataset,
            interchange_target=target,
            batch_size=batch_size,
            output_scores=False,
        )
    pbar.close()

    # Score using the core scoring function
    scores = score_intervention_outputs(
        raw_results=raw_results,
        dataset=dataset,
        metric=metric,
        causal_model=causal_model,
        original_outputs=original_outputs,
    )

    # Compute overall average score
    avg_score = float(sum(scores.values()) / len(scores)) if scores else 0.0

    # Create metadata
    model_name = getattr(pipeline, "model_or_name", None)
    metadata = {
        "experiment_type": "custom_evaluation",
        "model": model_name,
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "num_examples": len(dataset),
        "num_targets": len(scores),
        "avg_score": float(avg_score),
        "needs_causal_expected": metric.needs_causal_expected,
        "needs_original_output": metric.needs_original_output,
        "target_variables": list(metric.target_variables)
        if metric.target_variables
        else None,
        "component_type": component_type,
    }

    if component_type == "attention_head":
        metadata["layers"] = grid_dims["layers"]
        metadata["heads"] = grid_dims["heads"]
    else:
        metadata["layers"] = grid_dims["layers"]
        metadata["token_position_ids"] = grid_dims["token_position_ids"]

    # Generate visualizations
    output_paths = {}

    if save_results:
        heatmap_dir = os.path.join(output_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)

        # Generate heatmap
        heatmap_path = os.path.join(heatmap_dir, f"{dataset_name}_scores.png")
        plot_score_heatmap(
            scores=scores,
            interchange_targets=interchange_targets,
            title=f"Custom Metric Scores - {dataset_name}",
            save_path=heatmap_path,
        )

        output_paths["heatmap_dir"] = heatmap_dir

    return {
        "scores": scores,
        "component_type": component_type,
        "metadata": metadata,
        "output_paths": output_paths,
    }
