"""
Trace information flow through residual stream interventions.

This module provides a function to perform interchange interventions at each (layer, token_position)
combination to understand how information flows through the model's residual stream.

The function swaps activations from a counterfactual prompt and visualizes how this
affects the model's output predictions at different layers and positions.

Usage Example:
==============

```python
from causalab.experiments.scripts.residual_stream_tracing import run_residual_stream_tracing
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_position_builder import get_list_of_each_token

# Create pipeline
pipeline = LMPipeline("gpt2", max_new_tokens=1, device="cuda")

# Define prompts
prompt = "The sky is blue"
counterfactual_prompt = "The sky is red"

# Create token positions (use all tokens in prompt)
token_positions = get_list_of_each_token(prompt, pipeline)

# Run tracing experiment
result = run_residual_stream_tracing(
    pipeline=pipeline,
    prompt=prompt,
    counterfactual_prompt=counterfactual_prompt,
    token_positions=token_positions,
    output_dir="./tracing_results",
    layers=None,  # All layers
    generate_visualization=True,
    verbose=True
)

# Access results
print(f"Total interventions: {result['metadata']['total_interventions']}")
print(f"Saved to: {result['output_paths']['results_path']}")
```

Output Files:
============
output_dir/
├── results/                       # Results directory
│   ├── intervention_results.json        # Raw intervention string outputs
│   └── intervention_results.safetensors # Raw intervention tensor outputs
├── metadata.json                  # Experiment configuration and statistics
└── residual_stream_heatmap.png    # Visualization of output tokens
"""

import logging
import os
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm

from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_position_builder import TokenPosition
from causalab.causal.counterfactual_dataset import CounterfactualDataset
from causalab.neural.pyvene_core.interchange import run_interchange_interventions
from causalab.experiments.visualizations.string_heatmap import (
    plot_residual_stream_intervention_heatmap,
)
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.experiments.io import (
    save_experiment_metadata,
    save_json_results,
    save_tensor_results,
)

logger = logging.getLogger(__name__)


def run_residual_stream_tracing(
    pipeline: LMPipeline,
    prompt: str,
    counterfactual_prompt: str,
    token_positions: List[TokenPosition],
    output_dir: str,
    layers: Optional[List[int]] = None,
    generate_visualization: bool = True,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Trace information flow through residual stream interventions.

    Args:
        pipeline: LMPipeline object with loaded model
        prompt: Original prompt text
        counterfactual_prompt: Counterfactual prompt text
        token_positions: List of TokenPosition objects to analyze
        output_dir: Output directory for results
        layers: Specific layers to analyze (default: all layers including -1)
        generate_visualization: Whether to generate visualization (default: True)
        save_results: Whether to save metadata and results to disk (default: True)
        verbose: Whether to print progress information

    Returns:
        Dictionary containing:
            - intervention_results: raw intervention outputs
            - metadata: experiment configuration
            - output_paths: paths to saved files
    """
    # Configure logging level based on verbose flag
    if verbose:
        logging.getLogger("causalab").setLevel(logging.DEBUG)

    # Create Dataset
    # Create a single-example counterfactual dataset
    dataset_dict = {
        "input": [{"raw_input": prompt}],
        "counterfactual_inputs": [[{"raw_input": counterfactual_prompt}]],
    }

    counterfactual_dataset = CounterfactualDataset.from_dict(dataset_dict)

    # Determine Layers
    num_layers = pipeline.model.config.num_hidden_layers
    if layers is None:
        layers = [-1] + list(range(num_layers))

    # Build all targets at once
    targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=layers,
        token_positions=token_positions,
        mode="one_target_per_unit",
    )

    # Run interventions for each (layer, position) combination
    intervention_results = {}
    for (layer, pos_id), target in tqdm(
        targets.items(), desc="Running interventions", disable=not verbose
    ):
        outputs = run_interchange_interventions(
            pipeline=pipeline,
            counterfactual_dataset=counterfactual_dataset,
            interchange_target=target,
            batch_size=1,
            output_scores=False,
        )
        intervention_results[(layer, pos_id)] = outputs

    # Create metadata inline (following attention_head_DBM pattern)
    model_name = getattr(pipeline, "model_or_name", None)
    token_position_ids = [pos.id for pos in token_positions]
    metadata = {
        "experiment_type": "tracing",
        "model": model_name,
        "prompt": prompt,
        "counterfactual_prompt": counterfactual_prompt,
        "num_layers": num_layers,
        "layers_used": layers,
        "num_token_positions": len(token_positions),
        "token_position_ids": token_position_ids,
        "total_interventions": len(layers) * len(token_positions),
    }

    # Save results (conditional)
    output_paths = {}

    if save_results:
        # Save results to results directory
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Save raw intervention results - split into JSON (strings) and safetensors (sequences)
        # Convert tuple keys to strings for JSON
        results_json = {}
        results_tensors = {}
        for (layer, pos), result in intervention_results.items():
            key = f"{layer}__{pos}"
            results_json[key] = {"string": result.get("string", [])}
            if "sequences" in result and result["sequences"]:
                # Concatenate batch tensors into a single tensor
                sequences_list = result["sequences"]
                results_tensors[key] = torch.cat(sequences_list, dim=0)

        results_json_path = save_json_results(
            data=results_json,
            output_dir=results_dir,
            filename="intervention_results.json",
        )
        output_paths["results_json_path"] = results_json_path

        if results_tensors:
            results_tensors_path = save_tensor_results(
                tensors=results_tensors,
                output_dir=results_dir,
                filename="intervention_results.safetensors",
            )
            output_paths["results_tensors_path"] = results_tensors_path
        output_paths["results_dir"] = results_dir

        # Generate Visualization
        if generate_visualization:
            heatmap_path = os.path.join(output_dir, "residual_stream_heatmap.png")

            plot_residual_stream_intervention_heatmap(
                intervention_results=intervention_results,
                prompt=prompt,
                layers=layers,
                token_positions=token_positions,
                pipeline=pipeline,
                correct_answer=None,  # No expected answer in task-agnostic mode
                title="Residual Stream Tracing: Output Tokens After Intervention",
                save_path=heatmap_path,
                color_by_frequency=True,  # Color by output token frequency
            )

            output_paths["heatmap_path"] = heatmap_path

        # Save metadata
        metadata_path = save_experiment_metadata(
            metadata=metadata,
            output_dir=output_dir,
        )
        output_paths["metadata_path"] = metadata_path

    return {
        "intervention_results": intervention_results,
        "metadata": metadata,
        "output_paths": output_paths,
    }
