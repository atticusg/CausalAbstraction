#!/usr/bin/env -S uv run python
"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Step 2: Run Intervention Experiments

This script:
1. Loads the filtered counterfactual dataset
2. Loads the specified language model
3. Runs residual stream patching interventions across specified layers
4. Saves raw intervention results (no scoring yet - done in step 3)

Usage:
    python run_interventions.py --config CONFIG --dataset PATH --model MODEL [options]
    python run_interventions.py --config love --dataset datasets/love/filtered_dataset --model ...
    python run_interventions.py --config action --dataset ... --test  # Test mode: layer 0 only
"""

import sys

sys.path.append("/mnt/polished-lake/home/atticus/CausalAbstraction")

import argparse
import torch
import pickle
from pathlib import Path
import json

from causalab.tasks.entity_binding.experiment_config import (
    get_task_config,
    get_token_positions,
    get_pipeline_config,
)
from causalab.neural.pipeline import LMPipeline
from causalab.causal.counterfactual_dataset import CounterfactualExample
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(description="Run interventions on entity binding")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Task configuration: love, action, or positional_entity",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to filtered dataset directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use (default: meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="residual_stream",
        choices=["residual_stream", "attention_heads"],
        help="Experiment type (default: residual_stream)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Layer range as start:end (default: -1:num_layers)",
    )
    parser.add_argument(
        "--token-positions",
        type=str,
        default="last_token",
        help="Token position type: last_token or both_e1 (default: last_token)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for interventions (default: 8)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: layer 0 only, 8 examples, batch_size=8",
    )

    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.batch_size = 8
        print("\n*** TEST MODE: layer 0 only, 8 examples, batch_size=8 ***\n")

    # Auto-generate output path if not specified
    if args.output is None:
        test_suffix = "_test" if args.test else ""
        args.output = f"tasks/entity_binding/results/{args.config}{test_suffix}"

    # Configuration
    print("=" * 70)
    print("Entity Binding Intervention Experiments")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Task config: {args.config}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Experiment type: {args.experiment_type}")
    print(f"  Token positions: {args.token_positions}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {args.output}")
    print(f"  Test mode: {args.test}")
    print()

    # Get task configuration
    try:
        config = get_task_config(args.config)
        print("Task configuration loaded:")
        print(f"  Max groups: {config.max_groups}")
        print()
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    try:
        hf_dataset = load_from_disk(args.dataset)
        dataset: list[CounterfactualExample] = list(hf_dataset)  # type: ignore[assignment]
        print(f"✓ Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return 1
    print()

    # Show example
    print("Example from dataset:")
    print(f"  Input:  {dataset[0]['input']['raw_input'][:80]}...")
    print(f"  Counter: {dataset[0]['counterfactual_inputs'][0]['raw_input'][:80]}...")
    print()

    # Load language model
    print(f"Loading language model: {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    pipeline_cfg = get_pipeline_config(args.config)
    pipeline = LMPipeline(
        args.model,
        max_new_tokens=pipeline_cfg["max_new_tokens"],
        device=device,
        max_length=pipeline_cfg["max_length"],
    )
    num_layers = pipeline.get_num_layers()
    print(f"✓ Model loaded ({num_layers} layers)")
    print()

    # Create token positions
    token_positions_dict = get_token_positions(pipeline, config, args.token_positions)
    token_positions = list(token_positions_dict.values())
    print("Token positions:")
    for tp in token_positions:
        print(f"  {tp.id}")
    print()

    # Determine layers to test
    if args.test:
        layers = [0]  # Only layer 0 for testing
        test_size = 8
        test_dataset = dataset[: min(test_size, len(dataset))]
        datasets_dict = {"dataset": test_dataset}
        print(f"TEST MODE: Using layer 0 only, {len(test_dataset)} examples")
    else:
        if args.layers:
            start, end = map(int, args.layers.split(":"))
            if end == -1:
                end = num_layers
            layers = list(range(start, end))
        else:
            layers = list(range(-1, num_layers))
        datasets_dict = {"dataset": dataset}
        print(f"Running on {len(layers)} layers: {layers[0]} to {layers[-1]}")

    print(f"  Layers: {layers}")
    print()

    # Create experiment
    experiment_config = {
        "batch_size": args.batch_size,
        "id": f"entity_binding_{args.config}",
    }

    if args.experiment_type == "residual_stream":
        experiment = PatchResidualStream(
            pipeline, layers, token_positions, config=experiment_config
        )
    else:
        print(f"Error: Experiment type {args.experiment_type} not yet implemented")
        return 1

    # Run interventions
    print("Running interventions...")
    print("=" * 70)
    print(
        "NOTE: This only runs the interventions. Scoring/visualization is done in step 3."
    )
    print("=" * 70)
    raw_results = experiment.perform_interventions(datasets_dict, verbose=True)
    print("=" * 70)
    print("✓ Interventions complete")
    print()

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    raw_results_path = output_dir / "raw_results.pkl"
    print(f"Saving raw results to {raw_results_path}...")
    with open(raw_results_path, "wb") as f:
        pickle.dump(raw_results, f)
    print("✓ Raw results saved")
    print()

    # Save metadata
    metadata = {
        "config_name": args.config,
        "model": args.model,
        "experiment_id": experiment_config["id"],
        "experiment_type": args.experiment_type,
        "dataset_path": args.dataset,  # Important: for visualize_results.py to find dataset
        "dataset_size": len(datasets_dict["dataset"]),
        "layers": layers,
        "token_positions": list(token_positions_dict.keys()),
        "test_mode": args.test,
    }

    metadata_path = output_dir / "experiment_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")
    print()

    print("=" * 70)
    print("✓ Intervention experiment complete!")
    print("=" * 70)
    print()
    print(f"Results saved to: {output_dir}")
    print("  - Raw results: raw_results.pkl")
    print("  - Metadata: experiment_metadata.json")
    print()
    print(
        "Next step: Run visualize_results.py to compute scores and generate visualizations:"
    )
    print(f"  python visualize_results.py --results {raw_results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
