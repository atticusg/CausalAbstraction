#!/usr/bin/env -S uv run python
"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Universal Dataset Generation and Filtering Script for Entity Binding

This script generates counterfactual datasets and filters them based on model performance.
Saves both the original (unfiltered) and filtered datasets.

Usage:
    python generate_and_filter_dataset.py --config CONFIG --model MODEL [options]
    python generate_and_filter_dataset.py --config love --model meta-llama/Llama-3.1-8B-Instruct
    python generate_and_filter_dataset.py --config action --test  # Test mode
"""

import sys

sys.path.append("/mnt/polished-lake/home/atticus/CausalAbstraction")

import argparse
import torch
from pathlib import Path
import json

from causalab.tasks.entity_binding.experiment_config import (
    get_task_config,
    get_causal_model,
    get_counterfactual_generator,
    get_checker,
)
from causalab.neural.pipeline import LMPipeline
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.causal_utils import (
    generate_counterfactual_samples,
    save_counterfactual_examples,
)
from causalab.experiments.filter import filter_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate and filter entity binding dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Task configuration: love, action, or positional_entity",
    )
    parser.add_argument(
        "--counterfactual",
        type=str,
        default="swap_query_group",
        help="Counterfactual generator name (default: swap_query_group)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use (default: meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Number of counterfactual pairs to generate (default: 128)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: auto-generated from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for filtering (default: 32)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: size=8, batch_size=8"
    )

    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.size = 8
        args.batch_size = 8
        print("\n*** TEST MODE: size=8, batch_size=8 ***\n")

    # Auto-generate output path if not specified
    if args.output is None:
        test_suffix = "_test" if args.test else ""
        args.output = f"tasks/entity_binding/datasets/{args.config}_{args.counterfactual}{test_suffix}"

    # Configuration
    print("=" * 70)
    print("Entity Binding Dataset Generation and Filtering")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Task config: {args.config}")
    print(f"  Counterfactual: {args.counterfactual}")
    print(f"  Model: {args.model}")
    print(f"  Dataset size: {args.size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {args.output}")
    print(f"  Test mode: {args.test}")
    print()

    # Get task configuration
    try:
        config = get_task_config(args.config)
        print("Task configuration loaded:")
        print(f"  Max groups: {config.max_groups}")
        print(f"  Entities per group: {config.max_entities_per_group}")
        print(f"  Entity roles: {config.entity_roles}")
        print(f"  Template: {config.statement_template}")
        print()
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Create causal model
    causal_model = get_causal_model(config)
    print(f"Causal model: {causal_model.id}")
    print()

    # Get counterfactual generator
    try:
        cf_generator = get_counterfactual_generator(args.counterfactual, config)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Generate counterfactual dataset
    print(f"Generating {args.size} counterfactual pairs using {args.counterfactual}...")
    dataset: list[CounterfactualExample] = generate_counterfactual_samples(
        args.size, cf_generator
    )
    print(f"✓ Generated {len(dataset)} pairs")
    print()

    # Show example
    print("Example pair:")
    print(f"  Input:  {dataset[0]['input']['raw_input']}")
    print(f"  Counter: {dataset[0]['counterfactual_inputs'][0]['raw_input']}")
    print()

    # Load language model
    print(f"Loading language model: {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    pipeline = LMPipeline(args.model, max_new_tokens=5, device=device, max_length=256)
    print("✓ Model loaded")
    print()

    # Get checker function
    checker = get_checker()

    # Filter the dataset
    print("Filtering dataset based on model performance...")
    filtered_dataset = filter_dataset(
        dataset=dataset,
        pipeline=pipeline,
        causal_model=causal_model,
        metric=checker,
        batch_size=args.batch_size,
    )
    print()
    print("Filtering results:")
    print(f"  Original: {len(dataset)} examples")
    print(f"  Filtered: {len(filtered_dataset)} examples")
    print(f"  Keep rate: {len(filtered_dataset) / len(dataset) * 100:.1f}%")
    print()

    # Check if we have enough data
    if len(filtered_dataset) == 0:
        print(
            "⚠ WARNING: No examples passed filtering! Model may not be capable of this task."
        )
        return 1
    elif len(filtered_dataset) < args.size * 0.5 and not args.test:
        print(
            f"⚠ WARNING: Only {len(filtered_dataset)}/{args.size} examples passed. "
            f"Consider increasing dataset size or checking model capability."
        )
        print()

    # Save datasets
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save original (unfiltered) dataset
    original_path = output_path / "original_dataset.json"
    print(f"Saving original dataset to {original_path}...")
    save_counterfactual_examples(dataset, str(original_path))
    print(f"✓ Original dataset saved ({len(dataset)} examples)")

    # Save filtered dataset
    filtered_path = output_path / "filtered_dataset.json"
    print(f"Saving filtered dataset to {filtered_path}...")
    save_counterfactual_examples(filtered_dataset, str(filtered_path))
    print(f"✓ Filtered dataset saved ({len(filtered_dataset)} examples)")
    print()

    # Save metadata
    metadata = {
        "config_name": args.config,
        "counterfactual_type": args.counterfactual,
        "model": args.model,
        "original_size": len(dataset),
        "filtered_size": len(filtered_dataset),
        "keep_rate": len(filtered_dataset) / len(dataset),
        "task_config": {
            "max_groups": config.max_groups,
            "entities_per_group": config.max_entities_per_group,
            "entity_roles": config.entity_roles,
            "template": config.statement_template,
        },
        "test_mode": args.test,
    }

    metadata_path = output_path / "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")
    print()

    print("=" * 70)
    print("✓ Dataset generation and filtering complete!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_path}")
    print(f"  - Original dataset: {original_path} ({len(dataset)} examples)")
    print(f"  - Filtered dataset: {filtered_path} ({len(filtered_dataset)} examples)")
    print(f"  - Metadata: {metadata_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
