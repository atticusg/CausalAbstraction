#!/usr/bin/env -S uv run python
"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Script 1: Generate and Filter Counterfactual Dataset

This script:
1. Generates a counterfactual dataset for addition
2. Loads a specified language model
3. Filters the dataset to keep only examples where the model performs correctly
4. Saves the filtered dataset to disk

Usage:
    python 01_generate_and_filter_dataset.py --model MODEL --digits D [--output OUTPUT] [--size SIZE]
    python 01_generate_and_filter_dataset.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --digits 2
    python 01_generate_and_filter_dataset.py --test  # Run in test mode (small dataset)
"""

import sys

sys.path.append("/mnt/polished-lake/home/atticus/CausalAbstraction")

import argparse
import torch
from pathlib import Path
import json

from causalab.tasks.general_addition.config import (
    create_two_number_two_digit_config,
    create_two_number_three_digit_config,
    create_general_config,
)
from causalab.tasks.general_addition.causal_models import create_basic_addition_model
from causalab.tasks.general_addition.counterfactuals import random_counterfactual
from datasets import Dataset
from causalab.neural.pipeline import LMPipeline
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.causal_utils import generate_counterfactual_samples
from causalab.experiments.filter import filter_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate and filter addition dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model to use (default: meta-llama/Meta-Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--digits", type=int, default=2, help="Number of digits per number (2, 3, or 4)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Number of counterfactual pairs to generate (default: 256)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: auto-generated)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for filtering (default: 256)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode (size=8, single batch)"
    )

    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.size = 8
        args.batch_size = 8
        print("\n*** TEST MODE: Using size=8, batch_size=8 ***\n")

    # Auto-generate output path if not specified
    if args.output is None:
        model_short = args.model.split("/")[-1].replace("-", "_").lower()
        # Save in datasets/ directory with model name and config
        args.output = (
            f"tasks/general_addition/datasets/random_cf_{model_short}_{args.digits}d"
        )

    # Configuration
    print("=" * 70)
    print("Addition Dataset Generation and Filtering")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Digits: {args.digits}")
    print(f"  Dataset size: {args.size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {args.output}")
    print(f"  Test mode: {args.test}")
    print()

    # Create task configuration
    if args.digits == 2:
        config = create_two_number_two_digit_config()
    elif args.digits == 3:
        config = create_two_number_three_digit_config()
    elif args.digits == 4:
        config = create_general_config(2, 4)
    else:
        raise ValueError(f"Unsupported digits: {args.digits}. Use 2, 3, or 4.")

    print("Task configuration:")
    print(f"  Max numbers: {config.max_numbers}")
    print(f"  Max digits per number: {config.max_digits}")
    print(f"  Template: {config.templates[0]}")
    print()

    # Create causal model
    causal_model = create_basic_addition_model(config)
    print(f"Causal model: {causal_model.id}")
    print()

    # Generate counterfactual dataset
    print(f"Generating {args.size} counterfactual pairs using random_counterfactual...")
    dataset: list[CounterfactualExample] = generate_counterfactual_samples(
        args.size, lambda: random_counterfactual(config, 2, args.digits)
    )
    print(f"✓ Generated {len(dataset)} pairs")
    print()

    # Show example
    print("Example pair:")
    example_input = dataset[0]["input"]
    example_cf = dataset[0]["counterfactual_inputs"][0]
    # example_input can be dict or CausalTrace - handle both
    raw_input = example_input["raw_input"] if isinstance(example_input, dict) else example_input["raw_input"]
    print(f"  Input:  {raw_input}")
    # Compute answer for display - new_trace expects dict, convert if needed
    input_dict = example_input if isinstance(example_input, dict) else example_input.to_dict()  # type: ignore[union-attr]
    example_output = causal_model.new_trace(input_dict)
    print(f"  Answer: {example_output['raw_output']}")
    cf_raw_input = example_cf["raw_input"] if isinstance(example_cf, dict) else example_cf["raw_input"]
    print(f"  Counter: {cf_raw_input}")
    cf_dict = example_cf if isinstance(example_cf, dict) else example_cf.to_dict()  # type: ignore[union-attr]
    cf_output = causal_model.new_trace(cf_dict)
    print(f"  Answer: {cf_output['raw_output']}")
    print()

    # Load language model
    print(f"Loading language model: {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    pipeline = LMPipeline(
        args.model,
        max_new_tokens=3,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        max_length=64,
    )
    print("✓ Model loaded")
    print()

    # Define checker function
    def checker(neural_output, causal_output):
        """Check if neural network's first generated token exactly matches expected output."""
        # Get the generated token sequences
        sequences = neural_output.get("sequences")
        if sequences is None or len(sequences) == 0 or sequences.shape[1] == 0:
            return False

        # Get first generated token ID
        first_token_id = sequences[0, 0].item()

        # Decode first token
        first_token_str = pipeline.tokenizer.decode(
            [first_token_id], skip_special_tokens=True
        ).strip()

        # Expected output (causal model's answer)
        expected = causal_output.strip()

        # Exact match on first token
        return first_token_str == expected

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

    # Save filtered dataset
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_path = output_path / "filtered_dataset"
    print(f"Saving filtered dataset to {dataset_path}...")
    Dataset.from_list(filtered_dataset).save_to_disk(str(dataset_path))
    print("✓ Dataset saved")
    print()

    # Save metadata
    metadata = {
        "original_size": len(dataset),
        "filtered_size": len(filtered_dataset),
        "keep_rate": len(filtered_dataset) / len(dataset),
        "model": args.model,
        "task": f"addition_{args.digits}d",
        "num_addends": 2,
        "num_digits": args.digits,
        "counterfactual_type": "random",
        "config": {
            "max_numbers": config.max_numbers,
            "max_digits": config.max_digits,
            "template": config.templates[0],
        },
    }

    metadata_path = output_path / "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")
    print()

    print("=" * 70)
    print("✓ Dataset generation and filtering complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
