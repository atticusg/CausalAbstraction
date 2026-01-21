#!/usr/bin/env -S uv run python
"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Step 2: Run Residual Stream Patching Experiments

This script:
1. Loads the filtered counterfactual dataset
2. Loads the specified language model
3. Runs full vector residual stream patching at token positions across all layers
4. Saves raw intervention results (no scoring yet - done in step 3)

Usage:
    python 02_run_interventions.py --model MODEL --digits D [--dataset PATH] [--output OUTPUT]
    python 02_run_interventions.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --digits 2
    python 02_run_interventions.py --test  # Run in test mode (2 layers, 1 batch)
"""

import sys

sys.path.append("/mnt/polished-lake/home/atticus/CausalAbstraction")

import argparse
import torch
from pathlib import Path
import json
import pickle

from causalab.tasks.general_addition.config import (
    create_two_number_two_digit_config,
    create_two_number_three_digit_config,
    create_general_config,
)
from causalab.tasks.general_addition.causal_models import create_basic_addition_model
from causalab.tasks.general_addition.experiments.tokenization_config import (
    get_all_number_token_indices,
    get_tokens_per_number,
)
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_position_builder import TokenPosition, get_last_token_index
from causalab.causal.counterfactual_dataset import CounterfactualExample
from datasets import load_from_disk, Dataset


def create_token_positions_for_experiment(pipeline, config, num_digits, model_name):
    """
    Create token positions for the experiment.

    We create separate token positions for each token of each number.
    For multi-token numbers (e.g., Gemma with 2-digit = 2 tokens), we create
    number_0_tok0, number_0_tok1, etc.

    The number of tokens per number is determined by the model and digit count:
    - Llama: 1 token for 2/3-digit, 2 tokens for 4-digit
    - Gemma: 2 tokens for 2-digit, 3 for 3-digit, 4 for 4-digit
    - OLMo: 1 token for 2/3-digit, 2 tokens for 4-digit

    Args:
        pipeline: LM pipeline
        config: Task configuration
        num_digits: Number of digits per number
        model_name: Model name to determine tokenization pattern
    """
    token_positions_dict = {}

    tokens_per_number = get_tokens_per_number(model_name, num_digits)
    print(
        f"  Model tokenization: {tokens_per_number} token(s) per {num_digits}-digit number"
    )

    # Create separate token positions for each token of each number
    for k in range(2):  # Two addends
        for tok_idx in range(tokens_per_number):

            def make_number_token_indexer(addend_idx, token_idx):
                def indexer(input_sample):
                    all_tokens = get_all_number_token_indices(
                        input_sample, pipeline, addend_idx, num_digits, model_name
                    )
                    # Return the specific token (must be a list with exactly one element)
                    if token_idx < len(all_tokens):
                        return [all_tokens[token_idx]]
                    return []

                return indexer

            token_positions_dict[f"number_{k}_tok{tok_idx}"] = TokenPosition(
                make_number_token_indexer(k, tok_idx),
                pipeline,
                id=f"number_{k}_tok{tok_idx}",
            )

    # Last token position
    token_positions_dict["last_token"] = TokenPosition(
        lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token"
    )

    return token_positions_dict


def main():
    parser = argparse.ArgumentParser(
        description="Run residual stream patching on addition"
    )
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
        "--dataset",
        type=str,
        default=None,
        help="Path to filtered dataset (default: auto-generated from model/digits)",
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
        default=512,
        help="Batch size for interventions (default: 512)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (only first 2 layers, 1 batch)",
    )

    args = parser.parse_args()

    # Note: We always save raw-only in this script. Scoring is done in step 3.
    # This is intentional separation of concerns: interventions (GPU) vs scoring (CPU)

    # Auto-generate paths if not specified
    model_short = args.model.split("/")[-1].replace("-", "_").lower()
    if args.dataset is None:
        args.dataset = f"tasks/general_addition/datasets/random_cf_{model_short}_{args.digits}d/filtered_dataset"
    if args.output is None:
        args.output = f"tasks/general_addition/results/{model_short}_{args.digits}d"

    # Configuration
    print("=" * 70)
    print("Addition Residual Stream Patching")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Digits: {args.digits}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {args.output}")
    print(f"  Batch size: {args.batch_size}")
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

    # Create causal model
    causal_model = create_basic_addition_model(config)
    print(f"Causal model: {causal_model.id}")
    print()

    # Load dataset
    print(f"Loading filtered dataset from {args.dataset}...")
    hf_dataset = load_from_disk(args.dataset)
    dataset: list[CounterfactualExample] = list(hf_dataset)  # type: ignore[assignment]
    print(f"✓ Loaded {len(dataset)} examples")
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
        max_length=16,
    )
    print("✓ Model loaded")
    print()

    # Create token positions
    print("Creating token positions...")
    token_positions_dict = create_token_positions_for_experiment(
        pipeline, config, args.digits, args.model
    )
    token_positions = list(token_positions_dict.values())
    print(f"✓ Created token positions: {[tp.id for tp in token_positions]}")
    print()

    # Show example with highlighted tokens
    print("Example with token highlights:")
    example = dataset[0]["input"]
    print(f"  Prompt: {example['raw_input']}")
    for tp in token_positions:
        highlighted = tp.highlight_selected_token(example)
        print(f"  {tp.id}: {highlighted}")
    print()

    # Determine layers to test
    num_layers = pipeline.get_num_layers()
    if args.test:
        layers = [0]  # Only layer 0 for quick testing
        # Take only one batch of data
        test_size = args.batch_size
        test_dataset = dataset[: min(test_size, len(dataset))]
        datasets_dict = {"random_cf": test_dataset}
        print(f"*** TEST MODE: layer 0 only, {len(test_dataset)} examples ***")
    else:
        layers = list(range(-1, num_layers))  # -1 (input) through all layers
        datasets_dict = {"random_cf": dataset}
        print(f"Running on all {len(layers)} layers (-1 to {num_layers - 1})")

    print(f"  Layers: {layers}")
    print()

    # Create experiment configuration (NEW: must include "id")
    experiment_config = {
        "batch_size": args.batch_size,
        "output_scores": False,  # Don't store logits to save memory
        "id": f"addition_{args.digits}d_{model_short}",  # Required in new API
    }

    # Create experiment (NEW: no causal_model, no checker)
    experiment = PatchResidualStream(
        pipeline, layers, token_positions, config=experiment_config
    )

    # Run interventions (NEW: returns raw results, no target_variables_list)
    print("Running residual stream patching interventions...")
    print("=" * 70)
    print("NOTE: This only runs the interventions. Scoring is done in step 3.")
    print("=" * 70)
    raw_results = experiment.perform_interventions(datasets_dict, verbose=True)
    print("=" * 70)
    print("✓ Interventions complete")
    print()

    # Save raw results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_subdir = output_dir / ("test_results" if args.test else "full_results")
    results_subdir.mkdir(parents=True, exist_ok=True)

    raw_results_path = results_subdir / "raw_results.pkl"
    print(f"Saving raw results to {raw_results_path}...")
    with open(raw_results_path, "wb") as f:
        pickle.dump(raw_results, f)
    print("✓ Raw results saved")
    print()

    # Save experiment metadata
    metadata = {
        "model": args.model,
        "experiment_id": experiment_config["id"],
        "num_digits": args.digits,
        "dataset_path": args.dataset,  # Include dataset path for recompute_scores
        "dataset_size": len(datasets_dict["random_cf"]),
        "num_layers": len(layers),
        "layers": layers,
        "token_positions": [tp.id for tp in token_positions],
        "test_mode": args.test,
    }

    metadata_path = results_subdir / "experiment_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")
    print()

    print("=" * 70)
    print("✓ Intervention experiment complete!")
    print("=" * 70)
    print()
    print("Next step: Run step 3 to compute scores and visualize:")
    print(f"  python 03_compute_scores_and_visualize.py --results {raw_results_path}")
    print()
    print("Tip: You can rerun step 3 with different --target-vars without")
    print("     re-running these expensive interventions!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
