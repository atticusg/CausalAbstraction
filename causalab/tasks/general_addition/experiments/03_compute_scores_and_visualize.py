#!/usr/bin/env -S uv run python
"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Step 3: Compute Scores and Generate Visualizations

This script:
1. Loads raw intervention results from step 2
2. Loads the causal model and dataset
3. Computes interchange scores for specified target variables
4. Generates heatmaps and text analysis for each variable

Usage:
    python 03_compute_scores_and_visualize.py --results PATH [--target-vars VARS]
    python 03_compute_scores_and_visualize.py --results tasks/general_addition/results/llama_2d/full_results/raw_results.pkl
    python 03_compute_scores_and_visualize.py --results ... --target-vars digit_0_0 digit_0_1 raw_output
"""

import sys

sys.path.append("/mnt/polished-lake/home/atticus/CausalAbstraction")

# Set matplotlib backend to non-interactive for headless environments
import matplotlib

matplotlib.use("Agg")

import argparse
import pickle
from pathlib import Path
import json

from causalab.tasks.general_addition.config import (
    create_two_number_two_digit_config,
    create_two_number_three_digit_config,
    create_general_config,
)
from causalab.tasks.general_addition.causal_models import create_basic_addition_model
from causalab.causal.counterfactual_dataset import CounterfactualExample
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(
        description="Compute scores and visualizations from raw results"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to raw_results.pkl file from step 2",
    )
    parser.add_argument(
        "--target-vars",
        type=str,
        nargs="*",
        default=None,
        help="Target variables to analyze (default: all digit variables + raw_output)",
    )
    parser.add_argument(
        "--include-pairs",
        action="store_true",
        help="Also include all pairs of digit variables (can be slow)",
    )

    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1

    # Load metadata to get experiment configuration
    metadata_path = results_path.parent / "experiment_metadata.json"
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return 1

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print("=" * 70)
    print("Compute Scores and Generate Visualizations")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Results file: {results_path}")
    print(f"  Model: {metadata['model']}")
    print(f"  Digits: {metadata['num_digits']}")
    print(f"  Experiment ID: {metadata['experiment_id']}")
    print()

    # Load raw results
    print("Loading raw results...")
    with open(results_path, "rb") as f:
        raw_results = pickle.load(f)
    print("✓ Raw results loaded")
    print()

    # Create task configuration
    num_digits = metadata["num_digits"]
    if num_digits == 2:
        config = create_two_number_two_digit_config()
    elif num_digits == 3:
        config = create_two_number_three_digit_config()
    elif num_digits == 4:
        config = create_general_config(2, 4)
    else:
        raise ValueError(f"Unsupported digits: {num_digits}")

    # Create causal model
    causal_model = create_basic_addition_model(config)
    print(f"Causal model: {causal_model.id}")
    print()

    # Load dataset (reconstruct path from results path)
    model_short = metadata["model"].split("/")[-1].replace("-", "_").lower()
    dataset_path = f"tasks/general_addition/datasets/random_cf_{model_short}_{num_digits}d/filtered_dataset"

    print(f"Loading dataset from {dataset_path}...")
    hf_dataset = load_from_disk(dataset_path)
    dataset: list[CounterfactualExample] = list(hf_dataset)  # type: ignore[assignment]
    print(f"✓ Loaded {len(dataset)} examples")
    print()

    # Load only the tokenizer for checker function (don't load full model - wasteful!)
    from transformers import AutoTokenizer

    # Define checker function
    def checker(neural_output, causal_output):
        """Check if neural network's first generated token exactly matches expected output."""

        # Expected output (causal model's answer)
        expected = causal_output.strip()

        neural_output = neural_output["string"].strip()

        # Exact match on first token
        return neural_output[0] == expected[0]

    # Determine target variables to analyze
    if args.target_vars is not None:
        # User specified variables
        target_variables_list = [[var] for var in args.target_vars]
    else:
        # Default: all digit variables + raw_output
        target_variables_list = []

        # Collect all digit variable names
        all_digit_vars = []
        for k in range(2):
            for d in range(num_digits):
                all_digit_vars.append(f"digit_{k}_{d}")

        # Track each digit variable individually
        for var in all_digit_vars:
            target_variables_list.append([var])

        # Track all pairs of digit variables if requested
        if args.include_pairs:
            for i, var1 in enumerate(all_digit_vars):
                for var2 in all_digit_vars[i + 1 :]:
                    target_variables_list.append([var1, var2])

        # Track raw_output
        target_variables_list.append(["raw_output"])

    print("Target variables to analyze:")
    for vars in target_variables_list:
        print(f"  {vars}")
    print()

    # Compute scores for each target variable group
    datasets_dict = {"random_cf": dataset}

    results_by_target = {}

    for target_vars in target_variables_list:
        var_name = "_".join(target_vars)
        print(f"Computing scores for {var_name}...")

        # NEW: Use compute_interchange_scores to add scoring
        results = compute_interchange_scores(
            raw_results=raw_results,
            causal_model=causal_model,
            datasets=datasets_dict,
            target_variables_list=[target_vars],
            checker=checker,
        )

        results_by_target[var_name] = results
        print("  ✓ Scores computed")

    print()
    print("=" * 70)
    print("✓ All scores computed")
    print("=" * 70)
    print()

    # Generate visualizations for each target variable
    print("Generating visualizations...")
    print()

    # Use static method - no need to load the full model!
    output_dir = results_path.parent

    for var_name, results in results_by_target.items():
        # Reconstruct target_vars from var_name (was joined with "_")
        # var_name could be "digit_0_0" or "raw_output"
        # We need to get back the original list like ["digit_0_0"]
        target_vars = [var_name]  # Keep as single element list

        print(f"Generating outputs for '{var_name}'...")

        # Generate heatmap using static method (no model loading!)
        heatmap_path = str(output_dir / f"heatmap_{var_name}")
        print(f"    Saving to: {heatmap_path}")
        PatchResidualStream.plot_heatmaps_from_results(
            results=results,
            target_variables=target_vars,
            layers=metadata["layers"],
            token_position_ids=metadata["token_positions"],
            save_path=heatmap_path,
        )
        print("  ✓ Heatmap saved")

        # Save text analysis
        text_path = output_dir / f"analysis_{var_name}.txt"
        with open(text_path, "w") as f:
            f.write(f"Analysis for variable: {var_name}\n")
            f.write("=" * 70 + "\n\n")

            # Extract key statistics from results
            if "dataset" in results and "random_cf" in results["dataset"]:
                dataset_results = results["dataset"]["random_cf"]
                if "target_variables" in dataset_results:
                    var_key = str(target_vars)
                    if var_key in dataset_results["target_variables"]:
                        var_data = dataset_results["target_variables"][var_key]

                        f.write(
                            "Interchange Intervention Accuracy by Layer and Position:\n"
                        )
                        f.write("-" * 70 + "\n")

                        # Get all layer-position keys and sort them
                        if "accuracy" in var_data:
                            keys = sorted(var_data["accuracy"].keys())
                            for key in keys:
                                acc = var_data["accuracy"][key]
                                f.write(f"  {key}: {acc:.3f}\n")

                        f.write("\n")
        print("  ✓ Analysis saved")
        print()

    print("=" * 70)
    print("✓ All visualizations complete!")
    print("=" * 70)
    print()
    print(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
