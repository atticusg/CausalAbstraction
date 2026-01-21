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
    python visualize_results.py --results PATH [--target-vars VARS]
    python visualize_results.py --results tasks/entity_binding/results/love/raw_results.pkl
    python visualize_results.py --results ... --target-vars query_entity positional_query_group
    python visualize_results.py --results ... --test  # Test mode
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

from causalab.tasks.entity_binding.experiment_config import (
    get_task_config,
    get_causal_model,
    get_checker,
    get_default_target_variables,
)
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
        help="Target variables to analyze (default: config defaults)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: same as results directory)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode (for compatibility, no special behavior)",
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
    print(f"  Config: {metadata.get('config_name', 'unknown')}")
    print(f"  Model: {metadata.get('model', 'unknown')}")
    print(f"  Experiment ID: {metadata.get('experiment_id', 'unknown')}")
    if args.test:
        print("  Test mode: yes")
    print()

    # Load raw results
    print("Loading raw results...")
    with open(results_path, "rb") as f:
        raw_results = pickle.load(f)
    print("✓ Raw results loaded")
    print()

    # Get configuration
    config_name = metadata.get("config_name")
    if not config_name:
        print("Error: config_name not found in metadata")
        return 1

    try:
        config = get_task_config(config_name)
        print(f"Task configuration: {config_name}")
    except ValueError as e:
        print(f"Error loading config: {e}")
        return 1

    # Create causal model
    causal_model_type = metadata.get("causal_model_type", "positional")
    causal_model = get_causal_model(config, causal_model_type)
    print(f"Causal model: {causal_model.id}")
    print()

    # Load dataset
    dataset_path = metadata.get("dataset_path")
    if not dataset_path:
        print("Error: dataset_path not found in metadata")
        print(
            "Please ensure the original experiment was run with updated run_interventions.py"
        )
        return 1

    print(f"Loading dataset from {dataset_path}...")
    try:
        hf_dataset = load_from_disk(dataset_path)
        dataset: list[CounterfactualExample] = list(hf_dataset)  # type: ignore[assignment]
        print(f"✓ Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print(f"Dataset path from metadata: {dataset_path}")
        return 1
    print()

    # Get checker function
    checker = get_checker()

    # Determine target variables to analyze
    if args.target_vars is not None:
        # User specified variables - wrap each in a list
        target_variables_list = [[var] for var in args.target_vars]
    else:
        # Use defaults from config
        try:
            target_variables_list = get_default_target_variables(config_name)
        except ValueError as e:
            print(f"Error getting default target variables: {e}")
            return 1

    print("Target variables to analyze:")
    for vars in target_variables_list:
        print(f"  {vars}")
    print()

    # Compute scores for each target variable group
    datasets_dict = {"dataset": dataset}

    results_by_target = {}

    for target_vars in target_variables_list:
        var_name = "_".join(target_vars) if len(target_vars) > 1 else target_vars[0]
        print(f"Computing scores for {var_name}...")

        # Compute interchange scores
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

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = results_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save processed results with scores
    results_path_scored = output_dir / "results_with_scores.pkl"
    print(f"Saving results with scores to {results_path_scored}...")
    with open(results_path_scored, "wb") as f:
        # Save the first result (they all have the same structure, just different target_variables)
        # If we want all results, we'd need to merge them
        first_result = next(iter(results_by_target.values()))
        pickle.dump(first_result, f)
    print("✓ Results with scores saved")
    print()

    # Generate visualizations for each target variable
    print("Generating visualizations...")
    print()

    # Create output directories
    heatmaps_dir = output_dir / "heatmaps"
    analysis_dir = output_dir / "analysis"
    heatmaps_dir.mkdir(exist_ok=True)
    analysis_dir.mkdir(exist_ok=True)

    # Get layers and token position IDs from metadata (no need to load pipeline!)
    layers = metadata.get("layers", [0])
    token_position_ids = metadata.get("token_positions", ["last_token"])

    try:
        for var_name, results in results_by_target.items():
            # Reconstruct target_vars from the results
            # The var_name is joined with "_", but we need the original list
            # Check if this is an arrow syntax variable or regular variable
            if any(arrow in var_name for arrow in ["<-", "->"]):
                # Arrow syntax - don't split
                target_vars = [var_name]
            else:
                # Regular variable - get from results_by_target key
                # Actually, we should get it from the original target_variables_list
                # Find the matching entry
                target_vars = None
                for tv_list in target_variables_list:
                    tv_name = "_".join(tv_list) if len(tv_list) > 1 else tv_list[0]
                    if tv_name == var_name:
                        target_vars = tv_list
                        break
                if target_vars is None:
                    target_vars = [var_name]  # Fallback

            var_name_safe = var_name.replace("<-", "_from_").replace(">", "_to_")

            print(f"Generating outputs for '{var_name}'...")

            # Can distinguish analysis
            if "dataset" in datasets_dict:
                print(f"  Can distinguish {target_vars} from no intervention:")
                causal_model.can_distinguish_with_dataset(
                    datasets_dict["dataset"], target_vars, None
                )

                if target_vars != ["raw_output"]:
                    print(f"  Can distinguish {target_vars} from raw_output:")
                    causal_model.can_distinguish_with_dataset(
                        datasets_dict["dataset"], target_vars, ["raw_output"]
                    )

            # Generate heatmap using NEW static method (no pipeline needed!)
            try:
                PatchResidualStream.plot_heatmaps_from_results(
                    results=results,
                    target_variables=target_vars,
                    layers=layers,
                    token_position_ids=token_position_ids,
                    save_path=str(heatmaps_dir / f"heatmap_{var_name_safe}"),
                )
                print("  ✓ Heatmap saved")
            except Exception as e:
                print(f"  ✗ Error generating heatmap: {e}")

            # Save text analysis
            text_path = analysis_dir / f"analysis_{var_name_safe}.txt"
            with open(text_path, "w") as f:
                f.write(f"Analysis for variable: {var_name}\n")
                f.write("=" * 70 + "\n\n")

                # Extract key statistics from results
                if "dataset" in results and "dataset" in results["dataset"]:
                    dataset_results = results["dataset"]["dataset"]
                    if "target_variables" in dataset_results:
                        var_key = str(target_vars)
                        if var_key in dataset_results["target_variables"]:
                            var_data = dataset_results["target_variables"][var_key]

                            f.write("Interchange Intervention Accuracy by Layer:\n")
                            f.write("-" * 70 + "\n")

                            layers = metadata.get("layers", [])
                            for layer in layers:
                                key = f"Layer-{layer}"
                                if key in var_data.get("accuracy", {}):
                                    acc = var_data["accuracy"][key]
                                    f.write(f"  Layer {layer:3d}: {acc:.3f}\n")

                            f.write("\n" + "=" * 70 + "\n")
                            f.write("Summary Statistics:\n")
                            f.write("-" * 70 + "\n")

                            # Find best layer
                            best_acc = 0.0
                            best_layer = None
                            for layer in layers:
                                key = f"Layer-{layer}"
                                if key in var_data.get("accuracy", {}):
                                    acc = var_data["accuracy"][key]
                                    if acc > best_acc:
                                        best_acc = acc
                                        best_layer = layer

                            if best_layer is not None:
                                f.write(f"  Best layer: {best_layer}\n")
                                f.write(f"  Best accuracy: {best_acc:.3f}\n")

            print("  ✓ Analysis saved")
            print()

    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")
        print(
            "Scores have been computed and saved, but visualization generation failed"
        )

    print("=" * 70)
    print("✓ All visualizations complete!")
    print("=" * 70)
    print()
    print(f"Results saved to: {output_dir}")
    print("  - Results with scores: results_with_scores.pkl")
    print("  - Heatmaps: heatmaps/")
    print("  - Text analysis: analysis/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
