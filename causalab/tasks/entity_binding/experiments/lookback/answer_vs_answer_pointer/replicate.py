#!/usr/bin/env python
"""
Answer vs Answer Pointer Experiment

This experiment tests whether interchange interventions can distinguish between
the semantic answer (raw_output) and the positional answer pointer (positional_answer).

Experiment setup:
- Query: food (index 1) - "Who loves {food}?"
- Answer: person (index 0) - expects a person name
- change_answer=True: counterfactual has a different person as the answer
- Config: love (2 groups, 2 entities per group: person, food)

The key finding is that:
- positional_answer information peaks in middle layers (23-26)
- raw_output (semantic answer) emerges in final layers (32-35)

Usage:
    python -m tasks.entity_binding.experiments.lookback.answer_vs_answer_pointer.replicate
    python -m tasks.entity_binding.experiments.lookback.answer_vs_answer_pointer.replicate --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
from pathlib import Path

import torch

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.causal_utils import (
    generate_counterfactual_samples,
    save_counterfactual_examples,
)
from causalab.experiments.filter import filter_dataset
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.experiments.jobs.interchange_score_grid import (
    run_interchange_score_heatmap,
)
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_position_builder import build_token_position_factories

from causalab.tasks.entity_binding.counterfactuals import swap_query_group
from causalab.tasks.entity_binding.causal_models import (
    create_positional_entity_causal_model,
)
from causalab.tasks.entity_binding.experiment_config import get_task_config, get_checker


# =============================================================================
# Experiment Configuration (hardcoded for reproducibility)
# =============================================================================

EXPERIMENT_CONFIG = {
    "config_name": "love",
    "query_indices": (1,),  # Query food
    "answer_index": 0,  # Answer is person
    "change_answer": True,  # Counterfactual has different answer
    "dataset_size": 128,
    "batch_size": 32,
    "target_variables": ["positional_answer", "raw_output"],
}

# Directory where this script lives - outputs go here
SCRIPT_DIR = Path(__file__).parent


def create_token_positions(pipeline: LMPipeline, task_config):
    """Create token positions for entity binding task."""
    template = task_config.build_mega_template(
        active_groups=task_config.max_groups,
        query_indices=EXPERIMENT_CONFIG["query_indices"],
        answer_index=EXPERIMENT_CONFIG["answer_index"],
    )

    token_position_specs = {
        "last_token": {"type": "index", "position": -1},
    }

    factories = build_token_position_factories(token_position_specs, template)
    token_positions = {}
    for name, factory in factories.items():
        token_positions[name] = factory(pipeline)

    return token_positions


def run_experiment(model_name: str, verbose: bool = True):
    """
    Run the complete answer vs answer pointer experiment.

    Args:
        model_name: HuggingFace model name
        verbose: Print progress information
    """
    # Setup paths
    datasets_dir = SCRIPT_DIR / "datasets"
    results_dir = SCRIPT_DIR / "results"

    datasets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("ANSWER VS ANSWER POINTER EXPERIMENT")
        print("=" * 70)
        print(f"\nModel: {model_name}")
        print(f"Query indices: {EXPERIMENT_CONFIG['query_indices']} (food)")
        print(f"Answer index: {EXPERIMENT_CONFIG['answer_index']} (person)")
        print(f"Change answer: {EXPERIMENT_CONFIG['change_answer']}")
        print(f"Dataset size: {EXPERIMENT_CONFIG['dataset_size']}")
        print()

    # =========================================================================
    # Setup
    # =========================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if verbose:
        print(f"Device: {device}")
        print(f"Dtype: {dtype}")
        print()

    # Create task config with fixed query/answer indices
    task_config = get_task_config(EXPERIMENT_CONFIG["config_name"])
    task_config.fixed_query_indices = EXPERIMENT_CONFIG["query_indices"]
    task_config.fixed_answer_index = EXPERIMENT_CONFIG["answer_index"]

    # Create causal model
    causal_model = create_positional_entity_causal_model(task_config)

    if verbose:
        print(f"Task config: {EXPERIMENT_CONFIG['config_name']}")
        print(f"Causal model: {causal_model.id}")
        print()

    # =========================================================================
    # Load Model
    # =========================================================================
    if verbose:
        print("=" * 70)
        print("LOADING MODEL")
        print("=" * 70)

    pipeline = LMPipeline(
        model_name,
        max_new_tokens=5,
        device=device,
        dtype=dtype,
        max_length=256,
    )
    pipeline.tokenizer.padding_side = "left"

    num_layers = pipeline.model.config.num_hidden_layers

    if verbose:
        print(f"Model loaded: {model_name}")
        print(f"Layers: {num_layers}")
        print()

    # =========================================================================
    # Generate Dataset
    # =========================================================================
    if verbose:
        print("=" * 70)
        print("GENERATING COUNTERFACTUAL DATASET")
        print("=" * 70)

    def generator():
        return swap_query_group(
            task_config, change_answer=EXPERIMENT_CONFIG["change_answer"]
        )

    dataset: list[CounterfactualExample] = generate_counterfactual_samples(
        EXPERIMENT_CONFIG["dataset_size"], generator
    )

    if verbose:
        print(f"Generated {len(dataset)} counterfactual pairs")

        # Show example
        example = dataset[0]
        input_sample = example["input"]
        cf_sample = example["counterfactual_inputs"][0]

        if "raw_input" in input_sample:
            print("\nExample:")
            print(f"  Input: {input_sample['raw_input']}")
        if "raw_input" in cf_sample:
            print(f"  Counterfactual: {cf_sample['raw_input']}")
        print()

    # =========================================================================
    # Filter Dataset
    # =========================================================================
    if verbose:
        print("=" * 70)
        print("FILTERING DATASET")
        print("=" * 70)

    checker = get_checker()
    filtered_dataset = filter_dataset(
        dataset=dataset,
        pipeline=pipeline,
        causal_model=causal_model,
        metric=checker,
        batch_size=EXPERIMENT_CONFIG["batch_size"],
    )

    keep_rate = len(filtered_dataset) / len(dataset) * 100

    if verbose:
        print("\nFiltering results:")
        print(f"  Original: {len(dataset)} examples")
        print(f"  Filtered: {len(filtered_dataset)} examples")
        print(f"  Keep rate: {keep_rate:.1f}%")
        print()

    # Save filtered dataset
    cf_dataset_dir = datasets_dir / "swap_query_group"
    cf_dataset_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = cf_dataset_dir / "filtered_dataset.json"
    original_path = cf_dataset_dir / "original_dataset.json"

    save_counterfactual_examples(dataset, str(original_path))
    save_counterfactual_examples(filtered_dataset, str(filtered_path))

    if verbose:
        print(f"Saved datasets to: {datasets_dir}")
        print()

    # =========================================================================
    # Run Interchange Interventions
    # =========================================================================
    if verbose:
        print("=" * 70)
        print("RUNNING INTERCHANGE INTERVENTIONS")
        print("=" * 70)

    # Create token positions
    token_positions_dict = create_token_positions(pipeline, task_config)
    token_positions = list(token_positions_dict.values())

    # Build interchange targets for all layers
    layers = [-1] + list(range(num_layers))  # -1 is embeddings

    targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=layers,
        token_positions=token_positions,
        mode="one_target_per_unit",
    )

    if verbose:
        print(
            f"Analyzing {len(layers)} layers (embedding + {num_layers} transformer layers)"
        )
        print(f"Token positions: {[tp.id for tp in token_positions]}")
        print(f"Target variables: {EXPERIMENT_CONFIG['target_variables']}")
        print()

    # Run heatmap experiment
    # Convert target_variables list to tuple format expected by run_interchange_score_heatmap
    target_variable_groups = [tuple([v]) for v in EXPERIMENT_CONFIG["target_variables"]]
    result = run_interchange_score_heatmap(
        causal_model=causal_model,
        interchange_targets=targets,
        dataset_path=str(filtered_path),
        pipeline=pipeline,
        target_variable_groups=target_variable_groups,
        batch_size=EXPERIMENT_CONFIG["batch_size"],
        output_dir=str(results_dir),
        metric=checker,
        verbose=verbose,
    )

    if verbose:
        print()
        print("=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print(f"\nResults saved to: {results_dir}")
        print(f"Heatmaps saved to: {results_dir / 'heatmaps'}")

        # Print summary scores
        if "scores" in result:
            print("\nScore summary by variable:")
            for var_name, scores in result["scores"].items():
                avg_score = sum(scores.values()) / len(scores) if scores else 0
                print(f"  {var_name}: {avg_score:.3f} (avg)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the Answer vs Answer Pointer experiment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
