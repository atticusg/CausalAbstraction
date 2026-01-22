#!/usr/bin/env python
"""
Binding Address vs Payload Experiment

This experiment tests whether swapping the "address" (token position) of entity
bindings while keeping the same "payload" (entity values) can successfully
transfer binding information.

Experiment setup:
- Query: food (index 1) - "Who loves {food}?"
- Answer: person (index 0) - expects a person name
- change_answer=False: counterfactual has the SAME answer (just swapped positions)
- Config: love (2 groups, 2 entities per group: person, food)

With swap_query_group and change_answer=False:
- Original: "Pete loves pie and Ann loves cake. Who loves pie?" → "Pete"
- Counterfactual: "Ann loves cake and Pete loves pie. Who loves pie?" → "Pete"

Both person (e0) and food (e1) entities are swapped between groups:
- Original: g0_e0="Pete", g0_e1="pie", g1_e0="Ann", g1_e1="cake"
- Counterfactual: g0_e0="Ann", g0_e1="cake", g1_e0="Pete", g1_e1="pie"

We test cross-patching: swap entire binding group addresses to see if the payload transfers.
Token positions: original=[g0_e0, g0_e1, g1_e0, g1_e1], counterfactual=[g1_e0, g1_e1, g0_e0, g0_e1]
Target variables: All 4 positional_entity variables swapped between groups

Usage:
    python -m tasks.entity_binding.experiments.lookback.binding_address_payload.replicate
    python -m tasks.entity_binding.experiments.lookback.binding_address_payload.replicate --model meta-llama/Llama-3.2-1B-Instruct
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
from causalab.neural.token_position_builder import (
    TokenPosition,
    paired_token_position,
    combined_token_position,
)

from causalab.tasks.entity_binding.counterfactuals import swap_query_group
from causalab.tasks.entity_binding.causal_models import (
    create_positional_entity_causal_model,
)
from causalab.tasks.entity_binding.experiment_config import get_task_config, get_checker
from causalab.tasks.entity_binding.token_positions import get_entity_token_positions


# =============================================================================
# Experiment Configuration (hardcoded for reproducibility)
# =============================================================================

EXPERIMENT_CONFIG = {
    "config_name": "love",
    "query_indices": (1,),  # Query food
    "answer_index": 0,  # Answer is person
    "change_answer": False,  # Counterfactual has SAME answer (swapped positions)
    "dataset_size": 128,
    "batch_size": 32,
    "target_variables": [
        [
            "positional_entity_g0_e0<-positional_entity_g1_e0",
            "positional_entity_g0_e1<-positional_entity_g1_e1",
            "positional_entity_g1_e0<-positional_entity_g0_e0",
            "positional_entity_g1_e1<-positional_entity_g0_e1",
        ]
    ],
}

# Directory where this script lives - outputs go here
SCRIPT_DIR = Path(__file__).parent


def _fix_query_indices(input_sample):
    """Convert query_indices from list to tuple if needed (HuggingFace serialization fix)."""
    # Handle both CausalTrace objects and dicts (after deserialization)
    if hasattr(input_sample, "to_dict"):
        sample = input_sample.to_dict()
    else:
        sample = dict(input_sample)
    if isinstance(sample.get("query_indices"), list):
        sample["query_indices"] = tuple(sample["query_indices"])
    return sample


def create_token_positions(pipeline: LMPipeline, task_config):
    """
    Create token positions for the binding address/payload experiment.

    Creates a paired token position where:
    - Original: [g0_e0, g0_e1, g1_e0, g1_e1] (person0, food0, person1, food1)
    - Counterfactual: [g1_e0, g1_e1, g0_e0, g0_e1] (swapped entire groups)

    This enables cross-patching of entire binding group addresses.
    """

    # Create indexer functions for all 4 entity positions
    def make_indexer(group_idx, entity_idx):
        def indexer(input_sample, is_original=True):
            sample = _fix_query_indices(input_sample)
            return get_entity_token_positions(
                input_sample=sample,
                pipeline=pipeline,
                config=task_config,
                group_idx=group_idx,
                entity_idx=entity_idx,
                token_idx=-1,  # Last token
            )

        return indexer

    # Build individual token positions for all 4 entities
    g0_e0_last = TokenPosition(make_indexer(0, 0), pipeline, id="g0_e0_last")
    g0_e1_last = TokenPosition(make_indexer(0, 1), pipeline, id="g0_e1_last")
    g1_e0_last = TokenPosition(make_indexer(1, 0), pipeline, id="g1_e0_last")
    g1_e1_last = TokenPosition(make_indexer(1, 1), pipeline, id="g1_e1_last")

    # Create combined positions for original and counterfactual
    # Original order: [g0_e0, g0_e1, g1_e0, g1_e1] (group 0 first, then group 1)
    original_combined = combined_token_position(
        [g0_e0_last, g0_e1_last, g1_e0_last, g1_e1_last], id="g0_g1"
    )

    # Counterfactual order: [g1_e0, g1_e1, g0_e0, g0_e1] (group 1 first, then group 0 - swapped)
    counterfactual_combined = combined_token_position(
        [g1_e0_last, g1_e1_last, g0_e0_last, g0_e1_last], id="g1_g0"
    )

    # Create paired position that uses different orders for original vs counterfactual
    paired = paired_token_position(
        original_combined, counterfactual_combined, id="group_swap"
    )

    return {"group_swap": paired}


def run_experiment(model_name: str, verbose: bool = True):
    """
    Run the complete binding address vs payload experiment.

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
        print("BINDING ADDRESS VS PAYLOAD EXPERIMENT")
        print("=" * 70)
        print(f"\nModel: {model_name}")
        print(f"Query indices: {EXPERIMENT_CONFIG['query_indices']} (food)")
        print(f"Answer index: {EXPERIMENT_CONFIG['answer_index']} (person)")
        print(f"Change answer: {EXPERIMENT_CONFIG['change_answer']}")
        print(f"Dataset size: {EXPERIMENT_CONFIG['dataset_size']}")
        print(f"Target variables: {EXPERIMENT_CONFIG['target_variables']}")
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
    # Convert target_variables to tuple format expected by run_interchange_score_heatmap
    target_variable_groups = [
        tuple(group) for group in EXPERIMENT_CONFIG["target_variables"]
    ]
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
        description="Run the Binding Address vs Payload experiment"
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
