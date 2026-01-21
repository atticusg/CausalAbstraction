#!/usr/bin/env -S uv run python
"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Test script to determine optimal batch size for GPU memory.

This script tests increasing batch sizes until we hit OOM (Out of Memory),
helping us find the maximum batch size that fits on the GPU.
"""

import sys

sys.path.append("/mnt/polished-lake/home/atticus/CausalAbstraction")

import torch
import gc
from causalab.tasks.general_addition.config import create_two_number_two_digit_config
from causalab.tasks.general_addition.counterfactuals import random_counterfactual
from causalab.neural.pipeline import LMPipeline
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.causal_utils import generate_counterfactual_samples
from causalab.tasks.general_addition.experiments.tokenization_config import (
    get_all_number_token_indices,
    get_tokens_per_number,
)
from causalab.neural.token_position_builder import TokenPosition, get_last_token_index


def check_batch_size(model_name, batch_size, num_examples=64):
    """Test if a batch size fits in GPU memory."""
    print(f"\nTesting batch_size={batch_size}...")

    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Create config
        config = create_two_number_two_digit_config()

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = LMPipeline(
            model_name, max_new_tokens=10, device=device, max_length=64
        )

        # Generate small dataset
        dataset: list[CounterfactualExample] = generate_counterfactual_samples(
            num_examples, lambda: random_counterfactual(config, 2, 2)
        )

        # Create token positions
        tokens_per_number = get_tokens_per_number(model_name, 2)
        token_positions_dict = {}
        for k in range(2):
            for tok_idx in range(tokens_per_number):

                def make_indexer(addend_idx, token_idx):
                    def indexer(input_sample):
                        return get_all_number_token_indices(
                            input_sample, pipeline, addend_idx, 2, model_name
                        )[token_idx : token_idx + 1]

                    return indexer

                token_positions_dict[f"number_{k}_tok{tok_idx}"] = TokenPosition(
                    make_indexer(k, tok_idx), pipeline, id=f"number_{k}_tok{tok_idx}"
                )
        token_positions_dict["last_token"] = TokenPosition(
            lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token"
        )
        token_positions = list(token_positions_dict.values())

        # Create experiment with limited layers for speed
        layers = [-1, 0, 1]  # Just test a few layers
        experiment = PatchResidualStream(
            pipeline, layers, token_positions, config={"batch_size": batch_size}
        )

        # Run intervention on subset
        datasets_dict = {"test": dataset}
        target_variables_list = [["raw_output"]]  # Just test one variable

        print(f"  Running intervention with batch_size={batch_size}...")
        experiment.perform_interventions(
            datasets_dict, verbose=False, target_variables_list=target_variables_list
        )

        # Check memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            print(f"  ✓ Success! Peak GPU memory: {memory_used:.2f} GB")
            return True, memory_used
        else:
            print("  ✓ Success! (CPU mode)")
            return True, 0

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ✗ OOM: Out of memory with batch_size={batch_size}")
            return False, 0
        else:
            print(f"  ✗ Error: {e}")
            raise
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def find_optimal_batch_size(model_name):
    """Binary search to find optimal batch size."""
    print(f"\n{'=' * 70}")
    print(f"Finding optimal batch size for: {model_name}")
    print(f"{'=' * 70}")

    # Test increasing batch sizes
    batch_sizes_to_test = [8, 16, 32, 64, 128]

    max_working = None
    max_memory = 0

    for batch_size in batch_sizes_to_test:
        success, memory = check_batch_size(model_name, batch_size)
        if success:
            max_working = batch_size
            max_memory = memory
        else:
            print(f"\n  Stopping at batch_size={batch_size} (OOM)")
            break

    print(f"\n{'=' * 70}")
    if max_working:
        print(f"✓ Maximum working batch size: {max_working}")
        print(f"  Peak GPU memory: {max_memory:.2f} GB")
    else:
        print("✗ Even batch_size=8 caused OOM")
    print(f"{'=' * 70}")

    return max_working


def main():
    """Test batch sizes for all three models."""
    print("\n" + "=" * 70)
    print("BATCH SIZE OPTIMIZATION TEST")
    print("=" * 70)

    models = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "google/gemma-2-9b",
        "allenai/OLMo-2-1124-13B",
    ]

    results = {}

    for model in models:
        model_short = model.split("/")[-1]
        try:
            optimal = find_optimal_batch_size(model)
            results[model_short] = optimal
        except Exception as e:
            print(f"\n✗ Error testing {model_short}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n\n" + "=" * 70)
    print("RECOMMENDED BATCH SIZES")
    print("=" * 70)
    for model_short, batch_size in results.items():
        print(f"{model_short}: {batch_size}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
