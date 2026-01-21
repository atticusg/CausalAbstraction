"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Test token positions across all three models.

This script tests token position functions with Llama, Gemma, and OLMo
to verify they handle different tokenization patterns correctly.
"""

import sys

sys.path.append("/mnt/polished-lake/home/atticus/CausalAbstraction")

import torch
from causalab.tasks.general_addition.config import create_two_number_two_digit_config
from causalab.tasks.general_addition.causal_models import (
    create_basic_addition_model,
    sample_valid_addition_input,
)
from causalab.tasks.general_addition.token_positions import create_token_positions
from causalab.neural.pipeline import LMPipeline


def check_model_quick(model_name: str):
    """Quick test with one sample for a model."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {model_name}")
    print(f"{'=' * 70}")

    # Create config
    config = create_two_number_two_digit_config()
    causal_model = create_basic_addition_model(config)

    # Load model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = LMPipeline(model_name, max_new_tokens=5, device=device, max_length=256)
    print(f"✓ Loaded on {device}")

    # Generate sample
    input_sample = sample_valid_addition_input(config, 2, 2)
    output = causal_model.new_trace(input_sample)
    print(f"\nPrompt: {output['raw_input']}")
    print(f"Answer: {output['raw_output']}")

    # Create and test token positions
    token_positions = create_token_positions(pipeline, 2, 2)

    # Show a few key highlights
    print(
        f"\ndigit_0_0: {token_positions['digit_0_0'].highlight_selected_token(output)}"
    )
    print(f"digit_1_0: {token_positions['digit_1_0'].highlight_selected_token(output)}")
    print(
        f"last_token: {token_positions['last_token'].highlight_selected_token(output)}"
    )

    print(f"\n✓ Token positions work for {model_name}")


def main():
    """Test all three models."""
    print("\n" + "=" * 70)
    print("TOKEN POSITION TESTS - ALL MODELS")
    print("=" * 70)

    models = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "google/gemma-2-9b",
        "allenai/OLMo-2-1124-13B",
    ]

    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Testing {model.split('/')[-1]}...")
        try:
            check_model_quick(model)
        except Exception as e:
            print(f"\n✗ Error with {model}: {e}")
            import traceback

            traceback.print_exc()
            return 1

    print("\n" + "=" * 70)
    print("✓ ALL MODELS TESTED SUCCESSFULLY")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
