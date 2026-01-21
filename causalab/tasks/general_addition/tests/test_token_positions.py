"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Test token positions for general addition task.

This script tests token position functions by loading a model and using
highlight_selected_token to visualize which tokens are selected.
"""

import sys

sys.path.append("/mnt/polished-lake/home/atticus/CausalAbstraction")

import torch
from causalab.tasks.general_addition.causal_models import (
    create_basic_addition_model,
    sample_valid_addition_input,
)
from causalab.tasks.general_addition.token_positions import create_token_positions
from causalab.neural.pipeline import LMPipeline


def check_token_positions_with_model(model_name: str, num_digits: int = 2):
    """
    Test token positions with a specific model.

    Args:
        model_name: Name of the model to test
        num_digits: Number of digits per number
    """
    print("\n" + "=" * 70)
    print(f"Testing Token Positions: {model_name}, {num_digits}-digit")
    print("=" * 70)

    # Create config and model
    if num_digits == 2:
        from causalab.tasks.general_addition.config import create_two_number_two_digit_config

        config = create_two_number_two_digit_config()
    elif num_digits == 3:
        from causalab.tasks.general_addition.config import create_two_number_three_digit_config

        config = create_two_number_three_digit_config()
    else:
        from causalab.tasks.general_addition.config import create_general_config

        config = create_general_config(2, num_digits)

    causal_model = create_basic_addition_model(config)

    # Load language model
    print(f"\nLoading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = LMPipeline(model_name, max_new_tokens=5, device=device, max_length=256)
    print(f"✓ Model loaded on {device}")

    # Generate a sample input
    print("\nGenerating sample input...")
    input_sample = sample_valid_addition_input(config, 2, num_digits)
    output = causal_model.new_trace(input_sample)

    print(f"\nPrompt: {output['raw_input']}")
    print(f"Answer: {output['raw_output']}")

    # Extract the numbers for reference
    print("\nInput numbers:")
    for k in range(2):
        digits = [str(input_sample[f"digit_{k}_{d}"]) for d in range(num_digits)]
        number = "".join(digits)
        print(f"  Number {k}: {number}")

    # Create token positions
    print("\nCreating token positions...")
    token_positions = create_token_positions(pipeline, 2, num_digits)
    print(f"✓ Created {len(token_positions)} token positions")

    # Test each token position
    print("\n" + "-" * 70)
    print("Token Position Highlights:")
    print("-" * 70)

    # Test digit positions
    for k in range(2):
        for d in range(num_digits):
            key = f"digit_{k}_{d}"
            tp = token_positions[key]
            highlighted = tp.highlight_selected_token(output)
            digit_value = input_sample[f"digit_{k}_{d}"]
            print(f"\n{key} (value={digit_value}):")
            print(f"  {highlighted}")

    # Test delimiter positions
    print("\ndelimiter_and:")
    print(f"  {token_positions['delimiter_and'].highlight_selected_token(output)}")

    print("\ndelimiter_is:")
    print(f"  {token_positions['delimiter_is'].highlight_selected_token(output)}")

    # Test last token
    print("\nlast_token:")
    print(f"  {token_positions['last_token'].highlight_selected_token(output)}")

    print("\n" + "=" * 70)
    print(f"✓ Token position test complete for {model_name}")
    print("=" * 70)


def main():
    """Run token position tests."""
    print("\n" + "=" * 70)
    print("GENERAL ADDITION - TOKEN POSITION TESTS")
    print("=" * 70)

    # Test with Llama on 2-digit addition
    print("\n[1/3] Testing Llama 3.1 8B with 2-digit addition...")
    try:
        check_token_positions_with_model(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", num_digits=2
        )
    except Exception as e:
        print(f"\n✗ Error testing Llama: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test with 3-digit addition
    print("\n[2/3] Testing Llama 3.1 8B with 3-digit addition...")
    try:
        check_token_positions_with_model(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", num_digits=3
        )
    except Exception as e:
        print(f"\n✗ Error testing 3-digit: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test with 4-digit addition
    print("\n[3/3] Testing Llama 3.1 8B with 4-digit addition...")
    try:
        check_token_positions_with_model(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", num_digits=4
        )
    except Exception as e:
        print(f"\n✗ Error testing 4-digit: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("✓ ALL TOKEN POSITION TESTS PASSED")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
