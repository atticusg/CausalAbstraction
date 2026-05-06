"""
Validation tests that require loading a language model.

These tests verify:
1. Example forward pass through both LM and causal model
2. Token position validation and highlighting
3. Token alignment between raw_output and model-generated tokens

Run this before executing experiments to catch issues early.
"""

import argparse
import traceback

from causalab.neural.pipeline import LMPipeline

from ..causal_models import causal_model
from ..checker import checker
from ..config import MAX_TASK_TOKENS, MAX_NEW_TOKENS
from ..token_positions import create_token_positions
from ..templates import TEMPLATES

from causalab.neural.pipeline import resolve_device

DEVICE = resolve_device()


def test_example_forward_pass(pipeline: LMPipeline) -> bool:
    """
    Test 1: Run a single example through both the language model and causal model.
    Verify the checker function works correctly.
    """
    print("\n" + "=" * 80)
    print("TEST 1: EXAMPLE FORWARD PASS")
    print("=" * 80)

    print(
        "\nRunning a single example through both the language model and causal model..."
    )

    # Get a single example
    full_setting = causal_model.sample_input()
    example_input = full_setting["raw_input"]

    print(f"\nInput: {example_input}")

    # Run through language model
    lm_output = pipeline.generate([full_setting], output_scores=False)
    lm_answer = lm_output["string"]
    print(f"\nLanguage Model Output: '{lm_answer}'")

    # Run through causal model
    cm_answer = full_setting["raw_output"]
    print(f"Causal Model Output: '{cm_answer}'")

    # Use checker to verify
    match = checker(lm_output, full_setting["raw_output"])
    print(f"\nChecker Result: {match}")

    if match:
        print(
            "  ✓ Checker says match! If the two outputs are the same, then the checker is correct. Otherwise, the checker is incorrect!"
        )
    else:
        print(
            "  ✗ Checker says no match! If the two outputs are different, then the checker is correct. Otherwise, the checker is incorrect!"
        )

    return match


def test_token_positions(pipeline: LMPipeline) -> bool:
    """
    Test 2: Create token positions and verify they target the correct tokens.
    Highlights each position on example input for visual inspection.
    """
    print("\n" + "=" * 80)
    print("TEST 2: TESTING TOKEN POSITIONS")
    print("=" * 80)

    print("\nCreating token positions from declarative specs...")

    # Get template and create token positions
    template = TEMPLATES[0] if TEMPLATES else None
    if template is None:
        raise ValueError("No templates defined")

    token_position_factories = create_token_positions(pipeline, template)

    print(
        f"✓ Created {len(token_position_factories)} token positions: {list(token_position_factories.keys())}"
    )

    # Get an example to highlight
    full_setting = causal_model.sample_input()
    example_input = full_setting["raw_input"]

    print("\nHighlighting each token position on example input:")
    print(f"Input: {example_input}\n")

    for name, factory in token_position_factories.items():
        token_pos = factory(pipeline)
        highlighted = token_pos.highlight_selected_token(full_setting)
        print(f"{name}:")
        print(f"  {highlighted}")
        print()

    print("✓ Review the highlights above - do they target the correct tokens?")
    print("  If not, adjust your token_position_specs and re-run this test.")

    return True


def test_token_alignment(pipeline: LMPipeline) -> bool:
    """
    Test 3: Verify that raw_output tokenizes to the same tokens the model generates.

    A mismatch (e.g. " 13" vs "13") is invisible to .strip()-based checkers but
    breaks experiments that operate at the token level (loss functions, metrics, etc.).
    """
    print("\n" + "=" * 80)
    print("TEST 3: TOKEN ALIGNMENT")
    print("=" * 80)

    print("\nChecking that raw_output tokens match model's actual output tokens...")

    num_samples = 5
    all_match = True

    for i in range(num_samples):
        full_setting = causal_model.sample_input()
        raw_output = full_setting["raw_output"]

        # Get the model's actual output token IDs
        gen = pipeline.generate([full_setting])
        actual_ids = gen["sequences"][0].tolist()

        # Tokenize the expected raw_output
        expected_ids = pipeline.tokenizer.encode(raw_output, add_special_tokens=False)

        if actual_ids == expected_ids:
            print(f"\n  Example {i + 1}: ✓ tokens match")
            print(f"    Input:    {full_setting['raw_input']!r}")
            print(f"    Output:   {raw_output!r}")
            print(f"    Tokens:   {expected_ids}")
        else:
            all_match = False
            actual_tokens = [pipeline.tokenizer.decode([t]) for t in actual_ids]
            expected_tokens = [pipeline.tokenizer.decode([t]) for t in expected_ids]
            print(f"\n  Example {i + 1}: ✗ TOKEN MISMATCH")
            print(f"    Input:      {full_setting['raw_input']!r}")
            print(f"    raw_output: {raw_output!r}")
            print(f"    Expected:   {expected_ids} -> {expected_tokens}")
            print(f"    Actual:     {actual_ids} -> {actual_tokens}")

    if all_match:
        print("\n✓ All token alignments match.")
    else:
        print(
            "\n✗ Token mismatch detected! Token-level experiments will silently fail."
        )
        print(
            "  Common fix: if template ends with a space, remove leading space from raw_output."
        )

    return all_match


def main():
    parser = argparse.ArgumentParser(description="Run model-dependent validation tests")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--skip-forward-pass", action="store_true", help="Skip forward pass test"
    )
    parser.add_argument(
        "--skip-token-positions", action="store_true", help="Skip token position test"
    )
    parser.add_argument(
        "--skip-token-alignment", action="store_true", help="Skip token alignment test"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MODEL-DEPENDENT VALIDATION TESTS")
    print("=" * 80)
    print(f"\nModel: {args.model}")

    # Load Model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    pipeline = LMPipeline(
        args.model,
        max_new_tokens=MAX_NEW_TOKENS,
        device=DEVICE,
        max_length=MAX_TASK_TOKENS,
    )

    print(f"✓ Model loaded: {args.model}")
    print(f"  Device: {pipeline.model.device}")
    print(f"  Layers: {pipeline.model.config.num_hidden_layers}")

    # Run Tests
    results = {}

    if not args.skip_forward_pass:
        try:
            results["forward_pass"] = test_example_forward_pass(pipeline)
        except Exception as e:
            print(f"\n✗ Forward pass test failed with error: {e}")
            traceback.print_exc()
            results["forward_pass"] = False

    if not args.skip_token_positions:
        try:
            results["token_positions"] = test_token_positions(pipeline)
        except Exception as e:
            print(f"\n✗ Token position test failed with error: {e}")
            traceback.print_exc()
            results["token_positions"] = False

    if not args.skip_token_alignment:
        try:
            results["token_alignment"] = test_token_alignment(pipeline)
        except Exception as e:
            print(f"\n✗ Token alignment test failed with error: {e}")
            traceback.print_exc()
            results["token_alignment"] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed! Ready to run experiments.")
    else:
        print("\n✗ Some tests failed. Fix issues before running experiments.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
