"""
Validation test suite for causal model validation.

This script validates:
1. Template variable uniqueness
2. Sample outputs inspection
3. Correctness of all variables
4. Distribution analysis
5. Distinguishability across datasets
6. Dataset justifications
7. Root prefix uniqueness (all answers share same prefix)
"""

import argparse
import re
from collections import defaultdict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..causal_models import causal_model
from ..counterfactuals import (
    sample_valid_input,
    COUNTERFACTUAL_GENERATORS,
)
from ..templates import TEMPLATES
from ..test_config import (
    DISTINGUISHABILITY_TESTS,
    DATASET_JUSTIFICATIONS,
    check_correctness,
    extract_distribution_variables,
    get_expected_ranges,
)
from causalab.causal.causal_utils import generate_counterfactual_samples

ALL_VARIABLES = causal_model.variables

TEST_CONFIG = {
    "correctness_samples": 1000,
    "distribution_samples": 1000,
    "token_length_sanity_samples": 50,
    "distinguishability_samples": 200,
}


# =============================================================================
# Tests
# =============================================================================


def run_all_tests(tokenizer: PreTrainedTokenizerBase):
    """Run all validation tests."""
    print("=" * 80)
    print("CAUSAL MODEL VALIDATION TEST SUITE")
    print("=" * 80)

    # =========================================================================
    # Test 1: Template Variable Uniqueness
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: TEMPLATE VARIABLE UNIQUENESS")
    print("=" * 80)

    templates = TEMPLATES

    if templates:
        print(f"\nChecking {len(templates)} template(s) for duplicate variables...\n")

        all_passed = True
        for i, template in enumerate(templates):
            # Find all {variable} occurrences in the template
            placeholders = re.findall(r"\{([^}]+)\}", template)

            # Count occurrences of each variable
            var_counts = {}
            for var in placeholders:
                var_counts[var] = var_counts.get(var, 0) + 1

            # Check for duplicates
            duplicates = {var: count for var, count in var_counts.items() if count > 1}

            if duplicates:
                print(f"  ✗ Template {i}: DUPLICATE VARIABLES FOUND")
                print(f"    Template: {template}")
                for var, count in duplicates.items():
                    print(f"    Variable '{var}' appears {count} times")
                all_passed = False
            else:
                print(f"  ✓ Template {i}: All variables unique")

        if all_passed:
            print("\n✓ All templates pass uniqueness check")
        else:
            print(
                "\n✗ BLOCKER: Templates have duplicate variables. If duplicates are absolutely necessary, add a dummy intermediate variable for each duplicate occurrence."
            )
    else:
        print("  Note: No templates found")

    # =========================================================================
    # Test 2: Sample Outputs Inspection
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: SAMPLE OUTPUTS INSPECTION")
    print("=" * 80)
    print("\nGenerating sample outputs to verify they aren't surprising...\n")

    n_samples = 5
    for i in range(n_samples):
        full_setting = (
            sample_valid_input()
        )  # Returns CausalTrace with all values computed

        print(f"Sample {i + 1}:")
        for var_name in ALL_VARIABLES:
            if var_name in full_setting:
                print(f"  {var_name}: {full_setting[var_name]}")
        print()

    print("✓ Review the samples above to verify outputs look reasonable")

    # =========================================================================
    # Test 3: Correctness
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: CORRECTNESS FOR ALL VARIABLES")
    print("=" * 80)

    n_tests = TEST_CONFIG["correctness_samples"]
    print(f"\nTesting {n_tests} samples...")

    errors = []
    all_results = []

    for i in range(n_tests):
        full_setting = (
            sample_valid_input()
        )  # Returns CausalTrace with all values computed
        is_correct, error_messages = check_correctness(full_setting)

        all_results.append({"correct": is_correct, "setting": full_setting})
        if not is_correct:
            errors.append((i, error_messages))

    accuracy = sum(1 for r in all_results if r["correct"]) / len(all_results)
    print(f"Accuracy: {accuracy:.2%} ({len(errors)} errors)")

    if errors:
        print("\nErrors (first 5):")
        for idx, msgs in errors[:5]:
            print(f"  Sample {idx}: {msgs}")

    print(
        f"\n{'✓ PASS' if len(errors) == 0 else '✗ FAIL'}: Model produces correct outputs"
    )

    # =========================================================================
    # Test 4: Distribution
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: DISTRIBUTION ANALYSIS")
    print("=" * 80)

    expected_ranges = get_expected_ranges()
    all_distributions = defaultdict(lambda: defaultdict(int))

    for result in all_results:
        vars_to_analyze = extract_distribution_variables(result["setting"])
        for var_name, value in vars_to_analyze.items():
            all_distributions[var_name][value] += 1

    for var_name, distribution in all_distributions.items():
        print(f"\n{var_name}:")
        for value in sorted(distribution.keys(), key=str):
            count = distribution[value]
            pct = count / len(all_results) * 100
            bar = "█" * int(pct / 2)
            print(f"  {value}: {count:4d} ({pct:5.1f}%) {bar}")

        if var_name in expected_ranges:
            expected = expected_ranges[var_name]
            coverage = sum(1 for v in expected if distribution[v] > 0)
            total = len(list(expected))
            print(f"  Coverage: {coverage}/{total}")

    # =========================================================================
    # Test 5: Distinguishability
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 5: DISTINGUISHABILITY ACROSS DATASETS")
    print("=" * 80)

    n_samples = TEST_CONFIG["distinguishability_samples"]
    print(f"\nGenerating datasets ({n_samples} examples each)...")

    datasets = {}
    for dataset_name, generator in COUNTERFACTUAL_GENERATORS.items():
        datasets[dataset_name] = generate_counterfactual_samples(n_samples, generator)

    for dataset_name, dataset in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 80}")

        tests = DISTINGUISHABILITY_TESTS.get(dataset_name, [])

        if not tests:
            print("⚠ No distinguishability tests specified, skipping...")
            continue

        for i, test in enumerate(tests, 1):
            vars1_raw = test["variables1"]
            vars2_raw = test["variables2"]
            description = test.get("description", "")
            expected = test.get("expected", "unknown")

            # Ensure vars are lists
            vars1: list[str] = (
                [vars1_raw] if isinstance(vars1_raw, str) else (vars1_raw or [])
            )
            vars2: list[str] | None = (
                [vars2_raw] if isinstance(vars2_raw, str) else vars2_raw
            )

            # Format variable names for display
            vars1_str = str(vars1) if vars1 else "None"
            vars2_str = str(vars2) if vars2 else "None"

            print(f"\nTest {i}: {description}")
            print(f"  Variables 1: {vars1_str}")
            print(f"  Variables 2: {vars2_str}")
            print(f"  Expected: {expected} distinguishability")

            # Run the distinguishability test
            result_dict = causal_model.can_distinguish_with_dataset(
                examples=dataset,
                target_variables1=vars1,
                target_variables2=vars2,
            )
            result = result_dict["proportion"]

            # Interpret result
            if result >= 0.8:
                result_str = "HIGH"
            elif result >= 0.4:
                result_str = "MEDIUM"
            else:
                result_str = "LOW"

            expected_str = str(expected)
            status = (
                "✓"
                if (expected_str.upper() == result_str or expected_str == "unknown")
                else "✗"
            )
            print(f"  Result: {result:.1%} ({result_str}) {status}")

    # =========================================================================
    # Test 6: Dataset Justifications
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 6: DATASET JUSTIFICATIONS")
    print("=" * 80)

    for dataset_name in COUNTERFACTUAL_GENERATORS.keys():
        print(f"\n{dataset_name}:")

        if dataset_name in DATASET_JUSTIFICATIONS:
            print(f"  {DATASET_JUSTIFICATIONS[dataset_name]}")
        else:
            print("  ⚠ No justification provided")

        if dataset_name in DISTINGUISHABILITY_TESTS:
            num_tests = len(DISTINGUISHABILITY_TESTS[dataset_name])
            print(f"  Distinguishability tests: {num_tests}")

    coverage = sum(
        1 for name in COUNTERFACTUAL_GENERATORS.keys() if name in DATASET_JUSTIFICATIONS
    )
    total = len(COUNTERFACTUAL_GENERATORS)
    print(
        f"\n{'✓ PASS' if coverage == total else '⚠ WARNING'}: {coverage}/{total} datasets have justifications"
    )

    # =========================================================================
    # Test 7: Root Prefix Uniqueness
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 7: TOKEN PREFIX UNIQUENESS")
    print("=" * 80)

    n_samples = TEST_CONFIG["correctness_samples"]
    print(f"\nChecking if all {n_samples} answers share a common token prefix...")

    # Collect all raw_output token sequences
    token_sequences = []
    for result in all_results:
        raw_output = result["setting"].get("raw_output", "")
        if raw_output:
            tokens = tokenizer.encode(raw_output, add_special_tokens=False)
            if tokens:
                token_sequences.append(tokens)

    if not token_sequences:
        print("  ⚠ No non-empty outputs found to analyze")
    else:
        # Find maximum shared prefix
        shared_prefix = []
        if token_sequences:
            # Start with the first sequence as reference
            reference = token_sequences[0]
            for i in range(len(reference)):
                token = reference[i]
                # Check if all sequences have this token at position i
                if all(len(seq) > i and seq[i] == token for seq in token_sequences):
                    shared_prefix.append(token)
                else:
                    break

        if shared_prefix:
            prefix_str = tokenizer.decode(shared_prefix)
            print(
                f"\n  ⚠ BLOCKER: All answers share a common prefix of {len(shared_prefix)} token(s):"
            )
            print(f"    Prefix: '{prefix_str}'")
            print(f"    Token IDs: {shared_prefix}")
            print(
                "\n  → This is not a string prefix, it is a token prefix. Answers should not share a common token prefix. Add this to the end of the template."
            )
        else:
            print("\n  ✓ PASS: Answers have no common token prefix")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run causal model validation tests")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name or path (for tokenizer)"
    )
    args = parser.parse_args()

    print(f"Loading tokenizer for: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("✓ Tokenizer loaded\n")

    run_all_tests(tokenizer)
