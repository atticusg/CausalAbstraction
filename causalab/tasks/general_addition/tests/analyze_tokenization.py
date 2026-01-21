"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Analyze tokenization patterns for each model and digit configuration.

This script determines how many tokens each model uses for numbers
with 2, 3, and 4 digits. This information is critical for setting up
experiments correctly.
"""

import sys
sys.path.append('/mnt/polished-lake/home/atticus/CausalAbstraction')

import torch
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_position_builder import get_substring_token_ids


def analyze_model_tokenization(model_name: str):
    """
    Analyze how a model tokenizes numbers of different lengths.

    Args:
        model_name: Name of the model to analyze
    """
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = LMPipeline(
        model_name,
        max_new_tokens=5,
        device=device,
        max_length=256
    )

    # Test different number lengths and values
    test_cases = [
        # 2-digit numbers
        ("2-digit", [
            ("The sum of 23 and 45 is", "23", "45"),
            ("The sum of 9 and 78 is", "9", "78"),
            ("The sum of 99 and 11 is", "99", "11"),
        ]),
        # 3-digit numbers
        ("3-digit", [
            ("The sum of 123 and 456 is", "123", "456"),
            ("The sum of 999 and 100 is", "999", "100"),
            ("The sum of 734 and 262 is", "734", "262"),
        ]),
        # 4-digit numbers
        ("4-digit", [
            ("The sum of 1234 and 5678 is", "1234", "5678"),
            ("The sum of 9999 and 1000 is", "9999", "1000"),
            ("The sum of 6455 and 5162 is", "6455", "5162"),
        ]),
    ]

    results = {}

    for digit_count, cases in test_cases:
        print(f"\n{digit_count} numbers:")
        print("-" * 50)

        token_counts = []

        for prompt, num1, num2 in cases:
            # Get token counts for each number
            tokens1 = get_substring_token_ids(prompt, num1, pipeline, add_special_tokens=False)
            tokens2 = get_substring_token_ids(prompt, num2, pipeline, add_special_tokens=False)

            # Decode tokens to see what they are
            decoded1 = [pipeline.tokenizer.decode([tid]) for tid in tokens1]
            decoded2 = [pipeline.tokenizer.decode([tid]) for tid in tokens2]

            print(f"  '{num1}' → {len(tokens1)} token(s): {decoded1}")
            print(f"  '{num2}' → {len(tokens2)} token(s): {decoded2}")

            token_counts.append((len(tokens1), len(tokens2)))

        # Determine pattern
        min_tokens = min(min(tc) for tc in token_counts)
        max_tokens = max(max(tc) for tc in token_counts)

        if min_tokens == max_tokens:
            pattern = f"Consistent: {min_tokens} token(s) per number"
        else:
            pattern = f"Variable: {min_tokens}-{max_tokens} token(s) per number"

        print(f"\n  Pattern: {pattern}")
        results[digit_count] = {
            'min_tokens': min_tokens,
            'max_tokens': max_tokens,
            'pattern': pattern
        }

    return results


def main():
    """Analyze all three models."""
    print("\n" + "="*70)
    print("TOKENIZATION ANALYSIS FOR ADDITION EXPERIMENTS")
    print("="*70)

    models = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "google/gemma-2-9b",
        "allenai/OLMo-2-1124-13B"
    ]

    all_results = {}

    for model in models:
        model_short = model.split('/')[-1]
        print(f"\n\n{'='*70}")
        print(f"Analyzing: {model_short}")
        print(f"{'='*70}")

        try:
            results = analyze_model_tokenization(model)
            all_results[model_short] = results
        except Exception as e:
            print(f"\n✗ Error analyzing {model_short}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary table
    print("\n\n" + "="*70)
    print("SUMMARY: Tokens per Number")
    print("="*70)
    print(f"\n{'Model':<25} {'2-digit':<15} {'3-digit':<15} {'4-digit':<15}")
    print("-" * 70)

    for model_short, results in all_results.items():
        tokens_2d = f"{results['2-digit']['min_tokens']}-{results['2-digit']['max_tokens']}" \
                    if results['2-digit']['min_tokens'] != results['2-digit']['max_tokens'] \
                    else str(results['2-digit']['min_tokens'])
        tokens_3d = f"{results['3-digit']['min_tokens']}-{results['3-digit']['max_tokens']}" \
                    if results['3-digit']['min_tokens'] != results['3-digit']['max_tokens'] \
                    else str(results['3-digit']['min_tokens'])
        tokens_4d = f"{results['4-digit']['min_tokens']}-{results['4-digit']['max_tokens']}" \
                    if results['4-digit']['min_tokens'] != results['4-digit']['max_tokens'] \
                    else str(results['4-digit']['min_tokens'])

        print(f"{model_short:<25} {tokens_2d:<15} {tokens_3d:<15} {tokens_4d:<15}")

    print("\n" + "="*70)
    print("Analysis complete. Use these results to configure experiments.")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
