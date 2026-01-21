"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Tokenization configuration for different models.

This module provides lookup tables for how many tokens each model uses
for numbers of different lengths. This is critical for creating the correct
token positions for interventions.
"""

from typing import Dict


# Tokenization lookup: (model_name_pattern, num_digits) -> tokens_per_number
# Based on empirical testing from analyze_tokenization.py
TOKENIZATION_CONFIG = {
    # Llama models
    "llama": {
        2: 1,  # 2-digit numbers → 1 token
        3: 1,  # 3-digit numbers → 1 token
        4: 2,  # 4-digit numbers → 2 tokens
    },
    # Gemma models
    "gemma": {
        2: 2,  # 2-digit numbers → 2 tokens (digit-by-digit)
        3: 3,  # 3-digit numbers → 3 tokens
        4: 4,  # 4-digit numbers → 4 tokens
    },
    # OLMo models
    "olmo": {
        2: 1,  # 2-digit numbers → 1 token
        3: 1,  # 3-digit numbers → 1 token
        4: 2,  # 4-digit numbers → 2 tokens
    },
}


def get_model_family(model_name: str) -> str:
    """
    Determine the model family from the model name.

    Args:
        model_name: Full model name (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")

    Returns:
        Model family identifier ("llama", "gemma", or "olmo")

    Raises:
        ValueError: If model family cannot be determined
    """
    model_lower = model_name.lower()

    if "llama" in model_lower:
        return "llama"
    elif "gemma" in model_lower:
        return "gemma"
    elif "olmo" in model_lower:
        return "olmo"
    else:
        raise ValueError(
            f"Unknown model family for '{model_name}'. "
            f"Expected model name to contain 'llama', 'gemma', or 'olmo'."
        )


def get_tokens_per_number(model_name: str, num_digits: int) -> int:
    """
    Get the number of tokens per number for a given model and digit count.

    Args:
        model_name: Full model name
        num_digits: Number of digits in each number (2, 3, or 4)

    Returns:
        Number of tokens the model uses for numbers with this many digits

    Raises:
        ValueError: If configuration not found for this model/digit combination

    Examples:
        >>> get_tokens_per_number("meta-llama/Meta-Llama-3.1-8B-Instruct", 2)
        1
        >>> get_tokens_per_number("google/gemma-2-9b", 2)
        2
        >>> get_tokens_per_number("allenai/OLMo-2-1124-13B", 4)
        2
    """
    model_family = get_model_family(model_name)

    if model_family not in TOKENIZATION_CONFIG:
        raise ValueError(f"No tokenization config for model family: {model_family}")

    config = TOKENIZATION_CONFIG[model_family]

    if num_digits not in config:
        raise ValueError(
            f"No tokenization config for {model_family} with {num_digits} digits. "
            f"Available: {list(config.keys())}"
        )

    return config[num_digits]


def get_all_number_token_indices(
    input_sample: Dict, pipeline, addend_idx: int, num_digits: int, model_name: str
) -> list:
    """
    Get all token indices for a number, based on model tokenization.

    Args:
        input_sample: Input dictionary from causal model
        pipeline: LM pipeline
        addend_idx: Which addend (0 or 1)
        num_digits: Number of digits per number
        model_name: Model name to determine tokenization

    Returns:
        List of token indices for this number
    """
    from causalab.tasks.general_addition.token_positions import get_digit_token_position

    # Get how many tokens this model uses for numbers with this many digits
    tokens_per_number = get_tokens_per_number(model_name, num_digits)

    # Get tokens for the first digit
    tokens = get_digit_token_position(input_sample, pipeline, addend_idx, 0)

    # Return the correct number of tokens
    return tokens[:tokens_per_number]
