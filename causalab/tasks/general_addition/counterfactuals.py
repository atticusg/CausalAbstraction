"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Counterfactual dataset generation for addition tasks.

This module provides functions to generate counterfactual examples
for testing causal hypotheses about addition.
"""

from typing import Dict, Any
from .causal_models import (
    create_basic_addition_model,
    sample_valid_addition_input,
    sample_sum_to_nine_input
)
from .config import AdditionTaskConfig


def random_counterfactual(config: AdditionTaskConfig, num_addends: int, num_digits: int) -> Dict[str, Any]:
    """
    Generate a completely random counterfactual by sampling two independent inputs.

    This is a baseline condition - the counterfactual is unrelated to the input.

    Args:
        config: Task configuration
        num_addends: Number of numbers to add
        num_digits: Digits per number

    Returns:
        Dictionary with "input" and "counterfactual_inputs" keys
    """
    model = create_basic_addition_model(config)

    # Sample two independent inputs and generate raw_input only
    input_sample = sample_valid_addition_input(config, num_addends, num_digits)

    counterfactual = sample_valid_addition_input(config, num_addends, num_digits)

    return {"input": input_sample.to_dict() if hasattr(input_sample, "to_dict") else input_sample, "counterfactual_inputs": [counterfactual]}


def sum_to_nine_counterfactual(config: AdditionTaskConfig, digit_position: int) -> Dict[str, Any]:
    """
    Generate counterfactual where input has digits summing to 9 at a specific position.

    The input is sampled such that the specified digit position sums to 9 across
    all addends (e.g., if digit_position=0, the tens digits sum to 9).
    The counterfactual is completely random.

    This tests whether the model treats "sum to 9" cases differently (e.g.,
    because they don't generate a carry).

    Args:
        config: Task configuration
        digit_position: Which digit position should sum to 9 in the input

    Returns:
        Dictionary with "input" and "counterfactual_inputs" keys
    """
    model = create_basic_addition_model(config)

    # Input: digits at digit_position sum to 9
    input_sample = sample_sum_to_nine_input(config, digit_position)

    # Counterfactual: completely random
    num_digits = config.max_digits
    counterfactual = sample_valid_addition_input(config, 2, num_digits)

    return {"input": input_sample.to_dict() if hasattr(input_sample, "to_dict") else input_sample, "counterfactual_inputs": [counterfactual]}


