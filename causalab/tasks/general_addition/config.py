"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Configuration for general addition tasks.

This module provides the configuration data structure for addition tasks
with K numbers each having D digits.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class AdditionTaskConfig:
    """
    Configuration for an addition task.

    This defines the structure of the task including:
    - Maximum dimensions (numbers to add K, digits per number D)
    - Templates for generating prompts
    - Prompt formatting (prefix/suffix for instruction tuning)

    The task handles addition of K numbers, each with D digits.
    Input: K × D digit variables (each 0-9)
    Output: D + 1 digit variables (accounting for potential carry)
    """

    max_numbers: int  # Maximum number of addends (K)
    max_digits: int  # Maximum digits per number (D)
    templates: List[str]  # Template strings for generating prompts
    prompt_prefix: str = ""  # Text to prepend before the prompt
    prompt_suffix: str = ""  # Text to append after the prompt


def create_two_number_two_digit_config() -> AdditionTaskConfig:
    """
    Create a configuration for adding two 2-digit numbers.

    Example: "The sum of 23 and 45 is"

    Returns:
        AdditionTaskConfig for 2 numbers with 2 digits each
    """
    return AdditionTaskConfig(
        max_numbers=2,
        max_digits=2,
        templates=[
            "The sum of {num0} and {num1} is",
        ],
        prompt_prefix="",
        prompt_suffix="",
    )


def create_two_number_three_digit_config() -> AdditionTaskConfig:
    """
    Create a configuration for adding two 3-digit numbers.

    Example: "The sum of 123 and 456 is"

    Returns:
        AdditionTaskConfig for 2 numbers with 3 digits each
    """
    return AdditionTaskConfig(
        max_numbers=2,
        max_digits=3,
        templates=[
            "The sum of {num0} and {num1} is",
        ],
        prompt_prefix="",
        prompt_suffix="",
    )


def create_three_number_two_digit_config() -> AdditionTaskConfig:
    """
    Create a configuration for adding three 2-digit numbers.

    Example: "The sum of 12, 34, and 56 is"

    Returns:
        AdditionTaskConfig for 3 numbers with 2 digits each
    """
    return AdditionTaskConfig(
        max_numbers=3,
        max_digits=2,
        templates=[
            "The sum of {num0}, {num1}, and {num2} is",
        ],
        prompt_prefix="",
        prompt_suffix="",
    )


def create_general_config(max_numbers: int, max_digits: int) -> AdditionTaskConfig:
    """
    Create a general configuration for K numbers with D digits.

    Generates a template that can handle the specified number of addends.

    Args:
        max_numbers: Number of numbers to add (K)
        max_digits: Digits per number (D)

    Returns:
        AdditionTaskConfig for K numbers with D digits each
    """
    # Generate template based on number of addends
    if max_numbers == 1:
        template = "The sum of {num0} is"
    elif max_numbers == 2:
        template = "The sum of {num0} and {num1} is"
    else:
        # For 3+: "The sum of {num0}, {num1}, ..., and {numK-1} is"
        parts = ["The sum of"]
        for i in range(max_numbers - 1):
            parts.append(f"{{num{i}}},")
        parts.append(f"and {{num{max_numbers - 1}}} is")
        template = " ".join(parts)

    return AdditionTaskConfig(
        max_numbers=max_numbers,
        max_digits=max_digits,
        templates=[template],
        prompt_prefix="",
        prompt_suffix="",
    )
