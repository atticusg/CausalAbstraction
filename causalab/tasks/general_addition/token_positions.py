"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Token position functions for general addition tasks.

This module provides functions to locate specific tokens in addition prompts,
such as individual digits, delimiters, and the last token.
"""

from typing import List, Dict, Any
from causalab.neural.token_position_builder import TokenPosition, get_last_token_index, get_substring_token_ids
from causalab.neural.pipeline import LMPipeline


def get_digit_token_position(
    input_sample: Dict[str, Any],
    pipeline: LMPipeline,
    addend_idx: int,
    digit_idx: int
) -> List[int]:
    """
    Get token position(s) for a specific digit in a specific addend.

    This function locates the digit at position digit_idx in addend addend_idx
    by constructing a unique context substring around the number.

    Args:
        input_sample: Input dictionary from causal model
        pipeline: LM pipeline with tokenizer
        addend_idx: Which addend (0-indexed, e.g., 0 for first number)
        digit_idx: Which digit position (0-indexed from left, e.g., 0 for tens in "23")

    Returns:
        List of token position indices

    Example:
        For "The sum of 23 and 45 is":
        - get_digit_token_position(input, pipeline, 0, 0) -> tokens for "2" in "23"
        - get_digit_token_position(input, pipeline, 1, 1) -> tokens for "5" in "45"
    """
    prompt = input_sample['raw_input']
    num_addends = input_sample['num_addends']
    num_digits = input_sample['num_digits']

    # Build the full number for this addend and format it (remove leading zeros)
    number_digits = []
    for d in range(num_digits):
        number_digits.append(input_sample[f'digit_{addend_idx}_{d}'])

    # Convert to integer to remove leading zeros, then back to string
    number_value = sum(d * (10 ** (num_digits - 1 - i)) for i, d in enumerate(number_digits))
    number_str = str(number_value)

    # Build unique context around this number
    # Strategy: include enough surrounding text to make it unique
    if addend_idx == 0:
        # First number: use "of {number} and" or "of {number}," context
        if num_addends == 2:
            context = f"of {number_str} and"
        else:
            context = f"of {number_str},"
    elif addend_idx == num_addends - 1:
        # Last number: use "and {number} is" context
        context = f"and {number_str} is"
    else:
        # Middle number: use ", {number}," or ", {number}, and" context
        context = f", {number_str},"

    # Now we need to identify which token(s) correspond to the specific digit
    # The digit might be part of a multi-digit token or a separate token
    # We'll get tokens for the full number and then try to narrow down

    # Get token positions for just the number substring within the context
    number_tokens = get_substring_token_ids(prompt, number_str, pipeline, add_special_tokens=False)

    # If the number is tokenized as a single token, we return that token for any digit
    if len(number_tokens) == 1:
        return number_tokens

    # If the number is tokenized into multiple tokens, we need to figure out which
    # token corresponds to which digit. This is model-dependent and tricky.
    # For now, we'll return all tokens for the number and note this in documentation.
    # A more sophisticated approach would decode each token and match to digits.

    return number_tokens


def create_digit_token_position(pipeline: LMPipeline, addend_idx: int, digit_idx: int) -> TokenPosition:
    """
    Create a TokenPosition for a specific digit in a specific addend.

    Args:
        pipeline: LM pipeline with tokenizer
        addend_idx: Which addend (0-indexed)
        digit_idx: Which digit position (0-indexed from left)

    Returns:
        TokenPosition object
    """
    return TokenPosition(
        lambda x: get_digit_token_position(x, pipeline, addend_idx, digit_idx),
        pipeline,
        id=f"digit_{addend_idx}_{digit_idx}"
    )


def get_delimiter_token_position(
    input_sample: Dict[str, Any],
    pipeline: LMPipeline,
    delimiter_type: str
) -> List[int]:
    """
    Get token position(s) for a delimiter in the prompt.

    Args:
        input_sample: Input dictionary from causal model
        pipeline: LM pipeline with tokenizer
        delimiter_type: Type of delimiter ("and", "is", "comma")

    Returns:
        List of token position indices
    """
    prompt = input_sample['raw_input']

    # Define delimiter strings
    if delimiter_type == "and":
        # The " and " between numbers
        delimiter = " and "
    elif delimiter_type == "is":
        # The " is" at the end before answer
        delimiter = " is"
    elif delimiter_type == "comma":
        # The ", " between numbers (for 3+ addends)
        delimiter = ", "
    else:
        raise ValueError(f"Unknown delimiter type: {delimiter_type}")

    # Get token positions for the delimiter
    return get_substring_token_ids(prompt, delimiter, pipeline, add_special_tokens=False)


def create_delimiter_token_position(pipeline: LMPipeline, delimiter_type: str) -> TokenPosition:
    """
    Create a TokenPosition for a delimiter.

    Args:
        pipeline: LM pipeline with tokenizer
        delimiter_type: Type of delimiter ("and", "is", "comma")

    Returns:
        TokenPosition object
    """
    return TokenPosition(
        lambda x: get_delimiter_token_position(x, pipeline, delimiter_type),
        pipeline,
        id=f"delimiter_{delimiter_type}"
    )


def create_last_token_position(pipeline: LMPipeline) -> TokenPosition:
    """
    Create a TokenPosition for the last token in the prompt.

    Args:
        pipeline: LM pipeline with tokenizer

    Returns:
        TokenPosition object
    """
    return TokenPosition(
        lambda x: get_last_token_index(x, pipeline),
        pipeline,
        id="last_token"
    )


def create_token_positions(pipeline: LMPipeline, num_addends: int, num_digits: int) -> Dict[str, TokenPosition]:
    """
    Create all token positions for an addition task.

    Args:
        pipeline: LM pipeline with tokenizer
        num_addends: Number of numbers being added
        num_digits: Number of digits per number

    Returns:
        Dictionary mapping token position names to TokenPosition objects
    """
    token_positions = {}

    # Create token positions for each digit of each addend
    for k in range(num_addends):
        for d in range(num_digits):
            key = f"digit_{k}_{d}"
            token_positions[key] = create_digit_token_position(pipeline, k, d)

    # Create delimiter token positions
    token_positions["delimiter_and"] = create_delimiter_token_position(pipeline, "and")
    token_positions["delimiter_is"] = create_delimiter_token_position(pipeline, "is")
    if num_addends > 2:
        token_positions["delimiter_comma"] = create_delimiter_token_position(pipeline, "comma")

    # Create last token position
    token_positions["last_token"] = create_last_token_position(pipeline)

    return token_positions
