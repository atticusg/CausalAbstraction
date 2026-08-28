"""
Checker function for validating model outputs against causal model outputs.
"""

from typing import Any


def checker(neural_output: dict[str, Any], causal_output: str) -> bool:
    """
    Check if model output matches expected output.

    Args:
        neural_output: Dict with "string" key containing model output
        causal_output: Expected output string from causal model

    Returns:
        bool: True if outputs match
    """
    expected = causal_output.strip()
    actual = neural_output["string"].strip()

    # in some cases, you may need to use something less strict,
    # such as actual.startswith(expected), expected.in(actual), etc.
    return actual == expected
