"""
Test configuration for causal model validation.

This file contains all task-specific test definitions and helper functions
that need to be customized by the agent.
"""

from .causal_models import causal_model
from typing import Any

from causalab.causal.trace import CausalTrace

# TODO: Specify distinguishability tests for each dataset
# Each test compares a target variable to null
DISTINGUISHABILITY_TESTS = {
    "random_counterfactual": [
        {
            "variables1": ["target_intermediate"],
            "variables2": None,
            "description": "Target variable vs null",
            "expected": "medium",
        }
    ],
    "some_other_counterfactual": [
        {
            "variables1": ["target_intermediate"],
            "variables2": None,
            "description": "Target variable vs null",
            "expected": "high",
        }
    ],
    # etc... for each counterfactual dataset
}

# TODO: Specify dataset justifications
DATASET_JUSTIFICATIONS = {
    "random_counterfactual": "Completely independent inputs provide baseline distinguishability",
    "some_other_counterfactual": "Changes variable X while keeping Y constant",
}


# =============================================================================
# Helper Functions
# =============================================================================


# TODO: Implement correctness checker
def check_correctness(full_setting: CausalTrace) -> tuple[bool, list[str]]:
    """
    Check if all variables in a full setting are correct.

    Returns: (is_correct: bool, errors: list)
    """
    # Example:
    # expected = compute_expected(full_setting['input_x'])
    # actual = full_setting['output_foo']
    # is_correct = expected == actual
    # errors = [] if is_correct else [f"mismatch: expected {expected}, got {actual}"]
    # return (is_correct, errors)

    raise NotImplementedError("TODO: Implement check_correctness")


# TODO: Implement distribution extractor
def extract_distribution_variables(full_setting: dict[str, Any]) -> dict[str, Any]:
    """Extract all variables for distribution analysis."""
    return {
        "output_foo": full_setting["output_foo"],
        "intermediate_bar": full_setting["intermediate_bar"],
    }


# TODO: Implement expected ranges
def get_expected_ranges() -> dict[str, Any]:
    """Get expected value ranges for all variables."""
    return {
        "output_foo": range(10),
        "intermediate_bar": causal_model.values["intermediate_bar"],
    }
