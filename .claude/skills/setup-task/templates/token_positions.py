"""
Token position definitions and helper functions for the task.
"""

from typing import Any, Callable, Dict

from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import (
    build_token_position_factories,
)


def create_token_positions(
    pipeline: LMPipeline, template: str
) -> Dict[str, Callable[..., Any]]:
    """
    Create token position factories for the task.

    Args:
        pipeline: The LMPipeline to use for tokenization
        template: The template string for the task

    Returns:
        Dictionary of token position factories
    """

    # ============================================================================
    # DECLARATIVE TOKEN POSITION SYSTEM (PREFERRED APPROACH)
    # ============================================================================

    # TODO: Update these specs based on your task requirements
    token_position_specs = {
        # -------------------------------------------------------------------------
        # PATTERN 1: Fixed Positions
        # -------------------------------------------------------------------------
        # Use these for positions that don't depend on variable values
        # "last": {"type": "index", "position": -1},  # Last token in sequence
        # "first": {"type": "index", "position": 0},  # First token in sequence
        # "third": {"type": "index", "position": 2},  # Third token (0-indexed)
        # -------------------------------------------------------------------------
        # PATTERN 2: Variable Positions
        # -------------------------------------------------------------------------
        # Use these to target where a template variable appears
        # Example: For template "The sum of {x} and {y} is ", this finds all tokens
        # that make up the value of x when it's substituted into the template
        # "x": {"type": "variable", "name": "x"},
        # "y": {"type": "variable", "name": "y"},
        # "answer": {"type": "variable", "name": "answer"},
        # -------------------------------------------------------------------------
        # PATTERN 3: Indexed Positions (nth token within a variable)
        # -------------------------------------------------------------------------
        # Use when a variable tokenizes to multiple tokens and you want a specific one
        # Example: If "answer" = "Paris" tokenizes to ["Par", "is"], position 0 gets "Par"
        # "first_token_of_answer": {
        #     "type": "index",
        #     "position": 0,  # First token of the variable
        #     "scope": {"variable": "answer"}
        # },
        # "last_token_of_answer": {
        #     "type": "index",
        #     "position": -1,  # Last token of the variable
        #     "scope": {"variable": "answer"}
        # },
        # "second_token_of_answer": {
        #     "type": "index",
        #     "position": 1,  # Second token (0-indexed)
        #     "scope": {"variable": "answer"}
        # },
        # -------------------------------------------------------------------------
        # PATTERN 4: Relative Positions
        # -------------------------------------------------------------------------
        # Use these to target tokens immediately before/after a variable
        # Example: For "The sum of {x} and {y}", position +1 relative to x gets "and"
        # "delimiter_after_x": {
        #     "type": "index",
        #     "position": +1,  # Token immediately after x
        #     "relative_to": {"variable": "x"}
        # },
        # "token_before_y": {
        #     "type": "index",
        #     "position": -1,  # Token immediately before y
        #     "relative_to": {"variable": "y"}
        # },
        # "two_tokens_after_x": {
        #     "type": "index",
        #     "position": +2,  # Two tokens after x
        #     "relative_to": {"variable": "x"}
        # },
        # -------------------------------------------------------------------------
        # PATTERN 5: Dynamic Positions (function-based specs)
        # -------------------------------------------------------------------------
        # Use these when the position depends on causal model variables not in the template
        # The function receives the full causal model setting and returns a declarative spec
        # Example: Find the position of whichever symbol is the correct answer
        # "correct_answer": lambda setting: {
        #     "type": "variable",
        #     "name": "option_Z" if setting["answer_letter"] == 'Z' else "option_X"
        # }
        # Example: Use a derived variable to determine which template variable to target
        # "target_entity": lambda setting: {
        #     "type": "variable",
        #     "name": setting["entity_to_track"]  # Assumes entity_to_track is a variable name
        # },
        # Example: Conditional logic based on task structure
        # "relevant_position": lambda setting: {
        #     "type": "index",
        #     "position": -1 if setting["task_type"] == "completion" else 0
        # },
    }

    # ============================================================================
    # FALLBACK: Custom Python Functions (only if declarative system insufficient)
    # ============================================================================

    # If the declarative system can't express your token position, use this fallback.
    # IMPORTANT: Document in issues.md why you needed the fallback!
    # See neural.token_position_builder for helpers: get_last_token_index, get_substring_token_ids, etc.

    # def get_example_token_indices(input_sample, pipeline):
    #     """Returns list of token indices for intervention position."""
    #     token_ids = get_substring_token_ids(input_sample["raw_input"], "substring", pipeline)
    #     return [token_ids[-1]]

    # Create factory functions and add to task.token_positions dict
    # token_position_specs.update({
    #     "last": lambda pipeline: TokenPosition(
    #         lambda x: get_last_token_index(x, pipeline), pipeline, id="last"
    #     ),
    #     "position1": lambda pipeline: TokenPosition(
    #         lambda x: get_example_token_indices(x, pipeline), pipeline, id="position1"
    #     ),
    # })

    # IMPORTANT: Return the result of build_token_position_factories() directly.
    # This returns Dict[str, Callable] (factory functions), NOT TokenPosition objects.
    # Do NOT call the factories (e.g., factory(pipeline)) — the experiment framework calls them.
    return build_token_position_factories(token_position_specs, template)
