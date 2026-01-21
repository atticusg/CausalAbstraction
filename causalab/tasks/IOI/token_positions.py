"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Token position functions for the IOI task.

This module provides functions to locate specific tokens in IOI prompts,
such as name positions and the last token.
"""

import re
from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.token_position_builder import TokenPosition, get_last_token_index


def get_last_token_of_name(input_sample, pipeline, name_var):
    """
    Find the last token index of a specific name in the prompt.

    Args:
        input_sample (Dict): The input dictionary to a causal model
        pipeline: The tokenizer pipeline
        name_var (str): The name variable to find ("name_A", "name_B", or "name_C")

    Returns:
        list[int]: List containing the index of the last token of the name
    """
    target_name = input_sample[name_var]
    prompt = input_sample["raw_input"]

    # Find all occurrences of the name in the prompt
    # We need to find the correct occurrence based on the variable
    # name_A and name_B appear first, name_C appears after them

    # Count which occurrence we're looking for
    name_A = input_sample["name_A"]
    name_B = input_sample["name_B"]
    name_C = input_sample["name_C"]

    # Find all word boundaries with names
    words = re.finditer(r"\b\w+\b", prompt)
    name_positions = []

    for match in words:
        word = match.group()
        if word in [name_A, name_B, name_C]:
            name_positions.append((word, match.start(), match.end()))

    # Identify which occurrence corresponds to our variable
    # First occurrence of name_A -> name_A
    # First occurrence of name_B -> name_B
    # Last occurrence of name_C's value -> name_C (which might equal name_A or name_B)
    occurrence_map = {}
    seen_A = False
    seen_B = False

    for i, (name, start, end) in enumerate(name_positions):
        if name == name_A and not seen_A:
            occurrence_map["name_A"] = (start, end)
            seen_A = True
        elif name == name_B and not seen_B:
            occurrence_map["name_B"] = (start, end)
            seen_B = True

        # name_C: keep updating to get the LAST occurrence of name_C's value
        if name == name_C:
            occurrence_map["name_C"] = (start, end)

    if name_var not in occurrence_map:
        raise ValueError(
            f"Could not find {name_var} ({target_name}) in prompt: {prompt}"
        )

    _, end_pos = occurrence_map[name_var]

    # Helper to create CausalTrace from string
    def _make_trace(text: str) -> CausalTrace:
        return CausalTrace(
            mechanisms={
                "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
            },
            inputs={"raw_input": text},
        )

    # Tokenization approach similar to MCQA
    tokenized_prompt_padded = list(pipeline.load([_make_trace(prompt)])["input_ids"][0])
    pad_token_id = pipeline.tokenizer.pad_token_id

    # Find where content starts
    content_start_idx = 0
    for i, token in enumerate(tokenized_prompt_padded):
        if token != pad_token_id:
            content_start_idx = i
            break

    # Extract content tokens
    content_tokens = [t for t in tokenized_prompt_padded if t != pad_token_id]

    # Tokenize substring up to and including the name
    substring = prompt[:end_pos]
    tokenized_substring = list(
        pipeline.load([_make_trace(substring)], no_padding=True)["input_ids"][0]
    )

    # Find where substring ends in content
    m = len(tokenized_substring)
    if m == 0:
        raise ValueError(f"Substring tokenized to empty sequence: {substring}")

    end_idx_in_content = next(
        (
            i + m
            for i in range(len(content_tokens) - m + 1)
            if content_tokens[i : i + m] == tokenized_substring
        ),
        -1,
    )

    if end_idx_in_content == -1:
        raise ValueError("Could not find tokenized substring in prompt")

    # Convert to padded coordinate system (last token of the name)
    token_index_in_padded = content_start_idx + end_idx_in_content - 1

    return [token_index_in_padded]


def create_name_A_token_position(pipeline):
    """Create a TokenPosition for the last token of name_A."""
    return TokenPosition(
        lambda x: get_last_token_of_name(x, pipeline, "name_A"), pipeline, id="name_A"
    )


def create_name_B_token_position(pipeline):
    """Create a TokenPosition for the last token of name_B."""
    return TokenPosition(
        lambda x: get_last_token_of_name(x, pipeline, "name_B"), pipeline, id="name_B"
    )


def create_name_C_token_position(pipeline):
    """Create a TokenPosition for the last token of name_C."""
    return TokenPosition(
        lambda x: get_last_token_of_name(x, pipeline, "name_C"), pipeline, id="name_C"
    )


def create_last_token_position(pipeline):
    """Create a TokenPosition for the last token in the input."""
    return TokenPosition(
        lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token"
    )


def create_token_positions(pipeline):
    """
    Create all token positions for the IOI task.

    Args:
        pipeline: The tokenizer pipeline

    Returns:
        dict: Dictionary mapping token position names to TokenPosition objects
    """
    return {
        "name_A": create_name_A_token_position(pipeline),
        "name_B": create_name_B_token_position(pipeline),
        "name_C": create_name_C_token_position(pipeline),
        "last_token": create_last_token_position(pipeline),
    }
