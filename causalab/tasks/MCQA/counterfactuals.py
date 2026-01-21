"""Counterfactual dataset generators for the MCQA task.

This module provides functions to generate counterfactual pairs for testing
different causal hypotheses about the MCQA task.
"""

import random

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace
from .causal_models import (
    positional_causal_model,
    COLORS,
    ALPHABET,
    NUM_CHOICES,
    TEMPLATES,
    OBJECTS,
)


def sample_answerable_question() -> CausalTrace:
    """Sample a question where the correct answer appears in the choices."""
    # Sample core inputs manually (not using sample_input() to avoid triggering
    # get_answer_position with random choices that don't include the color)
    template = random.choice(TEMPLATES)
    obj = random.choice(OBJECTS)
    color = random.choice(COLORS)

    # Build choices that INCLUDE the correct color from the start
    other_colors = [c for c in COLORS if c != color]
    other_choices = random.sample(other_colors, NUM_CHOICES - 1)
    answer_position = random.randint(0, NUM_CHOICES - 1)
    choices = (
        other_choices[:answer_position] + [color] + other_choices[answer_position:]
    )

    # Sample unique symbols
    symbols = random.sample(ALPHABET, NUM_CHOICES)

    # Build input dict - color is guaranteed to be in choices
    input_dict = {
        "template": template,
        "object": obj,
        "color": color,
    }
    for idx in range(NUM_CHOICES):
        input_dict[f"choice{idx}"] = choices[idx]
        input_dict[f"symbol{idx}"] = symbols[idx]

    # Create trace - answer_position will always be valid since color is in choices
    return positional_causal_model.new_trace(input_dict)


def same_symbol_different_position() -> CounterfactualExample:
    """
    Generate a counterfactual where the answer position changes but symbols stay the same.
    This swaps the choices and symbols at two positions.
    """
    input_sample = sample_answerable_question()
    pos = input_sample["answer_position"]
    new_pos = random.choice([i for i in range(NUM_CHOICES) if i != pos])

    # Build input dict with swapped values (only input variables, not computed ones)
    cf_dict = {var: input_sample[var] for var in positional_causal_model.inputs}
    cf_dict[f"choice{pos}"], cf_dict[f"choice{new_pos}"] = (
        cf_dict[f"choice{new_pos}"],
        cf_dict[f"choice{pos}"],
    )
    cf_dict[f"symbol{pos}"], cf_dict[f"symbol{new_pos}"] = (
        cf_dict[f"symbol{new_pos}"],
        cf_dict[f"symbol{pos}"],
    )

    counterfactual = positional_causal_model.new_trace(cf_dict)
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def different_symbol() -> CounterfactualExample:
    """
    Generate a counterfactual where all symbols change (but choices stay the same).
    """
    input_sample = sample_answerable_question()

    # Build input dict with new symbols (only input variables, not computed ones)
    cf_dict = {var: input_sample[var] for var in positional_causal_model.inputs}
    current_symbols = [input_sample[f"symbol{i}"] for i in range(NUM_CHOICES)]
    complement = [x for x in ALPHABET if x not in current_symbols]
    new_symbols = random.sample(complement, NUM_CHOICES)
    for i in range(NUM_CHOICES):
        cf_dict[f"symbol{i}"] = new_symbols[i]

    counterfactual = positional_causal_model.new_trace(cf_dict)
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def random_counterfactual() -> CounterfactualExample:
    """
    Generate a completely random counterfactual by sampling two independent inputs.
    """
    input_sample = sample_answerable_question()
    counterfactual = sample_answerable_question()
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}
