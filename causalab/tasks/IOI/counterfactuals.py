"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Counterfactual dataset generators for the IOI task.

This module provides functions to generate counterfactual pairs for testing
different causal hypotheses about the IOI (Indirect Object Identification) task.
"""

import random
from .causal_models import positional_causal_model, NAMES


def sample_well_formed_input():
    """Sample a well-formed IOI input where name_C matches either name_A or name_B."""
    model = positional_causal_model
    input_sample = model.sample_input()

    # Ensure unique name_A and name_B
    while input_sample["name_A"] == input_sample["name_B"]:
        input_sample["name_B"] = random.choice(NAMES)

    # Set name_C to match either name_A or name_B
    input_sample["name_C"] = random.choice([input_sample["name_A"], input_sample["name_B"]])

    # input_sample is already a CausalTrace with computed values including raw_input
    return input_sample


def swap_names():
    """
    Generate a counterfactual where name_A and name_B are swapped.
    """
    model = positional_causal_model
    input_sample = sample_well_formed_input()

    # Create counterfactual with swapped names
    counterfactual = model.new_trace({
        "name_A": input_sample["name_B"],
        "name_B": input_sample["name_A"],
        "name_C": input_sample["name_C"],
    })

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def flip_name_C():
    """
    Generate a counterfactual where name_C is set to the other name.
    """
    model = positional_causal_model
    input_sample = sample_well_formed_input()

    # Flip name_C to the other name
    if input_sample["name_C"] == input_sample["name_A"]:
        new_name_C = input_sample["name_B"]
    else:
        new_name_C = input_sample["name_A"]

    counterfactual = model.new_trace({
        "name_A": input_sample["name_A"],
        "name_B": input_sample["name_B"],
        "name_C": new_name_C,
    })

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def flip_name_C_sample_new_IO():
    """
    Generate a counterfactual where name_C is set to the other name and a new IO is sampled.
    """
    model = positional_causal_model
    input_sample = sample_well_formed_input()

    # Flip name_C to the other name and sample a new IO name
    if input_sample["name_C"] == input_sample["name_A"]:
        new_name_C = input_sample["name_B"]
        new_name_A = random.choice([n for n in NAMES if n != new_name_C and n != input_sample["name_A"]])
        new_name_B = input_sample["name_B"]
    else:
        new_name_C = input_sample["name_A"]
        new_name_A = input_sample["name_A"]
        new_name_B = random.choice([n for n in NAMES if n != new_name_C and n != input_sample["name_B"]])

    counterfactual = model.new_trace({
        "name_A": new_name_A,
        "name_B": new_name_B,
        "name_C": new_name_C,
    })

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def swap_names_and_flip_C():
    """
    Generate a counterfactual where name_A and name_B are swapped AND name_C is flipped.
    """
    model = positional_causal_model
    input_sample = sample_well_formed_input()

    # Flip name_C to the other name (in original coordinate system)
    if input_sample["name_C"] == input_sample["name_A"]:
        new_name_C = input_sample["name_B"]
    else:
        new_name_C = input_sample["name_A"]

    counterfactual = model.new_trace({
        "name_A": input_sample["name_B"],
        "name_B": input_sample["name_A"],
        "name_C": new_name_C,
    })

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def random_counterfactual():
    """
    Generate a completely random counterfactual by sampling two independent inputs.
    """
    input_sample = sample_well_formed_input()
    counterfactual = sample_well_formed_input()

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def sample_three_random_distinct_names():
    """
    Sample three distinct names from the NAMES list.

    Returns:
        tuple: (name_A, name_B, name_C) where all three names are distinct
    """
    # Sample without replacement to ensure distinct names
    sampled_names = random.sample(NAMES, 3)
    return sampled_names[0], sampled_names[1], sampled_names[2]


def random_ABC():
    """
    Generate a counterfactual where the original has a well-formed IOI pattern
    (name_C matches name_A or name_B), but the counterfactual has three distinct
    random names (name_A, name_B, name_C all different).

    This creates an ambiguous scenario where there is no definitive answer for
    which name should come next, since name_C doesn't match either name_A or name_B.
    """
    model = positional_causal_model

    # Original: well-formed input
    input_sample = sample_well_formed_input()

    # Counterfactual: three distinct random names
    cf_name_A, cf_name_B, cf_name_C = sample_three_random_distinct_names()
    counterfactual = model.new_trace({
        "name_A": cf_name_A,
        "name_B": cf_name_B,
        "name_C": cf_name_C,
    })

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}
