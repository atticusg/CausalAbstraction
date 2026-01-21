"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Causal models for the IOI (Indirect Object Identification) task.

This module defines the causal model structure for the IOI task, including
variables, values, parent relationships, and mechanisms.
"""

import random
import json
import os
from pathlib import Path
from causalab.causal.causal_model import CausalModel


# Load data files
def get_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# Get paths to data files
IOI_DIR = Path(__file__).resolve().parent
NAMES = get_data(os.path.join(IOI_DIR, 'names.json'))
OBJECTS = get_data(os.path.join(IOI_DIR, 'objects.json'))
PLACES = get_data(os.path.join(IOI_DIR, 'places.json'))
TEMPLATES = get_data(os.path.join(IOI_DIR, 'templates.json'))


# Causal Model Mechanisms
def fill_template(trace):
    """Fill in the template with names, place, and object."""
    template = trace["template"]
    name_A = trace["name_A"]
    name_B = trace["name_B"]
    name_C = trace["name_C"]
    place = trace["place"]
    object_name = trace["object"]

    filled = template.replace("{name_A}", name_A)
    filled = filled.replace("{name_B}", name_B)
    filled = filled.replace("{name_C}", name_C)
    filled = filled.replace("{place}", place)
    # Handle both "a" and "an" article cases
    filled = filled.replace("{object}", object_name)
    return filled


def get_output_position(trace):
    """Determine the output position (0 for A, 1 for B)."""
    name_A = trace["name_A"]
    name_B = trace["name_B"]
    name_C = trace["name_C"]

    if name_C == name_A:
        return 1  # Answer is B
    elif name_C == name_B:
        return 0  # Answer is A
    else:
        # This shouldn't happen in well-formed inputs
        return None


def get_output_token(trace):
    """Determine the correct output token based on name_C."""
    name_A = trace["name_A"]
    name_B = trace["name_B"]
    name_C = trace["name_C"]

    if name_C == name_A:
        return name_B
    elif name_C == name_B:
        return name_A
    else:
        # This shouldn't happen in well-formed inputs
        return None


def create_ioi_causal_model(bias=2.0, token_coeff=-1.0, pos_coeff=-1.0):
    """
    Create and return the IOI causal model with specified parameters.

    Args:
        bias: Baseline bias for logit difference
        token_coeff: Coefficient for token signal contribution
        pos_coeff: Coefficient for position signal contribution

    Returns:
        CausalModel instance for IOI task
    """

    def get_logits(trace):
        """
        Calculate logits for name_A and name_B.

        Returns a dictionary mapping each name to its logit value, or None if
        there is no definitive answer (output_token or output_position is None).
        """
        name_A = trace["name_A"]
        name_B = trace["name_B"]
        output_token = trace["output_token"]
        output_position = trace["output_position"]

        # Calculate signal for name_A
        token_signal_A = 1 if output_token == name_A else 0
        pos_signal_A = 1 if output_position == 0 else 0
        logit_A = bias - token_coeff * token_signal_A - pos_coeff * pos_signal_A

        # Calculate signal for name_B
        token_signal_B = 1 if output_token == name_B else 0
        pos_signal_B = 1 if output_position == 1 else 0
        logit_B = bias - token_coeff * token_signal_B - pos_coeff * pos_signal_B

        return {name_A: logit_A, name_B: logit_B}

    def get_raw_output(trace):
        """
        Generate the raw output as a dictionary with token and logits.
        Token is whichever name has the higher logit.
        """
        name_A = trace["name_A"]
        name_B = trace["name_B"]
        logits = trace["logits"]

        # Pick the name with higher logit
        if logits[name_A] > logits[name_B]:
            token = name_A
        elif logits[name_B] > logits[name_A]:
            token = name_B
        else:
            token = name_A

        return {
            "string": " " + token if token is not None else None,
            "logits": logits,
        }

    variables = [
        "template", "name_A", "name_B", "name_C", "place", "object",
        "raw_input", "output_position", "output_token", "logits", "raw_output"
    ]

    values = {
        "template": TEMPLATES,
        "name_A": NAMES,
        "name_B": NAMES,
        "name_C": NAMES,
        "place": PLACES,
        "object": OBJECTS,
        "raw_input": None,
        "output_position": [0, 1],
        "output_token": NAMES,
        "logits": None,
        "raw_output": None,
    }

    parents = {
        "template": [],
        "name_A": [],
        "name_B": [],
        "name_C": [],
        "place": [],
        "object": [],
        "raw_input": ["template", "name_A", "name_B", "name_C", "place", "object"],
        "output_position": ["name_A", "name_B", "name_C"],
        "output_token": ["name_A", "name_B", "name_C"],
        "logits": ["name_A", "name_B", "output_token", "output_position"],
        "raw_output": ["name_A", "name_B", "logits"],
    }

    mechanisms = {
        "template": lambda t: random.choice(TEMPLATES),
        "name_A": lambda t: random.choice(NAMES),
        "name_B": lambda t: random.choice(NAMES),
        "name_C": lambda t: random.choice(NAMES),
        "place": lambda t: random.choice(PLACES),
        "object": lambda t: random.choice(OBJECTS),
        "raw_input": fill_template,
        "output_position": get_output_position,
        "output_token": get_output_token,
        "logits": get_logits,
        "raw_output": get_raw_output,
    }

    return CausalModel(
        variables,
        values,
        parents,
        mechanisms,
        id="ioi"
    )


# Default model instance for backward compatibility
positional_causal_model = create_ioi_causal_model()
