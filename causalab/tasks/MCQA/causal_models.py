"""Causal models for the MCQA task.

This module defines the causal model structure for multiple choice question answering,
including variables, values, parent relationships, and mechanisms.
"""

import random
from causalab.causal.causal_model import CausalModel

# Constants
OBJECTS = ["ball", "car", "house", "shirt", "flower", "pen", "cup", "hat", "bag", "shoe"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white"]

NUM_CHOICES = 2
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEMPLATES = [
    "The {object} is {color}. What color is the {object}?" +
    "".join([f"\n{{symbol{str(i)}}}. {{choice{str(i)}}}" for i in range(NUM_CHOICES)]) +
    "\nAnswer:"
]


# Causal Model Mechanisms
def fill_template(*args):
    """Fill in the template with object, color, symbols, and choices."""
    template = args[0]
    object_name = args[1]
    color = args[2]
    symbols = args[3:3 + NUM_CHOICES]
    choices = args[3 + NUM_CHOICES:3 + 2 * NUM_CHOICES]

    filled_template = template.replace("{object}", object_name).replace("{color}", color)
    for i, symbol in enumerate(symbols):
        filled_template = filled_template.replace(f"{{symbol{i}}}", symbol)
    for i, choice in enumerate(choices):
        filled_template = filled_template.replace(f"{{choice{i}}}", choice)
    return filled_template


def get_answer_position(color, *choices):
    """Determine which choice position contains the correct answer."""
    for i, choice in enumerate(choices):
        if choice == color:
            return i


def get_answer(answer_position, *symbols):
    """Get the symbol corresponding to the correct answer position."""
    if answer_position is None:
        return None
    return symbols[answer_position]


# Causal Model Definition
variables = (
    ["template", "object", "color", "raw_input"] +
    ["symbol" + str(x) for x in range(NUM_CHOICES)] +
    ["choice" + str(x) for x in range(NUM_CHOICES)] +
    ["answer_position", "answer", "raw_output"]
)

values = {"choice" + str(x): COLORS for x in range(NUM_CHOICES)}
values.update({"symbol" + str(x): ALPHABET for x in range(NUM_CHOICES)})
values.update({"answer_position": range(NUM_CHOICES), "answer": ALPHABET})
values.update({"template": TEMPLATES})
values.update({"object": OBJECTS, "color": COLORS})
values.update({"raw_input": None, "raw_output": None})

parents = {
    "template": [],
    "object": [],
    "color": [],
    "raw_input": (
        ["template", "object", "color"] +
        ["symbol" + str(x) for x in range(NUM_CHOICES)] +
        ["choice" + str(x) for x in range(NUM_CHOICES)]
    ),
    "answer_position": (
        ["color"] +
        ["choice" + str(x) for x in range(NUM_CHOICES)]
    ),
    "answer": (
        ["answer_position"] +
        ["symbol" + str(x) for x in range(NUM_CHOICES)]
    ),
    "raw_output": ["answer"],
}
parents.update({"choice" + str(x): [] for x in range(NUM_CHOICES)})
parents.update({"symbol" + str(x): [] for x in range(NUM_CHOICES)})

mechanisms = {
    "template": lambda: random.choice(TEMPLATES),
    "object": lambda: random.choice(OBJECTS),
    "color": lambda: random.choice(COLORS),
    **{f"symbol{i}": lambda: random.choice(ALPHABET) for i in range(NUM_CHOICES)},
    **{f"choice{i}": lambda: random.choice(COLORS) for i in range(NUM_CHOICES)},
    "raw_input": fill_template,
    "answer_position": get_answer_position,
    "answer": get_answer,
    "raw_output": lambda x: " " + x if x is not None else None,
}

positional_causal_model = CausalModel(
    variables,
    values,
    parents,
    mechanisms,
    id=f"{NUM_CHOICES}_answer_MCQA"
)
