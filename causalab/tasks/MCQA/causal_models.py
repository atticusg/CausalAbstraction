"""Causal models for the MCQA task.

This module defines the causal model structure for multiple choice question answering,
including variables, values, parent relationships, and mechanisms.
"""

from causalab.causal.causal_model import CausalModel, build_output_tokens
from causalab.causal.trace import CausalTrace, Mechanism, input_var

# Constants
OBJECTS = [
    "ball",
    "car",
    "house",
    "shirt",
    "flower",
    "pen",
    "cup",
    "hat",
    "bag",
    "shoe",
]
COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "brown",
    "black",
    "white",
]

NUM_CHOICES = 2
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEMPLATES = [
    "The {object} is {color}. What color is the {object}?"
    + "".join([f"\n{{symbol{str(i)}}}. {{choice{str(i)}}}" for i in range(NUM_CHOICES)])
    + "\nAnswer:"
]


# Causal Model Definition
values: dict[str, str | list[str] | list[int] | None] = {}
values.update({"choice" + str(x): COLORS for x in range(NUM_CHOICES)})
values.update({"symbol" + str(x): list(ALPHABET) for x in range(NUM_CHOICES)})
values.update({"answer_position": list(range(NUM_CHOICES)), "answer": list(ALPHABET)})
values.update({"template": TEMPLATES})
values.update({"object": OBJECTS, "color": COLORS})
values.update({"raw_input": None, "raw_output": None})


def _fill_template(t: CausalTrace) -> str:
    """Fill in the template with object, color, symbols, and choices."""
    template = t["template"]
    object_name = t["object"]
    color = t["color"]

    filled_template = template.replace("{object}", object_name).replace(
        "{color}", color
    )
    for i in range(NUM_CHOICES):
        filled_template = filled_template.replace(f"{{symbol{i}}}", t[f"symbol{i}"])
    for i in range(NUM_CHOICES):
        filled_template = filled_template.replace(f"{{choice{i}}}", t[f"choice{i}"])
    return filled_template


def _get_answer_position(t: CausalTrace) -> int:
    """Determine which choice position contains the correct answer."""
    color = t["color"]
    choices = [t[f"choice{i}"] for i in range(NUM_CHOICES)]
    for i in range(NUM_CHOICES):
        if choices[i] == color:
            return i
    raise ValueError(
        f"No correct answer position found for color {color} in choices {choices}"
    )


def _get_answer(t: CausalTrace) -> str:
    """Get the symbol corresponding to the correct answer position."""
    answer_position = t["answer_position"]
    return t[f"symbol{answer_position}"]


# Define mechanisms using the new Mechanism API
mechanisms = {
    # Input variables (no parents)
    "template": input_var(TEMPLATES),
    "object": input_var(OBJECTS),
    "color": input_var(COLORS),
    **{f"symbol{i}": input_var(list(ALPHABET)) for i in range(NUM_CHOICES)},
    **{f"choice{i}": input_var(COLORS) for i in range(NUM_CHOICES)},
    # Computed variables
    "raw_input": Mechanism(
        parents=(
            ["template", "object", "color"]
            + ["symbol" + str(x) for x in range(NUM_CHOICES)]
            + ["choice" + str(x) for x in range(NUM_CHOICES)]
        ),
        compute=_fill_template,
    ),
    "answer_position": Mechanism(
        parents=(["color"] + ["choice" + str(x) for x in range(NUM_CHOICES)]),
        compute=_get_answer_position,
    ),
    "answer": Mechanism(
        parents=(["answer_position"] + ["symbol" + str(x) for x in range(NUM_CHOICES)]),
        compute=_get_answer,
    ),
    "raw_output": Mechanism(
        parents=["answer"],
        compute=lambda t: " " + t["answer"],
    ),
}

# ``output_tokens`` declares the surface forms for the two variables MCQA is
# scored on: ``answer`` (the option *letter*, the variable configs localize on)
# drives the probability path, and ``answer_position`` (the module
# ``TARGET_VARIABLE``) drives the derived checker. The letter/value the model
# emits is example-dependent (the per-example ``CLASS_TOKEN_IDS`` path handles
# class scoring), so the derived checker falls back to a literal exact match on
# ``raw_output`` — reproducing the former checker.py for both the letter and the
# ``score_by: value`` conventions (#296).
positional_causal_model = CausalModel(
    mechanisms,
    values,
    id=f"{NUM_CHOICES}_answer_MCQA",
    output_tokens={
        "answer": build_output_tokens(list(ALPHABET)),
        "answer_position": build_output_tokens(list(range(NUM_CHOICES))),
    },
)


# ---------------------------------------------------------------------------
# Standard exports for load_task()
# ---------------------------------------------------------------------------

CAUSAL_MODEL = positional_causal_model
TARGET_VARIABLE = "answer_position"
TEMPLATE = TEMPLATES[0]


def PREDICT_CLASS(ex, generated: str) -> int | None:
    """Map a model's generated string back to an answer_position index.

    Returns None if the generated token doesn't match any symbol in the example.
    """
    trace = ex["input"]
    generated = generated.strip()
    for i in range(NUM_CHOICES):
        if generated == trace[f"symbol{i}"]:
            return i
    return None


def CLASS_TOKEN_IDS(ex, tokenizer) -> list[int]:
    """Return one token ID per class for this example.

    For MCQA, each class corresponds to a choice position whose answer symbol
    varies per example.  Returns [token_id_for_position_0, token_id_for_position_1, ...].
    """
    trace = ex["input"]
    ids = []
    for i in range(NUM_CHOICES):
        symbol = trace[f"symbol{i}"]
        toks = tokenizer.encode(f" {symbol}", add_special_tokens=False)
        ids.append(toks[-1])
    return ids


# ---------------------------------------------------------------------------
# Value-based scoring (the ``score_by: value`` mode)
# ---------------------------------------------------------------------------
#
# The default convention above scores the option *letter* (``symbol{i}``): the
# prompt ends ``…Y. orange\nAnswer:`` and the model is expected to emit ``Y``.
# A base model trained on MCQA-style data does that; an instruct/chat model
# fed the same prompt answers with the *value* — the colour word ``orange`` —
# and so scores ~0 under the letter contract even though it solved the task.
#
# Value mode is mode-agnostic: it accepts the choice *value* (the colour)
# OR the option *letter*, because both identify the correct choice and the
# chat instruct model is bimodal — it answers most MCQA questions with the
# colour but a minority with the letter (measured at seed 0 on
# Qwen3-4B-Instruct: 24/30 colour, 6/30 letter, exactly disjoint → 30/30
# accept-either, vs 0.8 colour-only / 0.2 letter-only). It is opt-in via
# ``task.score_by: value``. All ten colours and the A/B letters are
# single-token, so ``max_new_tokens=1`` and ``prob_accuracy`` are retained.


def _answer_value(ex) -> list[str]:
    """Accepted base-accuracy answers for value scoring: the correct colour
    OR its option letter.

    Returns both with a leading space (mirroring the letter convention's
    ``" " + answer``). ``compute_base_accuracy`` any-matches the list for
    accuracy and unions their token variants for prob_accuracy, and it tries
    both the space-prefixed and bare forms — so this captures the bare
    ``orange`` / ``B`` token the chat model actually emits. Accepting either
    notation is the correct MCQA contract (both name the right choice) and is
    what lets the golden clear the 0.9 floor on the bimodal chat model.
    """
    trace = ex["input"]
    return [" " + trace["color"], " " + trace["answer"]]


def _predict_class_value(ex, generated: str) -> int | None:
    """Map a generated string back to an answer_position by matching the colour.

    Value-mode counterpart of :func:`PREDICT_CLASS`: matches ``choice{i}`` (the
    colour) rather than ``symbol{i}`` (the letter). Returns None on no match.
    """
    trace = ex["input"]
    generated = generated.strip()
    for i in range(NUM_CHOICES):
        if generated == trace[f"choice{i}"]:
            return i
    return None


def _class_token_ids_value(ex, tokenizer) -> list[int]:
    """Return one colour-token ID per choice for this example.

    Value-mode counterpart of :func:`CLASS_TOKEN_IDS`. Uses the **bare** colour
    token (no leading space) because that is what the chat model emits as the
    first generated token.
    """
    trace = ex["input"]
    ids = []
    for i in range(NUM_CHOICES):
        color = trace[f"choice{i}"]
        toks = tokenizer.encode(color, add_special_tokens=False)
        ids.append(toks[-1])
    return ids


# Scoring conventions selectable via ``task.score_by`` (consumed by
# ``Task.apply_score_mode`` in causalab/tasks/loader.py). ``"letter"`` is the
# default: an empty override leaves the module-level PREDICT_CLASS /
# CLASS_TOKEN_IDS exports (and raw_output scoring) in place.
SCORE_MODES = {
    "letter": {},
    "value": {
        "answer": _answer_value,
        "predict_class": _predict_class_value,
        "class_token_ids": _class_token_ids_value,
    },
}
