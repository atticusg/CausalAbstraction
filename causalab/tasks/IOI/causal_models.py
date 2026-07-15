"""Causal model for the IOI (Indirect Object Identification) task.

Canonical IOI: a sentence introduces two distinct named entities at positions
A and B, then names one of them as the subject (``name_C``) of a giving
action. The model must predict the *other* name — the indirect object (IO).
For example::

    After Alice and Bob went to the park, Alice gave a ball to ____   →   Bob

Input variables:
    template:  prompt template (single canonical form for the coverage runner).
    name_A:    first introduced name.
    name_B:    second introduced name (distinct from ``name_A``).
    name_C:    subject of the giving action — equal to either ``name_A`` or
               ``name_B``.
    place:     filler for ``{place}``.
    object:    filler for ``{object}``.

Computed variables:
    IO:          the indirect object — whichever of ``name_A`` / ``name_B``
                 differs from ``name_C``. The target variable.
    raw_input:   filled template string.
    raw_output:  ``" " + IO`` (the model's expected next-token output).

This is a coverage-oriented task: it exists so the runner pipeline can
anchor a smoke + golden tier on ``causalab.tasks.IOI.*`` (see docs/TESTS.md
"Coverage-oriented runners"). It is **not** a scientifically meaningful
run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from causalab.causal.causal_model import CausalModel, build_output_tokens
from causalab.causal.trace import CausalTrace, Mechanism, input_var


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load(path: Path) -> list[str]:
    with open(path, "r") as f:
        return json.load(f)


IOI_DIR = Path(__file__).resolve().parent
NAMES: list[str] = _load(IOI_DIR / "names.json")
OBJECTS: list[str] = _load(IOI_DIR / "objects.json")
PLACES: list[str] = _load(IOI_DIR / "places.json")
ALL_TEMPLATES: list[str] = _load(IOI_DIR / "templates.json")

# Coverage runner uses a single canonical template — keeps token-position
# logic simple and avoids the multi-template factory plumbing. The other
# templates remain available via ``ALL_TEMPLATES`` for ad-hoc work.
CANONICAL_TEMPLATE: str = (
    "After {name_A} and {name_B} went to the {place}, {name_C} gave a {object} to"
)
TEMPLATES: list[str] = [CANONICAL_TEMPLATE]


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------


def _fill_template(t: CausalTrace) -> str:
    """Render ``raw_input`` from the input variables."""
    s = t["template"]
    s = s.replace("{name_A}", t["name_A"])
    s = s.replace("{name_B}", t["name_B"])
    s = s.replace("{name_C}", t["name_C"])
    s = s.replace("{place}", t["place"])
    s = s.replace("{object}", t["object"])
    return s


def _compute_io(t: CausalTrace) -> str:
    """The indirect object: the name in ``{name_A, name_B}`` that is *not*
    ``name_C``.

    Sampling rejects ill-formed inputs via ``_input_filter``, but the
    causal-trace machinery eagerly recomputes descendants before the filter
    runs, so this function tolerates ``name_C ∉ {name_A, name_B}``. In that
    case the trace is ill-formed and will be rejected by the filter; we
    return ``name_A`` as a placeholder so eager recomputation doesn't crash.
    """
    name_A = t["name_A"]
    name_B = t["name_B"]
    name_C = t["name_C"]
    if name_C == name_B:
        return name_A
    # Covers the well-formed ``name_C == name_A`` case AND the ill-formed
    # ``name_C ∉ {name_A, name_B}`` fallback (filter rejects the latter).
    return name_B


def _compute_raw_output(t: CausalTrace) -> str:
    """Expected next-token output: a leading space then the IO name."""
    return " " + t["IO"]


def _input_filter(t: CausalTrace) -> bool:
    """Reject inputs where the IOI question is ill-formed.

    Requires ``name_A != name_B`` (distinct entities) and
    ``name_C in {name_A, name_B}`` (subject is one of the introduced names).
    """
    name_A = t["name_A"]
    name_B = t["name_B"]
    name_C = t["name_C"]
    return name_A != name_B and name_C in (name_A, name_B)


# ---------------------------------------------------------------------------
# Causal model
# ---------------------------------------------------------------------------


def _build_causal_model() -> CausalModel:
    mechanisms: dict[str, Mechanism] = {
        "template": input_var(TEMPLATES),
        "name_A": input_var(NAMES),
        "name_B": input_var(NAMES),
        "name_C": input_var(NAMES),
        "place": input_var(PLACES),
        "object": input_var(OBJECTS),
        "IO": Mechanism(
            parents=["name_A", "name_B", "name_C"],
            compute=_compute_io,
        ),
        "raw_input": Mechanism(
            parents=["template", "name_A", "name_B", "name_C", "place", "object"],
            compute=_fill_template,
        ),
        "raw_output": Mechanism(
            parents=["IO"],
            compute=_compute_raw_output,
        ),
    }

    values: dict[str, list | None] = {
        "template": TEMPLATES,
        "name_A": NAMES,
        "name_B": NAMES,
        "name_C": NAMES,
        "place": PLACES,
        "object": OBJECTS,
        "IO": NAMES,
        "raw_input": None,
        "raw_output": None,
    }

    return CausalModel(
        mechanisms=mechanisms,
        values=values,
        id="ioi",
        # The answer is the indirect-object name (``raw_output = " " + IO``);
        # the model emits a single name token, so exact match on the declared
        # ``[" name", "name"]`` forms is the right grader (#296). This derives
        # the checker in place of the former checker.py.
        output_tokens={"IO": build_output_tokens(NAMES)},
        input_filter=_input_filter,
    )


positional_causal_model = _build_causal_model()


# ---------------------------------------------------------------------------
# Standard exports for load_task()
# ---------------------------------------------------------------------------

CAUSAL_MODEL = positional_causal_model
TARGET_VARIABLE = "IO"
TEMPLATE = CANONICAL_TEMPLATE
