"""Candidate causal model(s) and the hypotheses defined over them.

This is a SCAFFOLD. Replace the toy addition example with the causal model(s)
for your task. It ships working so that ``run_hypothesis_generation.py`` runs
before you edit it.

A hypothesis here is a (causal model, target-variable subset) pair: the claim
that those variables are the faithful abstraction of some neural location.

Two design rules carry most of the weight. See the surrounding
`hypothesis-generation.md` document for the full design guidance.

1. ONE VALUE PER VARIABLE (near-hard rule). One variable per conceptual unit,
   each holding a single value -- do NOT bundle several things into a
   list/dict/tuple value. (Values may be discrete, like a weekday or a boolean,
   or numeric; the point is one thing, not a collection.) A hypothesis about a
   bundled variable can only ask about a representation of the WHOLE structure;
   separate variables let you localize each piece. The model will look like an
   over-engineered version of a one-line program -- that is the point, not a smell.

2. INDEFINITE ARITY via a factory. To support an arbitrary number of entities /
   positions / slots, generate a FAMILY of separate variables in a loop over
   config dimensions (a factory function), rather than one list-valued variable.
   See entity_binding's ``create_positional_entity_causal_model`` looping over
   max_groups x max_entities_per_group to emit entity_g{g}_e{e}, query_e{e},
   positional_query_e{e}, ... -- arbitrarily many, each still a single variable.

The hypothesis-generation harness reads these values from the bottom of this file:
    MODELS         : name -> CausalModel
    DEFAULT_MODEL  : which model the auto-injected null/all reference hypotheses use
    HYPOTHESES     : name -> (model name, [target variables])
    TARGETS        : the focal hypotheses (a GROUP, not necessarily one) that the
                     alternatives are scored against

The two reference hypotheses are injected automatically if absent:
    "null" : (default model, [])              -- intervene on nothing
    "all"  : (default model, ["raw_output"])  -- transplant the whole output
"""

from __future__ import annotations

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var

# ---------------------------------------------------------------------------
# Example: single-digit addition with an explicit carry. raw_output is
# reconstructed FROM carry and ones so both intermediates lie on a live path to
# the output (if raw_output read `total` directly, carry/ones would be dead-end
# leaves -- indistinguishable from the null no matter the counterfactual).
# ---------------------------------------------------------------------------

DIGITS = list(range(10))

_values = {
    "X": DIGITS,
    "Y": DIGITS,
    "total": list(range(19)),
    "carry": [False, True],
    "ones": DIGITS,
    "raw_input": None,
    "raw_output": None,
}

_mechanisms = {
    "X": input_var(DIGITS),
    "Y": input_var(DIGITS),
    "total": Mechanism(parents=["X", "Y"], compute=lambda t: t["X"] + t["Y"]),
    "carry": Mechanism(parents=["total"], compute=lambda t: t["total"] >= 10),
    "ones": Mechanism(parents=["total"], compute=lambda t: t["total"] % 10),
    "raw_input": Mechanism(parents=["X", "Y"], compute=lambda t: f"{t['X']}+{t['Y']}="),
    "raw_output": Mechanism(
        parents=["carry", "ones"],
        compute=lambda t: str(int(t["carry"]) * 10 + t["ones"]),
    ),
}

addition = CausalModel(_mechanisms, _values, id="addition_with_carry")


# ---------------------------------------------------------------------------
# Registries read by the hypothesis-generation harness
# ---------------------------------------------------------------------------

MODELS: dict[str, CausalModel] = {"addition": addition}
DEFAULT_MODEL = "addition"

HYPOTHESES: dict[str, tuple[str, list[str]]] = {
    "carry": ("addition", ["carry"]),
    "ones": ("addition", ["ones"]),
    # "null" and "all" are injected automatically (see the docstring) -- do not
    # list them by hand.
}

# The focal hypothesis/hypotheses. Alternatives (everything else, plus the
# injected null/all) are scored against each target as interpretive baselines.
TARGETS: list[str] = ["carry"]
