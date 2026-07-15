"""
Hierarchical equality task for causal abstraction experiments.

Tests whether a model computes double equality: given four inputs (a, b, c, d),
checks if (a == b) == (c == d). Uses in-context learning with 60 balanced examples.
"""

from .causal_models import CAUSAL_MODEL
from .counterfactuals import COUNTERFACTUAL_GENERATORS

__all__ = ["CAUSAL_MODEL", "COUNTERFACTUAL_GENERATORS"]
