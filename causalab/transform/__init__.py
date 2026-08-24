"""Deterministic, versioned transforms over a workflow's saved artifacts.

The fourth workflow step type (docs/workflow_protocol.md §2.4) runs an ``op``
from the registry here: a pure function of its declared inputs and parameters,
turning a table or a tensor into another table or tensor. Fitting a basis,
aggregating per-head statistics, running a paired t-test — analyses that touch
no model, and that before this lived outside the record as post-hoc notebook
work.

Nothing in this package imports torch at module level: the op *records* — name,
version, parameter schema, input and output slots, and the columns of every
table an op writes — are what the pure CLI verbs read to refuse a bad document,
and they must stay cheap. The numerics live inside the op function bodies.
"""

from __future__ import annotations

from causalab.transform.registry import TransformOp, lookup, op_ids, register
from causalab.transform.schema import (
    Bool,
    Float,
    Int,
    Str,
    Table,
    Tensor,
    TransformError,
    validate_params,
)

__all__ = [
    "Bool",
    "Float",
    "Int",
    "Str",
    "Table",
    "Tensor",
    "TransformError",
    "TransformOp",
    "lookup",
    "op_ids",
    "register",
    "validate_params",
]
