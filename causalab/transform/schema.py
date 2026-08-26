"""Slot and parameter vocabulary for transform ops (workflow spec §2.4).

Torch-free by construction. A :class:`~causalab.transform.registry.TransformOp`
record describes an op's *shape* — its parameters, its input and output slots,
and the columns of every table it writes — and this module is that vocabulary.
Keeping it free of numerics is what lets ``causalab validate`` / ``digest`` /
``explain`` refuse a bad document on a machine with no torch, before a single
step runs (docs/CODEBASE.md §1, "``protocol/`` is torch-free").

Two slot kinds, mirroring the two things a workflow step can already produce:
a :class:`Table` (a ``.parquet`` metric table) and a :class:`Tensor` (a
``.safetensors`` bundle). An **output** table must declare its columns — that
declaration is what a downstream ``select``/``plot`` step's column references
are checked against at load, the same load-time bite rule 7 has for a protocol
producer's sweep axes. An **input** table may leave them unconstrained.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

from causalab.protocol.errors import ProtocolError, suggest

__all__ = [
    "COLUMN_DTYPES",
    "Bool",
    "Float",
    "Int",
    "Param",
    "Slot",
    "Str",
    "Table",
    "Tensor",
    "TransformError",
    "validate_params",
]


class TransformError(ProtocolError):
    """An op record was violated: an unknown op, a bad parameter, a slot the
    op does not declare. Code ``T1``.

    The workflow loader re-raises this as a ``WorkflowError(1, …)`` so the
    document-level contract stays the §5 checklist; the runner lets it
    propagate, where it is a run-time refusal."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        #: the bare text, so the workflow loader can re-raise it under its own
        #: rule code instead of nesting ``[T1]`` inside ``[W1]``
        self.message = message
        super().__init__("T1", message, path=path)


#: Closed set of column dtypes a table slot may declare. Deliberately narrow:
#: these are the types that survive a parquet round-trip and a strict re-parse
#: by a consuming document (``protocol/workflow.py`` `_decode_cell`).
COLUMN_DTYPES: tuple[str, ...] = ("int64", "float64", "bool", "string")


# --------------------------------------------------------------------------- #
# slots
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Table:
    """A ``.parquet`` table slot.

    ``columns`` maps column name to a :data:`COLUMN_DTYPES` entry. It is
    required on an output slot and optional on an input slot — an op that
    consumes a metric table cannot know which sweep-coordinate columns the
    producing document stamped, so constraining them would be a lie."""

    columns: Mapping[str, str] | None = None
    description: str | None = None

    #: the file extension a document must give this slot (rule 8)
    suffix: str = ".parquet"


@dataclasses.dataclass(frozen=True)
class Tensor:
    """A ``.safetensors`` bundle slot holding one tensor."""

    description: str | None = None

    suffix: str = ".safetensors"


Slot = Table | Tensor


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Int:
    default: int | None = None
    min: int | None = None
    max: int | None = None
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class Float:
    default: float | None = None
    min: float | None = None
    max: float | None = None
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class Str:
    default: str | None = None
    choices: tuple[str, ...] | None = None
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class Bool:
    default: bool | None = None
    description: str | None = None


Param = Int | Float | Str | Bool


def _type_name(spec: Param) -> str:
    return {Int: "an integer", Float: "a number", Str: "a string", Bool: "a boolean"}[
        type(spec)
    ]


def _coerce(name: str, spec: Param, value: Any, path: str) -> Any:
    """One authored parameter value, type-checked and normalized.

    ``bool`` is rejected for the numeric kinds on purpose: it is an ``int``
    subclass in Python, so ``{"k": true}`` would silently mean ``k = 1``."""
    if isinstance(spec, Bool):
        if not isinstance(value, bool):
            raise TransformError(
                f"parameter {name!r} is a boolean, got {value!r}", path=path
            )
        return value
    if isinstance(value, bool):
        raise TransformError(
            f"parameter {name!r} is {_type_name(spec)}, got the boolean {value!r}",
            path=path,
        )
    if isinstance(spec, Int):
        if not isinstance(value, int):
            raise TransformError(
                f"parameter {name!r} is an integer, got {value!r}", path=path
            )
        return _bounded(name, value, spec.min, spec.max, path)
    if isinstance(spec, Float):
        if not isinstance(value, (int, float)):
            raise TransformError(
                f"parameter {name!r} is a number, got {value!r}", path=path
            )
        return float(_bounded(name, float(value), spec.min, spec.max, path))
    if not isinstance(value, str):
        raise TransformError(
            f"parameter {name!r} is a string, got {value!r}", path=path
        )
    if spec.choices is not None and value not in spec.choices:
        raise TransformError(
            f"parameter {name!r} rejects {value!r}{suggest(value, spec.choices)}",
            path=path,
        )
    return value


def _bounded(name: str, value: Any, low: Any, high: Any, path: str) -> Any:
    if low is not None and value < low:
        raise TransformError(
            f"parameter {name!r} is below its minimum {low}", path=path
        )
    if high is not None and value > high:
        raise TransformError(
            f"parameter {name!r} is above its maximum {high}", path=path
        )
    return value


def validate_params(
    schema: Mapping[str, Param], authored: Any, *, path: str
) -> dict[str, Any]:
    """Check an authored ``params`` table against an op's schema and return
    it with every default **materialized**.

    Materializing here is what puts the defaults into the canonical form and
    therefore the digest (§7) — the same treatment ``_canon_train`` gives
    optimizer defaults, and the reason a later change to a default is a
    visible loader migration rather than a silent renumbering."""
    if authored is None:
        authored = {}
    if not isinstance(authored, Mapping) or not all(
        isinstance(key, str) for key in authored
    ):
        raise TransformError("'params' maps parameter names to values", path=path)
    known: Sequence[str] = tuple(schema)
    for key in authored:
        if key not in schema:
            raise TransformError(
                f"unknown parameter {key!r}{suggest(key, known)}", path=path
            )
    out: dict[str, Any] = {}
    for name, spec in schema.items():
        if name in authored:
            out[name] = _coerce(name, spec, authored[name], f"{path}.{name}")
            continue
        if spec.default is None:
            raise TransformError(f"missing required parameter {name!r}", path=path)
        out[name] = spec.default
    return out
