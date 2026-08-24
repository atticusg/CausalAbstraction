"""The closed-but-growable registry of transform ops (workflow spec §2.4).

A ``transform`` step names its op as ``name@version``. Both halves are part of
the record: two runs of the same document must agree numerically, so pinning
``fit_pca@1`` means a behavioural change ships as ``fit_pca@2`` and documents
written against the old one keep digesting — and running — as written. The
op's *implementation* is never in the digest, the same rule that keeps backends
out of protocol digests.

**Determinism is the admission criterion.** An op must be a pure function of
its declared inputs and params; anything stochastic takes an explicit ``seed``
parameter and must be bit-stable across devices. An op that cannot meet that
does not belong here, and that refusal is the point of a closed set — the
registry grows by pull request, never by document.

The registry is **torch-free**: op modules keep their numerics inside the
function body (``import torch`` at call time, the idiom the runner already uses
for pandas and matplotlib), so importing this module to *validate* a document
costs nothing but stdlib. ``tests/test_architecture_layering.py`` enforces it.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable, Mapping

from causalab.protocol.errors import suggest
from causalab.protocol.resolve import ARTIFACT_IDENTITY_KEYS
from causalab.transform.schema import (
    COLUMN_DTYPES,
    Param,
    Slot,
    Table,
    TransformError,
)

__all__ = ["TransformOp", "lookup", "op_ids", "register"]

_OP_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

#: The callable an op registers: ``(inputs, params) -> {slot: value}``. Values
#: are in-memory objects (a DataFrame for a table slot, a tensor for a tensor
#: slot); the *runner* owns paths, file formats and identity stamping, so an op
#: stays a pure function and its unit test needs no filesystem.
#: Typed as returning ``object`` rather than the mapping it owes: an op body is
#: ordinary Python, so the runner *checks* the shape it got instead of trusting
#: an annotation to have been true.
OpCallable = Callable[..., object]


@dataclasses.dataclass(frozen=True)
class TransformOp:
    """One registered op: everything the loader needs, plus the body."""

    name: str
    version: int
    params: Mapping[str, Param]
    inputs: Mapping[str, Slot]
    outputs: Mapping[str, Slot]
    fn: OpCallable
    #: ``{identity field: parameter name}`` — identity a tensor output gets
    #: from its own params rather than by inheritance from a tensor input
    #: (``fit_pca@1`` sets the featurizer rank ``k`` this way).
    identity_from_params: Mapping[str, str] = dataclasses.field(default_factory=dict)
    description: str | None = None

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


_REGISTRY: dict[str, TransformOp] = {}


def _check_record(op: TransformOp) -> None:
    """Refuse a malformed record at import time — a registry entry is part of
    the protocol's surface, so a bad one is a bug in this repository, not a
    user error."""
    if not _OP_NAME.match(op.name):
        raise AssertionError(f"op name {op.name!r} is not a lower_snake identifier")
    if op.version < 1:
        raise AssertionError(f"op {op.name!r} has a non-positive version")
    if not op.outputs:
        raise AssertionError(f"op {op.id} declares no outputs")
    for slot, decl in op.outputs.items():
        if not isinstance(decl, Table):
            continue
        if decl.columns is None:
            raise AssertionError(
                f"op {op.id} output {slot!r} is a table with no declared columns — "
                "a consuming select/plot step's column references are checked "
                "against that declaration at load"
            )
        for column, dtype in decl.columns.items():
            if dtype not in COLUMN_DTYPES:
                raise AssertionError(
                    f"op {op.id} output {slot!r} column {column!r} has unknown "
                    f"dtype {dtype!r} (one of {list(COLUMN_DTYPES)})"
                )
    for field, param in op.identity_from_params.items():
        if field not in ARTIFACT_IDENTITY_KEYS:
            raise AssertionError(
                f"op {op.id} maps unknown ArtifactIdentity field {field!r}"
            )
        if param not in op.params:
            raise AssertionError(
                f"op {op.id} maps identity {field!r} from undeclared parameter "
                f"{param!r}"
            )


def register(
    *,
    name: str,
    version: int,
    inputs: Mapping[str, Slot],
    outputs: Mapping[str, Slot],
    params: Mapping[str, Param] | None = None,
    identity_from_params: Mapping[str, str] | None = None,
    description: str | None = None,
) -> Callable[[OpCallable], OpCallable]:
    """Register one op under ``name@version``.

    Returns the function unchanged, so an op's unit test can call it directly
    without going through the registry."""

    def decorate(fn: OpCallable) -> OpCallable:
        op = TransformOp(
            name=name,
            version=version,
            params=dict(params or {}),
            inputs=dict(inputs),
            outputs=dict(outputs),
            fn=fn,
            identity_from_params=dict(identity_from_params or {}),
            description=description
            or (fn.__doc__ or "").strip().split("\n")[0]
            or None,
        )
        _check_record(op)
        if op.id in _REGISTRY:
            raise AssertionError(f"op {op.id} is registered twice")
        _REGISTRY[op.id] = op
        return fn

    return decorate


def _ensure_ops_imported() -> None:
    """Populate the registry. Importing the op modules runs their decorators;
    it does **not** import torch, which stays inside the function bodies.

    By module path rather than by name: the import is for its side effect, and
    nothing here uses the module object."""
    import importlib

    importlib.import_module("causalab.transform.ops")


def op_ids() -> tuple[str, ...]:
    """Every registered ``name@version``, sorted."""
    _ensure_ops_imported()
    return tuple(sorted(_REGISTRY))


def lookup(op_id: Any, *, path: str | None = None) -> TransformOp:
    """The op a document names, or a refusal with suggestions.

    An unknown *version of a known op* gets its own message: the usual
    did-you-mean over the whole id would suggest ``fit_pca@1`` for
    ``fit_pca@2`` without saying why, and "this op has versions 1" is the
    thing the author actually needs to read."""
    _ensure_ops_imported()
    if not isinstance(op_id, str):
        raise TransformError(
            f"'op' is a string 'name@version', got {op_id!r}", path=path
        )
    name, sep, version_text = op_id.partition("@")
    if not sep or not version_text:
        raise TransformError(
            f"op {op_id!r} is not spelled 'name@version' — the version is part "
            "of the record, so a document pins the numerics it was written "
            f"against{suggest(op_id, op_ids())}",
            path=path,
        )
    known_versions = sorted(op.version for op in _REGISTRY.values() if op.name == name)
    if not known_versions:
        raise TransformError(
            f"unknown op {name!r}{suggest(name, sorted({op.name for op in _REGISTRY.values()}))}",
            path=path,
        )
    if not version_text.isdigit():
        raise TransformError(
            f"op version {version_text!r} is not an integer (op {name!r} has "
            f"{_versions_text(known_versions)})",
            path=path,
        )
    op = _REGISTRY.get(f"{name}@{int(version_text)}")
    if op is None:
        raise TransformError(
            f"op {name!r} has no version {int(version_text)} — it has "
            f"{_versions_text(known_versions)}",
            path=path,
        )
    return op


def _versions_text(versions: list[int]) -> str:
    joined = ", ".join(str(v) for v in versions)
    return f"version{'s' if len(versions) > 1 else ''} {joined}"
