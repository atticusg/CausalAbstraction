"""Reading a transform step's inputs and writing its outputs.

The runner owns this seam, not the ops: an op is a pure
``(inputs, params) -> {slot: value}`` function, so paths, file formats and
provenance stamping all live here. Both formats are the ones the rest of the
stack already uses — JSON for tables, a ``.safetensors`` bundle with an
``entries`` table and an ArtifactIdentity for tensors — because a transform op
is **not** a new tensor-passing channel; it reuses the one #29 built.

Heavy imports (pandas, safetensors, torch) are function-local: this module is
importable, and its declarations readable, without them.

Provenance of a transform-written bundle
----------------------------------------
``check_artifact_identity`` refuses a bundle carrying no identity, so a tensor
a later *protocol* step loads has to be stamped. Three sources, in order:

1. **inherited** from the tensor inputs, keeping only the fields they all
   agree on (the ``record_common`` rule, ``pytorch_hooks/outputs.py``) — a fit
   over activations from model X at site S really is bound to X and S;
2. **from params**, for the fields the op's own parameters define rather than
   inherit (``identity_from_params``, e.g. a fitted basis's rank ``k``);
3. **stamped here**: ``produced_by`` (the step's digest — its provenance unit),
   ``backend`` and ``dtype``.

``commit`` is deliberately *not* stamped. The code identity of a transform
output is its op's **version**, which is in the document and therefore in the
digest; a git sha would say less and drift more.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.protocol.bundles import entry_key, select_entry
from causalab.protocol.resolve import (
    ARTIFACT_IDENTITY_KEYS,
    build_artifact_identity,
    read_safetensors_metadata,
)
from causalab.transform.schema import Table, TransformError

__all__ = [
    "inherited_identity",
    "read_table",
    "read_tensor",
    "write_table",
    "write_tensor",
]

#: torch dtype → the protocol's ``precision`` spelling (``protocol/schema.py``
#: ``PRECISION_DTYPES``), so a stamped ``dtype`` is comparable with what a
#: consuming featurizer spec declares.
_DTYPE_NAMES = {
    "torch.float32": "fp32",
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
}


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #


def read_table(path: Path) -> Any:
    """One JSON metric table as a DataFrame.

    The file is an array of row objects (``protocol.tables``); pandas is only
    the in-memory shape an op body wants, never the storage format."""
    import pandas as pd

    from causalab.protocol.tables import read_table as read_rows

    if not path.is_file():
        raise TransformError(f"input table {str(path)!r} does not exist")
    return pd.DataFrame(read_rows(path))


def write_table(frame: Any, path: Path, decl: Table, *, what: str) -> None:
    """Write a table output, checked against the op's declared columns.

    The declaration is the contract a downstream ``select``/``plot`` step was
    validated against at load, so an op that returns something else is refused
    here rather than producing a table whose columns silently disagree with
    the document that was checked."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TransformError(f"{what}: expected a table, got {type(frame).__name__}")
    columns = dict(decl.columns or {})
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise TransformError(
            f"{what}: the op declares columns {sorted(columns)} but returned "
            f"{sorted(map(str, frame.columns))} — missing {missing}"
        )
    extra = [str(name) for name in frame.columns if name not in columns]
    if extra:
        raise TransformError(
            f"{what}: the op returned undeclared columns {sorted(extra)} — a "
            "table's columns are part of its record (§2.4)"
        )
    ordered = frame[list(columns)]
    for name, dtype in columns.items():
        ordered = ordered.astype({name: "string" if dtype == "string" else dtype})
    from causalab.protocol.tables import write_table as write_rows

    write_rows(path, ordered.to_dict(orient="records"))


# --------------------------------------------------------------------------- #
# tensors
# --------------------------------------------------------------------------- #


def _entry_table(metadata: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not metadata:
        return {}
    raw = metadata.get("entries")
    if not isinstance(raw, str):
        return {}
    try:
        table = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return table if isinstance(table, dict) else {}


def _identity_of(metadata: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    """The identity of one bundle entry: the file-level stamp, overridden by
    whatever the ``entries`` table records for that key (the per-entry fields
    a swept producer could not stamp file-wide)."""
    if not metadata:
        return {}
    identity = {
        field: value
        for field, value in metadata.items()
        if field in ARTIFACT_IDENTITY_KEYS
    }
    entry = _entry_table(metadata).get(key, {})
    identity.update(
        {
            field: value
            for field, value in entry.items()
            if field in ARTIFACT_IDENTITY_KEYS
        }
    )
    return identity


def read_tensor(
    path: Path,
    *,
    slot: str | None,
    entry: Mapping[str, Any] | None,
    what: str,
) -> tuple[Any, dict[str, Any]]:
    """One tensor out of a ``.safetensors`` bundle, plus its identity.

    Selection reuses :func:`causalab.protocol.bundles.select_entry`, so a
    single-entry bundle needs no selector and an ambiguous one refuses with a
    listing instead of picking first. ``implicit`` is always ``False``: a
    transform step has no sweep coordinates of its own, so there is nothing to
    match implicitly — a swept producer must be addressed by an authored
    ``entry``."""
    from safetensors.torch import load_file

    if not path.is_file():
        raise TransformError(f"{what}: input tensor {str(path)!r} does not exist")
    metadata = read_safetensors_metadata(path)
    tensors = load_file(str(path))
    entries = _entry_table(metadata)
    key = select_entry(
        tensors.keys(),
        slot if slot is not None else _sole_slot(tensors.keys(), entries, what),
        entry,
        what=what,
        coords_by_key=entries or None,
        implicit=False,
    )
    return tensors[key], _identity_of(metadata, key)


def _sole_slot(keys: Any, entries: Mapping[str, Mapping[str, Any]], what: str) -> str:
    """The slot name when the document did not say which.

    A bundle written by one un-swept producer holds one slot, which is then
    unambiguous; anything else has to be named, because guessing is exactly
    the failure ``select_entry`` exists to prevent."""
    from causalab.protocol.bundles import RAGGED_SUFFIX, parse_entry_key

    slots = set()
    for key in keys:
        if str(key).endswith(RAGGED_SUFFIX):
            continue
        stored = entries.get(str(key), {})
        slots.add(str(stored.get("slot", parse_entry_key(str(key))[0])))
    if len(slots) == 1:
        return slots.pop()
    raise TransformError(
        f"{what}: the bundle holds slots {sorted(slots)} — name one with "
        "'slot' in the input's entry selector"
    )


def inherited_identity(sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The identity fields every tensor input agrees on.

    Same rule as ``TensorFile.record_common``: a field the sources disagree
    about says nothing true about the output, so it is dropped rather than
    letting one input speak for the result."""
    if not sources:
        return {}
    common = {
        field: value
        for field, value in sources[0].items()
        if field in ARTIFACT_IDENTITY_KEYS
    }
    for other in sources[1:]:
        for field in list(common):
            if str(common[field]) != str(other.get(field)):
                del common[field]
    return common


def write_tensor(
    tensor: Any,
    path: Path,
    *,
    slot: str,
    identity: Mapping[str, Any],
    what: str,
) -> None:
    """Write one tensor as a single-entry bundle, stamped so a protocol step
    may load it (module docstring)."""
    import torch
    from safetensors.torch import save_file

    if not isinstance(tensor, torch.Tensor):
        raise TransformError(f"{what}: expected a tensor, got {type(tensor).__name__}")
    value = tensor.detach().to("cpu").contiguous()
    key = entry_key(slot, "")
    stamped = dict(identity)
    stamped.setdefault("dtype", _DTYPE_NAMES.get(str(value.dtype), str(value.dtype)))
    metadata = build_artifact_identity(**stamped)
    metadata["entries"] = json.dumps(
        {key: {"slot": slot, "coords": {}}}, sort_keys=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value}, str(path), metadata=metadata)
