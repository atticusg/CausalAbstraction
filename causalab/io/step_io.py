"""Reading a step script's inputs and writing its outputs.

Two formats, so two pairs of functions (workflow spec §2.5): JSON — metric
tables and values objects — and ``.safetensors`` bundles for dense numerics.
A script imports what it needs; nothing forces it to use any of this.

The division of labour with the runner is deliberate. A script **writes its own
files**, because a plot step and a report step both want to, and paying for two
contracts to spare op tests a ``tmp_path`` fixture is not worth it. But
**identity stamping stays the runner's job** (:func:`stamp_tensor`): a bundle
carrying no ArtifactIdentity is refused when a later protocol step loads it, so
a script that forgot to stamp would produce a file that fails much later, in
someone else's step. The runner cannot forget.

Provenance of a script-written bundle — three sources, in order:

1. **inherited** from the step's tensor inputs, keeping only the fields they
   all agree on (:func:`inherited_identity`, the ``record_common`` rule from
   ``pytorch_hooks/outputs.py``) — a fit over activations from model X at site
   S really is bound to X and S;
2. **stamped** by the runner: ``produced_by`` (the step's digest — its
   provenance unit), ``backend`` and ``dtype``;
3. never ``commit``: the code identity of a script output is its
   ``script_sha256``, which is in the document's canonical form and therefore
   in the digest. A git sha would say less and drift more.

Heavy imports (pandas, safetensors, torch) are function-local, so this module
is importable and its declarations readable without them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.protocol.bundles import entry_key, select_entry
from causalab.protocol.errors import ProtocolError
from causalab.protocol.resolve import (
    ARTIFACT_IDENTITY_KEYS,
    build_artifact_identity,
    read_safetensors_metadata,
)
from causalab.protocol.tables import read_table, write_table

__all__ = [
    "StepError",
    "entry_identity",
    "frame",
    "inherited_identity",
    "read_table",
    "read_tensor",
    "read_values",
    "stamp_tensor",
    "write_frame",
    "write_table",
    "write_tensor",
    "write_values",
]


class StepError(ProtocolError):
    """A step script's IO contract was violated. Code ``S1``."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        self.message = message
        super().__init__("S1", message, path=path)


#: torch dtype → the protocol's ``precision`` spelling (``protocol/schema.py``
#: ``PRECISION_DTYPES``), so a stamped ``dtype`` is comparable with what a
#: consuming featurizer spec declares.
_DTYPE_NAMES = {
    "torch.float32": "fp32",
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
}


# --------------------------------------------------------------------------- #
# JSON: tables and values objects
# --------------------------------------------------------------------------- #


def frame(path: Path) -> Any:
    """One JSON metric table as a pandas DataFrame.

    pandas is the in-memory shape a reduction wants; it is never the storage
    format."""
    import pandas as pd

    return pd.DataFrame(read_table(Path(path)))


def write_frame(df: Any, path: Path) -> None:
    """A DataFrame as a JSON metric table."""
    write_table(Path(path), df.to_dict(orient="records"))


def read_values(path: Path) -> dict[str, Any]:
    """One values object — a flat JSON mapping, what a ``key`` selector reads."""
    target = Path(path)
    if not target.is_file():
        raise StepError(f"values file {str(target)!r} does not exist")
    with target.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise StepError(
            f"{target.name} is not a values object (a JSON mapping of name to "
            "value) — an array of row objects is a metric table instead"
        )
    return payload


def write_values(path: Path, values: Mapping[str, Any]) -> None:
    """Write a values object, the shape an ``outputs.<slot>.keys`` declaration
    promises."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(values), indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# safetensors
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


def entry_identity(metadata: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    """The identity of one bundle entry: the file-level stamp, overridden by
    whatever the ``entries`` table records for that key (the per-entry fields a
    swept producer could not stamp file-wide)."""
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
    slot: str | None = None,
    entry: Mapping[str, Any] | None = None,
    what: str | None = None,
) -> Any:
    """One tensor out of a ``.safetensors`` bundle.

    Selection reuses :func:`causalab.protocol.bundles.select_entry`, so a
    single-entry bundle needs no selector and an ambiguous one refuses with a
    listing instead of picking first. ``implicit`` is always ``False``: a script
    step has no sweep coordinates of its own, so there is nothing to match
    implicitly — a swept producer must be addressed by an authored ``entry``."""
    tensor, _ = read_tensor_with_identity(path, slot=slot, entry=entry, what=what)
    return tensor


def read_tensor_with_identity(
    path: Path,
    *,
    slot: str | None = None,
    entry: Mapping[str, Any] | None = None,
    what: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """:func:`read_tensor`, plus the entry's identity — what the runner needs
    in order to inherit provenance."""
    from safetensors.torch import load_file

    target = Path(path)
    label = what or str(target)
    if not target.is_file():
        raise StepError(f"{label}: input tensor {str(target)!r} does not exist")
    metadata = read_safetensors_metadata(target)
    tensors = load_file(str(target))
    entries = _entry_table(metadata)
    key = select_entry(
        tensors.keys(),
        slot if slot is not None else _sole_slot(tensors.keys(), entries, label),
        entry,
        what=label,
        coords_by_key=entries or None,
        implicit=False,
    )
    return tensors[key], entry_identity(metadata, key)


def _sole_slot(keys: Any, entries: Mapping[str, Mapping[str, Any]], what: str) -> str:
    """The slot name when the document did not say which.

    A bundle written by one un-swept producer holds one slot, which is then
    unambiguous; anything else has to be named, because guessing is exactly the
    failure ``select_entry`` exists to prevent."""
    from causalab.protocol.bundles import RAGGED_SUFFIX, parse_entry_key

    slots = set()
    for key in keys:
        if str(key).endswith(RAGGED_SUFFIX):
            continue
        stored = entries.get(str(key), {})
        slots.add(str(stored.get("slot", parse_entry_key(str(key))[0])))
    if len(slots) == 1:
        return slots.pop()
    raise StepError(
        f"{what}: the bundle holds slots {sorted(slots)} — name one with "
        "'slot' in the input's selector"
    )


def write_tensor(
    path: Path,
    tensor: Any,
    *,
    slot: str = "weight",
    identity: Mapping[str, Any] | None = None,
) -> None:
    """Write one tensor as a single-entry bundle.

    The runner stamps provenance and inherited identity afterwards
    (:func:`stamp_tensor`), so a script cannot produce a bundle that a later
    protocol step will refuse for lack of a stamp.

    ``identity`` is for the fields **only the script knows** — a fitted basis's
    rank ``k``, say, which is a parameter rather than something inheritable from
    the input. A consuming featurizer spec's identity check requires those, so a
    script that produces a loadable artifact has to declare them. Inherited
    fields win on conflict: what the inputs prove beats what a script claims."""
    import torch
    from safetensors.torch import save_file

    if not isinstance(tensor, torch.Tensor):
        raise StepError(f"expected a tensor, got {type(tensor).__name__}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().to("cpu").contiguous()
    metadata = {str(k): str(v) for k, v in identity.items()} if identity else None
    save_file({entry_key(slot, ""): value}, str(target), metadata=metadata)


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


def stamp_tensor(path: Path, identity: Mapping[str, Any], *, what: str) -> None:
    """Stamp an ArtifactIdentity onto a bundle a script just wrote.

    Rewrites the file with its tensors unchanged and the metadata filled in —
    safetensors headers are not editable in place, and a bundle is small enough
    that a round-trip is cheaper than teaching every script to stamp."""
    import torch
    from safetensors.torch import load_file, save_file

    from causalab.protocol.bundles import RAGGED_SUFFIX, parse_entry_key

    target = Path(path)
    try:
        tensors = load_file(str(target))
    except Exception as err:  # noqa: BLE001 — any unreadable bundle is the same bug
        raise StepError(f"{what}: not a readable .safetensors bundle: {err}") from err
    existing = read_safetensors_metadata(target) or {}
    # what the script declared about fields only it knows (a basis's rank),
    # overlaid by what the inputs *prove* — inherited beats claimed
    declared = {
        field: value
        for field, value in existing.items()
        if field in ARTIFACT_IDENTITY_KEYS
    }
    stamped = {**declared, **dict(identity)}
    dtypes = {
        str(value.dtype)
        for key, value in tensors.items()
        if isinstance(value, torch.Tensor) and not key.endswith(RAGGED_SUFFIX)
    }
    if len(dtypes) == 1:
        sole = dtypes.pop()
        stamped.setdefault("dtype", _DTYPE_NAMES.get(sole, sole))
    metadata = build_artifact_identity(**stamped)
    entries = _entry_table(existing)
    if not entries:
        entries = {
            key: {"slot": parse_entry_key(str(key))[0], "coords": {}}
            for key in tensors
            if not str(key).endswith(RAGGED_SUFFIX)
        }
    metadata["entries"] = json.dumps(entries, sort_keys=True)
    save_file(tensors, str(target), metadata=metadata)
