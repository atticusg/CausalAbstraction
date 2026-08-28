"""Engine-neutral §8 services: tensor-bundle loading, input-role resolution,
and site identity records.

These were the reference engine's private helpers, moved here because a
second engine needs them verbatim (plan §2.4/§4.1): nothing in them touches a
hook, a trace, or a loaded model — they read the document and the resolution
environment.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path
from typing import Any

import torch

from causalab.protocol.engine import ExecutionRequest
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import Document

__all__ = [
    "BundlePoint",
    "TensorBundle",
    "load_tensors",
    "resolve_roles",
    "site_identity",
]


@dataclasses.dataclass(frozen=True)
class BundlePoint:
    """One producing point's slice of a bundle: the tensors sharing a
    coordinate suffix, plus that entry's stamped record.

    Slicing by suffix rather than selecting each slot on its own is what
    keeps a multi-slot bundle coherent — an SAE's ``enc`` and ``dec`` must
    come from the same fit, not from whichever entries each lookup found.
    """

    tensors: dict[str, torch.Tensor]
    suffix: str
    record: dict[str, Any]
    what: str

    def tensor(self, slot: str) -> torch.Tensor:
        key = f"{slot}{self.suffix}"
        if key not in self.tensors:
            raise ProtocolError(
                "P2",
                f"{self.what}: the bundle has no {key!r} — an entry's slots "
                f"must be complete (has {sorted(self.tensors)})",
            )
        return self.tensors[key]


@dataclasses.dataclass(frozen=True)
class TensorBundle:
    """One loaded ``.safetensors`` file: its tensors plus the ``entries``
    table from the header (§8, :mod:`causalab.protocol.bundles`).

    :meth:`point` is the only way in. A bundle written by a swept document
    holds one entry per point per slot, so asking for a bare slot name would
    either ``KeyError`` or — worse — silently take whichever entry a plain
    dict lookup happened to find.
    """

    tensors: dict[str, torch.Tensor]
    entry_coords: dict[str, Any]

    def point(
        self,
        slot: str,
        want: Any,
        *,
        what: str,
        implicit: bool = False,
    ) -> BundlePoint:
        """The entry for ``slot`` selected by ``want`` (a coordinate
        mapping; ``implicit`` when derived from the consuming point rather
        than authored), as a coherent slice of the bundle."""
        from causalab.protocol.bundles import select_entry

        key = select_entry(
            self.tensors.keys(),
            slot,
            want,
            what=what,
            coords_by_key=self.entry_coords or None,
            implicit=implicit,
        )
        record = self.entry_coords.get(key, {})
        return BundlePoint(
            tensors=self.tensors,
            suffix=key[len(slot) :],
            record=record if isinstance(record, dict) else {},
            what=what,
        )


def site_identity(doc: Document, site_name: str | None) -> dict[str, Any] | None:
    """One site as the ArtifactIdentity records it — the non-null address
    fields only, the shape ``loader.py`` builds its expectation in."""
    if site_name is None or site_name not in doc.sites:
        return None
    record = doc.sites[site_name]
    return {
        key: value
        for key, value in {
            "component": record.component,
            "layer": record.layer,
            "head": record.head,
            "expert": record.expert,
            "stream": record.stream,
        }.items()
        if value is not None
    }


def resolve_roles(
    doc: Document, request: ExecutionRequest
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Dataset rows + field selector per input role, rows paired by index.

    ``rows`` is part of the :class:`~causalab.protocol.resolve.DatasetResolver`
    contract, so this reads it directly — a resolver without it is a typing
    error at construction, not a surprise at run time."""
    rows_of = request.env.datasets.rows
    role_rows: dict[str, list[dict[str, Any]]] = {}
    role_fields: dict[str, str] = {}
    lengths: dict[str, int] = {}
    for role, value in doc.data.items():
        entries = value if isinstance(value, tuple) else (value,)
        for j, role_spec in enumerate(entries):
            role_name = role if not isinstance(value, tuple) else f"{role}[{j}]"
            role_rows[role_name] = rows_of(str(role_spec.dataset))
            role_fields[role_name] = str(role_spec.field)
            lengths[role_name] = len(role_rows[role_name])
    if len(set(lengths.values())) > 1:
        raise ProtocolError(
            "P2",
            f"input roles have unequal row counts {lengths} — rows are paired "
            "by index (§2.2)",
        )
    return role_rows, role_fields


@functools.lru_cache(maxsize=32)
def _read_bundle(path: str, _stamp: tuple[int, int]) -> TensorBundle:
    """One bundle, read once. The cache matters: a write operand resolves
    its ``params`` tensor on every application, so an uncached read would
    re-open the same file for every batch of every point.

    ``_stamp`` is the file's (mtime, size), so a path rewritten in the same
    process — a step re-run into an existing run tree — is a cache miss
    rather than a stale tensor."""
    from safetensors.torch import load_file

    from causalab.protocol.resolve import read_safetensors_metadata

    meta = read_safetensors_metadata(Path(path)) or {}
    raw_entries = meta.get("entries")
    entry_coords: dict[str, Any] = {}
    if isinstance(raw_entries, str):
        try:
            decoded = json.loads(raw_entries)
        except json.JSONDecodeError as err:
            raise ProtocolError(
                "P2", f"{path}: unreadable 'entries' table in the header — {err}"
            ) from err
        if isinstance(decoded, dict):
            entry_coords = decoded
    return TensorBundle(tensors=load_file(path), entry_coords=entry_coords)


def load_tensors(request: ExecutionRequest, file_path: str) -> TensorBundle:
    """Load a tensor bundle referenced by a featurizer/params file_path,
    resolved through the artifact store (which owns the run-tree/external
    overlay inside a workflow)."""
    artifacts = request.env.artifacts
    resolve = getattr(artifacts, "resolve_path", None)
    if resolve is not None:
        target = Path(resolve(file_path))
    else:
        root = getattr(artifacts, "root", None)
        if root is None:
            raise ProtocolError("P2", "artifact store exposes no filesystem root")
        target = Path(root) / file_path
    stat = target.stat()
    return _read_bundle(str(target), (stat.st_mtime_ns, stat.st_size))
