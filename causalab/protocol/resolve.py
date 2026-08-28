"""Resolution: the services a document is loaded *against*.

A protocol document references things outside itself — datasets by ref,
prior runs' artifacts, a model's static config. The spec makes resolving
them part of loading (§1 artifact-valued fields, §2.2 dataset digests,
§5.15), but *what* they resolve against is an environment, not a global:
tests resolve against fixture files, production against task-generated
datasets and real run directories. :class:`ResolutionEnv` bundles the three
services; everything here is stdlib-only.

* **Artifacts** — ``{"artifact": "<ref>", "key": "<field>"}`` reads one
  value from a prior run at load. A ref names a JSON value table:
  ``<root>/<ref>.json`` or ``<root>/<ref>/values.json`` (first hit wins).
  Missing artifact or key = load error, never a default (§5.15).
* **Datasets** — a ref is a local path (relative to the data root) holding
  a serialized table; the resolver reports its content digest (stamped into
  the canonical form, §2.2), its columns (checked by ``validate --data``)
  and its rows (what a run consumes). The repo's task datasets are
  *generated*, but they are generated **ahead of the load**, by
  :mod:`causalab.tasks.serialize`, and enter here as ordinary tables — so
  resolution stays stdlib-only and a document's digest never depends on
  importing task code or a tokenizer.
* **Models** — static config metadata via
  :mod:`causalab.protocol.registry`.

Featurizer ``file_path`` artifacts get their existence checked here and
their ``ArtifactIdentity`` (§8) read from the safetensors header — the
header is a JSON prefix, so no tensor library is involved at load.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from causalab.protocol.errors import ValidationError
from causalab.protocol.registry import ModelInfo, get_model_info

__all__ = [
    "ArtifactStore",
    "DatasetResolver",
    "FileArtifacts",
    "FileDatasets",
    "ResolutionEnv",
    "read_safetensors_metadata",
    "resolve_artifact_fields",
]


class ArtifactStore(Protocol):
    """Where prior runs' outputs are found."""

    def read_value(self, artifact: str, key: str) -> Any: ...

    def file_digest(self, file_path: str) -> str: ...

    def read_identity(self, file_path: str) -> Mapping[str, Any] | None: ...


class DatasetResolver(Protocol):
    """Where dataset refs resolve. ``digest`` is the content digest stamped
    into canonical forms; ``columns`` backs ``validate --data``; ``rows`` is
    the table content a run consumes.

    All three are one contract on purpose. The pure verbs
    (``validate``/``explain``/``digest``) only need the first two, but a
    resolver that cannot produce rows cannot back a ``run`` — so the
    requirement is declared here instead of being discovered by a ``getattr``
    probe deep inside an engine."""

    def digest(self, ref: str) -> str: ...

    def columns(self, ref: str) -> tuple[str, ...]: ...

    def rows(self, ref: str) -> list[dict[str, Any]]: ...


@dataclasses.dataclass(frozen=True)
class ResolutionEnv:
    """The three resolution services a load runs against."""

    datasets: DatasetResolver
    artifacts: ArtifactStore
    model_info: Callable[[str], ModelInfo] = get_model_info


# --------------------------------------------------------------------------- #
# artifact-valued fields (§1, §5.15)
# --------------------------------------------------------------------------- #


def resolve_artifact_fields(
    raw: Any,
    env: ResolutionEnv,
    *,
    _path: str = "",
    _seen: frozenset[tuple[str, str]] = frozenset(),
) -> Any:
    """Replace every ``{"artifact": …, "key": …}`` node in a raw tree with
    the value it reads — recursively, so an artifact may itself store a
    reference (a cycle is a load error, not a hang). Runs before the parse
    gate, so a ref is legal anywhere a value is (§1); a mapping that
    *looks* like a ref but is malformed refuses rather than loading as a
    literal dict."""
    if isinstance(raw, Mapping):
        if isinstance(raw.get("artifact"), str):
            if set(raw) != {"artifact", "key"} or not isinstance(raw.get("key"), str):
                raise ValidationError(
                    15,
                    f"malformed artifact reference {dict(raw)!r} — the shape is "
                    '{"artifact": "<ref>", "key": "<field>"} exactly (§1)',
                    path=_path,
                )
            pair = (str(raw["artifact"]), str(raw["key"]))
            if pair in _seen:
                raise ValidationError(
                    15, f"artifact reference cycle through {pair!r}", path=_path
                )
            try:
                value = env.artifacts.read_value(*pair)
            except (FileNotFoundError, KeyError) as err:
                raise ValidationError(
                    15,
                    f"artifact-valued field did not resolve: {err} — a missing "
                    "artifact is a load error, never a default (§1)",
                    path=_path,
                ) from err
            return resolve_artifact_fields(
                value, env, _path=_path, _seen=_seen | {pair}
            )
        return {
            key: resolve_artifact_fields(
                value, env, _path=f"{_path}.{key}" if _path else key, _seen=_seen
            )
            for key, value in raw.items()
        }
    if isinstance(raw, list):
        return [
            resolve_artifact_fields(item, env, _path=_path, _seen=_seen) for item in raw
        ]
    return raw


# --------------------------------------------------------------------------- #
# file-backed implementations
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class FileArtifacts:
    """Artifacts under one root directory (a run-output tree)."""

    root: Path

    def _value_table(self, artifact: str) -> Mapping[str, Any]:
        for candidate in (
            self.root / f"{artifact}.json",
            self.root / artifact / "values.json",
        ):
            if candidate.is_file():
                table = json.loads(candidate.read_text())
                if not isinstance(table, dict):
                    raise ValidationError(
                        15, f"artifact {artifact!r} is not a JSON object of values"
                    )
                return table
        raise FileNotFoundError(f"no artifact {artifact!r} under {self.root}")

    def read_value(self, artifact: str, key: str) -> Any:
        table = self._value_table(artifact)
        if key not in table:
            raise KeyError(
                f"artifact {artifact!r} has no key {key!r} (has {sorted(table)})"
            )
        return table[key]

    def file_digest(self, file_path: str) -> str:
        target = self.root / file_path
        if not target.is_file():
            raise ValidationError(
                15, f"artifact file {file_path!r} not found under {self.root} (§5.15)"
            )
        return hashlib.sha256(target.read_bytes()).hexdigest()

    def read_identity(self, file_path: str) -> Mapping[str, Any] | None:
        target = self.root / file_path
        if not target.is_file():
            raise ValidationError(
                15, f"artifact file {file_path!r} not found under {self.root} (§5.15)"
            )
        return read_safetensors_metadata(target)

    def resolve_path(self, file_path: str) -> Path:
        return self.root / file_path


def read_safetensors_metadata(path: Path) -> Mapping[str, Any] | None:
    """The ``__metadata__`` table of a safetensors file — a pure header read
    (8-byte little-endian header length, then a JSON object), no tensor
    library involved. Returns ``None`` when the file carries no metadata.

    Format reference: https://github.com/huggingface/safetensors#format.
    """
    with path.open("rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise ValidationError(
                15, f"{path} is not a safetensors file (truncated header)"
            )
        (header_len,) = struct.unpack("<Q", prefix)
        header = json.loads(fh.read(header_len))
    meta = header.get("__metadata__")
    return meta if isinstance(meta, Mapping) else None


# --------------------------------------------------------------------------- #
# ArtifactIdentity (§8)
# --------------------------------------------------------------------------- #

#: The stamped-identity schema for featurizer bundles: these keys live in the
#: safetensors ``__metadata__`` table (string-valued, per the format). The
#: engine stamps them at save; the loader refuses a ``file_path`` load whose
#: stamped values contradict the document (§2.5).
#:
#: ⚠️ **Migration.** ``model_dtype`` and ``model_quantization`` joined this
#: schema when precision entered the record (§2.1). A bundle fitted before
#: that carries neither, so it no longer matches a document that names them
#: and ``_check_loaded_featurizers`` refuses it. That is intended and not a
#: bug to route around — a rotation fitted in bf16 is not the same artifact as
#: one fitted in fp32, and pretending otherwise is what the stamp exists to
#: prevent. Locally kept fitted artifacts must be re-fitted once; nothing in
#: the repo ships one.
ARTIFACT_IDENTITY_KEYS: tuple[str, ...] = (
    "produced_by",
    "model_key",
    "model_revision",
    "model_dtype",
    "model_quantization",
    "tokenizer",
    "site",
    "k",
    "parametrization",
    "dtype",
    "trained_on",
    "trained_on_digest",
    "engine",
    "commit",
)


def build_artifact_identity(**fields: Any) -> dict[str, str]:
    """Stringify identity fields for a safetensors ``__metadata__`` table
    (the format only carries ``str -> str``). Unknown keys are refused so
    the schema stays closed; absent fields are simply not stamped."""
    unknown = set(fields) - set(ARTIFACT_IDENTITY_KEYS)
    if unknown:
        raise AssertionError(f"unknown ArtifactIdentity fields {sorted(unknown)}")
    return {
        key: value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        for key, value in fields.items()
        if value is not None
    }


def check_artifact_identity(
    stamped: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
    *,
    what: str,
) -> None:
    """Refuse a loaded bundle whose stamped identity contradicts the
    document (§2.5). A bundle with no identity at all is refused too — an
    unverifiable artifact is a provenance hole, not a pass."""
    if stamped is None:
        raise ValidationError(
            15,
            f"{what}: the artifact carries no ArtifactIdentity metadata — "
            "nothing to check, so the load refuses (§2.5)",
        )
    normalized_expected = build_artifact_identity(**expected)
    for key, want in normalized_expected.items():
        got = stamped.get(key)
        if got is not None and str(got) != want:
            raise ValidationError(
                15,
                f"{what}: ArtifactIdentity mismatch on {key!r} — the document "
                f"implies {want!r} but the bundle was stamped {got!r} (§2.5)",
            )
        if got is None:
            raise ValidationError(
                15,
                f"{what}: ArtifactIdentity is missing {key!r} — the bundle "
                "cannot prove it matches the document (§2.5)",
            )


@dataclasses.dataclass(frozen=True)
class FileDatasets:
    """Dataset refs as serialized JSON tables under one data root.

    A ref ``weekdays/train`` resolves to ``<root>/weekdays/train.json`` — a
    JSON array of row objects. The content digest is the sha256 of the
    file's bytes; columns are the union of row keys. Task-generated tables
    are written by :mod:`causalab.tasks.serialize` into this same layout
    (deterministically, so the digest is reproducible), with their build
    provenance in a ``<ref>.manifest.json`` sidecar that nothing here reads.
    """

    root: Path

    def _file(self, ref: str) -> Path:
        candidate = self.root / f"{ref}.json"
        if not candidate.is_file():
            candidate = self.root / ref
        if not candidate.is_file():
            raise ValidationError(
                4, f"dataset {ref!r} not found under {self.root}", path="data"
            )
        return candidate

    def digest(self, ref: str) -> str:
        return hashlib.sha256(self._file(ref).read_bytes()).hexdigest()

    def columns(self, ref: str) -> tuple[str, ...]:
        rows = self.rows(ref)
        cols: set[str] = set()
        for row in rows:
            cols.update(row)
        return tuple(sorted(cols))

    def rows(self, ref: str) -> list[dict[str, Any]]:
        rows = json.loads(self._file(ref).read_text())
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise ValidationError(
                4, f"dataset {ref!r} is not a JSON array of row objects"
            )
        return rows
