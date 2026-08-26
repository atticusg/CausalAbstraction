"""The load pipeline: text → validated, expanded, canonicalized protocol.

One entry point, :func:`load`, owns the order the spec implies:

1. read + strict-parse the authored tree (rules 1–2, shape checks);
2. resolve artifact-valued fields against the environment (rule 15);
3. expand sweeps into point protocols (rule 14, point cap);
4. parse + validate every point (rules 3–13 — a point is exactly as valid
   as the same document written by hand);
5. canonicalize: the document form (wrappers intact — the campaign) and
   every point form (fully materialized — the provenance units), with
   their digests.

``--set path=value`` overrides (§9) are applied to the authored tree before
anything else — exploration only; the digest of an overridden document is
the overridden document's digest, so the record never lies.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Mapping

from causalab.protocol import canonical as _canonical
from causalab.protocol.bundles import select_entry, selector_slot
from causalab.protocol.errors import ParseError, ValidationError
from causalab.protocol.method import (
    document_type,
    is_split,
    method_digest,
    split_document,
)
from causalab.protocol.resolve import ResolutionEnv, resolve_artifact_fields
from causalab.protocol.schema import (
    FEATURIZER_SLOTS,
    OPTIONAL_METRIC_FIELDS,
    Document,
    PositionSpec,
    load_raw,
    parse_document,
)
from causalab.protocol.sweep import DEFAULT_POINT_CAP, Expansion, expand
from causalab.protocol.validate import validate_document

__all__ = ["LoadedProtocol", "apply_overrides", "flatten", "load", "load_text"]


@dataclasses.dataclass(frozen=True)
class LoadedProtocol:
    """Everything a verb or a backend needs after one load."""

    document: Document
    raw: Mapping[str, Any]
    expansion: Expansion
    point_documents: tuple[Document, ...]
    canonical_document: Mapping[str, Any]
    document_digest: str
    canonical_points: tuple[Mapping[str, Any], ...]
    point_digests: tuple[str, ...]
    #: The method this document was composed from (§1.1), when it was written
    #: in split form: its content hash, and the ``method`` reference when the
    #: method came from a reusable file rather than inline. The *composed*
    #: document digests as if it had been written flat, so method provenance is
    #: reported and stamped, never folded into the canonical bytes (§7).
    method_digest: str | None = None
    method_ref: str | None = None


def load_text(path: Path) -> dict[str, Any]:
    """Read one authored file. JSON is the normative surface; ``.yaml`` /
    ``.yml`` parse through a duplicate-key-rejecting SafeLoader into the
    same object model (strict keys hold on both surfaces, §5.1)."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        raw = _load_yaml(text)
        if not isinstance(raw, dict):
            raise ParseError("P1", "the top level must be a mapping")
        _check_json_values(raw)
        return raw
    return load_raw(text)


def _load_yaml(text: str) -> Any:
    import yaml  # the optional authoring surface — not a load-path dependency

    class _StrictLoader(yaml.SafeLoader):
        pass

    def _mapping(loader: Any, node: Any) -> dict[Any, Any]:
        out: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in out:
                raise ParseError("P2", f"duplicate key {key!r} in one object")
            out[key] = loader.construct_object(value_node)
        return out

    _StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
    )
    try:
        return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 — SafeLoader subclass
    except yaml.YAMLError as err:
        raise ParseError("P1", f"not valid YAML: {err}") from err


def _check_json_values(raw: Any, *, _path: str = "") -> None:
    """The JSON object model is normative (§0): every mapping key is a
    string and every number is finite — whatever surface (YAML, an artifact
    store) produced the tree."""
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if not isinstance(key, str):
                raise ParseError(
                    "P2",
                    f"mapping key {key!r} is not a string — quote it at the "
                    "authoring surface",
                    path=_path or None,
                )
            _check_json_values(value, _path=f"{_path}.{key}" if _path else str(key))
    elif isinstance(raw, list):
        for item in raw:
            _check_json_values(item, _path=_path)
    elif isinstance(raw, float) and (
        raw != raw or raw in (float("inf"), float("-inf"))
    ):
        raise ParseError(
            "P2",
            f"non-finite number at {_path or '<root>'} — the object model is JSON",
            path=_path or None,
        )


def flatten(
    raw: Mapping[str, Any], *, base_dir: Path | None = None
) -> tuple[dict[str, Any], str | None, str | None]:
    """One protocol document as a flat tree, whichever form it was written in
    (§1.1). Returns the flat document, the method's content digest, and the
    ``method`` reference when the method came from a file.

    Everything that addresses fields by path — ``--set`` overrides, a workflow
    step's ``set`` block, the run verb's model pre-registration — flattens
    first, so a dotted path means the same thing in both forms.
    """
    if not is_split(raw):
        return dict(raw), None, None
    composed, method_raw, method_ref = split_document(raw, base_dir=base_dir)
    return composed, method_digest(method_raw), method_ref


def load(
    source: Path | Mapping[str, Any],
    env: ResolutionEnv,
    *,
    overrides: Mapping[str, Any] | None = None,
    point_cap: int | None = DEFAULT_POINT_CAP,
    backend_is_local: bool | None = None,
    base_dir: Path | None = None,
) -> LoadedProtocol:
    """Load one protocol document through the full pipeline.

    ``base_dir`` is where a relative ``method`` reference resolves from when
    the document arrives as a tree rather than a file (a workflow step reads
    its inner document itself); a document loaded from a path uses its own
    directory. An inlined method — the usual case, one file per run — needs
    neither.

    A *split* document (§1.1) is flattened first: the composition is an
    ordinary protocol document, and everything after this line — overrides,
    artifact fields, sweeps, validation, canonicalization — cannot tell how the
    document was authored. ``--set`` paths therefore address the *composed*
    document, whichever form it was written in.
    """
    raw = dict(load_text(source)) if isinstance(source, Path) else dict(source)
    kind = document_type(raw)
    if kind == "method":
        raise ValidationError(
            18,
            "this is a method file: it names no network, no data and no "
            "addresses, so there is nothing to run. Bind it from a document's "
            "`application` half (§1.1), or ask for its signature with "
            "`causalab explain`.",
            path="type",
        )
    raw, method_digest_value, method_ref = flatten(
        raw, base_dir=source.parent if isinstance(source, Path) else base_dir
    )
    if overrides:
        raw = apply_overrides(raw, overrides)
    # artifact fields resolve first (§1: legal anywhere a value is), then the
    # strict parse gates the fully-literal authored form
    resolved = resolve_artifact_fields(raw, env)
    _check_json_values(resolved)
    parse_document(resolved)  # authored-form shape gate (sweeps intact)
    expansion = expand(resolved, point_cap=point_cap)
    point_documents: list[Document] = []
    for point in expansion.points:
        pdoc = parse_document(point.raw)
        validate_document(pdoc, backend_is_local=backend_is_local)
        _check_loaded_featurizers(pdoc, env)
        point_documents.append(pdoc)
    canonical_document = _canonical.canonicalize(resolved, env)
    canonical_points = tuple(
        _canonical.canonicalize(point.raw, env) for point in expansion.points
    )
    return LoadedProtocol(
        document=parse_document(resolved),
        raw=resolved,
        expansion=expansion,
        point_documents=tuple(point_documents),
        canonical_document=canonical_document,
        document_digest=_canonical.digest(canonical_document),
        canonical_points=canonical_points,
        point_digests=tuple(_canonical.digest(c) for c in canonical_points),
        method_digest=method_digest_value,
        method_ref=method_ref,
    )


def _check_loaded_featurizers(doc: Document, env: ResolutionEnv) -> None:
    """§2.5/§8: every ``file_path`` load is checked at load time. A
    featurizer bundle's stamped ArtifactIdentity must match what the
    document implies (model, site record, k, parametrization, dtype); a
    ``params`` entry's file must exist, and its identity — when stamped —
    must name the same model (free constant tensors may come from outside
    causalab, so an unstamped params file is existence-checked only; a
    stamped one must not contradict the document).

    A bundle written by a *swept* producer stamps per entry, not per file
    (§8): the fields that differ between points live in the header's
    ``entries`` table. The check therefore looks at the record of the entry
    this document selects, whenever that entry is knowable here — an
    authored ``entry``, or a bundle holding exactly one record for the slot.
    A selection that only the executing point can make (implicit matching
    off its own coordinates, §2.5) is checked when the stage is built, where
    the point is known."""
    import dataclasses as _dc

    from causalab.protocol.resolve import check_artifact_identity

    defers = getattr(env.artifacts, "defers", None)

    for pname, pspec in doc.params.items():
        if not isinstance(pspec.file_path, str):
            continue
        if defers is not None and defers(pspec.file_path):
            continue  # a run-tree path inside a workflow — checked at run time
        stamped = env.artifacts.read_identity(pspec.file_path)  # V15 if missing
        if stamped is not None:
            stamped = _entry_identity(
                stamped,
                slot=selector_slot(pspec.entry, "value"),
                authored=pspec.entry,
                what=f"params entry {pname!r} ({pspec.file_path})",
            )
        if stamped is not None:
            check_artifact_identity(
                stamped,
                {"model_key": doc.model.key, "model_revision": doc.model.revision},
                what=f"params entry {pname!r} ({pspec.file_path})",
            )

    for fname, spec in doc.featurizers.items():
        if not isinstance(spec.file_path, str):
            continue
        if defers is not None and defers(spec.file_path):
            continue  # a run-tree path inside a workflow — checked at run time
        used_sites: list[str] = []
        for entry in (*doc.reads.values(), *doc.writes.values()):
            ref = entry.featurizer
            chain = (
                (ref,)
                if isinstance(ref, str)
                else tuple(ref)
                if isinstance(ref, tuple)
                else ()
            )
            if fname in chain and str(entry.site) not in used_sites:
                used_sites.append(str(entry.site))
        realization = _canonical.canonical_model(doc.raw["model"])
        expected: dict[str, Any] = {
            "model_key": doc.model.key,
            "model_revision": doc.model.revision,
            # the realization the bundle was fitted against is part of its
            # identity (§8): a rotation fitted in bf16 does not apply to fp32
            # activations just because the shapes agree
            "model_dtype": realization["dtype"],
            "model_quantization": realization.get("quantization"),
            "k": spec.k,
            "parametrization": spec.parametrization,
            "dtype": spec.dtype if spec.dtype is not None else "fp32",
        }
        if len(used_sites) == 1:
            site = doc.sites[used_sites[0]]
            expected["site"] = {
                key: value
                for key, value in _dc.asdict(site).items()
                if value is not None
            }
        what = f"featurizer {fname!r} ({spec.file_path})"
        stamped = env.artifacts.read_identity(spec.file_path)
        slot = FEATURIZER_SLOTS.get(
            spec.kind if isinstance(spec.kind, str) else "identity", ()
        )
        resolved = (
            _entry_identity(stamped, slot=slot[0], authored=spec.entry, what=what)
            if stamped is not None and slot
            else stamped
        )
        if resolved is None:
            continue  # only the executing point can select — checked at build
        check_artifact_identity(
            resolved,
            {key: value for key, value in expected.items() if value is not None},
            what=what,
        )


def _entry_identity(
    stamped: Mapping[str, Any],
    *,
    slot: str,
    authored: Any,
    what: str,
) -> Mapping[str, Any] | None:
    """The identity of the one bundle entry a spec selects: the file-level
    stamp, overlaid with that entry's record from the header's ``entries``
    table (§8).

    Returns ``None`` when the table holds several candidates and the
    document authored no ``entry`` — the selection is then the executing
    point's, and so is the check. A bundle with no table at all is
    file-level only, which is exactly what an un-swept or hand-made bundle
    stamps.
    """
    raw = stamped.get("entries")
    if not isinstance(raw, str):
        return stamped
    try:
        table = json.loads(raw)
    except json.JSONDecodeError:
        return stamped
    if not isinstance(table, dict) or not table:
        return stamped
    try:
        key = select_entry(
            table.keys(),
            slot,
            authored,
            what=what,
            coords_by_key=table,
        )
    except ValidationError as err:
        if authored:
            raise  # an authored selection that misses is a load error
        if "selects none" in str(err):
            return None
        raise
    record = table[key]
    merged = {k: v for k, v in stamped.items() if k != "entries"}
    merged.update({k: v for k, v in record.items() if k not in ("slot", "coords")})
    return merged


_INDEX = re.compile(r"^(.*)\[(\d+)\]$")

#: Dotted paths an override may *create*. An override normally has to hit a
#: field that exists — inventing structure is how a typo becomes an
#: experiment. These two are the exception because the document is never
#: really silent about them: canonicalization materializes both (§7), so
#: setting one fills a default rather than adding a field.
CREATABLE_PATHS: frozenset[str] = frozenset({"model.dtype", "model.revision"})


def apply_overrides(
    raw: dict[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply ``--set path=value`` overrides (§9): dotted paths, ``[i]`` for
    list entries, values as JSON (bare words fall back to strings). The
    path must exist — an override that would *create* structure is a typo,
    not an experiment — except for :data:`CREATABLE_PATHS`, the fields the
    canonical form materializes whether or not they are authored."""
    out = json.loads(json.dumps(raw))  # deep copy, stays plain JSON types
    for dotted, value in overrides.items():
        node: Any = out
        parts = dotted.split(".")
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            match = _INDEX.match(part)
            key, index = (
                (match.group(1), int(match.group(2))) if match else (part, None)
            )
            if not isinstance(node, dict) or (
                key not in node and not (last and dotted in CREATABLE_PATHS)
            ):
                raise ParseError(
                    "P2",
                    f"--set {dotted}: {key!r} does not exist in the document",
                    path=dotted,
                )
            if last and index is None:
                node[key] = value
            elif index is not None:
                target = node[key]
                if not isinstance(target, list) or index >= len(target):
                    raise ParseError(
                        "P2",
                        f"--set {dotted}: {key}[{index}] is out of range",
                        path=dotted,
                    )
                if last:
                    target[index] = value
                else:
                    node = target[index]
            else:
                node = node[key]
    return out


def check_data_columns(loaded: LoadedProtocol, env: ResolutionEnv) -> list[str]:
    """The ``validate --data`` pass (§2.2): every dataset field selector, every
    metric column reference and every ``column`` position must exist in the
    resolved tables. Returns the checked column names (for reporting); raises
    on a miss.

    Column references are checked against the *union* of the resolved tables'
    columns: rows are paired across roles (§2.2), and a metric or a column
    position addresses the row, not one role's text."""
    doc = loaded.point_documents[0]
    columns: set[str] = set()
    refs: list[str] = []
    for role_value in doc.data.values():
        roles = role_value if isinstance(role_value, tuple) else (role_value,)
        for role in roles:
            if isinstance(role.dataset, str):
                columns.update(env.datasets.columns(role.dataset))
                field = str(role.field)
                base_field = field.split("[", 1)[0]
                if base_field not in env.datasets.columns(role.dataset):
                    raise ValidationError(
                        4,
                        f"data field {field!r} is not a column of {role.dataset!r}",
                        path="data",
                    )
    for qname, metric in doc.metrics.items():
        for field, value in metric.fields.items():
            if (
                metric.kind == "kl"
                or field in ("k", *OPTIONAL_METRIC_FIELDS.get(str(metric.kind), ()))
                or not isinstance(value, str)
            ):
                continue
            refs.append(value)
            if value not in columns:
                raise ValidationError(
                    4,
                    f"metric {qname!r} references column {value!r}, which none of "
                    f"the resolved datasets provide",
                    path=f"metrics.{qname}.{field}",
                )
    for where, name in _column_position_refs(doc):
        refs.append(name)
        if name not in columns:
            raise ValidationError(
                4,
                f"position {where} references column {name!r}, which none of the "
                f"resolved datasets provide",
                path=where,
            )
    return refs


def _column_position_refs(doc: Document) -> list[tuple[str, str]]:
    """``(where, column)`` for every ``column`` position in a document — the
    named entries plus the inline specs on reads and writes (§2.3)."""
    found: list[tuple[str, str]] = []

    def visit(where: str, spec: Any) -> None:
        if not isinstance(spec, PositionSpec):
            return
        if isinstance(spec.column, str):
            found.append((where, spec.column))
        anchor = spec.scope if spec.scope is not None else spec.relative_to
        if spec.anchor_source == "column" and isinstance(anchor, str):
            found.append((where, anchor))

    for name, entry in doc.positions.items():
        visit(f"positions.{name}", entry)
    for section, table in (("reads", doc.reads), ("writes", doc.writes)):
        for name, spec in table.items():
            visit(f"{section}.{name}.pos", spec.pos)
    return found
