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
    NON_COLUMN_METRIC_FIELDS,
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
    """Everything a verb or an engine needs after one load."""

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
    engine_is_local: bool | None = None,
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
        validate_document(pdoc, engine_is_local=engine_is_local)
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


#: A ``<column>[j]`` field selector (§2.2). Same shape as the engine's
#: ``encoding._LIST_FIELD``; kept here so ``protocol/`` needs no import from
#: ``neural/`` to answer a question about a table.
_LIST_FIELD = re.compile(r"^([A-Za-z0-9_]+)\[(\d+)\]$")


def check_data_columns(loaded: LoadedProtocol, env: ResolutionEnv) -> list[str]:
    """The ``validate --data`` pass (§2.2): every dataset field selector, every
    metric column reference and every ``column``/``variable`` position must
    exist in the resolved tables. Returns the checked names (for reporting);
    raises on a miss.

    References are checked against the *union* of the resolved tables: rows are
    paired across roles (§2.2), and a metric or a column position addresses the
    row, not one role's text.

    **Every expanded point, not just the first.** A swept axis is a set of
    documents (§3), and the coordinate that names a bad column need not be
    coordinate 0 — ``weekdays_locate_scan`` swept its tap over
    ``[{"index": -1}, {"variable": "subject"}]`` and this pass, reading
    ``point_documents[0]``, only ever saw the index. Table reads are cached per
    dataset ref, so the cost is the number of distinct refs, not the number of
    points.

    A ``variable`` position is checked for **existence only**. What a prompt
    variable resolves to — a char span, hence a token count — needs a
    tokenizer, and the pure verbs hold none (``ResolutionEnv`` carries datasets
    and artifacts, and stays torch- and network-free). So a variable that no
    role can name is refused here, while a variable whose window turns out to
    be ragged across rows is still a run-time refusal ([V19]). That split is
    the whole of what is answerable without a model.
    """
    columns_by_ref: dict[str, set[str]] = {}
    variables_by_role: dict[tuple[str, str], set[str]] = {}

    def columns_of(ref: str) -> set[str]:
        if ref not in columns_by_ref:
            columns_by_ref[ref] = set(env.datasets.columns(ref))
        return columns_by_ref[ref]

    def variables_of(ref: str, field: str) -> set[str]:
        key = (ref, field)
        if key not in variables_by_role:
            variables_by_role[key] = _role_variables(env.datasets.rows(ref), field)
        return variables_by_role[key]

    refs: list[str] = []
    seen: set[tuple[str, str]] = set()

    def record(where: str, name: str) -> bool:
        """Report the reference, and say whether it still needs checking."""
        refs.append(name)
        if (where, name) in seen:
            return False
        seen.add((where, name))
        return True

    for doc in loaded.point_documents:
        columns: set[str] = set()
        variables: set[str] = set()
        for role_value in doc.data.values():
            roles = role_value if isinstance(role_value, tuple) else (role_value,)
            for role in roles:
                if not isinstance(role.dataset, str):
                    continue
                columns.update(columns_of(role.dataset))
                field = str(role.field)
                base_field = field.split("[", 1)[0]
                if base_field not in columns_of(role.dataset):
                    raise ValidationError(
                        4,
                        f"data field {field!r} is not a column of {role.dataset!r}",
                        path="data",
                    )
                variables.update(variables_of(role.dataset, field))
        for qname, metric in doc.metrics.items():
            for field, value in metric.fields.items():
                if (
                    metric.kind == "kl"
                    or field in NON_COLUMN_METRIC_FIELDS
                    or field in OPTIONAL_METRIC_FIELDS.get(str(metric.kind), ())
                    or not isinstance(value, str)
                ):
                    continue
                if not record(f"metrics.{qname}.{field}", value):
                    continue
                if value not in columns:
                    raise ValidationError(
                        4,
                        f"metric {qname!r} references column {value!r}, which none "
                        f"of the resolved datasets provide",
                        path=f"metrics.{qname}.{field}",
                    )
        for where, name in _column_position_refs(doc):
            if not record(where, name):
                continue
            if name not in columns:
                raise ValidationError(
                    4,
                    f"position {where} references column {name!r}, which none of "
                    f"the resolved datasets provide",
                    path=where,
                )
        # a prompt variable resolves per role: the role's <col>_variables
        # sibling first, then a same-named column (§2.3). Either spelling counts.
        resolvable = variables | columns
        for where, name in _variable_position_refs(doc):
            if not record(where, name):
                continue
            if name not in resolvable:
                raise ValidationError(
                    4,
                    f"position {where} references prompt variable {name!r}, which "
                    f"none of the resolved datasets provide — no role's "
                    f"'<field>_variables' names it and there is no {name!r} "
                    f"column (have {sorted(resolvable)})",
                    path=where,
                )
    return refs


def _role_variables(rows: list[dict[str, Any]], field: str) -> set[str]:
    """The prompt variables one data role can name, from its rows.

    Mirrors ``neural.shared.encoding.variable_value``, which is the authority
    at run time; duplicated rather than imported because ``protocol/`` stays
    torch-free (``test_load_is_torch_free``). Only the *sibling* half is here —
    the plain-column fallback is the caller's ``columns`` set.

    Union across rows, matching ``FileDatasets.columns``: a table whose rows
    disagree about their variables is a table defect, and refusing the
    *document* for it would point at the wrong thing."""
    match = _LIST_FIELD.match(field)
    column = match.group(1) if match else field
    index = int(match.group(2)) if match else None
    found: set[str] = set()
    for row in rows:
        sibling = row.get(f"{column}_variables")
        if index is not None and isinstance(sibling, list):
            sibling = sibling[index] if index < len(sibling) else None
        if isinstance(sibling, Mapping):
            found.update(str(key) for key in sibling)
    return found


def _variable_position_refs(doc: Document) -> list[tuple[str, str]]:
    """``(where, variable)`` for every prompt-variable position in a document —
    the named entries plus the inline specs on reads and writes, and the
    ``scope``/``relative_to`` anchors spelled as a variable (§2.3)."""
    found: list[tuple[str, str]] = []

    def visit(where: str, spec: Any) -> None:
        if not isinstance(spec, PositionSpec):
            return
        if isinstance(spec.variable, str):
            found.append((where, spec.variable))
        anchor = spec.scope if spec.scope is not None else spec.relative_to
        if spec.anchor_source == "variable" and isinstance(anchor, str):
            found.append((where, anchor))

    for name, entry in doc.positions.items():
        visit(f"positions.{name}", entry)
    for section, table in (("reads", doc.reads), ("writes", doc.writes)):
        for name, spec in table.items():
            visit(f"{section}.{name}.pos", spec.pos)
    return found


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
