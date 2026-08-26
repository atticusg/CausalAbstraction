"""Methods and applications: the two halves of a protocol document (§1.1).

A protocol document answers two different questions at once. *What is the
experiment* — the causal hypothesis, which values are read, what is written
into whom, how the result is scored — and *what was it run on* — which
network, at which addresses, in which precision. The first half transfers
between models; the second half is exactly the part that cannot.

So a document may be authored as two files:

* a **method** (``"type": "method"``) — a protocol document with the
  network-facing fields left open. It declares the site *names* it addresses
  (``"target": {}``) without saying where they are, and it never declares
  ``model``: a method that named a network would not be a method.
* an **application** — the binding. It names the method, supplies ``model``
  (key, revision, dtype, quantization) and the addresses, and may fill in
  anything else the method left open.

:func:`compose` puts them back together, and the composition is an ordinary
protocol document: it validates, expands, canonicalizes and digests exactly
as the same experiment written as one file. That transparency is the point —
splitting a document is an authoring choice, never a second dialect, and a
point protocol keeps digesting identically however it was reached (§7).

The composition rule is one sentence: **an application may complete the
method, never contradict it.** Every leaf comes from exactly one side, or
from both with the same value (a restatement, cross-checked like the ``save``
manifest's bindings, §2.12). A leaf neither side supplies is a hole, and an
unfilled hole is a load error naming what is missing — not a parse failure
somewhere downstream.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

from causalab.protocol.errors import ParseError, ValidationError, suggest
from causalab.protocol.schema import (
    DOCUMENT_TYPES,
    LAYERLESS_COMPONENTS,
    NAMED_SECTIONS,
    RESERVED_NAMES,
    SECTION_ORDER,
)

__all__ = [
    "MethodDocument",
    "MethodSignature",
    "compose",
    "document_type",
    "is_method",
    "method_digest",
    "parse_method",
    "resolve_method_source",
    "signature_of",
]

#: Sections an application supplies on top of a method. Everything else is
#: legal on both sides — the split is a boundary an author draws, and the
#: only structural rule is that the *method* never names the network.
_APPLICATION_ONLY: frozenset[str] = frozenset({"model"})


def document_type(raw: Mapping[str, Any]) -> str:
    """Which of the four document types ``raw`` is (§1.1).

    Structure decides — ``steps`` is a workflow, ``method`` is an application,
    anything else is a protocol document — except for methods, which are the
    one shape structure cannot name (a method is a protocol document with
    pieces missing, and so is a broken protocol document). A method therefore
    declares ``"type": "method"``, and any document *may* declare its type to
    have the claim checked.
    """
    declared = raw.get("type")
    if declared is not None and declared not in DOCUMENT_TYPES:
        raise ParseError(
            "P4",
            f"unknown document type {declared!r} — one of {list(DOCUMENT_TYPES)}"
            f"{suggest(str(declared), DOCUMENT_TYPES)}",
            path="type",
        )
    if declared == "method":
        return "method"
    structural, because = (
        ("workflow", "it has a 'steps' section")
        if "steps" in raw
        else ("application", "it has a 'method' section")
        if "method" in raw
        else ("protocol", "it has neither a 'method' nor a 'steps' section")
    )
    if declared is not None and declared != structural:
        raise ParseError(
            "P2",
            f"this file declares type {declared!r} but reads as a {structural} "
            f"document — {because}",
            path="type",
        )
    return structural


def is_method(raw: Mapping[str, Any]) -> bool:
    """``True`` for a method document — the one type that cannot be run."""
    return document_type(raw) == "method"


@dataclasses.dataclass(frozen=True)
class MethodSignature:
    """What an application must supply to close one method.

    ``sites`` maps a declared site name to the address fields still missing;
    ``data`` names the input roles the method reads but does not bind;
    ``model`` is always in the signature (a method never names a network).
    """

    model: bool
    sites: Mapping[str, tuple[str, ...]]
    data: tuple[str, ...]

    def is_closed(self) -> bool:
        return not self.model and not self.data and not any(self.sites.values())

    def lines(self) -> tuple[str, ...]:
        """One human-readable line per thing still to bind."""
        out: list[str] = []
        if self.model:
            out.append("model: key, revision, dtype (+ optional quantization)")
        for name, fields in self.sites.items():
            if fields:
                out.append(f"sites.{name}: {', '.join(fields)}")
        for role in self.data:
            out.append(f"data.{role}: dataset, field")
        return tuple(out)


@dataclasses.dataclass(frozen=True)
class MethodDocument:
    """One parsed method: the raw tree plus the signature it exposes."""

    raw: Mapping[str, Any]
    signature: MethodSignature
    description: str | None = None


def parse_method(raw: Mapping[str, Any]) -> MethodDocument:
    """Strict-parse a method document.

    A method is checked for everything that is checkable without a network:
    its own key set and section order (§5.1–5.2), one global namespace
    (§5.3), no reserved names, every site it addresses declared, and the two
    sections that make it a method at all — ``reads`` (what it measures) and
    ``save`` (what leaves the run). The rest of the §5 checklist needs the
    addresses, and runs on the composition.
    """
    if raw.get("type") != "method":
        raise ParseError(
            "P2",
            'a method document declares "type": "method" — it is the one '
            "document type that cannot be recognized by its shape",
            path="type",
        )
    unknown = [key for key in raw if key not in SECTION_ORDER]
    if unknown:
        raise ParseError(
            "P3",
            f"unknown section {unknown[0]!r}{suggest(unknown[0], SECTION_ORDER)}",
            path=unknown[0],
        )
    _check_order(list(raw), what="method")
    if raw.get("version") != "1":
        raise ParseError(
            "P2",
            f"unsupported version {raw.get('version')!r}; this loader reads "
            'version "1"',
        )
    for section in _APPLICATION_ONLY:
        if section in raw:
            raise ValidationError(
                18,
                f"a method declares no {section!r} — naming the network is what "
                "makes a document an application (§1.1)",
                path=section,
            )
    for section in ("reads", "save"):
        if section not in raw:
            raise ValidationError(
                18,
                f"a method needs {section!r}: the reads and the save manifest are "
                "what the method *is* — an application supplies addresses, never "
                "what is measured (§1.1)",
                path=section,
            )
    _check_namespace(raw)
    signature = signature_of(raw)
    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise ParseError("P2", "description is free text", path="description")
    return MethodDocument(raw=raw, signature=signature, description=description)


def _check_order(sections: list[str], *, what: str) -> None:
    """§5.2 on a half-document: the sections a method or an application does
    carry appear in the §1 order, and ``save`` — carried only by a method — is
    still last."""
    ranks = {name: i for i, name in enumerate(SECTION_ORDER)}
    order = [ranks[section] for section in sections if section in ranks]
    if order != sorted(order):
        raise ValidationError(
            2,
            f"{what} sections out of order: got {sections}, expected the "
            f"docs/intervention_protocol.md §1 order {list(SECTION_ORDER)}",
        )
    if "save" in sections and sections[-1] != "save":
        raise ValidationError(2, "'save' must be the last section")


def _check_namespace(raw: Mapping[str, Any]) -> None:
    """§5.3 on a raw tree — the one checklist rule that needs no addresses."""
    seen: dict[str, str] = {}
    for section in NAMED_SECTIONS:
        table = raw.get(section)
        if not isinstance(table, Mapping):
            continue
        for name in table:
            if name in RESERVED_NAMES or name.startswith("counterfactual["):
                raise ValidationError(
                    3,
                    f"{name!r} is a reserved name and may not be declared",
                    path=f"{section}.{name}",
                )
            if name in seen:
                raise ValidationError(
                    3,
                    f"duplicate name {name!r} — declared in {seen[name]!r} and "
                    f"{section!r} (§1: one global namespace)",
                    path=f"{section}.{name}",
                )
            seen[name] = section


def signature_of(raw: Mapping[str, Any]) -> MethodSignature:
    """What is still open in ``raw`` — computed the same way for a method and
    for a composition, so "what is missing" and "did the application close it"
    are one question asked twice."""
    sites_raw = raw.get("sites")
    sites: dict[str, tuple[str, ...]] = {}
    for name in _site_names(raw):
        entry = sites_raw.get(name) if isinstance(sites_raw, Mapping) else None
        if entry is None:
            sites[name] = ("component",)
            continue
        if not isinstance(entry, Mapping):
            raise ParseError("P2", "a site is an object", path=f"sites.{name}")
        missing: list[str] = []
        component = entry.get("component")
        if component is None:
            missing.append("component")
        elif component not in LAYERLESS_COMPONENTS and "layer" not in entry:
            missing.append("layer")
        sites[name] = tuple(missing)
    data_raw = raw.get("data")
    data: list[str] = []
    for role in _roles(raw):
        entry = data_raw.get(role) if isinstance(data_raw, Mapping) else None
        if entry is None or (
            isinstance(entry, Mapping) and not {"dataset", "field"} <= set(entry)
        ):
            data.append(role)
    return MethodSignature(
        model="model" not in raw, sites=sites, data=tuple(data)
    )


def _site_names(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """Every site name the document addresses — the declared inventory plus
    anything a read or write references (a reference to an undeclared site is
    rule 4's business, but it still belongs in the signature)."""
    names: list[str] = []
    declared = raw.get("sites")
    if isinstance(declared, Mapping):
        names.extend(declared)
    for section in ("reads", "writes"):
        table = raw.get(section)
        if not isinstance(table, Mapping):
            continue
        for entry in table.values():
            site = entry.get("site") if isinstance(entry, Mapping) else None
            if isinstance(site, str) and site not in names:
                names.append(site)
    return tuple(names)


def _roles(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """The input roles the document reads: ``base`` (always — §2.2) plus every
    role a read or an intervened_model names."""
    roles: list[str] = ["base"]
    for section in ("reads", "intervened_models"):
        table = raw.get(section)
        if not isinstance(table, Mapping):
            continue
        for entry in table.values():
            role = entry.get("input") if isinstance(entry, Mapping) else None
            if isinstance(role, str) and role not in roles:
                roles.append(role.split("[", 1)[0] if role.startswith("counterfactual[") else role)
    declared = raw.get("data")
    if isinstance(declared, Mapping):
        for role in declared:
            if role not in roles:
                roles.append(role)
    return tuple(dict.fromkeys(roles))


def resolve_method_source(
    reference: Any, *, base_dir: Path | None
) -> tuple[Mapping[str, Any], str | None]:
    """Read an application's ``method`` field: an inline method object, or a
    path relative to the application file (the same rule a workflow step's
    ``document`` follows). Returns the method tree and the reference as
    written (``None`` for an inline method)."""
    if isinstance(reference, Mapping):
        return reference, None
    if not isinstance(reference, str):
        raise ParseError(
            "P2",
            "method is a path to a method document, or the method object inline",
            path="method",
        )
    if base_dir is None:
        raise ParseError(
            "P2",
            f"method {reference!r} is a path, but this application was loaded "
            "from memory — pass the document as a file, or inline the method",
            path="method",
        )
    from causalab.protocol.loader import load_text

    path = Path(reference)
    resolved = path if path.is_absolute() else base_dir / path
    if not resolved.exists():
        raise ValidationError(
            18,
            f"method {reference!r} does not exist (looked in {resolved})",
            path="method",
        )
    return load_text(resolved), reference


def compose(
    method_raw: Mapping[str, Any], application_raw: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge a method and an application into one protocol document (rule 18).

    Sections merge recursively; at every leaf exactly one side supplies a
    value, or both supply the same one. A contradiction is a load error rather
    than a silent override: an application that could overrule its method
    would make the method's digest a claim about nothing.
    """
    method = parse_method(method_raw)
    _check_application_shape(application_raw)
    if application_raw.get("version") != method_raw.get("version"):
        raise ValidationError(
            18,
            f"version mismatch: the method is version "
            f"{method_raw.get('version')!r}, the application "
            f"{application_raw.get('version')!r}",
            path="version",
        )
    merged: dict[str, Any] = {}
    for section in SECTION_ORDER:
        if section in ("type", "description"):
            continue
        in_method = section in method_raw
        in_app = section in application_raw
        if in_method and in_app:
            merged[section] = _merge(
                method_raw[section], application_raw[section], path=section
            )
        elif in_method:
            merged[section] = method_raw[section]
        elif in_app:
            merged[section] = application_raw[section]
    description = _join_descriptions(
        method_raw.get("description"), application_raw.get("description")
    )
    composed: dict[str, Any] = {"version": merged.pop("version")}
    if description is not None:
        composed["description"] = description
    composed.update(merged)
    _check_closed(composed)
    return composed


def _check_application_shape(raw: Mapping[str, Any]) -> None:
    kind = document_type(raw)  # a declared `type` is checked against the shape
    if kind != "application" and "method" not in raw:
        raise ValidationError(
            18, "an application names the method it binds", path="method"
        )
    allowed = ("method", *SECTION_ORDER)
    unknown = [key for key in raw if key not in allowed]
    if unknown:
        raise ParseError(
            "P3",
            f"unknown section {unknown[0]!r}{suggest(unknown[0], allowed)}",
            path=unknown[0],
        )
    if "method" not in raw:
        raise ValidationError(
            18, "an application names the method it binds", path="method"
        )
    ordered = [key for key in raw if key != "method"]
    _check_order(ordered, what="application")


def _merge(left: Any, right: Any, *, path: str) -> Any:
    """Recursive disjoint-or-equal merge — the composition rule itself."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        out = dict(left)
        for key, value in right.items():
            out[key] = (
                _merge(out[key], value, path=f"{path}.{key}") if key in out else value
            )
        return out
    if left == right:
        return left  # a restatement, cross-checked (§2.12)
    raise ValidationError(
        18,
        f"the application sets {path} to {right!r}, but its method already "
        f"fixed it at {left!r} — an application completes a method, it never "
        "overrules one (§1.1). Drop the field here, or fork the method.",
        path=path,
    )


def _join_descriptions(method: Any, application: Any) -> str | None:
    """Both halves describe themselves; the composition keeps both, method
    first — one document, one ``description``, nothing dropped."""
    parts = [part for part in (method, application) if isinstance(part, str) and part]
    if not parts:
        return None
    return "\n\n".join(parts)


def _check_closed(composed: Mapping[str, Any]) -> None:
    """Every hole filled — reported as the list of what is still open, so a
    half-bound application says what it forgot instead of failing as a parse
    error three layers down."""
    signature = signature_of(composed)
    if signature.is_closed():
        return
    lines = "\n  ".join(signature.lines())
    raise ValidationError(
        18,
        "the composition is still open — the application binds no value for:"
        f"\n  {lines}",
    )


def method_digest(method_raw: Mapping[str, Any]) -> str:
    """The method's content hash.

    Not a canonical protocol digest: a method has no model, so nothing about
    it can be derived (widths, dataset digests) and there is nothing to
    materialize. It is the sha256 of the method's own bytes under the
    canonical serialization (sorted keys, normalized numbers), minus the
    ``type`` declaration — enough to answer "is this the same method?", which
    is the question a shared method is asked.
    """
    from causalab.protocol.canonical import canonical_bytes
    import hashlib

    body = {key: value for key, value in method_raw.items() if key != "type"}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()
