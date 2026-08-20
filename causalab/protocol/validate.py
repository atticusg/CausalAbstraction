"""The load-error checklist (spec §5) over one concrete document.

:func:`validate_document` runs rules 3–13 on a parsed, *concrete*
:class:`~causalab.protocol.schema.Document` — a point protocol, or an
un-swept document. The other rules live where their information lives:

* rules 1–2 (strict keys, section order) — :mod:`causalab.protocol.schema`;
* rule 14 (sweep wrappers, point cap) — :mod:`causalab.protocol.sweep`;
* rule 15 (artifact-valued fields) — :mod:`causalab.protocol.resolve`;
* rule 13 needs a selected backend, so it only fires when one is passed.

Interpretations this module commits to (each surfaced in the PR notes):

* **Rule 8 overlap is conservative.** Two positions "overlap" unless they
  are *provably* disjoint from the document alone: distinct indices of the
  same sign, non-intersecting spans in the same frame, or two different
  prompt variables (template variables occupy disjoint spans). An index
  compared against a variable window is unknowable at load and treated as
  overlapping — refuse-before-run rather than collide-at-run.
* **Rules 8 and 9 compose.** Full-width writes have ``dims = all``. At one
  (site, overlapping pos, model) address: at most one full-width absolute
  write (rule 8), and no two *absolute* writes may intersect in dims
  (rule 9) — additive deltas may overlap anything, absolute-then-additive
  order is the mechanism-class rule (§2.8).
* **Dead declarations** (§0) are reported under rule 11 — the sink rule
  names reads explicitly; unused sites, positions, featurizers and params
  are the same principle and share the rule number.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from causalab.protocol.errors import ValidationError
from causalab.protocol.schema import (
    ADDITIVE_MECHANISMS,
    FEATURIZER_SLOTS,
    Do,
    Document,
    WriteSpec,
    NAMED_SECTIONS,
    PositionSpec,
    RESERVED_NAMES,
    SaveEntry,
)

__all__ = ["validate_document"]

_COUNTERFACTUAL_INDEXED = re.compile(r"^counterfactual\[(\d+)\]$")

#: Trainable featurizer kinds — the ones with gradient-trainable slots
#: (§5.12). ``pca`` / ``standardize`` are computed from data, ``identity``
#: has nothing to fit.
_TRAINABLE_KINDS = frozenset({"subspace", "gate", "sae"})


def validate_document(doc: Document, *, backend_is_local: bool | None = None) -> None:
    """Run checklist rules 3–13 (and 13 only when ``backend_is_local`` is
    given). Raises :class:`ValidationError` on the first violation."""
    names = _check_namespace(doc)  # rule 3
    _check_references(doc, names)  # rule 4 (+ the rule-5 read bindings)
    _check_writes_inert(doc, names)  # rule 6
    _check_membership_and_acyclic(doc)  # rule 7
    _check_write_collisions(doc)  # rules 8 + 9
    _check_save(doc)  # rule 10
    _check_sinks(doc)  # rule 11
    _check_trainability(doc)  # rule 12
    if backend_is_local is not None:
        _check_pytorch_fn(doc, backend_is_local)  # rule 13


# --------------------------------------------------------------------------- #
# rule 3 — one global namespace, no reserved names
# --------------------------------------------------------------------------- #


def _check_namespace(doc: Document) -> dict[str, str]:
    names: dict[str, str] = {}
    for section in NAMED_SECTIONS:
        table = getattr(doc, section)
        for name in table:
            if name in RESERVED_NAMES or _COUNTERFACTUAL_INDEXED.match(name):
                raise ValidationError(
                    3, f"{name!r} is a reserved name", path=f"{section}.{name}"
                )
            if name in names:
                raise ValidationError(
                    3,
                    f"name {name!r} is declared in both {names[name]!r} and {section!r} — "
                    "sections 6–13 share one namespace",
                    path=f"{section}.{name}",
                )
            names[name] = section
    return names


# --------------------------------------------------------------------------- #
# rule 4 — every reference resolves (and rule 5, read bindings)
# --------------------------------------------------------------------------- #


def _valid_roles(doc: Document) -> set[str]:
    roles = {"base"}
    counterfactual = doc.data.get("counterfactual")
    if counterfactual is None:
        return roles
    if isinstance(counterfactual, tuple):
        roles.update(f"counterfactual[{j}]" for j in range(len(counterfactual)))
        # a singular reference to an array-valued counterfactual is not a role
    else:
        roles.add("counterfactual")
    return roles


def _check_pos_ref(doc: Document, pos: Any, path: str) -> None:
    if isinstance(pos, str):
        if pos not in doc.positions:
            raise ValidationError(
                4, f"position {pos!r} is not declared in positions", path=path
            )
    elif not isinstance(pos, PositionSpec):
        raise ValidationError(4, f"unresolvable position {pos!r}", path=path)


def _check_featurizer_ref(doc: Document, ref: Any, path: str) -> None:
    if ref is None:
        return
    chain = (
        (ref,) if isinstance(ref, str) else tuple(ref) if isinstance(ref, tuple) else ()
    )
    for name in chain:
        if name not in doc.featurizers:
            raise ValidationError(4, f"featurizer {name!r} is not declared", path=path)


def _param_slot_names(doc: Document) -> set[str]:
    """Every addressable param name: ``params`` entries plus the
    auto-declared ``<featurizer>.<slot>`` names (§2.5)."""
    out = set(doc.params)
    for fname, spec in doc.featurizers.items():
        kind = spec.kind if isinstance(spec.kind, str) else "identity"
        for slot in FEATURIZER_SLOTS.get(kind, ()):
            out.add(f"{fname}.{slot}")
    return out


def _check_references(doc: Document, names: dict[str, str]) -> None:
    roles = _valid_roles(doc)
    model_names = {"original", *doc.intervened_models}

    for rname, read in doc.reads.items():
        p = f"reads.{rname}"
        if read.site not in doc.sites:
            raise ValidationError(
                4, f"site {read.site!r} is not declared", path=f"{p}.site"
            )
        _check_pos_ref(doc, read.pos, f"{p}.pos")
        if read.model not in model_names:
            raise ValidationError(
                5,
                f"read model {read.model!r} is neither 'original' nor a declared "
                "intervened_model",
                path=f"{p}.model",
            )
        if read.input not in roles:
            raise ValidationError(
                5,
                f"read input {read.input!r} is not a valid role ({sorted(roles)})",
                path=f"{p}.input",
            )
        if read.model != "original":
            im = doc.intervened_models[str(read.model)]
            if read.input != im.input:
                raise ValidationError(
                    5,
                    f"read {rname!r} declares input {read.input!r} but its model "
                    f"{read.model!r} runs on {im.input!r} — the restated binding "
                    "must match (§2.7)",
                    path=f"{p}.input",
                )
        _check_featurizer_ref(doc, read.featurizer, f"{p}.featurizer")

    for ename, write in doc.writes.items():
        p = f"writes.{ename}"
        if write.site not in doc.sites:
            raise ValidationError(
                4, f"site {write.site!r} is not declared", path=f"{p}.site"
            )
        _check_pos_ref(doc, write.pos, f"{p}.pos")
        _check_featurizer_ref(doc, write.featurizer, f"{p}.featurizer")

    for mname, im in doc.intervened_models.items():
        p = f"intervened_models.{mname}"
        if im.input not in roles:
            raise ValidationError(
                7,
                f"intervened_model input {im.input!r} is not a valid role",
                path=f"{p}.input",
            )
        for ename in _im_writes(im.writes):
            if ename not in doc.writes:
                raise ValidationError(
                    4, f"write {ename!r} is not declared", path=f"{p}.writes"
                )

    for qname, metric in doc.metrics.items():
        p = f"metrics.{qname}"
        if metric.of not in doc.reads:
            raise ValidationError(
                4, f"metric 'of' {metric.of!r} is not a read", path=f"{p}.of"
            )
        of_read = doc.reads[str(metric.of)]
        of_site = doc.sites[str(of_read.site)]
        if metric.kind != "kl" and of_site.component != "lm_head":
            # not in the §5 checklist: the token-space kinds name vocab
            # entries, which only lm_head produces (an interpretation this
            # loader commits to; surfaced in the PR notes)
            raise ValidationError(
                4,
                f"metric {qname!r} names vocabulary tokens, but its read "
                f"{metric.of!r} taps {of_site.component!r} — token-space metric "
                "kinds bind to lm_head reads",
                path=f"{p}.of",
            )
        if metric.kind == "kl":
            target = metric.fields.get("target")
            if target not in doc.reads:
                raise ValidationError(
                    4, f"kl target {target!r} is not a read (§2.10)", path=f"{p}.target"
                )
            target_site = doc.sites[str(doc.reads[str(target)].site)]
            if target_site.component != of_site.component:
                raise ValidationError(
                    4,
                    f"kl compares two reads' distributions, but {metric.of!r} taps "
                    f"{of_site.component!r} and {target!r} taps "
                    f"{target_site.component!r}",
                    path=f"{p}.target",
                )

    if doc.train is not None:
        _check_train_references(doc)

    for i, entry in enumerate(doc.save):
        if (
            entry.value not in doc.reads
            and entry.value not in doc.metrics
            and entry.value not in doc.featurizers
            and entry.value not in names  # declared-but-unsaveable is rule 10
        ):
            raise ValidationError(
                4,
                f"save value {entry.value!r} is not declared",
                path=f"save[{i}].value",
            )


def _check_train_references(doc: Document) -> None:
    assert doc.train is not None
    train = doc.train
    slot_names = _param_slot_names(doc)
    for i, (_weight, target) in enumerate(train.objective):
        p = f"train.objective[{i}]"
        if isinstance(target, str):
            if target not in doc.metrics:
                raise ValidationError(
                    4, f"objective metric {target!r} is not declared", path=p
                )
        elif isinstance(target, tuple):
            _kind, reg_target = target
            if reg_target not in doc.featurizers and reg_target not in slot_names:
                raise ValidationError(
                    4,
                    f"regularizer target {reg_target!r} is neither a featurizer "
                    "nor a dotted param slot",
                    path=p,
                )
    for i, pname in enumerate(train.params):
        if (
            pname not in doc.featurizers
            and pname not in slot_names
            and pname not in doc.params
        ):
            raise ValidationError(
                4,
                f"train.params entry {pname!r} is neither a featurizer, a dotted "
                "slot, nor a params entry",
                path=f"train.params[{i}]",
            )
    if train.eval is not None:
        for mname in train.eval["metrics"]:
            if mname not in doc.metrics:
                raise ValidationError(
                    4,
                    f"eval metric {mname!r} is not declared",
                    path="train.eval.metrics",
                )
    if train.early_stop is not None:
        es_metric = train.early_stop["metric"]
        if es_metric not in doc.metrics:
            raise ValidationError(
                4,
                f"early_stop metric {es_metric!r} is not declared",
                path="train.early_stop.metric",
            )
    if train.anneal is not None:
        for dotted in train.anneal:
            parts = dotted.split(".")
            if len(parts) < 3:
                raise ValidationError(
                    4,
                    f"anneal target {dotted!r} is not a dotted "
                    "<featurizer>.<slot>.<hyperparameter> path",
                    path="train.anneal",
                )
            fname, slot = parts[0], parts[1]
            spec = doc.featurizers.get(fname)
            if spec is None:
                raise ValidationError(
                    4,
                    f"anneal target {dotted!r}: {fname!r} is not a featurizer",
                    path="train.anneal",
                )
            kind = spec.kind if isinstance(spec.kind, str) else "identity"
            if slot not in FEATURIZER_SLOTS.get(kind, ()):
                raise ValidationError(
                    4,
                    f"anneal target {dotted!r}: {kind!r} featurizers have no slot {slot!r}",
                    path="train.anneal",
                )


# --------------------------------------------------------------------------- #
# rule 6 — writes are inert; operands are reads, params, or literal scalars
# --------------------------------------------------------------------------- #


def _operand_names(do: Do) -> tuple[str, ...]:
    """The names a mechanism's operands reference (literal scalars excluded)."""
    mech = do.mechanism
    if mech == "swap":
        return (do.payload,) if isinstance(do.payload, str) else ()
    if mech in ("add_scaled", "lerp"):
        out: list[str] = []
        for field in ("op", "alpha"):
            value = do.payload.get(field)
            if isinstance(value, str):
                out.append(value)
        return tuple(out)
    if mech == "affine":
        return tuple(
            v for v in (do.payload.get("A"), do.payload.get("b")) if isinstance(v, str)
        )
    return ()


def _check_writes_inert(doc: Document, names: dict[str, str]) -> None:
    slot_names = _param_slot_names(doc)
    for ename, write in doc.writes.items():
        if write.do.mechanism == "affine":
            for field in ("A", "b"):
                target = write.do.payload.get(field)
                if isinstance(target, str) and (
                    target in doc.reads
                    or (
                        target in names
                        and target not in doc.params
                        and target not in slot_names
                    )
                ):
                    raise ValidationError(
                        6,
                        f"affine {field!r} must name a param (§2.8 types both "
                        f"fields as params); {target!r} is a "
                        f"{names.get(target, 'read')}",
                        path=f"writes.{ename}.do",
                    )
        for operand in _operand_names(write.do):
            if operand in doc.reads or operand in doc.params or operand in slot_names:
                continue
            if operand in names:
                raise ValidationError(
                    6,
                    f"write operand {operand!r} names a {names[operand]} entry — "
                    "operands are reads, params, or literal scalars (§2.8)",
                    path=f"writes.{ename}.do",
                )
            raise ValidationError(
                4,
                f"write operand {operand!r} is not declared",
                path=f"writes.{ename}.do",
            )


# --------------------------------------------------------------------------- #
# rule 7 — membership and the acyclic model graph
# --------------------------------------------------------------------------- #


def _im_writes(writes: Any) -> tuple[str, ...]:
    return tuple(writes) if isinstance(writes, tuple) else ()


def _check_membership_and_acyclic(doc: Document) -> None:
    in_force: dict[str, set[str]] = {ename: set() for ename in doc.writes}
    for mname, im in doc.intervened_models.items():
        seen: set[str] = set()
        for ename in _im_writes(im.writes):
            if ename in seen:
                raise ValidationError(
                    7,
                    f"write {ename!r} listed twice",
                    path=f"intervened_models.{mname}.writes",
                )
            seen.add(ename)
            in_force[ename].add(mname)
    for ename, hosts in in_force.items():
        if not hosts:
            raise ValidationError(
                7,
                f"write {ename!r} appears in no intervened_model — every declared "
                "write must be in force somewhere (§2.9)",
                path=f"writes.{ename}",
            )

    # model graph: an edge M -> M' when a read in M' is an operand of a write
    # in force in M (M' must run first). 'original' has no out-edges.
    graph: dict[str, set[str]] = {m: set() for m in doc.intervened_models}
    for mname, im in doc.intervened_models.items():
        for ename in _im_writes(im.writes):
            for operand in _operand_names(doc.writes[ename].do):
                read = doc.reads.get(operand)
                if read is not None and read.model != "original":
                    graph[mname].add(str(read.model))

    state: dict[str, int] = {}  # 0 in-progress, 1 done

    def visit(node: str, trail: tuple[str, ...]) -> None:
        mark = state.get(node)
        if mark == 1:
            return
        if mark == 0:
            cycle = " -> ".join((*trail[trail.index(node) :], node))
            raise ValidationError(
                7,
                f"the intervened-model graph has a cycle: {cycle} — operand flow "
                "must be acyclic (§2.9)",
            )
        state[node] = 0
        for nxt in graph[node]:
            visit(nxt, trail + (node,))
        state[node] = 1

    for mname in graph:
        visit(mname, ())


# --------------------------------------------------------------------------- #
# rules 8 + 9 — absolute-write and dims collisions per address
# --------------------------------------------------------------------------- #


def _pos_key(doc: Document, pos: Any) -> PositionSpec:
    if isinstance(pos, str):
        entry = doc.positions[pos]
        if isinstance(entry, PositionSpec):
            return entry
        raise ValidationError(4, f"position {pos!r} did not resolve to a concrete spec")
    assert isinstance(pos, PositionSpec)
    return pos


def _provably_disjoint(a: PositionSpec, b: PositionSpec) -> bool:
    """True when two position specs cannot address a common token on any row
    (see the module docstring for the conservative reading)."""
    if a.scope != b.scope or a.relative_to != b.relative_to:
        return False  # different frames — incomparable, assume overlap
    if a.variable is not None and b.variable is not None:
        return a.variable != b.variable
    if a.variable is not None or b.variable is not None:
        return False
    a_index = a.index if isinstance(a.index, int) else None
    b_index = b.index if isinstance(b.index, int) else None
    a_span = tuple(a.span) if isinstance(a.span, tuple) else None
    b_span = tuple(b.span) if isinstance(b.span, tuple) else None
    if a_index is not None and b_index is not None:
        return a_index != b_index and (a_index < 0) == (b_index < 0)
    if a_span is not None and b_span is not None:
        # comparable only within one sign regime (end-relative bounds are a
        # different frame; mixed-sign pairs are unknowable at load)
        if _span_regime(a_span) is None or _span_regime(a_span) != _span_regime(b_span):
            return False
        (a0, a1), (b0, b1) = a_span, b_span
        return a1 <= b0 or b1 <= a0
    if a_index is not None and b_span is not None:
        return _index_outside_span(a_index, b_span)
    if b_index is not None and a_span is not None:
        return _index_outside_span(b_index, a_span)
    return False


def _span_regime(span: tuple[int, int]) -> str | None:
    """ "forward" for fully non-negative spans, "end" for fully end-relative
    ones, None for mixed (incomparable at load)."""
    lo, hi = span
    if lo >= 0 and hi >= 0:
        return "forward"
    if lo < 0 and hi <= 0:
        return "end"
    return None


def _index_outside_span(index: int, span: tuple[int, int]) -> bool:
    regime = _span_regime(span)
    if regime == "forward":
        if index < 0:
            return False  # end-relative index vs a forward window — unknowable
        lo, hi = span
        return not lo <= index < hi
    if regime == "end":
        if index >= 0:
            return False  # forward index vs an end-relative window — unknowable
        lo, hi = span
        return not lo <= index < (hi if hi != 0 else 0)
    return False  # mixed-sign span — unknowable


def _is_absolute(write: WriteSpec) -> bool:
    return write.do.mechanism not in ADDITIVE_MECHANISMS


def _dims_intersect(a: WriteSpec, b: WriteSpec) -> bool:
    a_dims = a.dims if isinstance(a.dims, tuple) else None
    b_dims = b.dims if isinstance(b.dims, tuple) else None
    if a_dims is None or b_dims is None:
        return True  # full width intersects everything
    return bool(set(a_dims) & set(b_dims))


def _check_write_collisions(doc: Document) -> None:
    for mname, im in doc.intervened_models.items():
        writes = [(ename, doc.writes[ename]) for ename in _im_writes(im.writes)]
        for i in range(len(writes)):
            for j in range(i + 1, len(writes)):
                (name_a, a), (name_b, b) = writes[i], writes[j]
                if a.site != b.site:
                    continue
                if _provably_disjoint(_pos_key(doc, a.pos), _pos_key(doc, b.pos)):
                    continue
                both_dims = isinstance(a.dims, tuple) and isinstance(b.dims, tuple)
                if both_dims and _dims_intersect(a, b):
                    # §5.9, read literally: explicit dims selections at one
                    # address are pairwise disjoint — additive included
                    # (surfaced as a spec question; a steer inside a swapped
                    # subspace needs featurizer composition instead)
                    raise ValidationError(
                        9,
                        f"writes {name_a!r} and {name_b!r} select intersecting dims "
                        f"at site {a.site!r} in model {mname!r} — co-occurring "
                        "dims selections must be disjoint (§5.9)",
                    )
                if not (_is_absolute(a) and _is_absolute(b)):
                    continue  # one absolute + additive deltas is the §2.8 class order
                if not both_dims:
                    raise ValidationError(
                        8,
                        f"writes {name_a!r} and {name_b!r} are two absolute writes at "
                        f"site {a.site!r} with overlapping positions in model "
                        f"{mname!r} — at most one absolute write per address (§2.8)",
                    )


# --------------------------------------------------------------------------- #
# rule 10 — the save manifest
# --------------------------------------------------------------------------- #


def _trained_featurizers(doc: Document) -> set[str]:
    if doc.train is None:
        return set()
    trained: set[str] = set()
    for pname in doc.train.params:
        root = pname.split(".", 1)[0]
        if root in doc.featurizers:
            trained.add(root)
    return trained


def _check_save(doc: Document) -> None:
    trained = _trained_featurizers(doc)
    seen_values: set[str] = set()
    seen_paths: set[str] = set()
    for i, entry in enumerate(doc.save):
        p = f"save[{i}]"
        if entry.file_path in seen_paths:
            raise ValidationError(
                10,
                f"two save entries write {entry.file_path!r} — one file per "
                "entry, or the later silently clobbers the earlier",
                path=p,
            )
        seen_paths.add(entry.file_path)
        if entry.value in seen_values:
            raise ValidationError(
                10,
                f"{entry.value!r} is saved twice — one manifest entry per value",
                path=p,
            )
        seen_values.add(entry.value)
        if entry.value in doc.featurizers:
            _check_save_featurizer(doc, entry, trained, p)
        elif entry.value in doc.reads or entry.value in doc.metrics:
            _check_save_binding(doc, entry, p)
        else:
            raise ValidationError(
                10,
                f"{entry.value!r} is not saveable — only reads, metrics, and "
                "trained featurizers leave a run (§2.12)",
                path=p,
            )
    for qname in doc.metrics:
        if qname not in seen_values:
            raise ValidationError(
                10, f"metric {qname!r} is not saved — every metric must be (§2.12)"
            )
    for fname in trained:
        if fname not in seen_values:
            raise ValidationError(
                10,
                f"trained featurizer {fname!r} is not saved — every fit must be (§2.12)",
            )


def _check_save_binding(doc: Document, entry: SaveEntry, path: str) -> None:
    if entry.site is not None or entry.model is None or entry.input is None:
        raise ValidationError(
            10, "a read/metric save entry binds with model + input", path=path
        )
    read = doc.reads.get(entry.value)
    if read is None:
        read = doc.reads[str(doc.metrics[entry.value].of)]
    if entry.model != read.model or entry.input != read.input:
        raise ValidationError(
            10,
            f"save entry for {entry.value!r} restates (model={entry.model!r}, "
            f"input={entry.input!r}) but the declaration chain resolves to "
            f"(model={read.model!r}, input={read.input!r}) — bindings are "
            "cross-checked, never trusted (§2.12)",
            path=path,
        )
    is_metric = entry.value in doc.metrics
    if entry.reduce is not None and is_metric:
        raise ValidationError(
            10,
            f"save entry for metric {entry.value!r} carries 'reduce' — a metric "
            "is already a reduction over its read; reduce applies to reads "
            "(§2.12)",
            path=path,
        )
    expected_ext = ".parquet" if is_metric else ".safetensors"
    if not entry.file_path.endswith(expected_ext):
        raise ValidationError(
            10,
            f"{entry.value!r} is a {'metric' if is_metric else 'read'} — its "
            f"file_path must end in {expected_ext!r} (§2.12)",
            path=path,
        )


def _check_save_featurizer(
    doc: Document, entry: SaveEntry, trained: set[str], path: str
) -> None:
    if entry.site is None or entry.model is not None or entry.input is not None:
        raise ValidationError(
            10, "a featurizer save entry binds with 'site' alone", path=path
        )
    if entry.value not in trained:
        spec = doc.featurizers[entry.value]
        reason = (
            "a file_path-loaded featurizer is a pointless copy"
            if spec.file_path is not None
            else "an untrained featurizer has nothing to save"
        )
        raise ValidationError(
            10, f"{entry.value!r} is not trained — {reason} (§2.12)", path=path
        )
    used_sites = _featurizer_sites(doc, entry.value)
    if entry.site not in used_sites:
        raise ValidationError(
            10,
            f"featurizer {entry.value!r} is used at site(s) {sorted(used_sites)}, "
            f"not {entry.site!r} — the restated site is cross-checked (§2.12)",
            path=path,
        )
    if entry.reduce is not None:
        raise ValidationError(
            10,
            "a featurizer bundle carries fitted parameters, not gathered rows — "
            "'reduce' applies to reads (§2.12)",
            path=path,
        )
    if not entry.file_path.endswith(".safetensors"):
        raise ValidationError(
            10, "a featurizer bundle's file_path must end in '.safetensors'", path=path
        )


def _featurizer_sites(doc: Document, fname: str) -> set[str]:
    used: set[str] = set()
    for read in doc.reads.values():
        if _references_featurizer(read.featurizer, fname):
            used.add(str(read.site))
    for write in doc.writes.values():
        if _references_featurizer(write.featurizer, fname):
            used.add(str(write.site))
    return used


def _references_featurizer(ref: Any, fname: str) -> bool:
    if ref is None:
        return False
    chain = (ref,) if isinstance(ref, str) else tuple(ref)
    return fname in chain


# --------------------------------------------------------------------------- #
# rule 11 — sinks: nothing declared is dead
# --------------------------------------------------------------------------- #


def _check_sinks(doc: Document) -> None:
    saved = {entry.value for entry in doc.save}
    metric_inputs: set[str] = set()
    for metric in doc.metrics.values():
        metric_inputs.add(str(metric.of))
        if metric.kind == "kl":
            target = metric.fields.get("target")
            if isinstance(target, str):
                metric_inputs.add(target)
    operands: set[str] = set()
    for write in doc.writes.values():
        operands.update(_operand_names(write.do))
    for rname in doc.reads:
        if rname not in saved and rname not in metric_inputs and rname not in operands:
            raise ValidationError(
                11,
                f"read {rname!r} is dead: neither saved, nor a metric input, nor "
                "a write operand (§5.11)",
                path=f"reads.{rname}",
            )
    _check_dead_declarations(doc, operands)


def _check_dead_declarations(doc: Document, operands: set[str]) -> None:
    """§0's uniform rule, reported under the sink rule's number: every
    declared site/position/featurizer/param must be referenced."""
    used_sites = {str(r.site) for r in doc.reads.values()} | {
        str(e.site) for e in doc.writes.values()
    }
    for sname in doc.sites:
        if sname not in used_sites:
            raise ValidationError(
                11, f"site {sname!r} is declared but never used", path=f"sites.{sname}"
            )
    used_pos = {r.pos for r in doc.reads.values() if isinstance(r.pos, str)} | {
        e.pos for e in doc.writes.values() if isinstance(e.pos, str)
    }
    for pname in doc.positions:
        if pname not in used_pos:
            raise ValidationError(
                11,
                f"position {pname!r} is declared but never used",
                path=f"positions.{pname}",
            )
    used_feat: set[str] = set()
    for read in doc.reads.values():
        used_feat.update(_featurizer_chain(read.featurizer))
    for write in doc.writes.values():
        used_feat.update(_featurizer_chain(write.featurizer))
    for fname in doc.featurizers:
        if fname not in used_feat:
            raise ValidationError(
                11,
                f"featurizer {fname!r} is declared but never used",
                path=f"featurizers.{fname}",
            )
    train_params: set[str] = set(doc.train.params) if doc.train is not None else set()
    for pname in doc.params:
        if pname in operands or pname in train_params:
            continue
        if any(pname in _operand_names(e.do) for e in doc.writes.values()):
            continue
        raise ValidationError(
            11,
            f"params entry {pname!r} is declared but never used",
            path=f"params.{pname}",
        )
    read_models = {str(r.model) for r in doc.reads.values()}
    for mname in doc.intervened_models:
        if mname not in read_models:
            raise ValidationError(
                11,
                f"intervened_model {mname!r} is never read — its writes can reach "
                "no sink (§0)",
                path=f"intervened_models.{mname}",
            )


def _featurizer_chain(ref: Any) -> Iterable[str]:
    if ref is None:
        return ()
    return (ref,) if isinstance(ref, str) else tuple(ref)


# --------------------------------------------------------------------------- #
# rule 12 — trainability declarations are consistent
# --------------------------------------------------------------------------- #


def _check_trainability(doc: Document) -> None:
    trained = _trained_featurizers(doc)
    for fname in trained:
        spec = doc.featurizers[fname]
        if spec.file_path is not None:
            raise ValidationError(
                12,
                f"featurizer {fname!r} is loaded from file_path and appears in "
                "train.params — loading and fitting the same artifact is a "
                "contradiction (§2.5)",
                path=f"featurizers.{fname}",
            )
        kind = spec.kind if isinstance(spec.kind, str) else "identity"
        if kind not in _TRAINABLE_KINDS:
            raise ValidationError(
                12,
                f"featurizer {fname!r} has kind {kind!r}, which has no trainable "
                "slots (§5.12)",
                path=f"featurizers.{fname}",
            )
    if doc.train is not None:
        for pname in doc.train.params:
            spec = doc.params.get(pname)
            if spec is not None and spec.file_path is not None:
                raise ValidationError(
                    12,
                    f"params entry {pname!r} is a loaded constant (file_path) and "
                    "appears in train.params — loading and fitting the same "
                    "tensor is a contradiction (§2.6)",
                    path=f"params.{pname}",
                )
        for pname, spec in doc.params.items():
            if spec.shape is not None and pname not in doc.train.params:
                raise ValidationError(
                    12,
                    f"params entry {pname!r} declares shape/init (trainable) but is "
                    "not in train.params (§2.6)",
                    path=f"params.{pname}",
                )
    else:
        for pname, spec in doc.params.items():
            if spec.shape is not None:
                raise ValidationError(
                    12,
                    f"params entry {pname!r} is trainable but the document has no "
                    "train section (§2.6)",
                    path=f"params.{pname}",
                )


# --------------------------------------------------------------------------- #
# rule 13 — pytorch_fn is local-only
# --------------------------------------------------------------------------- #


def _check_pytorch_fn(doc: Document, backend_is_local: bool) -> None:
    if backend_is_local:
        return
    for ename, write in doc.writes.items():
        if write.do.mechanism == "pytorch_fn":
            raise ValidationError(
                13,
                f"write {ename!r} uses pytorch_fn, which only a local backend may "
                "run (§2.8) — the selected backend is not local",
                path=f"writes.{ename}.do",
            )
