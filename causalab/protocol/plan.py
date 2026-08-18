"""The planning skeleton: models → forward groups, shared work interned.

Execution semantics (spec §4) derive everything from the document: for each
expanded point, the models to run are ``original`` on every input it is
read on, plus each intervened model on its declared input — one **forward
group** each. ``num_forwards`` is a property of the plan, never authored.

Across the points of a swept document, the planner **content-dedups**: a
forward group's identity is the digest of its full dependency closure
(model + in-force edits + their operand reads' closures + input binding),
so a harvest shared by nine fits interns to one group and the sharing falls
out of value identity, not scheduling cleverness (§3). Backends consume
this plan as data; fusion, batching and staging stay their call (§8).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping

from causalab.protocol.schema import Document, PositionSpec

__all__ = ["ForwardGroup", "PointPlan", "Tap", "plan_point", "closure_digest"]

#: Intra-block execution order of the component vocabulary — backend-free
#: data used only to find a group's deepest tap (elision, §4). ``ln_final``
#: and ``lm_head`` sort after every block.
_COMPONENT_RANK: dict[str, int] = {
    "embeddings": 0,
    "block_input": 10,
    "attention_probs": 20,
    "attention_value": 30,
    "attention_output": 40,
    "mlp_input": 50,
    "mlp_activation": 60,
    "mlp_output": 70,
    "expert_output": 71,
    "router_logits": 72,
    "block_output": 80,
    "ln_final": 90,
    "lm_head": 100,
}


@dataclasses.dataclass(frozen=True)
class Tap:
    """One value to materialize in a group's forward: a read's address."""

    read: str
    site: str
    depth: tuple[int, int]  # (layer, component rank) — the elision key


@dataclasses.dataclass(frozen=True)
class ForwardGroup:
    """One forward pass: a model (original or an IM) on one input role.

    ``digest`` is the content identity of everything that determines this
    group's activations — equal digests across points mean one shared
    forward."""

    model: str
    input: str
    taps: tuple[Tap, ...]
    digest: str

    @property
    def stop_after(self) -> tuple[int, int] | None:
        """The deepest tap's depth — a backend may end the forward there
        (§4 elision). ``None`` when the group has no taps."""
        return max((tap.depth for tap in self.taps), default=None)


@dataclasses.dataclass(frozen=True)
class PointPlan:
    """The derived execution shape of one concrete point protocol."""

    groups: tuple[ForwardGroup, ...]

    @property
    def num_forwards(self) -> int:
        return len(self.groups)


def plan_point(
    doc: Document, *, data_identity: Mapping[str, Any] | None = None
) -> PointPlan:
    """Derive the forward groups of one concrete document.

    ``data_identity`` (role → stamped identity, e.g. dataset digests from
    the canonical form) folds the input data into group digests so two
    points reading different datasets never intern together; omit it for a
    purely structural plan."""
    groups: list[ForwardGroup] = []
    seen: set[tuple[str, str]] = set()
    # original, once per input it is read on (§4), in read declaration order
    for read in doc.reads.values():
        if read.model == "original" and ("original", str(read.input)) not in seen:
            seen.add(("original", str(read.input)))
            groups.append(_build_group(doc, "original", str(read.input), data_identity))
    for im_name, im in doc.intervened_models.items():
        groups.append(_build_group(doc, im_name, str(im.input), data_identity))
    return PointPlan(groups=tuple(groups))


def _build_group(
    doc: Document,
    model: str,
    input_role: str,
    data_identity: Mapping[str, Any] | None,
) -> ForwardGroup:
    taps = tuple(
        Tap(read=rname, site=str(read.site), depth=_depth(doc, str(read.site)))
        for rname, read in doc.reads.items()
        if read.model == model and str(read.input) == input_role
    )
    body = {
        "model": _model_closure(doc, model, set()),
        "input": input_role,
        "data": dict(data_identity or {}).get(input_role),
    }
    digest = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), default=_encode
        ).encode()
    ).hexdigest()
    return ForwardGroup(model=model, input=input_role, taps=taps, digest=digest)


def closure_digest(doc: Document, read_name: str) -> str:
    """The content identity of one read's value: its address plus the full
    closure of the model it reads in. Equal digests across points mean one
    shared harvest (§3)."""
    body = _read_closure(doc, read_name, set())
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), default=_encode
        ).encode()
    ).hexdigest()


def _depth(doc: Document, site_name: str) -> tuple[int, int]:
    site = doc.sites[site_name]
    layer = site.layer if isinstance(site.layer, int) else 0
    component = site.component if isinstance(site.component, str) else "lm_head"
    rank = _COMPONENT_RANK.get(component, 100)
    if component in ("ln_final", "lm_head"):
        return (1_000_000, rank)  # after every block
    return (layer, rank)


def _model_closure(doc: Document, model: str, visiting: set[str]) -> Any:
    """Everything that determines a model's activations: for an IM, the
    in-force edits and, recursively, their operand reads' closures. The
    validated acyclicity (§5.7) bounds the recursion; ``visiting`` is a
    belt-and-braces guard."""
    if model == "original":
        return "original"
    if model in visiting:
        raise AssertionError(
            f"cycle through {model!r} — validation should have refused this"
        )
    im = doc.intervened_models[model]
    train_dep: Any = None
    edits: dict[str, Any] = {}
    for ename in sorted(im.edits if isinstance(im.edits, tuple) else ()):
        edit = doc.edits[ename]
        operands = {}
        for op in _edit_operand_names(doc, ename):
            if op in doc.reads:
                operands[op] = _read_closure(doc, op, visiting | {model})
        edits[ename] = {
            "site": _entry(doc.sites, str(edit.site)),
            "pos": _pos_entry(doc, edit.pos),
            "featurizer": _featurizer_entry(doc, edit.featurizer),
            "dims": edit.dims,
            "do": {str(edit.do.mechanism): edit.do.payload},
            "operands": operands,
        }
        if doc.train is not None and _uses_trained_featurizer(doc, edit.featurizer):
            # a trained featurizer's weights are a function of the whole fit —
            # two points differing only in train.seed must never intern
            train_dep = dataclasses.asdict(doc.train)
    return {"input": im.input, "edits": edits, "train": train_dep}


def _uses_trained_featurizer(doc: Document, ref: Any) -> bool:
    if doc.train is None or ref is None:
        return False
    trained = {p.split(".", 1)[0] for p in doc.train.params}
    chain = (ref,) if isinstance(ref, str) else tuple(ref)
    return any(name in trained for name in chain)


def _read_closure(doc: Document, read_name: str, visiting: set[str]) -> Any:
    read = doc.reads[read_name]
    return {
        "site": _entry(doc.sites, str(read.site)),
        "pos": _pos_entry(doc, read.pos),
        "featurizer": _featurizer_entry(doc, read.featurizer),
        "dims": read.dims,
        "input": read.input,
        "model": _model_closure(doc, str(read.model), visiting),
        "train": dataclasses.asdict(doc.train)
        if doc.train is not None and _uses_trained_featurizer(doc, read.featurizer)
        else None,
    }


def _edit_operand_names(doc: Document, ename: str) -> tuple[str, ...]:
    do = doc.edits[ename].do
    payload = do.payload
    names: list[str] = []
    if isinstance(payload, str):
        names.append(payload)
    elif isinstance(payload, Mapping):
        names.extend(v for v in payload.values() if isinstance(v, str))
    return tuple(n for n in names if n in doc.reads)


def _entry(table: Mapping[str, Any], name: str) -> Any:
    return dataclasses.asdict(table[name])


def _pos_entry(doc: Document, pos: Any) -> Any:
    if isinstance(pos, str):
        resolved = doc.positions[pos]
        return (
            dataclasses.asdict(resolved)
            if isinstance(resolved, PositionSpec)
            else str(resolved)
        )
    return dataclasses.asdict(pos) if isinstance(pos, PositionSpec) else pos


def _featurizer_entry(doc: Document, ref: Any) -> Any:
    if ref is None:
        return None
    chain = (ref,) if isinstance(ref, str) else tuple(ref)
    return [dataclasses.asdict(doc.featurizers[name]) for name in chain]


def _encode(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"unencodable {type(obj).__name__} in a plan closure")
