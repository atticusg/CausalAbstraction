"""The planning skeleton: models → forward groups, shared work interned.

Execution semantics (spec §4) derive everything from the document: for each
expanded point, the models to run are ``original`` on every input it is
read on, plus each intervened model on its declared input — one **forward
group** each. ``num_forwards`` is a property of the plan, never authored.

Across the points of a swept document, the planner **content-dedups**: a
forward group's identity is the digest of its full dependency closure
(model + in-force writes + their operand reads' closures + input binding),
so a harvest shared by nine fits interns to one group and the sharing falls
out of value identity, not scheduling cleverness (§3). Backends consume
this plan as data; fusion, batching and staging stay their call (§8).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping

from causalab.protocol.schema import (
    METRIC_DOMAINS,
    Document,
    PositionSpec,
    concrete_int,
)

__all__ = [
    "COMPONENT_RANK",
    "UNRANKED",
    "ForwardGroup",
    "Materialization",
    "PointPlan",
    "Tap",
    "generated_budget",
    "plan_point",
    "closure_digest",
]

#: Intra-block execution order of the component vocabulary — backend-free
#: data used only to find a group's deepest tap (elision, §4). ``ln_final``
#: and ``lm_head`` sort after every block.
#:
#: **Numbered in hundreds, with the attention band deliberately spread.** The
#: values are ordinal and nothing outside this table reads them, but changing
#: one changes group elision and therefore every closure digest — so the point
#: of the spacing is that inserting a component never renumbers an existing one.
#: The attention interior is where the vocabulary is still growing (round 2 adds
#: the pre-RoPE projections, the gate, the post-RoPE q/k, the scores, the mixer
#: output and the per-head result), and the reserved slots below say where each
#: goes so that each PR is an insertion rather than a re-pin.
#:
#: Still reserved, in forward order through the mixer::
#:
#:     200  attention_query             post-RoPE q, at the interface
#:     210  attention_key               post-RoPE k, at the interface
#:     220  attention_scores            the softmax's input
#:     240  attention_z                 the interface's return, pre-gate
#:     350  attention_result            the per-head contribution to the stream
COMPONENT_RANK: dict[str, int] = {
    "input_ids": -10,  # the model's input: before every activation
    "embeddings": 0,
    "block_input": 100,
    "attention_input_norm": 150,  # input_layernorm, between resid_pre and mixer
    # The mixer's interior, in the order the forward computes it. All four are
    # module boundaries: q_norm/k_norm run BEFORE RoPE and are nn.Modules, so
    # the pre-RoPE projections are ordinary forward hooks rather than taps
    # inside the attention function.
    "attention_query_pre_rope": 160,
    "attention_key_pre_rope": 170,
    "attention_value_states": 180,
    # produced with q (one fused projection) and consumed at the very end, at
    # `attn_output * sigmoid(gate)` — ranked where it is produced
    "attention_gate": 190,
    "attention_probs": 230,
    # 🔤 `attention_premix` was `attention_value` until round 2. It is the
    # o-projection's INPUT — on a gated family `z · σ(gate)`, on an ungated one
    # `z` — which is the mixer's output just before it is mixed back into the
    # residual stream, and is not the value vectors that name suggested. Round 2
    # introduces those separately, and two components a letter apart in meaning
    # and identical in name is nnterp#51's cautionary tale happening to us.
    "attention_premix": 300,
    "attention_output": 400,
    # resid_mid is post_attention_layernorm's INPUT and mlp_input_norm its
    # OUTPUT, so the two straddle that one module in this order
    "block_mid": 450,
    "mlp_input_norm": 470,
    "mlp_input": 500,
    # The MoE interior, between the block's input and its output: the router
    # fires first, then the experts, then the combine.
    "router_logits": 510,
    "router_scores": 520,
    "expert_idx": 530,
    "expert_output": 540,  # per-expert interior (still refused — follow-up F2)
    "routed_output": 550,
    "mlp_activation": 600,
    # the shared expert runs beside the routed ones; its gate is *consumed*
    # last, at the multiply that produces the (derived) gated output
    "shared_expert_gate_proj": 610,
    "shared_expert_up_proj": 620,
    "shared_expert_activation": 630,
    "shared_expert_output": 640,
    "shared_expert_gate": 650,
    "mlp_output": 700,
    "block_output": 800,
    "ln_final": 900,
    "lm_head": 1000,
}

#: The rank of a component the table does not know. Deliberately past
#: ``lm_head``: an unranked tap sorts last, so it is treated as the deepest and
#: nothing is elided behind it. Being wrong in the other direction would elide a
#: forward that a later tap still needed.
UNRANKED = 10_000


@dataclasses.dataclass(frozen=True)
class Tap:
    """One value to materialize in a group's forward: a read's address."""

    read: str
    site: str
    depth: tuple[int, int]  # (layer, component rank) — the elision key


@dataclasses.dataclass(frozen=True)
class Materialization:
    """What one continuation read obliges a backend to build.

    ``needs_distribution`` is the expensive bit: a vocabulary-wide tensor
    per addressed position. It is false when the read is neither saved nor
    reduced by a metric that consumes distributions, in which case the
    backend must not build one (§8). *How* it avoids building one — a
    narrowed projection, per-step captures, a replay — is the backend's
    choice; this is the requirement, not the mechanism."""

    read: str
    site: str
    needs_distribution: bool


@dataclasses.dataclass(frozen=True)
class ForwardGroup:
    """One forward pass: a model (original or an IM) on one input role.

    ``digest`` is the content identity of everything that determines this
    group's activations — equal digests across points mean one shared
    forward. ``decode_depth`` is the greedy budget this group must decode
    for (0 = prefill only), and ``materialize`` states what its
    continuation reads oblige — both derived, never authored (§6)."""

    model: str
    input: str
    taps: tuple[Tap, ...]
    digest: str
    decode_depth: int = 0
    materialize: tuple[Materialization, ...] = ()

    @property
    def stop_after(self) -> tuple[int, int] | None:
        """The deepest tap's depth — a backend may end the forward there
        (§4 elision). ``None`` when the group has no taps, and also when it
        decodes: every decode step needs the head, so there is nothing to
        elide."""
        if self.decode_depth:
            return None
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
    identity = dict(data_identity or {})
    body = {
        "network": {"key": doc.model.key, "revision": doc.model.revision},
        "model": _model_closure(doc, model, set(), identity),
        "input": input_role,
        "data": identity.get(input_role),
    }
    digest = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), default=_encode
        ).encode()
    ).hexdigest()
    # The digest stays activation-identity: a decode changes what the group
    # *produces*, not what its prefill computes — the same reason taps are
    # not in it. Two points that differ only in decode depth share a prefill.
    depth = 0
    materialize: list[Materialization] = []
    for rname, read in doc.reads.items():
        if read.model != model or str(read.input) != input_role:
            continue
        budget = generated_budget(doc, read.pos)
        if budget is None:
            continue
        depth = max(depth, budget)
        materialize.append(
            Materialization(
                read=rname,
                site=str(read.site),
                needs_distribution=_needs_distribution(doc, rname),
            )
        )
    return ForwardGroup(
        model=model,
        input=input_role,
        taps=taps,
        digest=digest,
        decode_depth=depth,
        materialize=tuple(materialize),
    )


def generated_budget(doc: Document, pos: Any) -> int | None:
    """The decode budget of a position, or ``None`` for the prompt frame.

    Takes the spelling a read carries (a positions-table name or an inline
    spec) and returns the concrete budget — points are concrete by the time
    they are planned, so a surviving sweep wrapper is a caller error."""
    spec = doc.positions.get(pos) if isinstance(pos, str) else pos
    if not isinstance(spec, PositionSpec) or spec.generated is None:
        return None
    return concrete_int(spec.generated["max_new_tokens"], "generated.max_new_tokens")


def _needs_distribution(doc: Document, read: str) -> bool:
    """Whether anything downstream of ``read`` consumes a full distribution.

    Saving the read is the obvious case. So is any metric in the
    ``distribution`` domain (§2.10). An ``ids`` kind does **not** count: it
    consumes the tokens the decode produced, so a text-only probe obliges no
    vocabulary projection anywhere — which is the whole point of stating the
    requirement rather than always paying it."""
    if any(entry.value == read for entry in doc.save):
        return True
    for metric in doc.metrics.values():
        domain = METRIC_DOMAINS.get(str(metric.kind), "distribution")
        if str(metric.of) == read and domain == "distribution":
            return True
        target = metric.fields.get("target")
        if isinstance(target, str) and target == read:
            return True
    return False


def closure_digest(
    doc: Document, read_name: str, *, data_identity: Mapping[str, Any] | None = None
) -> str:
    """The content identity of one read's value: its address plus the full
    closure of the model it reads in. Equal digests across points mean one
    shared harvest (§3)."""
    body = {
        "network": {"key": doc.model.key, "revision": doc.model.revision},
        "read": _read_closure(doc, read_name, set(), dict(data_identity or {})),
    }
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), default=_encode
        ).encode()
    ).hexdigest()


def _depth(doc: Document, site_name: str) -> tuple[int, int]:
    site = doc.sites[site_name]
    layer = site.layer if isinstance(site.layer, int) else 0
    component = site.component if isinstance(site.component, str) else "lm_head"
    rank = COMPONENT_RANK.get(component, UNRANKED)
    if component in ("ln_final", "lm_head"):
        return (1_000_000, rank)  # after every block
    return (layer, rank)


def _model_closure(
    doc: Document, model: str, visiting: set[str], identity: Mapping[str, Any]
) -> Any:
    """Everything that determines a model's activations: for an IM, the
    in-force writes and, recursively, their operand reads' closures. The
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
    writes: dict[str, Any] = {}
    for ename in sorted(im.writes if isinstance(im.writes, tuple) else ()):
        write = doc.writes[ename]
        operands: dict[str, Any] = {}
        for op in _write_operand_names(doc, ename):
            if op in doc.reads:
                operands[op] = _read_closure(doc, op, visiting | {model}, identity)
        for op in _write_param_names(doc, ename):
            operands[op] = _entry(doc.params, op) if op in doc.params else op
        writes[ename] = {
            "site": _entry(doc.sites, str(write.site)),
            "pos": _pos_entry(doc, write.pos),
            "featurizer": _featurizer_entry(doc, write.featurizer),
            "dims": write.dims,
            "do": {str(write.do.mechanism): write.do.payload},
            "operands": operands,
        }
        if doc.train is not None and _uses_trained_featurizer(doc, write.featurizer):
            # a trained featurizer's weights are a function of the whole fit —
            # two points differing only in train.seed must never intern
            train_dep = dataclasses.asdict(doc.train)
    return {
        "input": im.input,
        "data": identity.get(str(im.input)),
        "writes": writes,
        "train": train_dep,
    }


def _write_param_names(doc: Document, ename: str) -> tuple[str, ...]:
    """Operand names that resolve to params entries or featurizer slots —
    their specs are part of the written value's identity."""
    do = doc.writes[ename].do
    payload = do.payload
    names: list[str] = []
    if isinstance(payload, str):
        names.append(payload)
    elif isinstance(payload, Mapping):
        names.extend(v for v in payload.values() if isinstance(v, str))
    return tuple(n for n in names if n not in doc.reads)


def _uses_trained_featurizer(doc: Document, ref: Any) -> bool:
    if doc.train is None or ref is None:
        return False
    trained = {p.split(".", 1)[0] for p in doc.train.params}
    chain = (ref,) if isinstance(ref, str) else tuple(ref)
    return any(name in trained for name in chain)


def _read_closure(
    doc: Document, read_name: str, visiting: set[str], identity: Mapping[str, Any]
) -> Any:
    read = doc.reads[read_name]
    return {
        "site": _entry(doc.sites, str(read.site)),
        "pos": _pos_entry(doc, read.pos),
        "featurizer": _featurizer_entry(doc, read.featurizer),
        "dims": read.dims,
        "input": read.input,
        "data": identity.get(str(read.input)),
        "model": _model_closure(doc, str(read.model), visiting, identity),
        "train": dataclasses.asdict(doc.train)
        if doc.train is not None and _uses_trained_featurizer(doc, read.featurizer)
        else None,
    }


def _write_operand_names(doc: Document, ename: str) -> tuple[str, ...]:
    do = doc.writes[ename].do
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
