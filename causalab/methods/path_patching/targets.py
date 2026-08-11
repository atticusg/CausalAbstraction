"""Path-patching target construction for a sender → receiver edge.

Resolves the *locations* of a path-patched edge onto the Site stack: the
**sender** (a :class:`~causalab.neural.specs.SiteSpec` at the API boundary —
born holding a real engine :class:`~causalab.neural.site.Site` /
:class:`~causalab.neural.head_view.HeadSite`, WU4 #506), the **receiver**
(:class:`ReceiverSpec` → :func:`build_receiver_site`), and the **restorer set**
(:func:`build_restorer_sites` — the components frozen to their clean-base value
between them). The hard correctness rule lives here — restorers are only
``attention_output`` / ``mlp_output`` components, **never** ``block_output``
(which bundles the residual stream carrying the sender's direct contribution
and would erase the path).

Receivers are ``output`` (the logits, the degenerate one-pass case) or an
internal ``head_value_input`` / ``head_query_input`` / ``mlp_input`` /
``residual`` read-point that the two-pass plan collects and re-injects. The
restorer **range** stops at the receiver's read point via the forward-order
depth rule below, reducing to "sender-layer MLP + every layer above" when the
receiver is the output.

The restorer set *is the estimand definition*, exposed via ``restore``:

* ``("attention", "mlp")`` (default) — freeze every attention **and** MLP output
  above the sender. The only surviving sender→logits route is the bare residual
  stream, so this measures a strict residual-to-output direct effect (the pyvene
  IOI-tutorial ``path_patching_config``).
* ``("attention",)`` — freeze only other attention outputs; MLPs (and LayerNorm)
  recompute. This is the Wang et al. (2022) §3.1 direct effect: the sender's
  contribution may flow through residual + MLPs, just not through other attention
  heads.

(Caveat vs the paper: the whole-block ``attention_output`` restorer does not
freeze the *sender layer's other heads* — a small over-inclusion of same-layer
attention paths.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from causalab.neural.head_view import HeadSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec
from causalab.neural.token_positions import TokenPosition

RestoreFamily = Literal["attention", "mlp"]
_VALID_FAMILIES: frozenset[str] = frozenset({"attention", "mlp"})

ReceiverKind = Literal[
    "output", "head_value_input", "head_query_input", "mlp_input", "residual"
]


@dataclass(frozen=True)
class ReceiverSpec:
    """What the receiver of a path-patched edge is.

    * ``output`` (default) — the logits / unembedding (the IOI Fig. 3
      direct-effect case). The two-pass procedure collapses to a *single*
      intervened forward (there is nothing downstream of the output to strip), so
      this never needs the collect/inject machinery.
    * ``head_value_input`` — a head's value path at ``(layer, head)``, realized as
      its per-head value vector (:class:`~causalab.neural.head_view.HeadSite`
      kind ``"value"``). The canonical "v of head h" receiver.
    * ``head_query_input`` — a head's query path at ``(layer, head)``, realized as
      its per-head query vector (:class:`~causalab.neural.head_view.HeadSite`
      kind ``"query"``). The "q of head h"
      receiver — the Fig. 4 S-Inhibition → Name-Mover *query* edge. Read at the
      same depth as the value receiver (``2·layer``); the query vector is
      per-query-head, so unlike the value path it needs no GQA KV-group remap.
    * ``mlp_input`` — the input to the MLP sublayer at ``layer``.
    * ``residual`` — a residual-stream position at ``layer`` (``block_output`` by
      default, or ``block_input`` via ``residual_point``).

    Every internal receiver carries the ``token_position`` it reads at; the
    residual stream at a position is written only by components at that position,
    so that position also fixes where the restorers freeze.
    """

    kind: ReceiverKind = "output"
    layer: int | None = None
    head: int | None = None
    token_position: TokenPosition | None = None
    residual_point: Literal["block_input", "block_output"] = "block_output"

    def __post_init__(self) -> None:
        if self.kind == "output":
            if any(x is not None for x in (self.layer, self.head, self.token_position)):
                raise ValueError(
                    "ReceiverSpec(kind='output') takes no layer/head/token_position."
                )
            return
        if self.layer is None or self.token_position is None:
            raise ValueError(
                f"ReceiverSpec(kind={self.kind!r}) requires layer and token_position."
            )
        if self.kind in ("head_value_input", "head_query_input"):
            if self.head is None:
                raise ValueError(f"ReceiverSpec(kind={self.kind!r}) requires head.")
        elif self.head is not None:
            raise ValueError(f"ReceiverSpec(kind={self.kind!r}) does not take a head.")


OUTPUT = ReceiverSpec(kind="output")


def build_receiver_site(
    pipeline: LMPipeline, receiver: ReceiverSpec
) -> Site | HeadSite | None:
    """Resolve the receiver's read/write location to collect (PASS 1) and inject
    (PASS 2).

    Returns ``None`` for ``kind='output'`` (the degenerate single-forward case has
    no receiver location — the metric is read straight off the logits). Internal
    receivers resolve to a per-head :class:`HeadSite` (``value`` / ``query``) or a
    whole-component :class:`Site` (``mlp_input``, ``block_input`` /
    ``block_output``); the receiver's ``token_position`` stays on the spec and is
    resolved per batch by the plan builder. Unlike pyvene 0.1.8 (which sliced
    value vectors at ``hidden // n_head``), :class:`HeadSite` honours a decoupled
    ``config.head_dim`` (e.g. Qwen3), so value receivers on such models are
    supported.
    """
    if receiver.kind == "output":
        return None
    config = pipeline.model.config
    assert receiver.layer is not None and receiver.token_position is not None
    if receiver.kind == "head_value_input":
        assert receiver.head is not None
        n_head = config.num_attention_heads
        n_kv_heads = getattr(config, "num_key_value_heads", None) or n_head
        if not 0 <= receiver.head < n_head:
            raise ValueError(
                f"head_value_input receiver head={receiver.head} is out of range "
                f"0..{n_head - 1} (num_attention_heads={n_head})."
            )
        # The value vector is indexed in KV-head space (v_proj splits by
        # num_key_value_heads). Under grouped-/multi-query attention (n_kv < n_head)
        # each value vector is *shared* by a group of query heads, so the receiver's
        # query head maps to its KV group `head // (n_head // n_kv)`. On non-GQA
        # (n_kv == n_head) this is the identity. Consequence: injecting the group's
        # value vector in PASS 2 reaches every query head in the group, so the measured
        # edge is "into the group's shared value", not one isolated head — a
        # per-query-head value path does not exist in GQA.
        kv_head = receiver.head // (n_head // n_kv_heads)
        return HeadSite("value", receiver.layer, kv_head)
    if receiver.kind == "head_query_input":
        assert receiver.head is not None
        n_head = config.num_attention_heads
        if not 0 <= receiver.head < n_head:
            raise ValueError(
                f"head_query_input receiver head={receiver.head} is out of range "
                f"0..{n_head - 1} (num_attention_heads={n_head})."
            )
        # Queries are per-query-head even under GQA — only k/v are shared across a
        # KV group — so the receiver's head maps to itself with no KV-group remap
        # (contrast head_value_input above).
        return HeadSite("query", receiver.layer, receiver.head)
    if receiver.kind == "mlp_input":
        return Site("mlp_input", receiver.layer)
    if receiver.kind == "residual":
        return Site(receiver.residual_point, receiver.layer)
    raise ValueError(f"Unknown receiver kind {receiver.kind!r}.")


def _normalize_restore(restore: Iterable[str]) -> tuple[str, ...]:
    """Validate and de-duplicate the restorer-family selection."""
    families = tuple(dict.fromkeys(restore))  # order-stable de-dup
    if not families:
        raise ValueError(
            "restore must name at least one component family from "
            f"{sorted(_VALID_FAMILIES)}; got an empty selection."
        )
    bad = set(families) - _VALID_FAMILIES
    if bad:
        raise ValueError(
            f"restore families {sorted(bad)} are not valid; choose from "
            f"{sorted(_VALID_FAMILIES)}."
        )
    return families


# Residual-stream writers/readers laid out in forward order as integer "depths":
# layer i's attention output writes at 2*i, its MLP output at 2*i+1. A restorer
# belongs strictly *between* the sender's write and the receiver's read — i.e. its
# depth d satisfies sender_depth < d < receiver_read_depth — and is filtered by the
# `restore` families. This one rule subsumes the old "sender-layer MLP + every layer
# above" logic (recovered exactly when the receiver is the output) and the mid-block
# membership for internal receivers (e.g. an MLP input reads after its layer's
# attention, so that attention is a restorer; a head's value input reads before its
# layer's attention, so it is not).


def _sender_write_depth(sender: SiteSpec) -> int:
    # Structural placement off the spec's engine site: every per-head site
    # (HeadSite, whatever its kind) writes within its layer's attention, as
    # does the whole attention sublayer output.
    site = sender.fsite.site
    layer = site.layer  # type: ignore[attr-defined]  # Site and HeadSite both carry it
    if isinstance(site, HeadSite):
        return 2 * layer
    component = site.component  # type: ignore[attr-defined]
    if component == "attention_output":
        return 2 * layer
    if component in ("mlp_output", "mlp_input", "mlp_activation"):
        return 2 * layer + 1
    if component == "block_output":
        return 2 * layer + 1
    if component == "block_input":
        # Reads/writes before layer L's attention; at layer 0 this is depth -1
        # (before everything), which is intentional — it places such a sender
        # upstream of all writers.
        return 2 * layer - 1
    raise ValueError(
        f"Cannot place sender component {component!r} in the residual order."
    )


def _receiver_read_depth(receiver: ReceiverSpec, n_layers: int) -> int:
    """Forward-order depth at which the receiver reads the residual stream.

    Restorers are everything written before this point (and after the sender).
    """
    if receiver.kind == "output":
        return 2 * n_layers  # past every writer — freeze the whole stack above
    assert receiver.layer is not None
    if receiver.kind in ("head_value_input", "head_query_input"):
        # Both the value vector (W_V·x) and the query vector (W_Q·x) are projected
        # from the residual *entering* attention at `layer`, so they share a read
        # depth — the sender→receiver restorer set is identical; only q vs v differs.
        return 2 * receiver.layer  # reads the residual entering attention `layer`
    if receiver.kind == "mlp_input":
        return 2 * receiver.layer + 1  # reads after attention `layer`, before its MLP
    if receiver.kind == "residual":
        return (
            2 * receiver.layer + 2
            if receiver.residual_point == "block_output"
            else 2 * receiver.layer
        )
    raise ValueError(f"Unknown receiver kind {receiver.kind!r}.")


def sender_reaches_receiver(
    pipeline: LMPipeline, sender: SiteSpec, receiver: ReceiverSpec
) -> bool:
    """True iff the sender writes to the residual stream before the receiver reads it.

    When ``False`` the sender sits at or downstream of the receiver's read point, so no
    forward path connects them and the edge's direct effect is structurally zero (a
    wasted run). ``output`` receivers read past the whole stack, so every sender reaches
    them.
    """
    n_layers = pipeline.model.config.num_hidden_layers
    return _sender_write_depth(sender) < _receiver_read_depth(receiver, n_layers)


def deepest_receiver(
    pipeline: LMPipeline, receiver_specs: list[ReceiverSpec]
) -> ReceiverSpec:
    """The receiver in the set that reads the residual stream latest (forward order).

    The restorer set for a receiver *set* is built against this receiver (see
    :func:`build_restorer_set`): freezing every component between the sender and the
    deepest read point isolates the direct sender→R_k edge for *every* R_k in the set
    simultaneously, by forward-order hook firing — a restorer above a shallower
    receiver's read depth fires after that receiver has already projected its
    query/value, so it cannot perturb it; a restorer below correctly freezes the
    indirect path. Ties (same read depth) resolve to the first such spec; the restorer
    set is identical for any of them.
    """
    if not receiver_specs:
        raise ValueError("receiver_specs is empty")
    n_layers = pipeline.model.config.num_hidden_layers
    return max(receiver_specs, key=lambda r: _receiver_read_depth(r, n_layers))


def sender_reaches_any(
    pipeline: LMPipeline,
    sender: SiteSpec,
    receiver_specs: list[ReceiverSpec],
) -> bool:
    """True iff the sender reaches at least one receiver in the set.

    Equivalent to reaching the *deepest* receiver: ``deepest`` has the maximal read
    depth, so a sender at or downstream of it is at or downstream of every receiver
    (no forward path to any — the set's direct effect is a structural zero). When this
    is ``False`` the scan scores the cell ``nan`` rather than running a wasted forward.
    """
    return sender_reaches_receiver(
        pipeline, sender, deepest_receiver(pipeline, receiver_specs)
    )


def build_restorer_sites(
    pipeline: LMPipeline,
    sender: SiteSpec,
    receiver: ReceiverSpec = OUTPUT,
    *,
    restore: Iterable[str] = ("attention", "mlp"),
) -> list[Site]:
    """Construct the restorer locations for the edge ``sender → receiver``.

    Returns ``attention_output`` / ``mlp_output`` :class:`Site`\\ s, never
    ``block_output`` (which bundles the residual stream carrying the sender's
    direct contribution). A component is a restorer iff its forward-order depth
    lies strictly between the sender's write and the receiver's read (see the
    module-level depth note), and its family is in ``restore`` (``"attention"``
    and/or ``"mlp"``).

    For ``receiver = output`` this is the sender-layer MLP plus every attention /
    MLP output above the sender — the IOI-tutorial restorer set. For an internal
    receiver the range stops at the receiver's read point. The *position* the
    restorers freeze at is the plan builder's job (the receiver's token position,
    falling back to the sender's for the output case — the residual at a position
    is written only by components at that position, so freezing there isolates
    the direct edge while staying single-token).
    """
    families = _normalize_restore(restore)
    n_layers = pipeline.model.config.num_hidden_layers
    sender_depth = _sender_write_depth(sender)
    read_depth = _receiver_read_depth(receiver, n_layers)

    restorers: list[Site] = []
    for i in range(n_layers):
        if "attention" in families and sender_depth < 2 * i < read_depth:
            restorers.append(Site("attention_output", i))
        if "mlp" in families and sender_depth < 2 * i + 1 < read_depth:
            restorers.append(Site("mlp_output", i))
    return restorers
