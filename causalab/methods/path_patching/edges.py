"""Path specification for the analytic patching engine.

A patch is an explicit set of edges over {sender, receiver MLPs, logits}:

* ``sender -> receiver_k``   the sender's trunk delta enters receiver k's input
* ``sender -> logits``       the sender's direct edge to the final logits
* ``receiver_j -> receiver_k``  receiver j's output delta enters receiver k
* ``receiver_k -> logits``   receiver k's output delta reaches the logits

Each receiver's input delta sums only its included incoming edges; its output
delta propagates only along its included outgoing edges. Receivers are always
MLP blocks (the Hanna et al. 2023 receiver vocabulary); ordering between
receivers follows the architecture's block order, which the engine checks.

Granularity ceiling (documented, deliberate): edges are atomic. One edge
cannot carry different values per downstream branch (the "treeified" patching
expressible in rust-circuit); expressing that requires per-path value routing
the cache-arithmetic engine does not model.

The common case is the *closed cascade*: sender feeds the receiver set, every
receiver feeds every later receiver in the set and the logits. Build it with
:meth:`PathSpec.cascade`, the documented default entry point; the explicit
edge set is the power-user form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = ["PathSpec"]

Multipath = Literal["first", "all"]


@dataclass(frozen=True)
class PathSpec:
    """An explicit edge set over {sender, receiver MLPs, logits}.

    Attributes
    ----------
    receivers:
        MLP layers that recompute, in upstream-to-downstream order.
    sender_to:
        Receivers whose input includes the sender's trunk delta.
    receiver_to_receiver:
        Included ``(j, k)`` edges; j must be upstream of k. Any pair of
        receivers not listed here is an *excluded* edge: k recomputes as if
        j's output had stayed clean.
    sender_to_logits:
        Whether the sender's direct edge into the logits is patched.
    receivers_to_logits:
        Receivers whose output delta reaches the logits.
    """

    receivers: tuple[int, ...] = ()
    sender_to: frozenset[int] = frozenset()
    receiver_to_receiver: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    sender_to_logits: bool = True
    receivers_to_logits: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        rec = tuple(self.receivers)
        if len(set(rec)) != len(rec):
            raise ValueError(f"duplicate receivers: {rec}")
        if list(rec) != sorted(rec):
            raise ValueError(
                f"receivers must be in upstream-to-downstream (ascending layer) "
                f"order, got {rec}"
            )
        rset = set(rec)
        for bad in self.sender_to - rset:
            raise ValueError(f"sender_to names non-receiver layer {bad}")
        for bad in self.receivers_to_logits - rset:
            raise ValueError(f"receivers_to_logits names non-receiver layer {bad}")
        for j, k in self.receiver_to_receiver:
            if j not in rset or k not in rset:
                raise ValueError(f"edge ({j}, {k}) names a non-receiver layer")
            if j >= k:
                raise ValueError(
                    f"edge ({j}, {k}) is not upstream->downstream (j < k required)"
                )

    # ------------------------------------------------------------------
    @classmethod
    def cascade(
        cls,
        receivers: list[int] | tuple[int, ...] = (),
        *,
        direct_to_logits: bool = True,
        multipath: Multipath = "first",
    ) -> "PathSpec":
        """The closed cascade: the documented default entry point.

        ``multipath="first"``: the sender's delta enters only the furthest-
        upstream receiver (the via-MLP-k iterative path patch of Hanna et al.
        Fig 3C). ``multipath="all"``: it enters every receiver directly (the
        paper's §3.3 indirect split, patching all of the sender's paths
        through the receiver set simultaneously). In both, every receiver
        feeds every later receiver in the set and the logits.
        """
        rec = tuple(sorted(receivers))
        if not rec:
            return cls(sender_to_logits=direct_to_logits)
        sender_to = frozenset(rec) if multipath == "all" else frozenset({rec[0]})
        r2r = frozenset((j, k) for i, j in enumerate(rec) for k in rec[i + 1 :])
        return cls(
            receivers=rec,
            sender_to=sender_to,
            receiver_to_receiver=r2r,
            sender_to_logits=direct_to_logits,
            receivers_to_logits=frozenset(rec),
        )

    # ------------------------------------------------------------------
    def without_edge(self, j: int, k: int) -> "PathSpec":
        """A copy with the receiver_j -> receiver_k edge excluded."""
        if (j, k) not in self.receiver_to_receiver:
            raise ValueError(f"edge ({j}, {k}) is not in the spec")
        return PathSpec(
            receivers=self.receivers,
            sender_to=self.sender_to,
            receiver_to_receiver=self.receiver_to_receiver - {(j, k)},
            sender_to_logits=self.sender_to_logits,
            receivers_to_logits=self.receivers_to_logits,
        )

    def describe(self) -> str:
        parts = []
        if self.sender_to_logits:
            parts.append("S->logits")
        parts += [f"S->m{k}" for k in sorted(self.sender_to)]
        parts += [f"m{j}->m{k}" for j, k in sorted(self.receiver_to_receiver)]
        parts += [f"m{k}->logits" for k in sorted(self.receivers_to_logits)]
        return " + ".join(parts) if parts else "(empty)"
