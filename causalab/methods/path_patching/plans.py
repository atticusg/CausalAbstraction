"""Lower a path-patched edge onto the Plan IR — the PL5 core.

One batch of counterfactual examples becomes **one**
:class:`~causalab.neural.plan.Plan` whose named inputs and cross-input
``ReadSource`` edges spell out the dataflow the pyvene implementation encoded
in a mixed intervenable model plus the ``sorted_keys`` collect-order contract
(``docs/PATH_PATCHING.md`` §8.3):

* ``source`` — the counterfactual batch. The sender's interchange reads it.
* ``base`` — the base batch, carrying PASS 1's interventions: the sender
  frozen to ``source``, every restorer frozen to ``clean``.
* ``clean`` — the *same* base tensors as a separate, unintervened input: the
  restorers' freeze values. A same-input read at the written site would see
  the already-patched activation, so the clean pass is declared explicitly
  (the duplicate-input convention, :mod:`causalab.neural.staged`). Only
  present when there are restorers.
* ``final`` — the base tensors again, carrying PASS 2: each receiver
  overwritten with ``ReadSource(receiver_site, input="base")`` — the
  receiver's activation *under* PASS 1's interventions (``v*``). The
  collect∘inject that pyvene split across two models and an ordering contract
  is a single named edge per receiver; order-invariance holds by
  construction. Only present for internal receivers; the degenerate
  ``receiver = output`` case saves the ``base`` logits directly.

How many forward passes actually run is the compiler's decision
(:func:`~causalab.neural.plan.run_plan`, ``lowering="auto"``): the
no-restorer output case is the canonical single fused trace; the two-pass
shape stages ``final`` after ``base`` because ``base`` is itself an in-trace
consumer — exactly the minimal pass structure.

Positions resolve per batch through the ST2 bridge
(:func:`causalab.neural.positions.resolve_positions`): the sender reads the
source at the source-side positions and writes the base at the base-side
positions; restorers and receivers read/write at the receiver's token
position (the sender's for the output case). Width guards fire here, before
any forward pass, with role-specific messages.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.path_patching.targets import (
    OUTPUT,
    ReceiverSpec,
    build_restorer_sites,
)
from causalab.neural.edit import Edit, ReadSource
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.head_view import HeadSite
from causalab.neural.pipeline import LMPipeline, ensure_position_ids
from causalab.neural.plan import EditOp, Plan
from causalab.neural.positions import resolve_positions
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec

__all__ = ["build_edge_plan"]


def _swap(f: Any, f_src: Any) -> Any:
    """The interchange ``g``: replace the base features with the source's."""
    return f_src


def _check_uniform(rows: Sequence[Sequence[int]], role: str) -> None:
    """Every example must select the same number of tokens for ``role``.

    A ragged span cannot batch as one ``(batch, k)`` gather/scatter; it would
    otherwise surface as :func:`causalab.neural.site._sequence_index`'s generic
    ragged error (or a backend broadcast error) without saying which edge role
    is at fault.
    """
    widths = sorted({len(row) for row in rows})
    if len(widths) > 1:
        first = len(rows[0])
        offender = next(i for i, row in enumerate(rows) if len(row) != first)
        raise ValueError(
            f"A path-patching {role} selects a variable number of tokens across "
            f"the batch (example 0 selects {first}, example {offender} selects "
            f"{len(rows[offender])}). Batched interchange writes one "
            f"(batch, k) position block per site, so every example must select "
            f"the same number of tokens. Use a fixed-width {role} position "
            f"(e.g. the last token)."
        )


def _check_matched(
    base_rows: Sequence[Sequence[int]], src_rows: Sequence[Sequence[int]]
) -> None:
    """The sender writes ``len(source positions)`` vectors into
    ``len(base positions)`` slots — a per-example mismatch would reach the
    backend as a broadcast error mid-trace."""
    for i, (b, s) in enumerate(zip(base_rows, src_rows)):
        if len(b) != len(s):
            raise ValueError(
                f"Sender position width differs between base and source for "
                f"example {i}: the base selects {len(b)} token(s) but the "
                f"source selects {len(s)}. The interchange writes the source's "
                f"vectors into the base's slots one-to-one, so both sides must "
                f"select the same number of tokens."
            )


def build_edge_plan(
    pipeline: LMPipeline,
    examples: list[CounterfactualExample],
    sender: SiteSpec,
    receiver_sites: Sequence[Site | HeadSite],
    range_receiver: ReceiverSpec = OUTPUT,
    *,
    restore: Iterable[str] = ("attention", "mlp"),
    restorer_sites: list[Site] | None = None,
) -> tuple[Plan, str]:
    """Assemble the Plan for one batch of a path-patched edge.

    ``examples`` follow the counterfactual-dataset shape (``input`` +
    ``counterfactual_inputs``, the source first). ``receiver_sites`` are the
    resolved internal receiver locations (empty for the ``output`` receiver —
    the one-pass direct effect). ``range_receiver`` is the
    :class:`ReceiverSpec` the restorer *range* and freeze position are built
    against — the receiver itself for one receiver, the deepest member for a
    set (its read point bounds the union restorer set, isolating the direct
    sender→R_k edge for every member at once).

    ``restorer_sites`` overrides the built set (``[]`` for an explicitly
    restorer-free edge — the parity-test seam); ``None`` builds it from
    ``restore`` via :func:`build_restorer_sites`.

    Returns ``(plan, logits_key)`` — the key (``"final"`` or ``"base"``) whose
    saved logits carry the patched run's output.
    """
    if restorer_sites is None:
        restorer_sites = build_restorer_sites(
            pipeline, sender, range_receiver, restore=restore
        )

    base_traces = [ex["input"] for ex in examples]
    source_traces = [ex["counterfactual_inputs"][0] for ex in examples]
    base_in = ensure_position_ids(pipeline.load(base_traces))
    source_in = ensure_position_ids(pipeline.load(source_traces))

    # -- positions (ST2 bridge; padded-batch frames) --------------------------- #
    sender_pos = sender.positions
    if sender_pos is None:
        raise ValueError(
            f"sender {sender.key!r} has positions=None (unbound): path patching "
            "needs the sender's token positions on both sides of the edge. Bind "
            "them via spec.with_positions(...), literal rows, or "
            "load_site_specs(dir, token_positions=...)."
        )
    src_rows = resolve_positions(
        sender_pos, source_traces, source_in["attention_mask"], is_original=False
    )
    base_rows = resolve_positions(
        sender_pos, base_traces, base_in["attention_mask"], is_original=True
    )
    _check_uniform(src_rows, "sender")
    _check_uniform(base_rows, "sender")
    _check_matched(base_rows, src_rows)

    freeze_rows: list[list[int]] | None = None
    if restorer_sites or receiver_sites:
        # Restorers and receivers freeze/read at the receiver's token position
        # (the sender's for the output case — no internal read point).
        freeze_pos = (
            range_receiver.token_position
            if range_receiver.kind != "output"
            else sender_pos
        )
        assert freeze_pos is not None  # internal specs carry one by validation
        freeze_rows = resolve_positions(
            freeze_pos, base_traces, base_in["attention_mask"], is_original=True
        )
        _check_uniform(freeze_rows, "restorer" if not receiver_sites else "receiver")

    # -- ops -------------------------------------------------------------------- #
    sender_fsite = sender.fsite
    ops: list[EditOp] = [
        EditOp(
            "base",
            Edit(
                sender_fsite,
                g=_swap,
                read_sources=(
                    ReadSource(sender_fsite, positions=src_rows, input="source"),
                ),
                positions=base_rows,
            ),
        )
    ]
    for site in restorer_sites:
        fsite = FeaturizedSite(site)
        ops.append(
            EditOp(
                "base",
                Edit(
                    fsite,
                    g=_swap,
                    read_sources=(
                        ReadSource(fsite, positions=freeze_rows, input="clean"),
                    ),
                    positions=freeze_rows,
                ),
            )
        )
    for site in receiver_sites:
        fsite = FeaturizedSite(site)
        ops.append(
            EditOp(
                "final",
                Edit(
                    fsite,
                    g=_swap,
                    read_sources=(
                        # v*: the receiver read UNDER the "base" invoke's
                        # interventions — the collect∘inject as one named edge.
                        ReadSource(fsite, positions=freeze_rows, input="base"),
                    ),
                    positions=freeze_rows,
                ),
            )
        )

    inputs: dict[str, Any] = {"source": source_in, "base": base_in}
    if restorer_sites:
        inputs["clean"] = base_in
    logits_key = "base"
    if receiver_sites:
        inputs["final"] = base_in
        logits_key = "final"
    plan = Plan(inputs=inputs, ops=tuple(ops), save_logits=(logits_key,))
    return plan, logits_key
