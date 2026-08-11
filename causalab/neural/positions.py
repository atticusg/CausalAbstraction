"""Resolver output → padded-batch-frame indices: the position-resolver bridge.

The ST2 half of the Site story (#397). :mod:`causalab.neural.token_positions`
already resolves *which tokens* an intervention touches — ``Template``
char→token tracking over ``offset_mapping``, declarative specs, the
``TokenPosition`` combinators — and is backbone-agnostic, reused as-is. But
its indexers return positions in each example's **unpadded** tokenization
frame, while everything that consumes positions at run time
(:meth:`causalab.neural.site.Site.read` / :meth:`~causalab.neural.site.Site.write`
slicing an in-trace activation) indexes the **padded batch** the model
actually runs on. Under the pipeline's left-padding convention each row's
offset differs, so the rewrite is per-example.

This module is that bridge, and the single home of the padding-frame shift:

* :func:`shift_to_padded_frame` — rewrite per-example unpadded-frame index
  rows into the padded-batch frame given the batch ``attention_mask``,
  bounds-checked per row. The single home of the shift semantics (the
  pyvene-era unit-side wrapper that delegated here was deleted in the
  where-unification sweep, #508).
* :func:`resolve_positions` — resolve a position resolver (any
  ``ComponentIndexer``, including ``TokenPosition`` and its paired / combined
  / dynamic combinators, or a static index list broadcast to every row) over
  a batch of traces, then shift the rows into the padded frame.
* :func:`resolve_positions_batched` — the batch-first path (PL3, #405):
  resolve directly against the batch's own *run encoding*
  (``pipeline.load(traces, return_offsets_mapping=True)``), so declarative
  positions are **born in the padded frame** — one tokenization per batch,
  no unpadded→padded shift, and no way to index a different tokenization
  than the run (the #176 stale-index class). Hand-built indexer closures
  fall back to the per-example resolve + shift, anchored to the same run
  encoding's mask.

The output rows feed :meth:`Site.read` / :meth:`Site.write` directly:
equal-width rows batch as one per-row slice; ragged rows (a multi-token
variable whose width differs across examples) also resolve here — batching
them is the plan compiler's job (PL3).

Deliberately torch-only and duck-typed on the resolver, so trace-side
consumers — :mod:`causalab.neural.site` today, the plan compiler (PL1/PL3)
tomorrow — can import it without dragging in the pipeline or the pyvene
layer.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, cast

import torch

__all__ = [
    "PositionResolver",
    "resolve_positions",
    "resolve_positions_batched",
    "shift_to_padded_frame",
]


class PositionResolver(Protocol):
    """The ``ComponentIndexer.index`` contract this module consumes.

    ``TokenPosition`` and every combinator over it satisfy this by
    inheritance; anything else only needs the same call shape. Structural on
    purpose — importing the concrete class would pull the pyvene-era
    ``units`` module into trace-side import graphs.
    """

    def index(
        self, input: Any, batch: bool = False, is_original: bool | None = None
    ) -> Any: ...


def _shift_row(
    positions: Sequence[int], shift: int, padded_len: int, ex_idx: int
) -> list[int]:
    """Add ``shift`` to every position, bounds-checked against ``padded_len``."""
    shifted = [int(p) + shift for p in positions]
    oob = [p for p in shifted if p < 0 or p >= padded_len]
    if oob:
        raise ValueError(
            f"Position {oob} is out of bounds for padded length {padded_len} "
            f"(example {ex_idx}, per-row shift {shift}). Token-position indices "
            "must fall within the padded sequence; out of bounds they would "
            "silently address the wrong token, or reach the backend's "
            "gather/scatter as a CUDA assertion that poisons the context. "
            "Common causes: a position computed for a differently-shaped input "
            "(e.g. a base position reused on a shorter counterfactual, #176), "
            "or — when `pipeline.max_length` is set — an indexer returning "
            "positions already in the padded frame so the shift pushes them "
            "past the end (use `pipeline.max_length=None`)."
        )
    return shifted


def shift_to_padded_frame(
    rows: Sequence[Sequence[int]], attention_mask: torch.Tensor
) -> list[list[int]]:
    """Rewrite per-example unpadded-frame rows into the padded-batch frame.

    ``rows[i]`` holds example ``i``'s token indices in its own *unpadded*
    tokenization frame (what every indexer in
    :mod:`causalab.neural.token_positions` returns). ``attention_mask`` is the
    padded batch's mask (``pipeline.load(...)['attention_mask']``); its
    ``argmax`` per row gives the offset — ``0`` for right-padding,
    ``padded_len - unpadded_len`` for left-padding. Assumes contiguous padding
    (a prefix of zeros for left-pad or a suffix for right-pad); interior zeros
    are not supported.

    Every row is bounds-checked against the padded length **even when its
    shift is zero**: a stale index computed for a differently-shaped input
    (#176) must fail here as a catchable ``ValueError``, not downstream as a
    silent wrong-token read or a poisoned CUDA context.
    """
    n_rows = int(attention_mask.shape[0])
    if len(rows) != n_rows:
        raise ValueError(
            f"got {len(rows)} position rows for an attention_mask with "
            f"{n_rows} rows — positions must be resolved for exactly the "
            "examples in the padded batch"
        )
    per_row_shift = torch.argmax(attention_mask.int(), dim=1).tolist()
    padded_len = int(attention_mask.shape[1])
    return [
        _shift_row(row, per_row_shift[i], padded_len, ex_idx=i)
        for i, row in enumerate(rows)
    ]


def resolve_positions(
    positions: PositionResolver | Sequence[int] | torch.Tensor,
    traces: Sequence[Any],
    attention_mask: torch.Tensor | None,
    *,
    is_original: bool | None = None,
) -> list[list[int]]:
    """Resolve ``positions`` over ``traces`` into padded-batch-frame rows.

    ``positions`` is either a resolver (:class:`PositionResolver` — any
    ``ComponentIndexer``, so ``TokenPosition`` and its paired / combined /
    dynamic combinators all work) called once per trace in the unpadded
    frame, or a static index list / 1-D tensor broadcast to every row.
    ``is_original`` is threaded through to the resolver so paired base /
    counterfactual positions route to the right side (ignored for static
    positions).

    ``attention_mask`` is the padded batch's mask for these same traces
    (``pipeline.load(traces)['attention_mask']``); the resolved rows are
    shifted into that frame via :func:`shift_to_padded_frame`. Pass ``None``
    to opt out and keep each row in its example's unpadded frame — only
    meaningful for diagnostics or single-example calls that never index into
    a padded tensor.

    Returns one list of indices per trace. Rows may be ragged (a multi-token
    variable whose width differs across examples); equal-width rows can be
    handed straight to ``Site.read`` / ``Site.write``.
    """
    if isinstance(positions, torch.Tensor):
        if positions.dim() != 1:
            raise ValueError(
                f"static positions tensor must be 1-D, got {positions.dim()}-D"
            )
        positions = [int(p) for p in positions.tolist()]
    if isinstance(positions, (list, tuple, range)):
        static = [int(p) for p in positions]
        rows: Sequence[Sequence[int]] = [list(static) for _ in traces]
    else:
        resolver = cast(PositionResolver, positions)
        rows = resolver.index(list(traces), batch=True, is_original=is_original)
    if attention_mask is None:
        return [list(row) for row in rows]
    return shift_to_padded_frame(rows, attention_mask)


def resolve_positions_batched(
    positions: Any,
    traces: Sequence[Any],
    encoding: Any,
    *,
    is_original: bool | None = None,
) -> list[list[int]]:
    """Resolve ``positions`` batch-first, against the batch's run encoding.

    ``encoding`` is the padded batch the model actually runs — ONE
    ``pipeline.load(traces, return_offsets_mapping=True)`` for these traces.
    Declarative resolvers (spec-built ``TokenPosition`` and its paired /
    combined / dynamic combinators) resolve as pure functions of their
    encoding row via ``index_on_encoding``, so the returned indices are born
    in the padded frame: one tokenization per batch instead of one per
    example per spec (the N+1), and no second frame to drift out of (#176).

    Static index lists and hand-built indexer closures take the legacy
    per-example path, shifted with the *run encoding's own* attention mask —
    same result frame, one apparatus. Rows may be ragged.

    Duck-typed on ``index_on_encoding`` so this module still imports without
    the pipeline layer.
    """
    index_on_encoding = getattr(positions, "index_on_encoding", None)
    if index_on_encoding is not None:
        rows = index_on_encoding(list(traces), encoding, is_original=is_original)
        if rows is not None:
            return [list(row) for row in rows]
    return resolve_positions(
        positions, traces, encoding["attention_mask"], is_original=is_original
    )
