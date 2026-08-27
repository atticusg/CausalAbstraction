"""Tap layouts: reconciling a module's native tensor shape with the executor's.

The executor works in one shape throughout — ``(batch, position, feature)``.
``PointExecutor._gather`` indexes ``tensor[rows, idx]`` (dim 0 batch, dim 1
position), ``_finalize_read`` slices features on dim ``-1``, and
``_address_writer`` mutates ``tensor[rows, idx][..., fslice]`` in place. That is
the *contract*.

Most taps already satisfy it, which is why the contract could stay implicit.
Some do not, and the shapes are architecture facts rather than bugs:

===================  ==========================  ====================================
layout               native shape                where it shows up
===================  ==========================  ====================================
``"bsd"``            ``(batch, seq, feature)``    the contract; every tap so far
``"flat_td"``        ``(batch * seq, feature)``   MoE taps — ``Qwen3_5MoeSparseMoeBlock``
                                                  reshapes to ``(-1, hidden)`` before
                                                  the router, so the router, experts
                                                  and shared-expert taps are flat
``"bds"``            ``(batch, feature, seq)``    channels-first — the Gated DeltaNet
                                                  ``conv1d`` output
``"bs"``             ``(batch, seq)``             no feature axis at all — ``input_ids``,
                                                  one integer token id per position
===================  ==========================  ====================================

A tap declares its layout on :class:`~causalab.neural.pytorch_hooks.sites.ResolvedSite`
and this module converts in both directions: :func:`to_contract` on the way out of
a hook, :func:`from_contract` on the way back in for writes. Reads and writes
therefore share one description of the shape instead of each growing a special
case, which is the point — the general feature-shape descriptor (a typed answer
for ``attention_probs``, whose feature axis *is* a position axis) is later work,
and it should extend this rather than unpick per-component branches.

Separately, a module may return a **tuple** and the interesting tensor may not be
element 0: ``Qwen3_5MoeTopKRouter.forward`` returns
``(router_logits, router_scores, router_indices)``. A tap declares which element
it means with ``tuple_index``; the default keeps the historical behaviour of
taking element 0 of any tuple.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

import torch

#: The layouts a tap may declare. ``"bsd"`` is the executor's contract.
#:
#: ``"bs"`` is the degenerate case and is deliberately *here* rather than
#: special-cased in the executor: a tensor with no feature axis still has a
#: native shape that differs from the contract, which is exactly what this
#: module exists to reconcile. It is not the typed feature-shape descriptor that
#: ``attention_probs`` needs (whose feature axis *is* a position axis) — there is
#: no feature-axis ambiguity to describe, only an absent axis to add.
Layout = Literal["bsd", "flat_td", "bds", "bs"]

#: Every valid layout string, for validation and error messages.
LAYOUTS: tuple[str, ...] = get_args(Layout)


class LayoutError(ValueError):
    """A tensor does not have the shape its tap's layout claims.

    Deliberately *not* a :class:`~causalab.protocol.errors.ProtocolError`, and
    so it carries no ``P``/``V`` code: ``layout`` is a field the site table
    sets on :class:`~causalab.neural.pytorch_hooks.sites.ResolvedSite`, never
    something a document author writes, so this can only fire on a mismatch
    between our table and a model's real module — an internal invariant, and
    the protocol codes exist to name rules a *document* broke.

    What would change that: making the layout authorable (a per-family
    override in a document), at which point a bad entry becomes an author
    error and this should either derive from ``ProtocolError`` or be wrapped
    at the executor boundary with a code.
    """


def to_contract(
    tensor: torch.Tensor, layout: Layout, *, batch_size: int
) -> torch.Tensor:
    """Reshape a tap's native tensor into ``(batch, position, feature)``.

    Args:
        tensor: the tensor as the module produced or consumed it.
        layout: the tap's declared native layout.
        batch_size: rows in the encoded batch. Needed to un-flatten
            ``"flat_td"``, where the position count is only recoverable as
            ``tensor.shape[0] // batch_size``.

    Returns:
        A ``(batch, position, feature)`` view where possible, so that an
        in-place write through the result reaches the original storage.

        📐 **That aliasing is not load-bearing** — it is an artefact of
        ``view``/``transpose``, not a guarantee this function makes, and no
        behaviour depends on it. Writes are correct because the hook passes
        :func:`from_contract`'s result *back* to the model and
        ``_address_writer`` mutates through the returned chain, so a layout
        that had to ``copy`` (an incompatible stride, say) would be equally
        correct and simply slower. Stated because the reverse is the tempting
        mistake: nothing in the suite fails if aliasing is lost, so a later
        change that relies on it would be silently unpinned.

    Raises:
        LayoutError: the tensor's rank or leading dimension contradicts the
            declared layout. Silently reinterpreting a mismatched shape is the
            failure mode this exists to prevent — a wrong tap that still
            produces plausible numbers.
    """
    if layout == "bsd":
        return tensor
    if layout == "flat_td":
        if tensor.dim() != 2:
            raise LayoutError(
                f"layout 'flat_td' expects a 2-D (batch*seq, feature) tensor, "
                f"got shape {tuple(tensor.shape)}"
            )
        flat, feature = tensor.shape
        if batch_size <= 0 or flat % batch_size:
            raise LayoutError(
                f"layout 'flat_td' cannot split {flat} rows into batch_size "
                f"{batch_size}: the tap is flattened over (batch, seq), so the "
                "row count must be a multiple of the batch size"
            )
        return tensor.view(batch_size, flat // batch_size, feature)
    if layout == "bds":
        if tensor.dim() != 3:
            raise LayoutError(
                f"layout 'bds' expects a 3-D (batch, feature, seq) tensor, got "
                f"shape {tuple(tensor.shape)}"
            )
        return tensor.transpose(1, 2)
    if layout == "bs":
        if tensor.dim() != 2:
            raise LayoutError(
                f"layout 'bs' expects a 2-D (batch, seq) tensor, got shape "
                f"{tuple(tensor.shape)}"
            )
        # unsqueeze, not reshape: a view, so the aliasing the write path relies
        # on survives (even though the only 'bs' tap today is read-only)
        return tensor.unsqueeze(-1)
    raise LayoutError(f"unknown tap layout {layout!r}; expected one of {LAYOUTS}")


def from_contract(
    tensor: torch.Tensor, layout: Layout, *, batch_size: int
) -> torch.Tensor:
    """Invert :func:`to_contract`, returning the module's native shape.

    ``from_contract(to_contract(x, l), l)`` round-trips to ``x``'s shape for
    every layout. Used on the write path, where the model must receive back the
    shape it was going to produce.
    """
    if layout == "bsd":
        return tensor
    if layout == "flat_td":
        if tensor.dim() != 3:
            raise LayoutError(
                f"layout 'flat_td' expects a 3-D contract tensor to flatten, got "
                f"shape {tuple(tensor.shape)}"
            )
        return tensor.reshape(-1, tensor.shape[-1])
    if layout == "bds":
        if tensor.dim() != 3:
            raise LayoutError(
                f"layout 'bds' expects a 3-D contract tensor, got shape "
                f"{tuple(tensor.shape)}"
            )
        return tensor.transpose(1, 2)
    if layout == "bs":
        if tensor.dim() != 3 or tensor.shape[-1] != 1:
            raise LayoutError(
                f"layout 'bs' expects a 3-D contract tensor of width 1 to drop "
                f"back to (batch, seq), got shape {tuple(tensor.shape)}"
            )
        return tensor.squeeze(-1)
    raise LayoutError(f"unknown tap layout {layout!r}; expected one of {LAYOUTS}")


def tap_tensor(payload: Any, tuple_index: int | None) -> torch.Tensor:
    """The tensor a tap means, out of a module's output or input payload.

    ``tuple_index=None`` keeps the historical rule — element 0 of a tuple,
    otherwise the payload itself. An explicit index addresses a specific element
    and fails loudly if the payload is not a tuple, because a tap that asked for
    element 2 and silently got the whole tensor would read the wrong thing.
    """
    if tuple_index is None:
        return payload[0] if isinstance(payload, tuple) else payload
    if not isinstance(payload, tuple):
        raise LayoutError(
            f"tap declares tuple_index={tuple_index} but the module payload is "
            f"{type(payload).__name__}, not a tuple"
        )
    if not -len(payload) <= tuple_index < len(payload):
        raise LayoutError(
            f"tap declares tuple_index={tuple_index} but the module returned a "
            f"{len(payload)}-tuple"
        )
    return payload[tuple_index]


def rebuild_payload(payload: Any, tuple_index: int | None, value: torch.Tensor) -> Any:
    """Put ``value`` back where :func:`tap_tensor` took it from.

    Preserves the rest of a tuple payload so a write does not drop the cache or
    the attention weights a module also returned.
    """
    if not isinstance(payload, tuple):
        return value
    index = 0 if tuple_index is None else tuple_index
    items = list(payload)
    items[index] = value
    return tuple(items)
