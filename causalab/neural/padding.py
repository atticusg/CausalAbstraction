"""Padding-aware adjustment of per-example token indices.

Token positions in causalab are computed per example on the *unpadded*
tokenization (every indexer in ``token_positions.py`` tokenizes a single
example, where padding is a no-op). Batched execution then pads to the
batch's longest sequence — on the left by default — which shifts every real
token of a shorter example right by ``(padded length - true length)``.
Passing unpadded indices to a left-padded batch therefore intervenes on the
wrong positions (or on padding) for every example shorter than the batch
maximum.

These helpers convert unpadded per-example indices to padded batch
coordinates using the attention mask. They are applied at the single point
where indices meet a padded batch (``collect.py`` /
``interchange_mode.py``), so indexers stay pad-agnostic. Right padding needs
no shift; the mask-based computation handles both sides.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

__all__ = ["shift_indices_for_padding", "shift_unit_indices_for_padding"]


def _per_example_shifts(attention_mask: Tensor) -> list[int]:
    mask = attention_mask.long()
    total = mask.shape[1]
    true_len = mask.sum(dim=1)
    first_real = mask.argmax(
        dim=1
    )  # (total - true_len) under left padding, 0 under right
    # first_real is where unpadded index 0 was placed; that IS the shift.
    del total, true_len
    return [int(s) for s in first_real]


def shift_indices_for_padding(
    indices: list[list[int]], attention_mask: Tensor
) -> list[list[int]]:
    """Shift per-example position lists into padded batch coordinates.

    ``indices`` has one list of token positions per example (the ``"pos"``
    unit layout). Positions are unpadded per-example coordinates.
    """
    shifts = _per_example_shifts(attention_mask)
    if len(indices) != len(shifts):
        raise ValueError(
            f"{len(indices)} index lists for a batch of {len(shifts)} examples"
        )
    out: list[list[int]] = []
    seq_len = attention_mask.shape[1]
    true_lens = attention_mask.long().sum(dim=1).tolist()
    for row, shift, true_len in zip(indices, shifts, true_lens):
        # A negative index is end-relative in the example's TRUE (unpadded)
        # sequence — the documented ``TokenPosition(lambda x: [-1], ...)``
        # API — so it resolves against the real last token, not the padded
        # tensor edge.
        shifted = []
        for orig in row:
            q = orig if orig >= 0 else true_len + orig
            if not 0 <= q < true_len:
                raise IndexError(
                    f"unpadded index {orig} is out of range for its example "
                    f"(true length {true_len})"
                )
            p = q + shift
            if not 0 <= p < seq_len:
                raise IndexError(
                    f"unpadded index {orig} resolves to padded position {p}, "
                    f"outside the padded sequence (length {seq_len})"
                )
            shifted.append(p)
        out.append(shifted)
    return out


def shift_unit_indices_for_padding(
    unit: Any, indices: Any, attention_mask: Tensor
) -> Any:
    """Shift one model unit's batched indices into padded batch coordinates.

    Understands the two index layouts units produce: ``"pos"`` (one position
    list per example) and ``"h.pos"`` (a ``[heads, positions]`` pair whose
    second element is one position list per example). Head indices are
    positionless and pass through untouched.
    """
    unit_kind = getattr(unit, "unit", None)
    if unit_kind == "pos":
        return shift_indices_for_padding(indices, attention_mask)
    if unit_kind == "h.pos":
        heads, positions = indices
        return [heads, shift_indices_for_padding(positions, attention_mask)]
    # Unknown or missing unit layout (e.g. a mock in tests, or a custom unit
    # class): leave indices untouched — the pre-fix behavior — rather than
    # guess at the structure.
    return indices
