"""Direct tests for the ``attention_mask`` kwarg on ``index_component``.

Exercises both axis-layout branches (single-axis via ``ResidualStream``, multi-
axis via ``AttentionHead``) without mocks, so the polymorphism in
``_apply_padding_shift`` is what's under test.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.LM_units import AttentionHead, ResidualStream
from causalab.neural.units import ComponentIndexer

pytestmark = pytest.mark.unit


def _const_indexer(positions: list[int]) -> ComponentIndexer:
    """ComponentIndexer that returns ``positions`` regardless of input."""
    return ComponentIndexer(lambda _x: positions, id=f"const_{positions}")


def test_explicit_none_mask_is_noop_residual_stream():
    """Explicit ``attention_mask=None`` is the documented opt-out path."""
    unit = ResidualStream(layer=0, token_indices=_const_indexer([0, 1]))
    result = unit.index_component(["a", "b"], batch=True, attention_mask=None)
    assert result == [[0, 1], [0, 1]]


def test_explicit_none_mask_is_noop_attention_head():
    """Explicit ``attention_mask=None`` is the documented opt-out path."""
    unit = AttentionHead(layer=0, head=7, token_indices=_const_indexer([0, 1]))
    result = unit.index_component(["a", "b"], batch=True, attention_mask=None)
    assert result == [[[7], [7]], [[0, 1], [0, 1]]]


def test_omitted_mask_raises_for_batched_multi_residual_stream():
    """Omitting ``attention_mask`` on batched multi-input must fail loudly."""
    unit = ResidualStream(layer=0, token_indices=_const_indexer([0, 1]))
    with pytest.raises(ValueError, match="requires `attention_mask`"):
        unit.index_component(["a", "b"], batch=True)


def test_omitted_mask_raises_for_batched_multi_attention_head():
    """The check fires on subclasses too — `AttentionHead` overrides
    ``index_component``."""
    unit = AttentionHead(layer=0, head=7, token_indices=_const_indexer([0, 1]))
    with pytest.raises(ValueError, match="requires `attention_mask`"):
        unit.index_component(["a", "b", "c"], batch=True)


def test_omitted_mask_ok_for_batched_singleton():
    """Batch of one has no padding; the kwarg is genuinely optional there."""
    unit = ResidualStream(layer=0, token_indices=_const_indexer([0, 1]))
    result = unit.index_component(["a"], batch=True)
    assert result == [[0, 1]]


def test_omitted_mask_ok_for_unbatched():
    """``batch=False`` is the single-example path; no padding to correct."""
    unit = ResidualStream(layer=0, token_indices=_const_indexer([0, 1]))
    result = unit.index_component("a")
    assert result == [0, 1]


def test_right_padding_fastpath_residual_stream():
    """No left-padding ⇒ no shift; in-bounds positions return unchanged. The
    bounds check still runs (see test_oob_raises_without_left_padding_*)."""
    unit = ResidualStream(layer=0, token_indices=_const_indexer([0, 1]))
    mask = torch.ones((2, 3), dtype=torch.long)  # no padding → argmax is 0
    result = unit.index_component(["a", "b"], batch=True, attention_mask=mask)
    assert result == [[0, 1], [0, 1]]


def test_left_padding_residual_stream_batch():
    unit = ResidualStream(layer=0, token_indices=_const_indexer([0, 1]))
    # row 0: 2 pad tokens, row 1: 1 pad token. padded_len=5.
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.long)
    result = unit.index_component(["a", "b"], batch=True, attention_mask=mask)
    assert result == [[2, 3], [1, 2]]


def test_left_padding_attention_head_position_axis_only():
    unit = AttentionHead(layer=0, head=7, token_indices=_const_indexer([0, 1]))
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.long)
    head_axis, position_axis = unit.index_component(
        ["a", "b"], batch=True, attention_mask=mask
    )
    # head axis untouched
    assert head_axis == [[7], [7]]
    # position axis shifted by per-row offset
    assert position_axis == [[2, 3], [1, 2]]


def test_oob_raises_value_error():
    # padded_len=3, row 0 has shift=2; position 2 → 4 which is OOB.
    unit = ResidualStream(layer=0, token_indices=_const_indexer([2]))
    mask = torch.tensor([[0, 0, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="out of bounds for padded length 3"):
        unit.index_component(["a"], batch=True, attention_mask=mask)


def test_oob_raises_without_left_padding_residual_stream():
    """An index >= sequence length must raise even with no left-padding to
    shift — the #176 single_pair_trace case (batch_size=1, no padding). The
    old fast-path returned such indices unchecked and they tripped the CUDA
    gather assert downstream."""
    unit = ResidualStream(layer=0, token_indices=_const_indexer([5]))
    mask = torch.ones((1, 3), dtype=torch.long)  # padded_len=3, no shift
    with pytest.raises(ValueError, match="out of bounds for padded length 3"):
        unit.index_component(["a"], batch=True, attention_mask=mask)


def test_oob_position_raises_attention_head_head_axis_not_flagged():
    """For AttentionHead the bounds check covers the position axis only — a
    head index larger than the sequence length must NOT be flagged, while an
    out-of-bounds position must raise."""
    mask = torch.ones((1, 3), dtype=torch.long)  # padded_len=3, no shift

    # head=7 (> padded_len) with a valid position: head axis untouched, no raise.
    ok = AttentionHead(layer=0, head=7, token_indices=_const_indexer([1]))
    head_axis, position_axis = ok.index_component(
        ["a"], batch=True, attention_mask=mask
    )
    assert head_axis == [[7]]  # head 7 not bounds-checked against padded_len 3
    assert position_axis == [[1]]

    # Same head, out-of-bounds position: raises.
    bad = AttentionHead(layer=0, head=7, token_indices=_const_indexer([5]))
    with pytest.raises(ValueError, match="out of bounds for padded length 3"):
        bad.index_component(["a"], batch=True, attention_mask=mask)


def test_single_example_path_residual_stream():
    unit = ResidualStream(layer=0, token_indices=_const_indexer([0, 1]))
    mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)
    result = unit.index_component("a", batch=False, attention_mask=mask)
    assert result == [2, 3]


def test_single_example_path_attention_head():
    unit = AttentionHead(layer=0, head=7, token_indices=_const_indexer([0, 1]))
    mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)
    # AttentionHead wraps positions as list-of-rows even for batch=False:
    # the unmasked shape is [[[head]], [[positions]]].
    head_axis, position_axis = unit.index_component(
        "a", batch=False, attention_mask=mask
    )
    assert head_axis == [[7]]
    assert position_axis == [[2, 3]]
