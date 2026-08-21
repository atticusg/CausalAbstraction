"""PositionFrame resolution (spec §2.3, §8): index/variable/span, scope and
relative_to, left-pad frame math, and the out-of-bounds refusal contract
(a stale position must fail legibly, never address the wrong token)."""

from __future__ import annotations

import pytest

from causalab.neural.pytorch_hooks.encoding import encode, resolve_position
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import PositionSpec

from tests.neural.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def tokenizer():
    from causalab.neural.pytorch_hooks.loading import load_model

    return load_model(TINY_LLAMA).tokenizer


def test_negative_index_is_the_last_real_token(tokenizer):
    batch = encode(tokenizer, ["one two three", "a much longer sentence right here"])
    for row in range(2):
        (pos,) = resolve_position(PositionSpec(index=-1), batch, row)
        assert pos == batch.padded_len - 1  # left padding right-aligns content


def test_nonnegative_index_counts_from_content_start(tokenizer):
    batch = encode(tokenizer, ["one two three", "a much longer sentence right here"])
    # row 0 is padded on the left; its index 0 is its first real token
    (pos,) = resolve_position(PositionSpec(index=0), batch, 0)
    assert pos == batch.content_start(0)
    assert batch.attention_mask[0, pos] == 1
    if pos > 0:
        assert batch.attention_mask[0, pos - 1] == 0


def test_variable_window_covers_the_value(tokenizer):
    text = "If today is Thursday, tomorrow is"
    batch = encode(tokenizer, [text])
    run = resolve_position(
        PositionSpec(variable="subject"),
        batch,
        0,
        dataset_row={"input": text, "subject": "Thursday"},
        field="input",
    )
    decoded = tokenizer.decode(batch.input_ids[0, run])
    assert "Thursday" in decoded


def test_ambiguous_variable_occurrence_refuses(tokenizer):
    text = "day after day after day"
    batch = encode(tokenizer, [text])
    with pytest.raises(ProtocolError) as err:
        resolve_position(
            PositionSpec(variable="word"),
            batch,
            0,
            dataset_row={"input": text, "word": "day"},
            field="input",
        )
    assert "occurs" in str(err.value)


def test_scope_indexes_inside_the_variable_window(tokenizer):
    text = "If today is Thursday, tomorrow is"
    batch = encode(tokenizer, [text])
    row = {"input": text, "subject": "Thursday"}
    window = resolve_position(
        PositionSpec(variable="subject"), batch, 0, dataset_row=row, field="input"
    )
    (first,) = resolve_position(
        PositionSpec(index=0, scope="subject"), batch, 0, dataset_row=row, field="input"
    )
    (last,) = resolve_position(
        PositionSpec(index=-1, scope="subject"),
        batch,
        0,
        dataset_row=row,
        field="input",
    )
    assert first == window[0] and last == window[-1]


def test_relative_to_offsets_from_the_window(tokenizer):
    text = "If today is Thursday, tomorrow is"
    batch = encode(tokenizer, [text])
    row = {"input": text, "subject": "Thursday"}
    window = resolve_position(
        PositionSpec(variable="subject"), batch, 0, dataset_row=row, field="input"
    )
    (after,) = resolve_position(
        PositionSpec(index=1, relative_to="subject"),
        batch,
        0,
        dataset_row=row,
        field="input",
    )
    (before,) = resolve_position(
        PositionSpec(index=-1, relative_to="subject"),
        batch,
        0,
        dataset_row=row,
        field="input",
    )
    assert after == window[-1] + 1 and before == window[0] - 1


def test_span_is_a_content_frame_window(tokenizer):
    batch = encode(tokenizer, ["one two three", "a much longer sentence right here"])
    window = resolve_position(PositionSpec(span=(0, 2)), batch, 0)
    start = batch.content_start(0)
    assert window == [start, start + 1]


def test_all_is_every_content_token(tokenizer):
    """``{"all": true}`` is the row's real tokens: contiguous from the row's
    content start to the padded end, nothing under the left pad."""
    batch = encode(tokenizer, ["one two three", "a much longer sentence right here"])
    for row in range(2):
        positions = resolve_position(PositionSpec(all=True), batch, row)
        start = batch.content_start(row)
        assert positions == list(range(start, batch.padded_len))
        assert all(batch.attention_mask[row, p] == 1 for p in positions)
    # the short row is genuinely shorter — the ragged case reads must carry
    assert len(resolve_position(PositionSpec(all=True), batch, 0)) < len(
        resolve_position(PositionSpec(all=True), batch, 1)
    )


def test_all_needs_no_dataset_row(tokenizer):
    """Unlike a variable window, an all spec is frame-only — it resolves
    without the dataset row, which is what makes it usable in any document."""
    batch = encode(tokenizer, ["one two three"])
    positions = resolve_position(PositionSpec(all=True), batch, 0)
    assert positions == list(range(batch.content_start(0), batch.padded_len))


def test_all_covers_the_index_forms(tokenizer):
    """The endpoints agree with the index spellings they generalize."""
    batch = encode(tokenizer, ["one two three", "a much longer sentence right here"])
    for row in range(2):
        positions = resolve_position(PositionSpec(all=True), batch, row)
        (first,) = resolve_position(PositionSpec(index=0), batch, row)
        (last,) = resolve_position(PositionSpec(index=-1), batch, row)
        assert positions[0] == first and positions[-1] == last


def test_out_of_bounds_refuses(tokenizer):
    batch = encode(tokenizer, ["one two three"])
    with pytest.raises(ProtocolError) as err:
        resolve_position(PositionSpec(index=500), batch, 0)
    assert "out of bounds" in str(err.value)
