"""Tap-layout conversions: the shape contract, made explicit and reversible.

These are pure-function tests on small tensors — no model, no hooks. What they
pin is that a tap can declare a native shape other than
``(batch, position, feature)`` without any component-specific branching, and
that a declared shape which does not match reality fails loudly instead of
being silently reinterpreted.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.pytorch_hooks.layout import (
    LAYOUTS,
    LayoutError,
    from_contract,
    rebuild_payload,
    tap_tensor,
    to_contract,
)

pytestmark = pytest.mark.unit

BATCH, SEQ, FEATURE = 2, 3, 5

#: The layouts that actually convert something. ``"native"`` is excluded on
#: purpose: it is the *absence* of a description (``attention_probs``, whose
#: feature axis is a position axis), so there is no contract shape to assert it
#: reaches. It gets its own test below instead of a vacuous parametrized one.
CONVERTING_LAYOUTS = tuple(layout for layout in LAYOUTS if layout != "native")


def _feature(layout: str) -> int:
    """The feature width the contract has for this layout.

    ``"bs"`` has no native feature axis, so its contract width is 1 by
    definition rather than by choice — the generic assertions below are
    parametrized on this instead of on the module-level ``FEATURE``."""
    return 1 if layout == "bs" else FEATURE


def _native(layout: str) -> torch.Tensor:
    if layout == "bsd":
        return torch.arange(BATCH * SEQ * FEATURE, dtype=torch.float32).reshape(
            BATCH, SEQ, FEATURE
        )
    if layout == "flat_td":
        return torch.arange(BATCH * SEQ * FEATURE, dtype=torch.float32).reshape(
            BATCH * SEQ, FEATURE
        )
    if layout == "bds":
        return torch.arange(BATCH * FEATURE * SEQ, dtype=torch.float32).reshape(
            BATCH, FEATURE, SEQ
        )
    if layout == "bs":
        return torch.arange(BATCH * SEQ, dtype=torch.float32).reshape(BATCH, SEQ)
    raise AssertionError(layout)


@pytest.mark.parametrize("layout", CONVERTING_LAYOUTS)
def test_to_contract_yields_the_executor_shape(layout: str) -> None:
    """Whatever the tap's native shape, the executor sees (batch, pos, feature)."""
    got = to_contract(_native(layout), layout, batch_size=BATCH)
    assert got.shape == (BATCH, SEQ, _feature(layout))


@pytest.mark.parametrize("layout", CONVERTING_LAYOUTS)
def test_round_trip_restores_the_native_shape(layout: str) -> None:
    """The write path must hand the model back the shape it expected."""
    native = _native(layout)
    contract = to_contract(native, layout, batch_size=BATCH)
    back = from_contract(contract, layout, batch_size=BATCH)
    assert back.shape == native.shape
    assert torch.equal(back, native)


@pytest.mark.parametrize("layout", CONVERTING_LAYOUTS)
def test_an_in_place_edit_through_the_contract_reaches_the_native_tensor(
    layout: str,
) -> None:
    """This is the property the write path depends on.

    ``_address_writer`` mutates the contract-shaped tensor in place, then the
    hook converts back. If the conversion copied instead of viewing, and the
    copy were discarded, writes would silently no-op — so assert the edit
    survives the round trip.
    """
    native = _native(layout)
    contract = to_contract(native, layout, batch_size=BATCH)
    cell = (0, 1, _feature(layout) - 1)
    contract[cell] = -99.0
    back = from_contract(contract, layout, batch_size=BATCH)
    assert to_contract(back, layout, batch_size=BATCH)[cell] == -99.0


def test_native_is_the_identity_and_claims_nothing() -> None:
    """``"native"`` passes any shape through untouched, in both directions.

    It exists so a tap whose shape this module cannot honestly describe cannot
    be mistaken for the contract by defaulting to ``"bsd"`` — where the
    conversions are also the identity, but the *claim* would be false. There is
    no conversion case here for the typed feature-shape descriptor to unpick.
    """
    probs = torch.zeros(2, 4, 3, 3)  # (batch, heads, query, key)
    assert to_contract(probs, "native", batch_size=2) is probs
    assert from_contract(probs, "native", batch_size=2) is probs
    # and, unlike every other layout, it does NOT promise a 3-D contract shape
    assert "native" not in CONVERTING_LAYOUTS


def test_bsd_is_the_identity() -> None:
    """The default layout must not copy or reshape anything."""
    native = _native("bsd")
    assert to_contract(native, "bsd", batch_size=BATCH) is native
    assert from_contract(native, "bsd", batch_size=BATCH) is native


def test_flat_td_positions_are_recovered_in_order() -> None:
    """Row r of the flat tap is (batch r // seq, position r % seq)."""
    native = _native("flat_td")
    contract = to_contract(native, "flat_td", batch_size=BATCH)
    for b in range(BATCH):
        for pos in range(SEQ):
            assert torch.equal(contract[b, pos], native[b * SEQ + pos])


def test_bds_moves_the_feature_axis_last() -> None:
    native = _native("bds")
    contract = to_contract(native, "bds", batch_size=BATCH)
    assert torch.equal(contract[:, :, 0], native[:, 0, :])


# ------------------------------------------------------------------ #
# a declared layout that contradicts the tensor must fail loudly
# ------------------------------------------------------------------ #


def test_flat_td_rejects_a_batch_size_that_does_not_divide() -> None:
    with pytest.raises(LayoutError, match="multiple of the batch size"):
        to_contract(_native("flat_td"), "flat_td", batch_size=4)


def test_flat_td_rejects_a_non_2d_tensor() -> None:
    with pytest.raises(LayoutError, match="expects a 2-D"):
        to_contract(_native("bsd"), "flat_td", batch_size=BATCH)


def test_bds_rejects_a_non_3d_tensor() -> None:
    with pytest.raises(LayoutError, match="expects a 3-D"):
        to_contract(_native("flat_td"), "bds", batch_size=BATCH)


def test_unknown_layout_is_refused() -> None:
    with pytest.raises(LayoutError, match="unknown tap layout"):
        to_contract(_native("bsd"), "nonsense", batch_size=BATCH)  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# tuple payloads
# ------------------------------------------------------------------ #


def test_default_tuple_rule_is_unchanged() -> None:
    """No tuple_index keeps the historical behaviour exactly."""
    t0, t1 = torch.zeros(2), torch.ones(2)
    assert tap_tensor((t0, t1), None) is t0
    assert tap_tensor(t0, None) is t0


def test_tuple_index_addresses_a_later_element() -> None:
    """A router returning (logits, scores, indices) needs element 1 and 2."""
    logits, scores, indices = torch.zeros(2), torch.ones(2), torch.full((2,), 7.0)
    payload = (logits, scores, indices)
    assert tap_tensor(payload, 0) is logits
    assert tap_tensor(payload, 1) is scores
    assert tap_tensor(payload, 2) is indices
    assert tap_tensor(payload, -1) is indices


def test_tuple_index_on_a_bare_tensor_is_refused() -> None:
    """Silently returning the whole tensor would read the wrong thing."""
    with pytest.raises(LayoutError, match="not a tuple"):
        tap_tensor(torch.zeros(2), 1)


def test_tuple_index_out_of_range_is_refused() -> None:
    with pytest.raises(LayoutError, match="2-tuple"):
        tap_tensor((torch.zeros(2), torch.ones(2)), 5)


def test_rebuild_preserves_the_rest_of_the_payload() -> None:
    """A write must not drop the cache or weights a module also returned."""
    hidden, cache = torch.zeros(2), object()
    out = rebuild_payload((hidden, cache), None, torch.ones(2))
    assert isinstance(out, tuple) and len(out) == 2
    assert torch.equal(out[0], torch.ones(2))
    assert out[1] is cache


def test_rebuild_targets_the_declared_element() -> None:
    a, b, c = torch.zeros(2), torch.ones(2), torch.full((2,), 3.0)
    out = rebuild_payload((a, b, c), 1, torch.full((2,), -1.0))
    assert out[0] is a and out[2] is c
    assert torch.equal(out[1], torch.full((2,), -1.0))


def test_rebuild_on_a_bare_payload_returns_the_value() -> None:
    assert torch.equal(
        rebuild_payload(torch.zeros(2), None, torch.ones(2)), torch.ones(2)
    )
