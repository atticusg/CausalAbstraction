"""Unit tests for fixed-rotation orientation resolution (issue #255).

``subspace method: fixed`` used to read ``k = rotation.shape[1]`` and reject a
``(k, d_model)`` artifact even though ``characterize_subspace``'s adaptive loader
auto-transposes the *same* file — so one artifact behaved two ways depending on
the entry point. :func:`orient_rotation` resolves orientation deterministically
against the model's known ``d_model``; these tests pin that contract.

Pure-tensor ``unit`` tier: no model, no device — just shape/value assertions.
"""

from __future__ import annotations

import pytest
import torch

from causalab.analyses.subspace import orient_rotation

pytestmark = pytest.mark.unit

D_MODEL = 16
K = 4


def test_d_model_k_returned_unchanged() -> None:
    """A ``(d_model, k)`` rotation is already canonical → returned as-is."""
    rot = torch.randn(D_MODEL, K)
    out = orient_rotation(rot, d_model=D_MODEL)
    assert out.shape == (D_MODEL, K)
    assert out.shape[1] == K  # resolves to k features
    assert torch.equal(out, rot)


def test_k_d_model_is_transposed() -> None:
    """A ``(k, d_model)`` rotation is transposed to ``(d_model, k)``."""
    rot = torch.randn(K, D_MODEL)
    out = orient_rotation(rot, d_model=D_MODEL)
    assert out.shape == (D_MODEL, K)
    assert out.shape[1] == K  # resolves to the same k features
    assert torch.equal(out, rot.t())


def test_square_requires_k_features_hint() -> None:
    """A square rotation is orientation-ambiguous; a hint is required."""
    rot = torch.randn(D_MODEL, D_MODEL)
    with pytest.raises(ValueError, match=r"ambiguous"):
        orient_rotation(rot, d_model=D_MODEL)


def test_square_with_matching_hint_unchanged() -> None:
    """Square + ``k_features_hint == d_model`` resolves and is returned as-is."""
    rot = torch.randn(D_MODEL, D_MODEL)
    out = orient_rotation(rot, d_model=D_MODEL, k_features_hint=D_MODEL)
    assert out.shape == (D_MODEL, D_MODEL)
    assert out.shape[1] == D_MODEL
    assert torch.equal(out, rot)


def test_square_with_mismatched_hint_raises() -> None:
    """A hint that doesn't match the square axis is rejected."""
    rot = torch.randn(D_MODEL, D_MODEL)
    with pytest.raises(ValueError, match=r"does not match"):
        orient_rotation(rot, d_model=D_MODEL, k_features_hint=D_MODEL - 1)


def test_square_not_matching_d_model_raises() -> None:
    """A square rotation whose dim isn't d_model can't project."""
    rot = torch.randn(K, K)
    with pytest.raises(ValueError, match=r"neither axis"):
        orient_rotation(rot, d_model=D_MODEL, k_features_hint=K)


def test_neither_axis_matches_raises_clear_message() -> None:
    """Neither dim matching ``d_model`` raises an orientation-specific error."""
    rot = torch.randn(7, 5)
    with pytest.raises(ValueError, match=r"neither axis"):
        orient_rotation(rot, d_model=D_MODEL)


@pytest.mark.parametrize(
    "shape",
    [(D_MODEL,), (2, D_MODEL, K)],
)
def test_non_2d_raises(shape: tuple[int, ...]) -> None:
    """1-D or >2-D rotations are rejected with a shape message."""
    with pytest.raises(ValueError, match=r"2D"):
        orient_rotation(torch.randn(*shape), d_model=D_MODEL)


def test_idempotent() -> None:
    """Orienting an already-oriented rotation is a no-op (safe to call twice)."""
    rot = torch.randn(K, D_MODEL)
    once = orient_rotation(rot, d_model=D_MODEL)
    twice = orient_rotation(once, d_model=D_MODEL)
    assert torch.equal(once, twice)
    assert twice.shape == (D_MODEL, K)
