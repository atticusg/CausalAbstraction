"""``fit_pca@1`` against a hand-computed oracle.

The fixture is a cross in the xy-plane of R^3, centred at the origin:
points (±2, 0, 0) and (0, ±1, 0). With n = 4 and ddof = 1 the variances are
x: (4 + 4)/3 = 8/3, y: (1 + 1)/3 = 2/3, z: 0 — so the ratios are exactly
0.8 and 0.2, and the components are the x and y axes.
"""

from __future__ import annotations

import pytest
import torch

from causalab.transform.ops.fit_pca import fit_pca
from causalab.transform.schema import TransformError

pytestmark = pytest.mark.numerical_unit

CROSS = torch.tensor(
    [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
)


def test_components_and_spectrum_match_the_oracle() -> None:
    out = fit_pca(inputs={"acts": CROSS}, params={"k": 2})
    weight, spectrum = out["weight"], out["spectrum"]
    assert weight.shape == (3, 2)  # (d, k)
    assert torch.allclose(weight[:, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(weight[:, 1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert spectrum["pc"].tolist() == [0, 1]
    assert spectrum["explained_variance"].tolist() == pytest.approx([8 / 3, 2 / 3])
    assert spectrum["explained_variance_ratio"].tolist() == pytest.approx([0.8, 0.2])


def test_leading_dimensions_are_flattened() -> None:
    """[examples, positions, d] and [rows, d] must mean the same fit."""
    flat = fit_pca(inputs={"acts": CROSS}, params={"k": 2})
    nested = fit_pca(inputs={"acts": CROSS.reshape(2, 2, 3)}, params={"k": 2})
    assert torch.equal(flat["weight"], nested["weight"])


def test_the_sign_convention_is_pinned() -> None:
    """SVD fixes a component only up to a sign. Negating the input must not
    negate the basis, or two equally 'correct' fits would disagree."""
    positive = fit_pca(inputs={"acts": CROSS}, params={"k": 2})["weight"]
    negated = fit_pca(inputs={"acts": -CROSS}, params={"k": 2})["weight"]
    assert torch.equal(positive, negated)
    # and the pinned sign is "largest-magnitude entry positive"
    for column in range(positive.shape[1]):
        component = positive[:, column]
        assert component[component.abs().argmax()] > 0


def test_the_fit_is_deterministic() -> None:
    """Bit-identical, twice in one process — the registry's admission
    criterion, and why this op declares no seed."""
    first = fit_pca(inputs={"acts": CROSS}, params={"k": 2})
    second = fit_pca(inputs={"acts": CROSS}, params={"k": 2})
    assert torch.equal(first["weight"], second["weight"])
    assert first["spectrum"].equals(second["spectrum"])


def test_k_beyond_the_available_rank_is_refused() -> None:
    with pytest.raises(TransformError) as err:
        fit_pca(inputs={"acts": CROSS}, params={"k": 5})
    assert "exceeds the rank available" in str(err.value)


def test_a_one_dimensional_input_is_refused() -> None:
    with pytest.raises(TransformError):
        fit_pca(inputs={"acts": torch.zeros(4)}, params={"k": 1})


def test_a_single_row_has_no_variance_to_fit() -> None:
    with pytest.raises(TransformError):
        fit_pca(inputs={"acts": torch.zeros(1, 3)}, params={"k": 1})


def test_a_degenerate_input_reports_zero_ratios_rather_than_nan() -> None:
    """All-identical rows have zero total variance; 0/0 would be NaN, which
    would then travel silently into a plot."""
    out = fit_pca(inputs={"acts": torch.ones(4, 3)}, params={"k": 2})
    assert out["spectrum"]["explained_variance_ratio"].tolist() == [0.0, 0.0]
