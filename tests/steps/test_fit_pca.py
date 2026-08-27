"""``causalab:fit_pca`` against a hand-computed oracle.

Ported from ``fit_pca@1``'s registry-era test; the numerics assertions are
unchanged, which is the point — the op moved from a versioned registry entry to
a shipped script without its behaviour moving.

The fixture is a cross in the xy-plane of R^3, centred at the origin: points
(±2, 0, 0) and (0, ±1, 0). With n = 4 and ddof = 1 the variances are
x: (4 + 4)/3 = 8/3, y: (1 + 1)/3 = 2/3, z: 0 — so the ratios are exactly 0.8 and
0.2, and the components are the x and y axes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from causalab.protocol.tables import read_table
from causalab.steps.builtin import fit_pca
from causalab.steps.io import StepError, read_tensor
from tests.steps._run import run_step

pytestmark = pytest.mark.numerical_unit

CROSS = torch.tensor(
    [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
)


def _fit(tmp_path: Path, acts: torch.Tensor, k: int = 2, tag: str = "a"):
    out = tmp_path / tag
    out.mkdir(parents=True, exist_ok=True)
    run_step(
        fit_pca,
        {"acts": acts, "k": k},
        {"weight": out / "w.safetensors", "spectrum": out / "s.json"},
    )
    return read_tensor(out / "w.safetensors"), read_table(out / "s.json")


def test_components_and_spectrum_match_the_oracle(tmp_path):
    weight, spectrum = _fit(tmp_path, CROSS)
    assert weight.shape == (3, 2)  # (d, k)
    assert torch.allclose(weight[:, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(weight[:, 1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert [row["pc"] for row in spectrum] == [0, 1]
    assert [row["explained_variance"] for row in spectrum] == pytest.approx(
        [8 / 3, 2 / 3]
    )
    assert [row["explained_variance_ratio"] for row in spectrum] == pytest.approx(
        [0.8, 0.2]
    )


def test_leading_dimensions_are_flattened(tmp_path):
    """[examples, positions, d] and [rows, d] must mean the same fit."""
    flat, _ = _fit(tmp_path, CROSS, tag="flat")
    nested, _ = _fit(tmp_path, CROSS.reshape(2, 2, 3), tag="nested")
    assert torch.equal(flat, nested)


def test_the_sign_convention_is_pinned(tmp_path):
    """SVD fixes a component only up to a sign. Negating the input must not
    negate the basis, or two equally 'correct' fits would disagree."""
    positive, _ = _fit(tmp_path, CROSS, tag="pos")
    negated, _ = _fit(tmp_path, -CROSS, tag="neg")
    assert torch.equal(positive, negated)
    for column in range(positive.shape[1]):
        component = positive[:, column]
        assert component[component.abs().argmax()] > 0


def test_the_fit_is_deterministic(tmp_path):
    """Bit-identical, twice — the property the registry used to enforce by
    admission and a script now simply has."""
    first, first_spectrum = _fit(tmp_path, CROSS, tag="one")
    second, second_spectrum = _fit(tmp_path, CROSS, tag="two")
    assert torch.equal(first, second)
    assert first_spectrum == second_spectrum


def test_k_beyond_the_available_rank_is_refused(tmp_path):
    with pytest.raises(StepError) as err:
        _fit(tmp_path, CROSS, k=5)
    assert "exceeds the rank available" in str(err.value)


def test_a_one_dimensional_input_is_refused(tmp_path):
    with pytest.raises(StepError):
        _fit(tmp_path, torch.tensor([1.0, 2.0, 3.0]))


def test_a_tensor_path_input_is_read(tmp_path):
    """The runner hands over a path when the document used no selector, so the
    script must accept either."""
    from causalab.steps.io import write_tensor

    bundle = tmp_path / "acts.safetensors"
    write_tensor(bundle, CROSS, slot="acts")
    out = tmp_path / "viapath"
    out.mkdir()
    run_step(
        fit_pca,
        {"acts": bundle, "k": 2},
        {"weight": out / "w.safetensors", "spectrum": out / "s.json"},
    )
    weight = read_tensor(out / "w.safetensors")
    assert weight.shape == (3, 2)
