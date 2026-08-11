"""
pytest unit-tests for the FeatureInterpolateIntervention class.

Tests use SubspaceFeaturizer and random tensors — no real model required.

Run with:
    pytest -q tests/test_pyvene_core/test_interpolate.py
"""

from __future__ import annotations


import torch
import pytest

import causalab.neural.featurizer as F
from causalab.methods.trained_subspace.subspace import (
    SubspaceFeaturizer as _SubspaceFeaturizer,
)


pytestmark = pytest.mark.unit


F.SubspaceFeaturizer = _SubspaceFeaturizer  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
#  Helpers / fixtures                                                         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(42)
    return g


def randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).float()


@pytest.fixture(scope="module")
def identity_featurizer() -> F.Featurizer:
    """Identity featurizer — featurize is a no-op, no error."""
    return F.Featurizer(n_features=8, id="identity")


@pytest.fixture(scope="module")
def square_subspace_featurizer() -> F.SubspaceFeaturizer:
    """Square (lossless) subspace featurizer: 8-dim in, 8-dim features."""
    return F.SubspaceFeaturizer(shape=(8, 8), trainable=False, id="square")


@pytest.fixture(scope="module")
def lossy_subspace_featurizer() -> F.SubspaceFeaturizer:
    """Lossy subspace featurizer: 8-dim in, 4-dim features (drops half)."""
    return F.SubspaceFeaturizer(shape=(8, 4), trainable=False, id="lossy")


def apply_interpolation(featurizer, base, src, fn, **params):
    """The interpolate-mode math the engine applies in-trace (SH2, #411):
    featurize both sides, apply ``fn`` in feature space, inverse with the
    BASE error term — ``EditSpec(mode="interpolate")``'s per-tensor contract."""
    f_base, err = featurizer.featurize(base)
    f_src, _ = featurizer.featurize(src)
    return featurizer.inverse_featurize(
        fn(f_base=f_base, f_src=f_src, **params), err
    ).to(base.dtype)


# --------------------------------------------------------------------------- #
#  Tests                                                                      #
# --------------------------------------------------------------------------- #
def test_identity_fn(identity_featurizer: F.Featurizer, rng: torch.Generator) -> None:
    """fn that returns f_base → output equals base reconstructed (identity round-trip)."""
    base = randn((2, 8), rng)
    src = randn((2, 8), rng)

    out = apply_interpolation(
        identity_featurizer, base, src, lambda f_base, f_src: f_base
    )
    assert torch.allclose(out, base, atol=1e-5), "identity fn should return base"


def test_interchange_fn(
    identity_featurizer: F.Featurizer, rng: torch.Generator
) -> None:
    """fn that returns f_src → output matches full interchange (copy source features)."""
    base = randn((2, 8), rng)
    src = randn((2, 8), rng)

    out = apply_interpolation(
        identity_featurizer, base, src, lambda f_base, f_src: f_src
    )
    # For identity featurizer, interchange == copy src activations directly
    assert torch.allclose(out, src, atol=1e-5), "interchange fn should return src"


def test_linear_interp_midpoint(
    square_subspace_featurizer: F.SubspaceFeaturizer, rng: torch.Generator
) -> None:
    """Linear interp at alpha=0.5 → output is midpoint of base and source in feature space."""
    base = randn((3, 8), rng)
    src = randn((3, 8), rng)

    def linear_interp(
        f_base: torch.Tensor, f_src: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        return (1 - alpha) * f_base + alpha * f_src

    out = apply_interpolation(
        square_subspace_featurizer, base, src, linear_interp, alpha=0.5
    )

    # Compute expected: featurize both, average, inverse-featurize
    feat = square_subspace_featurizer
    f_base, base_err = feat.featurize(base)
    f_src, _ = feat.featurize(src)
    f_mid = 0.5 * f_base + 0.5 * f_src
    expected = feat.inverse_featurize(f_mid, base_err).to(base.dtype)

    assert torch.allclose(out, expected, atol=1e-5), "midpoint interpolation mismatch"


def test_base_error_preserved(
    lossy_subspace_featurizer: F.SubspaceFeaturizer, rng: torch.Generator
) -> None:
    """Lossy featurizer: base_err is added back regardless of fn output."""
    base = randn((2, 8), rng)
    src = randn((2, 8), rng)

    # fn copies source features — base_err (from base) should still be preserved
    out = apply_interpolation(
        lossy_subspace_featurizer, base, src, lambda f_base, f_src: f_src
    )

    # Manually reconstruct: featurize base to get base_err, featurize src for f_src
    feat = lossy_subspace_featurizer
    _, base_err = feat.featurize(base)
    f_src, _ = feat.featurize(src)
    expected = feat.inverse_featurize(f_src, base_err).to(base.dtype)

    assert torch.allclose(out, expected, atol=1e-5), "base_err should come from base"
