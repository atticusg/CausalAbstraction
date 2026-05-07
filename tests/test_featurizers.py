"""
pytest unit-tests for featurizers.py

Run with:
    pytest -q test_featurizers.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

import causalab.neural.featurizer as F
from causalab.methods.trained_subspace.subspace import (
    SubspaceFeaturizer as _SubspaceFeaturizer,
)
from causalab.methods.standardize import StandardizeFeaturizer as _StandardizeFeaturizer

F.SubspaceFeaturizer = _SubspaceFeaturizer  # type: ignore[attr-defined]
F.StandardizeFeaturizer = _StandardizeFeaturizer  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
#  Helpers / fixtures                                                         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).float()


# --------------------------------------------------------------------------- #
#  Identity featurizer                                                        #
# --------------------------------------------------------------------------- #
def test_identity_roundtrip(rng: torch.Generator) -> None:
    x = randn((2, 4), rng)

    feat = F.Featurizer(n_features=4)  # identity by default
    f, err = feat.featurize(x)
    assert err is None
    assert torch.equal(f, x)

    x_rec = feat.inverse_featurize(f, err)
    assert torch.equal(x, x_rec)


# --------------------------------------------------------------------------- #
#  Subspace featurizer                                                        #
# --------------------------------------------------------------------------- #
def test_subspace_roundtrip(rng: torch.Generator) -> None:
    x = randn((3, 6), rng)

    sub = F.SubspaceFeaturizer(shape=(6, 6), trainable=False)
    f, err = sub.featurize(x)
    x_rec = sub.inverse_featurize(f, err)

    assert torch.allclose(x, x_rec, atol=1e-5)


# --------------------------------------------------------------------------- #
#  Interchange intervention                                                   #
# --------------------------------------------------------------------------- #
def test_interchange_swaps_when_subspaces_none(rng: torch.Generator) -> None:
    x_base = randn((2, 4), rng)
    x_src = randn((2, 4), rng)

    feat = F.Featurizer(n_features=4)
    Interchange = feat.get_interchange_intervention()
    inter = Interchange()  # default kwargs ok

    out = inter(x_base, x_src, subspaces=None)
    assert torch.equal(out, x_src)


# --------------------------------------------------------------------------- #
#  Mask intervention basic invariants                                         #
# --------------------------------------------------------------------------- #
def test_mask_requires_n_features() -> None:
    feat = F.Featurizer()  # n_features=None
    with pytest.raises(ValueError):
        _ = feat.get_mask_intervention()


def test_mask_forward_training_and_eval(rng: torch.Generator) -> None:
    x_base = randn((1, 4), rng)
    x_src = randn((1, 4), rng)

    feat = F.Featurizer(n_features=4)
    MaskCls = feat.get_mask_intervention()
    mask = MaskCls()

    # Temperature must be set first
    mask.set_temperature(1.0)  # pyright: ignore[reportCallIssue]

    # --------------------------------------------------------------------- #
    # 1. Training mode – push mask to 1 ⇒ output ≈ src                      #
    # --------------------------------------------------------------------- #
    mask.train()
    mask.mask.data.fill_(20.0)  # pyright: ignore[reportCallIssue] # sigmoid(20) ≈ 1 − 2e-9
    out_train = mask(x_base, x_src)
    assert torch.allclose(out_train, x_src, atol=1e-6)

    # --------------------------------------------------------------------- #
    # 2. Eval mode – binary gate                                            #
    # --------------------------------------------------------------------- #
    mask.eval()
    out_eval = mask(x_base, x_src)
    assert torch.allclose(out_eval, x_src, atol=1e-5)


# --------------------------------------------------------------------------- #
#  Serialization round-trips                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "factory",
    [
        lambda: F.Featurizer(n_features=4),
        lambda: F.SubspaceFeaturizer(shape=(4, 4), trainable=False),
    ],
)
def test_save_load_roundtrip(factory, tmp_path: Path, rng: torch.Generator) -> None:
    feat = factory()
    x = randn((2, 4), rng)

    f, err = feat.featurize(x)
    x_rec = feat.inverse_featurize(f, err)

    path_root = tmp_path / "unit"
    feat.save_modules(str(path_root))

    loaded = F.Featurizer.load_modules(str(path_root))
    f2, err2 = loaded.featurize(x)
    x_rec2 = loaded.inverse_featurize(f2, err2)

    # reconstruction stays the same after reload
    assert torch.allclose(x_rec, x_rec2, atol=1e-6)
    # and round-trip still faithful to original input
    assert torch.allclose(x, x_rec2, atol=1e-5)


# --------------------------------------------------------------------------- #
#  __str__ helpers                                                            #
# --------------------------------------------------------------------------- #
def test_collect_str() -> None:
    feat = F.Featurizer(n_features=4, id="my_feat")
    Collect = feat.get_collect_intervention()
    col = Collect()
    assert "my_feat" in str(col)


# --------------------------------------------------------------------------- #
#  Composed featurizers                                                        #
# --------------------------------------------------------------------------- #
def test_standardize_roundtrip(rng: torch.Generator) -> None:
    """StandardizeFeaturizer is bijective - perfect reconstruction."""
    x = randn((3, 4), rng)
    mean = torch.randn(4)
    std = torch.rand(4) + 0.1  # ensure positive

    standardize = F.StandardizeFeaturizer(mean, std)
    f, err = standardize.featurize(x)

    assert err is None  # bijective, no error term
    x_rec = standardize.inverse_featurize(f, err)
    assert torch.allclose(x, x_rec, atol=1e-6)


def test_composition_basic(rng: torch.Generator) -> None:
    """Test basic composition with >> operator."""
    _ = randn((3, 6), rng)  # consume rng for consistency

    # Create two subspace featurizers
    sub1 = F.SubspaceFeaturizer(shape=(6, 4), trainable=False)
    sub2 = F.SubspaceFeaturizer(shape=(4, 2), trainable=False)

    # Compose them
    composed = sub1 >> sub2

    assert isinstance(composed, F.ComposedFeaturizer)
    assert composed.n_features == 2
    assert "subspace >> subspace" in composed.id


def test_composition_roundtrip_lossy(rng: torch.Generator) -> None:
    """Composed featurizer with lossy stage preserves error for reconstruction."""
    x = randn((3, 6), rng)

    # Lossy: projects 6 -> 4
    sub = F.SubspaceFeaturizer(shape=(6, 4), trainable=False)
    # Bijective: standardizes
    mean = torch.zeros(4)
    std = torch.ones(4)
    standardize = F.StandardizeFeaturizer(mean, std)

    composed = sub >> standardize
    features, errors = composed.featurize(x)

    # Check error structure
    assert isinstance(errors, list)
    assert len(errors) == 2
    assert errors[0] is not None  # lossy stage has error
    assert errors[1] is None  # bijective stage has no error

    # Reconstruct
    x_rec = composed.inverse_featurize(features, errors)
    assert torch.allclose(x, x_rec, atol=1e-5)


def test_composition_roundtrip_bijective(rng: torch.Generator) -> None:
    """Two bijective stages compose to bijective."""
    x = randn((3, 4), rng)

    mean1, std1 = torch.zeros(4), torch.ones(4)
    mean2, std2 = torch.randn(4), torch.rand(4) + 0.1

    std1 = F.StandardizeFeaturizer(mean1, std1)
    std2 = F.StandardizeFeaturizer(mean2, std2)

    composed = std1 >> std2
    features, errors = composed.featurize(x)

    # Both None (bijective)
    assert errors == [None, None]

    x_rec = composed.inverse_featurize(features, errors)
    assert torch.allclose(x, x_rec, atol=1e-6)


def test_composition_associativity(rng: torch.Generator) -> None:
    """(a >> b) >> c == a >> (b >> c) in terms of structure."""
    x = randn((3, 8), rng)

    a = F.SubspaceFeaturizer(shape=(8, 6), trainable=False)
    b = F.SubspaceFeaturizer(shape=(6, 4), trainable=False)
    c = F.StandardizeFeaturizer(torch.zeros(4), torch.ones(4))

    # Two ways to compose
    left = (a >> b) >> c
    right = a >> (b >> c)

    # Both should be flat with 3 stages
    assert len(left.stages) == 3
    assert len(right.stages) == 3

    # Same n_features
    assert left.n_features == right.n_features == 4

    # Both produce same-length error lists
    _, errors_left = left.featurize(x)
    _, errors_right = right.featurize(x)
    assert len(errors_left) == len(errors_right) == 3


def test_composition_multiple_lossy(rng: torch.Generator) -> None:
    """Multiple lossy stages preserve all errors."""
    x = randn((3, 8), rng)

    # Two lossy projections
    sub1 = F.SubspaceFeaturizer(shape=(8, 6), trainable=False)
    sub2 = F.SubspaceFeaturizer(shape=(6, 4), trainable=False)

    composed = sub1 >> sub2
    features, errors = composed.featurize(x)

    # Both stages should have error
    assert len(errors) == 2
    assert errors[0] is not None
    assert errors[1] is not None

    # Full reconstruction
    x_rec = composed.inverse_featurize(features, errors)
    assert torch.allclose(x, x_rec, atol=1e-5)


def test_composed_to_dict_from_dict(rng: torch.Generator) -> None:
    """Serialization roundtrip for composed featurizers."""
    x = randn((3, 6), rng)

    sub = F.SubspaceFeaturizer(shape=(6, 4), trainable=False)
    standardize = F.StandardizeFeaturizer(torch.zeros(4), torch.ones(4))
    composed = sub >> standardize

    # Serialize
    data = composed.to_dict()
    assert data is not None
    assert data["model_info"]["featurizer_class"] == "ComposedFeaturizer"
    assert len(data["stages"]) == 2

    # Deserialize
    loaded = F.Featurizer.from_dict(data)
    assert isinstance(loaded, F.ComposedFeaturizer)
    assert len(loaded.stages) == 2

    # Check equivalence
    f1, e1 = composed.featurize(x)
    f2, e2 = loaded.featurize(x)
    assert torch.allclose(f1, f2, atol=1e-6)

    x_rec1 = composed.inverse_featurize(f1, e1)
    x_rec2 = loaded.inverse_featurize(f2, e2)
    assert torch.allclose(x_rec1, x_rec2, atol=1e-6)


def test_standardize_to_dict_from_dict(rng: torch.Generator) -> None:
    """Serialization roundtrip for StandardizeFeaturizer."""
    x = randn((3, 4), rng)
    mean = torch.randn(4)
    std = torch.rand(4) + 0.1

    original = F.StandardizeFeaturizer(mean, std, eps=1e-5)

    data = original.to_dict()
    assert data["model_info"]["featurizer_class"] == "StandardizeFeaturizerModule"

    loaded = F.Featurizer.from_dict(data)
    assert isinstance(loaded, F.StandardizeFeaturizer)

    # Check equivalence
    f1, _ = original.featurize(x)
    f2, _ = loaded.featurize(x)
    assert torch.allclose(f1, f2, atol=1e-6)


def test_composed_intervention_compatibility(rng: torch.Generator) -> None:
    """Composed featurizers work with intervention builders."""
    x_base = randn((2, 6), rng)
    x_src = randn((2, 6), rng)

    sub = F.SubspaceFeaturizer(shape=(6, 4), trainable=False)
    standardize = F.StandardizeFeaturizer(torch.zeros(4), torch.ones(4))
    composed = sub >> standardize

    # Get interchange intervention
    Interchange = composed.get_interchange_intervention()
    inter = Interchange()

    # Should swap features
    out = inter(x_base, x_src, subspaces=None)

    # Verify: output should be x_src passed through compose-then-inverse
    f_src, _ = composed.featurize(x_src)
    _, base_errors = composed.featurize(x_base)
    expected = composed.inverse_featurize(f_src, base_errors)

    assert torch.allclose(out, expected, atol=1e-5)


def test_composed_steering_intervention(rng: torch.Generator) -> None:
    """Composed featurizers work with steering interventions."""
    x = randn((2, 6), rng)
    steering_vec = randn((2, 4), rng)  # in feature space

    sub = F.SubspaceFeaturizer(shape=(6, 4), trainable=False)
    standardize = F.StandardizeFeaturizer(torch.zeros(4), torch.ones(4))
    composed = sub >> standardize

    Steering = composed.get_steering_intervention()
    steer = Steering()

    out = steer(x, steering_vec)

    # Manually compute expected
    features, errors = composed.featurize(x)
    steered = features + steering_vec
    expected = composed.inverse_featurize(steered, errors)

    assert torch.allclose(out, expected, atol=1e-5)
