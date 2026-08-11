"""
pytest unit-tests for steer.py

Tests steering-vector helpers against the spec surface (WU4, #506): sites are
:class:`~causalab.neural.specs.SiteSpec` values and every per-site dict keys
on ``spec.key``.

Run with:
    pytest -q tests/methods/test_steer.py
"""

from __future__ import annotations

from typing import Any

import torch
import pytest

import causalab.neural.featurizer as F
from causalab.methods.trained_subspace.subspace import (
    SubspaceFeaturizer as _SubspaceFeaturizer,
)

# Patch must happen before importing modules that capture F.SubspaceFeaturizer
# at import time.
F.SubspaceFeaturizer = _SubspaceFeaturizer  # type: ignore[attr-defined]
from causalab.neural.featurized_site import FeaturizedSite  # noqa: E402
from causalab.neural.site import Site  # noqa: E402
from causalab.neural.specs import SiteSpec  # noqa: E402
from causalab.methods.steer.steer import (  # noqa: E402  (must follow monkeypatch above)
    make_zero_features,
    validate_steering_vectors,
)


pytestmark = pytest.mark.unit


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


@pytest.fixture
def identity_featurizer() -> F.Featurizer:
    """Create an identity featurizer with 8 features."""
    return F.Featurizer(n_features=8, id="identity")


@pytest.fixture
def subspace_featurizer() -> F.SubspaceFeaturizer:
    """Create a subspace featurizer that projects 16-dim to 4-dim."""
    return F.SubspaceFeaturizer(shape=(16, 4), trainable=False, id="subspace")


@pytest.fixture
def identity_site(identity_featurizer: F.Featurizer) -> SiteSpec:
    """Create a site spec with identity featurizer (static position 0)."""
    return SiteSpec(
        fsite=FeaturizedSite(Site("block_output", 0), identity_featurizer),
        positions=(0,),
        key="unit_identity",
        width=8,
    )


@pytest.fixture
def subspace_site(subspace_featurizer: F.SubspaceFeaturizer) -> SiteSpec:
    """Create a site spec with subspace featurizer (static position 0)."""
    return SiteSpec(
        fsite=FeaturizedSite(Site("block_output", 1), subspace_featurizer),
        positions=(0,),
        key="unit_subspace",
        width=16,
    )


@pytest.fixture
def sites_identity(identity_site: SiteSpec) -> list[SiteSpec]:
    """Flat site list with identity featurizer."""
    return [identity_site]


@pytest.fixture
def sites_mixed(identity_site: SiteSpec, subspace_site: SiteSpec) -> list[SiteSpec]:
    """Flat site list with both featurizer types."""
    return [identity_site, subspace_site]


# --------------------------------------------------------------------------- #
#  make_zero_features tests                                                   #
# --------------------------------------------------------------------------- #
class TestMakeZeroFeatures:
    """Tests for the make_zero_features helper."""

    def test_creates_zeros_for_each_site(self, sites_identity: list[SiteSpec]) -> None:
        """Test that zeros are created for each site."""
        zeros = make_zero_features(sites_identity)

        assert len(zeros) == 1
        assert "unit_identity" in zeros
        assert zeros["unit_identity"].shape == (8,)
        assert torch.all(zeros["unit_identity"] == 0)

    def test_creates_zeros_mixed_sites(self, sites_mixed: list[SiteSpec]) -> None:
        """Test with mixed featurizer types."""
        zeros = make_zero_features(sites_mixed)

        assert len(zeros) == 2
        assert zeros["unit_identity"].shape == (8,)
        assert zeros["unit_subspace"].shape == (4,)
        assert torch.all(zeros["unit_identity"] == 0)
        assert torch.all(zeros["unit_subspace"] == 0)

    def test_respects_device_and_dtype(self, sites_identity: list[SiteSpec]) -> None:
        """Test that device and dtype are respected."""
        zeros = make_zero_features(
            sites_identity,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        assert zeros["unit_identity"].device.type == "cpu"
        assert zeros["unit_identity"].dtype == torch.float64

    def test_falls_back_to_width(self) -> None:
        """An identity featurizer without n_features sizes from spec.width."""
        spec = SiteSpec(
            fsite=FeaturizedSite(Site("block_output", 0)),
            positions=(0,),
            key="unit_raw",
            width=6,
        )
        zeros = make_zero_features([spec])
        assert zeros["unit_raw"].shape == (6,)

    def test_raises_for_none_n_features(self) -> None:
        """Test that ValueError is raised when n_features and width are None."""
        featurizer = F.Featurizer(n_features=None, id="no_features")
        spec = SiteSpec(
            fsite=FeaturizedSite(Site("block_output", 0), featurizer),
            positions=(0,),
            key="unit_no_features",
            width=None,
        )

        with pytest.raises(ValueError, match="n_features=None"):
            make_zero_features([spec])


# --------------------------------------------------------------------------- #
#  validate_steering_vectors tests                                            #
# --------------------------------------------------------------------------- #
class TestValidateSteeringVectors:
    """Tests for steering vector validation."""

    def test_valid_broadcast_mode(self, sites_identity: list[SiteSpec]) -> None:
        """Test validation passes for broadcast mode."""
        vectors: dict[str, Any] = {"unit_identity": torch.randn(8)}
        # Should not raise
        validate_steering_vectors(vectors, sites_identity, n_examples=10)

    def test_valid_per_example_mode(self, sites_identity: list[SiteSpec]) -> None:
        """Test validation passes for per-example mode."""
        vectors: dict[str, Any] = {"unit_identity": torch.randn(10, 8)}
        # Should not raise
        validate_steering_vectors(vectors, sites_identity, n_examples=10)

    def test_raises_for_missing_site(self, sites_identity: list[SiteSpec]) -> None:
        """Test that ValueError is raised for missing sites."""
        vectors: dict[str, Any] = {}  # Missing unit_identity
        with pytest.raises(ValueError, match="Missing steering vectors"):
            validate_steering_vectors(vectors, sites_identity, n_examples=10)

    def test_raises_for_wrong_feature_dim_broadcast(
        self, sites_identity: list[SiteSpec]
    ) -> None:
        """Test that ValueError is raised for wrong feature dimension in broadcast mode."""
        vectors: dict[str, Any] = {"unit_identity": torch.randn(5)}  # Should be 8
        with pytest.raises(ValueError, match="has 5 features.*expects 8"):
            validate_steering_vectors(vectors, sites_identity, n_examples=10)

    def test_raises_for_wrong_feature_dim_per_example(
        self, sites_identity: list[SiteSpec]
    ) -> None:
        """Test that ValueError is raised for wrong feature dimension in per-example mode."""
        vectors: dict[str, Any] = {
            "unit_identity": torch.randn(10, 5)
        }  # Should be (10, 8)
        with pytest.raises(ValueError, match="has 5 features.*expects 8"):
            validate_steering_vectors(vectors, sites_identity, n_examples=10)

    def test_raises_for_wrong_example_count(
        self, sites_identity: list[SiteSpec]
    ) -> None:
        """Test that ValueError is raised for wrong number of examples."""
        vectors: dict[str, Any] = {
            "unit_identity": torch.randn(5, 8)
        }  # Should be 10 examples
        with pytest.raises(ValueError, match="has 5 examples.*has 10 examples"):
            validate_steering_vectors(vectors, sites_identity, n_examples=10)

    def test_raises_for_invalid_shape(self, sites_identity: list[SiteSpec]) -> None:
        """Test that ValueError is raised for invalid tensor shape."""
        vectors: dict[str, Any] = {"unit_identity": torch.randn(2, 3, 8)}  # 3D tensor
        with pytest.raises(ValueError, match="invalid shape"):
            validate_steering_vectors(vectors, sites_identity, n_examples=10)

    def test_valid_mixed_sites(self, sites_mixed: list[SiteSpec]) -> None:
        """Test validation with mixed featurizer types."""
        vectors: dict[str, Any] = {
            "unit_identity": torch.randn(8),  # broadcast
            "unit_subspace": torch.randn(10, 4),  # per-example
        }
        # Should not raise
        validate_steering_vectors(vectors, sites_mixed, n_examples=10)
