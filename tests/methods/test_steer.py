"""
pytest unit-tests for steer.py

Tests steering interventions with both Identity and Subspace featurizers.

Run with:
    pytest -q tests/methods/test_steer.py
"""

from __future__ import annotations

from typing import Any

import torch
import pytest

import causalab.neural.featurizer as F
from causalab.neural.interventions import build_intervention
from causalab.methods.trained_subspace.subspace import (
    SubspaceFeaturizer as _SubspaceFeaturizer,
)

# Patch must happen before importing units/steer modules that capture
# F.SubspaceFeaturizer at import time.
F.SubspaceFeaturizer = _SubspaceFeaturizer  # type: ignore[attr-defined]
from causalab.neural.units import (  # noqa: E402  (must follow monkeypatch above)
    AtomicModelUnit,
    InterchangeTarget,
    ComponentIndexer,
)
from causalab.methods.steer.steer import (  # noqa: E402  (must follow monkeypatch above)
    make_zero_features,
    validate_steering_vectors,
    get_batch_steering_vectors,
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
def static_indexer() -> ComponentIndexer:
    """Create a static indexer that always returns [0]."""
    return ComponentIndexer(lambda _: [0], id="static_pos0")


@pytest.fixture
def identity_model_unit(
    identity_featurizer: F.Featurizer, static_indexer: ComponentIndexer
) -> AtomicModelUnit:
    """Create a model unit with identity featurizer."""
    return AtomicModelUnit(
        layer=0,
        component_type="block_output",
        indices_func=static_indexer,
        featurizer=identity_featurizer,
        id="unit_identity",
    )


@pytest.fixture
def subspace_model_unit(
    subspace_featurizer: F.SubspaceFeaturizer, static_indexer: ComponentIndexer
) -> AtomicModelUnit:
    """Create a model unit with subspace featurizer."""
    return AtomicModelUnit(
        layer=1,
        component_type="block_output",
        indices_func=static_indexer,
        featurizer=subspace_featurizer,
        id="unit_subspace",
    )


@pytest.fixture
def interchange_target_identity(
    identity_model_unit: AtomicModelUnit,
) -> InterchangeTarget:
    """Create an InterchangeTarget with identity featurizer."""
    return InterchangeTarget([[identity_model_unit]])


@pytest.fixture
def interchange_target_subspace(
    subspace_model_unit: AtomicModelUnit,
) -> InterchangeTarget:
    """Create an InterchangeTarget with subspace featurizer."""
    return InterchangeTarget([[subspace_model_unit]])


@pytest.fixture
def interchange_target_mixed(
    identity_model_unit: AtomicModelUnit, subspace_model_unit: AtomicModelUnit
) -> InterchangeTarget:
    """Create an InterchangeTarget with both featurizer types."""
    return InterchangeTarget([[identity_model_unit], [subspace_model_unit]])


# --------------------------------------------------------------------------- #
#  Steering intervention class tests                                          #
# --------------------------------------------------------------------------- #
class TestSteeringIntervention:
    """Tests for the steering intervention class itself."""

    def test_steering_adds_vector_identity(
        self, rng: torch.Generator, identity_featurizer: F.Featurizer
    ) -> None:
        """Test that steering adds vector in identity feature space."""
        x_base = randn((2, 8), rng)
        steering_vec = randn((2, 8), rng)

        steering = build_intervention(identity_featurizer, "add")

        out = steering(x_base, steering_vec)

        # With identity featurizer, output should be base + steering
        expected = x_base + steering_vec
        assert torch.allclose(out, expected, atol=1e-6)

    def test_steering_adds_vector_subspace(
        self, rng: torch.Generator, subspace_featurizer: F.SubspaceFeaturizer
    ) -> None:
        """Test that steering adds vector in subspace feature space."""
        # Input is 16-dim, feature space is 4-dim
        x_base = randn((2, 16), rng)
        steering_vec = randn((2, 4), rng)  # In feature space

        steering = build_intervention(subspace_featurizer, "add")

        out = steering(x_base, steering_vec)

        # Verify the output has the same shape as input
        assert out.shape == x_base.shape

        # Verify the steering was applied in feature space
        # Featurize both and check the difference
        base_features, base_error = subspace_featurizer.featurize(x_base)
        out_features, out_error = subspace_featurizer.featurize(out)

        # Features should differ by the steering vector
        feature_diff = out_features - base_features
        assert torch.allclose(
            feature_diff, steering_vec.to(feature_diff.dtype), atol=1e-5
        )

        # Error term should be preserved (orthogonal component unchanged)
        assert out_error is not None and base_error is not None
        assert torch.allclose(out_error, base_error, atol=1e-5)

    def test_steering_preserves_error_term(
        self, rng: torch.Generator, subspace_featurizer: F.SubspaceFeaturizer
    ) -> None:
        """Test that steering preserves the error (orthogonal) component."""
        x_base = randn((3, 16), rng)
        steering_vec = randn((3, 4), rng)

        # Get the original error term
        _, base_error = subspace_featurizer.featurize(x_base)

        steering = build_intervention(subspace_featurizer, "add")

        out = steering(x_base, steering_vec)

        # Get the output error term
        _, out_error = subspace_featurizer.featurize(out)

        # Error terms should be identical
        assert out_error is not None and base_error is not None
        assert torch.allclose(out_error, base_error, atol=1e-5)

    def test_steering_zero_vector_no_change(
        self, rng: torch.Generator, identity_featurizer: F.Featurizer
    ) -> None:
        """Test that zero steering vector produces no change."""
        x_base = randn((2, 8), rng)
        steering_vec = torch.zeros(2, 8)

        steering = build_intervention(identity_featurizer, "add")

        out = steering(x_base, steering_vec)

        assert torch.allclose(out, x_base, atol=1e-6)

    def test_steering_repr_names_the_mode(
        self, identity_featurizer: F.Featurizer
    ) -> None:
        """The object identifies its mode. The legacy ``__str__`` that carried
        the featurizer id is gone with the class factory — an ordinary module's
        repr names its class."""
        steering = build_intervention(identity_featurizer, "add")
        assert "SteeringIntervention" in repr(steering)


# --------------------------------------------------------------------------- #
#  make_zero_features tests                                                   #
# --------------------------------------------------------------------------- #
class TestMakeZeroFeatures:
    """Tests for the make_zero_features helper."""

    def test_creates_zeros_for_each_unit(
        self, interchange_target_identity: InterchangeTarget
    ) -> None:
        """Test that zeros are created for each unit."""
        zeros = make_zero_features(interchange_target_identity)

        assert len(zeros) == 1
        assert "unit_identity" in zeros
        assert zeros["unit_identity"].shape == (8,)
        assert torch.all(zeros["unit_identity"] == 0)

    def test_creates_zeros_mixed_units(
        self, interchange_target_mixed: InterchangeTarget
    ) -> None:
        """Test with mixed featurizer types."""
        zeros = make_zero_features(interchange_target_mixed)

        assert len(zeros) == 2
        assert zeros["unit_identity"].shape == (8,)
        assert zeros["unit_subspace"].shape == (4,)
        assert torch.all(zeros["unit_identity"] == 0)
        assert torch.all(zeros["unit_subspace"] == 0)

    def test_respects_device_and_dtype(
        self, interchange_target_identity: InterchangeTarget
    ) -> None:
        """Test that device and dtype are respected."""
        zeros = make_zero_features(
            interchange_target_identity,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        assert zeros["unit_identity"].device.type == "cpu"
        assert zeros["unit_identity"].dtype == torch.float64

    def test_raises_for_none_n_features(self, static_indexer: ComponentIndexer) -> None:
        """Test that ValueError is raised when n_features is None."""
        featurizer = F.Featurizer(n_features=None, id="no_features")
        unit = AtomicModelUnit(
            layer=0,
            component_type="block_output",
            indices_func=static_indexer,
            featurizer=featurizer,
            id="unit_no_features",
        )
        target = InterchangeTarget([[unit]])

        with pytest.raises(ValueError, match="n_features=None"):
            make_zero_features(target)


# --------------------------------------------------------------------------- #
#  validate_steering_vectors tests                                            #
# --------------------------------------------------------------------------- #
class TestValidateSteeringVectors:
    """Tests for steering vector validation."""

    def test_valid_broadcast_mode(
        self, interchange_target_identity: InterchangeTarget
    ) -> None:
        """Test validation passes for broadcast mode."""
        vectors: dict[str, Any] = {"unit_identity": torch.randn(8)}
        # Should not raise
        validate_steering_vectors(vectors, interchange_target_identity, n_examples=10)

    def test_valid_per_example_mode(
        self, interchange_target_identity: InterchangeTarget
    ) -> None:
        """Test validation passes for per-example mode."""
        vectors: dict[str, Any] = {"unit_identity": torch.randn(10, 8)}
        # Should not raise
        validate_steering_vectors(vectors, interchange_target_identity, n_examples=10)

    def test_raises_for_missing_unit(
        self, interchange_target_identity: InterchangeTarget
    ) -> None:
        """Test that ValueError is raised for missing units."""
        vectors: dict[str, Any] = {}  # Missing unit_identity
        with pytest.raises(ValueError, match="Missing steering vectors"):
            validate_steering_vectors(
                vectors, interchange_target_identity, n_examples=10
            )

    def test_raises_for_wrong_feature_dim_broadcast(
        self, interchange_target_identity: InterchangeTarget
    ) -> None:
        """Test that ValueError is raised for wrong feature dimension in broadcast mode."""
        vectors: dict[str, Any] = {"unit_identity": torch.randn(5)}  # Should be 8
        with pytest.raises(ValueError, match="has 5 features.*expects 8"):
            validate_steering_vectors(
                vectors, interchange_target_identity, n_examples=10
            )

    def test_raises_for_wrong_feature_dim_per_example(
        self, interchange_target_identity: InterchangeTarget
    ) -> None:
        """Test that ValueError is raised for wrong feature dimension in per-example mode."""
        vectors: dict[str, Any] = {
            "unit_identity": torch.randn(10, 5)
        }  # Should be (10, 8)
        with pytest.raises(ValueError, match="has 5 features.*expects 8"):
            validate_steering_vectors(
                vectors, interchange_target_identity, n_examples=10
            )

    def test_raises_for_wrong_example_count(
        self, interchange_target_identity: InterchangeTarget
    ) -> None:
        """Test that ValueError is raised for wrong number of examples."""
        vectors: dict[str, Any] = {
            "unit_identity": torch.randn(5, 8)
        }  # Should be 10 examples
        with pytest.raises(ValueError, match="has 5 examples.*has 10 examples"):
            validate_steering_vectors(
                vectors, interchange_target_identity, n_examples=10
            )

    def test_raises_for_invalid_shape(
        self, interchange_target_identity: InterchangeTarget
    ) -> None:
        """Test that ValueError is raised for invalid tensor shape."""
        vectors: dict[str, Any] = {"unit_identity": torch.randn(2, 3, 8)}  # 3D tensor
        with pytest.raises(ValueError, match="invalid shape"):
            validate_steering_vectors(
                vectors, interchange_target_identity, n_examples=10
            )

    def test_valid_mixed_targets(
        self, interchange_target_mixed: InterchangeTarget
    ) -> None:
        """Test validation with mixed featurizer types."""
        vectors: dict[str, Any] = {
            "unit_identity": torch.randn(8),  # broadcast
            "unit_subspace": torch.randn(10, 4),  # per-example
        }
        # Should not raise
        validate_steering_vectors(vectors, interchange_target_mixed, n_examples=10)


# --------------------------------------------------------------------------- #
#  get_batch_steering_vectors tests                                           #
# --------------------------------------------------------------------------- #
class TestGetBatchSteeringVectors:
    """Tests for batch steering vector extraction."""

    def test_broadcast_mode_expands(
        self, interchange_target_identity: InterchangeTarget
    ) -> None:
        """Test that broadcast mode expands vectors correctly."""
        vec = torch.randn(8)
        vectors: dict[str, Any] = {"unit_identity": vec}

        batch_vecs = get_batch_steering_vectors(
            vectors,
            interchange_target_identity,
            batch_start=0,
            batch_size=4,
            device=torch.device("cpu"),
        )

        assert len(batch_vecs) == 1
        assert batch_vecs[0].shape == (4, 1, 8)
        # Each row should be identical to the original vector
        for i in range(4):
            assert torch.allclose(batch_vecs[0][i, 0], vec)

    def test_per_example_mode_slices(
        self, interchange_target_identity: InterchangeTarget
    ) -> None:
        """Test that per-example mode slices correctly."""
        vec = torch.randn(10, 8)
        vectors: dict[str, Any] = {"unit_identity": vec}

        batch_vecs = get_batch_steering_vectors(
            vectors,
            interchange_target_identity,
            batch_start=2,
            batch_size=3,
            device=torch.device("cpu"),
        )

        assert len(batch_vecs) == 1
        assert batch_vecs[0].shape == (3, 1, 8)
        assert torch.allclose(batch_vecs[0][:, 0, :], vec[2:5])

    def test_mixed_modes(self, interchange_target_mixed: InterchangeTarget) -> None:
        """Test with mixed broadcast and per-example modes."""
        vectors: dict[str, Any] = {
            "unit_identity": torch.randn(8),  # broadcast
            "unit_subspace": torch.randn(10, 4),  # per-example
        }

        batch_vecs = get_batch_steering_vectors(
            vectors,
            interchange_target_mixed,
            batch_start=0,
            batch_size=3,
            device=torch.device("cpu"),
        )

        assert len(batch_vecs) == 2
        assert batch_vecs[0].shape == (3, 1, 8)  # expanded broadcast
        assert batch_vecs[1].shape == (3, 1, 4)  # sliced per-example


# --------------------------------------------------------------------------- #
#  Steering-mode selection on an AtomicModelUnit                              #
# --------------------------------------------------------------------------- #
class TestSteeringModeSelection:
    """``add`` builds a steering intervention over the unit's feature space.

    This replaces two classes: one asserting the shape of the config dict the
    pyvene backbone consumed (``create_intervention_config``), and one asserting
    that ``Featurizer.get_steering_intervention`` returned a cached *class* —
    that backbone instantiated intervention classes itself, so the featurizer had to hand it
    one. Both are gone: a mode is now an ordinary module instance, and each call
    builds a fresh one so a caller can configure it without affecting others.
    """

    def test_add_mode_builds_a_steering_intervention(
        self, identity_model_unit: AtomicModelUnit
    ) -> None:
        from causalab.neural.interventions import SteeringIntervention

        intervention = build_intervention(identity_model_unit.featurizer, "add")
        assert isinstance(intervention, SteeringIntervention)

    def test_returns_a_configured_instance(
        self, identity_featurizer: F.Featurizer
    ) -> None:
        steering = build_intervention(identity_featurizer, "add")
        assert isinstance(steering, torch.nn.Module)
        assert steering._featurizer is identity_featurizer

    def test_each_call_is_a_fresh_object(
        self, identity_featurizer: F.Featurizer
    ) -> None:
        first = build_intervention(identity_featurizer, "add")
        second = build_intervention(identity_featurizer, "add")
        assert first is not second


# --------------------------------------------------------------------------- #
#  Replace intervention                                                        #
# --------------------------------------------------------------------------- #
class TestReplaceIntervention:
    """Tests for the replace intervention class."""

    def test_replace_replaces_features_identity(
        self, rng: torch.Generator, identity_featurizer: F.Featurizer
    ) -> None:
        """Test that replace replaces features in identity feature space."""
        x_base = randn((2, 8), rng)
        replacement_vec = randn((2, 8), rng)

        replace = build_intervention(identity_featurizer, "replace")

        out = replace(x_base, replacement_vec)

        # With identity featurizer, output should equal replacement vector
        # (since error term is None/zero for identity featurizer)
        assert torch.allclose(out, replacement_vec, atol=1e-6)

    def test_replace_ignores_base_features_subspace(
        self, rng: torch.Generator, subspace_featurizer: F.SubspaceFeaturizer
    ) -> None:
        """Test that replace ignores base features in subspace feature space."""
        # Input is 16-dim, feature space is 4-dim
        x_base = randn((2, 16), rng)
        replacement_vec = randn((2, 4), rng)  # In feature space

        replace = build_intervention(subspace_featurizer, "replace")

        out = replace(x_base, replacement_vec)

        # Verify the output has the same shape as input
        assert out.shape == x_base.shape

        # Verify the features were replaced (not added)
        out_features, _ = subspace_featurizer.featurize(out)

        # Output features should equal the replacement vector
        assert torch.allclose(
            out_features, replacement_vec.to(out_features.dtype), atol=1e-5
        )

    def test_replace_preserves_error_term(
        self, rng: torch.Generator, subspace_featurizer: F.SubspaceFeaturizer
    ) -> None:
        """Test that replace preserves the error (orthogonal) component from base."""
        x_base = randn((3, 16), rng)
        replacement_vec = randn((3, 4), rng)

        # Get the original error term
        _, base_error = subspace_featurizer.featurize(x_base)

        replace = build_intervention(subspace_featurizer, "replace")

        out = replace(x_base, replacement_vec)

        # Get the output error term
        _, out_error = subspace_featurizer.featurize(out)

        # Error terms should be identical (preserved from base)
        assert out_error is not None and base_error is not None
        assert torch.allclose(out_error, base_error, atol=1e-5)

    def test_replace_with_zeros_ablates_features(
        self, rng: torch.Generator, subspace_featurizer: F.SubspaceFeaturizer
    ) -> None:
        """Test that replacing with zeros ablates the feature contribution."""
        x_base = randn((2, 16), rng)
        zero_vec = torch.zeros(2, 4)

        replace = build_intervention(subspace_featurizer, "replace")

        out = replace(x_base, zero_vec)

        # Verify features are zeroed
        out_features, _ = subspace_featurizer.featurize(out)
        assert torch.allclose(out_features, zero_vec.to(out_features.dtype), atol=1e-5)

        # But output is not all zeros (error term preserved)
        _, base_error = subspace_featurizer.featurize(x_base)
        assert base_error is not None
        # Output should equal just the error term (inverse of zeros + error)
        expected = subspace_featurizer.inverse_featurize(
            zero_vec.to(base_error.dtype), base_error
        )
        assert torch.allclose(out, expected, atol=1e-5)

    def test_replace_differs_from_steering(
        self, rng: torch.Generator, identity_featurizer: F.Featurizer
    ) -> None:
        """Test that replace produces different results than steering."""
        x_base = randn((2, 8), rng)
        vec = randn((2, 8), rng)

        steering = build_intervention(identity_featurizer, "add")
        replace = build_intervention(identity_featurizer, "replace")

        out_steer = steering(x_base, vec)
        out_replace = replace(x_base, vec)

        # Steering adds: base + vec
        # Replace replaces: vec
        # They should differ (unless base is zero)
        assert not torch.allclose(out_steer, out_replace, atol=1e-6)

    def test_repr_names_the_mode(self, identity_featurizer: F.Featurizer) -> None:
        """The object identifies its mode — the interventions are ordinary
        modules now, so the repr is torch's rather than a hand-written ``__str__``
        that had to carry the featurizer id to be identifiable."""
        replace = build_intervention(identity_featurizer, "replace")
        assert "ReplaceIntervention" in repr(replace)


# --------------------------------------------------------------------------- #
#  Building a replace intervention                                             #
# --------------------------------------------------------------------------- #
class TestBuildReplaceIntervention:
    """``build_intervention(featurizer, "replace")`` construction.

    The old API returned a *class* per featurizer, cached, because the pyvene
    backbone instantiated intervention classes itself. It now returns an ordinary module
    instance, so each call is a fresh object — which is what lets a caller
    configure one (a mask's logits, an interpolation's function) without
    affecting anyone else's.
    """

    def test_returns_a_configured_instance(
        self, identity_featurizer: F.Featurizer
    ) -> None:
        replace = build_intervention(identity_featurizer, "replace")
        assert isinstance(replace, torch.nn.Module)
        assert replace._featurizer is identity_featurizer

    def test_each_call_is_a_fresh_object(
        self, identity_featurizer: F.Featurizer
    ) -> None:
        first = build_intervention(identity_featurizer, "replace")
        second = build_intervention(identity_featurizer, "replace")
        assert first is not second

    def test_works_with_subspace_featurizer(
        self, subspace_featurizer: F.SubspaceFeaturizer
    ) -> None:
        """Test that replace works with subspace featurizer."""
        replace = build_intervention(subspace_featurizer, "replace")
        assert replace is not None


# --------------------------------------------------------------------------- #
#  Replace-mode site resolution on an AtomicModelUnit                         #
# --------------------------------------------------------------------------- #
class TestReplaceSiteResolution:
    """A ``replace`` unit resolves to a real, writable site.

    This replaces tests of ``create_intervention_config``, which asserted the
    shape of the config dict the old backbone consumed (component / unit / layer /
    group_key / intervention_type). There is no config dict any more — a unit is
    resolved to a site and paired with an intervention — so the equivalent
    contract is that both halves come out well-formed.
    """

    def test_unit_resolves_to_a_site(
        self, identity_model_unit: AtomicModelUnit, mock_tiny_lm
    ) -> None:
        from causalab.neural.components import resolve_site

        site = resolve_site(
            mock_tiny_lm.nnsight,
            identity_model_unit.component_type,
            identity_model_unit.layer,
        )
        assert site.key in ("input", "output")
        assert site.envoy is not None

    def test_replace_mode_builds_a_replace_intervention(
        self, identity_model_unit: AtomicModelUnit
    ) -> None:
        from causalab.neural.interventions import ReplaceIntervention

        intervention = build_intervention(identity_model_unit.featurizer, "replace")
        assert isinstance(intervention, ReplaceIntervention)
