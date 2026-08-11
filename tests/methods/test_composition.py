"""Composition tests across concrete featurizer subclasses.

Exercises :class:`causalab.neural.featurizer.ComposedFeaturizer` and the
``>>`` operator with subclasses from ``causalab.methods.*`` (subspace,
standardize). These tests need both concrete subclasses, so they live here
rather than in ``tests/neural/test_featurizer.py`` (which restricts itself to
the base module and its identity modules).
"""

from __future__ import annotations

import pytest
import torch

from causalab.methods.standardize import StandardizeFeaturizer
from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.featurizer import ComposedFeaturizer, Featurizer


@pytest.fixture(scope="module")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).float()


class TestComposedFeaturizerWithSubclassesProperty:
    """``>>`` composition with concrete subclasses preserves invariants."""

    pytestmark = pytest.mark.property

    def test_basic_composition_metadata(self, rng: torch.Generator) -> None:
        sub1 = SubspaceFeaturizer(shape=(6, 4), trainable=False)
        sub2 = SubspaceFeaturizer(shape=(4, 2), trainable=False)

        composed = sub1 >> sub2

        assert isinstance(composed, ComposedFeaturizer)
        assert composed.n_features == 2
        assert "subspace >> subspace" in composed.id

    def test_lossy_then_bijective_roundtrip(self, rng: torch.Generator) -> None:
        x = randn((3, 6), rng)

        sub = SubspaceFeaturizer(shape=(6, 4), trainable=False)
        standardize = StandardizeFeaturizer(torch.zeros(4), torch.ones(4))

        composed = sub >> standardize
        features, errors = composed.featurize(x)

        assert isinstance(errors, list)
        assert len(errors) == 2
        assert errors[0] is not None  # lossy stage carries error
        assert errors[1] is None  # bijective stage has no error

        x_rec = composed.inverse_featurize(features, errors)
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_bijective_then_bijective_roundtrip(self, rng: torch.Generator) -> None:
        x = randn((3, 4), rng)
        std1 = StandardizeFeaturizer(torch.zeros(4), torch.ones(4))
        std2 = StandardizeFeaturizer(torch.randn(4), torch.rand(4) + 0.1)

        composed = std1 >> std2
        features, errors = composed.featurize(x)

        assert errors == [None, None]

        x_rec = composed.inverse_featurize(features, errors)
        assert torch.allclose(x, x_rec, atol=1e-6)

    def test_associativity_with_three_stages(self, rng: torch.Generator) -> None:
        x = randn((3, 8), rng)

        a = SubspaceFeaturizer(shape=(8, 6), trainable=False)
        b = SubspaceFeaturizer(shape=(6, 4), trainable=False)
        c = StandardizeFeaturizer(torch.zeros(4), torch.ones(4))

        left = (a >> b) >> c
        right = a >> (b >> c)

        assert len(left.stages) == 3
        assert len(right.stages) == 3
        assert left.n_features == right.n_features == 4

        _, errors_left = left.featurize(x)
        _, errors_right = right.featurize(x)
        assert len(errors_left) == len(errors_right) == 3

    def test_multiple_lossy_stages_preserve_errors(self, rng: torch.Generator) -> None:
        x = randn((3, 8), rng)

        sub1 = SubspaceFeaturizer(shape=(8, 6), trainable=False)
        sub2 = SubspaceFeaturizer(shape=(6, 4), trainable=False)

        composed = sub1 >> sub2
        features, errors = composed.featurize(x)

        assert len(errors) == 2
        assert errors[0] is not None
        assert errors[1] is not None

        x_rec = composed.inverse_featurize(features, errors)
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_to_dict_from_dict_roundtrip(self, rng: torch.Generator) -> None:
        x = randn((3, 6), rng)

        sub = SubspaceFeaturizer(shape=(6, 4), trainable=False)
        standardize = StandardizeFeaturizer(torch.zeros(4), torch.ones(4))
        composed = sub >> standardize

        data = composed.to_dict()
        assert data is not None
        assert data["model_info"]["featurizer_class"] == "ComposedFeaturizer"
        assert len(data["stages"]) == 2

        loaded = Featurizer.from_dict(data)
        assert isinstance(loaded, ComposedFeaturizer)
        assert len(loaded.stages) == 2

        f1, e1 = composed.featurize(x)
        f2, e2 = loaded.featurize(x)
        assert torch.allclose(f1, f2, atol=1e-6)

        x_rec1 = composed.inverse_featurize(f1, e1)
        x_rec2 = loaded.inverse_featurize(f2, e2)
        assert torch.allclose(x_rec1, x_rec2, atol=1e-6)
