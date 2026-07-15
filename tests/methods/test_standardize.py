"""Direct tests for ``causalab.methods.standardize``.

Covers the bijective affine standardization round-trip (no error term) and
the dict-based ``to_dict``/``from_dict`` serialization path that
``Featurizer.from_dict`` dispatches through.
"""

from __future__ import annotations

import pytest
import torch

from causalab.methods.standardize import StandardizeFeaturizer
from causalab.neural.featurizer import Featurizer


pytestmark = pytest.mark.property


@pytest.fixture(scope="module")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).float()


class TestStandardizeFeaturizerProperty:
    """StandardizeFeaturizer is bijective: perfect reconstruction, no error term."""

    def test_roundtrip_is_lossless(self, rng: torch.Generator) -> None:
        x = randn((3, 4), rng)
        mean = torch.randn(4)
        std = torch.rand(4) + 0.1  # ensure positive

        standardize = StandardizeFeaturizer(mean, std)
        f, err = standardize.featurize(x)

        assert err is None
        x_rec = standardize.inverse_featurize(f, err)
        assert torch.allclose(x, x_rec, atol=1e-6)

    def test_to_dict_from_dict_roundtrip(self, rng: torch.Generator) -> None:
        x = randn((3, 4), rng)
        mean = torch.randn(4)
        std = torch.rand(4) + 0.1

        original = StandardizeFeaturizer(mean, std, eps=1e-5)

        data = original.to_dict()
        assert data["model_info"]["featurizer_class"] == "StandardizeFeaturizerModule"

        loaded = Featurizer.from_dict(data)
        assert isinstance(loaded, StandardizeFeaturizer)

        f1, _ = original.featurize(x)
        f2, _ = loaded.featurize(x)
        assert torch.allclose(f1, f2, atol=1e-6)
