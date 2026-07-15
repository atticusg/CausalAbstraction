"""Direct tests for ``causalab.methods.trained_subspace.subspace``.

Covers the lossy projection round-trip (orthogonal-complement preserved via
the error term) and the save/load path that round-trips a SubspaceFeaturizer
through :meth:`Featurizer.save_modules`/:meth:`Featurizer.load_modules`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.featurizer import Featurizer


pytestmark = pytest.mark.property


@pytest.fixture(scope="module")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).float()


class TestSubspaceFeaturizerProperty:
    """SubspaceFeaturizer is lossy but reconstructs exactly via the error term."""

    def test_roundtrip_full_rank_reconstructs_exactly(
        self, rng: torch.Generator
    ) -> None:
        x = randn((3, 6), rng)
        sub = SubspaceFeaturizer(shape=(6, 6), trainable=False)
        f, err = sub.featurize(x)
        x_rec = sub.inverse_featurize(f, err)
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_save_load_roundtrip_preserves_reconstruction(
        self, tmp_path: Path, rng: torch.Generator
    ) -> None:
        x = randn((2, 4), rng)
        sub = SubspaceFeaturizer(shape=(4, 4), trainable=False)

        f, err = sub.featurize(x)
        x_rec = sub.inverse_featurize(f, err)

        path_root = tmp_path / "unit"
        sub.save_modules(str(path_root))

        loaded = Featurizer.load_modules(str(path_root))
        f2, err2 = loaded.featurize(x)
        x_rec2 = loaded.inverse_featurize(f2, err2)

        assert torch.allclose(x_rec, x_rec2, atol=1e-6)
        assert torch.allclose(x, x_rec2, atol=1e-5)
