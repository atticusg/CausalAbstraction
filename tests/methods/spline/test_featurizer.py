"""Rank-polymorphism of the manifold featurizer modules under ``keep_last_dim``.

Feature interventions set ``keep_last_dim=True`` (PR #334) so a featurizer is
handed ``(batch, num_pos, d)`` — and even the *single-token* path arrives as
``(batch, 1, d)`` rather than ``(batch, d)``. The spline manifold's
``encode``/``fwd``/``inv`` hard-code a 2-D row layout (``argmin(dim=1)``,
``z[:, :k]``, ``torch.arange(x.shape[0])``), so before the leading-dim shim a
single-token interchange / path-steering run crashed inside ``Manifold.encode``
with ``IndexError: ... shapes [b], [b, n_centroids]``.

The broadcast featurizers in ``test_interchange_mode`` (``x * w``, ``x @ R``)
operate on the last dim only, so they pass a ``(b, 1, d)`` tensor through
transparently and are structurally blind to this rank change. The manifold
featurizer is the regression surface: these tests lock that it accepts arbitrary
leading dims and applies the manifold *per point*.
"""

import pytest
import torch

from causalab.methods.spline.builders import build_spline_manifold
from causalab.methods.spline.featurizer import (
    ManifoldFeaturizerModule,
    ManifoldInverseFeaturizerModule,
    ManifoldProjectFeaturizerModule,
)

pytestmark = pytest.mark.unit

_AMBIENT = 4


def _tiny_manifold():
    """A 1-D intrinsic curve embedded in 4-D ambient space (mirrors the
    path-steering setup: ``intrinsic_dim=1``, last-token span). ``n_centroids=5``
    so a 3-D input would make ``argmin(dim=1)`` return a ``(b, 5)`` index — the
    shape that triggered the original broadcast crash."""
    control_points = torch.linspace(0.0, 1.0, 5).unsqueeze(1)  # (5, 1)
    t = torch.linspace(0.0, 1.0, 5)
    centroids = torch.stack([t, t**2, torch.sin(3 * t), torch.cos(3 * t)], dim=1)
    return build_spline_manifold(
        control_points, centroids, intrinsic_dim=1, ambient_dim=_AMBIENT
    )


@pytest.mark.parametrize("num_pos", [1, 3])
def test_manifold_featurizer_module_is_rank_polymorphic(num_pos: int) -> None:
    module = ManifoldFeaturizerModule(_tiny_manifold())
    torch.manual_seed(0)
    b = 2
    x3 = torch.randn(b, num_pos, _AMBIENT)

    z3, err = module(x3)
    assert err is None
    assert z3.shape == (b, num_pos, _AMBIENT)

    # Per-point application: flattening the leading (batch, num_pos) dims into one
    # row axis must commute with the featurizer (each position projected alone).
    z_flat, _ = module(x3.reshape(b * num_pos, _AMBIENT))
    assert torch.allclose(z3.reshape(b * num_pos, _AMBIENT), z_flat, atol=1e-5)

    # The single-token contract: (b, 1, d) is numerically identical to the plain
    # 2-D (b, d) result, unsqueezed. This is the case the regression broke.
    if num_pos == 1:
        z2, _ = module(x3.squeeze(1))
        assert z2.shape == (b, _AMBIENT)
        assert torch.allclose(z3, z2.unsqueeze(1), atol=1e-5)


@pytest.mark.parametrize("num_pos", [1, 3])
def test_manifold_inverse_featurizer_module_is_rank_polymorphic(num_pos: int) -> None:
    module = ManifoldInverseFeaturizerModule(_tiny_manifold())
    torch.manual_seed(1)
    b = 2
    z3 = torch.randn(b, num_pos, _AMBIENT)

    x3 = module(z3, None)
    assert x3.shape == (b, num_pos, _AMBIENT)

    x_flat = module(z3.reshape(b * num_pos, _AMBIENT), None)
    assert torch.allclose(x3.reshape(b * num_pos, _AMBIENT), x_flat, atol=1e-5)

    if num_pos == 1:
        x2 = module(z3.squeeze(1), None)
        assert torch.allclose(x3, x2.unsqueeze(1), atol=1e-5)


@pytest.mark.parametrize("num_pos", [1, 3])
def test_manifold_project_featurizer_module_is_rank_polymorphic(num_pos: int) -> None:
    manifold = _tiny_manifold()
    mean = torch.zeros(_AMBIENT)
    std = torch.ones(_AMBIENT)
    module = ManifoldProjectFeaturizerModule(manifold, mean, std)
    torch.manual_seed(2)
    b = 2
    x3 = torch.randn(b, num_pos, _AMBIENT)

    x_proj3, err = module(x3)
    assert err is None
    assert x_proj3.shape == (b, num_pos, _AMBIENT)

    x_proj_flat, _ = module(x3.reshape(b * num_pos, _AMBIENT))
    assert torch.allclose(
        x_proj3.reshape(b * num_pos, _AMBIENT), x_proj_flat, atol=1e-5
    )

    if num_pos == 1:
        x_proj2, _ = module(x3.squeeze(1))
        assert torch.allclose(x_proj3, x_proj2.unsqueeze(1), atol=1e-5)
