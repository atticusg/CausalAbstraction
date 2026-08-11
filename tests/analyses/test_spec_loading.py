"""Constructive featurizer loading for the analysis layer (WU5, #507).

Pins the two rebuilt loaders:

* ``causalab.analyses.activation_manifold.loading.load_featurizer`` /
  ``apply_loaded_featurizer`` — read a per-cell WU1 spec bundle
  (``models/<layer>__<pos_id>``) and return the stored featurizer (+ feature
  ids) for functional application; the legacy path mutated a caller-prebuilt
  target in place.
* ``causalab.analyses.subspace.loading.load_subspace_onto_spec`` — the
  pca branch keeps its rotation-artifact contract
  (``rotation.safetensors["rotation_matrix"]``) and returns an updated spec.
"""

from __future__ import annotations

import os

import pytest
import torch
from safetensors.torch import save_file

from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec, save_site_specs

pytestmark = pytest.mark.unit

WIDTH = 8
K = 3


class _StubPos:
    """Minimal named PositionResolver (the grid builders bind real ones)."""

    def __init__(self, name: str) -> None:
        self.id = name

    def index(self, inp):  # pragma: no cover - never resolved here
        return [0]


def _spec(layer: int = 2, pos: str = "last") -> SiteSpec:
    return SiteSpec(
        fsite=FeaturizedSite(Site("block_output", layer)),
        positions=_StubPos(pos),
        key=f"residual_stream.L{layer}.block_output.{pos}",
        width=WIDTH,
    )


def _subspace_featurizer():
    from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer

    return SubspaceFeaturizer(
        rotation_subspace=torch.randn(WIDTH, K), trainable=False, id="PCA"
    )


class TestLoadFeaturizer:
    def test_none_path_returns_identity(self) -> None:
        from causalab.analyses.activation_manifold.loading import load_featurizer

        featurizer, feature_ids = load_featurizer(None, 2, "last")
        assert featurizer.is_trivial()
        assert feature_ids is None

    def test_missing_dir_falls_back_to_identity(self, tmp_path) -> None:
        from causalab.analyses.activation_manifold.loading import load_featurizer

        featurizer, feature_ids = load_featurizer(str(tmp_path), 2, "last")
        assert featurizer.is_trivial()
        assert feature_ids is None

    def test_round_trip_from_spec_bundle(self, tmp_path) -> None:
        from causalab.analyses.activation_manifold.loading import load_featurizer

        trained = _spec().with_featurizer(_subspace_featurizer())
        bundle_dir = os.path.join(str(tmp_path), "models", "2__last")
        save_site_specs([trained], bundle_dir)

        featurizer, feature_ids = load_featurizer(str(tmp_path), 2, "last")
        assert not featurizer.is_trivial()
        assert featurizer.n_features == K
        assert feature_ids is None

    def test_apply_loaded_featurizer_is_functional(self, tmp_path) -> None:
        from causalab.analyses.activation_manifold.loading import (
            apply_loaded_featurizer,
        )

        trained = (
            _spec().with_featurizer(_subspace_featurizer()).with_feature_ids([0, 2])
        )
        bundle_dir = os.path.join(str(tmp_path), "models", "2__last")
        save_site_specs([trained], bundle_dir)

        base = _spec()
        updated, featurizer = apply_loaded_featurizer(base, str(tmp_path), 2, "last")
        # Functional: the input spec is unchanged; the returned spec carries
        # the loaded featurizer and stored feature ids.
        assert base.fsite.featurizer.is_trivial()
        assert not updated.fsite.featurizer.is_trivial()
        assert updated.fsite.feature_ids == (0, 2)
        assert updated.key == base.key
        assert featurizer is updated.fsite.featurizer


class TestLoadSubspaceOntoSpec:
    def test_pca_branch_reads_rotation_artifact(self, tmp_path) -> None:
        from causalab.analyses.subspace import load_subspace_onto_spec

        rotation = torch.randn(WIDTH, K)
        save_file(
            {"rotation_matrix": rotation},
            os.path.join(str(tmp_path), "rotation.safetensors"),
        )

        base = _spec()
        updated = load_subspace_onto_spec(base, str(tmp_path), "pca", K)
        assert base.fsite.featurizer.is_trivial()
        assert not updated.fsite.featurizer.is_trivial()
        assert updated.fsite.featurizer.n_features == K

    def test_missing_pca_artifact_returns_spec_unchanged(self, tmp_path) -> None:
        from causalab.analyses.subspace import load_subspace_onto_spec

        base = _spec()
        updated = load_subspace_onto_spec(base, str(tmp_path), "pca", K)
        assert updated is base

    def test_das_branch_reads_bundle_by_spec_position_name(self, tmp_path) -> None:
        from causalab.analyses.subspace import load_subspace_onto_spec

        trained = _spec().with_featurizer(_subspace_featurizer())
        bundle_dir = os.path.join(str(tmp_path), "das", "models", "2__last")
        save_site_specs([trained], bundle_dir)

        base = _spec()
        updated = load_subspace_onto_spec(base, str(tmp_path), "das", K, layer=2)
        assert not updated.fsite.featurizer.is_trivial()
        assert updated.fsite.featurizer.n_features == K
