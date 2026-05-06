"""
Tests for dual manifold viewer integration with evaluate stage.

Run with:
    pytest -q tests/test_experiments/test_dual_manifold.py
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from sklearn.decomposition import PCA

from causalab.methods.spline.manifold import SplineManifold


def _make_simple_manifold(
    n_centroids: int = 7, intrinsic_dim: int = 1, ambient_dim: int = 8
):
    """Build a simple SplineManifold for testing (no periodicity)."""
    control_points = torch.linspace(0, 2 * math.pi, n_centroids).unsqueeze(1)
    if intrinsic_dim == 2:
        cp1 = torch.linspace(0, 2 * math.pi, n_centroids)
        cp2 = torch.linspace(0, 2 * math.pi, n_centroids)
        g1, g2 = torch.meshgrid(cp1, cp2, indexing="ij")
        control_points = torch.stack([g1.flatten(), g2.flatten()], dim=-1)
        n_centroids = control_points.shape[0]
    target_points = torch.randn(n_centroids, ambient_dim)
    return SplineManifold(
        control_points=control_points,
        target_points=target_points,
        intrinsic_dim=intrinsic_dim,
        ambient_dim=ambient_dim,
    )


def _make_hellinger_pca(W: int = 7, n_components: int = 3, n_samples: int = 100):
    """Build a fitted sklearn PCA that acts as the Hellinger PCA."""
    rng = np.random.RandomState(42)
    sqrt_dists = rng.dirichlet(np.ones(W + 1), size=n_samples)
    sqrt_dists = np.sqrt(sqrt_dists)
    pca = PCA(n_components=n_components)
    pca.fit(sqrt_dists)
    return pca


def _make_test_artifacts(n_classes=7, k=8, n_pairs=3, n_steps=15, n_prompts=5, W=7):
    """Build a complete set of test artifacts for from_evaluate_artifacts."""
    manifold = _make_simple_manifold(n_centroids=n_classes, ambient_dim=k)
    feat_mean = torch.zeros(k)
    feat_std = torch.ones(k)
    pca_features = torch.randn(100, k)
    hellinger_pca = _make_hellinger_pca(W=W)
    natural_dists = torch.rand(100, W + 1)
    natural_dists = natural_dists / natural_dists.sum(-1, keepdim=True)

    geo_grid_points = torch.linspace(0, 2 * math.pi, n_pairs * n_steps).reshape(
        n_pairs, n_steps, 1
    )
    geo_distributions = torch.rand(n_pairs, n_steps, n_prompts, W)
    geo_distributions = geo_distributions / geo_distributions.sum(-1, keepdim=True)

    pairs = [(i, i + 1) for i in range(n_pairs)]
    class_labels = [str(i) for i in range(n_classes)]

    feature_classes = np.random.randint(0, n_classes, size=100)
    bel_class_assignments_true = np.random.randint(0, n_classes, size=100)

    return dict(
        geo_grid_points=geo_grid_points,
        geo_distributions=geo_distributions,
        lin_grid_points=None,
        lin_distributions=None,
        pairs=pairs,
        manifold_obj=manifold,
        feat_mean=feat_mean,
        feat_std=feat_std,
        pca_features=pca_features,
        subspace_featurizer=None,
        feature_classes=feature_classes,
        belief_manifold=None,
        hellinger_pca=hellinger_pca,
        natural_dists=natural_dists,
        class_labels=class_labels,
        bel_class_assignments_true=bel_class_assignments_true,
    )


class TestFromEvaluateArtifacts:
    """from_evaluate_artifacts should produce correctly shaped per-pair data."""

    def test_basic_shapes(self):
        from causalab.io.plots.dual_manifold import DualManifoldData

        n_pairs, n_steps, n_classes = 3, 15, 7
        artifacts = _make_test_artifacts(
            n_classes=n_classes,
            n_pairs=n_pairs,
            n_steps=n_steps,
        )
        data = DualManifoldData.from_evaluate_artifacts(**artifacts)

        assert len(data.pairs) == n_pairs
        assert len(data.geo_act_paths_3d) == n_pairs
        assert len(data.geo_bel_paths_3d) == n_pairs
        for i in range(n_pairs):
            assert data.geo_act_paths_3d[i].shape == (n_steps, 3)
            assert data.geo_bel_paths_3d[i].shape == (n_steps, 3)
        assert data.act_centroids_3d.shape == (n_classes, 3)
        assert data.bel_centroids_3d.shape == (n_classes, 3)
        assert data.n_classes == n_classes

    def test_belief_paths_are_actual_distributions_not_tps(self):
        """bel_paths_3d should reflect actual decoded distributions, not TPS fit."""
        from causalab.io.plots.dual_manifold import DualManifoldData

        artifacts = _make_test_artifacts(n_pairs=2, n_steps=10, n_prompts=3, W=5)
        data = DualManifoldData.from_evaluate_artifacts(**artifacts)

        # Manual computation for first pair
        hellinger_pca = artifacts["hellinger_pca"]
        mean_dists = artifacts["geo_distributions"][0].mean(dim=1).numpy()
        other = np.clip(1.0 - mean_dists.sum(axis=-1, keepdims=True), 0, None)
        full = np.concatenate([mean_dists, other], axis=-1)
        expected_3d = hellinger_pca.transform(
            np.sqrt(np.clip(full, 0, None)).astype(np.float32)
        )

        np.testing.assert_allclose(data.geo_bel_paths_3d[0], expected_3d, atol=1e-5)

    def test_tps_curve_separate_from_paths(self):
        """When belief_manifold is provided, TPS curve should be stored separately."""
        from causalab.io.plots.dual_manifold import DualManifoldData

        artifacts = _make_test_artifacts(n_classes=5, k=8, n_pairs=2, n_steps=10, W=5)
        # Create a simple belief manifold
        bel_manifold = _make_simple_manifold(n_centroids=5, ambient_dim=6)
        artifacts["belief_manifold"] = bel_manifold

        data = DualManifoldData.from_evaluate_artifacts(**artifacts)

        # TPS curve should exist and be separate from paths
        assert data.bel_tps_curve_3d is not None
        assert data.bel_tps_curve_3d.shape[1] == 3
        # Paths should still be from actual distributions
        assert len(data.geo_bel_paths_3d) == 2

    def test_activation_paths_decode_through_manifold(self):
        """Activation paths should be manifold-decoded, not raw intrinsic coords."""
        from causalab.io.plots.dual_manifold import DualManifoldData

        artifacts = _make_test_artifacts(n_pairs=2, n_steps=10)
        data = DualManifoldData.from_evaluate_artifacts(**artifacts)

        # Total path points should equal n_pairs * n_steps
        assert data.geo_act_paths_3d[0].shape[0] == 10
        assert data.geo_act_paths_3d[1].shape[0] == 10


class TestSaveDualManifoldHtml:
    """save_dual_manifold_html should write a valid HTML file."""

    def _make_dummy_data(self):
        from causalab.io.plots.dual_manifold import DualManifoldData

        n_classes, n_steps, n_bg, n_train, W = 3, 10, 50, 80, 3
        return DualManifoldData(
            pairs=[(0, 1), (1, 2)],
            geo_act_paths_3d=[np.random.randn(n_steps, 3) for _ in range(2)],
            geo_bel_paths_3d=[np.random.randn(n_steps, 3) for _ in range(2)],
            geo_dists=[np.random.dirichlet(np.ones(W), n_steps) for _ in range(2)],
            lin_act_paths_3d=[np.random.randn(n_steps, 3) for _ in range(2)],
            lin_bel_paths_3d=[np.random.randn(n_steps, 3) for _ in range(2)],
            lin_dists=[np.random.dirichlet(np.ones(W), n_steps) for _ in range(2)],
            act_mesh_traces=[],
            bel_tps_curve_3d=np.random.randn(20, 3),
            act_centroids_3d=np.random.randn(n_classes, 3),
            bel_centroids_3d=np.random.randn(n_classes, 3),
            act_features_3d=np.random.randn(n_train, 3),
            act_feature_classes=np.random.randint(0, n_classes, size=n_train),
            bel_background_3d=np.random.randn(n_bg, 3),
            bel_class_assignments=np.random.randint(0, n_classes, size=n_bg),
            n_classes=n_classes,
            class_labels=["A", "B", "C"],
        )

    def test_produces_valid_html(self, tmp_path):
        from causalab.io.plots.dual_manifold import save_dual_manifold_html

        data = self._make_dummy_data()
        out = str(tmp_path / "test.html")
        save_dual_manifold_html(data, out)
        assert os.path.isfile(out)
        content = open(out).read()
        assert "Plotly" in content
        assert "act-plot" in content
        assert "bel-plot" in content
        assert "step-slider" in content
        assert "pair-select" in content

    def test_html_contains_labels_and_font(self, tmp_path):
        from causalab.io.plots.dual_manifold import save_dual_manifold_html

        data = self._make_dummy_data()
        out = str(tmp_path / "test.html")
        save_dual_manifold_html(data, out, colormap="rainbow")
        content = open(out).read()
        assert "A" in content
        assert "B" in content
        assert "Avenir" in content
