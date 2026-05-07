"""Tests for isometry.py."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from torch import Tensor

from omegaconf import OmegaConf

from causalab.methods.scores.isometry import (
    _build_variable_value_hover,
    _decoded_path_length,
    _save_isometry_artifacts,
    compute_isometry_from_manifolds,
    compute_isometry_metrics,
    visualize_isometry,
)
from causalab.methods.distances import (
    _fit_gaussian_params,
    euclidean_log_prob as euclidean_log_prob_distance,
    fisher_rao as fisher_rao_distance,
    fisher_rao_gaussian as fisher_rao_gaussian_distance,
    pairwise_output_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class IdentityManifold:
    """Trivial manifold where decode is the identity function."""

    def __init__(
        self,
        control_points: Tensor | None = None,
        periodic_dims: list[int] | None = None,
        periods: list[float] | None = None,
    ):
        self.control_points = (
            control_points if control_points is not None else torch.zeros(0, 1)
        )
        self._periodic_dims = periodic_dims or []
        self._periods = periods or []

    @property
    def periodic_dims(self) -> list[int]:
        return self._periodic_dims

    @property
    def periods(self) -> list[float]:
        return list(self._periods)

    def decode(self, u: Tensor, r: Tensor | None = None) -> Tensor:
        return u

    def parameters(self):
        return iter([])

    def buffers(self):
        return iter([])


class ScalingManifold:
    """Manifold that scales each dimension differently."""

    def __init__(
        self,
        scales: Tensor,
        control_points: Tensor | None = None,
        periodic_dims: list[int] | None = None,
        periods: list[float] | None = None,
    ):
        self.scales = scales  # (n,)
        self.control_points = (
            control_points
            if control_points is not None
            else torch.zeros(0, scales.shape[0])
        )
        self._periodic_dims = periodic_dims or []
        self._periods = periods or []

    @property
    def periodic_dims(self) -> list[int]:
        return self._periodic_dims

    @property
    def periods(self) -> list[float]:
        return list(self._periods)

    def decode(self, u: Tensor, r: Tensor | None = None) -> Tensor:
        return u * self.scales

    def parameters(self):
        return iter([])

    def buffers(self):
        return iter([])


# ---------------------------------------------------------------------------
# fisher_rao_distance
# ---------------------------------------------------------------------------


class TestFisherRaoDistance:
    def test_identical_distributions(self):
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        d = fisher_rao_distance(p, p)
        assert d.item() == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_distributions(self):
        """Disjoint support -> maximal distance = pi."""
        p = torch.tensor([1.0, 0.0, 0.0])
        q = torch.tensor([0.0, 1.0, 0.0])
        d = fisher_rao_distance(p, q)
        assert d.item() == pytest.approx(np.pi, abs=1e-5)

    def test_symmetry(self):
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.1, 0.3, 0.6])
        assert fisher_rao_distance(p, q).item() == pytest.approx(
            fisher_rao_distance(q, p).item(), abs=1e-6
        )

    def test_batch(self):
        p = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
        q = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
        d = fisher_rao_distance(p, q)
        assert d.shape == (2,)
        assert d[0].item() == pytest.approx(0.0, abs=1e-6)
        assert d[1].item() == pytest.approx(np.pi, abs=1e-5)

    def test_triangle_inequality(self):
        p = torch.tensor([0.6, 0.3, 0.1])
        q = torch.tensor([0.1, 0.6, 0.3])
        r = torch.tensor([0.3, 0.1, 0.6])
        d_pq = fisher_rao_distance(p, q).item()
        d_qr = fisher_rao_distance(q, r).item()
        d_pr = fisher_rao_distance(p, r).item()
        assert d_pr <= d_pq + d_qr + 1e-6


# ---------------------------------------------------------------------------
# fisher_rao_gaussian_distance
# ---------------------------------------------------------------------------


class TestFitGaussianParams:
    def test_delta_distribution(self):
        probs = torch.zeros(1, 5)
        probs[0, 2] = 1.0
        mu, sigma = _fit_gaussian_params(probs)
        assert mu.item() == pytest.approx(2.0, abs=1e-4)
        assert sigma.item() < 0.01

    def test_uniform_distribution(self):
        probs = torch.ones(1, 5) / 5.0
        mu, sigma = _fit_gaussian_params(probs)
        assert mu.item() == pytest.approx(2.0, abs=1e-4)
        expected_var = sum((k - 2.0) ** 2 for k in range(5)) / 5.0
        assert sigma.item() == pytest.approx(expected_var**0.5, abs=1e-3)

    def test_custom_bin_positions(self):
        probs = torch.zeros(1, 3)
        probs[0, 1] = 1.0
        bins = torch.tensor([10.0, 20.0, 30.0])
        mu, sigma = _fit_gaussian_params(probs, bin_positions=bins)
        assert mu.item() == pytest.approx(20.0, abs=1e-4)

    def test_batch(self):
        probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        mu, sigma = _fit_gaussian_params(probs)
        assert mu.shape == (2,)
        assert mu[0].item() == pytest.approx(0.0, abs=1e-4)
        assert mu[1].item() == pytest.approx(2.0, abs=1e-4)


class TestFisherRaoGaussianDistance:
    def test_identical(self):
        d = fisher_rao_gaussian_distance(
            torch.tensor(5.0),
            torch.tensor(1.0),
            torch.tensor(5.0),
            torch.tensor(1.0),
        )
        assert d.item() == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        mu1, s1 = torch.tensor(2.0), torch.tensor(1.0)
        mu2, s2 = torch.tensor(5.0), torch.tensor(2.0)
        d12 = fisher_rao_gaussian_distance(mu1, s1, mu2, s2)
        d21 = fisher_rao_gaussian_distance(mu2, s2, mu1, s1)
        assert d12.item() == pytest.approx(d21.item(), abs=1e-6)

    def test_pure_sigma_change(self):
        mu = torch.tensor(0.0)
        s1 = torch.tensor(1.0)
        s2 = torch.tensor(2.0)
        d = fisher_rao_gaussian_distance(mu, s1, mu, s2)
        expected = (2.0**0.5) * abs(np.log(2.0))
        assert d.item() == pytest.approx(expected, abs=1e-4)

    def test_triangle_inequality(self):
        mu1, s1 = torch.tensor(0.0), torch.tensor(1.0)
        mu2, s2 = torch.tensor(3.0), torch.tensor(2.0)
        mu3, s3 = torch.tensor(6.0), torch.tensor(0.5)
        d12 = fisher_rao_gaussian_distance(mu1, s1, mu2, s2).item()
        d23 = fisher_rao_gaussian_distance(mu2, s2, mu3, s3).item()
        d13 = fisher_rao_gaussian_distance(mu1, s1, mu3, s3).item()
        assert d13 <= d12 + d23 + 1e-6

    def test_batch(self):
        mu1 = torch.tensor([0.0, 0.0])
        s1 = torch.tensor([1.0, 1.0])
        mu2 = torch.tensor([0.0, 5.0])
        s2 = torch.tensor([1.0, 1.0])
        d = fisher_rao_gaussian_distance(mu1, s1, mu2, s2)
        assert d.shape == (2,)
        assert d[0].item() == pytest.approx(0.0, abs=1e-6)
        assert d[1].item() > 0

    def test_larger_mu_separation_gives_larger_distance(self):
        s = torch.tensor(1.0)
        d_small = fisher_rao_gaussian_distance(
            torch.tensor(0.0),
            s,
            torch.tensor(1.0),
            s,
        )
        d_large = fisher_rao_gaussian_distance(
            torch.tensor(0.0),
            s,
            torch.tensor(5.0),
            s,
        )
        assert d_large.item() > d_small.item()

    def test_small_sigma_amplifies_mu_distance(self):
        mu1, mu2 = torch.tensor(0.0), torch.tensor(1.0)
        d_large_sigma = fisher_rao_gaussian_distance(
            mu1,
            torch.tensor(10.0),
            mu2,
            torch.tensor(10.0),
        )
        d_small_sigma = fisher_rao_gaussian_distance(
            mu1,
            torch.tensor(0.1),
            mu2,
            torch.tensor(0.1),
        )
        assert d_small_sigma.item() > d_large_sigma.item()


# ---------------------------------------------------------------------------
# euclidean_log_prob_distance
# ---------------------------------------------------------------------------


class TestEuclideanLogProbDistance:
    def test_identical_distributions(self):
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        d = euclidean_log_prob_distance(p, p)
        assert d.item() == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.1, 0.3, 0.6])
        assert euclidean_log_prob_distance(p, q).item() == pytest.approx(
            euclidean_log_prob_distance(q, p).item(), abs=1e-6
        )

    def test_batch(self):
        p = torch.tensor([[0.5, 0.5], [0.9, 0.1]])
        q = torch.tensor([[0.5, 0.5], [0.1, 0.9]])
        d = euclidean_log_prob_distance(p, q)
        assert d.shape == (2,)
        assert d[0].item() == pytest.approx(0.0, abs=1e-6)
        assert d[1].item() > 0

    def test_triangle_inequality(self):
        p = torch.tensor([0.6, 0.3, 0.1])
        q = torch.tensor([0.1, 0.6, 0.3])
        r = torch.tensor([0.3, 0.1, 0.6])
        d_pq = euclidean_log_prob_distance(p, q).item()
        d_qr = euclidean_log_prob_distance(q, r).item()
        d_pr = euclidean_log_prob_distance(p, r).item()
        assert d_pr <= d_pq + d_qr + 1e-6

    def test_near_zero_probs_handled(self):
        p = torch.tensor([1e-15, 1.0 - 1e-15])
        q = torch.tensor([0.5, 0.5])
        d = euclidean_log_prob_distance(p, q)
        assert torch.isfinite(d).all()


# ---------------------------------------------------------------------------
# pairwise_output_distance
# ---------------------------------------------------------------------------


class TestPairwiseOutputDistance:
    def test_log_prob_metric(self):
        dists = torch.tensor([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
        D = pairwise_output_distance(dists, metric="log_prob")
        assert D.shape == (3, 3)
        np.testing.assert_array_almost_equal(np.diag(D), 0.0)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_fisher_rao_metric(self):
        dists = torch.tensor([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
        D = pairwise_output_distance(dists, metric="fisher_rao")
        assert D.shape == (3, 3)
        np.testing.assert_array_almost_equal(np.diag(D), 0.0)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_fisher_rao_gaussian_metric(self):
        dists = torch.tensor([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
        D = pairwise_output_distance(dists, metric="fisher_rao_gaussian")
        assert D.shape == (3, 3)
        np.testing.assert_array_almost_equal(np.diag(D), 0.0)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown distance function"):
            pairwise_output_distance(torch.tensor([[0.5, 0.5]]), metric="bad")

    def test_hellinger_metric(self):
        dists = torch.tensor([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
        D = pairwise_output_distance(dists, metric="hellinger")
        assert D.shape == (3, 3)
        np.testing.assert_array_almost_equal(np.diag(D), 0.0)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_wasserstein1_cyclic_metric(self):
        dists = torch.tensor([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
        D = pairwise_output_distance(dists, metric="wasserstein1_cyclic")
        assert D.shape == (3, 3)
        np.testing.assert_array_almost_equal(np.diag(D), 0.0, decimal=5)

    def test_wasserstein2_cyclic_metric(self):
        dists = torch.tensor([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
        D = pairwise_output_distance(dists, metric="wasserstein2_cyclic")
        assert D.shape == (3, 3)
        np.testing.assert_array_almost_equal(np.diag(D), 0.0, decimal=5)


# ---------------------------------------------------------------------------
# compute_isometry_metrics
# ---------------------------------------------------------------------------


class TestComputeIsometryMetrics:
    def test_perfect_correlation(self):
        rng = np.random.RandomState(42)
        D = np.abs(rng.randn(10, 10))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        metrics = compute_isometry_metrics(D, 3.0 * D)
        assert metrics["pearson_r"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["n_pairs"] == 45  # 10*9/2

    def test_1d_input(self):
        dx = np.array([1.0, 2.0, 3.0, 4.0])
        dy = 2.0 * dx
        metrics = compute_isometry_metrics(dx, dy)
        assert metrics["pearson_r"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["n_pairs"] == 4

    def test_anticorrelation(self):
        dx = np.array([1.0, 2.0, 3.0, 4.0])
        dy = -dx
        metrics = compute_isometry_metrics(dx, dy)
        assert metrics["pearson_r"] == pytest.approx(-1.0, abs=1e-6)

    def test_empty(self):
        D = np.empty((0, 0))
        metrics = compute_isometry_metrics(D, D)
        assert metrics["n_pairs"] == 0
        assert np.isnan(metrics["pearson_r"])

    def test_constant_returns_nan(self):
        dx = np.ones(5)
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = compute_isometry_metrics(dx, dy)
        assert np.isnan(metrics["pearson_r"])
        assert metrics["n_pairs"] == 5


# ---------------------------------------------------------------------------
# _decoded_path_length
# ---------------------------------------------------------------------------


class TestDecodedPathLength:
    def test_identity_euclidean(self):
        m = IdentityManifold()
        u_a = torch.tensor([0.0, 0.0])
        u_b = torch.tensor([3.0, 4.0])
        length = _decoded_path_length(u_a, u_b, m.decode, n_steps=100)
        assert length == pytest.approx(5.0, abs=0.01)

    def test_zero_length(self):
        m = IdentityManifold()
        u = torch.tensor([1.0, 2.0, 3.0])
        length = _decoded_path_length(u, u, m.decode, n_steps=10)
        assert length == pytest.approx(0.0, abs=1e-8)

    def test_scaling_manifold(self):
        scales = torch.tensor([2.0, 1.0])
        m = ScalingManifold(scales)
        u_a = torch.tensor([0.0, 0.0])
        u_b = torch.tensor([1.0, 0.0])
        length = _decoded_path_length(u_a, u_b, m.decode, n_steps=100)
        assert length == pytest.approx(2.0, abs=0.01)

    def test_periodic_takes_shorter_arc(self):
        m = IdentityManifold()
        # Period=10, u_a=1, u_b=9: direct delta=8 > period/2=5, wraps to -2
        u_a = torch.tensor([1.0])
        u_b = torch.tensor([9.0])
        length = _decoded_path_length(
            u_a,
            u_b,
            m.decode,
            n_steps=100,
            periodic_dims=[0],
            periods=[10.0],
        )
        assert length == pytest.approx(2.0, abs=0.05)


# ---------------------------------------------------------------------------
# compute_isometry_from_manifolds
# ---------------------------------------------------------------------------


class TestComputeIsometryFromManifolds:
    def test_perfect_isometry_identity_manifolds(self):
        # Both manifolds are the identity; their geometries are identical
        # up to scaling, so Pearson r should be 1.
        cps = torch.linspace(0, 5, 6).unsqueeze(-1)  # (6, 1)
        act = IdentityManifold(control_points=cps)
        bel = IdentityManifold(control_points=cps)
        mean = torch.zeros(1)
        std = torch.ones(1)
        metrics, D_X, D_Y, grid, _ = compute_isometry_from_manifolds(
            act,
            mean,
            std,
            bel,
            n_arc_steps=20,
        )
        assert metrics["pearson_r"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["n_pairs"] == 15  # 6*5/2
        assert D_X.shape == (6, 6)
        assert D_Y.shape == (6, 6)
        np.testing.assert_array_almost_equal(D_X, D_X.T)
        np.testing.assert_array_almost_equal(np.diag(D_X), 0.0)
        # Belief length is divided by sqrt(2)
        np.testing.assert_array_almost_equal(D_Y * (2.0**0.5), D_X, decimal=4)
        assert grid.shape == (6, 1)

    def test_isometry_with_scaled_belief(self):
        cps = torch.linspace(0, 1, 4).unsqueeze(-1)
        act = IdentityManifold(control_points=cps)
        bel = ScalingManifold(torch.tensor([3.0]), control_points=cps)
        mean = torch.zeros(1)
        std = torch.ones(1)
        metrics, D_X, D_Y, _, _ = compute_isometry_from_manifolds(
            act,
            mean,
            std,
            bel,
            n_arc_steps=20,
        )
        # Both monotone in the same intrinsic distance -> perfect r
        assert metrics["pearson_r"] == pytest.approx(1.0, abs=1e-4)

    def test_different_geometries_lower_correlation(self):
        # Cubic belief vs linear activation: D_Y[i,j] = u_j^3 - u_i^3 depends
        # on (u_a + u_b), not just Δu, so it cannot be perfectly correlated
        # with D_X[i,j] = Δu.
        cps = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
        act = IdentityManifold(control_points=cps)

        class CubicManifold:
            def __init__(self, control_points):
                self.control_points = control_points
                self._periodic_dims = []
                self._periods = []

            @property
            def periodic_dims(self) -> list[int]:
                return self._periodic_dims

            @property
            def periods(self) -> list[float]:
                return list(self._periods)

            def decode(self, u):
                return u**3

            def parameters(self):
                return iter([])

            def buffers(self):
                return iter([])

        bel = CubicManifold(cps)
        mean = torch.zeros(1)
        std = torch.ones(1)
        metrics, _, _, _, _ = compute_isometry_from_manifolds(
            act,
            mean,
            std,
            bel,
            n_arc_steps=50,
        )
        assert metrics["pearson_r"] < 0.95

    def test_mismatched_centroid_count_raises(self):
        act = IdentityManifold(control_points=torch.zeros(3, 1))
        bel = IdentityManifold(control_points=torch.zeros(4, 1))
        with pytest.raises(ValueError, match="cannot align by class index"):
            compute_isometry_from_manifolds(
                act,
                torch.zeros(1),
                torch.ones(1),
                bel,
            )

    def test_periodic_handling(self):
        # Activation has period=10 on dim 0; belief is identity (no wrap).
        # u-points 1 and 9: act sees shortest-arc (length 2), bel sees direct (length 8).
        cps = torch.tensor([[1.0], [9.0]])
        act = IdentityManifold(
            control_points=cps,
            periodic_dims=[0],
            periods=[10.0],
        )
        bel = IdentityManifold(control_points=cps)
        mean = torch.zeros(1)
        std = torch.ones(1)
        _, D_X, D_Y, _, _ = compute_isometry_from_manifolds(
            act,
            mean,
            std,
            bel,
            n_arc_steps=50,
        )
        assert D_X[0, 1] == pytest.approx(2.0, abs=0.1)
        # Belief is non-periodic identity, so it sees the direct delta of 8
        # (divided by sqrt(2) for Hellinger scaling).
        assert D_Y[0, 1] == pytest.approx(8.0 / (2.0**0.5), abs=0.1)


# ---------------------------------------------------------------------------
# Vertex-set parity across path_modes (Option X — same manifold-derived
# vertices for both geometric and linear; modes differ only in distance metric).
# ---------------------------------------------------------------------------


class _CurvedManifold:
    """1D manifold whose decode is u -> (u, u^2). Control points lie on the parabola.

    Used to construct a manifold where the geodesic arc length and the chord
    differ in a known way, so we can pin both metrics against hand-computed values.
    """

    def __init__(
        self,
        control_points: Tensor,
        periodic_dims: list[int] | None = None,
        periods: list[float] | None = None,
    ):
        self.control_points = control_points
        self._periodic_dims = periodic_dims or []
        self._periods = periods or []

    @property
    def periodic_dims(self) -> list[int]:
        return self._periodic_dims

    @property
    def periods(self) -> list[float]:
        return list(self._periods)

    def decode(self, u: Tensor, r: Tensor | None = None) -> Tensor:
        # u: (..., 1) -> (..., 2): (u, u^2)
        u_col = u[..., 0:1]
        return torch.cat([u_col, u_col**2], dim=-1)

    def parameters(self):
        return iter([])

    def buffers(self):
        return iter([])


class TestIsometryVertexParity:
    """Both geometric and linear use the same manifold-derived vertex set."""

    def test_centroid_baseline_linear_chord_equals_pca_distance(self):
        # On an Identity manifold, act_decode(u_i) = u_i, so the linear chord
        # between centroids equals u_j - u_i (in 1D). Same number as the
        # arc length under geometric mode (since the manifold is flat here).
        cps = torch.tensor([[0.0], [1.0], [2.5], [4.0]])
        act = IdentityManifold(control_points=cps)
        bel = IdentityManifold(control_points=cps)
        mean = torch.zeros(1)
        std = torch.ones(1)

        _, D_lin, _, _, _ = compute_isometry_from_manifolds(
            act,
            mean,
            std,
            bel,
            n_arc_steps=50,
            path_mode="linear",
        )
        _, D_geo, _, _, _ = compute_isometry_from_manifolds(
            act,
            mean,
            std,
            bel,
            n_arc_steps=50,
            path_mode="geometric",
        )
        # Hand-computed chord distances (same as arc length on flat manifold).
        expected = torch.tensor(
            [
                [0.0, 1.0, 2.5, 4.0],
                [1.0, 0.0, 1.5, 3.0],
                [2.5, 1.5, 0.0, 1.5],
                [4.0, 3.0, 1.5, 0.0],
            ]
        ).numpy()
        np.testing.assert_array_almost_equal(D_lin, expected, decimal=4)
        np.testing.assert_array_almost_equal(D_geo, expected, decimal=2)

    def test_linear_interior_uses_manifold_decoded_points(self):
        # On the curved manifold u -> (u, u^2), an interior vertex on geodesic
        # (i, j) at fraction f sits at (u_i + f*Δu, (u_i + f*Δu)^2). The linear
        # chord between two such vertices is hand-computable and is NOT equal
        # to the chord between the raw control points (which is what the OLD
        # code would have computed). This test locks "vertices are manifold-
        # derived, not chord-of-raw-centroids."
        cps = torch.tensor([[0.0], [1.0], [2.0]])
        act = _CurvedManifold(control_points=cps)
        bel = IdentityManifold(control_points=cps)
        mean = torch.zeros(2)
        std = torch.ones(2)

        K = 2
        _, D_lin, _, _, _ = compute_isometry_from_manifolds(
            act,
            mean,
            std,
            bel,
            n_arc_steps=50,
            path_mode="linear",
            n_interior_per_pair=K,
        )

        # Vertex layout: 3 centroids first (W=3), then K=2 interior points per
        # ordered pair (0,1), (0,2), (1,2). Fractions = [1/3, 2/3].
        # Build the expected manifold-decoded vertex coords by hand.
        def decoded(u: float) -> tuple[float, float]:
            return (u, u * u)

        verts: list[tuple[float, float]] = []
        for u_val in [0.0, 1.0, 2.0]:
            verts.append(decoded(u_val))
        for ui, uj in [(0.0, 1.0), (0.0, 2.0), (1.0, 2.0)]:
            for f in [1.0 / 3.0, 2.0 / 3.0]:
                verts.append(decoded(ui + f * (uj - ui)))

        # Expected D_X = pairwise Euclidean over those decoded points.
        V = len(verts)
        expected = np.zeros((V, V))
        for a in range(V):
            for b in range(a + 1, V):
                dx = verts[a][0] - verts[b][0]
                dy = verts[a][1] - verts[b][1]
                d = (dx * dx + dy * dy) ** 0.5
                expected[a, b] = expected[b, a] = d

        assert D_lin.shape == (V, V)
        np.testing.assert_array_almost_equal(D_lin, expected, decimal=5)

        # Sanity: the chord between manifold-decoded interior points differs
        # from a hypothetical chord-of-raw-centroids construction. For (0,1)
        # interior at f=1/3, that chord-of-cps would put the vertex at
        # (1/3, 0) (since cps are 1D); manifold decode puts it at (1/3, 1/9).
        # Distance to centroid 0 = (0, 0): chord-version sqrt(1/9) ≈ 0.333,
        # manifold-version sqrt(1/9 + 1/81) ≈ 0.351. The metric reflects the
        # manifold version.
        d_centroid0_to_first_interior = D_lin[0, 3]  # vertex 3 is (0,1) at f=1/3
        chord_of_cps = 1.0 / 3.0  # what the old code would have computed
        manifold_decoded = ((1.0 / 3.0) ** 2 + (1.0 / 9.0) ** 2) ** 0.5
        assert d_centroid0_to_first_interior == pytest.approx(
            manifold_decoded, abs=1e-5
        )
        assert d_centroid0_to_first_interior > chord_of_cps + 0.01

    def test_geometric_arc_length_geq_linear_chord(self):
        # Triangle inequality / curvature property: arc length on the manifold
        # is at least the straight-line chord between the same two endpoints.
        # On the curved manifold, the inequality is strict for non-zero Δu.
        cps = torch.tensor([[0.0], [1.0], [2.0]])
        act = _CurvedManifold(control_points=cps)
        bel = IdentityManifold(control_points=cps)
        mean = torch.zeros(2)
        std = torch.ones(2)

        _, D_lin, _, _, _ = compute_isometry_from_manifolds(
            act,
            mean,
            std,
            bel,
            n_arc_steps=200,
            path_mode="linear",
            n_interior_per_pair=0,
        )
        _, D_geo, _, _, _ = compute_isometry_from_manifolds(
            act,
            mean,
            std,
            bel,
            n_arc_steps=200,
            path_mode="geometric",
            n_interior_per_pair=0,
        )
        # Off-diagonal: D_geo >= D_lin, strictly greater on the curved manifold.
        for i in range(3):
            for j in range(i + 1, 3):
                assert D_geo[i, j] >= D_lin[i, j] - 1e-5
                assert D_geo[i, j] > D_lin[i, j] + 1e-3

    def test_linear_subspace_no_longer_a_valid_path_mode(self):
        # Option X collapses linear_subspace into linear; the inner function
        # only knows about "geometric" and "linear". Anything else raises.
        cps = torch.tensor([[0.0], [1.0]])
        act = IdentityManifold(control_points=cps)
        bel = IdentityManifold(control_points=cps)
        mean = torch.zeros(1)
        std = torch.ones(1)
        with pytest.raises(ValueError, match="Unknown path_mode"):
            compute_isometry_from_manifolds(
                act,
                mean,
                std,
                bel,
                path_mode="linear_subspace",
            )
        with pytest.raises(ValueError, match="Unknown path_mode"):
            compute_isometry_from_manifolds(
                act,
                mean,
                std,
                bel,
                path_mode="banana",
            )


# ---------------------------------------------------------------------------
# _save_isometry_artifacts
# ---------------------------------------------------------------------------


class TestSaveIsometryArtifacts:
    def test_saves_metrics_and_tensors(self, tmp_path):
        D_X = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        D_Y = 2.0 * D_X
        metrics = compute_isometry_metrics(D_X, D_Y)
        metadata = {"n_arc_steps": 50, "n_centroids": 3}
        out_dir = str(tmp_path / "eval_out")

        _save_isometry_artifacts(metrics, D_X, D_Y, None, out_dir, metadata)

        assert os.path.exists(os.path.join(out_dir, "metrics.json"))
        assert os.path.exists(os.path.join(out_dir, "metadata.json"))
        assert os.path.exists(os.path.join(out_dir, "tensors.safetensors"))

        with open(os.path.join(out_dir, "metrics.json")) as f:
            saved = json.load(f)
        assert "pearson_r" in saved
        assert "nrmsd" not in saved
        assert "alpha_star" not in saved

        from safetensors.torch import load_file

        tensors = load_file(os.path.join(out_dir, "tensors.safetensors"))
        assert "D_manifold" in tensors
        assert "D_output" in tensors
        assert "grid_points_valid" not in tensors
        assert tensors["D_manifold"].shape == (3, 3)

    def test_saves_grid_points_when_provided(self, tmp_path):
        D = np.zeros((2, 2))
        metrics = compute_isometry_metrics(D, D)
        grid = torch.tensor([[0.0], [1.0]])
        out_dir = str(tmp_path / "eval_out2")

        _save_isometry_artifacts(metrics, D, D, grid, out_dir)

        from safetensors.torch import load_file

        tensors = load_file(os.path.join(out_dir, "tensors.safetensors"))
        assert "grid_points_valid" in tensors


# ---------------------------------------------------------------------------
# visualize_isometry
# ---------------------------------------------------------------------------


class TestVisualizeIsometry:
    def _save_test_artifacts(self, artifact_dir, n_points=5):
        """Create and save synthetic artifacts."""
        cps = torch.linspace(0, 2, n_points).unsqueeze(-1)
        act = IdentityManifold(control_points=cps)
        bel = IdentityManifold(control_points=cps)
        metrics, D_X, D_Y, grid, grid_belief = compute_isometry_from_manifolds(
            act,
            torch.zeros(1),
            torch.ones(1),
            bel,
            n_arc_steps=20,
        )
        _save_isometry_artifacts(
            metrics,
            D_X,
            D_Y,
            grid,
            artifact_dir,
            grid_points_belief=grid_belief,
        )
        return metrics

    def test_3d_mds(self, tmp_path):
        artifact_dir = os.path.join(str(tmp_path), "isometry")
        self._save_test_artifacts(artifact_dir)

        viz_cfg = OmegaConf.create(
            {
                "n_mds_components": 3,
                "hover_label_format": "grid_coords",
                "figure_format": "png",
            }
        )
        result = visualize_isometry(
            artifact_dir=artifact_dir,
            viz_cfg=viz_cfg,
            distance_function="hellinger",
        )
        assert os.path.exists(result["scatter"])
        assert os.path.exists(result["mds"])

    def test_2d_mds(self, tmp_path):
        artifact_dir = os.path.join(str(tmp_path), "isometry")
        self._save_test_artifacts(artifact_dir)

        viz_cfg = OmegaConf.create(
            {
                "n_mds_components": 2,
                "hover_label_format": "grid_coords",
                "figure_format": "png",
            }
        )
        result = visualize_isometry(
            artifact_dir=artifact_dir,
            viz_cfg=viz_cfg,
            distance_function="hellinger",
        )
        assert os.path.exists(result["scatter"])
        assert os.path.exists(result["mds"])

    def test_variable_values_hover(self, tmp_path):
        artifact_dir = os.path.join(str(tmp_path), "isometry")
        self._save_test_artifacts(artifact_dir, n_points=7)

        viz_cfg = OmegaConf.create(
            {"n_mds_components": 3, "hover_label_format": "variable_values"}
        )
        result = visualize_isometry(
            artifact_dir=artifact_dir,
            viz_cfg=viz_cfg,
            distance_function="fisher_rao",
            variable_values=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            grid_range=[0, 7],
        )
        assert os.path.exists(result["mds"])


# ---------------------------------------------------------------------------
# _build_variable_value_hover
# ---------------------------------------------------------------------------


class TestBuildVariableValueHover:
    def test_maps_grid_to_values(self):
        grid_points = np.array([[0.0], [1.0], [2.0], [3.0]])
        values = ["Mon", "Tue", "Wed", "Thu"]
        labels = _build_variable_value_hover(grid_points, values, [0, 3])
        assert len(labels) == 4
        assert "Mon" in labels[0]
        assert "Thu" in labels[3]

    def test_includes_coordinates(self):
        grid_points = np.array([[1.5]])
        values = ["A", "B", "C"]
        labels = _build_variable_value_hover(grid_points, values, [0, 2])
        assert "u0=" in labels[0]

    def test_clamps_to_range(self):
        grid_points = np.array([[-1.0], [10.0]])
        values = ["X", "Y", "Z"]
        labels = _build_variable_value_hover(grid_points, values, [0, 2])
        assert "X" in labels[0]
        assert "Z" in labels[1]
