"""Tests for distances.py — cyclic costs, displacement weights, Hellinger, Wasserstein."""

from __future__ import annotations

import pytest
import torch

from causalab.methods.distances import (
    _get_cyclic_costs,
    cyclic_displacement_weights,
    hellinger,
    wasserstein1_cyclic,
    wasserstein2_cyclic,
)


# ---------------------------------------------------------------------------
# _get_cyclic_costs
# ---------------------------------------------------------------------------


class TestGetCyclicCosts:
    def test_shapes(self):
        cost_sq, cost, displacement = _get_cyclic_costs(5)
        assert cost_sq.shape == (5, 5)
        assert cost.shape == (5, 5)
        assert displacement.shape == (5, 5)

    def test_diagonal_zero(self):
        cost_sq, cost, displacement = _get_cyclic_costs(6)
        for mat in (cost_sq, cost, displacement):
            assert torch.allclose(mat.diag(), torch.zeros(6))

    def test_cost_symmetry(self):
        cost_sq, cost, _ = _get_cyclic_costs(7)
        assert torch.allclose(cost_sq, cost_sq.T)
        assert torch.allclose(cost, cost.T)

    def test_displacement_antisymmetry(self):
        """displacement[i,j] = -displacement[j,i]."""
        _, _, displacement = _get_cyclic_costs(7)
        assert torch.allclose(displacement, -displacement.T)

    def test_max_distance_even(self):
        n = 8
        _, cost, _ = _get_cyclic_costs(n)
        assert cost.max().item() == n // 2

    def test_max_distance_odd(self):
        n = 7
        _, cost, _ = _get_cyclic_costs(n)
        assert cost.max().item() == n // 2  # floor(7/2) = 3

    def test_cost_equals_sqrt_cost_sq(self):
        cost_sq, cost, _ = _get_cyclic_costs(6)
        assert torch.allclose(cost, cost_sq.sqrt())

    def test_known_values_n4(self):
        """n=4 cycle: 0-1-2-3-0. Distances: d(0,2)=2, d(0,1)=1, d(1,3)=2."""
        cost_sq, cost, displacement = _get_cyclic_costs(4)
        assert cost[0, 1].item() == 1.0
        assert cost[0, 2].item() == 2.0
        assert cost[0, 3].item() == 1.0
        assert cost[1, 3].item() == 2.0
        # Displacement: 0->1 = +1, 0->3 = -1
        assert displacement[0, 1].item() == 1.0
        assert displacement[0, 3].item() == -1.0

    def test_lru_caching(self):
        """Calling twice returns the same objects."""
        a = _get_cyclic_costs(10)
        b = _get_cyclic_costs(10)
        assert a[0] is b[0]
        assert a[1] is b[1]
        assert a[2] is b[2]


# ---------------------------------------------------------------------------
# cyclic_displacement_weights
# ---------------------------------------------------------------------------


class TestCyclicDisplacementWeights:
    def test_linear_interp_integer_position(self):
        """Integer position → all weight on one bin."""
        pos = torch.tensor(2.0)
        w = cyclic_displacement_weights(pos, sigma=None, W=5)
        assert w.shape == (5,)
        assert w[2].item() == pytest.approx(1.0, abs=1e-10)

    def test_linear_interp_midpoint(self):
        """Position 1.5 → equal weight on bins 1 and 2."""
        pos = torch.tensor(1.5)
        w = cyclic_displacement_weights(pos, sigma=None, W=5)
        assert w[1].item() == pytest.approx(0.5, abs=1e-10)
        assert w[2].item() == pytest.approx(0.5, abs=1e-10)

    def test_linear_interp_wrapping(self):
        """Position near W boundary wraps around."""
        pos = torch.tensor(4.5)
        w = cyclic_displacement_weights(pos, sigma=None, W=5)
        assert w[4].item() == pytest.approx(0.5, abs=1e-10)
        assert w[0].item() == pytest.approx(0.5, abs=1e-10)

    def test_linear_interp_sums_to_one(self):
        pos = torch.tensor(2.7)
        w = cyclic_displacement_weights(pos, sigma=None, W=7)
        assert w.sum().item() == pytest.approx(1.0, abs=1e-10)

    def test_vonmises_sums_to_one(self):
        pos = torch.tensor(3.2)
        w = cyclic_displacement_weights(pos, sigma=0.5, W=7)
        assert w.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_vonmises_peaks_at_position(self):
        pos = torch.tensor(3.0)
        w = cyclic_displacement_weights(pos, sigma=0.3, W=7)
        assert w.argmax().item() == 3

    def test_vonmises_non_negative(self):
        pos = torch.tensor(1.0)
        w = cyclic_displacement_weights(pos, sigma=1.0, W=5)
        assert (w >= 0).all()

    def test_batch_shape(self):
        pos = torch.tensor([0.0, 1.5, 3.0])
        w = cyclic_displacement_weights(pos, sigma=0.5, W=5)
        assert w.shape == (3, 5)
        # Each row sums to 1
        assert torch.allclose(w.sum(-1), torch.ones(3, dtype=torch.float64), atol=1e-6)


# ---------------------------------------------------------------------------
# hellinger
# ---------------------------------------------------------------------------


class TestHellinger:
    def test_identical(self):
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert hellinger(p, p).item() == pytest.approx(0.0, abs=1e-6)

    def test_disjoint_support(self):
        p = torch.tensor([1.0, 0.0, 0.0])
        q = torch.tensor([0.0, 1.0, 0.0])
        assert hellinger(p, q).item() == pytest.approx(1.0, abs=1e-5)

    def test_symmetry(self):
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.1, 0.3, 0.6])
        assert hellinger(p, q).item() == pytest.approx(hellinger(q, p).item(), abs=1e-6)

    def test_range(self):
        """Hellinger distance is in [0, 1]."""
        p = torch.tensor([0.9, 0.05, 0.05])
        q = torch.tensor([0.05, 0.9, 0.05])
        d = hellinger(p, q).item()
        assert 0.0 <= d <= 1.0 + 1e-6

    def test_triangle_inequality(self):
        p = torch.tensor([0.6, 0.3, 0.1])
        q = torch.tensor([0.1, 0.6, 0.3])
        r = torch.tensor([0.3, 0.1, 0.6])
        d_pq = hellinger(p, q).item()
        d_qr = hellinger(q, r).item()
        d_pr = hellinger(p, r).item()
        assert d_pr <= d_pq + d_qr + 1e-6

    def test_batch_shape(self):
        p = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
        q = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
        d = hellinger(p, q)
        assert d.shape == (2,)
        assert d[0].item() == pytest.approx(0.0, abs=1e-6)
        assert d[1].item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# wasserstein1_cyclic
# ---------------------------------------------------------------------------


class TestWasserstein1Cyclic:
    def test_identical(self):
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert wasserstein1_cyclic(p, p).item() == pytest.approx(0.0, abs=1e-5)

    def test_symmetry(self):
        p = torch.tensor([0.7, 0.2, 0.05, 0.05])
        q = torch.tensor([0.1, 0.3, 0.3, 0.3])
        assert wasserstein1_cyclic(p, q).item() == pytest.approx(
            wasserstein1_cyclic(q, p).item(), abs=1e-5
        )

    def test_triangle_inequality(self):
        p = torch.tensor([0.6, 0.3, 0.05, 0.05])
        q = torch.tensor([0.1, 0.6, 0.2, 0.1])
        r = torch.tensor([0.1, 0.1, 0.3, 0.5])
        d_pq = wasserstein1_cyclic(p, q).item()
        d_qr = wasserstein1_cyclic(q, r).item()
        d_pr = wasserstein1_cyclic(p, r).item()
        assert d_pr <= d_pq + d_qr + 1e-5

    def test_cyclic_awareness(self):
        """On a 7-cycle, delta(0) vs delta(6) should have distance 1 (shortest path wraps)."""
        p = torch.zeros(7)
        p[0] = 1.0
        q = torch.zeros(7)
        q[6] = 1.0
        d = wasserstein1_cyclic(p, q).item()
        assert d == pytest.approx(1.0, abs=1e-5)

    def test_batch_shape(self):
        p = torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
        q = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
        d = wasserstein1_cyclic(p, q)
        assert d.shape == (2,)
        assert d[0].item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# wasserstein2_cyclic
# ---------------------------------------------------------------------------


class TestWasserstein2Cyclic:
    def test_identical(self):
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert wasserstein2_cyclic(p, p).item() == pytest.approx(0.0, abs=1e-5)

    def test_symmetry(self):
        p = torch.tensor([0.7, 0.2, 0.05, 0.05])
        q = torch.tensor([0.1, 0.3, 0.3, 0.3])
        assert wasserstein2_cyclic(p, q).item() == pytest.approx(
            wasserstein2_cyclic(q, p).item(), abs=1e-5
        )

    def test_triangle_inequality(self):
        p = torch.tensor([0.6, 0.3, 0.05, 0.05])
        q = torch.tensor([0.1, 0.6, 0.2, 0.1])
        r = torch.tensor([0.1, 0.1, 0.3, 0.5])
        d_pq = wasserstein2_cyclic(p, q).item()
        d_qr = wasserstein2_cyclic(q, r).item()
        d_pr = wasserstein2_cyclic(p, r).item()
        assert d_pr <= d_pq + d_qr + 1e-5

    def test_cyclic_awareness(self):
        """On a 7-cycle, delta(0) vs delta(6) should have distance 1."""
        p = torch.zeros(7)
        p[0] = 1.0
        q = torch.zeros(7)
        q[6] = 1.0
        d = wasserstein2_cyclic(p, q).item()
        assert d == pytest.approx(1.0, abs=1e-5)

    def test_batch_shape(self):
        p = torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
        q = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
        d = wasserstein2_cyclic(p, q)
        assert d.shape == (2,)
        assert d[0].item() == pytest.approx(0.0, abs=1e-5)
