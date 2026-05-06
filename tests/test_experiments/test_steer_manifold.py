"""
Tests for steer_manifold.py

Run with:
    pytest -q tests/test_experiments/test_steer_manifold.py
"""

from __future__ import annotations

import pytest
import torch

from causalab.methods.steer.collect import (
    make_intrinsic_steering_grid,
)
from causalab.methods.steer.collect import (
    construct_steering_vectors,
)


class TestMakeIntrinsicSteeringGrid:
    """Tests for make_intrinsic_steering_grid."""

    def test_dim1_shape(self):
        """d=1 should produce (n_points, 1) tensor."""
        grid = make_intrinsic_steering_grid(intrinsic_dim=1, n_points_per_dim=11)
        assert grid.shape == (11, 1)

    def test_dim1_range(self):
        """d=1 grid values should span the specified range."""
        grid = make_intrinsic_steering_grid(
            intrinsic_dim=1, n_points_per_dim=5, range_min=-2.0, range_max=2.0
        )
        assert grid.min().item() == pytest.approx(-2.0)
        assert grid.max().item() == pytest.approx(2.0)

    def test_dim2_shape(self):
        """d=2 should produce (n^2, 2) meshgrid tensor."""
        grid = make_intrinsic_steering_grid(intrinsic_dim=2, n_points_per_dim=7)
        assert grid.shape == (49, 2)  # 7*7 = 49

    def test_dim2_covers_all_combinations(self):
        """d=2 grid should cover all combinations of coordinates."""
        grid = make_intrinsic_steering_grid(
            intrinsic_dim=2, n_points_per_dim=3, range_min=-1.0, range_max=1.0
        )
        # Should have 9 points: (-1,-1), (-1,0), (-1,1), (0,-1), ..., (1,1)
        assert grid.shape == (9, 2)
        # Each dimension should have 3 unique values
        assert len(torch.unique(grid[:, 0])) == 3
        assert len(torch.unique(grid[:, 1])) == 3

    def test_dim3_meshgrid_shape(self):
        """d>2 should produce full meshgrid, capped to ~512 points."""
        grid = make_intrinsic_steering_grid(intrinsic_dim=3, n_points_per_dim=5)
        assert grid.shape == (125, 3)  # 5^3 = 125 (5 < cap of 8)

    def test_dim3_meshgrid_capped(self):
        """d=3 with n=21 should cap to 8 per dim -> 512 points."""
        grid = make_intrinsic_steering_grid(intrinsic_dim=3, n_points_per_dim=21)
        assert grid.shape == (512, 3)  # 8^3 = 512

    def test_dim3_meshgrid_covers_ranges(self):
        """d>2 full meshgrid should cover all dimension ranges."""
        grid = make_intrinsic_steering_grid(
            intrinsic_dim=3,
            n_points_per_dim=5,
            ranges=((1.0, 10.0), (0.0, 4.0), (0.0, 4.0)),
        )
        assert grid.shape == (125, 3)
        for d in range(3):
            assert len(grid[:, d].unique()) == 5

    def test_default_range(self):
        """Default range should be [-3, 3]."""
        grid = make_intrinsic_steering_grid(intrinsic_dim=1, n_points_per_dim=11)
        assert grid.min().item() == pytest.approx(-3.0)
        assert grid.max().item() == pytest.approx(3.0)


class TestConstructSteeringVectors:
    """Tests for construct_steering_vectors."""

    def test_shape(self):
        """Output should have shape (n_points, total_dim)."""
        grid = torch.randn(10, 2)
        vectors = construct_steering_vectors(grid, intrinsic_dim=2, total_dim=8)
        assert vectors.shape == (10, 8)

    def test_intrinsic_preserved(self):
        """First d columns should match the input grid."""
        grid = torch.randn(10, 3)
        vectors = construct_steering_vectors(grid, intrinsic_dim=3, total_dim=10)
        assert torch.allclose(vectors[:, :3], grid)

    def test_residual_zeros(self):
        """Last (k-d) columns should be zeros."""
        grid = torch.randn(10, 2)
        vectors = construct_steering_vectors(grid, intrinsic_dim=2, total_dim=8)
        assert torch.allclose(vectors[:, 2:], torch.zeros(10, 6))

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        grid = torch.randn(5, 2, dtype=torch.float32)
        vectors = construct_steering_vectors(grid, intrinsic_dim=2, total_dim=4)
        assert vectors.dtype == torch.float32

        grid64 = torch.randn(5, 2, dtype=torch.float64)
        vectors64 = construct_steering_vectors(grid64, intrinsic_dim=2, total_dim=4)
        assert vectors64.dtype == torch.float64

    def test_single_point(self):
        """Should work with a single grid point."""
        grid = torch.tensor([[1.0, 2.0]])
        vectors = construct_steering_vectors(grid, intrinsic_dim=2, total_dim=5)
        expected = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(vectors, expected)

    def test_no_residual(self):
        """Edge case: total_dim == intrinsic_dim (no residual)."""
        grid = torch.randn(5, 3)
        vectors = construct_steering_vectors(grid, intrinsic_dim=3, total_dim=3)
        assert torch.allclose(vectors, grid)


class TestMakeIntrinsicSteeringGridPerDimension:
    """Tests for per-dimension ranges in make_intrinsic_steering_grid."""

    def test_dim1_per_dimension_range(self):
        """d=1 with per-dimension ranges should use the single range."""
        ranges = ((-1.0, 2.0),)
        grid = make_intrinsic_steering_grid(
            intrinsic_dim=1, n_points_per_dim=5, ranges=ranges
        )
        assert grid.shape == (5, 1)
        assert grid.min().item() == pytest.approx(-1.0)
        assert grid.max().item() == pytest.approx(2.0)

    def test_dim2_per_dimension_ranges(self):
        """d=2 with per-dimension ranges should have different ranges per dimension."""
        ranges = ((-1.0, 2.0), (-4.0, 5.0))
        grid = make_intrinsic_steering_grid(
            intrinsic_dim=2, n_points_per_dim=5, ranges=ranges
        )
        assert grid.shape == (25, 2)  # 5*5

        # Check dimension 0 spans its own range
        assert grid[:, 0].min().item() == pytest.approx(-1.0)
        assert grid[:, 0].max().item() == pytest.approx(2.0)

        # Check dimension 1 spans its own range
        assert grid[:, 1].min().item() == pytest.approx(-4.0)
        assert grid[:, 1].max().item() == pytest.approx(5.0)

    def test_dim2_asymmetric_ranges(self):
        """d=2 with asymmetric ranges correctly creates rectangular grid."""
        # One dimension narrow, one wide
        ranges = ((0.0, 1.0), (-10.0, 10.0))
        grid = make_intrinsic_steering_grid(
            intrinsic_dim=2, n_points_per_dim=3, ranges=ranges
        )

        # Verify the unique values in each dimension
        unique_dim0 = torch.unique(grid[:, 0])
        unique_dim1 = torch.unique(grid[:, 1])

        assert len(unique_dim0) == 3
        assert len(unique_dim1) == 3

        # Check actual values
        assert torch.allclose(unique_dim0, torch.tensor([0.0, 0.5, 1.0]))
        assert torch.allclose(unique_dim1, torch.tensor([-10.0, 0.0, 10.0]))

    def test_dim3_per_dimension_ranges(self):
        """d>2 full meshgrid with per-dimension ranges."""
        ranges = ((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0))
        grid = make_intrinsic_steering_grid(
            intrinsic_dim=3, n_points_per_dim=5, ranges=ranges
        )
        assert grid.shape == (125, 3)  # 5^3

        # Each dim should cover its full range
        assert grid[:, 0].min().item() == pytest.approx(-1.0)
        assert grid[:, 0].max().item() == pytest.approx(1.0)
        assert grid[:, 1].min().item() == pytest.approx(-2.0)
        assert grid[:, 1].max().item() == pytest.approx(2.0)
        assert grid[:, 2].min().item() == pytest.approx(-3.0)
        assert grid[:, 2].max().item() == pytest.approx(3.0)

    def test_ranges_overrides_range_min_max(self):
        """ranges parameter should override range_min/range_max."""
        ranges = ((10.0, 20.0),)
        grid = make_intrinsic_steering_grid(
            intrinsic_dim=1,
            n_points_per_dim=5,
            range_min=-100.0,  # Should be ignored
            range_max=100.0,  # Should be ignored
            ranges=ranges,
        )
        assert grid.min().item() == pytest.approx(10.0)
        assert grid.max().item() == pytest.approx(20.0)

    def test_backward_compat_no_ranges(self):
        """Without ranges parameter, should use range_min/range_max."""
        grid = make_intrinsic_steering_grid(
            intrinsic_dim=2, n_points_per_dim=3, range_min=-2.0, range_max=2.0
        )
        # Both dimensions should have the same range
        assert grid[:, 0].min().item() == pytest.approx(-2.0)
        assert grid[:, 0].max().item() == pytest.approx(2.0)
        assert grid[:, 1].min().item() == pytest.approx(-2.0)
        assert grid[:, 1].max().item() == pytest.approx(2.0)

    def test_ranges_wrong_length_raises(self):
        """ranges with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="ranges has 3 entries"):
            make_intrinsic_steering_grid(
                intrinsic_dim=2,
                n_points_per_dim=3,
                ranges=((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)),  # 3 ranges for d=2
            )


class TestMakeAndConstructIntegration:
    """Integration tests combining grid generation and vector construction."""

    def test_dim1_full_pipeline(self):
        """Create grid and construct vectors for d=1."""
        grid = make_intrinsic_steering_grid(intrinsic_dim=1, n_points_per_dim=5)
        vectors = construct_steering_vectors(grid, intrinsic_dim=1, total_dim=8)

        assert grid.shape == (5, 1)
        assert vectors.shape == (5, 8)
        assert torch.allclose(vectors[:, :1], grid)
        assert torch.allclose(vectors[:, 1:], torch.zeros(5, 7))

    def test_dim2_full_pipeline(self):
        """Create grid and construct vectors for d=2."""
        grid = make_intrinsic_steering_grid(intrinsic_dim=2, n_points_per_dim=3)
        vectors = construct_steering_vectors(grid, intrinsic_dim=2, total_dim=10)

        assert grid.shape == (9, 2)  # 3*3
        assert vectors.shape == (9, 10)
        assert torch.allclose(vectors[:, :2], grid)
        assert torch.allclose(vectors[:, 2:], torch.zeros(9, 8))

    def test_dim2_per_dimension_pipeline(self):
        """Create grid with per-dimension ranges and construct vectors."""
        ranges = ((-1.0, 2.0), (-4.0, 5.0))
        grid = make_intrinsic_steering_grid(
            intrinsic_dim=2, n_points_per_dim=3, ranges=ranges
        )
        vectors = construct_steering_vectors(grid, intrinsic_dim=2, total_dim=8)

        assert grid.shape == (9, 2)
        assert vectors.shape == (9, 8)
        assert torch.allclose(vectors[:, :2], grid)
        assert torch.allclose(vectors[:, 2:], torch.zeros(9, 6))
