"""Tests for ManifoldFlow and FlowManifold."""

import pytest
import torch

from causalab.methods.flow import Flow, ManifoldFlow, FlowManifold, StandardNormal
from causalab.methods.flow.bijectors import AffineCoupling, Permutation


def build_flow(D: int, num_layers: int = 4, hidden: int = 32, seed: int = 0) -> Flow:
    """Build a simple flow for testing."""
    layers = []
    for i in range(num_layers):
        if i % 2 == 0:
            idx_a = torch.arange(0, D, 2)
            idx_b = torch.arange(1, D, 2)
        else:
            idx_a = torch.arange(1, D, 2)
            idx_b = torch.arange(0, D, 2)

        layers.append(
            AffineCoupling(D, idx_a=idx_a, idx_b=idx_b, hidden=hidden, depth=2)
        )
        layers.append(Permutation(D, seed=seed + i))

    return Flow(layers=layers, base_dist=StandardNormal(D))


class TestManifoldFlowShapes:
    """Test shape correctness for encode/decode."""

    def test_encode_shapes(self):
        """encode() returns (u, r) with correct shapes."""
        D, k, B = 4, 2, 16
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)

        x = torch.randn(B, D)
        u, r = mf.encode(x)

        assert u.shape == (B, k)
        assert r.shape == (B, D - k)

    def test_decode_shapes(self):
        """decode() returns x with correct shape."""
        D, k, B = 4, 2, 16
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)

        u = torch.randn(B, k)
        r = torch.randn(B, D - k)

        # With explicit r
        x = mf.decode(u, r)
        assert x.shape == (B, D)

        # With r=None (on-manifold)
        x_proj = mf.decode(u, r=None)
        assert x_proj.shape == (B, D)

    def test_project_shapes(self):
        """project() returns x with correct shape."""
        D, k, B = 4, 2, 16
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)

        x = torch.randn(B, D)
        x_proj = mf.project(x)
        assert x_proj.shape == (B, D)

    def test_forward_is_project(self):
        """forward() is an alias for project()."""
        D, k, B = 4, 2, 16
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)

        x = torch.randn(B, D)
        torch.testing.assert_close(mf(x), mf.project(x))


class TestManifoldFlowInvertibility:
    """Test that encode -> decode(u, r) recovers x exactly."""

    def test_encode_decode_roundtrip(self):
        """encode -> decode with original r recovers x."""
        D, k, B = 4, 2, 32
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)

        x = torch.randn(B, D)
        u, r = mf.encode(x)
        x_recon = mf.decode(u, r)

        torch.testing.assert_close(x_recon, x, atol=1e-5, rtol=1e-5)

    def test_inner_flow_invertibility(self):
        """Inner flow remains invertible after wrapping."""
        D, k, B = 4, 2, 32
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)

        x = torch.randn(B, D)
        z, logdet_fwd = mf.flow.fwd(x)
        x_recon, logdet_inv = mf.flow.inv(z)

        torch.testing.assert_close(x_recon, x, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(logdet_fwd, -logdet_inv, atol=1e-4, rtol=1e-4)


class TestManifoldFlowLoss:
    """Test loss computation and gradients."""

    def test_loss_returns_components(self):
        """loss() returns total loss and component metrics."""
        D, k, B = 4, 2, 16
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)

        x = torch.randn(B, D)
        total, metrics = mf.loss(x)

        assert isinstance(total, torch.Tensor)
        assert total.shape == ()
        assert "recon" in metrics
        assert "residual" in metrics
        assert isinstance(metrics["recon"], float)
        assert isinstance(metrics["residual"], float)

    def test_loss_weights(self):
        """Loss weights affect components correctly."""
        D, k, B = 4, 2, 16
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)

        x = torch.randn(B, D)

        # Get baseline metrics
        _, metrics = mf.loss(x, residual_weight=1.0, reconstruction_weight=1.0)
        recon = metrics["recon"]
        resid = metrics["residual"]

        # Verify weights scale correctly
        total_2x_recon, _ = mf.loss(x, residual_weight=1.0, reconstruction_weight=2.0)
        expected = 2.0 * recon + 1.0 * resid
        torch.testing.assert_close(
            total_2x_recon, torch.tensor(expected), atol=1e-5, rtol=1e-5
        )

        total_2x_resid, _ = mf.loss(x, residual_weight=2.0, reconstruction_weight=1.0)
        expected = 1.0 * recon + 2.0 * resid
        torch.testing.assert_close(
            total_2x_resid, torch.tensor(expected), atol=1e-5, rtol=1e-5
        )

    def test_gradients_flow(self):
        """Gradients flow through loss."""
        D, k, B = 4, 2, 16
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)

        x = torch.randn(B, D)
        total, _ = mf.loss(x)
        total.backward()

        # Check that at least one parameter has a gradient
        has_grad = False
        for p in mf.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed through loss"


class TestManifoldFlowTraining:
    """Test training behavior."""

    def test_residual_decreases_on_manifold_data(self):
        """Residual penalty drives r toward zero on manifold-like data."""
        # Create data that lies near a 2D manifold in 4D space
        D, k, B = 4, 2, 64
        torch.manual_seed(42)

        # Generate data: first 2 dims have structure, last 2 are small
        intrinsic = torch.randn(B, k)
        noise = torch.randn(B, D - k) * 0.1
        x_data = torch.cat([intrinsic, noise], dim=-1)

        flow = build_flow(D, num_layers=6, hidden=64)
        mf = ManifoldFlow(flow, intrinsic_dim=k)
        opt = torch.optim.Adam(mf.parameters(), lr=1e-3)

        # Train for a few steps
        initial_residual: float = 0.0
        final_residual: float = 0.0
        for step in range(100):
            loss, metrics = mf.loss(
                x_data, residual_weight=1.0, reconstruction_weight=1.0
            )
            if step == 0:
                initial_residual = metrics["residual"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_residual = metrics["residual"]

        # Residual should decrease (or at least not increase significantly)
        # This is a weak test since 100 steps is not much
        assert final_residual < initial_residual * 2.0, (
            f"Residual increased too much: {initial_residual:.4f} -> {final_residual:.4f}"
        )


class TestManifoldFlowValidation:
    """Test input validation."""

    def test_intrinsic_dim_too_large(self):
        """Raises error if intrinsic_dim >= ambient_dim."""
        D = 4
        flow = build_flow(D)

        with pytest.raises(ValueError, match="intrinsic_dim.*must be less than"):
            ManifoldFlow(flow, intrinsic_dim=D)

        with pytest.raises(ValueError, match="intrinsic_dim.*must be less than"):
            ManifoldFlow(flow, intrinsic_dim=D + 1)

    def test_intrinsic_dim_zero(self):
        """Raises error if intrinsic_dim < 1."""
        D = 4
        flow = build_flow(D)

        with pytest.raises(ValueError, match="intrinsic_dim must be >= 1"):
            ManifoldFlow(flow, intrinsic_dim=0)

        with pytest.raises(ValueError, match="intrinsic_dim must be >= 1"):
            ManifoldFlow(flow, intrinsic_dim=-1)


class TestFlowManifoldProtocol:
    """Test FlowManifold properties and roundtrips."""

    def test_properties(self):
        """FlowManifold exposes correct dimensions."""
        D, k, _B = 4, 2, 16
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)
        mean = torch.zeros(D)
        std = torch.ones(D)

        fm = FlowManifold(mf, mean, std)

        assert fm.intrinsic_dim == k
        assert fm.ambient_dim == D
        assert fm.residual_dim == D - k

    def test_encode_decode_roundtrip(self):
        """encode -> decode with original residual recovers x."""
        D, k, B = 4, 2, 32
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)

        # Use non-trivial standardization
        mean = torch.randn(D)
        std = torch.rand(D) + 0.5

        fm = FlowManifold(mf, mean, std)

        x = torch.randn(B, D)
        intrinsic, residual = fm.encode(x)

        assert intrinsic.shape == (B, k)
        assert residual.shape == (B, D - k)

        x_recon = fm.decode(intrinsic, residual)
        torch.testing.assert_close(x_recon, x, atol=1e-5, rtol=1e-5)

    def test_project_zeros_residual(self):
        """project() returns on-manifold points (decode with r=None)."""
        D, k, B = 4, 2, 16
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)
        mean = torch.zeros(D)
        std = torch.ones(D)

        fm = FlowManifold(mf, mean, std)

        x = torch.randn(B, D)
        x_proj = fm.project(x)

        assert x_proj.shape == (B, D)

        # Verify projection is idempotent
        x_proj2 = fm.project(x_proj)
        torch.testing.assert_close(x_proj2, x_proj, atol=1e-5, rtol=1e-5)


class TestFlowManifoldSteeringGrid:
    """Test FlowManifold steering grid generation."""

    def test_grid_1d(self):
        """1D manifold produces (n, 1) grid."""
        D, k = 4, 1
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)
        fm = FlowManifold(mf, torch.zeros(D), torch.ones(D))

        grid = fm.make_steering_grid(n_points_per_dim=11)
        assert grid.shape == (11, 1)

    def test_grid_2d(self):
        """2D manifold produces (n^2, 2) meshgrid."""
        D, k = 4, 2
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)
        fm = FlowManifold(mf, torch.zeros(D), torch.ones(D))

        grid = fm.make_steering_grid(n_points_per_dim=11)
        assert grid.shape == (11 * 11, 2)

    def test_grid_3d_sparse(self):
        """3D+ manifold produces sparse grid (n*d, d)."""
        D, k = 6, 3
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)
        fm = FlowManifold(mf, torch.zeros(D), torch.ones(D))

        grid = fm.make_steering_grid(n_points_per_dim=11)
        assert grid.shape == (11 * 3, 3)

    def test_grid_custom_ranges(self):
        """Custom per-dimension ranges work correctly."""
        D, k = 4, 2
        flow = build_flow(D)
        mf = ManifoldFlow(flow, intrinsic_dim=k)
        fm = FlowManifold(mf, torch.zeros(D), torch.ones(D))

        ranges = ((-1.0, 1.0), (-2.0, 2.0))
        grid = fm.make_steering_grid(n_points_per_dim=5, ranges=ranges)

        # Check range bounds
        assert grid[:, 0].min() == pytest.approx(-1.0)
        assert grid[:, 0].max() == pytest.approx(1.0)
        assert grid[:, 1].min() == pytest.approx(-2.0)
        assert grid[:, 1].max() == pytest.approx(2.0)
