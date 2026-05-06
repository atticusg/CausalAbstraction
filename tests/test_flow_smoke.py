"""Smoke tests for flow model."""

import torch

from causalab.methods.flow import Flow, StandardNormal, Permutation, AffineCoupling


def build_test_flow(D: int, num_layers: int = 4, hidden: int = 64) -> Flow:
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
            AffineCoupling(D, idx_a, idx_b, hidden=hidden, depth=2, s_scale=2.0)
        )
        layers.append(Permutation(D, seed=100 + i))

    return Flow(layers, base_dist=StandardNormal(D))


class TestFlowSmoke:
    """Smoke tests for flow.log_prob and flow.sample."""

    def test_log_prob_shape_and_finiteness(self):
        """Test log_prob returns correct shape with finite values."""
        torch.manual_seed(0)
        B, D = 32, 6
        flow = build_test_flow(D)

        x = torch.randn(B, D)
        lp = flow.log_prob(x)

        assert lp.shape == (B,)
        assert torch.isfinite(lp).all()

    def test_sample_shape_and_finiteness(self):
        """Test sample returns correct shape with finite values."""
        torch.manual_seed(0)
        D = 6
        n = 10
        flow = build_test_flow(D)

        s = flow.sample(n)

        assert s.shape == (n, D)
        assert torch.isfinite(s).all()

    def test_fwd_inv_shapes(self):
        """Test fwd and inv return correct shapes."""
        torch.manual_seed(0)
        B, D = 32, 4
        flow = build_test_flow(D)

        x = torch.randn(B, D)
        z, ld_fwd = flow.fwd(x)
        x2, ld_inv = flow.inv(z)

        assert z.shape == (B, D)
        assert ld_fwd.shape == (B,)
        assert x2.shape == (B, D)
        assert ld_inv.shape == (B,)

    def test_log_prob_different_batch_sizes(self):
        """Test log_prob works with various batch sizes."""
        torch.manual_seed(0)
        D = 4
        flow = build_test_flow(D)

        for B in [1, 16, 128]:
            x = torch.randn(B, D)
            lp = flow.log_prob(x)
            assert lp.shape == (B,)
            assert torch.isfinite(lp).all()

    def test_sample_different_sizes(self):
        """Test sample works with various sample sizes."""
        torch.manual_seed(0)
        D = 4
        flow = build_test_flow(D)

        for n in [1, 10, 100]:
            s = flow.sample(n)
            assert s.shape == (n, D)
            assert torch.isfinite(s).all()

    def test_log_prob_gradient_exists(self):
        """Test that gradients flow through log_prob."""
        torch.manual_seed(0)
        B, D = 16, 4
        flow = build_test_flow(D)

        x = torch.randn(B, D)
        lp = flow.log_prob(x)
        loss = -lp.mean()
        loss.backward()

        # Check that at least one parameter has gradients
        has_grad = False
        for p in flow.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def test_base_dist_log_prob_shape(self):
        """Test base distribution log_prob returns (B,)."""
        B, D = 32, 6
        base = StandardNormal(D)
        z = torch.randn(B, D)
        lp = base.log_prob(z)

        assert lp.shape == (B,)
        assert torch.isfinite(lp).all()

    def test_base_dist_sample_shape(self):
        """Test base distribution sample returns correct shape."""
        D = 6
        n = 10
        base = StandardNormal(D)
        z = base.sample((n,))

        assert z.shape == (n, D)
        assert torch.isfinite(z).all()


class TestFlowEdgeCases:
    """Test edge cases and potential failure modes."""

    def test_small_dimension(self):
        """Test flow works with D=2 (minimum useful dimension)."""
        torch.manual_seed(0)
        D = 2
        flow = build_test_flow(D, num_layers=4)

        x = torch.randn(16, D)
        lp = flow.log_prob(x)
        assert torch.isfinite(lp).all()

        s = flow.sample(10)
        assert torch.isfinite(s).all()

    def test_large_batch(self):
        """Test flow handles large batches."""
        torch.manual_seed(0)
        D = 4
        B = 1024
        flow = build_test_flow(D)

        x = torch.randn(B, D)
        lp = flow.log_prob(x)
        assert lp.shape == (B,)
        assert torch.isfinite(lp).all()

    def test_extreme_inputs(self):
        """Test flow handles somewhat extreme inputs."""
        torch.manual_seed(0)
        D = 4
        flow = build_test_flow(D)

        # Large but not too extreme values
        x = torch.randn(16, D) * 3.0
        lp = flow.log_prob(x)
        assert torch.isfinite(lp).all()

    def test_eval_mode(self):
        """Test flow works in eval mode."""
        torch.manual_seed(0)
        D = 4
        flow = build_test_flow(D)
        flow.eval()

        x = torch.randn(16, D)
        lp = flow.log_prob(x)
        assert torch.isfinite(lp).all()

        s = flow.sample(10)
        assert torch.isfinite(s).all()
