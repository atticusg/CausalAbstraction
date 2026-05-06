"""Tests for log-determinant consistency."""

import torch

from causalab.methods.flow import (
    Flow,
    StandardNormal,
    Permutation,
    AffineCoupling,
)


class TestLogdetConsistency:
    """
    Test that logdet_fwd + logdet_inv ≈ 0 for paired samples.

    This is a critical correctness requirement for normalizing flows.
    """

    def test_permutation_logdet_consistency(self):
        """Permutation should have zero logdet in both directions."""
        torch.manual_seed(0)
        B, D = 64, 10
        x = torch.randn(B, D)
        perm = Permutation(D, seed=123)

        y, ld_fwd = perm.forward(x)
        _x2, ld_inv = perm.inverse(y)

        # Both should be zero
        assert torch.max(torch.abs(ld_fwd)).item() == 0.0
        assert torch.max(torch.abs(ld_inv)).item() == 0.0
        # Sum should be zero
        assert torch.max(torch.abs(ld_fwd + ld_inv)).item() == 0.0

    def test_affine_coupling_logdet_consistency(self):
        """Affine coupling logdet_fwd + logdet_inv should be ~0."""
        torch.manual_seed(0)
        B, D = 64, 8
        idx_a = torch.arange(0, D, 2)
        idx_b = torch.arange(1, D, 2)

        layer = AffineCoupling(
            D, idx_a=idx_a, idx_b=idx_b, hidden=64, depth=2, s_scale=2.0
        )

        x = torch.randn(B, D)
        y, ld_fwd = layer.forward(x)
        _x2, ld_inv = layer.inverse(y)

        # logdet_inv should be -logdet_fwd for paired x->y
        assert torch.max(torch.abs(ld_fwd + ld_inv)).item() < 1e-5

    def test_flow_logdet_consistency(self):
        """Full flow logdet_fwd + logdet_inv should be ~0."""
        torch.manual_seed(0)
        B, D = 32, 6
        layers = []
        for i in range(4):
            if i % 2 == 0:
                idx_a = torch.arange(0, D, 2)
                idx_b = torch.arange(1, D, 2)
            else:
                idx_a = torch.arange(1, D, 2)
                idx_b = torch.arange(0, D, 2)

            layers.append(
                AffineCoupling(D, idx_a, idx_b, hidden=64, depth=2, s_scale=2.0)
            )
            layers.append(Permutation(D, seed=100 + i))

        flow = Flow(layers, base_dist=StandardNormal(D))

        x = torch.randn(B, D)
        z, ld_fwd = flow.fwd(x)
        _x2, ld_inv = flow.inv(z)

        assert torch.max(torch.abs(ld_fwd + ld_inv)).item() < 1e-5

    def test_affine_coupling_logdet_values(self):
        """Test that logdet equals sum of scale factors."""
        torch.manual_seed(42)
        B, D = 16, 4
        idx_a = torch.tensor([0, 2])
        idx_b = torch.tensor([1, 3])

        layer = AffineCoupling(
            D, idx_a=idx_a, idx_b=idx_b, hidden=32, depth=1, s_scale=2.0
        )

        x = torch.randn(B, D)
        _y, ld = layer.forward(x)

        # Manually compute expected logdet
        xa = x[:, idx_a]
        st = layer.nn(xa)
        raw_s, _t = st.chunk(2, dim=-1)
        s = layer.s_scale * torch.tanh(raw_s)
        expected_ld = s.sum(dim=-1)

        assert torch.allclose(ld, expected_ld, atol=1e-6)
