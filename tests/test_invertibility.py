"""Tests for bijector invertibility."""

import torch

from causalab.methods.flow import Permutation, AffineCoupling


class TestPermutationInvertibility:
    """Test Permutation bijector invertibility."""

    def test_permutation_roundtrip(self):
        """Test x -> fwd -> inv recovers x."""
        torch.manual_seed(0)
        B, D = 64, 10
        x = torch.randn(B, D)
        perm = Permutation(D, seed=123)

        y, ld1 = perm.forward(x)
        x2, ld2 = perm.inverse(y)

        assert torch.max(torch.abs(x - x2)).item() < 1e-6
        assert torch.max(torch.abs(ld1)).item() == 0.0
        assert torch.max(torch.abs(ld2)).item() == 0.0

    def test_permutation_shapes(self):
        """Test output shapes are correct."""
        torch.manual_seed(0)
        B, D = 32, 8
        x = torch.randn(B, D)
        perm = Permutation(D, seed=42)

        y, ld = perm.forward(x)

        assert y.shape == (B, D)
        assert ld.shape == (B,)

    def test_permutation_with_explicit_perm(self):
        """Test with explicit permutation tensor."""
        D = 5
        perm_tensor = torch.tensor([4, 3, 2, 1, 0])
        perm = Permutation(D, perm=perm_tensor)

        x = torch.arange(D).unsqueeze(0).float()  # [[0, 1, 2, 3, 4]]
        y, _ = perm.forward(x)

        expected = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
        assert torch.allclose(y, expected)


class TestAffineCouplingInvertibility:
    """Test AffineCoupling bijector invertibility."""

    def test_affine_coupling_roundtrip(self):
        """Test x -> fwd -> inv recovers x."""
        torch.manual_seed(0)
        B, D = 64, 8
        idx_a = torch.arange(0, D, 2)
        idx_b = torch.arange(1, D, 2)

        layer = AffineCoupling(
            D, idx_a=idx_a, idx_b=idx_b, hidden=64, depth=2, s_scale=2.0
        )

        x = torch.randn(B, D)
        y, _ld1 = layer.forward(x)
        x2, _ld2 = layer.inverse(y)

        assert torch.max(torch.abs(x - x2)).item() < 1e-5

    def test_affine_coupling_shapes(self):
        """Test output shapes are correct."""
        torch.manual_seed(0)
        B, D = 32, 6
        idx_a = torch.arange(0, D, 2)
        idx_b = torch.arange(1, D, 2)

        layer = AffineCoupling(D, idx_a=idx_a, idx_b=idx_b)

        x = torch.randn(B, D)
        y, ld = layer.forward(x)

        assert y.shape == (B, D)
        assert ld.shape == (B,)

    def test_affine_coupling_conditioning_unchanged(self):
        """Test that conditioning dimensions are unchanged."""
        torch.manual_seed(0)
        B, D = 16, 8
        idx_a = torch.arange(0, D, 2)  # [0, 2, 4, 6]
        idx_b = torch.arange(1, D, 2)  # [1, 3, 5, 7]

        layer = AffineCoupling(D, idx_a=idx_a, idx_b=idx_b)

        x = torch.randn(B, D)
        y, _ = layer.forward(x)

        # Conditioning dims should be identical
        assert torch.allclose(x[:, idx_a], y[:, idx_a])


class TestFlowInvertibility:
    """Test full flow invertibility."""

    def test_flow_roundtrip(self):
        """Test x -> fwd -> inv recovers x for full flow."""
        torch.manual_seed(0)
        from causalab.methods.flow import Flow, StandardNormal

        B, D = 32, 4
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
        z, _ld_fwd = flow.fwd(x)
        x2, _ld_inv = flow.inv(z)

        assert torch.max(torch.abs(x - x2)).item() < 1e-5
