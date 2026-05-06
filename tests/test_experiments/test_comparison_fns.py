"""Tests for distribution comparison functions in metric.py."""

import math

import pytest
import torch

from causalab.methods.metric import (
    DISTRIBUTION_COMPARISONS,
    kl_divergence,
    hellinger_distance,
)


class TestKLDivergence:
    def test_identical_distributions(self):
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        result = kl_divergence(p, p)
        assert result.shape == (1,)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_distributions(self):
        p = torch.tensor([[1.0, 0.0, 0.0]])
        q = torch.tensor([[0.0, 1.0, 0.0]])
        result = kl_divergence(p, q)
        # KL should be large (log(1/eps))
        assert result.item() > 10.0

    def test_batch(self):
        p = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
        q = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        result = kl_divergence(p, q)
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(0.0, abs=1e-6)
        assert result[1].item() > 0.0

    def test_non_negative(self):
        torch.manual_seed(42)
        for _ in range(10):
            logits = torch.randn(5, 4)
            p = torch.softmax(logits, dim=-1)
            q = torch.softmax(torch.randn(5, 4), dim=-1)
            result = kl_divergence(p, q)
            assert (result >= -1e-6).all(), f"KL should be non-negative, got {result}"

    def test_zero_reference_entries_ignored(self):
        p = torch.tensor([[0.5, 0.5, 0.0]])
        q = torch.tensor([[0.3, 0.3, 0.4]])
        result = kl_divergence(p, q)
        # Should only sum over non-zero entries of p
        expected = 0.5 * math.log(0.5 / 0.3) + 0.5 * math.log(0.5 / 0.3)
        assert result.item() == pytest.approx(expected, abs=1e-5)


class TestHellingerDistance:
    def test_identical_distributions(self):
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        result = hellinger_distance(p, p)
        assert result.shape == (1,)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_distributions(self):
        p = torch.tensor([[1.0, 0.0]])
        q = torch.tensor([[0.0, 1.0]])
        result = hellinger_distance(p, q)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_range_zero_to_one(self):
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(5, 4), dim=-1)
            q = torch.softmax(torch.randn(5, 4), dim=-1)
            result = hellinger_distance(p, q)
            assert (result >= -1e-6).all()
            assert (result <= 1.0 + 1e-6).all()

    def test_batch(self):
        p = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
        q = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
        result = hellinger_distance(p, q)
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(0.0, abs=1e-6)
        assert result[1].item() == pytest.approx(1.0, abs=1e-5)

    def test_symmetric(self):
        p = torch.tensor([[0.7, 0.3]])
        q = torch.tensor([[0.4, 0.6]])
        assert hellinger_distance(p, q).item() == pytest.approx(
            hellinger_distance(q, p).item(), abs=1e-6
        )


class TestRegistry:
    def test_kl_in_registry(self):
        assert "kl" in DISTRIBUTION_COMPARISONS
        assert DISTRIBUTION_COMPARISONS["kl"] is kl_divergence

    def test_hellinger_in_registry(self):
        assert "hellinger" in DISTRIBUTION_COMPARISONS
        assert DISTRIBUTION_COMPARISONS["hellinger"] is hellinger_distance
