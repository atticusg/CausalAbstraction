"""Tests for multi-token class probability support.

Covers class_probabilities, _normalize_var_indices, _logits_to_class_probs,
compute_reference_distributions, and the pairwise/centroid layer scan
functions with list[list[int]] score_token_ids.
"""

import pytest
import torch
import torch.nn.functional as F

from causalab.methods.metric import (
    class_probabilities,
    _normalize_var_indices,
    _logits_to_class_probs,
    compute_reference_distributions,
)


# ---------------------------------------------------------------------------
# _normalize_var_indices
# ---------------------------------------------------------------------------


class TestNormalizeVarIndices:
    def test_flat_list(self):
        assert _normalize_var_indices([10, 20, 30]) == [[10], [20], [30]]

    def test_already_nested(self):
        nested = [[10, 11], [20, 21]]
        assert _normalize_var_indices(nested) == nested

    def test_tensor(self):
        t = torch.tensor([5, 6, 7])
        assert _normalize_var_indices(t) == [[5], [6], [7]]


# ---------------------------------------------------------------------------
# class_probabilities
# ---------------------------------------------------------------------------


class TestClassProbabilities:
    def test_basic_shape(self):
        logits = torch.randn(4, 100)
        probs = class_probabilities(logits, [10, 20, 30])
        assert probs.shape == (4, 3)

    def test_sums_to_one(self):
        logits = torch.randn(8, 50)
        probs = class_probabilities(logits, [0, 1, 2, 3])
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(8), atol=1e-5, rtol=0)

    def test_1d_input(self):
        logits = torch.randn(100)
        probs = class_probabilities(logits, [5, 10])
        assert probs.shape == (1, 2)

    def test_dtype_promotion(self):
        logits = torch.randn(2, 50).bfloat16()
        probs = class_probabilities(logits, [0, 1])
        assert probs.dtype == torch.float32

    def test_correct_indexing(self):
        """Make high logit at specific index → high probability."""
        logits = torch.zeros(1, 100)
        logits[0, 42] = 100.0  # dominate
        probs = class_probabilities(logits, [42, 7])
        assert probs[0, 0].item() > 0.99
        assert probs[0, 1].item() < 0.01


# ---------------------------------------------------------------------------
# _logits_to_class_probs (multi-token joint probabilities)
# ---------------------------------------------------------------------------


class TestLogitsToClassProbs:
    def test_single_token_classes(self):
        """Single-token: equivalent to class_probabilities on one step."""
        logits = torch.randn(4, 100)
        token_seqs = [[10], [20], [30]]
        probs = _logits_to_class_probs([logits], token_seqs)
        assert probs.shape == (4, 3)
        # Should sum to 1 (normalized)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=0)

    def test_multi_token_two_steps(self):
        """Two-token classes: joint = p(step0) * p(step1), renormalized."""
        N, V = 8, 50
        logits_step0 = torch.randn(N, V)
        logits_step1 = torch.randn(N, V)

        # 3 classes, each with 2 tokens
        token_seqs = [[10, 20], [11, 21], [12, 22]]

        probs = _logits_to_class_probs([logits_step0, logits_step1], token_seqs)
        assert probs.shape == (N, 3)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(N), atol=1e-5, rtol=0)

        # Verify it's actually multiplying: compute manually
        p0 = F.softmax(logits_step0[:, [10, 11, 12]].float(), dim=-1)
        p1 = F.softmax(logits_step1[:, [20, 21, 22]].float(), dim=-1)
        joint_raw = p0 * p1
        joint_norm = joint_raw / joint_raw.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(probs, joint_norm, atol=1e-5, rtol=0)

    def test_mixed_length_sequences(self):
        """Classes with different token counts (e.g. 1-token and 2-token)."""
        N, V = 4, 30
        logits_step0 = torch.randn(N, V)
        logits_step1 = torch.randn(N, V)

        # Class 0: 1 token, Class 1: 2 tokens, Class 2: 2 tokens
        token_seqs = [[5], [6, 15], [7, 16]]

        probs = _logits_to_class_probs([logits_step0, logits_step1], token_seqs)
        assert probs.shape == (N, 3)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(N), atol=1e-5, rtol=0)

    def test_dominant_class_wins(self):
        """High logits for one class across all steps → high joint probability."""
        N, V = 1, 20
        logits_step0 = torch.zeros(N, V)
        logits_step1 = torch.zeros(N, V)
        # Make class 0 dominate at both steps
        logits_step0[0, 10] = 100.0
        logits_step1[0, 20 % V] = 100.0

        token_seqs = [[10, 20 % V], [11, (21 % V)], [12, (22 % V)]]
        probs = _logits_to_class_probs([logits_step0, logits_step1], token_seqs)
        assert probs[0, 0].item() > 0.99


# ---------------------------------------------------------------------------
# compute_reference_distributions with multi-token
# ---------------------------------------------------------------------------


class TestComputeRefDistsMultiToken:
    """Test compute_reference_distributions with list[list[int]] token IDs."""

    def _make_mock_pipeline(self, n_classes, n_steps, vocab_size=100):
        """Create a mock pipeline that returns predictable logits."""
        from unittest.mock import MagicMock

        pipeline = MagicMock()
        call_count = [0]

        def mock_generate(inputs, **kwargs):
            bs = len(inputs)
            scores = []
            for step in range(n_steps):
                # Return random logits but seeded so it's deterministic
                torch.manual_seed(42 + call_count[0] * 100 + step)
                scores.append(torch.randn(bs, vocab_size))
            call_count[0] += 1
            return {"scores": scores}

        pipeline.generate = mock_generate
        return pipeline

    def test_shape_multi_token(self):
        """Reference distributions have correct shape with multi-token IDs."""
        n_classes = 5
        n_steps = 2
        # 2-token class IDs
        token_seqs = [[i, i + 10] for i in range(n_classes)]

        dataset = [{"input": {"class": i % n_classes}} for i in range(20)]

        pipeline = self._make_mock_pipeline(n_classes, n_steps)

        ref_dists = compute_reference_distributions(
            dataset=dataset,
            score_token_ids=token_seqs,
            n_classes=n_classes,
            example_to_class=lambda ex: ex["input"]["class"],
            pipeline=pipeline,
            score_token_index=0,
            batch_size=8,
        )

        assert ref_dists.shape == (n_classes, n_classes)
        # Each row should sum to ~1
        for i in range(n_classes):
            assert abs(ref_dists[i].sum().item() - 1.0) < 1e-4

    def test_single_token_backward_compat(self):
        """list[int] still works (backward compatibility)."""
        n_classes = 3
        token_ids = [10, 20, 30]

        dataset = [{"input": {"class": i % n_classes}} for i in range(12)]

        pipeline = self._make_mock_pipeline(n_classes, n_steps=2)

        ref_dists = compute_reference_distributions(
            dataset=dataset,
            score_token_ids=token_ids,
            n_classes=n_classes,
            example_to_class=lambda ex: ex["input"]["class"],
            pipeline=pipeline,
            score_token_index=0,
            batch_size=4,
        )

        assert ref_dists.shape == (n_classes, n_classes)
        for i in range(n_classes):
            assert abs(ref_dists[i].sum().item() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
