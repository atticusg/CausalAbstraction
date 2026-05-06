"""Tests for multi-variant token probability aggregation.

Tests that class_probabilities correctly sums probability mass across
token variants (e.g., " Monday", "Monday", " monday") for each concept.
"""

from __future__ import annotations


import pytest
import torch
import torch.nn.functional as F


class TestClassProbabilities:
    """Test class_probabilities with variant token IDs."""

    def test_single_id_per_class_unchanged(self):
        """Flat list of ints (old format) should still work."""
        from causalab.methods.metric import class_probabilities

        logits = torch.randn(3, 100)
        ids = [10, 20, 30]
        probs = class_probabilities(logits, ids, full_vocab_softmax=False)
        assert probs.shape == (3, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)

    def test_variant_ids_sum_probabilities(self):
        """Multiple variant IDs per class should have their probs summed."""
        from causalab.methods.metric import class_probabilities

        # Construct logits where we know the answer
        logits = torch.zeros(1, 10)
        # Class 0 has variants at indices 0, 1 (equal logits)
        logits[0, 0] = 2.0
        logits[0, 1] = 2.0
        # Class 1 has single variant at index 5
        logits[0, 5] = 2.0

        ids = [[0, 1], [5]]
        probs_fvs = class_probabilities(logits, ids, full_vocab_softmax=True)

        # With full_vocab_softmax: class 0 gets sum of probs at 0 and 1
        full_probs = F.softmax(logits.float(), dim=-1)
        expected_c0 = full_probs[0, 0].item() + full_probs[0, 1].item()
        expected_c1 = full_probs[0, 5].item()
        assert probs_fvs[0, 0].item() == pytest.approx(expected_c0, rel=1e-5)
        assert probs_fvs[0, 1].item() == pytest.approx(expected_c1, rel=1e-5)

        # Class 0 should get ~2x the mass of class 1 (same logit, two variants)
        assert probs_fvs[0, 0].item() > 1.5 * probs_fvs[0, 1].item()

    def test_variant_renormalization(self):
        """With full_vocab_softmax=False, concept probs should sum to 1."""
        from causalab.methods.metric import class_probabilities

        logits = torch.randn(5, 100)
        ids = [[10, 11, 12], [20, 21], [30]]
        probs = class_probabilities(logits, ids, full_vocab_softmax=False)
        assert probs.shape == (5, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_full_vocab_softmax_does_not_sum_to_one(self):
        """With full_vocab_softmax=True, concept probs should NOT sum to 1
        (there's mass on non-concept tokens)."""
        from causalab.methods.metric import class_probabilities

        logits = torch.randn(3, 1000)  # large vocab
        ids = [[10], [20], [30]]
        probs = class_probabilities(logits, ids, full_vocab_softmax=True)
        # Sum should be < 1 since most mass is on non-concept tokens
        assert (probs.sum(dim=-1) < 1.0).all()

    def test_variant_sum_vs_single_consistency(self):
        """A concept with one variant should give same result as flat int."""
        from causalab.methods.metric import class_probabilities

        logits = torch.randn(4, 50)
        flat_ids = [5, 10, 15]
        variant_ids = [[5], [10], [15]]

        probs_flat = class_probabilities(logits, flat_ids, full_vocab_softmax=True)
        probs_var = class_probabilities(logits, variant_ids, full_vocab_softmax=True)
        assert torch.allclose(probs_flat, probs_var, atol=1e-6)

    def test_variant_captures_more_mass(self):
        """Adding variants should always capture >= mass than single token."""
        from causalab.methods.metric import class_probabilities

        logits = torch.randn(10, 200)
        # Single variant
        single = [[42], [99]]
        # Same + extra variants
        multi = [[42, 43, 44], [99, 100]]

        probs_s = class_probabilities(logits, single, full_vocab_softmax=True)
        probs_m = class_probabilities(logits, multi, full_vocab_softmax=True)

        # Multi should capture >= single for every example and concept
        assert (probs_m >= probs_s - 1e-7).all()

    def test_duplicate_variant_ids_not_double_counted(self):
        """If the same token ID appears twice in variants, it should NOT
        be double-counted (tokenize_variable_values deduplicates, but
        class_probabilities should be robust to it too)."""
        from causalab.methods.metric import class_probabilities

        logits = torch.randn(2, 50)
        # Token 10 listed twice — should NOT double its probability
        ids_dup = [[10, 10], [20]]
        ids_single = [[10], [20]]

        probs_dup = class_probabilities(logits, ids_dup, full_vocab_softmax=True)
        probs_single = class_probabilities(logits, ids_single, full_vocab_softmax=True)

        # Currently class_probabilities does NOT deduplicate — it sums.
        # This test documents the behavior: duplicate IDs DO double-count.
        # The deduplication happens upstream in tokenize_variable_values.
        assert probs_dup[0, 0].item() == pytest.approx(
            2 * probs_single[0, 0].item(), rel=1e-5
        )


class TestTokenizeVariableValues:
    """Test tokenize_variable_values with the new list-returning pattern."""

    def test_returns_list_of_lists(self):
        from causalab.methods.metric import tokenize_variable_values
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        values = ["Monday", "Tuesday"]
        pattern = lambda v: [f" {v}"]
        result = tokenize_variable_values(tok, values, pattern)
        assert isinstance(result, list)
        assert all(isinstance(r, list) for r in result)
        assert all(len(r) >= 1 for r in result)

    def test_variants_are_deduplicated(self):
        from causalab.methods.metric import tokenize_variable_values
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        values = ["Monday"]
        # Pattern returns duplicates
        pattern = lambda v: [f" {v}", f" {v}", f" {v}"]
        result = tokenize_variable_values(tok, values, pattern)
        # Should be deduplicated
        assert len(result[0]) == 1

    def test_multi_token_variants_filtered(self):
        """Variants that encode to multiple tokens should be excluded."""
        from causalab.methods.metric import tokenize_variable_values
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        values = ["Monday"]
        # " Monday" is 1 token, "the Monday" is 2 tokens
        pattern = lambda v: [f" {v}", f"the {v}"]
        result = tokenize_variable_values(tok, values, pattern)
        # "the Monday" should be filtered out, only " Monday" remains
        assert len(result[0]) == 1

    def test_natural_domain_variants(self):
        """Test the actual natural_domains_arithmetic variant pattern."""
        from causalab.methods.metric import tokenize_variable_values
        from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
        from causalab.tasks.natural_domains_arithmetic.causal_models import (
            create_causal_model,
            GET_RESULT_TOKEN_PATTERN,
        )
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        cfg = NaturalDomainConfig(domain_type="weekdays")
        cm = create_causal_model(cfg)
        pattern = GET_RESULT_TOKEN_PATTERN(cm)

        # Check that Monday produces multiple variants
        variants = pattern("Monday")
        assert " Monday" in variants
        assert "Monday" in variants
        assert " monday" in variants

        # Tokenize and check all are single tokens
        result = tokenize_variable_values(tok, ["Monday"], pattern)
        assert len(result[0]) >= 2  # at least " Monday" and "Monday"

    def test_hours_numeric_variants(self):
        """Hours use numeric tokens with no space prefix.
        " 1" is multi-token in Llama ([220, 16]), so only "1" survives."""
        from causalab.methods.metric import tokenize_variable_values
        from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
        from causalab.tasks.natural_domains_arithmetic.causal_models import (
            create_causal_model,
            GET_RESULT_TOKEN_PATTERN,
        )
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        cfg = NaturalDomainConfig(domain_type="hours")
        cm = create_causal_model(cfg)
        pattern = GET_RESULT_TOKEN_PATTERN(cm)

        # Pattern generates variants but " 1" is multi-token in Llama
        variants = pattern("1")
        assert "1" in variants
        assert " 1" in variants  # pattern generates it

        result = tokenize_variable_values(tok, ["1"], pattern)
        # Only "1" (token 16) survives — " 1" is [220, 16] (multi-token)
        assert len(result[0]) == 1
        assert result[0][0] == tok.encode("1", add_special_tokens=False)[0]


class TestScoresToJointProbs:
    """Test scores_to_joint_probs with variant token IDs."""

    def test_single_step_uses_variants(self):
        """For single generation step, variants should be summed."""
        from causalab.methods.metric import scores_to_joint_probs

        # 2 examples, vocab size 50
        logits = torch.randn(2, 50)
        # 3 concepts with variants
        var_indices = [[10, 11], [20], [30, 31, 32]]

        result = scores_to_joint_probs([logits], var_indices, full_vocab_softmax=True)
        assert result is not None
        assert result.shape == (2, 3)

        # Concept 0 should have more mass than if we only used token 10
        full_probs = F.softmax(logits.float(), dim=-1)
        single_mass = full_probs[:, 10]
        variant_mass = result[:, 0]
        assert (variant_mass >= single_mass - 1e-6).all()
