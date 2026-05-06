"""Tests for interchange layer scan with KL metric.

Tests the generic KL metric infrastructure and reference distribution
computation, using graph_walk as the concrete task.
"""

from __future__ import annotations

import torch

from causalab.methods.metric import (
    compute_reference_distributions,
    make_kl_checker,
)
from causalab.tasks.graph_walk.causal_models import create_causal_model
from causalab.tasks.graph_walk.config import GraphWalkConfig
from causalab.tasks.graph_walk.counterfactuals import (
    generate_graph_walk_dataset,
)


# ---------------------------------------------------------------------------
# compute_reference_distributions
# ---------------------------------------------------------------------------


class TestReferenceDistributions:
    """Test reference distribution computation."""

    def test_shape_and_normalization(self):
        """Reference dists should be (n_classes, n_classes), rows sum to 1."""
        from unittest.mock import MagicMock

        n_nodes = 4
        concept_token_ids = [10, 20, 30, 40]

        config = GraphWalkConfig(
            graph_type="ring", graph_size=n_nodes, context_length=10, seed=42
        )
        causal_model = create_causal_model(config)
        dataset = generate_graph_walk_dataset(causal_model, n_examples=8, seed=42)

        # Mock pipeline.generate to return fake scores
        pipeline = MagicMock()
        vocab_size = 100

        def fake_generate(traces):
            bs = len(traces)
            logits = torch.randn(bs, vocab_size)
            return {
                "scores": [torch.randn(bs, vocab_size), logits],
                "sequences": torch.zeros(bs, 2, dtype=torch.long),
                "string": "fake",
            }

        pipeline.generate = fake_generate

        ref_dists = compute_reference_distributions(
            dataset=dataset,
            score_token_ids=concept_token_ids,
            n_classes=n_nodes,
            example_to_class=lambda ex: ex["input"]["walk_sequence"][-1],
            pipeline=pipeline,
        )

        assert ref_dists.shape == (n_nodes, n_nodes)
        row_sums = ref_dists.sum(dim=1)
        for i in range(n_nodes):
            assert abs(row_sums[i].item() - 1.0) < 0.01, (
                f"Row {i} sums to {row_sums[i].item()}, expected ~1.0"
            )

    def test_with_precomputed_logits(self):
        """When output_logits is provided, pipeline is not needed."""
        n_nodes = 3
        concept_token_ids = [0, 1, 2]

        config = GraphWalkConfig(
            graph_type="ring", graph_size=n_nodes, context_length=10, seed=42
        )
        causal_model = create_causal_model(config)
        dataset = generate_graph_walk_dataset(causal_model, n_examples=6, seed=42)

        # Create fake logits: (seq_len, vocab_size) per example
        output_logits = [torch.randn(10, 5) for _ in range(len(dataset))]

        ref_dists = compute_reference_distributions(
            dataset=dataset,
            score_token_ids=concept_token_ids,
            n_classes=n_nodes,
            example_to_class=lambda ex: ex["input"]["walk_sequence"][-1],
            output_logits=output_logits,
        )

        assert ref_dists.shape == (n_nodes, n_nodes)
        row_sums = ref_dists.sum(dim=1)
        for i in range(n_nodes):
            assert abs(row_sums[i].item() - 1.0) < 0.01


# ---------------------------------------------------------------------------
# make_kl_checker (generic KL metric)
# ---------------------------------------------------------------------------


class TestKLChecker:
    """Test the generic KL checker from metric.py."""

    def test_identical_distributions_kl_zero(self):
        ref_dists = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        checker = make_kl_checker(
            ref_dists=ref_dists,
            score_token_ids=[0, 1, 2, 3],
            label_to_class=lambda _: 0,
            score_token_index=0,
        )
        # Build intervention output with logits that softmax to uniform
        logits = torch.zeros(4)
        full_logits = torch.zeros(1, 10)
        full_logits[0, :4] = logits
        intervention_output = {
            "scores": [full_logits],
            "example_idx": 0,
        }
        kl = checker(intervention_output, "dummy_label")
        assert abs(kl) < 1e-5, f"KL should be ~0 for identical dists, got {kl}"

    def test_different_distributions_kl_positive(self):
        ref_dists = torch.tensor([[0.9, 0.05, 0.025, 0.025]])
        checker = make_kl_checker(
            ref_dists=ref_dists,
            score_token_ids=[0, 1, 2, 3],
            label_to_class=lambda _: 0,
            score_token_index=0,
        )
        full_logits = torch.zeros(1, 10)
        intervention_output = {
            "scores": [full_logits],
            "example_idx": 0,
        }
        kl = checker(intervention_output, "dummy_label")
        assert kl > 0, f"KL should be positive for different dists, got {kl}"

    def test_label_to_class_selects_correct_row(self):
        """Verify that label_to_class correctly indexes into ref_dists."""
        ref_dists = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        checker = make_kl_checker(
            ref_dists=ref_dists,
            score_token_ids=[0, 1],
            label_to_class=lambda label: label,
            score_token_index=0,
        )
        full_logits = torch.zeros(1, 10)
        full_logits[0, 0] = 10.0
        intervention_output = {
            "scores": [full_logits],
            "example_idx": 0,
        }
        kl_class0 = checker(intervention_output, 0)
        kl_class1 = checker(intervention_output, 1)
        assert kl_class0 < kl_class1


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


class TestDatasetGeneration:
    """Test that counterfactual datasets have different ending nodes."""

    def test_base_and_cf_end_at_different_nodes(self):
        config = GraphWalkConfig(
            graph_type="ring", graph_size=5, context_length=10, seed=42
        )
        causal_model = create_causal_model(config)
        dataset = generate_graph_walk_dataset(causal_model, n_examples=20, seed=42)

        for i, ex in enumerate(dataset):
            base_trace = ex["input"]
            cf_trace = ex["counterfactual_inputs"][0]

            base_node = base_trace["walk_sequence"][-1]
            cf_node = cf_trace["walk_sequence"][-1]

            assert base_node != cf_node, (
                f"Example {i}: base and counterfactual end at the same node "
                f"({base_node})"
            )
