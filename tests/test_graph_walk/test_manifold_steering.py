"""Tests for graph walk manifold steering helpers.

Tests graph-walk utilities in graphs.py, evaluation.py, and counterfactuals.py.
Pipeline-level tests live in tests/test_experiments/test_manifold_steering_pipeline.py.
"""

from __future__ import annotations


import torch

from causalab.tasks.graph_walk.graphs import make_ring_graph, make_grid_graph


# ---------------------------------------------------------------------------
# get_control_points_tensor
# ---------------------------------------------------------------------------


class TestGetControlPoints:
    """Test extraction of coordinate tensors from Graph objects."""

    def test_ring_graph_coords_and_periodic(self):
        import math
        from causalab.tasks.graph_walk.graphs import get_control_points_tensor

        graph = make_ring_graph(6)
        coords, periodic_info = get_control_points_tensor(graph)
        assert coords.shape == (6, 1)
        expected = torch.tensor([[2 * math.pi * i / 6] for i in range(6)])
        assert torch.allclose(coords, expected, atol=1e-5)
        assert "angle" in periodic_info
        assert abs(periodic_info["angle"] - 2 * math.pi) < 1e-5

    def test_grid_graph_coords_and_not_periodic(self):
        from causalab.tasks.graph_walk.graphs import get_control_points_tensor

        graph = make_grid_graph(3)
        coords, periodic_info = get_control_points_tensor(graph)
        assert coords.shape == (9, 2)
        assert torch.allclose(coords[0], torch.tensor([0.0, 0.0]))
        assert torch.allclose(coords[4], torch.tensor([1.0, 1.0]))
        assert torch.allclose(coords[8], torch.tensor([2.0, 2.0]))
        assert periodic_info == {}


# ---------------------------------------------------------------------------
# compute_expected_neighbor_distributions
# ---------------------------------------------------------------------------


class TestComputeExpectedNeighborDistributions:
    """Test expected neighbor distribution computation."""

    def test_ring_uniform_neighbors(self):
        from causalab.tasks.graph_walk.graphs import (
            compute_expected_neighbor_distributions,
        )

        graph = make_ring_graph(6)
        dist = compute_expected_neighbor_distributions(graph)
        assert dist.shape == (6, 6)
        for node in range(6):
            neighbors = graph.adjacency[node]
            for j in range(6):
                if j in neighbors:
                    assert abs(dist[node, j].item() - 0.5) < 1e-6
                else:
                    assert dist[node, j].item() == 0.0

    def test_grid_corner_node(self):
        from causalab.tasks.graph_walk.graphs import (
            compute_expected_neighbor_distributions,
        )

        graph = make_grid_graph(3)
        dist = compute_expected_neighbor_distributions(graph)
        assert dist.shape == (9, 9)
        assert abs(dist[0, 1].item() - 0.5) < 1e-6
        assert abs(dist[0, 3].item() - 0.5) < 1e-6
        assert dist[0, 0].item() == 0.0

    def test_rows_sum_to_one(self):
        from causalab.tasks.graph_walk.graphs import (
            compute_expected_neighbor_distributions,
        )

        graph = make_grid_graph(4)
        dist = compute_expected_neighbor_distributions(graph)
        row_sums = dist.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(16), atol=1e-6)


# ---------------------------------------------------------------------------
# KL divergence (F.kl_div usage, as used in evaluate_at_nodes)
# ---------------------------------------------------------------------------


class TestKLDivergence:
    """Test KL divergence computation as used in evaluate_at_nodes."""

    def test_perfect_match_gives_zero_kl(self):
        from causalab.tasks.graph_walk.graphs import (
            compute_expected_neighbor_distributions,
        )

        graph = make_ring_graph(6)
        ref_dists = compute_expected_neighbor_distributions(graph)
        steered_probs = ref_dists.clone()

        steered_log = steered_probs.clamp(min=1e-10).log()
        kl = torch.nn.functional.kl_div(
            steered_log, ref_dists, reduction="none", log_target=False
        ).sum(dim=1)
        assert kl.shape == (6,)
        assert torch.allclose(kl, torch.zeros(6), atol=1e-5)

    def test_uniform_steered_gives_positive_kl(self):
        from causalab.tasks.graph_walk.graphs import (
            compute_expected_neighbor_distributions,
        )

        graph = make_ring_graph(6)
        ref_dists = compute_expected_neighbor_distributions(graph)
        steered_probs = torch.ones(6, 6) / 6.0

        steered_log = steered_probs.clamp(min=1e-10).log()
        kl = torch.nn.functional.kl_div(
            steered_log, ref_dists, reduction="none", log_target=False
        ).sum(dim=1)
        assert kl.shape == (6,)
        assert (kl > 0).all()

    def test_grid_graph_perfect_match(self):
        from causalab.tasks.graph_walk.graphs import (
            compute_expected_neighbor_distributions,
        )

        graph = make_grid_graph(3)
        ref_dists = compute_expected_neighbor_distributions(graph)
        steered_probs = ref_dists.clone()

        steered_log = steered_probs.clamp(min=1e-10).log()
        kl = torch.nn.functional.kl_div(
            steered_log, ref_dists, reduction="none", log_target=False
        ).sum(dim=1)
        assert torch.allclose(kl, torch.zeros(9), atol=1e-5)
