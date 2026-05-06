"""Tests for graph_walk configuration."""

import pytest

from causalab.tasks.graph_walk.config import DEFAULT_CONCEPTS, GraphWalkConfig


class TestDefaultConcepts:
    def test_no_duplicates(self):
        assert len(DEFAULT_CONCEPTS) == len(set(DEFAULT_CONCEPTS))

    def test_enough_for_19x19_grid(self):
        assert len(DEFAULT_CONCEPTS) >= 361

    def test_all_lowercase_alpha(self):
        for word in DEFAULT_CONCEPTS:
            assert word.isalpha() and word.islower(), f"Bad concept: {word!r}"


class TestGraphWalkConfig:
    def test_ring_auto_concepts(self):
        config = GraphWalkConfig(graph_type="ring", graph_size=10)
        assert len(config.concepts) == 10
        assert config.concepts == DEFAULT_CONCEPTS[:10]

    def test_grid_auto_concepts(self):
        config = GraphWalkConfig(graph_type="grid", graph_size=3)
        assert len(config.concepts) == 9

    def test_hex_auto_concepts(self):
        config = GraphWalkConfig(graph_type="hex", graph_size=4)
        assert len(config.concepts) == 16

    def test_custom_concepts_accepted(self):
        concepts = [f"word{i}" for i in range(25)]
        config = GraphWalkConfig(graph_type="grid", graph_size=5, concepts=concepts)
        assert config.concepts == concepts

    def test_wrong_concept_count_raises(self):
        with pytest.raises(ValueError, match="Expected 9 concepts"):
            GraphWalkConfig(graph_type="grid", graph_size=3, concepts=["a", "b"])

    def test_invalid_graph_type_raises(self):
        with pytest.raises(ValueError, match="graph_type must be"):
            GraphWalkConfig(graph_type="triangle", graph_size=3)

    def test_too_large_graph_raises(self):
        """Graph needing more nodes than DEFAULT_CONCEPTS should error."""
        with pytest.raises(ValueError, match="default concepts available"):
            GraphWalkConfig(graph_type="grid", graph_size=30)  # 900 nodes

    def test_ring_concept_count_differs_from_grid(self):
        """Ring uses graph_size directly, not graph_size^2."""
        ring = GraphWalkConfig(graph_type="ring", graph_size=5)
        grid = GraphWalkConfig(graph_type="grid", graph_size=5)
        assert len(ring.concepts) == 5
        assert len(grid.concepts) == 25
