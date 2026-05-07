"""Tests for the graph_walk causal model.

Focus on structural correctness: does the walk end at the right node?
Are the outputs valid neighbors? Is the text format correct?
"""

import pytest

from causalab.tasks.graph_walk.causal_models import create_causal_model
from causalab.tasks.graph_walk.config import GraphWalkConfig
from causalab.tasks.graph_walk.graphs import build_graph


class TestCausalModelStructure:
    def test_has_required_variables(self):
        config = GraphWalkConfig(graph_type="ring", graph_size=5, context_length=10)
        model = create_causal_model(config)
        assert "raw_input" in model.variables
        assert "raw_output" in model.variables
        assert "node_coordinates" in model.variables
        assert "walk_sequence" in model.variables

    def test_node_coordinates_is_input(self):
        config = GraphWalkConfig(graph_type="ring", graph_size=5, context_length=10)
        model = create_causal_model(config)
        assert "node_coordinates" in model.inputs


class TestCausalModelWalkEndsAtNode:
    """The walk must end at the node matching the input coordinates."""

    @pytest.fixture
    def ring_setup(self):
        config = GraphWalkConfig(
            graph_type="ring",
            graph_size=5,
            context_length=10,
            seed=42,
        )
        model = create_causal_model(config)
        graph = build_graph("ring", 5)
        return model, graph, config

    def test_walk_ends_at_correct_node(self, ring_setup):
        model, graph, config = ring_setup
        coord_to_node = {tuple(graph.coordinates[i]): i for i in range(graph.n_nodes)}
        for _ in range(20):
            trace = model.sample_input()
            walk = trace["walk_sequence"]
            expected_node = coord_to_node[tuple(trace["node_coordinates"])]
            assert walk[-1] == expected_node, (
                f"Walk ends at {walk[-1]} but expected node {expected_node} "
                f"for coordinates {trace['node_coordinates']}"
            )

    def test_walk_length_equals_context_length(self, ring_setup):
        model, graph, config = ring_setup
        trace = model.sample_input()
        walk = trace["walk_sequence"]
        assert len(walk) == config.context_length

    def test_walk_steps_are_valid(self, ring_setup):
        """Every step in the walk must follow a graph edge."""
        model, graph, config = ring_setup
        for _ in range(20):
            trace = model.sample_input()
            walk = trace["walk_sequence"]
            for i in range(len(walk) - 1):
                assert walk[i + 1] in graph.adjacency[walk[i]], (
                    f"Invalid walk step: {walk[i]} -> {walk[i + 1]}"
                )


class TestCausalModelOutput:
    def test_raw_output_contains_valid_neighbors(self):
        config = GraphWalkConfig(
            graph_type="grid",
            graph_size=3,
            context_length=10,
            seed=42,
        )
        model = create_causal_model(config)
        graph = build_graph("grid", 3)
        coord_to_node = {tuple(graph.coordinates[i]): i for i in range(graph.n_nodes)}

        for _ in range(20):
            trace = model.sample_input()
            node_idx = coord_to_node[tuple(trace["node_coordinates"])]
            expected = [config.concepts[n] for n in graph.adjacency[node_idx]]
            assert set(trace["raw_output"]) == set(expected)


class TestCausalModelTextFormat:
    def test_raw_input_is_separator_joined_with_trailing_sep(self):
        config = GraphWalkConfig(
            graph_type="ring",
            graph_size=5,
            context_length=5,
            separator=",",
            seed=42,
        )
        model = create_causal_model(config)

        trace = model.sample_input()
        raw_input = trace["raw_input"]
        # Trailing separator means the text ends with ","
        assert raw_input.endswith(","), f"Expected trailing separator: {raw_input}"
        parts = raw_input.rstrip(",").split(",")
        assert len(parts) == 5, f"Expected 5 parts, got {len(parts)}: {raw_input}"

        # Each part should be a valid concept
        for part in parts:
            assert part in config.concepts, f"Unknown concept in text: {part!r}"

    def test_custom_separator(self):
        config = GraphWalkConfig(
            graph_type="ring",
            graph_size=5,
            context_length=5,
            separator=" | ",
            seed=42,
        )
        model = create_causal_model(config)
        trace = model.sample_input()
        raw = trace["raw_input"]
        assert raw.endswith(" | ")
        parts = raw.rstrip(" | ").split(" | ")
        assert len(parts) == 5

    def test_text_concepts_match_walk_sequence(self):
        """The concepts in the text should correspond to the walk nodes."""
        config = GraphWalkConfig(
            graph_type="ring",
            graph_size=5,
            context_length=5,
            separator=",",
            seed=42,
        )
        model = create_causal_model(config)

        for _ in range(10):
            trace = model.sample_input()
            walk = trace["walk_sequence"]
            text_concepts = trace["raw_input"].rstrip(",").split(",")
            walk_concepts = [config.concepts[n] for n in walk]
            assert text_concepts == walk_concepts
