"""Graph walk task for in-context learning of graph representations.

Replicates PCA visualization results from "ICLR: In-Context Learning of
Representations" (Park et al., ICLR 2025). When LLMs process random walk
sequences on graphs, their internal representations reorganize to reflect
the underlying graph structure.

Example usage::

    from causalab.tasks.graph_walk import GraphWalkConfig, build_graph, create_causal_model

    config = GraphWalkConfig(graph_type="ring", graph_size=10)
    graph = build_graph(config.graph_type, config.graph_size)
    model = create_causal_model(config)
    trace = model.sample_input()
    print(trace["raw_input"])
"""

from .causal_models import create_causal_model
from .config import TASK_NAME, GraphWalkConfig, DEFAULT_CONCEPTS
from .graphs import (
    Graph,
    build_graph,
    make_grid_graph,
    make_hex_graph,
    make_ring_graph,
    random_neighbor_pairs,
    random_walk,
)
from .token_positions import create_token_positions

__all__ = [
    "TASK_NAME",
    "GraphWalkConfig",
    "DEFAULT_CONCEPTS",
    "Graph",
    "build_graph",
    "make_grid_graph",
    "make_ring_graph",
    "make_hex_graph",
    "random_walk",
    "random_neighbor_pairs",
    "create_causal_model",
    "create_token_positions",
]
