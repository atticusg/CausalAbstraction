"""Causal model for the graph_walk task.

The causal DAG is:
    graph_type, graph_size (fixed per run)
          |
    node_coordinates  (input: coordinates of the node the walk currently visits)
          |
    walk_sequence  (mechanism: random walk ending at that node)
          |
      raw_input  (mechanism: format walk as text)
          |
     raw_output  (mechanism: list of valid neighbor concept tokens)
"""

from __future__ import annotations

import random

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import CausalTrace, Mechanism, input_var

from .config import TASK_NAME, GraphWalkConfig
from .graphs import build_graph


def create_causal_model(config: GraphWalkConfig) -> CausalModel:
    """Create a causal model for the graph_walk task.

    Args:
        config: GraphWalkConfig specifying graph type, size, concepts, etc.

    Returns:
        CausalModel with variables: node_coordinates, walk_sequence,
        raw_input, raw_output.
    """
    graph = build_graph(config.graph_type, config.graph_size, config.graph_size_2)
    concepts = config.concepts
    node_ids = list(range(graph.n_nodes))
    rng = random.Random(config.seed)

    node_to_concept = {i: concepts[i] for i in range(graph.n_nodes)}

    # Pre-compute coordinate tuples and reverse mapping
    node_coordinates = [tuple(graph.coordinates[i]) for i in node_ids]
    coord_to_node = {coord: i for i, coord in enumerate(node_coordinates)}

    values: dict = {
        "node_coordinates": node_coordinates,
        "concepts": concepts,
        "walk_sequence": None,
        "raw_input": None,
        "raw_output": None,
    }

    def _node_idx(t: CausalTrace) -> int:
        """Look up node index from coordinates."""
        return coord_to_node[tuple(t["node_coordinates"])]

    def _compute_walk(t: CausalTrace) -> list[int]:
        """Generate a random walk of context_length steps ending at the node."""
        target = _node_idx(t)
        # Generate forward walk from target, then reverse so it ends at target
        path = graph.random_walk_fast(
            target,
            config.context_length,
            rng=rng,
            no_backtrack=config.no_backtrack,
        )
        path.reverse()
        return path

    def _format_raw_input(t: CausalTrace) -> str:
        """Format walk sequence as separator-joined concept string.

        Appends a trailing separator so that the last token position is a
        separator — matching the position where the model has just processed
        the final concept and is predicting the next one.
        """
        walk = t["walk_sequence"]
        sep = config.separator
        return sep.join(node_to_concept[node] for node in walk) + sep

    def _compute_raw_output(t: CausalTrace) -> list[str]:
        """Return list of valid next-token concept strings (neighbors of last node)."""
        node = _node_idx(t)
        return [node_to_concept[n] for n in graph.adjacency[node]]

    mechanisms = {
        "node_coordinates": input_var(node_coordinates),
        "walk_sequence": Mechanism(
            parents=["node_coordinates"],
            compute=_compute_walk,
        ),
        "raw_input": Mechanism(
            parents=["walk_sequence"],
            compute=_format_raw_input,
            lazy=True,  # expensive text formatting, only computed when accessed
        ),
        "raw_output": Mechanism(
            parents=["node_coordinates"],
            compute=_compute_raw_output,
            lazy=True,
        ),
    }

    # Compute periods from graph's periodic dimensions
    periods: dict[str, float] = {}
    if graph.periodic_dims:
        coords = values["node_coordinates"]
        n_dims = len(coords[0]) if coords else 0
        for dim, period in graph.periodic_dims.items():
            key = "node_coordinates" if n_dims == 1 else f"node_coordinates_{dim}"
            periods[key] = period

    # The model predicts the next node's *concept* string, not its coordinate.
    # Declare that mapping as a plain ``{coordinate: [concept]}`` map (#296):
    # node ``i``'s coordinate tuple → its concept. This replaces the former
    # ``id()``-keyed result_token_pattern whose object-identity lookup broke when
    # a coordinate tuple was reconstructed (the #258 graph_walk regression). The
    # probability path reads the concept forms; the derived (exact) checker
    # matches the generated concept literally.
    output_tokens = {
        "node_coordinates": {
            node_coordinates[i]: [concepts[i]] for i in range(graph.n_nodes)
        }
    }

    model = CausalModel(
        mechanisms,
        values,
        id=TASK_NAME,
        embeddings=EMBEDDINGS,
        periods=periods,
        output_tokens=output_tokens,
    )
    # Store for coordinate_names access; CausalModel doesn't declare _graph,
    # but the attribute is set dynamically here and read by downstream code.
    model._graph = graph  # pyright: ignore[reportAttributeAccessIssue]
    return model


# --- Standard exports for load_task() ---
CREATE_CAUSAL_MODEL = create_causal_model


# graph_walk coordinates serve as the natural embedding
def _embed_coordinates(v: tuple) -> list[float]:
    """Identity embedding: coordinates are already numeric vectors."""
    return list(float(x) for x in v)


EMBEDDINGS: dict = {"node_coordinates": _embed_coordinates}
CYCLIC_VARIABLES: set[str] = set()  # determined per-graph by GET_CYCLIC_VARIABLES
TARGET_VARIABLE = "node_coordinates"


def EXAMPLE_TO_CLASS(ex: dict) -> int:
    """Map example to class index (last node visited in walk)."""
    return ex["input"]["walk_sequence"][-1]


def GET_VARIABLE_VALUES(model: CausalModel) -> dict[str, list]:
    """Derive variable values from the model's causal graph structure."""
    return {"node_coordinates": model.values["node_coordinates"]}


def GET_PERIODIC_INFO(model: CausalModel) -> dict[str, float] | None:
    """Derive periodic info from the graph's periodic_dims attribute."""
    graph = getattr(model, "_graph", None)
    if graph is None or not graph.periodic_dims:
        return None
    coords = model.values["node_coordinates"]
    n_dims = len(coords[0]) if coords else 0
    periodic_info = {}
    for dim, period in graph.periodic_dims.items():
        key = "node_coordinates" if n_dims == 1 else f"node_coordinates_{dim}"
        periodic_info[key] = period
    return periodic_info


def SCORE_TOKEN_IDS_FROM_MODEL(pipeline, concepts: list[str]) -> list[int]:
    """Get token IDs for graph node concepts."""
    return [pipeline.tokenizer.encode(c, add_special_tokens=False)[0] for c in concepts]
