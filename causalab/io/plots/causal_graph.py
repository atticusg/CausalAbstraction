"""
Visualization primitives for CausalModel.

Two output families:

- Dash/Cytoscape interactive graphs (``build_*_app`` / ``display_*``)
- Matplotlib/NetworkX static figures (``build_*_figure`` / ``print_*``)

Each family is factored into pure builders that return data/objects and thin
``display_*`` / ``print_*`` wrappers that actually launch a server or call
``plt.show``. Tests can exercise the builders without binding a port or opening
a window.
"""

from typing import TYPE_CHECKING, Any, Callable

import matplotlib.pyplot as plt
import networkx as nx
from dash import Dash, html
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from causalab.causal.causal_model import CausalModel
    from causalab.causal.trace import CausalTrace


class DEFAULT_COLORS:
    """Palette for Dash/Cytoscape causal-graph views."""

    BASE_INPUT = "#68AF9C"
    BASE_OUTPUT = "#A0CCC0"
    SOURCE_INPUT = "#EC3B82"
    SOURCE_OUTPUT = "#F276A8"
    COUNTERFACTUAL_OUTPUT = "#A59FD9"


def _get_descendants(
    model: "CausalModel", intervention: dict[str, Any], strict: bool = True
) -> list[str]:
    """BFS the variables affected by ``intervention``.

    With ``strict=True`` a variable is included only when *all* of its parents
    are already in the descendant set; with ``strict=False`` any variable whose
    parent set intersects the descendants is included.
    """
    descendants = [v for v in intervention]
    current_paths = [v for v in intervention]
    covered = [v for v in intervention]
    while current_paths:
        variable = current_paths.pop(0)
        for c in model.children[variable]:
            if c in covered:
                continue
            covered.append(c)
            if all(p in descendants for p in model.parents[c]) or not strict:
                descendants.append(c)
                current_paths.append(c)
    return descendants


# --------------------------------------------------------------------------- #
# Cytoscape element builders
# --------------------------------------------------------------------------- #


def build_variable_nodes(
    model: "CausalModel", *, suffix: str = ""
) -> list[dict[str, Any]]:
    """Cytoscape nodes for each variable in ``model``.

    ``suffix`` (e.g. ``"-source-0"``) is appended to node ids so multiple copies
    of the same DAG can coexist on one canvas without id collisions.
    """
    return [
        {
            "data": {"id": f"{var}{suffix}", "label": var},
            "position": {"x": 0, "y": 0},
            "classes": "variable",
        }
        for var in model.variables
    ]


def build_edges(model: "CausalModel", *, suffix: str = "") -> list[dict[str, Any]]:
    """Cytoscape edges for ``model.parents``, with optional id suffix."""
    return [
        {
            "data": {
                "id": f"{parent}->{child}{suffix}",
                "source": f"{parent}{suffix}",
                "target": f"{child}{suffix}",
            }
        }
        for child in model.variables
        for parent in model.parents[child]
    ]


def build_interchange_subgraphs(
    model: "CausalModel", counterfactual_inputs: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One source-DAG copy per counterfactual key, plus an interchange edge each.

    Returns ``(source_nodes, source_edges_with_interchange)``.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for i, source in enumerate(counterfactual_inputs.keys()):
        suffix = f"-source-{i}"
        nodes += build_variable_nodes(model, suffix=suffix)
        edges += build_edges(model, suffix=suffix)
        edges.append(
            {
                "data": {
                    "id": f"interchange-{i}",
                    "source": f"{source}{suffix}",
                    "target": source,
                },
                "classes": "interchange_edge",
            }
        )
    return nodes, edges


def classify_forward_node(
    var: str,
    *,
    inputs: dict[str, Any],
    intervention: dict[str, Any],
    intervention_only: list[str],
    counterfactual: list[str],
) -> str:
    """Cytoscape class for a value node under a (base + intervention) forward pass.

    Precedence: intervention overrides base, base inputs are next, then
    intervention-only descendants, then mixed (counterfactual) descendants,
    finally untouched base values.
    """
    if var in intervention:
        return "source_input"
    if var in inputs:
        return "base_input"
    if var in intervention_only:
        return "source_value"
    if var in counterfactual:
        return "counterfactual_value"
    return "base_value"


def build_stylesheet(
    colors: type[DEFAULT_COLORS],
    *,
    hide_variable_when_value_present: bool,
    include_interchange_edge: bool,
) -> list[dict[str, Any]]:
    """Cytoscape stylesheet shared across all three Dash views.

    ``hide_variable_when_value_present`` paints the bare variable node white
    so the value node (added on load) is what the user sees. The structure-only
    view leaves variable nodes visible.
    """
    variable_style: dict[str, Any] = {"text-valign": "top"}
    if hide_variable_when_value_present:
        variable_style["background-color"] = "white"

    stylesheet: list[dict[str, Any]] = [
        {
            "selector": "node",
            "style": {"content": "data(label)", "width": 50, "height": 50},
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "straight",
                "target-arrow-shape": "triangle",
            },
        },
        {"selector": ".variable", "style": variable_style},
        {
            "selector": ".base_input",
            "style": {
                "background-color": colors.BASE_INPUT,
                "color": "white",
                "text-valign": "center",
            },
        },
        {
            "selector": ".base_value",
            "style": {
                "background-color": colors.BASE_OUTPUT,
                "text-valign": "center",
            },
        },
        {
            "selector": ".source_input",
            "style": {
                "background-color": colors.SOURCE_INPUT,
                "color": "white",
                "text-valign": "center",
            },
        },
        {
            "selector": ".source_value",
            "style": {
                "background-color": colors.SOURCE_OUTPUT,
                "text-valign": "center",
            },
        },
        {
            "selector": ".counterfactual_value",
            "style": {
                "background-color": colors.COUNTERFACTUAL_OUTPUT,
                "text-valign": "center",
            },
        },
    ]
    if include_interchange_edge:
        stylesheet.append(
            {
                "selector": ".interchange_edge",
                "style": {
                    "curve-style": "unbundled-bezier",
                    "line-color": colors.SOURCE_INPUT,
                    "target-arrow-color": colors.SOURCE_INPUT,
                    "control-point-distances": "80 -80 80",
                },
            }
        )
    return stylesheet


# --------------------------------------------------------------------------- #
# Dash callback factories (pure functions returning callables — testable)
# --------------------------------------------------------------------------- #


def make_forward_pass_onload(
    model: "CausalModel",
    *,
    outputs: dict[str, Any],
    inputs: dict[str, Any],
    intervention: dict[str, Any],
) -> Callable[[str, list[dict[str, Any]]], tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Closure used as the ``hidden-onload-trigger`` callback for forward-pass views.

    Captures the precomputed forward-pass outputs and the intervention metadata
    so the callback only needs the rendered ``elements`` list at call time.
    """
    intervention_only = _get_descendants(model, intervention)
    counterfactual = _get_descendants(model, intervention, strict=False)

    def onload(
        _: str, elements: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        layout: dict[str, Any] = {"name": "preset"}
        for variable, value in outputs.items():
            classes = classify_forward_node(
                variable,
                inputs=inputs,
                intervention=intervention,
                intervention_only=intervention_only,
                counterfactual=counterfactual,
            )
            variable_node = [e for e in elements if e["data"]["id"] == variable][0]
            elements.append(
                {
                    "data": {"id": f"{variable}-value", "label": value},
                    "position": variable_node["position"],
                    "classes": classes,
                }
            )
        return layout, elements

    return onload


def make_interchange_onload(
    model: "CausalModel",
    *,
    outputs: dict[str, Any],
    inputs: "dict[str, Any] | CausalTrace",
    counterfactual_inputs: dict[str, Any],
    cf_traces: "dict[str, CausalTrace]",
) -> Callable[[str, list[dict[str, Any]]], tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Onload callback for the interchange view.

    Labels both the base DAG and each per-key source DAG. Base-DAG class
    assignment uses :func:`classify_forward_node` with ``intervention =
    counterfactual_inputs``.
    """
    intervention_only = _get_descendants(model, counterfactual_inputs)
    counterfactual = _get_descendants(model, counterfactual_inputs, strict=False)

    def onload(
        _: str, elements: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        layout: dict[str, Any] = {"name": "preset"}
        elements_by_id = {e["data"]["id"]: e for e in elements}

        # Per-key source DAGs.
        for i, (cf_key, source_trace) in enumerate(cf_traces.items()):
            source_outputs = source_trace.to_dict()
            original_inputs = counterfactual_inputs[cf_key]
            for variable, value in source_outputs.items():
                classes = (
                    "source_input" if variable in original_inputs else "source_value"
                )
                variable_node = elements_by_id.get(f"{variable}-source-{i}")
                if variable_node:
                    elements.append(
                        {
                            "data": {
                                "id": f"{variable}-source-{i}-value",
                                "label": value,
                            },
                            "position": variable_node["position"],
                            "classes": classes,
                        }
                    )

        # Base DAG.
        for variable, value in outputs.items():
            classes = classify_forward_node(
                variable,
                inputs=inputs,  # type: ignore[arg-type]
                intervention=counterfactual_inputs,
                intervention_only=intervention_only,
                counterfactual=counterfactual,
            )
            variable_node = elements_by_id.get(variable)
            if variable_node:
                elements.append(
                    {
                        "data": {"id": f"{variable}-value", "label": value},
                        "position": variable_node["position"],
                        "classes": classes,
                    }
                )
        return layout, elements

    return onload


def make_simple_on_moving_nodes(
    model: "CausalModel",
) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    """For forward-pass view: keep variable nodes anchored to their value nodes."""

    def on_moving_nodes(
        elements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        elements_by_id = {e["data"]["id"]: e for e in elements}
        for var in model.variables:
            variable_node = elements_by_id.get(var)
            value_node = elements_by_id.get(f"{var}-value")
            if variable_node and value_node:
                variable_node["position"] = value_node["position"]
        return elements

    return on_moving_nodes


def make_general_on_moving_nodes() -> Callable[
    [list[dict[str, Any]]], list[dict[str, Any]]
]:
    """For interchange view: anchor every labelled variable node (base and source DAGs)."""

    def on_moving_nodes(
        elements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        elements_by_id = {e["data"]["id"]: e for e in elements}
        for variable_node in elements:
            var_id = variable_node["data"]["id"]
            if not var_id.endswith("value") and "label" in variable_node["data"]:
                value_node = elements_by_id.get(f"{var_id}-value")
                if value_node:
                    variable_node["position"] = value_node["position"]
        return elements

    return on_moving_nodes


# --------------------------------------------------------------------------- #
# Dash app builders
# --------------------------------------------------------------------------- #


def _cytoscape(
    elements: list[dict[str, Any]],
    *,
    roots: str,
    stylesheet: list[dict[str, Any]],
) -> cyto.Cytoscape:
    return cyto.Cytoscape(
        id="causal-graph-visualization",
        elements=elements,
        layout={"name": "breadthfirst", "roots": roots},
        stylesheet=stylesheet,
    )


def build_structure_app(
    model: "CausalModel", colors: type[DEFAULT_COLORS] = DEFAULT_COLORS
) -> Dash:
    """Dash app showing only the DAG structure, no inputs."""
    elements = build_variable_nodes(model) + build_edges(model)
    stylesheet = build_stylesheet(
        colors,
        hide_variable_when_value_present=False,
        include_interchange_edge=False,
    )
    app = Dash()
    app.layout = html.Div(
        [_cytoscape(elements, roots="#raw_input", stylesheet=stylesheet)]
    )
    return app


def build_forward_pass_app(
    model: "CausalModel",
    inputs: dict[str, Any],
    intervention: dict[str, Any] | None = None,
    colors: type[DEFAULT_COLORS] = DEFAULT_COLORS,
) -> Dash:
    """Dash app for a forward pass (optionally with an intervention)."""
    if intervention is None:
        intervention = {}

    trace = model.new_trace(inputs)
    for var, val in intervention.items():
        trace.intervene(var, val)
    outputs = trace.to_dict()

    elements = build_variable_nodes(model) + build_edges(model)
    stylesheet = build_stylesheet(
        colors,
        hide_variable_when_value_present=True,
        include_interchange_edge=False,
    )

    app = Dash()
    app.layout = html.Div(
        [
            _cytoscape(elements, roots="#raw_input", stylesheet=stylesheet),
            html.Div(id="hidden-onload-trigger", style={"display": "none"}),
        ]
    )

    onload = make_forward_pass_onload(
        model, outputs=outputs, inputs=inputs, intervention=intervention
    )
    app.callback(
        [
            Output("causal-graph-visualization", "layout"),
            Output("causal-graph-visualization", "elements"),
        ],
        [Input("hidden-onload-trigger", "children")],
        [State("causal-graph-visualization", "elements")],
    )(onload)

    on_moving_nodes = make_simple_on_moving_nodes(model)
    app.callback(
        Output("causal-graph-visualization", "elements", allow_duplicate=True),
        Input("causal-graph-visualization", "elements"),
        prevent_initial_call=True,
    )(on_moving_nodes)

    return app


def build_interchange_app(
    model: "CausalModel",
    inputs: "dict[str, Any] | CausalTrace",
    counterfactual_inputs: dict[str, Any],
    colors: type[DEFAULT_COLORS] = DEFAULT_COLORS,
) -> Dash:
    """Dash app comparing the base run against per-key counterfactual sources."""
    input_trace = model.new_trace(inputs) if isinstance(inputs, dict) else inputs

    cf_traces: "dict[str, CausalTrace]" = {}
    for var, cf_input in counterfactual_inputs.items():
        cf_traces[var] = (
            model.new_trace(cf_input) if isinstance(cf_input, dict) else cf_input
        )

    result_trace = input_trace.copy()
    for var, cf_trace in cf_traces.items():
        result_trace.intervene(var, cf_trace[var])
    outputs = result_trace.to_dict()

    base_nodes = build_variable_nodes(model)
    base_edges = build_edges(model)
    source_nodes, source_edges = build_interchange_subgraphs(
        model, counterfactual_inputs
    )
    elements = base_nodes + source_nodes + base_edges + source_edges

    roots = ["#raw_input"] + [
        f"#raw_input-source-{i}" for i in range(len(counterfactual_inputs))
    ]
    stylesheet = build_stylesheet(
        colors,
        hide_variable_when_value_present=True,
        include_interchange_edge=True,
    )

    app = Dash()
    app.layout = html.Div(
        [
            _cytoscape(elements, roots=",".join(roots), stylesheet=stylesheet),
            html.Div(id="hidden-onload-trigger", style={"display": "none"}),
        ]
    )

    onload = make_interchange_onload(
        model,
        outputs=outputs,
        inputs=inputs,
        counterfactual_inputs=counterfactual_inputs,
        cf_traces=cf_traces,
    )
    app.callback(
        [
            Output("causal-graph-visualization", "layout"),
            Output("causal-graph-visualization", "elements"),
        ],
        [Input("hidden-onload-trigger", "children")],
        [State("causal-graph-visualization", "elements")],
    )(onload)

    on_moving_nodes = make_general_on_moving_nodes()
    app.callback(
        Output("causal-graph-visualization", "elements", allow_duplicate=True),
        Input("causal-graph-visualization", "elements"),
        prevent_initial_call=True,
    )(on_moving_nodes)

    return app


# --------------------------------------------------------------------------- #
# Dash display wrappers (launch the server)
# --------------------------------------------------------------------------- #


def display_structure(
    model: "CausalModel", colors: type[DEFAULT_COLORS] = DEFAULT_COLORS
) -> None:
    """Launch the structure-only Dash app in the browser."""
    build_structure_app(model, colors).run()


def display_forward_pass(
    model: "CausalModel",
    inputs: dict[str, Any],
    intervention: dict[str, Any] | None = None,
    colors: type[DEFAULT_COLORS] = DEFAULT_COLORS,
) -> None:
    """Launch the forward-pass Dash app in the browser."""
    build_forward_pass_app(model, inputs, intervention, colors).run()


def display_interchange(
    model: "CausalModel",
    inputs: "dict[str, Any] | CausalTrace",
    counterfactual_inputs: dict[str, Any],
    colors: type[DEFAULT_COLORS] = DEFAULT_COLORS,
) -> None:
    """Launch the interchange Dash app in the browser."""
    build_interchange_app(model, inputs, counterfactual_inputs, colors).run()


# --------------------------------------------------------------------------- #
# Matplotlib / NetworkX builders
# --------------------------------------------------------------------------- #


def _model_digraph(model: "CausalModel") -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (parent, child)
            for child in model.variables
            for parent in model.parents[child]
        ]
    )
    return graph


def build_structure_figure(
    model: "CausalModel", font: int = 12, node_size: int = 1000
) -> "Figure":
    """Build (but do not show) a matplotlib figure of the DAG structure."""
    graph = _model_digraph(model)
    figure = plt.figure(figsize=(10, 10))
    nx.draw_networkx(
        graph,
        with_labels=True,
        node_color="green",
        pos=model.print_pos,
        font_size=font,
        node_size=node_size,
    )
    return figure


def build_setting_figure(
    model: "CausalModel",
    total_setting: dict[str, Any],
    font: int = 12,
    node_size: int = 1000,
) -> "Figure":
    """Build (but do not show) a matplotlib figure of the DAG with values relabeled."""
    relabeler = {var: var + ":\n " + str(total_setting[var]) for var in model.variables}
    graph = nx.relabel_nodes(_model_digraph(model), relabeler)
    figure = plt.figure(figsize=(10, 10))
    newpos: dict[str, tuple[int, int]] = {}
    if model.print_pos is not None:
        for var in model.print_pos:
            newpos[relabeler[var]] = model.print_pos[var]
    nx.draw_networkx(
        graph,
        with_labels=True,
        node_color="green",
        pos=newpos,
        font_size=font,
        node_size=node_size,
    )
    return figure


def print_structure(
    model: "CausalModel", font: int = 12, node_size: int = 1000
) -> None:
    """Show the matplotlib DAG figure."""
    build_structure_figure(model, font=font, node_size=node_size)
    plt.show()


def print_setting(
    model: "CausalModel",
    total_setting: dict[str, Any],
    font: int = 12,
    node_size: int = 1000,
) -> None:
    """Show the matplotlib DAG figure with values relabeled into node strings."""
    build_setting_figure(model, total_setting, font=font, node_size=node_size)
    plt.show()
