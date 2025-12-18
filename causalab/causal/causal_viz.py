"""
Visualization utilities for causal models.

This module provides interactive and static visualization methods for CausalModel objects,
including graph-based visualizations using Dash/Cytoscape and NetworkX/Matplotlib.
"""

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from dash import Dash, html
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto


class DEFAULT_COLORS:
    """default colors for displaying causal model with dash"""

    BASE_INPUT = "#68AF9C"
    BASE_OUTPUT = "#A0CCC0"
    SOURCE_INPUT = "#EC3B82"
    SOURCE_OUTPUT = "#F276A8"
    COUNTERFACTUAL_OUTPUT = "#A59FD9"


def _get_descendants(model, intervention, strict=True):
    """
    Breadth-first search to get variables affected by intervention.

    Parameters:
    -----------
    model : CausalModel
        The causal model to traverse.
    intervention : dict
        A dictionary mapping variables to their intervened values.
    strict : bool, optional
        Controls which descendant variables are included (default True):
        - If True: Only includes variables where ALL parents are in the intervention
          or are themselves descendants. This ensures only directly affected variables
          are included.
        - If False: Includes any variable that has at least one parent in the intervention
          or descendant set, even if other parents are unaffected.

    Returns:
    --------
    list
        List of variables affected by the intervention.
    """
    descendants = [v for v in intervention]
    current_paths = [v for v in intervention]
    covered = [v for v in intervention]
    while current_paths:
        variable = current_paths.pop(0)
        for c in model.children[variable]:
            if c in covered:  # don't loop
                continue
            covered.append(c)
            if all(p in descendants for p in model.parents[c]) or not strict:
                descendants.append(c)
                current_paths.append(c)
    return descendants


def display_structure(model, colors=DEFAULT_COLORS):
    """
    Visualize the structure of the causal model without any inputs.

    Creates an interactive Dash/Cytoscape visualization showing the graph structure
    of the causal model.

    Parameters:
    -----------
    model : CausalModel
        The causal model to visualize.
    colors : class, optional
        Color scheme for the visualization (default is DEFAULT_COLORS).

    Returns:
    --------
    None
        Launches an interactive Dash application in the browser.
    """
    variables = [
        {
            "data": {"id": var, "label": var},
            # Updated by automatic layout
            "position": {"x": 0, "y": 0},
            # As opposed to their inner values
            "classes": "variable",
        }
        for var in model.variables
    ]
    edges = [
        {"data": {"id": f"{parent}->{child}", "source": parent, "target": child}}
        for child in model.variables
        for parent in model.parents[child]
    ]

    app = Dash()

    app.layout = html.Div(
        [
            cyto.Cytoscape(
                id="causal-graph-visualization",
                elements=variables + edges,
                layout={"name": "breadthfirst", "roots": "#raw_input"},
                stylesheet=[
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
                    {"selector": ".variable", "style": {"text-valign": "top"}},
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
                ],
            )
        ]
    )

    app.run()


def display_forward_pass(model, inputs, intervention=None, colors=DEFAULT_COLORS):
    """
    Visualize a forward pass of the model.

    Creates an interactive Dash/Cytoscape visualization showing how values propagate
    through the model with optional interventions.

    Parameters:
    -----------
    model : CausalModel
        The causal model to visualize.
    inputs : dict
        A dictionary mapping input variables to their values.
    intervention : dict, optional
        A dictionary mapping variables to their intervened values (default is None).
    colors : class, optional
        Color scheme for the visualization (default is DEFAULT_COLORS).

    Returns:
    --------
    None
        Launches an interactive Dash application in the browser.
    """
    if intervention is None:
        intervention = {}
    outputs = model.run_forward({**inputs, **intervention})

    intervention_only = _get_descendants(model, intervention)
    counterfactual = _get_descendants(model, intervention, strict=False)

    variables = [
        {
            "data": {"id": var, "label": var},
            # Updated by automatic layout
            "position": {"x": 0, "y": 0},
            # As opposed to their inner values
            "classes": "variable",
        }
        for var in model.variables
    ]
    edges = [
        {"data": {"id": f"{parent}->{child}", "source": parent, "target": child}}
        for child in model.variables
        for parent in model.parents[child]
    ]

    app = Dash()

    app.layout = html.Div(
        [
            cyto.Cytoscape(
                id="causal-graph-visualization",
                elements=variables + edges,
                layout={"name": "breadthfirst", "roots": "#raw_input"},
                stylesheet=[
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
                    {
                        "selector": ".variable",
                        "style": {
                            "text-valign": "top",
                            # don't show variable when moving value!
                            "background-color": "white",
                        },
                    },
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
                ],
            ),
            # hidden trigger for onload so we can add nodes with values
            html.Div(id="hidden-onload-trigger", style={"display": "none"}),
        ]
    )

    def onload(
        _: str, elements: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Add values as inner text of variables nodes.
        Needs to be done after positions have been rendered.
        """
        # update layout to preset to fix node positions
        layout: dict[str, Any] = {"name": "preset"}
        for variable, value in outputs.items():
            if variable in intervention:  # intervention first bc it overrides base
                classes = "source_input"
            elif variable in inputs:
                classes = "base_input"
            elif variable in intervention_only:
                classes = "source_value"
            elif variable in counterfactual:
                classes = "counterfactual_value"
            else:  # variable wasn't affected by intervention, so it's a base value
                classes = "base_value"

            # get original variable node
            variable_node = [e for e in elements if e["data"]["id"] == variable][0]
            elements.append(
                {
                    "data": {"id": f"{variable}-value", "label": value},
                    "position": variable_node["position"],
                    "classes": classes,
                }
            )
        return layout, elements

    app.callback(
        [
            Output("causal-graph-visualization", "layout"),
            Output("causal-graph-visualization", "elements"),
        ],
        [Input("hidden-onload-trigger", "children")],
        [State("causal-graph-visualization", "elements")],
    )(onload)

    def on_moving_nodes(
        elements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        for var in model.variables:
            variable_node = [e for e in elements if e["data"]["id"] == var][0]
            value_node = [e for e in elements if e["data"]["id"] == f"{var}-value"][0]
            variable_node["position"] = value_node["position"]
        return elements

    app.callback(
        Output("causal-graph-visualization", "elements", allow_duplicate=True),
        Input("causal-graph-visualization", "elements"),
        prevent_initial_call=True,
    )(on_moving_nodes)

    app.run()


def display_interchange(model, inputs, counterfactual_inputs, colors=DEFAULT_COLORS):
    """
    Visualize an interchange intervention between a base input and counterfactual input(s).

    Creates an interactive Dash/Cytoscape visualization comparing the base and counterfactual
    execution paths.

    Parameters:
    -----------
    model : CausalModel
        The causal model to visualize.
    inputs : dict
        A dictionary mapping input variables to their values for the base run.
    counterfactual_inputs : dict
        A dictionary mapping variables to their counterfactual input settings.
    colors : class, optional
        Color scheme for the visualization (default is DEFAULT_COLORS).

    Returns:
    --------
    None
        Launches an interactive Dash application in the browser.
    """
    outputs = model.run_interchange(inputs, counterfactual_inputs)

    intervention_only = _get_descendants(model, counterfactual_inputs)
    counterfactual = _get_descendants(model, counterfactual_inputs, strict=False)

    # CODE
    # set up base causal graph
    variables = [
        {
            "data": {"id": var, "label": var},
            # Updated by automatic layout
            "position": {"x": 0, "y": 0},
            # As opposed to their inner values
            "classes": "variable",
        }
        for var in model.variables
    ]
    edges = [
        {"data": {"id": f"{parent}->{child}", "source": parent, "target": child}}
        for child in model.variables
        for parent in model.parents[child]
    ]

    # set up source causal graphs
    for i, source in enumerate(counterfactual_inputs.keys()):
        source_variables = [
            {
                "data": {"id": f"{var}-source-{i}", "label": var},
                "position": {"x": 0, "y": 0},  # updated by automatic layout
                "classes": "variable",  # as opposed to their inner values
            }
            for var in model.variables
        ]
        source_edges = [
            {
                "data": {
                    "id": f"{parent}->{child}-source-{i}",
                    "source": f"{parent}-source-{i}",
                    "target": f"{child}-source-{i}",
                }
            }
            for child in model.variables
            for parent in model.parents[child]
        ]
        intervention_edge = [
            {
                "data": {
                    "id": f"interchange-{i}",
                    "source": f"{source}-source-{i}",
                    "target": source,
                },
                "classes": "interchange_edge",
            }
        ]
        variables += source_variables
        edges += source_edges + intervention_edge

    # set raw input as source for each graph
    roots = ["#raw_input"] + [
        f"#raw_input-source-{i}" for i in range(len(counterfactual_inputs))
    ]

    app = Dash()

    app.layout = html.Div(
        [
            cyto.Cytoscape(
                id="causal-graph-visualization",
                elements=variables + edges,
                layout={"name": "breadthfirst", "roots": ",".join(roots)},
                stylesheet=[
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
                    {
                        "selector": ".variable",
                        "style": {"text-valign": "top", "background-color": "white"},
                    },
                    # color variables by value
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
                    # color interchange arrows
                    {
                        "selector": ".interchange_edge",
                        "style": {
                            "curve-style": "unbundled-bezier",
                            "line-color": colors.SOURCE_INPUT,
                            "target-arrow-color": colors.SOURCE_INPUT,
                            "control-point-distances": "80 -80 80",
                        },
                    },
                ],
            ),
            # hidden trigger for onload so we can add nodes with values
            html.Div(id="hidden-onload-trigger", style={"display": "none"}),
        ]
    )

    def onload(
        _: str, elements: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Add values as inner text of variables nodes.
        Needs to be done after positions have been rendered.
        """
        # update layout to preset to fix node positions
        layout: dict[str, Any] = {"name": "preset"}

        # set up source graphs
        for i, source_inputs in enumerate(counterfactual_inputs.values()):
            source_outputs = model.run_forward(source_inputs)
            for variable, value in source_outputs.items():
                if variable in source_inputs:
                    classes = "source_input"
                else:
                    classes = "source_value"

                variable_node = [
                    e for e in elements if e["data"]["id"] == f"{variable}-source-{i}"
                ][0]
                elements.append(
                    {
                        "data": {"id": f"{variable}-source-{i}-value", "label": value},
                        "position": variable_node["position"],
                        "classes": classes,
                    }
                )

        # set up base graphs
        for variable, value in outputs.items():
            if (
                variable in counterfactual_inputs
            ):  # intervention first bc it overrides base
                classes = "source_input"
            elif variable in inputs:
                classes = "base_input"
            elif variable in intervention_only:
                classes = "source_value"
            elif variable in counterfactual:
                classes = "counterfactual_value"
            else:  # variable wasn't affected by intervention, so it's a base value
                classes = "base_value"

            # get original variable node
            variable_node = [e for e in elements if e["data"]["id"] == variable][0]
            elements.append(
                {
                    "data": {"id": f"{variable}-value", "label": value},
                    "position": variable_node["position"],
                    "classes": classes,
                }
            )
        return layout, elements

    app.callback(
        [
            Output("causal-graph-visualization", "layout"),
            Output("causal-graph-visualization", "elements"),
        ],
        [Input("hidden-onload-trigger", "children")],
        [State("causal-graph-visualization", "elements")],
    )(onload)

    def on_moving_nodes(
        elements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """make sure variable nodes follow their respective value nodes"""
        for variable_node in elements:
            var_id = variable_node["data"]["id"]
            if not var_id.endswith("value") and "label" in variable_node["data"]:
                value_node = [
                    e for e in elements if e["data"]["id"] == f"{var_id}-value"
                ][0]
                variable_node["position"] = value_node["position"]
        return elements

    app.callback(
        Output("causal-graph-visualization", "elements", allow_duplicate=True),
        Input("causal-graph-visualization", "elements"),
        prevent_initial_call=True,
    )(on_moving_nodes)

    app.run()


def print_structure(model, font=12, node_size=1000):
    """
    Print the graph structure of the causal model.

    Creates a static matplotlib visualization of the causal graph using NetworkX.

    Parameters:
    -----------
    model : CausalModel
        The causal model to visualize.
    font : int, optional
        Font size for node labels (default is 12).
    node_size : int, optional
        Size of nodes in the graph (default is 1000).

    Returns:
    --------
    None
        Displays a matplotlib figure showing the graph structure.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (parent, child)
            for child in model.variables
            for parent in model.parents[child]
        ]
    )
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(
        graph,
        with_labels=True,
        node_color="green",
        pos=model.print_pos,
        font_size=font,
        node_size=node_size,
    )
    plt.show()


def print_setting(model, total_setting, font=12, node_size=1000):
    """
    Print the graph with variable values.

    Creates a static matplotlib visualization showing the causal graph with variable
    values displayed as labels using NetworkX.

    Parameters:
    -----------
    model : CausalModel
        The causal model to visualize.
    total_setting : dict
        A dictionary mapping variables to their values.
    font : int, optional
        Font size for node labels (default is 12).
    node_size : int, optional
        Size of nodes in the graph (default is 1000).

    Returns:
    --------
    None
        Displays a matplotlib figure showing the graph structure with values.
    """
    relabeler = {var: var + ":\n " + str(total_setting[var]) for var in model.variables}
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (parent, child)
            for child in model.variables
            for parent in model.parents[child]
        ]
    )
    plt.figure(figsize=(10, 10))
    graph = nx.relabel_nodes(graph, relabeler)
    newpos = dict()
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
    plt.show()
