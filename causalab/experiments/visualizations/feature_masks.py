"""
Visualization functions for feature count heatmaps (DBM with tie_masks=False).

These visualizations show feature counts (number of selected features) for mask-based
interventions where each model unit can select a subset of features.

Supports both:
- Single score mode: One accuracy value for all units
- Per-layer score mode: Separate accuracy value per layer

n_features can be specified as:
- int: Same number of features for all units
- Dict[str, int]: Per-unit n_features keyed by unit_id

Component types:
- Attention heads: (layer, head) grid
- Residual stream: (layer, token_position) grid
- MLPs: (layer, token_position) grid
"""

from typing import Any, Dict, List, Optional, Union, cast

import numpy as np

from .unit_id import (
    extract_layer_from_unit_id,
    extract_token_position_from_unit_id,
    extract_layer_head_from_unit_id,
    detect_component_type,
    is_per_layer_mode,
    extract_grid_dimensions,
    FeatureIndicesSingle,
    FeatureIndicesPerLayer,
    FeatureIndicesUnion,
)
from .utils import create_feature_count_heatmap

# Type alias for n_features parameter
NFeatures = Union[int, Dict[str, int]]


# =============================================================================
# Helper Functions
# =============================================================================


def _get_n_features_for_unit(n_features: NFeatures, unit_id: str) -> int:
    """
    Get n_features for a specific unit.

    Args:
        n_features: Either an int (same for all units) or Dict[str, int] (per-unit)
        unit_id: The unit ID to look up

    Returns:
        Number of features for this unit

    Raises:
        ValueError: If n_features is a dict and unit_id is not found
    """
    if isinstance(n_features, int):
        return n_features
    else:
        if unit_id not in n_features:
            raise ValueError(
                f"Unit ID '{unit_id}' not found in n_features dict. "
                f"Available keys: {list(n_features.keys())}"
            )
        return n_features[unit_id]


def count_selected_features(indices: Optional[List[int]], n_features: int) -> int:
    """
    Count the number of selected features from feature_indices.

    Args:
        indices: Feature indices from InterchangeTarget.get_feature_indices().
                 None = all features selected, [] = none selected.
        n_features: Total number of features in the unit.

    Returns:
        Number of selected features.
    """
    if indices is None:
        return n_features
    return len(indices)


# =============================================================================
# Matrix Building Functions
# =============================================================================


def _build_attention_head_matrix(
    feature_indices: FeatureIndicesUnion,
    layers: List[int],
    heads: List[int],
    n_features: NFeatures,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Build feature count matrix for attention heads."""
    num_layers = len(layers)
    num_heads = len(heads)
    count_matrix = np.zeros((num_layers, num_heads))

    if is_per_layer_mode(feature_indices):
        per_layer_fi = cast(FeatureIndicesPerLayer, feature_indices)
        for layer, layer_dict in per_layer_fi.items():
            if layer not in layers:
                continue
            layer_idx = layers.index(layer)
            for unit_id, indices in layer_dict.items():
                if "AttentionHead" not in unit_id:
                    continue
                try:
                    _, head = extract_layer_head_from_unit_id(unit_id)
                    if head in heads:
                        head_idx = heads.index(head)
                        unit_n_features = _get_n_features_for_unit(n_features, unit_id)
                        count_matrix[layer_idx, head_idx] = count_selected_features(
                            indices, unit_n_features
                        )
                except ValueError:
                    continue
    else:
        single_fi = cast(FeatureIndicesSingle, feature_indices)
        for unit_id, indices in single_fi.items():
            if "AttentionHead" not in unit_id:
                continue
            try:
                layer, head = extract_layer_head_from_unit_id(unit_id)
                if layer in layers and head in heads:
                    layer_idx = layers.index(layer)
                    head_idx = heads.index(head)
                    unit_n_features = _get_n_features_for_unit(n_features, unit_id)
                    count_matrix[layer_idx, head_idx] = count_selected_features(
                        indices, unit_n_features
                    )
            except ValueError:
                continue

    return count_matrix


def _build_position_based_matrix(
    feature_indices: FeatureIndicesUnion,
    layers: List[int],
    token_position_ids: List[str],
    n_features: NFeatures,
    component_marker: str,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Build feature count matrix for residual stream or MLP."""
    num_layers = len(layers)
    num_positions = len(token_position_ids)
    count_matrix = np.zeros((num_layers, num_positions))

    if is_per_layer_mode(feature_indices):
        per_layer_fi = cast(FeatureIndicesPerLayer, feature_indices)
        for layer, layer_dict in per_layer_fi.items():
            if layer not in layers:
                continue
            layer_idx = layers.index(layer)
            for unit_id, indices in layer_dict.items():
                if component_marker not in unit_id:
                    continue
                try:
                    token_pos_id = extract_token_position_from_unit_id(unit_id)
                    if token_pos_id in token_position_ids:
                        pos_idx = token_position_ids.index(token_pos_id)
                        unit_n_features = _get_n_features_for_unit(n_features, unit_id)
                        count_matrix[layer_idx, pos_idx] = count_selected_features(
                            indices, unit_n_features
                        )
                except ValueError:
                    continue
    else:
        single_fi = cast(FeatureIndicesSingle, feature_indices)
        for unit_id, indices in single_fi.items():
            if component_marker not in unit_id:
                continue
            try:
                layer = extract_layer_from_unit_id(unit_id)
                token_pos_id = extract_token_position_from_unit_id(unit_id)
                if layer in layers and token_pos_id in token_position_ids:
                    layer_idx = layers.index(layer)
                    pos_idx = token_position_ids.index(token_pos_id)
                    unit_n_features = _get_n_features_for_unit(n_features, unit_id)
                    count_matrix[layer_idx, pos_idx] = count_selected_features(
                        indices, unit_n_features
                    )
            except ValueError:
                continue

    return count_matrix


# =============================================================================
# Public Plotting Functions
# =============================================================================


def plot_attention_head_feature_counts(
    feature_indices: Union[
        Dict[str, Optional[List[int]]], Dict[int, Dict[str, Optional[List[int]]]]
    ],
    scores: Union[float, Dict[int, float]],
    layers: List[int],
    heads: List[int],
    n_features: NFeatures,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a full grid heatmap of feature counts for attention heads.

    Supports both single score mode (one accuracy) and per-layer score mode.

    Args:
        feature_indices: Either:
            - Dict[str, Optional[List[int]]] for single mode
            - Dict[int, Dict[str, Optional[List[int]]]] for per-layer mode
        scores: Either:
            - float for single overall accuracy
            - Dict[int, float] for per-layer accuracies
        layers: List of layer indices (y-axis).
        heads: List of head indices (x-axis).
        n_features: Total number of features per head (head_dim).
            Can be int (same for all) or Dict[str, int] (per-unit).
        title: Optional custom title.
        save_path: Optional path to save figure.
    """
    count_matrix = _build_attention_head_matrix(
        feature_indices, layers, heads, n_features
    )

    x_labels = [f"H{h}" for h in heads]
    y_labels = [f"L{layer}" for layer in layers]

    has_per_layer_scores = isinstance(scores, dict)
    show_accuracy_column = has_per_layer_scores

    create_feature_count_heatmap(
        count_matrix=count_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        scores=scores,
        layers=layers,
        title=title or "Attention Heads: Features Selected",
        xlabel="Head",
        ylabel="Layer",
        colorbar_label="Feature Count",
        save_path=save_path,
        flip_vertical=True,  # Lowest layer at bottom, highest at top
        figsize=(max(12, len(heads) * 0.5 + 2), max(6, len(layers) * 0.4)),
        show_accuracy_column=show_accuracy_column,
    )


def plot_residual_stream_feature_counts(
    feature_indices: Union[
        Dict[str, Optional[List[int]]], Dict[int, Dict[str, Optional[List[int]]]]
    ],
    scores: Union[float, Dict[int, float]],
    layers: List[int],
    token_position_ids: List[str],
    n_features: NFeatures,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    score_label: str = "Acc",
) -> None:
    """
    Plot a tokens x layers heatmap for residual stream with accuracy column.

    Supports both single score mode and per-layer score mode.

    Args:
        feature_indices: Either:
            - Dict[str, Optional[List[int]]] for single mode
            - Dict[int, Dict[str, Optional[List[int]]]] for per-layer mode
        scores: Either:
            - float for single overall accuracy
            - Dict[int, float] for per-layer accuracies
        layers: List of layer indices (will be displayed bottom-to-top).
        token_position_ids: List of token position IDs (x-axis).
        n_features: Total number of features per unit (hidden_size).
            Can be int (same for all) or Dict[str, int] (per-unit).
        title: Optional custom title.
        save_path: Optional path to save figure.
        score_label: Label for the accuracy column (default: "Acc").
    """
    count_matrix = _build_position_based_matrix(
        feature_indices, layers, token_position_ids, n_features, "ResidualStream"
    )

    y_labels = [f"L{layer}" if layer >= 0 else "Emb" for layer in layers]

    create_feature_count_heatmap(
        count_matrix=count_matrix,
        x_labels=token_position_ids,
        y_labels=y_labels,
        scores=scores,
        layers=layers,
        title=title or "Residual Stream: Features Selected",
        xlabel="Token Position",
        ylabel="Layer",
        score_label=score_label,
        colorbar_label="Features Selected",
        save_path=save_path,
        flip_vertical=True,
        figsize=(max(10, len(token_position_ids) * 0.8 + 2), max(6, len(layers) * 0.4)),
        show_accuracy_column=True,
    )


def plot_mlp_feature_counts(
    feature_indices: Union[
        Dict[str, Optional[List[int]]], Dict[int, Dict[str, Optional[List[int]]]]
    ],
    scores: Union[float, Dict[int, float]],
    layers: List[int],
    token_position_ids: List[str],
    n_features: NFeatures,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a tokens x layers heatmap of feature counts for MLPs with accuracy column.

    Supports both single score mode and per-layer score mode.

    Args:
        feature_indices: Either:
            - Dict[str, Optional[List[int]]] for single mode
            - Dict[int, Dict[str, Optional[List[int]]]] for per-layer mode
        scores: Either:
            - float for single overall accuracy
            - Dict[int, float] for per-layer accuracies
        layers: List of layer indices.
        token_position_ids: List of token position IDs (x-axis).
        n_features: Total number of features per MLP (hidden_size).
            Can be int (same for all) or Dict[str, int] (per-unit).
        title: Optional custom title.
        save_path: Optional path to save figure.
    """
    count_matrix = _build_position_based_matrix(
        feature_indices, layers, token_position_ids, n_features, "MLP"
    )

    y_labels = [f"L{layer}" for layer in layers]

    create_feature_count_heatmap(
        count_matrix=count_matrix,
        x_labels=token_position_ids,
        y_labels=y_labels,
        scores=scores,
        layers=layers,
        title=title or "MLPs: Features Selected",
        xlabel="Token Position",
        ylabel="Layer",
        colorbar_label="Features Selected",
        save_path=save_path,
        flip_vertical=True,
        figsize=(max(10, len(token_position_ids) * 0.8 + 2), max(6, len(layers) * 0.4)),
        show_accuracy_column=True,
    )


# =============================================================================
# Unified Dispatcher
# =============================================================================


def plot_feature_counts(
    feature_indices: Union[
        Dict[str, Optional[List[int]]], Dict[int, Dict[str, Optional[List[int]]]]
    ],
    scores: Union[float, Dict[int, float]],
    n_features: NFeatures,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot feature counts, auto-detecting component type and grid dimensions.

    This is a unified dispatcher that:
    1. Detects the component type from unit IDs
    2. Detects if scores/feature_indices are per-layer or single
    3. Extracts grid dimensions from the feature_indices
    4. Calls the appropriate plotting function

    Args:
        feature_indices: Either:
            - Dict[str, Optional[List[int]]] for single mode (unit_id -> indices)
            - Dict[int, Dict[str, Optional[List[int]]]] for per-layer mode (layer -> {unit_id -> indices})
        scores: Either:
            - float for single overall accuracy
            - Dict[int, float] for per-layer accuracies
        n_features: Total number of features per unit. Can be:
            - int: Same number of features for all units
            - Dict[str, int]: Per-unit n_features keyed by unit_id
        title: Optional custom title.
        save_path: Optional path to save figure.
    """
    component_type = detect_component_type(feature_indices)
    dims = extract_grid_dimensions(component_type, feature_indices)

    if component_type == "attention_head":
        plot_attention_head_feature_counts(
            feature_indices=feature_indices,
            scores=scores,
            layers=dims["layers"],
            heads=dims["heads"],
            n_features=n_features,
            title=title,
            save_path=save_path,
        )
    elif component_type == "residual_stream":
        plot_residual_stream_feature_counts(
            feature_indices=feature_indices,
            scores=scores,
            layers=dims["layers"],
            token_position_ids=dims["token_position_ids"],
            n_features=n_features,
            title=title,
            save_path=save_path,
        )
    elif component_type == "mlp":
        plot_mlp_feature_counts(
            feature_indices=feature_indices,
            scores=scores,
            layers=dims["layers"],
            token_position_ids=dims["token_position_ids"],
            n_features=n_features,
            title=title,
            save_path=save_path,
        )
    else:
        raise ValueError(f"Unknown component type: {component_type}")
