"""
Visualization functions for binary mask heatmaps (DBM with tie_masks=True).

These visualizations show which model units (attention heads, residual stream positions,
or MLPs) were selected by DBM training. Selected units have mask=1 (indices=None),
unselected units have mask=0 (indices=[]).

This module consolidates all binary mask plotting for different component types:
- Attention heads: (layer, head) grid
- Residual stream: (layer, token_position) grid
- MLPs: (layer, token_position) grid
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np
import re

from .utils import create_binary_mask_heatmap


# =============================================================================
# Unit ID Parsing Helpers
# =============================================================================


def _extract_layer_from_unit_id(unit_id: str) -> int:
    """
    Extract layer number from a unit ID string.

    Handles formats like:
    - "ResidualStream(Layer-5,...)"
    - "MLP(Layer-3,...)"
    - "AttentionHead(Layer-2,Head-5,...)"
    """
    match = re.search(r"Layer-(\d+)", unit_id)
    if not match:
        raise ValueError(
            f"Could not parse layer from unit_id: {unit_id}. "
            f"Expected format containing 'Layer-X'"
        )
    return int(match.group(1))


def _extract_token_position_from_unit_id(unit_id: str) -> str:
    """
    Extract token position ID from a unit ID string.

    Handles formats like:
    - "ResidualStream(Layer-5,block_output,Token-last_token)"
    - "MLP(Layer-3,mlp_output,Token-last_token)"
    """
    match = re.search(r"Token-([^)]+)", unit_id)
    if not match:
        raise ValueError(
            f"Could not parse token position from unit_id: {unit_id}. "
            f"Expected format containing 'Token-<token_position_id>'"
        )
    return match.group(1)


def extract_layer_head_from_unit_id(unit_id: str) -> Tuple[int, int]:
    """
    Extract (layer, head) from an AttentionHead unit ID string.

    Args:
        unit_id: Unit ID string like "AttentionHead(Layer-2,Head-5,...)"

    Returns:
        Tuple of (layer, head) as integers.
    """
    match = re.search(r"Layer-(\d+),Head-(\d+)", unit_id)
    if not match:
        raise ValueError(
            f"Could not parse layer and head from unit_id: {unit_id}. "
            f"Expected format: 'AttentionHead(Layer-X,Head-Y,...)'"
        )
    return int(match.group(1)), int(match.group(2))


# =============================================================================
# Component Type Detection and Grid Extraction
# =============================================================================


def _detect_component_type(feature_indices: Dict[str, Optional[List[int]]]) -> str:
    """
    Infer component type from unit IDs in feature_indices.

    Args:
        feature_indices: Dict from InterchangeTarget.get_feature_indices()

    Returns:
        One of: "attention_head", "residual_stream", "mlp"

    Raises:
        ValueError: If component type cannot be determined
    """
    if not feature_indices:
        raise ValueError("feature_indices is empty, cannot detect component type")

    # Get first unit ID
    unit_id = next(iter(feature_indices.keys()))

    if "AttentionHead" in unit_id:
        return "attention_head"
    elif "ResidualStream" in unit_id:
        return "residual_stream"
    elif "MLP" in unit_id:
        return "mlp"
    else:
        raise ValueError(f"Unknown component type in unit_id: {unit_id}")


def extract_grid_dimensions(
    component_type: str, feature_indices: Dict[str, Optional[List[int]]]
) -> Dict[str, Any]:
    """
    Extract grid dimensions (layers, heads, or token_positions) from unit IDs.

    Args:
        component_type: One of "attention_head", "residual_stream", "mlp"
        feature_indices: Dict from InterchangeTarget.get_feature_indices()

    Returns:
        Dict with grid dimensions:
        - attention_head: {"layers": [...], "heads": [...]}
        - residual_stream/mlp: {"layers": [...], "token_position_ids": [...]}
    """
    if component_type == "attention_head":
        layers, heads = set(), set()
        for unit_id in feature_indices.keys():
            if "AttentionHead" not in unit_id:
                continue
            try:
                layer, head = extract_layer_head_from_unit_id(unit_id)
                layers.add(layer)
                heads.add(head)
            except ValueError:
                continue
        return {"layers": sorted(layers), "heads": sorted(heads)}
    else:
        # residual_stream or mlp
        layers, positions = set(), set()
        component_marker = (
            "ResidualStream" if component_type == "residual_stream" else "MLP"
        )
        for unit_id in feature_indices.keys():
            if component_marker not in unit_id:
                continue
            try:
                layer = _extract_layer_from_unit_id(unit_id)
                position = _extract_token_position_from_unit_id(unit_id)
                layers.add(layer)
                positions.add(position)
            except ValueError:
                continue
        return {"layers": sorted(layers), "token_position_ids": sorted(positions)}


# =============================================================================
# Selection Extractors
# =============================================================================


def get_selected_heads(
    feature_indices: Mapping[str, Optional[List[int]]],
) -> List[Tuple[int, int]]:
    """
    Extract list of (layer, head) pairs that were selected by DBM.

    Args:
        feature_indices: Dict from InterchangeTarget.get_feature_indices().

    Returns:
        List of (layer, head) tuples for selected heads, sorted by layer then head.
    """
    selected = []

    for unit_id, indices in feature_indices.items():
        if "AttentionHead" not in unit_id:
            continue

        try:
            layer, head = extract_layer_head_from_unit_id(unit_id)

            # None means all features selected (mask=1)
            if indices is None:
                selected.append((layer, head))

        except ValueError:
            continue

    # Sort by layer then head
    selected.sort(key=lambda x: (x[0], x[1]))
    return selected


def get_selected_residual_positions(
    feature_indices: Dict[str, Optional[List[int]]],
) -> List[Tuple[int, str]]:
    """
    Extract list of (layer, token_position_id) pairs that were selected by DBM.

    Args:
        feature_indices: Dict from InterchangeTarget.get_feature_indices().

    Returns:
        List of (layer, token_position_id) tuples for selected positions,
        sorted by layer then position.
    """
    selected = []

    for unit_id, indices in feature_indices.items():
        if "ResidualStream" not in unit_id:
            continue

        try:
            layer = _extract_layer_from_unit_id(unit_id)
            position = _extract_token_position_from_unit_id(unit_id)

            # None means all features selected (mask=1)
            if indices is None:
                selected.append((layer, position))

        except ValueError:
            continue

    # Sort by layer then position
    selected.sort(key=lambda x: (x[0], x[1]))
    return selected


def get_selected_mlps(
    feature_indices: Dict[str, Optional[List[int]]],
) -> List[Tuple[int, str]]:
    """
    Extract list of (layer, token_position_id) pairs for selected MLPs.

    Args:
        feature_indices: Dict from InterchangeTarget.get_feature_indices().

    Returns:
        List of (layer, token_position_id) tuples for selected MLPs,
        sorted by layer then position.
    """
    selected = []

    for unit_id, indices in feature_indices.items():
        if "MLP" not in unit_id:
            continue

        try:
            layer = _extract_layer_from_unit_id(unit_id)
            position = _extract_token_position_from_unit_id(unit_id)

            # None means all features selected (mask=1)
            if indices is None:
                selected.append((layer, position))

        except ValueError:
            continue

    # Sort by layer then position
    selected.sort(key=lambda x: (x[0], x[1]))
    return selected


def get_selected_units(
    feature_indices: Dict[str, Optional[List[int]]],
) -> List[Tuple[Any, ...]]:
    """
    Extract list of selected units, auto-detecting component type.

    Args:
        feature_indices: Dict from InterchangeTarget.get_feature_indices().

    Returns:
        List of tuples for selected units:
        - attention_head: List[Tuple[int, int]] - (layer, head)
        - residual_stream: List[Tuple[int, str]] - (layer, token_position_id)
        - mlp: List[Tuple[int, str]] - (layer, token_position_id)
    """
    component_type = _detect_component_type(feature_indices)

    if component_type == "attention_head":
        return get_selected_heads(feature_indices)
    elif component_type == "residual_stream":
        return get_selected_residual_positions(feature_indices)
    elif component_type == "mlp":
        return get_selected_mlps(feature_indices)
    else:
        raise ValueError(f"Unknown component type: {component_type}")


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_attention_head_mask(
    feature_indices: Dict[str, Optional[List[int]]],
    layers: List[int],
    heads: List[int],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot binary mask showing which attention heads were selected by DBM.

    Args:
        feature_indices: Dict from InterchangeTarget.get_feature_indices().
                        None = selected, [] = not selected.
        layers: List of layer indices (y-axis).
        heads: List of head indices (x-axis).
        title: Optional custom title.
        save_path: Optional path to save figure.
    """
    # Build binary mask matrix
    mask_matrix = np.full((len(heads), len(layers)), np.nan)

    for unit_id, indices in feature_indices.items():
        if "AttentionHead" not in unit_id:
            continue

        try:
            layer, head = extract_layer_head_from_unit_id(unit_id)

            if layer in layers and head in heads:
                layer_idx = layers.index(layer)
                head_idx = heads.index(head)

                # Convert feature_indices to binary mask
                # None means all features selected (mask=1)
                # [] means no features selected (mask=0)
                mask_value = 1 if indices is None else 0
                mask_matrix[head_idx, layer_idx] = mask_value

        except ValueError:
            continue

    # Transpose so layers are on y-axis
    mask_matrix = mask_matrix.T

    # Format labels
    x_labels = [f"H{head}" for head in heads]
    y_labels = [f"L{layer}" for layer in layers]

    # Auto-generate title if not provided
    if title is None:
        num_selected = int(np.nansum(mask_matrix))
        num_total = int(np.sum(~np.isnan(mask_matrix)))
        title = f"DBM Attention Head Mask ({num_selected}/{num_total} heads selected)"

    # Create the mask heatmap
    create_binary_mask_heatmap(
        mask_matrix=mask_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        save_path=save_path,
        xlabel="Head",
        ylabel="Layer",
        figsize=(max(12, len(heads) * 0.6), max(6, len(layers) * 0.8)),
    )


def plot_residual_stream_mask(
    feature_indices: Dict[str, Optional[List[int]]],
    layers: List[int],
    token_position_ids: List[str],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot binary mask showing which residual stream positions were selected by DBM.

    Args:
        feature_indices: Dict from InterchangeTarget.get_feature_indices().
                        None = selected, [] = not selected.
        layers: List of layer indices (y-axis).
        token_position_ids: List of token position IDs (x-axis).
        title: Optional custom title.
        save_path: Optional path to save figure.
    """
    # Build binary mask matrix: rows=positions, cols=layers
    mask_matrix = np.full((len(token_position_ids), len(layers)), np.nan)

    for unit_id, indices in feature_indices.items():
        if "ResidualStream" not in unit_id:
            continue

        try:
            layer = _extract_layer_from_unit_id(unit_id)
            position = _extract_token_position_from_unit_id(unit_id)

            if layer in layers and position in token_position_ids:
                layer_idx = layers.index(layer)
                pos_idx = token_position_ids.index(position)

                # None means selected (mask=1), [] means not selected (mask=0)
                mask_value = 1 if indices is None else 0
                mask_matrix[pos_idx, layer_idx] = mask_value

        except ValueError:
            continue

    # Transpose so layers are on y-axis
    mask_matrix = mask_matrix.T

    # Format labels
    x_labels = token_position_ids
    y_labels = [f"L{layer}" if layer >= 0 else "Emb" for layer in layers]

    # Auto-generate title if not provided
    if title is None:
        num_selected = int(np.nansum(mask_matrix))
        num_total = int(np.sum(~np.isnan(mask_matrix)))
        title = (
            f"DBM Residual Stream Mask ({num_selected}/{num_total} positions selected)"
        )

    # Create the mask heatmap
    create_binary_mask_heatmap(
        mask_matrix=mask_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        save_path=save_path,
        xlabel="Token Position",
        ylabel="Layer",
        figsize=(max(10, len(token_position_ids) * 0.8), max(6, len(layers) * 0.4)),
    )


def plot_mlp_mask(
    feature_indices: Dict[str, Optional[List[int]]],
    layers: List[int],
    token_position_ids: List[str],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot binary mask showing which MLP units were selected by DBM.

    Args:
        feature_indices: Dict from InterchangeTarget.get_feature_indices().
                        None = selected, [] = not selected.
        layers: List of layer indices (y-axis).
        token_position_ids: List of token position IDs (x-axis).
        title: Optional custom title.
        save_path: Optional path to save figure.
    """
    # Build binary mask matrix: rows=positions, cols=layers
    mask_matrix = np.full((len(token_position_ids), len(layers)), np.nan)

    for unit_id, indices in feature_indices.items():
        if "MLP" not in unit_id:
            continue

        try:
            layer = _extract_layer_from_unit_id(unit_id)
            position = _extract_token_position_from_unit_id(unit_id)

            if layer in layers and position in token_position_ids:
                layer_idx = layers.index(layer)
                pos_idx = token_position_ids.index(position)

                # None means selected (mask=1), [] means not selected (mask=0)
                mask_value = 1 if indices is None else 0
                mask_matrix[pos_idx, layer_idx] = mask_value

        except ValueError:
            continue

    # Transpose so layers are on y-axis
    mask_matrix = mask_matrix.T

    # Format labels
    x_labels = token_position_ids
    y_labels = [f"L{layer}" for layer in layers]

    # Auto-generate title if not provided
    if title is None:
        num_selected = int(np.nansum(mask_matrix))
        num_total = int(np.sum(~np.isnan(mask_matrix)))
        title = f"DBM MLP Mask ({num_selected}/{num_total} units selected)"

    # Create the mask heatmap
    create_binary_mask_heatmap(
        mask_matrix=mask_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        save_path=save_path,
        xlabel="Token Position",
        ylabel="Layer",
        figsize=(max(10, len(token_position_ids) * 0.8), max(6, len(layers) * 0.4)),
    )


def plot_binary_mask(
    feature_indices: Dict[str, Optional[List[int]]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot binary mask, auto-detecting component type and grid dimensions.

    This is a unified dispatcher that:
    1. Detects the component type from unit IDs
    2. Extracts grid dimensions from the feature_indices
    3. Calls the appropriate plotting function

    Args:
        feature_indices: Dict from InterchangeTarget.get_feature_indices().
                        None = selected, [] = not selected.
        title: Optional custom title.
        save_path: Optional path to save figure.
    """
    # Detect component type
    component_type = _detect_component_type(feature_indices)

    # Extract grid dimensions
    dims = extract_grid_dimensions(component_type, feature_indices)

    # Dispatch to appropriate plotting function
    if component_type == "attention_head":
        plot_attention_head_mask(
            feature_indices=feature_indices,
            layers=dims["layers"],
            heads=dims["heads"],
            title=title,
            save_path=save_path,
        )
    elif component_type == "residual_stream":
        plot_residual_stream_mask(
            feature_indices=feature_indices,
            layers=dims["layers"],
            token_position_ids=dims["token_position_ids"],
            title=title,
            save_path=save_path,
        )
    elif component_type == "mlp":
        plot_mlp_mask(
            feature_indices=feature_indices,
            layers=dims["layers"],
            token_position_ids=dims["token_position_ids"],
            title=title,
            save_path=save_path,
        )
    else:
        raise ValueError(f"Unknown component type: {component_type}")
