"""
InterchangeTarget Builder and Utility Functions

This module provides:
1. Builder functions to create InterchangeTarget objects for different model unit types
   (attention heads, residual streams, MLPs)
2. Utility functions for detecting component types and extracting grid dimensions

Each builder returns a Dict[tuple, InterchangeTarget] with grouping controlled by the `mode` parameter:
- "one_target_all_units": {("all",): target} - Single target containing all units
- "one_target_per_unit": {(layer, pos_id): target, ...} - One target per unit
- "one_target_per_layer": {(layer,): target, ...} - One target per layer
"""

from typing import List, Dict, Tuple, Any

from causalab.neural.model_units import InterchangeTarget, AtomicModelUnit
from causalab.neural.LM_units import AttentionHead, ResidualStream, MLP
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_position_builder import TokenPosition


# =============================================================================
# Component Type Detection
# =============================================================================


def detect_component_type_from_targets(
    interchange_targets: Dict[Tuple[Any, ...], InterchangeTarget],
) -> str:
    """
    Detect component type from the first unit in an interchange targets dict.

    This function inspects the unit IDs in the first InterchangeTarget to determine
    whether the targets represent attention heads, residual stream positions, or MLPs.

    Args:
        interchange_targets: Dict mapping keys (e.g., (layer, head)) to InterchangeTarget objects

    Returns:
        One of: "attention_head", "residual_stream", "mlp"

    Raises:
        ValueError: If targets dict is empty or component type cannot be determined
    """
    if not interchange_targets:
        raise ValueError("interchange_targets dict is empty")

    # Get first target
    first_target = next(iter(interchange_targets.values()))
    return _detect_component_type_from_single_target(first_target)


def _detect_component_type_from_single_target(target: InterchangeTarget) -> str:
    """
    Internal helper to detect component type from a single InterchangeTarget.

    Args:
        target: InterchangeTarget object

    Returns:
        One of: "attention_head", "residual_stream", "mlp"

    Raises:
        ValueError: If target has no units or component type cannot be determined
    """
    units = target.flatten()
    if not units:
        raise ValueError("InterchangeTarget has no units")

    sample_unit_id = units[0].id
    if "AttentionHead" in sample_unit_id:
        return "attention_head"
    elif "ResidualStream" in sample_unit_id:
        return "residual_stream"
    elif "MLP" in sample_unit_id:
        return "mlp"
    else:
        raise ValueError(f"Unknown component type in unit_id: {sample_unit_id}")


# =============================================================================
# Grid Dimension Extraction
# =============================================================================


def extract_grid_dimensions_from_targets(
    component_type: str,
    interchange_targets: Dict[Tuple[Any, ...], InterchangeTarget],
) -> Dict[str, List[int]]:
    """
    Extract grid dimensions from the keys of an interchange targets dict.

    For one_target_per_unit mode:
    - attention_head: keys are (layer, head)
    - residual_stream/mlp: keys are (layer, token_position_id)

    Args:
        component_type: One of "attention_head", "residual_stream", "mlp"
        interchange_targets: Dict mapping keys to InterchangeTarget objects

    Returns:
        Dict with grid dimensions:
        - attention_head: {"layers": [...], "heads": [...]}
        - residual_stream/mlp: {"layers": [...], "token_position_ids": [...]}
    """
    keys = list(interchange_targets.keys())

    if component_type == "attention_head":
        layers = sorted(set(k[0] for k in keys))
        heads = sorted(set(k[1] for k in keys))
        return {"layers": layers, "heads": heads}
    else:
        # residual_stream or mlp: keys are (layer, position_id)
        layers = sorted(set(k[0] for k in keys))
        # Token position IDs may be strings, preserve insertion order
        position_ids = []
        seen = set()
        for k in keys:
            if k[1] not in seen:
                position_ids.append(k[1])
                seen.add(k[1])
        return {"layers": layers, "token_position_ids": position_ids}


# =============================================================================
# InterchangeTarget Builder Functions
# =============================================================================


def _group_units_into_targets(
    units_with_keys: List[Tuple[Any, AtomicModelUnit]],
    layers: List[int],
    mode: str,
) -> Dict[Tuple[Any, ...], InterchangeTarget]:
    """
    Group units into InterchangeTargets based on the specified mode.

    Args:
        units_with_keys: List of (key, unit) tuples where key identifies the unit
        layers: List of layer indices (used for one_target_per_layer grouping)
        mode: One of "one_target_all_units", "one_target_per_unit", "one_target_per_layer"

    Returns:
        Dict mapping keys to InterchangeTarget objects
    """
    if mode == "one_target_all_units":
        units = [unit for _, unit in units_with_keys]
        return {("all",): InterchangeTarget([units])}

    elif mode == "one_target_per_unit":
        return {key: InterchangeTarget([[unit]]) for key, unit in units_with_keys}

    elif mode == "one_target_per_layer":
        layer_groups = {layer: [] for layer in layers}
        for (layer, *_), unit in units_with_keys:
            layer_groups[layer].append(unit)
        return {
            (layer,): InterchangeTarget([group])
            for layer in layers
            if (group := layer_groups[layer])
        }

    else:
        raise ValueError(
            f"Invalid mode: {mode}. "
            f"Expected: 'one_target_all_units', 'one_target_per_unit', 'one_target_per_layer'"
        )


def build_residual_stream_targets(
    pipeline: LMPipeline,
    layers: List[int],
    token_positions: List[TokenPosition],
    mode: str = "one_target_per_unit",
    target_output: bool = True,
) -> Dict[Tuple[Any, ...], InterchangeTarget]:
    """
    Build InterchangeTargets for residual stream interventions.

    Args:
        pipeline: LMPipeline with model configuration
        layers: List of layer indices to intervene on
        token_positions: List of ComponentIndexers for token positions
        mode: How to group units into targets:
            - "one_target_all_units": Single target with all units
            - "one_target_per_unit": One target per (layer, position) combination
            - "one_target_per_layer": One target per layer
        target_output: Whether to target block_output (True) or block_input (False)

    Returns:
        Dict mapping keys to InterchangeTarget objects:
        - "one_target_all_units": {("all",): target}
        - "one_target_per_unit": {(layer, position_id): target, ...}
        - "one_target_per_layer": {(layer,): target, ...}
    """

    hidden_size = pipeline.model.config.hidden_size

    # Build all units with their keys
    units_with_keys = []
    for layer in layers:
        for pos in token_positions:
            # Handle layer -1 special case (embeddings/block_input at layer 0)
            actual_layer = layer
            actual_target_output = target_output
            if layer == -1:
                actual_layer = 0
                actual_target_output = False

            unit = ResidualStream(
                layer=actual_layer,
                token_indices=pos,
                featurizer=None,  # Will default to identity Featurizer()
                shape=(hidden_size,),
                feature_indices=None,
                target_output=actual_target_output,
            )
            units_with_keys.append(((layer, pos.id), unit))

    return _group_units_into_targets(units_with_keys, layers, mode)


def build_mlp_targets(
    pipeline: LMPipeline,
    layers: List[int],
    token_positions: List[TokenPosition],
    mode: str = "one_target_per_unit",
    location: str = "mlp_output",
) -> Dict[Tuple[Any, ...], InterchangeTarget]:
    """
    Build InterchangeTargets for MLP interventions.

    Args:
        pipeline: LMPipeline with model configuration
        layers: List of layer indices to intervene on
        token_positions: List of TokenPosition objects for token positions
        mode: How to group units into targets:
            - "one_target_all_units": Single target with all units
            - "one_target_per_unit": One target per (layer, position) combination
            - "one_target_per_layer": One target per layer
        location: Which MLP component to target:
            - "mlp_input": Input to the MLP
            - "mlp_output": Output of the MLP (default)
            - "mlp_activation": Activation inside the MLP

    Returns:
        Dict mapping keys to InterchangeTarget objects:
        - "one_target_all_units": {("all",): target}
        - "one_target_per_unit": {(layer, position_id): target, ...}
        - "one_target_per_layer": {(layer,): target, ...}
    """

    # Get dimension from model config based on location
    p_config = pipeline.model.config
    if location == "mlp_activation":
        if hasattr(p_config, "n_inner") and p_config.n_inner is not None:
            feature_size = p_config.n_inner
        elif hasattr(p_config, "intermediate_size"):
            feature_size = p_config.intermediate_size
        else:
            feature_size = p_config.hidden_size * 4
    else:
        feature_size = p_config.hidden_size

    # Build all units with their keys
    units_with_keys = []
    for layer in layers:
        for pos in token_positions:
            unit = MLP(
                layer=layer,
                token_indices=pos,
                featurizer=None,  # Will default to identity Featurizer()
                shape=(feature_size,),
                feature_indices=None,
                location=location,
            )
            units_with_keys.append(((layer, pos.id), unit))

    return _group_units_into_targets(units_with_keys, layers, mode)


def build_attention_head_targets(
    pipeline: LMPipeline,
    layers: List[int],
    heads: List[int],
    token_position: TokenPosition,
    mode: str = "one_target_per_unit",
) -> Dict[Tuple[Any, ...], InterchangeTarget]:
    """
    Build InterchangeTargets for attention head interventions.

    Args:
        pipeline: LMPipeline with model configuration
        layers: List of layer indices
        heads: List of head indices
        token_position: ComponentIndexer for token positions
        mode: How to group units into targets:
            - "one_target_all_units": Single target with all heads
            - "one_target_per_unit": One target per (layer, head) combination
            - "one_target_per_layer": One target per layer (all heads in layer grouped)

    Returns:
        Dict mapping keys to InterchangeTarget objects:
        - "one_target_all_units": {("all",): target}
        - "one_target_per_unit": {(layer, head): target, ...}
        - "one_target_per_layer": {(layer,): target, ...}
    """

    # Calculate head dimension from model config
    p_config = pipeline.model.config
    if hasattr(p_config, "head_dim"):
        head_size = p_config.head_dim
    else:
        if hasattr(p_config, "n_head"):
            num_heads = p_config.n_head
        elif hasattr(p_config, "num_attention_heads"):
            num_heads = p_config.num_attention_heads
        elif hasattr(p_config, "num_heads"):
            num_heads = p_config.num_heads
        else:
            raise ValueError(
                "Could not determine number of heads from model config. "
                "Expected one of: head_dim, n_head, num_attention_heads, num_heads"
            )
        head_size = pipeline.model.config.hidden_size // num_heads

    # Build all units with their keys
    units_with_keys = []
    for layer in layers:
        for head in heads:
            unit = AttentionHead(
                layer=layer,
                head=head,
                token_indices=token_position,
                featurizer=None,  # Will default to identity Featurizer()
                feature_indices=None,
                target_output=True,
                shape=(head_size,),
            )
            units_with_keys.append(((layer, head), unit))

    return _group_units_into_targets(units_with_keys, layers, mode)
