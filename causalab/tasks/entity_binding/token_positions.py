"""Token position functions for the entity binding task.

This module provides functions to locate specific tokens in entity binding prompts,
such as entity tokens in statements and questions.

Uses the declarative token position system from neural.token_position_builder.
"""

from typing import Any, Dict, List, Optional

from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_position_builder import Template

from .config import EntityBindingTaskConfig


def _build_values_dict(
    input_sample: Dict[str, Any],
    config: EntityBindingTaskConfig,
) -> Dict[str, str]:
    """Build the values dict for filling the mega template."""
    active_groups = input_sample.get("active_groups")
    entities_per_group = input_sample.get("entities_per_group")
    query_indices = input_sample.get("query_indices")

    values = {}
    # Statement entities: g0_e0, g0_e1, g1_e0, g1_e1, ...
    for g in range(active_groups):
        for e in range(entities_per_group):
            entity = input_sample.get(f"entity_g{g}_e{e}")
            values[f"g{g}_e{e}"] = entity if entity is not None else f"MISSING_{g}_{e}"

    # Question entities: use query_e0, query_e1, etc. from input_sample
    values["query_entity"] = input_sample.get(f"query_e{query_indices[0]}")
    for e in range(entities_per_group):
        role_name = config.entity_roles.get(e, f"entity{e}")
        query_val = input_sample.get(f"query_e{e}")
        values[role_name] = query_val if query_val is not None else ""

    return values


def get_entity_token_positions(
    input_sample: Dict[str, Any],
    pipeline: LMPipeline,
    config: EntityBindingTaskConfig,
    group_idx: int,
    entity_idx: int,
    token_idx: Optional[int] = None,
) -> List[int]:
    """
    Get token positions for a statement entity.

    Args:
        input_sample: Input sample with entity values and raw_input
        pipeline: Pipeline with tokenizer
        config: Task configuration
        group_idx: Entity group index (0-indexed)
        entity_idx: Entity index within group (0-indexed)
        token_idx: If specified, return only the token at this index within the entity.
                   Supports negative indexing (-1 for last token).

    Returns:
        List of token position indices
    """
    # Build the mega template using config method
    active_groups = input_sample.get("active_groups")
    query_indices = input_sample.get("query_indices")
    answer_index = input_sample.get("answer_index")

    mega_template_str = config.build_mega_template(
        active_groups, query_indices, answer_index
    )
    mega_template = Template(mega_template_str)

    # Build values and get variable positions
    values = _build_values_dict(input_sample, config)
    variable_positions = mega_template.get_variable_positions(values, pipeline)

    # Get positions for the requested entity
    var_name = f"g{group_idx}_e{entity_idx}"
    if var_name not in variable_positions:
        raise ValueError(
            f"Variable '{var_name}' not found. Available: {list(variable_positions.keys())}"
        )

    positions = variable_positions[var_name]

    # Apply token_idx filter if specified
    if token_idx is not None:
        if not positions:
            raise ValueError(
                f"No tokens found for entity at group {group_idx}, position {entity_idx}"
            )
        if token_idx < -len(positions) or token_idx >= len(positions):
            raise ValueError(
                f"token_idx {token_idx} out of range for entity with {len(positions)} tokens"
            )
        positions = [positions[token_idx]]

    return positions


def get_question_entity_token_positions(
    input_sample: Dict[str, Any],
    pipeline: LMPipeline,
    config: EntityBindingTaskConfig,
    entity_idx: Optional[int] = None,
    role_name: Optional[str] = None,
    token_idx: Optional[int] = None,
) -> List[int]:
    """
    Get token positions for a question entity.

    The question contains entities from the query variables (query_e0, query_e1, ...).
    You can specify which entity either by its index or by its role name.

    Args:
        input_sample: Input sample with entity values and raw_input
        pipeline: Pipeline with tokenizer
        config: Task configuration
        entity_idx: Entity index (0-indexed), corresponding to query_e{entity_idx}.
                    Mutually exclusive with role_name.
        role_name: Role name from config.entity_roles (e.g., "person", "object").
                   Will be converted to entity_idx. Mutually exclusive with entity_idx.
        token_idx: If specified, return only the token at this index within the entity.
                   Supports negative indexing (-1 for last token).

    Returns:
        List of token position indices

    Raises:
        ValueError: If the requested entity is not present in the question template.
    """
    # Validate arguments
    if entity_idx is None and role_name is None:
        raise ValueError("Must specify either entity_idx or role_name")
    if entity_idx is not None and role_name is not None:
        raise ValueError("Cannot specify both entity_idx and role_name")

    # Convert role_name to entity_idx
    if entity_idx is None:
        for idx, name in config.entity_roles.items():
            if name == role_name:
                entity_idx = idx
                break
        if entity_idx is None:
            raise ValueError(
                f"Role '{role_name}' not found in config.entity_roles: "
                f"{config.entity_roles}"
            )

    # Build the mega template using config method
    active_groups = input_sample.get("active_groups")
    query_indices = input_sample.get("query_indices")
    answer_index = input_sample.get("answer_index")

    mega_template_str = config.build_mega_template(
        active_groups, query_indices, answer_index
    )
    mega_template = Template(mega_template_str)

    # Build values and get variable positions
    values = _build_values_dict(input_sample, config)
    variable_positions = mega_template.get_variable_positions(values, pipeline)

    # Check if this entity_idx is in query_indices (i.e., appears in the question)
    if entity_idx not in query_indices:
        raise ValueError(
            f"Entity index {entity_idx} (role '{config.entity_roles.get(entity_idx)}') "
            f"is not in query_indices {query_indices}. "
            f"Only entities at positions {query_indices} appear in the question."
        )

    # The question template uses role names like {person}, {object}, {location}
    # which are filled from query_e{e} values via _build_values_dict
    var_name = config.entity_roles.get(entity_idx, f"entity{entity_idx}")

    if var_name not in variable_positions:
        raise ValueError(
            f"Variable '{var_name}' not found in question template. "
            f"Available: {list(variable_positions.keys())}"
        )

    positions = variable_positions[var_name]

    # Apply token_idx filter if specified
    if token_idx is not None:
        if not positions:
            raise ValueError(f"No tokens found for entity_idx {entity_idx}")
        if token_idx < -len(positions) or token_idx >= len(positions):
            raise ValueError(
                f"token_idx {token_idx} out of range for entity with {len(positions)} tokens"
            )
        positions = [positions[token_idx]]

    return positions


def get_statement_entity_token_positions(
    input_sample: Dict[str, Any],
    pipeline: LMPipeline,
    config: EntityBindingTaskConfig,
    group_idx: int,
    entity_idx: Optional[int] = None,
    role_name: Optional[str] = None,
    token_idx: Optional[int] = None,
) -> List[int]:
    """
    Get token positions for a statement entity with role_name support.

    Args:
        input_sample: Input sample with entity values and raw_input
        pipeline: Pipeline with tokenizer
        config: Task configuration
        group_idx: Entity group index (0-indexed)
        entity_idx: Entity index within group (0-indexed).
                    Mutually exclusive with role_name.
        role_name: Role name from config.entity_roles (e.g., "person", "object").
                   Mutually exclusive with entity_idx.
        token_idx: If specified, return only the token at this index within the entity.
                   Supports negative indexing (-1 for last token).

    Returns:
        List of token position indices
    """
    # Validate arguments
    if entity_idx is None and role_name is None:
        raise ValueError("Must specify either entity_idx or role_name")
    if entity_idx is not None and role_name is not None:
        raise ValueError("Cannot specify both entity_idx and role_name")

    # Convert role_name to entity_idx if needed
    if entity_idx is None:
        for idx, name in config.entity_roles.items():
            if name == role_name:
                entity_idx = idx
                break
        if entity_idx is None:
            raise ValueError(
                f"Role '{role_name}' not found in config.entity_roles: "
                f"{config.entity_roles}"
            )

    return get_entity_token_positions(
        input_sample=input_sample,
        pipeline=pipeline,
        config=config,
        group_idx=group_idx,
        entity_idx=entity_idx,
        token_idx=token_idx,
    )
