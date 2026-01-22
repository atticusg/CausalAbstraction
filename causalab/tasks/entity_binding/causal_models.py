"""
Causal model implementation for entity binding tasks.

The positional model searches for the query entity, then retrieves from that position.
This tests how neural networks perform entity-based retrieval.
"""

import random
from typing import Any

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import CausalTrace, Mechanism, input_var
from causalab.neural.token_position_builder import Template

from .config import EntityBindingTaskConfig


def sample_valid_entity_binding_input(
    config: EntityBindingTaskConfig,
    model: CausalModel,
    ensure_positional_uniqueness: bool = True,
) -> CausalTrace:
    """
    Sample a valid input for entity binding causal models.

    This ensures that:
    - Active groups have all entities filled
    - Query group is within active groups
    - Query indices and answer index are valid for the group size
    - A question template exists for the query pattern
    - (Optional) Entities at the same position across groups are distinct

    Args:
        config: Task configuration
        ensure_positional_uniqueness: If True, ensures that for each entity position,
            all groups have different entities at that position. This is required for
            the positional model to avoid ambiguity.
        model: If provided, returns a CausalTrace instead of a dict. The model
            will compute derived variables (raw_input, raw_output, etc.).

    Returns:
        If model is None: Dictionary with sampled input values
        If model is provided: CausalTrace with input values and computed variables
    """
    max_attempts = 100

    for _ in range(max_attempts):
        # Sample basic parameters
        active_groups = random.randint(2, config.max_groups)
        query_group = random.randint(0, active_groups - 1)

        # Use fixed_query_indices from config if provided, otherwise sample randomly
        if config.fixed_query_indices is not None:
            query_indices = config.fixed_query_indices
        else:
            query_indices = tuple(
                [random.randint(0, config.max_entities_per_group - 1)]
            )

        # Use fixed_answer_index from config if provided, otherwise sample randomly
        if config.fixed_answer_index is not None:
            answer_index = config.fixed_answer_index
        else:
            answer_index = random.randint(0, config.max_entities_per_group - 1)

        # Ensure query and answer are different
        if answer_index in query_indices:
            continue

        # Check if template exists for this query pattern
        if (query_indices, answer_index) not in config.question_templates:
            continue

        # Sample entities for all active groups
        input_sample: dict[str, Any] = {
            "query_group": query_group,
            "query_indices": query_indices,
            "answer_index": answer_index,
            "active_groups": active_groups,
            "entities_per_group": config.max_entities_per_group,
        }

        # Sample entities with two constraints:
        # 1. Distinct within each group
        # 2. (Optional) Distinct at each position across groups
        used_entities_per_group = [set() for _ in range(active_groups)]
        used_entities_per_position = [
            set() for _ in range(config.max_entities_per_group)
        ]

        all_valid = True
        for g in range(config.max_groups):
            for e in range(config.max_entities_per_group):
                key = f"entity_g{g}_e{e}"

                if g < active_groups:
                    # Active group - sample an entity
                    if e in config.entity_pools:
                        # Build list of available entities
                        available = config.entity_pools[e][:]

                        # Exclude entities already used in this group
                        available = [
                            ent
                            for ent in available
                            if ent not in used_entities_per_group[g]
                        ]

                        # Optionally exclude entities already used at this position in other groups
                        if ensure_positional_uniqueness:
                            available = [
                                ent
                                for ent in available
                                if ent not in used_entities_per_position[e]
                            ]

                        if not available:
                            all_valid = False
                            break

                        entity = random.choice(available)
                        input_sample[key] = entity
                        used_entities_per_group[g].add(entity)
                        used_entities_per_position[e].add(entity)
                    else:
                        input_sample[key] = None
                else:
                    # Inactive group
                    input_sample[key] = None

            if not all_valid:
                break

        if all_valid:
            # Set query entity variables (query_e0, query_e1, ...) from the query group's entities
            for e in range(config.max_entities_per_group):
                input_sample[f"query_e{e}"] = input_sample.get(
                    f"entity_g{query_group}_e{e}"
                )
            # Add statement_template (required input variable)
            input_sample["statement_template"] = config.statement_template
            return model.new_trace(input_sample)

    # If we failed after max_attempts, raise an error with helpful context
    raise ValueError(
        f"Failed to sample valid entity binding input after {max_attempts} attempts. "
        f"This usually means the entity pools are too small for the constraints. "
        f"Config: max_groups={config.max_groups}, "
        f"max_entities_per_group={config.max_entities_per_group}, "
        f"entity_pools sizes={[len(pool) for pool in config.entity_pools.values()]}, "
        f"ensure_positional_uniqueness={ensure_positional_uniqueness}"
    )


# =============================================================================
# Compute functions for mechanisms
# =============================================================================


def _compute_question_template(t: CausalTrace, config: EntityBindingTaskConfig) -> str:
    """Compute question_template based on query_indices and answer_index."""
    query_indices = t["query_indices"]
    answer_index = t["answer_index"]

    if isinstance(query_indices, list):
        query_indices = tuple(query_indices)

    key = (query_indices, answer_index)
    if key in config.question_templates:
        return config.question_templates[key]
    return "What is the answer?"


def _compute_positional_query(
    t: CausalTrace, entity_position: int, config: EntityBindingTaskConfig
) -> tuple[int, ...]:
    """Compute positional_query_e{entity_position} - groups where query entity appears."""
    query_indices = t["query_indices"]
    active_groups = t["active_groups"]

    # Check if this entity position is in the query
    if entity_position not in query_indices:
        return ()

    # Get the query entity at this position
    query_entity = t[f"query_e{entity_position}"]
    if query_entity is None:
        return ()

    # Search for groups where this entity appears at this position
    matching_groups = []
    for g in range(active_groups):
        entity = t[f"entity_g{g}_e{entity_position}"]
        if entity == query_entity:
            # Found matching entity at this position
            group_pos = t[f"positional_entity_g{g}_e{entity_position}"]
            if group_pos is not None:
                matching_groups.append(group_pos)

    return tuple(matching_groups)


def _compute_positional_answer(
    t: CausalTrace, config: EntityBindingTaskConfig
) -> int | None:
    """Compute positional_answer (intersection of all positional queries)."""
    query_indices = t["query_indices"]

    if not query_indices:
        return None

    # Get positions for all queried entities
    candidate_sets = []
    for entity_idx in query_indices:
        query_positions = t[f"positional_query_e{entity_idx}"]
        if query_positions:
            candidate_sets.append(set(query_positions))

    if not candidate_sets:
        return None

    # Find intersection
    intersection = candidate_sets[0]
    for candidate_set in candidate_sets[1:]:
        intersection = intersection.intersection(candidate_set)

    if len(intersection) == 0:
        return None
    elif len(intersection) > 1:
        raise ValueError(
            f"Ambiguous query: {len(intersection)} positions match the query. "
            f"Matching positions: {sorted(intersection)}. "
            f"Entity binding should have distinct entities to avoid ambiguity."
        )

    return next(iter(intersection))


def _compute_raw_input(t: CausalTrace, config: EntityBindingTaskConfig) -> str:
    """Compute raw_input (the complete prompt text)."""
    query_indices = t["query_indices"]
    if isinstance(query_indices, list):
        query_indices = tuple(query_indices)
    answer_index = t["answer_index"]
    active_groups = t["active_groups"]
    entities_per_group = t["entities_per_group"]

    try:
        # Build mega template using config method
        mega_template_str = config.build_mega_template(
            active_groups, query_indices, answer_index
        )
        mega_template = Template(mega_template_str)

        # Build values dict with all variables
        values = {}

        # Statement entities: g0_e0, g0_e1, g1_e0, g1_e1, ...
        for g in range(active_groups):
            for e in range(entities_per_group):
                entity = t[f"entity_g{g}_e{e}"]
                values[f"g{g}_e{e}"] = (
                    entity if entity is not None else f"MISSING_{g}_{e}"
                )

        # Question entities: use query_e0, query_e1, etc.
        values["query_entity"] = t[f"query_e{query_indices[0]}"]
        for e in range(entities_per_group):
            role_name = config.entity_roles.get(e, f"entity{e}")
            values[role_name] = t[f"query_e{e}"]

        return mega_template.fill(values)
    except Exception as e:
        import warnings

        warnings.warn(
            f"Failed to compute raw_input: {e}. "
            f"query_indices={query_indices}, answer_index={answer_index}, "
            f"active_groups={active_groups}"
        )
        return "Invalid configuration"


def _compute_raw_output(t: CausalTrace, config: EntityBindingTaskConfig) -> str:
    """Compute raw_output (the expected answer)."""
    positional_answer = t["positional_answer"]
    answer_index = t["answer_index"]
    active_groups = t["active_groups"]
    entities_per_group = t["entities_per_group"]

    # Use positional_answer to retrieve
    if (
        positional_answer is not None
        and positional_answer < active_groups
        and answer_index < entities_per_group
    ):
        answer_entity = t[f"entity_g{positional_answer}_e{answer_index}"]
        if answer_entity is not None:
            return answer_entity

    return "UNKNOWN"


# =============================================================================
# Main model creation function
# =============================================================================


def create_positional_entity_causal_model(
    config: EntityBindingTaskConfig,
) -> CausalModel:
    """
    Create the POSITIONAL ENTITY binding causal model.

    This model makes position computation explicit through intermediate variables:
    - positional_entity_g{g}_e{e}: The position (group index) of each entity
    - positional_query_e{e}: Tuple of group positions where query entity appears
    - positional_answer: The final group position to retrieve from (intersection)

    The model breaks retrieval into stages:
    1. Compute position of each entity (trivially returns group index)
    2. For each query position, find which groups contain that query entity
    3. Take intersection to get single answer position
    4. Retrieve answer from that position

    Args:
        config: The task configuration

    Returns:
        A CausalModel instance
    """
    mechanisms: dict[str, Mechanism] = {}
    values: dict[str, Any] = {}

    # =========================================================================
    # Input Variables (use input_var)
    # =========================================================================

    # Entity variables - one for each possible position
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"entity_g{g}_e{e}"
            if e in config.entity_pools:
                pool = config.entity_pools[e] + [None]
                mechanisms[key] = input_var(pool)
                values[key] = pool
            else:
                mechanisms[key] = input_var([None])
                values[key] = [None]

    # Query entity variables - entities from the query group
    for e in range(config.max_entities_per_group):
        key = f"query_e{e}"
        if e in config.entity_pools:
            pool = config.entity_pools[e] + [None]
            mechanisms[key] = input_var(pool)
            values[key] = pool
        else:
            mechanisms[key] = input_var([None])
            values[key] = [None]

    # Control variables
    mechanisms["query_group"] = input_var(list(range(config.max_groups)))
    values["query_group"] = list(range(config.max_groups))

    query_indices_values = [tuple([i]) for i in range(config.max_entities_per_group)]
    mechanisms["query_indices"] = input_var(query_indices_values)
    values["query_indices"] = query_indices_values

    mechanisms["answer_index"] = input_var(list(range(config.max_entities_per_group)))
    values["answer_index"] = list(range(config.max_entities_per_group))

    mechanisms["active_groups"] = input_var(list(range(1, config.max_groups + 1)))
    values["active_groups"] = list(range(1, config.max_groups + 1))

    mechanisms["entities_per_group"] = input_var([config.max_entities_per_group])
    values["entities_per_group"] = [config.max_entities_per_group]

    mechanisms["statement_template"] = input_var([config.statement_template])
    values["statement_template"] = [config.statement_template]

    # =========================================================================
    # Computed Variables (use Mechanism)
    # =========================================================================

    # Positional entity variables - position of each entity (trivially = group index)
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"positional_entity_g{g}_e{e}"
            mechanisms[key] = Mechanism(
                parents=[f"entity_g{g}_e{e}"],
                compute=lambda t, g=g, e=e: g
                if t[f"entity_g{g}_e{e}"] is not None
                else None,
            )
            values[key] = list(range(config.max_groups)) + [None]

    # Question template selection
    mechanisms["question_template"] = Mechanism(
        parents=["query_indices", "answer_index"],
        compute=lambda t: _compute_question_template(t, config),
    )
    values["question_template"] = list(config.question_templates.values())

    # Positional query variables - find groups where query entity appears
    entity_vars = [
        f"entity_g{g}_e{e}"
        for g in range(config.max_groups)
        for e in range(config.max_entities_per_group)
    ]
    positional_entity_vars = [
        f"positional_entity_g{g}_e{e}"
        for g in range(config.max_groups)
        for e in range(config.max_entities_per_group)
    ]
    query_entity_vars = [f"query_e{e}" for e in range(config.max_entities_per_group)]

    for e in range(config.max_entities_per_group):
        key = f"positional_query_e{e}"
        mechanisms[key] = Mechanism(
            parents=(
                entity_vars
                + positional_entity_vars
                + query_entity_vars
                + ["query_indices", "active_groups", "entities_per_group"]
            ),
            compute=lambda t, e=e: _compute_positional_query(t, e, config),
        )
        values[key] = None  # Computed

    # Positional answer - intersection of all positional queries
    positional_query_vars = [
        f"positional_query_e{e}" for e in range(config.max_entities_per_group)
    ]
    mechanisms["positional_answer"] = Mechanism(
        parents=positional_query_vars + ["query_indices"],
        compute=lambda t: _compute_positional_answer(t, config),
    )
    values["positional_answer"] = None  # Computed

    # Raw input - the complete prompt text
    mechanisms["raw_input"] = Mechanism(
        parents=(
            entity_vars
            + query_entity_vars
            + [
                "statement_template",
                "question_template",
                "query_indices",
                "answer_index",
                "active_groups",
                "entities_per_group",
            ]
        ),
        compute=lambda t: _compute_raw_input(t, config),
    )
    values["raw_input"] = None  # Computed

    # Raw output - the expected answer
    mechanisms["raw_output"] = Mechanism(
        parents=(
            entity_vars
            + [
                "positional_answer",
                "answer_index",
                "active_groups",
                "entities_per_group",
            ]
        ),
        compute=lambda t: _compute_raw_output(t, config),
    )
    values["raw_output"] = None  # Computed

    # =========================================================================
    # Create the model
    # =========================================================================
    model_id = f"entity_binding_positional_entity_{config.max_groups}g_{config.max_entities_per_group}e"
    return CausalModel(mechanisms, values, id=model_id)
