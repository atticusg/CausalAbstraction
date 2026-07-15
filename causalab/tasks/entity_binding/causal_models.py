"""
Causal model implementation for entity binding tasks.

The positional entity model searches for the query entity across all groups,
then retrieves from the matched group position. This tests how neural networks
perform entity-based retrieval with an explicit position-finding intermediate.
"""

import random
from typing import Any

from causalab.causal.causal_model import CausalModel, build_output_tokens
from causalab.causal.trace import CausalTrace, Mechanism, input_var

from .config import EntityBindingTaskConfig, create_sample_love_config


def sample_valid_entity_binding_input(
    config: EntityBindingTaskConfig,
    model: CausalModel,
    ensure_positional_uniqueness: bool = True,
) -> CausalTrace:
    """
    Sample a valid input for entity binding causal models.

    Ensures:
    - Active groups have all entities filled
    - Query group is within active groups
    - Query indices and answer index are valid for the group size
    - A question template exists for the query pattern
    - (Optional) Entities at the same position across groups are distinct

    Args:
        config: Task configuration
        model: CausalModel used to create the trace
        ensure_positional_uniqueness: If True, entities at the same position are distinct
            across groups. Required for the positional model to avoid ambiguity.

    Returns:
        CausalTrace with input values and computed variables
    """
    max_attempts = 100

    for _ in range(max_attempts):
        active_groups = config.max_groups  # Always use max groups for simplicity
        query_group = random.randint(0, active_groups - 1)

        if config.fixed_query_indices is not None:
            query_indices = config.fixed_query_indices
        else:
            query_indices = tuple([random.randint(0, config.max_entities_per_group - 1)])

        if config.fixed_answer_index is not None:
            answer_index = config.fixed_answer_index
        else:
            answer_index = random.randint(0, config.max_entities_per_group - 1)

        if answer_index in query_indices:
            continue

        if (query_indices, answer_index) not in config.question_templates:
            continue

        input_sample: dict[str, Any] = {
            "query_group": query_group,
            "query_indices": query_indices,
            "answer_index": answer_index,
            "active_groups": active_groups,
            "entities_per_group": config.max_entities_per_group,
        }

        used_entities_per_group = [set() for _ in range(active_groups)]
        used_entities_per_position = [
            set() for _ in range(config.max_entities_per_group)
        ]

        all_valid = True
        for g in range(active_groups):
            for e in range(config.max_entities_per_group):
                key = f"entity_g{g}_e{e}"

                if e in config.entity_pools:
                    available = config.entity_pools[e][:]
                    available = [
                        ent for ent in available if ent not in used_entities_per_group[g]
                    ]

                    if ensure_positional_uniqueness:
                        available = [
                            ent for ent in available
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

            if not all_valid:
                break

        if all_valid:
            input_sample["statement_template"] = config.statement_template
            # query_e{e} are computed variables — do not pass them as inputs
            return model.new_trace(input_sample)

    raise ValueError(
        f"Failed to sample valid entity binding input after {max_attempts} attempts. "
        f"Entity pools may be too small for the constraints."
    )


# =============================================================================
# Compute functions for mechanisms
# =============================================================================


def _compute_query_entity(t: CausalTrace, entity_pos: int, config: EntityBindingTaskConfig) -> Any:
    """Compute query_e{entity_pos} — entity from the query group at that position."""
    query_group = t["query_group"]
    active_groups = t["active_groups"]

    if query_group < active_groups:
        return t[f"entity_g{query_group}_e{entity_pos}"]
    return None


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
    """Compute positional_query_e{entity_position} — groups where query entity appears."""
    query_indices = t["query_indices"]
    active_groups = t["active_groups"]

    if entity_position not in query_indices:
        return ()

    query_entity = t[f"query_e{entity_position}"]
    if query_entity is None:
        return ()

    matching_groups = []
    for g in range(active_groups):
        entity = t[f"entity_g{g}_e{entity_position}"]
        if entity == query_entity:
            group_pos = t[f"positional_entity_g{g}_e{entity_position}"]
            if group_pos is not None:
                matching_groups.append(group_pos)

    return tuple(matching_groups)


def _compute_positional_answer(
    t: CausalTrace, config: EntityBindingTaskConfig
) -> int | None:
    """Compute positional_answer — intersection of all positional queries."""
    query_indices = t["query_indices"]

    if not query_indices:
        return None

    candidate_sets = []
    for entity_idx in query_indices:
        query_positions = t[f"positional_query_e{entity_idx}"]
        if query_positions:
            candidate_sets.append(set(query_positions))

    if not candidate_sets:
        return None

    intersection = candidate_sets[0]
    for candidate_set in candidate_sets[1:]:
        intersection = intersection.intersection(candidate_set)

    if len(intersection) == 0:
        return None
    elif len(intersection) > 1:
        # Ambiguous: multiple groups match. Use sample_valid_entity_binding_input for
        # proper sampling that enforces positional uniqueness across groups.
        return None

    return next(iter(intersection))


def _compute_raw_input(t: CausalTrace, config: EntityBindingTaskConfig) -> str:
    """Compute raw_input — the complete prompt text."""
    query_indices = t["query_indices"]
    if isinstance(query_indices, list):
        query_indices = tuple(query_indices)
    answer_index = t["answer_index"]
    active_groups = t["active_groups"]
    entities_per_group = t["entities_per_group"]

    try:
        mega_template_str = config.build_mega_template(
            active_groups, query_indices, answer_index
        )

        values = {}
        for g in range(active_groups):
            for e in range(entities_per_group):
                entity = t[f"entity_g{g}_e{e}"]
                values[f"g{g}_e{e}"] = entity if entity is not None else f"MISSING_{g}_{e}"

        # Fill question entity role names from query_e{e} computed variables
        values["query_entity"] = t[f"query_e{query_indices[0]}"]
        for e in range(entities_per_group):
            role_name = config.entity_roles.get(e, f"entity{e}")
            values[role_name] = t[f"query_e{e}"]

        return config.fill_template(mega_template_str, values)
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to compute raw_input: {e}")
        return "Invalid configuration"


def _compute_raw_output(t: CausalTrace, config: EntityBindingTaskConfig) -> str:
    """Compute raw_output — the expected answer entity."""
    positional_answer = t["positional_answer"]
    answer_index = t["answer_index"]
    active_groups = t["active_groups"]
    entities_per_group = t["entities_per_group"]

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

    Makes position computation explicit through intermediate variables:
    - query_e{e}: Entity from the query group at position e (computed)
    - positional_entity_g{g}_e{e}: Group index of each entity (trivially = g)
    - positional_query_e{e}: Groups containing the query entity at position e
    - positional_answer: Intersection → single group position to retrieve from

    Args:
        config: The task configuration

    Returns:
        CausalModel instance
    """
    mechanisms: dict[str, Mechanism] = {}
    values: dict[str, Any] = {}

    # =========================================================================
    # Input Variables
    # =========================================================================

    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"entity_g{g}_e{e}"
            if e in config.entity_pools:
                pool = config.entity_pools[e]
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

    # Fix active_groups to max_groups so default sampling always produces valid inputs
    mechanisms["active_groups"] = input_var([config.max_groups])
    values["active_groups"] = [config.max_groups]

    mechanisms["entities_per_group"] = input_var([config.max_entities_per_group])
    values["entities_per_group"] = [config.max_entities_per_group]

    mechanisms["statement_template"] = input_var([config.statement_template])
    values["statement_template"] = [config.statement_template]

    # =========================================================================
    # Computed Variables
    # =========================================================================

    # query_e{e}: Entity from the query group at position e (derived from entity_g* and query_group)
    all_entity_vars = [
        f"entity_g{g}_e{e}"
        for g in range(config.max_groups)
        for e in range(config.max_entities_per_group)
    ]
    for e in range(config.max_entities_per_group):
        key = f"query_e{e}"
        mechanisms[key] = Mechanism(
            parents=[f"entity_g{g}_e{e}" for g in range(config.max_groups)] + ["query_group", "active_groups"],
            compute=lambda t, e=e: _compute_query_entity(t, e, config),
        )
        if e in config.entity_pools:
            values[key] = config.entity_pools[e] + [None]
        else:
            values[key] = [None]

    # Positional entity variables — position of each entity (trivially = group index)
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

    # Positional query variables — find groups where query entity appears
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
        values[key] = None

    # Positional answer — intersection of all positional queries
    positional_query_vars = [
        f"positional_query_e{e}" for e in range(config.max_entities_per_group)
    ]
    mechanisms["positional_answer"] = Mechanism(
        parents=positional_query_vars + ["query_indices"],
        compute=lambda t: _compute_positional_answer(t, config),
    )
    values["positional_answer"] = list(range(config.max_groups))

    # Raw input — complete prompt text
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
    values["raw_input"] = None

    # Raw output — expected answer
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
    values["raw_output"] = None

    model_id = (
        f"entity_binding_positional_entity_"
        f"{config.max_groups}g_{config.max_entities_per_group}e"
    )

    # The answer is a bound entity name (any of the pooled entities). Declare its
    # surface forms once: the deduped union of all entity pools, each as its
    # ``[" entity", "entity"]`` forms (#296). The probability path reads these
    # (dedup of the 12 answer tokens falls out of the distinct form-groups), and
    # the derived checker uses ``prefix`` — the entity may be followed by
    # continuation tokens under the multi-token (``max_new_tokens=4``) contract,
    # which is exactly what the former checker.py's ``startswith`` accepted.
    all_entities: list[str] = []
    for pool in config.entity_pools.values():
        all_entities.extend(pool)
    all_entities = list(dict.fromkeys(all_entities))
    return CausalModel(
        mechanisms,
        values,
        id=model_id,
        output_tokens={"positional_answer": build_output_tokens(all_entities)},
        match_modes={"positional_answer": "prefix"},
    )


# Module-level default causal model using the love config
_default_config = create_sample_love_config()
causal_model = create_positional_entity_causal_model(_default_config)

# Required exports for the causalab runner
CAUSAL_MODEL = causal_model
TARGET_VARIABLE = "positional_answer"
