"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Causal model implementations for entity binding tasks.

This module contains two variants of the entity binding causal model:

1. **Direct Model**: Uses query_group index directly to retrieve the answer
2. **Positional Model**: Searches for the query entity, then retrieves from that position

These models test different hypotheses about how neural networks perform retrieval.
"""

import random
from typing import Dict, Any
from causalab.causal.causal_model import CausalModel
from .config import EntityBindingTaskConfig
from .templates import TemplateProcessor, EntityGroup, BindingMatrix


def create_direct_causal_model(config: EntityBindingTaskConfig) -> CausalModel:
    """
    Create the DIRECT entity binding causal model.

    This model directly uses the query_group index to retrieve the answer.
    It represents the hypothesis that the model uses direct indexing/addressing.

    Variables:
    - Entity variables: entity_g{group}_e{entity} for each possible position
    - Query variables: query_group, query_indices, answer_index
    - Template variables: statement_template, question_template
    - Control variables: active_groups, entities_per_group
    - Output variables: raw_input, raw_output

    The key is that raw_output directly uses query_group to index into the entity matrix.

    Args:
        config: The task configuration

    Returns:
        A CausalModel instance
    """

    # Build variable list
    variables = []

    # Entity variables - one for each possible position
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            variables.append(f"entity_g{g}_e{e}")

    # Query and control variables
    variables.extend(
        [
            "query_group",  # Which group contains the query/answer (used directly)
            "query_indices",  # Tuple of entity indices mentioned in question
            "answer_index",  # Index of the answer entity within query group
            "active_groups",  # Number of groups actually used
            "entities_per_group",  # Number of entities per group
            "statement_template",  # Template for factual statements
            "question_template",  # Template for questions
            "raw_input",  # The complete prompt text
            "raw_output",  # The expected answer text
        ]
    )

    # Build values dictionary
    values = {}

    # Entity values - each entity can be any from its pool, or None for inactive
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"entity_g{g}_e{e}"
            if e in config.entity_pools:
                values[key] = config.entity_pools[e] + [
                    None
                ]  # Include None for inactive
            else:
                values[key] = [None]  # Only None if no pool defined

    # Query and control values
    values.update(
        {
            "query_group": list(range(config.max_groups)),
            "query_indices": [tuple([i]) for i in range(config.max_entities_per_group)],
            "answer_index": list(range(config.max_entities_per_group)),
            "active_groups": list(range(1, config.max_groups + 1)),
            "entities_per_group": [config.max_entities_per_group],
            "statement_template": [config.statement_template],
            "question_template": list(config.question_templates.values()),
            "raw_input": None,  # Generated
            "raw_output": None,  # Generated
        }
    )

    # Build parents dictionary
    parents = {}

    # Entity variables are independent
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            parents[f"entity_g{g}_e{e}"] = []

    # Control variables are independent
    parents.update(
        {
            "query_group": [],
            "query_indices": [],
            "answer_index": [],
            "active_groups": [],
            "entities_per_group": [],
            "statement_template": [],
        }
    )

    # Question template depends on query pattern
    parents["question_template"] = ["query_indices", "answer_index"]

    # Raw input depends on entities and templates
    entity_vars = [
        f"entity_g{g}_e{e}"
        for g in range(config.max_groups)
        for e in range(config.max_entities_per_group)
    ]
    parents["raw_input"] = entity_vars + [
        "statement_template",
        "question_template",
        "query_group",
        "query_indices",
        "active_groups",
        "entities_per_group",
    ]

    # Raw output depends on query_group DIRECTLY
    parents["raw_output"] = entity_vars + [
        "query_group",
        "answer_index",
        "active_groups",
        "entities_per_group",
    ]

    # Build mechanisms dictionary
    mechanisms = {}

    # Entity sampling mechanisms
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"entity_g{g}_e{e}"
            if e in config.entity_pools:
                pool = config.entity_pools[e]
                mechanisms[key] = lambda t, pool=pool: random.choice(pool)
            else:
                mechanisms[key] = lambda t: None

    # Control variable mechanisms
    mechanisms.update({
        "query_group": lambda t: random.randint(0, config.max_groups - 1),
        "query_indices": lambda t: tuple([random.randint(0, config.max_entities_per_group - 1)]),
        "answer_index": lambda t: random.randint(0, config.max_entities_per_group - 1),
        "active_groups": lambda t: random.randint(2, config.max_groups),
        "entities_per_group": lambda t: config.max_entities_per_group,
        "statement_template": lambda t: config.statement_template,
    })

    # Question template selection mechanism
    def select_question_template(trace):
        """Select question template based on query pattern."""
        query_indices = trace["query_indices"]
        answer_index = trace["answer_index"]

        # Convert list to tuple if needed (Dataset serialization can convert tuples to lists)
        if isinstance(query_indices, list):
            query_indices = tuple(query_indices)

        key = (query_indices, answer_index)
        if key in config.question_templates:
            return config.question_templates[key]
        else:
            return "What is the answer?"

    mechanisms["question_template"] = select_question_template

    # Raw input generation mechanism
    def generate_raw_input(trace):
        """Generate the complete prompt text."""
        question_template = trace["question_template"]
        query_group = trace["query_group"]
        query_indices = trace["query_indices"]
        active_groups = trace["active_groups"]
        entities_per_group = trace["entities_per_group"]

        # Build entity dictionary
        entity_dict = {}
        for g in range(config.max_groups):
            for e in range(config.max_entities_per_group):
                entity_dict[f"entity_g{g}_e{e}"] = trace[f"entity_g{g}_e{e}"]

        # Create binding matrix
        groups = []
        for g in range(active_groups):
            entities = []
            for e in range(entities_per_group):
                entity = entity_dict[f"entity_g{g}_e{e}"]
                if entity is not None:
                    entities.append(entity)
                else:
                    entities.append(f"MISSING_{g}_{e}")
            groups.append(EntityGroup(entities, g))

        matrix = BindingMatrix(groups, config.max_groups, config.max_entities_per_group)

        # Use template processor to generate text
        processor = TemplateProcessor(config)

        try:
            statement = processor.fill_statement_template(matrix)
            question = processor.fill_question_template(
                question_template, query_group, query_indices, matrix
            )
            separator = config.statement_question_separator
            core_prompt = f"{statement}{separator}{question}"
            return f"{config.prompt_prefix}{core_prompt}{config.prompt_suffix}"
        except Exception:
            return "Invalid configuration"

    mechanisms["raw_input"] = generate_raw_input

    # Raw output generation mechanism (DIRECT retrieval by index)
    def generate_raw_output(trace):
        """Generate the expected answer by DIRECT indexing."""
        query_group = trace["query_group"]
        answer_index = trace["answer_index"]
        active_groups = trace["active_groups"]
        entities_per_group = trace["entities_per_group"]

        # DIRECT retrieval: use query_group index directly
        if query_group < active_groups and answer_index < entities_per_group:
            answer_entity = trace[f"entity_g{query_group}_e{answer_index}"]
            if answer_entity is not None:
                return answer_entity

        return "UNKNOWN"

    mechanisms["raw_output"] = generate_raw_output

    # Create the causal model
    model_id = (
        f"entity_binding_direct_{config.max_groups}g_{config.max_entities_per_group}e"
    )
    return CausalModel(variables, values, parents, mechanisms, id=model_id)


def create_positional_causal_model(config: EntityBindingTaskConfig) -> CausalModel:
    """
    Create the POSITIONAL entity binding causal model.

    This model adds intermediate variables that perform a positional search:
    1. Identify which entity is being queried (by value)
    2. Search for that entity in the entity groups
    3. Retrieve the answer from the position where the entity was found

    This represents the hypothesis that the model performs content-based retrieval
    rather than direct indexing.

    New intermediate variables:
    - query_entity: The actual entity value being queried (e.g., "Ann")
    - positional_query_group: Which group contains the query_entity (found by search)

    The key difference: raw_output uses positional_query_group (found by search)
    instead of query_group (given as input).

    Args:
        config: The task configuration

    Returns:
        A CausalModel instance
    """

    # Build variable list
    variables = []

    # Entity variables - one for each possible position
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            variables.append(f"entity_g{g}_e{e}")

    # Query and control variables
    variables.extend(
        [
            "query_group",  # Which group SHOULD contain the query (input)
            "query_indices",  # Tuple of entity indices mentioned in question
            "answer_index",  # Index of the answer entity within query group
            "active_groups",  # Number of groups actually used
            "entities_per_group",  # Number of entities per group
            "statement_template",  # Template for factual statements
            "question_template",  # Template for questions
            # NEW INTERMEDIATE VARIABLES
            "query_entity",  # The actual entity being queried (value, not index)
            "positional_query_group",  # Which group contains query_entity (found by search)
            "raw_input",  # The complete prompt text
            "raw_output",  # The expected answer text
        ]
    )

    # Build values dictionary
    values = {}

    # Entity values
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"entity_g{g}_e{e}"
            if e in config.entity_pools:
                values[key] = config.entity_pools[e] + [None]
            else:
                values[key] = [None]

    # Query and control values
    values.update(
        {
            "query_group": list(range(config.max_groups)),
            "query_indices": [tuple([i]) for i in range(config.max_entities_per_group)],
            "answer_index": list(range(config.max_entities_per_group)),
            "active_groups": list(range(1, config.max_groups + 1)),
            "entities_per_group": [config.max_entities_per_group],
            "statement_template": [config.statement_template],
            "question_template": list(config.question_templates.values()),
            # Intermediate variables (computed)
            "query_entity": None,  # Will be computed from entities
            "positional_query_group": None,  # Will be computed by search
            "raw_input": None,
            "raw_output": None,
        }
    )

    # Build parents dictionary
    parents = {}

    # Entity variables are independent
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            parents[f"entity_g{g}_e{e}"] = []

    # Control variables are independent
    parents.update(
        {
            "query_group": [],
            "query_indices": [],
            "answer_index": [],
            "active_groups": [],
            "entities_per_group": [],
            "statement_template": [],
        }
    )

    # Question template depends on query pattern
    parents["question_template"] = ["query_indices", "answer_index"]

    # NEW: query_entity depends on entities and query parameters
    entity_vars = [
        f"entity_g{g}_e{e}"
        for g in range(config.max_groups)
        for e in range(config.max_entities_per_group)
    ]
    parents["query_entity"] = entity_vars + ["query_group", "query_indices"]

    # NEW: positional_query_group depends on entities, query_entity, and query_indices
    parents["positional_query_group"] = entity_vars + [
        "query_entity",
        "query_indices",
        "active_groups",
        "entities_per_group",
    ]

    # Raw input depends on entities and templates (same as direct model)
    parents["raw_input"] = entity_vars + [
        "statement_template",
        "question_template",
        "query_group",
        "query_indices",
        "active_groups",
        "entities_per_group",
    ]

    # Raw output depends on POSITIONAL_QUERY_GROUP (not query_group directly!)
    parents["raw_output"] = entity_vars + [
        "positional_query_group",
        "answer_index",
        "active_groups",
        "entities_per_group",
    ]

    # Build mechanisms dictionary
    mechanisms = {}

    # Entity sampling mechanisms (same as direct model)
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"entity_g{g}_e{e}"
            if e in config.entity_pools:
                pool = config.entity_pools[e]
                mechanisms[key] = lambda t, pool=pool: random.choice(pool)
            else:
                mechanisms[key] = lambda t: None

    # Control variable mechanisms (same as direct model)
    mechanisms.update({
        "query_group": lambda t: random.randint(0, config.max_groups - 1),
        "query_indices": lambda t: tuple([random.randint(0, config.max_entities_per_group - 1)]),
        "answer_index": lambda t: random.randint(0, config.max_entities_per_group - 1),
        "active_groups": lambda t: random.randint(2, config.max_groups),
        "entities_per_group": lambda t: config.max_entities_per_group,
        "statement_template": lambda t: config.statement_template,
    })

    # Question template selection mechanism (same as direct model)
    def select_question_template(trace):
        """Select question template based on query pattern."""
        query_indices = trace["query_indices"]
        answer_index = trace["answer_index"]

        # Convert list to tuple if needed (Dataset serialization can convert tuples to lists)
        if isinstance(query_indices, list):
            query_indices = tuple(query_indices)

        key = (query_indices, answer_index)
        if key in config.question_templates:
            return config.question_templates[key]
        else:
            return "What is the answer?"

    mechanisms["question_template"] = select_question_template

    # NEW: Query entity extraction mechanism
    def extract_query_entity(trace):
        """Extract the query entity values from the binding matrix.

        This identifies WHICH entities are being queried based on query_group and query_indices.
        Returns a tuple of entity values for all query indices.
        """
        query_group = trace["query_group"]
        query_indices = trace["query_indices"]

        # Extract all query entities as a tuple
        query_entities = []
        for query_entity_idx in query_indices:
            entity = trace[f"entity_g{query_group}_e{query_entity_idx}"]
            query_entities.append(entity)

        return tuple(query_entities) if query_entities else None

    mechanisms["query_entity"] = extract_query_entity

    # NEW: Positional search mechanism
    def search_for_query_entity(trace):
        """Search for a group that contains the query entities at their respective positions.

        Returns the group index where ALL query entities are found at their query positions.
        Raises an error if multiple groups match (ambiguous case).

        This simulates content-based retrieval rather than direct indexing.
        """
        query_entity = trace["query_entity"]  # This is now a tuple of entities
        query_indices = trace["query_indices"]
        active_groups = trace["active_groups"]

        if query_entity is None or not query_indices:
            return None

        # Convert to tuple if needed
        if not isinstance(query_entity, tuple):
            query_entity = (query_entity,)

        # Search for groups where ALL query entities match at their positions
        matching_groups = []

        for g in range(active_groups):
            # Check if this group has all query entities at the right positions
            all_match = True
            for i, query_entity_idx in enumerate(query_indices):
                entity = trace[f"entity_g{g}_e{query_entity_idx}"]
                if i < len(query_entity) and entity != query_entity[i]:
                    all_match = False
                    break

            if all_match:
                matching_groups.append(g)

        # Check for ambiguity
        if len(matching_groups) > 1:
            # Multiple groups match - this is ambiguous!
            # In this case, we could either:
            # 1. Raise an error (strict mode)
            # 2. Return the first match (lenient mode)
            # 3. Return None to indicate ambiguity
            # For now, let's raise an error to catch this case
            raise ValueError(
                f"Ambiguous query: {len(matching_groups)} groups match the query entities {query_entity} "
                f"at positions {query_indices}. Matching groups: {matching_groups}. "
                f"Entity binding should have distinct entities to avoid ambiguity."
            )

        if len(matching_groups) == 1:
            return matching_groups[0]

        # Not found
        return None

    mechanisms["positional_query_group"] = search_for_query_entity

    # Raw input generation mechanism (same as direct model)
    def generate_raw_input(trace):
        """Generate the complete prompt text."""
        question_template = trace["question_template"]
        query_group = trace["query_group"]
        query_indices = trace["query_indices"]
        active_groups = trace["active_groups"]
        entities_per_group = trace["entities_per_group"]

        # Build entity dictionary
        entity_dict = {}
        for g in range(config.max_groups):
            for e in range(config.max_entities_per_group):
                entity_dict[f"entity_g{g}_e{e}"] = trace[f"entity_g{g}_e{e}"]

        # Create binding matrix
        groups = []
        for g in range(active_groups):
            entities = []
            for e in range(entities_per_group):
                entity = entity_dict[f"entity_g{g}_e{e}"]
                if entity is not None:
                    entities.append(entity)
                else:
                    entities.append(f"MISSING_{g}_{e}")
            groups.append(EntityGroup(entities, g))

        matrix = BindingMatrix(groups, config.max_groups, config.max_entities_per_group)

        # Use template processor to generate text
        processor = TemplateProcessor(config)

        try:
            statement = processor.fill_statement_template(matrix)
            question = processor.fill_question_template(
                question_template, query_group, query_indices, matrix
            )
            separator = config.statement_question_separator
            core_prompt = f"{statement}{separator}{question}"
            return f"{config.prompt_prefix}{core_prompt}{config.prompt_suffix}"
        except Exception:
            return "Invalid configuration"

    mechanisms["raw_input"] = generate_raw_input

    # Raw output generation mechanism (POSITIONAL retrieval)
    def generate_raw_output(trace):
        """Generate the expected answer by POSITIONAL search."""
        positional_query_group = trace["positional_query_group"]  # Use POSITIONAL group (from search)
        answer_index = trace["answer_index"]
        active_groups = trace["active_groups"]
        entities_per_group = trace["entities_per_group"]

        # POSITIONAL retrieval: use positional_query_group (found by search)
        if positional_query_group is not None and positional_query_group < active_groups and answer_index < entities_per_group:
            answer_entity = trace[f"entity_g{positional_query_group}_e{answer_index}"]
            if answer_entity is not None:
                return answer_entity

        return "UNKNOWN"

    mechanisms["raw_output"] = generate_raw_output

    # Create the causal model
    model_id = f"entity_binding_positional_{config.max_groups}g_{config.max_entities_per_group}e"
    return CausalModel(variables, values, parents, mechanisms, id=model_id)


def sample_valid_entity_binding_input(
    config: EntityBindingTaskConfig, ensure_positional_uniqueness: bool = True
) -> Dict[str, Any]:
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

    Returns:
        Dictionary with sampled input values
    """
    max_attempts = 100

    for attempt in range(max_attempts):
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

        answer_index = random.randint(0, config.max_entities_per_group - 1)

        # Ensure query and answer are different
        if answer_index in query_indices:
            continue

        # Check if template exists for this query pattern
        if (query_indices, answer_index) not in config.question_templates:
            continue

        # Sample entities for all active groups
        input_sample = {
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
            return input_sample

    # If we failed after max_attempts, return a simple valid sample
    input_sample = {
        "query_group": 0,
        "query_indices": (0,),
        "answer_index": 1,
        "active_groups": 2,
        "entities_per_group": config.max_entities_per_group,
    }

    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"entity_g{g}_e{e}"
            if g < 2 and e in config.entity_pools:
                # Use different entities for fallback
                pool_idx = g if g < len(config.entity_pools[e]) else 0
                input_sample[key] = config.entity_pools[e][pool_idx]
            else:
                input_sample[key] = None

    return input_sample


def create_positional_entity_causal_model(
    config: EntityBindingTaskConfig,
) -> CausalModel:
    """
    Create the POSITIONAL ENTITY binding causal model.

    This is an extended version of the positional model that makes position computation
    more explicit through additional intermediate variables.

    New intermediate variables:
    - positional_entity_g{g}_e{e}: The position (group index) of each entity
    - positional_query_e{e}: Tuple of group positions where query entity at position e appears
    - positional_answer: The final group position to retrieve from (intersection of queries)

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

    # Build variable list
    variables = []

    # Entity variables - one for each possible position
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            variables.append(f"entity_g{g}_e{e}")

    # Positional entity variables - position of each entity
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            variables.append(f"positional_entity_g{g}_e{e}")

    # Query and control variables
    variables.extend(
        [
            "query_group",  # Which group SHOULD contain the query (input)
            "query_indices",  # Tuple of entity indices mentioned in question
            "answer_index",  # Index of the answer entity within query group
            "active_groups",  # Number of groups actually used
            "entities_per_group",  # Number of entities per group
            "statement_template",  # Template for factual statements
            "question_template",  # Template for questions
        ]
    )

    # Positional query variables - one for each possible entity position
    for e in range(config.max_entities_per_group):
        variables.append(f"positional_query_e{e}")

    # Positional answer variable - final position after intersection
    variables.extend(
        [
            "positional_answer",  # Which group contains the answer (from intersection)
            "raw_input",  # The complete prompt text
            "raw_output",  # The expected answer text
        ]
    )

    # Build values dictionary
    values = {}

    # Entity values
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"entity_g{g}_e{e}"
            if e in config.entity_pools:
                values[key] = config.entity_pools[e] + [None]
            else:
                values[key] = [None]

    # Positional entity values - can be any group index or None
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"positional_entity_g{g}_e{e}"
            values[key] = list(range(config.max_groups)) + [None]

    # Query and control values
    values.update(
        {
            "query_group": list(range(config.max_groups)),
            "query_indices": [tuple([i]) for i in range(config.max_entities_per_group)],
            "answer_index": list(range(config.max_entities_per_group)),
            "active_groups": list(range(1, config.max_groups + 1)),
            "entities_per_group": [config.max_entities_per_group],
            "statement_template": [config.statement_template],
            "question_template": list(config.question_templates.values()),
        }
    )

    # Positional query values - tuples of group indices
    for e in range(config.max_entities_per_group):
        key = f"positional_query_e{e}"
        values[key] = None  # Computed

    # Positional answer and output values
    values.update(
        {
            "positional_answer": None,  # Computed
            "raw_input": None,
            "raw_output": None,
        }
    )

    # Build parents dictionary
    parents = {}

    # Entity variables are independent
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            parents[f"entity_g{g}_e{e}"] = []

    # Positional entity variables depend only on their entity
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            parents[f"positional_entity_g{g}_e{e}"] = [f"entity_g{g}_e{e}"]

    # Control variables are independent
    parents.update(
        {
            "query_group": [],
            "query_indices": [],
            "answer_index": [],
            "active_groups": [],
            "entities_per_group": [],
            "statement_template": [],
        }
    )

    # Question template depends on query pattern
    parents["question_template"] = ["query_indices", "answer_index"]

    # Positional query variables depend on entities, positional entities, and query info
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

    for e in range(config.max_entities_per_group):
        parents[f"positional_query_e{e}"] = (
            entity_vars
            + positional_entity_vars
            + ["query_group", "query_indices", "active_groups", "entities_per_group"]
        )

    # Positional answer depends on all positional query variables
    positional_query_vars = [
        f"positional_query_e{e}" for e in range(config.max_entities_per_group)
    ]
    parents["positional_answer"] = positional_query_vars + ["query_indices"]

    # Raw input depends on entities and templates (same as other models)
    parents["raw_input"] = entity_vars + [
        "statement_template",
        "question_template",
        "query_group",
        "query_indices",
        "active_groups",
        "entities_per_group",
    ]

    # Raw output depends on POSITIONAL_ANSWER
    parents["raw_output"] = entity_vars + [
        "positional_answer",
        "answer_index",
        "active_groups",
        "entities_per_group",
    ]

    # Build mechanisms dictionary
    mechanisms = {}

    # Entity sampling mechanisms
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"entity_g{g}_e{e}"
            if e in config.entity_pools:
                pool = config.entity_pools[e]
                mechanisms[key] = lambda t, pool=pool: random.choice(pool)
            else:
                mechanisms[key] = lambda t: None

    # Positional entity mechanisms - simply return the group index
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"positional_entity_g{g}_e{e}"
            mechanisms[key] = lambda t, g=g: g if t[f"entity_g{g}_e{e}"] is not None else None

    # Control variable mechanisms
    mechanisms.update({
        "query_group": lambda t: random.randint(0, config.max_groups - 1),
        "query_indices": lambda t: tuple([random.randint(0, config.max_entities_per_group - 1)]),
        "answer_index": lambda t: random.randint(0, config.max_entities_per_group - 1),
        "active_groups": lambda t: random.randint(2, config.max_groups),
        "entities_per_group": lambda t: config.max_entities_per_group,
        "statement_template": lambda t: config.statement_template,
    })

    # Question template selection mechanism
    def select_question_template(trace):
        """Select question template based on query pattern."""
        query_indices = trace["query_indices"]
        answer_index = trace["answer_index"]

        if isinstance(query_indices, list):
            query_indices = tuple(query_indices)

        key = (query_indices, answer_index)
        if key in config.question_templates:
            return config.question_templates[key]
        else:
            return "What is the answer?"

    mechanisms["question_template"] = select_question_template

    # Positional query mechanisms - find groups where query entity appears at this position
    def create_positional_query_mechanism(entity_position):
        """Create mechanism for positional_query_e{entity_position}."""
        def mechanism(trace):
            query_group = trace["query_group"]
            query_indices = trace["query_indices"]
            active_groups = trace["active_groups"]

            # Check if this entity position is in the query
            if entity_position not in query_indices:
                # Not queried, return empty tuple
                return ()

            # Get the query entity at this position
            query_entity = trace[f"entity_g{query_group}_e{entity_position}"]
            if query_entity is None:
                return ()

            # Search for groups where this entity appears at this position
            matching_groups = []
            for g in range(active_groups):
                entity = trace[f"entity_g{g}_e{entity_position}"]
                if entity == query_entity:
                    # Found matching entity at this position
                    group_pos = trace[f"positional_entity_g{g}_e{entity_position}"]
                    if group_pos is not None:
                        matching_groups.append(group_pos)

            return tuple(matching_groups)

        return mechanism

    for e in range(config.max_entities_per_group):
        key = f"positional_query_e{e}"
        mechanisms[key] = create_positional_query_mechanism(e)

    # Positional answer mechanism - intersection of all query positions
    def compute_positional_answer(trace):
        """Compute the single position by intersecting all positional queries."""
        query_indices = trace["query_indices"]

        # Start with all positions from first queried entity
        if not query_indices:
            return None

        # Get positions for all queried entities
        candidate_sets = []
        for entity_idx in query_indices:
            query_positions = trace[f"positional_query_e{entity_idx}"]
            if query_positions:
                candidate_sets.append(set(query_positions))

        if not candidate_sets:
            return None

        # Find intersection
        intersection = candidate_sets[0]
        for candidate_set in candidate_sets[1:]:
            intersection = intersection.intersection(candidate_set)

        # Must have exactly one position
        if len(intersection) == 0:
            # No matching position found
            return None
        elif len(intersection) > 1:
            # Ambiguous - multiple positions match
            raise ValueError(
                f"Ambiguous query: {len(intersection)} positions match the query. "
                f"Matching positions: {sorted(intersection)}. "
                f"Entity binding should have distinct entities to avoid ambiguity."
            )

        # Return the single position
        return list(intersection)[0]

    mechanisms["positional_answer"] = compute_positional_answer

    # Raw input generation mechanism (same as other models)
    def generate_raw_input(trace):
        """Generate the complete prompt text."""
        question_template = trace["question_template"]
        query_group = trace["query_group"]
        query_indices = trace["query_indices"]
        active_groups = trace["active_groups"]
        entities_per_group = trace["entities_per_group"]

        # Build entity dictionary
        entity_dict = {}
        for g in range(config.max_groups):
            for e in range(config.max_entities_per_group):
                entity_dict[f"entity_g{g}_e{e}"] = trace[f"entity_g{g}_e{e}"]

        # Create binding matrix
        groups = []
        for g in range(active_groups):
            entities = []
            for e in range(entities_per_group):
                entity = entity_dict[f"entity_g{g}_e{e}"]
                if entity is not None:
                    entities.append(entity)
                else:
                    entities.append(f"MISSING_{g}_{e}")
            groups.append(EntityGroup(entities, g))

        matrix = BindingMatrix(groups, config.max_groups, config.max_entities_per_group)

        # Use template processor to generate text
        processor = TemplateProcessor(config)

        try:
            statement = processor.fill_statement_template(matrix)
            question = processor.fill_question_template(
                question_template, query_group, query_indices, matrix
            )
            separator = config.statement_question_separator
            core_prompt = f"{statement}{separator}{question}"
            return f"{config.prompt_prefix}{core_prompt}{config.prompt_suffix}"
        except Exception:
            return "Invalid configuration"

    mechanisms["raw_input"] = generate_raw_input

    # Raw output generation mechanism - uses positional_answer
    def generate_raw_output(trace):
        """Generate the expected answer using positional_answer."""
        positional_answer = trace["positional_answer"]
        answer_index = trace["answer_index"]
        active_groups = trace["active_groups"]
        entities_per_group = trace["entities_per_group"]

        # Use positional_answer to retrieve
        if positional_answer is not None and positional_answer < active_groups and answer_index < entities_per_group:
            answer_entity = trace[f"entity_g{positional_answer}_e{answer_index}"]
            if answer_entity is not None:
                return answer_entity

        return "UNKNOWN"

    mechanisms["raw_output"] = generate_raw_output

    # Create the causal model
    model_id = f"entity_binding_positional_entity_{config.max_groups}g_{config.max_entities_per_group}e"
    return CausalModel(variables, values, parents, mechanisms, id=model_id)
