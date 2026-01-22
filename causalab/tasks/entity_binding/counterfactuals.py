"""
Counterfactual dataset generation for entity binding tasks.

This module provides functions to generate counterfactual examples by
swapping entity groups while keeping the query the same.
"""

import random

from causalab.causal.counterfactual_dataset import CounterfactualExample

from .causal_models import (
    create_positional_entity_causal_model,
    sample_valid_entity_binding_input,
)
from .config import EntityBindingTaskConfig


def swap_query_group(
    config: EntityBindingTaskConfig, change_answer: bool = False
) -> CounterfactualExample:
    """
    Generate a counterfactual by swapping the queried entity group with another group.

    This tests whether the model correctly retrieves information based on which
    entity group is queried, rather than relying on positional information.

    Example with 3 groups:
        Input:
            Entities: g0=(Pete, jam), g1=(Ann, pie), g2=(Bob, cake)
            Query: group 1, entity 0 -> asking about Ann
            Prompt:  "Pete loves jam, Ann loves pie, and Bob loves cake. What does Ann love?"
            Answer:  "pie"

        Counterfactual (swapped g1 with g2):
            Entities: g0=(Pete, jam), g1=(Bob, cake), g2=(Ann, pie)
            Query: group 1, entity 0 -> now asking about Bob (who moved to g1)
            Prompt:  "Pete loves jam, Bob loves cake, and Ann loves pie. What does Bob love?"
            Answer:  "cake"

    The counterfactual swaps the entity groups but keeps the SAME QUERY POSITION.
    This means:
    - We're querying the same POSITION in the binding matrix (e.g., group 1, entity 0)
    - But different ENTITIES now occupy that position
    - The model must retrieve the binding at that position, not memorize entity names

    If config.fixed_query_indices is set, query_indices will be fixed to that value.

    Parameters
    ----------
    config : EntityBindingTaskConfig
        The task configuration
    change_answer : bool, optional
        If True, replace the answer entity in the counterfactual with a new entity
        from the same pool (different from all entities in the sample). This creates
        a counterfactual with a different expected answer. Default is False.

    Returns
    -------
    CounterfactualExample
    """
    # Create causal model
    model = create_positional_entity_causal_model(config)

    # Sample a valid input
    input_sample = sample_valid_entity_binding_input(config, model=model)

    # Get query_group directly from the input sample
    query_group = input_sample["query_group"]
    active_groups = input_sample["active_groups"]

    # Choose a different group to swap with
    other_groups = [g for g in range(active_groups) if g != query_group]
    if not other_groups:
        # Only one group active - cannot swap, fall back to random counterfactual
        import warnings

        warnings.warn(
            f"swap_query_group called with only one active group ({active_groups}). "
            "Falling back to random counterfactual sampling."
        )
        counterfactual = sample_valid_entity_binding_input(config, model=model)
        return CounterfactualExample(
            input=input_sample, counterfactual_inputs=[counterfactual]
        )

    swap_group = random.choice(other_groups)

    # Build counterfactual dict from INPUT variables only
    cf_dict = {var: input_sample[var] for var in model.inputs}

    # Swap entities between query_group and swap_group
    entities_per_group = input_sample["entities_per_group"]
    for e in range(entities_per_group):
        key_query = f"entity_g{query_group}_e{e}"
        key_swap = f"entity_g{swap_group}_e{e}"

        # Swap the entities
        cf_dict[key_query], cf_dict[key_swap] = cf_dict[key_swap], cf_dict[key_query]

    # KEY: Update query_group and query_e{e} to follow where the original query entity moved
    # After the swap, the original query entities are now at swap_group
    cf_dict["query_group"] = swap_group

    # Set query_e{e} from the entities at swap_group (which now has original entities)
    for e in range(entities_per_group):
        cf_dict[f"query_e{e}"] = cf_dict[f"entity_g{swap_group}_e{e}"]

    # Optionally change the answer entity to a new one
    if change_answer:
        answer_index = cf_dict["answer_index"]
        # The answer entity is at position answer_index in the queried group
        # After swap, the queried group is swap_group
        answer_key = f"entity_g{swap_group}_e{answer_index}"

        # Collect all entities currently in the sample
        used_entities = set()
        for g in range(cf_dict["active_groups"]):
            for e in range(cf_dict["entities_per_group"]):
                entity = cf_dict.get(f"entity_g{g}_e{e}")
                if entity:
                    used_entities.add(entity)

        # Get available entities from the same pool (same entity role)
        available = [
            ent for ent in config.entity_pools[answer_index] if ent not in used_entities
        ]

        if available:
            new_answer = random.choice(available)
            cf_dict[answer_key] = new_answer
            # Also update query_e{answer_index} if it was the answer position
            cf_dict[f"query_e{answer_index}"] = new_answer

    # Create counterfactual trace - this computes raw_input, raw_output, etc.
    counterfactual = model.new_trace(cf_dict)

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


def random_counterfactual(config: EntityBindingTaskConfig) -> CounterfactualExample:
    """
    Generate a completely random counterfactual by sampling two independent inputs.

    This is a baseline condition - the counterfactual is unrelated to the input.

    Parameters
    ----------
    config : EntityBindingTaskConfig
        The task configuration

    Returns
    -------
    CounterfactualExample
        Dictionary with:
        - "input": The original input sample (CausalTrace)
        - "counterfactual_inputs": List containing one counterfactual sample (CausalTrace)
    """
    model = create_positional_entity_causal_model(config)

    # Sample two independent inputs as CausalTraces
    input_sample = sample_valid_entity_binding_input(config, model=model)
    counterfactual = sample_valid_entity_binding_input(config, model=model)

    return CounterfactualExample(
        input=input_sample, counterfactual_inputs=[counterfactual]
    )
