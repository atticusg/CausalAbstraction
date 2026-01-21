"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Counterfactual dataset generation for entity binding tasks.

This module provides functions to generate counterfactual examples by
swapping entity groups while keeping the query the same.
"""

import random
from typing import Dict, Any
from .causal_models import create_direct_causal_model, sample_valid_entity_binding_input
from .config import EntityBindingTaskConfig


def swap_query_group(config: EntityBindingTaskConfig) -> Dict[str, Any]:
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

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - "input": The original input sample
        - "counterfactual_inputs": List containing one counterfactual sample
    """
    # Create causal model
    model = create_direct_causal_model(config)

    # Sample a valid input (will use config.fixed_query_indices if set)
    input_sample = sample_valid_entity_binding_input(config)

    # Regenerate raw_input for the input sample

    # Identify which group is being queried
    query_group = input_sample["query_group"]
    active_groups = input_sample["active_groups"]

    # Choose a different group to swap with
    other_groups = [g for g in range(active_groups) if g != query_group]
    if not other_groups:
        # Only one group active, return random counterfactual instead
        counterfactual_input = sample_valid_entity_binding_input(config)
        return {"input": input_sample.to_dict() if hasattr(input_sample, "to_dict") else input_sample, "counterfactual_inputs": [counterfactual_input]}

    swap_group = random.choice(other_groups)

    # Create counterfactual by swapping the entity groups
    counterfactual = input_sample.to_dict() if hasattr(input_sample, "to_dict") else input_sample.copy()

    # Swap entities between query_group and swap_group
    entities_per_group = input_sample["entities_per_group"]
    for e in range(entities_per_group):
        key_query = f"entity_g{query_group}_e{e}"
        key_swap = f"entity_g{swap_group}_e{e}"

        # Swap the entities
        counterfactual[key_query] = input_sample[key_swap]
        counterfactual[key_swap] = input_sample[key_query]

    # KEY: Update query_group to follow where the original query entity moved
    # After the swap, the original query entity is now at swap_group
    counterfactual["query_group"] = swap_group

    # This keeps the SAME QUESTION (asking about the same entity)
    # but the statement has changed (entities in different positions)

    # Remove raw_input if copied
    if "raw_input" in counterfactual:
        del counterfactual["raw_input"]


    return {"input": input_sample.to_dict() if hasattr(input_sample, "to_dict") else input_sample, "counterfactual_inputs": [counterfactual]}


def swap_query_group_preserve_answer(config: EntityBindingTaskConfig) -> Dict[str, Any]:
    """
    Generate a counterfactual by swapping entity groups AND adjusting the query.

    Unlike swap_query_group, this version keeps the ANSWER the same by also
    changing which entity is queried.

    Example:
        Input:     "Pete loves jam, and Ann loves pie. What does Pete love?"
        Expected answer: "jam"

        Counterfactual: "Ann loves pie, and Pete loves jam. What does Ann love?"
        Expected answer: "pie" (different answer because we're querying different entity)

    This tests whether the model can retrieve the correct binding regardless of
    which entity is queried.

    Parameters
    ----------
    config : EntityBindingTaskConfig
        The task configuration

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - "input": The original input sample
        - "counterfactual_inputs": List containing one counterfactual sample
    """
    # Create causal model
    model = create_direct_causal_model(config)

    # Sample a valid input
    input_sample = sample_valid_entity_binding_input(config)

    # Regenerate raw_input for the input sample

    # Identify which group is being queried
    query_group = input_sample["query_group"]
    active_groups = input_sample["active_groups"]

    # Choose a different group to swap with
    other_groups = [g for g in range(active_groups) if g != query_group]
    if not other_groups:
        # Only one group active, return random counterfactual instead
        counterfactual_input = sample_valid_entity_binding_input(config)
        return {"input": input_sample.to_dict() if hasattr(input_sample, "to_dict") else input_sample, "counterfactual_inputs": [counterfactual_input]}

    swap_group = random.choice(other_groups)

    # Create counterfactual by swapping the entity groups
    counterfactual = input_sample.to_dict() if hasattr(input_sample, "to_dict") else input_sample.copy()

    # Swap entities between query_group and swap_group
    entities_per_group = input_sample["entities_per_group"]
    for e in range(entities_per_group):
        key_query = f"entity_g{query_group}_e{e}"
        key_swap = f"entity_g{swap_group}_e{e}"

        # Swap the entities
        counterfactual[key_query] = input_sample[key_swap]
        counterfactual[key_swap] = input_sample[key_query]

    # ALSO swap which group is queried to preserve the semantic meaning
    counterfactual["query_group"] = swap_group

    # Regenerate the prompt and answer for the counterfactual
    # Remove raw_input if copied, will be regenerated
    if "raw_input" in counterfactual:
        del counterfactual["raw_input"]


    return {"input": input_sample.to_dict() if hasattr(input_sample, "to_dict") else input_sample, "counterfactual_inputs": [counterfactual]}


def swap_non_query_groups(config: EntityBindingTaskConfig) -> Dict[str, Any]:
    """
    Generate a counterfactual by swapping two NON-QUERIED entity groups.

    This tests whether the model is affected by changes to irrelevant entities.
    The query group stays the same, and so does the answer.

    Example:
        Input:     "Pete loves jam, Ann loves pie, and Bob loves cake. What does Pete love?"
        Expected answer: "jam"

        Counterfactual: "Pete loves jam, Bob loves cake, and Ann loves pie. What does Pete love?"
        Expected answer: "jam" (still correct, Ann and Bob swapped but Pete unchanged)

    This is a control condition - the model should give the same answer since
    the queried entity hasn't changed.

    Parameters
    ----------
    config : EntityBindingTaskConfig
        The task configuration

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - "input": The original input sample
        - "counterfactual_inputs": List containing one counterfactual sample
    """
    # Create causal model
    model = create_direct_causal_model(config)

    # Sample a valid input with at least 3 groups
    input_sample = sample_valid_entity_binding_input(config)
    # Ensure we have at least 3 active groups
    if input_sample["active_groups"] < 3:
        input_sample["active_groups"] = min(3, config.max_groups)

    # Regenerate raw_input for the input sample

    # Identify which group is being queried
    query_group = input_sample["query_group"]
    active_groups = input_sample["active_groups"]

    # Find two non-queried groups to swap
    non_query_groups = [g for g in range(active_groups) if g != query_group]
    if len(non_query_groups) < 2:
        # Not enough groups to swap, return same input
        return {"input": input_sample.to_dict() if hasattr(input_sample, "to_dict") else input_sample, "counterfactual_inputs": [input_sample.copy()]}

    group1, group2 = random.sample(non_query_groups, 2)

    # Create counterfactual by swapping these two groups
    counterfactual = input_sample.to_dict() if hasattr(input_sample, "to_dict") else input_sample.copy()

    # Swap entities between group1 and group2
    entities_per_group = input_sample["entities_per_group"]
    for e in range(entities_per_group):
        key1 = f"entity_g{group1}_e{e}"
        key2 = f"entity_g{group2}_e{e}"

        # Swap the entities
        counterfactual[key1] = input_sample[key2]
        counterfactual[key2] = input_sample[key1]

    # The query_group stays the same, so the answer should be the same

    # Regenerate the prompt for the counterfactual
    # Remove raw_input if copied, will be regenerated
    if "raw_input" in counterfactual:
        del counterfactual["raw_input"]


    return {"input": input_sample.to_dict() if hasattr(input_sample, "to_dict") else input_sample, "counterfactual_inputs": [counterfactual]}


def random_counterfactual(config: EntityBindingTaskConfig) -> Dict[str, Any]:
    """
    Generate a completely random counterfactual by sampling two independent inputs.

    This is a baseline condition - the counterfactual is unrelated to the input.

    Parameters
    ----------
    config : EntityBindingTaskConfig
        The task configuration

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - "input": The original input sample
        - "counterfactual_inputs": List containing one counterfactual sample
    """
    model = create_direct_causal_model(config)

    # Sample two independent inputs
    input_sample = sample_valid_entity_binding_input(config)

    counterfactual = sample_valid_entity_binding_input(config)

    return {"input": input_sample.to_dict() if hasattr(input_sample, "to_dict") else input_sample, "counterfactual_inputs": [counterfactual]}


# Example usage and testing
if __name__ == "__main__":
    from .config import create_sample_love_config

    config = create_sample_love_config()

    print("=" * 70)
    print("COUNTERFACTUAL DATASET EXAMPLES")
    print("=" * 70)

    # Test swap_query_group
    print("\n1. SWAP QUERY GROUP (query same entity, but it has different binding)")
    print("-" * 70)
    example = swap_query_group(config)
    input_s = example["input"]
    counter_s = example["counterfactual_inputs"][0]

    print(f"Input:          {input_s['raw_input']}")
    print(f"Answer:         {input_s['raw_output']}")
    print(f"Query group:    {input_s['query_group']}")
    print()
    print(f"Counterfactual: {counter_s['raw_input']}")
    print(f"Answer:         {counter_s['raw_output']}")
    print(f"Query group:    {counter_s['query_group']}")

    # Test swap_non_query_groups (if enough groups)
    if config.max_groups >= 3:
        print("\n2. SWAP NON-QUERY GROUPS (answer should stay the same)")
        print("-" * 70)
        example = swap_non_query_groups(config)
        input_s = example["input"]
        counter_s = example["counterfactual_inputs"][0]

        print(f"Input:          {input_s['raw_input']}")
        print(f"Answer:         {input_s['raw_output']}")
        print()
        print(f"Counterfactual: {counter_s['raw_input']}")
        print(f"Answer:         {counter_s['raw_output']}")
        print(f"Same answer?    {input_s['raw_output'] == counter_s['raw_output']}")

    # Test random_counterfactual
    print("\n3. RANDOM COUNTERFACTUAL (completely independent)")
    print("-" * 70)
    example = random_counterfactual(config)
    input_s = example["input"]
    counter_s = example["counterfactual_inputs"][0]

    print(f"Input:          {input_s['raw_input']}")
    print(f"Answer:         {input_s['raw_output']}")
    print()
    print(f"Counterfactual: {counter_s['raw_input']}")
    print(f"Answer:         {counter_s['raw_output']}")
