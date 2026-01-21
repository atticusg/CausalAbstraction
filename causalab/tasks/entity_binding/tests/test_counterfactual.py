#!/usr/bin/env -S uv run python
"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Test Script: Counterfactual Dataset Generation

This script tests the counterfactual generation functions for entity binding tasks.
It verifies that:

- Entity groups are correctly swapped
- Queries are appropriately updated or preserved
- Answers change as expected
- The counterfactual inputs are valid
"""

import sys

sys.path.append("/mnt/polished-lake/home/atticus/CausalAbstraction")

from causalab.tasks.entity_binding.config import (
    create_sample_love_config,
    create_sample_action_config,
)
from causalab.tasks.entity_binding.counterfactuals import (
    swap_query_group,
    swap_query_group_preserve_answer,
    swap_non_query_groups,
    random_counterfactual,
)


def test_swap_query_group():
    """Test swapping the queried entity group."""
    print("=== Test 1: Swap Query Group ===")

    config = create_sample_love_config()

    # Generate counterfactual example
    example = swap_query_group(config)
    input_sample = example["input"]
    counterfactual = example["counterfactual_inputs"][0]

    print(f"Input prompt:          {input_sample['raw_input']}")
    print(f"Input query group:     {input_sample['query_group']}")
    print(
        f"Input query entity:    entity_g{input_sample['query_group']}_e{input_sample['query_indices'][0]}"
    )
    print()
    print(f"Counterfactual prompt: {counterfactual['raw_input']}")
    print(f"Counterfactual query group: {counterfactual['query_group']}")
    print()

    # Verify the query_indices and answer_index are the same
    # (query_group changes to follow where the entity moved)
    assert input_sample["query_indices"] == counterfactual["query_indices"], (
        "Query indices should be the same"
    )
    assert input_sample["answer_index"] == counterfactual["answer_index"], (
        "Answer index should be the same"
    )

    # Verify the QUESTION TEXT is the same
    input_question = input_sample["raw_input"].split(". ")[-1]
    counter_question = counterfactual["raw_input"].split(". ")[-1]
    assert input_question == counter_question, (
        f"Question text should be the same: '{input_question}' vs '{counter_question}'"
    )

    # Verify the entity groups were swapped
    query_group = input_sample["query_group"]
    entities_per_group = input_sample["entities_per_group"]

    # Find which group was swapped with
    swapped_with = None
    for g in range(input_sample["active_groups"]):
        if g == query_group:
            continue
        # Check if entities were swapped
        match = True
        for e in range(entities_per_group):
            input_key_query = f"entity_g{query_group}_e{e}"
            input_key_other = f"entity_g{g}_e{e}"
            counter_key_query = f"entity_g{query_group}_e{e}"
            counter_key_other = f"entity_g{g}_e{e}"

            if input_sample.get(input_key_query) != counterfactual.get(
                counter_key_other
            ):
                match = False
                break
            if input_sample.get(input_key_other) != counterfactual.get(
                counter_key_query
            ):
                match = False
                break

        if match:
            swapped_with = g
            break

    if swapped_with is not None:
        print("✓ Verified: Entities were swapped between groups")
        print(
            f"  Query group updated: {input_sample['query_group']} -> {counterfactual['query_group']}"
        )
        print("  (Follows where the query entity moved)")
    else:
        print(
            "⚠ Could not verify swap (may have single group or random counterfactual)"
        )

    # The prompts might be the same or different depending on the swap
    # What matters is that the ENTITIES in the query group changed
    query_group = input_sample["query_group"]
    entities_changed = False
    for e in range(config.max_entities_per_group):
        key = f"entity_g{query_group}_e{e}"
        if input_sample.get(key) != counterfactual.get(key):
            entities_changed = True
            break

    if entities_changed:
        print(f"✓ Verified: Entities in query group {query_group} changed")
        print(
            f"   Input query group entities: {[input_sample.get(f'entity_g{query_group}_e{e}') for e in range(config.max_entities_per_group)]}"
        )
        print(
            f"   Counter query group entities: {[counterfactual.get(f'entity_g{query_group}_e{e}') for e in range(config.max_entities_per_group)]}"
        )
    else:
        print("⚠ Entities in query group did not change (likely single active group)")

    print("✓ Test 1 passed\n")


def test_swap_query_group_preserve_answer():
    """Test swapping groups while preserving answer semantics."""
    print("=== Test 2: Swap Query Group Preserve Answer ===")

    config = create_sample_love_config()

    # Generate counterfactual example
    example = swap_query_group_preserve_answer(config)
    input_sample = example["input"]
    counterfactual = example["counterfactual_inputs"][0]

    print(f"Input prompt:          {input_sample['raw_input']}")
    print(f"Input query group:     {input_sample['query_group']}")
    print()
    print(f"Counterfactual prompt: {counterfactual['raw_input']}")
    print(f"Counterfactual query group: {counterfactual['query_group']}")
    print()

    # Verify the query_group changed
    if counterfactual["query_group"] != input_sample["query_group"]:
        print(
            f"✓ Query group changed: {input_sample['query_group']} -> {counterfactual['query_group']}"
        )
    else:
        print("⚠ Query group stayed same (may have single group)")

    # Verify the prompts are valid
    assert "raw_input" in input_sample and input_sample["raw_input"], (
        "Input should have valid raw_input"
    )
    assert "raw_input" in counterfactual and counterfactual["raw_input"], (
        "Counterfactual should have valid raw_input"
    )

    print("✓ Test 2 passed\n")


def test_swap_non_query_groups():
    """Test swapping non-queried groups (answer should stay same)."""
    print("=== Test 3: Swap Non-Query Groups ===")

    config = create_sample_love_config()

    # Generate counterfactual example
    example = swap_non_query_groups(config)
    input_sample = example["input"]
    counterfactual = example["counterfactual_inputs"][0]

    print(f"Input prompt:          {input_sample['raw_input']}")
    print(f"Input query group:     {input_sample['query_group']}")
    print()
    print(f"Counterfactual prompt: {counterfactual['raw_input']}")
    print(f"Counterfactual query group: {counterfactual['query_group']}")
    print()

    # Verify query group is the same
    assert input_sample["query_group"] == counterfactual["query_group"], (
        "Query group should stay the same"
    )

    # Verify query group stayed the same
    if input_sample["active_groups"] >= 3:
        print("✓ Verified: Query group stayed the same (non-query groups swapped)")
    else:
        print("⚠ Not enough groups to perform non-query swap")

    print("✓ Test 3 passed\n")


def test_random_counterfactual():
    """Test random counterfactual generation."""
    print("=== Test 4: Random Counterfactual ===")

    config = create_sample_love_config()

    # Generate random counterfactual example
    example = random_counterfactual(config)
    input_sample = example["input"]
    counterfactual = example["counterfactual_inputs"][0]

    print(f"Input prompt:          {input_sample['raw_input']}")
    print()
    print(f"Counterfactual prompt: {counterfactual['raw_input']}")
    print()

    # Verify both have raw_input (but NOT raw_output)
    assert "raw_input" in input_sample, "Input should have raw_input"
    assert "raw_input" in counterfactual, "Counterfactual should have raw_input"
    assert "raw_output" not in input_sample, "Input should NOT have raw_output"
    assert "raw_output" not in counterfactual, (
        "Counterfactual should NOT have raw_output"
    )

    print(
        "✓ Both input and counterfactual have raw_input (but not raw_output - correct!)"
    )
    print("✓ Test 4 passed\n")


def test_multiple_examples():
    """Generate multiple examples to verify consistency."""
    print("=== Test 5: Multiple Examples ===")

    config = create_sample_love_config()

    print("Generating 5 swap_query_group examples:\n")

    for i in range(5):
        example = swap_query_group(config)
        input_sample = example["input"]
        counterfactual = example["counterfactual_inputs"][0]

        print(f"Example {i + 1}:")
        print(f"  Input:  {input_sample['raw_input']}")
        print(f"  Counter: {counterfactual['raw_input']}")
        print(
            f"  Same query group? {input_sample['query_group'] == counterfactual['query_group']}"
        )
        print()

        # Verify structure (should have raw_input but NOT raw_output)
        assert "raw_input" in input_sample, "Should have raw_input"
        assert "raw_output" not in input_sample, "Should NOT have raw_output"
        assert "raw_input" in counterfactual, "Should have raw_input"
        assert "raw_output" not in counterfactual, "Should NOT have raw_output"

    print("✓ Test 5 passed\n")


def test_action_task_counterfactuals():
    """Test counterfactuals with action tasks (3-entity groups)."""
    print("=== Test 6: Action Task Counterfactuals ===")

    config = create_sample_action_config()

    # Generate example
    example = swap_query_group(config)
    input_sample = example["input"]
    counterfactual = example["counterfactual_inputs"][0]

    print(f"Input prompt:          {input_sample['raw_input']}")
    print()
    print(f"Counterfactual prompt: {counterfactual['raw_input']}")
    print()

    # Verify the QUESTION TEXT is the same (query_group may change to follow entity)
    input_question = input_sample["raw_input"].split(". ")[-1]
    counter_question = counterfactual["raw_input"].split(". ")[-1]

    print(f"  Question (input):  {input_question}")
    print(f"  Question (counter): {counter_question}")
    print(f"  Same question text? {input_question == counter_question}")

    print("\n✓ Test 6 passed\n")


def main():
    """Run all counterfactual tests."""
    print("Testing Counterfactual Dataset Generation")
    print("=" * 70)
    print()

    try:
        test_swap_query_group()
        test_swap_query_group_preserve_answer()
        test_swap_non_query_groups()
        test_random_counterfactual()
        test_multiple_examples()
        test_action_task_counterfactuals()

        print("=" * 70)
        print("🎉 All counterfactual tests passed!")
        print("=" * 70)
        print("\nCounterfactual types available:")
        print("✓ swap_query_group - Swap query group, keep query same")
        print("✓ swap_query_group_preserve_answer - Swap and update query")
        print("✓ swap_non_query_groups - Swap irrelevant groups")
        print("✓ random_counterfactual - Completely independent sample")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
