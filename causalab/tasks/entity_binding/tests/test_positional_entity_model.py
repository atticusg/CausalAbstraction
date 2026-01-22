#!/usr/bin/env -S uv run python
"""Test Script: Positional Entity Causal Model.

This script tests the positional entity causal model which breaks down
positional computation into multiple stages with explicit intermediate variables.

Tests cover:
1. Basic functionality and correctness
2. Intervention effects on each variable type
3. Multiple query entities
"""

from causalab.tasks.entity_binding.config import (
    create_sample_love_config,
    create_sample_action_config,
)
from causalab.tasks.entity_binding.causal_models import (
    create_positional_entity_causal_model,
    sample_valid_entity_binding_input,
)


def test_basic_functionality():
    """Test that the model generates correct outputs."""
    print("=== Test 1: Basic Functionality ===")

    config = create_sample_love_config()
    model = create_positional_entity_causal_model(config)

    # Sample input
    input_sample = sample_valid_entity_binding_input(config)
    trace = model.new_trace(input_sample)

    print(f"Prompt: {trace['raw_input']}")
    print(f"Answer: {trace['raw_output']}")
    print()

    # Verify all intermediate variables are computed
    assert "positional_entity_g0_e0" in trace, "Should have positional entity variables"
    assert "positional_query_e0" in trace, "Should have positional query variables"
    assert "positional_answer" in trace, "Should have positional answer variable"
    assert trace["raw_output"] != "UNKNOWN", "Should have valid answer"

    print("✓ Test 1 passed\n")


def test_positional_entity_variables():
    """Test that positional entity variables return correct group indices."""
    print("=== Test 2: Positional Entity Variables ===")

    config = create_sample_love_config()
    model = create_positional_entity_causal_model(config)

    # Create specific input
    input_sample = {
        "entity_g0_e0": "Pete",
        "entity_g0_e1": "jam",
        "entity_g1_e0": "Ann",
        "entity_g1_e1": "pie",
        "entity_g2_e0": "Bob",
        "entity_g2_e1": "cake",
        "query_e0": "Ann",
        "query_e1": "pie",
        "query_group": 1,
        "query_indices": (0,),
        "answer_index": 1,
        "active_groups": 3,
        "entities_per_group": 2,
        "statement_template": config.statement_template,
    }

    trace = model.new_trace(input_sample)

    print("Entities and their positions:")
    print(f"  Pete at g0_e0 -> position: {trace['positional_entity_g0_e0']}")
    print(f"  Ann at g1_e0 -> position: {trace['positional_entity_g1_e0']}")
    print(f"  Bob at g2_e0 -> position: {trace['positional_entity_g2_e0']}")
    print()

    # Verify positional entity variables return group indices
    assert trace["positional_entity_g0_e0"] == 0, "Pete should be at position 0"
    assert trace["positional_entity_g1_e0"] == 1, "Ann should be at position 1"
    assert trace["positional_entity_g2_e0"] == 2, "Bob should be at position 2"

    print("✓ Test 2 passed\n")


def test_positional_query_variables():
    """Test that positional query variables find correct groups."""
    print("=== Test 3: Positional Query Variables ===")

    config = create_sample_love_config()
    model = create_positional_entity_causal_model(config)

    input_sample = {
        "entity_g0_e0": "Pete",
        "entity_g0_e1": "jam",
        "entity_g1_e0": "Ann",
        "entity_g1_e1": "pie",
        "entity_g2_e0": "Bob",
        "entity_g2_e1": "cake",
        "query_e0": "Ann",
        "query_e1": "pie",
        "query_group": 1,
        "query_indices": (0,),  # Querying position 0 (Ann)
        "answer_index": 1,
        "active_groups": 3,
        "entities_per_group": 2,
        "statement_template": config.statement_template,
    }

    trace = model.new_trace(input_sample)

    print(
        f"Query: asking about entity at position 0 in group {input_sample['query_group']}"
    )
    print(f"Query entity: {input_sample['query_e0']}")
    print(f"positional_query_e0: {trace['positional_query_e0']}")
    print(f"positional_query_e1: {trace['positional_query_e1']}")
    print()

    # Position 0 is queried, should find Ann at group 1
    assert trace["positional_query_e0"] == (1,), (
        f"Should find Ann at group 1, got {trace['positional_query_e0']}"
    )
    # Position 1 is not queried, should be empty
    assert trace["positional_query_e1"] == (), (
        f"Position 1 not queried, should be empty, got {trace['positional_query_e1']}"
    )

    print("✓ Test 3 passed\n")


def test_positional_answer_variable():
    """Test that positional answer computes correct intersection."""
    print("=== Test 4: Positional Answer Variable ===")

    config = create_sample_love_config()
    model = create_positional_entity_causal_model(config)

    input_sample = {
        "entity_g0_e0": "Pete",
        "entity_g0_e1": "jam",
        "entity_g1_e0": "Ann",
        "entity_g1_e1": "pie",
        "entity_g2_e0": "Bob",
        "entity_g2_e1": "cake",
        "query_e0": "Ann",
        "query_e1": "pie",
        "query_group": 1,
        "query_indices": (0,),
        "answer_index": 1,
        "active_groups": 3,
        "entities_per_group": 2,
        "statement_template": config.statement_template,
    }

    trace = model.new_trace(input_sample)

    print(f"Positional queries: {trace['positional_query_e0']}")
    print(f"Intersection (positional_answer): {trace['positional_answer']}")
    print("Expected: group 1 (where Ann is)")
    print()

    assert trace["positional_answer"] == 1, "Should retrieve from group 1"
    assert trace["raw_output"] == "pie", (
        f"Should answer 'pie', got {trace['raw_output']}"
    )

    print("✓ Test 4 passed\n")


def test_intervention_on_positional_entity():
    """Test interventions on positional entity variables."""
    print("=== Test 5: Intervention on Positional Entity Variables ===")

    config = create_sample_love_config()
    model = create_positional_entity_causal_model(config)

    input_sample = {
        "entity_g0_e0": "Pete",
        "entity_g0_e1": "jam",
        "entity_g1_e0": "Ann",
        "entity_g1_e1": "pie",
        "entity_g2_e0": "Bob",
        "entity_g2_e1": "cake",
        "query_e0": "Ann",
        "query_e1": "pie",
        "query_group": 1,
        "query_indices": (0,),
        "answer_index": 1,
        "active_groups": 3,
        "entities_per_group": 2,
        "statement_template": config.statement_template,
    }

    # Normal run
    normal = model.new_trace(input_sample)
    print(
        f"Normal: Query Ann (g1_e0) -> position {normal['positional_entity_g1_e0']} -> answer {normal['raw_output']}"
    )

    # Intervene on positional_entity_g1_e0 to make Ann appear at position 2
    intervened = normal.copy()
    intervened.intervene("positional_entity_g1_e0", 2)

    print(f"Intervened: Ann now at position {intervened['positional_entity_g1_e0']}")
    print(f"  positional_query_e0: {intervened['positional_query_e0']}")
    print(f"  positional_answer: {intervened['positional_answer']}")
    print(f"  Answer: {intervened['raw_output']}")
    print()

    # After intervention, should look for Ann at position 2
    assert intervened["positional_entity_g1_e0"] == 2, (
        "Intervention should change position"
    )
    assert intervened["positional_query_e0"] == (2,), (
        "Should search and find position 2"
    )
    assert intervened["positional_answer"] == 2, "Should retrieve from position 2"
    assert intervened["raw_output"] == "cake", "Should get Bob's answer (cake)"

    print("✓ Test 5 passed\n")


def test_intervention_on_positional_query():
    """Test interventions on positional query variables."""
    print("=== Test 6: Intervention on Positional Query Variables ===")

    config = create_sample_love_config()
    model = create_positional_entity_causal_model(config)

    input_sample = {
        "entity_g0_e0": "Pete",
        "entity_g0_e1": "jam",
        "entity_g1_e0": "Ann",
        "entity_g1_e1": "pie",
        "entity_g2_e0": "Bob",
        "entity_g2_e1": "cake",
        "query_e0": "Ann",
        "query_e1": "pie",
        "query_group": 1,
        "query_indices": (0,),
        "answer_index": 1,
        "active_groups": 3,
        "entities_per_group": 2,
        "statement_template": config.statement_template,
    }

    normal = model.new_trace(input_sample)
    print(
        f"Normal: positional_query_e0={normal['positional_query_e0']}, answer={normal['raw_output']}"
    )

    # Intervene to make positional_query_e0 point to group 0
    intervened = normal.copy()
    intervened.intervene("positional_query_e0", (0,))

    print(f"Intervened: positional_query_e0={intervened['positional_query_e0']}")
    print(f"  positional_answer: {intervened['positional_answer']}")
    print(f"  Answer: {intervened['raw_output']}")
    print()

    assert intervened["positional_query_e0"] == (0,), (
        "Intervention should change query result"
    )
    assert intervened["positional_answer"] == 0, "Should retrieve from group 0"
    assert intervened["raw_output"] == "jam", "Should get Pete's answer (jam)"

    print("✓ Test 6 passed\n")


def test_intervention_on_positional_answer():
    """Test interventions on positional answer variable."""
    print("=== Test 7: Intervention on Positional Answer ===")

    config = create_sample_love_config()
    model = create_positional_entity_causal_model(config)

    input_sample = {
        "entity_g0_e0": "Pete",
        "entity_g0_e1": "jam",
        "entity_g1_e0": "Ann",
        "entity_g1_e1": "pie",
        "entity_g2_e0": "Bob",
        "entity_g2_e1": "cake",
        "query_e0": "Ann",
        "query_e1": "pie",
        "query_group": 1,
        "query_indices": (0,),
        "answer_index": 1,
        "active_groups": 3,
        "entities_per_group": 2,
        "statement_template": config.statement_template,
    }

    normal = model.new_trace(input_sample)
    print(
        f"Normal: positional_answer={normal['positional_answer']}, answer={normal['raw_output']}"
    )

    # Intervene to force retrieval from group 2
    intervened = normal.copy()
    intervened.intervene("positional_answer", 2)

    print(f"Intervened: positional_answer={intervened['positional_answer']}")
    print(f"  Answer: {intervened['raw_output']}")
    print()

    assert intervened["positional_answer"] == 2, (
        "Intervention should change answer position"
    )
    assert intervened["raw_output"] == "cake", "Should retrieve from group 2 (cake)"

    print("✓ Test 7 passed\n")


def test_with_action_config():
    """Test with 3-entity action configuration."""
    print("=== Test 8: Action Configuration (3 entities) ===")

    config = create_sample_action_config()
    model = create_positional_entity_causal_model(config)

    # Test a few samples
    for i in range(3):
        input_sample = sample_valid_entity_binding_input(config)
        trace = model.new_trace(input_sample)

        print(f"Sample {i + 1}:")
        print(f"  Prompt: {trace['raw_input'][:70]}...")
        print(f"  Position: {trace['positional_answer']}")
        print(f"  Answer: {trace['raw_output']}")

        assert trace["raw_output"] != "UNKNOWN", "Should have valid answer"
        print()

    print("✓ Test 8 passed\n")


def test_multiple_query_entities():
    """Test with multiple entities in the query (action task with 2 query entities)."""
    print("=== Test 9: Multiple Query Entities ===")

    config = create_sample_action_config()
    model = create_positional_entity_causal_model(config)

    # Create specific input with 2-entity query
    input_sample = {
        "entity_g0_e0": "Pete",
        "entity_g0_e1": "jam",
        "entity_g0_e2": "cup",
        "entity_g1_e0": "Ann",
        "entity_g1_e1": "water",
        "entity_g1_e2": "box",
        "entity_g2_e0": "Bob",
        "entity_g2_e1": "book",
        "entity_g2_e2": "table",
        "query_e0": "Pete",
        "query_e1": "jam",
        "query_e2": "cup",
        "query_group": 0,
        "query_indices": (0, 1),  # Query both person and object
        "answer_index": 2,  # Answer is location
        "active_groups": 3,
        "entities_per_group": 3,
        "statement_template": config.statement_template,
    }

    trace = model.new_trace(input_sample)

    print(
        f"Query: person={input_sample['query_e0']}, object={input_sample['query_e1']}"
    )
    print(f"positional_query_e0 (person): {trace['positional_query_e0']}")
    print(f"positional_query_e1 (object): {trace['positional_query_e1']}")
    print(f"positional_query_e2 (location): {trace['positional_query_e2']}")
    print(f"Intersection (positional_answer): {trace['positional_answer']}")
    print(f"Answer: {trace['raw_output']}")
    print()

    # Both should point to group 0
    assert trace["positional_query_e0"] == (0,), "Pete is at group 0"
    assert trace["positional_query_e1"] == (0,), "jam is at group 0"
    assert trace["positional_query_e2"] == (), "location not queried"
    assert trace["positional_answer"] == 0, "Intersection should be group 0"
    assert trace["raw_output"] == "cup", "Answer should be cup"

    print("✓ Test 9 passed\n")


def test_model_properties():
    """Test model has correct properties."""
    print("=== Test 10: Model Properties ===")

    config = create_sample_love_config()
    model = create_positional_entity_causal_model(config)

    print(f"Model ID: {model.id}")
    print(f"Number of variables: {len(model.variables)}")
    print(f"Input variables: {model.inputs}")
    print()

    # Verify required variables exist
    assert "raw_input" in model.variables, "Should have raw_input"
    assert "raw_output" in model.variables, "Should have raw_output"
    assert "positional_answer" in model.variables, "Should have positional_answer"

    # Verify input variables
    assert "entity_g0_e0" in model.inputs, "Entity vars should be inputs"
    assert "query_e0" in model.inputs, "Query entity vars should be inputs"
    assert "query_group" in model.inputs, "Control vars should be inputs"

    print("✓ Test 10 passed\n")


def main():
    """Run all tests."""
    print("Testing Positional Entity Causal Model")
    print("=" * 70)
    print()

    try:
        test_basic_functionality()
        test_positional_entity_variables()
        test_positional_query_variables()
        test_positional_answer_variable()
        test_intervention_on_positional_entity()
        test_intervention_on_positional_query()
        test_intervention_on_positional_answer()
        test_with_action_config()
        test_multiple_query_entities()
        test_model_properties()

        print("\n" + "=" * 70)
        print("All tests passed!")
        print("=" * 70)
        print("\nPositional Entity Causal Model successfully implemented!")
        print("\nKey features verified:")
        print("- Produces correct answers")
        print("- Positional entity variables compute correct positions")
        print("- Positional query variables find matching groups")
        print("- Positional answer correctly computes intersection")
        print("- Interventions work on all variable types")
        print("- Handles multiple query entities correctly")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
