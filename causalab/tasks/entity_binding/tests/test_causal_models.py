#!/usr/bin/env -S uv run python
"""Test Script: Entity Binding Causal Model.

Tests the positional entity causal model which searches for query entities
and retrieves answers based on found positions.
"""

from causalab.tasks.entity_binding.config import (
    create_sample_love_config,
    create_sample_action_config,
)
from causalab.tasks.entity_binding.causal_models import (
    create_positional_entity_causal_model,
    sample_valid_entity_binding_input,
)


def test_model_basic():
    """Test basic functionality of positional entity model."""
    print("=== Test 1: Positional Entity Model Basic ===")

    config = create_sample_love_config()
    model = create_positional_entity_causal_model(config)

    # Sample input
    input_sample = sample_valid_entity_binding_input(config)
    output = model.new_trace(input_sample)

    print(f"Prompt: {output['raw_input']}")
    print(f"Answer: {output['raw_output']}")
    print(f"Query group (input): {input_sample['query_group']}")
    print()

    # Verify output has required fields
    assert "raw_input" in output, "Should have raw_input"
    assert "raw_output" in output, "Should have raw_output"
    assert output["raw_output"] != "UNKNOWN", "Should have valid answer"

    print("✓ Test 1 passed\n")


def test_model_with_multiple_samples():
    """Test model produces consistent results across samples."""
    print("=== Test 2: Multiple Samples ===")

    config = create_sample_love_config()
    model = create_positional_entity_causal_model(config)

    print("Testing 5 random samples:\n")

    for i in range(5):
        input_sample = sample_valid_entity_binding_input(config)
        output = model.new_trace(input_sample)

        print(f"Sample {i + 1}:")
        print(f"  Prompt: {output['raw_input']}")
        print(f"  Answer: {output['raw_output']}")

        # Verify basic structure
        assert "raw_input" in output
        assert "raw_output" in output
        assert output["raw_output"] != "UNKNOWN"

        print("  ✓ Valid output")
        print()

    print("✓ Test 2 passed\n")


def test_positional_search_mechanism():
    """Test the positional search finds entities correctly."""
    print("=== Test 3: Positional Search Mechanism ===")

    config = create_sample_love_config()
    model = create_positional_entity_causal_model(config)

    # Create specific example
    input_sample = {
        "entity_g0_e0": "Pete",
        "entity_g0_e1": "jam",
        "entity_g1_e0": "Ann",
        "entity_g1_e1": "pie",
        "entity_g2_e0": "Bob",
        "entity_g2_e1": "cake",
        "query_group": 1,
        "query_indices": (0,),
        "answer_index": 1,
        "active_groups": 3,
        "entities_per_group": 2,
        "query_e0": "Ann",
        "query_e1": "pie",
        "statement_template": config.statement_template,
    }

    output = model.new_trace(input_sample)

    print("Entities: g0=(Pete, jam), g1=(Ann, pie), g2=(Bob, cake)")
    print(
        f"Query: group {input_sample['query_group']}, entity {input_sample['query_indices'][0]}"
    )
    print()
    print(f"Raw input: {output['raw_input']}")
    print(f"Raw output: {output['raw_output']}")
    print()

    # Verify answer is correct
    assert output["raw_output"] == "pie", "Should answer 'pie' for Ann's love"

    print("✓ Test 3 passed\n")


def test_with_action_config():
    """Test model with action configuration."""
    print("=== Test 4: Action Configuration ===")

    config = create_sample_action_config()
    model = create_positional_entity_causal_model(config)

    input_sample = sample_valid_entity_binding_input(config)
    output = model.new_trace(input_sample)

    print("Action task:")
    print(f"  Prompt: {output['raw_input']}")
    print(f"  Answer: {output['raw_output']}")
    print()

    assert "raw_input" in output
    assert "raw_output" in output
    assert output["raw_output"] != "UNKNOWN"

    print("✓ Test 4 passed\n")


def test_model_structure():
    """Test the model has expected variables and structure."""
    print("=== Test 5: Model Structure ===")

    config = create_sample_love_config()
    model = create_positional_entity_causal_model(config)

    # Check model has expected variables
    input_sample = sample_valid_entity_binding_input(config)
    output = model.new_trace(input_sample)

    # Check for positional entity variables
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"positional_entity_g{g}_e{e}"
            assert key in output, f"Should have {key}"

    # Check for positional query variables
    for e in range(config.max_entities_per_group):
        key = f"positional_query_e{e}"
        assert key in output, f"Should have {key}"

    # Check for positional answer
    assert "positional_answer" in output, "Should have positional_answer"

    print(f"  positional_answer: {output['positional_answer']}")
    print(f"  raw_output: {output['raw_output']}")

    print("\n✓ Test 5 passed\n")


def main():
    """Run all tests."""
    print("Testing Entity Binding Causal Model")
    print("=" * 70)
    print()

    try:
        test_model_basic()
        test_model_with_multiple_samples()
        test_positional_search_mechanism()
        test_with_action_config()
        test_model_structure()

        print("\n" + "=" * 70)
        print("🎉 All causal model tests passed!")
        print("=" * 70)
        print("\nPositional entity model:")
        print("✓ Searches for query entity in all groups")
        print("✓ Uses positional intersection to find answer group")
        print("✓ Retrieves answer from found position")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
