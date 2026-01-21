"""
Test Script: MCQA Causal Models

This script tests the MCQA causal model structure, mechanisms, and forward execution.
"""

import pytest

from causalab.tasks.MCQA.causal_models import (
    positional_causal_model,
    OBJECTS,
    COLORS,
    ALPHABET,
    NUM_CHOICES,
    TEMPLATES,
)
from causalab.tasks.MCQA.counterfactuals import sample_answerable_question


def test_causal_model_structure():
    """Test that the causal model has the correct structure."""
    print("=== Test 1: Causal Model Structure ===")

    model = positional_causal_model

    # Check variables
    expected_vars = [
        "template",
        "object",
        "color",
        "raw_input",
        "symbol0",
        "symbol1",
        "choice0",
        "choice1",
        "answer_position",
        "answer",
        "raw_output",
    ]

    for var in expected_vars:
        assert var in model.variables, f"Variable '{var}' should be in model"
        print(f"✓ Variable '{var}' present")

    # Check parent relationships
    assert model.parents["raw_input"] == [
        "template",
        "object",
        "color",
        "symbol0",
        "symbol1",
        "choice0",
        "choice1",
    ]
    assert model.parents["answer_position"] == ["color", "choice0", "choice1"]
    assert model.parents["answer"] == ["answer_position", "symbol0", "symbol1"]

    print("✓ Parent relationships correct")
    print("✓ Test 1 passed\n")


def test_sample_answerable_question():
    """Test sampling valid answerable questions."""
    print("=== Test 2: Sample Answerable Question ===")

    # Sample 5 valid inputs using sample_answerable_question
    for i in range(5):
        input_sample = sample_answerable_question()

        # Check all input variables are present
        assert "template" in input_sample
        assert "object" in input_sample
        assert "color" in input_sample
        assert "symbol0" in input_sample
        assert "symbol1" in input_sample
        assert "choice0" in input_sample
        assert "choice1" in input_sample

        # Verify the color is in one of the choices (answerable question)
        assert input_sample["color"] in [
            input_sample["choice0"],
            input_sample["choice1"],
        ], "Color should be in choices for answerable question"

        print(
            f"Sample {i + 1}: {input_sample['object']} is {input_sample['color']} with choices {input_sample['choice0']}, {input_sample['choice1']}"
        )

    print("✓ Test 2 passed\n")


def test_fill_template_mechanism():
    """Test the template filling mechanism via new_trace."""
    print("=== Test 3: Fill Template Mechanism ===")

    model = positional_causal_model

    input_sample = {
        "template": TEMPLATES[0],
        "object": "banana",
        "color": "yellow",
        "symbol0": "A",
        "symbol1": "B",
        "choice0": "blue",
        "choice1": "yellow",
    }

    output = model.new_trace(input_sample)
    result = output["raw_input"]

    print(f"Template: {TEMPLATES[0]}")
    print(f"\nFilled result:\n{result}")

    # Check substitutions
    assert "banana" in result, "Object name should be in result"
    assert "yellow" in result, "Color should be in result"
    assert "A" in result, "Symbol0 should be in result"
    assert "B" in result, "Symbol1 should be in result"
    assert "blue" in result, "Choice0 should be in result"

    # Check no placeholders remain
    assert "{object}" not in result, "No placeholder should remain"
    assert "{color}" not in result, "No placeholder should remain"
    assert "{symbol0}" not in result, "No placeholder should remain"
    assert "{symbol1}" not in result, "No placeholder should remain"
    assert "{choice0}" not in result, "No placeholder should remain"
    assert "{choice1}" not in result, "No placeholder should remain"

    print("✓ All substitutions correct")
    print("✓ Test 3 passed\n")


def test_answer_position_mechanism():
    """Test the answer position mechanism via new_trace."""
    print("=== Test 4: Answer Position Mechanism ===")

    model = positional_causal_model

    # Test case 1: Answer in position 0
    output1 = model.new_trace(
        {
            "template": TEMPLATES[0],
            "object": "ball",
            "color": "yellow",
            "symbol0": "A",
            "symbol1": "B",
            "choice0": "yellow",
            "choice1": "blue",
        }
    )
    print(
        f"Test 1: Color 'yellow' in choices ['yellow', 'blue'] -> position {output1['answer_position']}"
    )
    assert output1["answer_position"] == 0, "Should find yellow at position 0"

    # Test case 2: Answer in position 1
    output2 = model.new_trace(
        {
            "template": TEMPLATES[0],
            "object": "ball",
            "color": "green",
            "symbol0": "A",
            "symbol1": "B",
            "choice0": "blue",
            "choice1": "green",
        }
    )
    print(
        f"Test 2: Color 'green' in choices ['blue', 'green'] -> position {output2['answer_position']}"
    )
    assert output2["answer_position"] == 1, "Should find green at position 1"

    # Test case 3: Answer not in choices (should raise ValueError)
    input3 = {
        "template": TEMPLATES[0],
        "object": "ball",
        "color": "yellow",
        "symbol0": "A",
        "symbol1": "B",
        "choice0": "blue",
        "choice1": "red",
    }
    print(
        "Test 3: Color 'yellow' in choices ['blue', 'red'] -> should raise ValueError"
    )
    with pytest.raises(ValueError, match="No correct answer position found"):
        model.new_trace(input3)

    # Test case 4: Duplicate colors (edge case - returns first match)
    output4 = model.new_trace(
        {
            "template": TEMPLATES[0],
            "object": "ball",
            "color": "white",
            "symbol0": "A",
            "symbol1": "B",
            "choice0": "white",
            "choice1": "white",
        }
    )
    print(
        f"Test 4: Color 'white' in choices ['white', 'white'] -> position {output4['answer_position']}"
    )
    assert output4["answer_position"] == 0, "Should return first match when duplicate"

    print("✓ Test 4 passed\n")


def test_answer_retrieval_mechanism():
    """Test the answer retrieval mechanism via new_trace."""
    print("=== Test 5: Answer Retrieval Mechanism ===")

    model = positional_causal_model

    # Test position 0 (answer at first choice)
    output1 = model.new_trace(
        {
            "template": TEMPLATES[0],
            "object": "ball",
            "color": "yellow",
            "symbol0": "A",
            "symbol1": "B",
            "choice0": "yellow",
            "choice1": "blue",
        }
    )
    print(f"Position 0 with symbols ['A', 'B'] -> answer '{output1['answer']}'")
    assert output1["answer"] == "A", "Position 0 should return first symbol"

    # Test position 1 (answer at second choice)
    output2 = model.new_trace(
        {
            "template": TEMPLATES[0],
            "object": "ball",
            "color": "blue",
            "symbol0": "A",
            "symbol1": "B",
            "choice0": "yellow",
            "choice1": "blue",
        }
    )
    print(f"Position 1 with symbols ['A', 'B'] -> answer '{output2['answer']}'")
    assert output2["answer"] == "B", "Position 1 should return second symbol"

    # Test color not in choices (should raise ValueError)
    input3 = {
        "template": TEMPLATES[0],
        "object": "ball",
        "color": "green",
        "symbol0": "A",
        "symbol1": "B",
        "choice0": "yellow",
        "choice1": "blue",
    }
    print("Color not in choices -> should raise ValueError")
    with pytest.raises(ValueError, match="No correct answer position found"):
        model.new_trace(input3)

    print("✓ Test 5 passed\n")


def test_forward_execution():
    """Test running the causal model forward."""
    print("=== Test 6: Forward Execution ===")

    model = positional_causal_model

    # Create specific input
    input_sample = {
        "template": TEMPLATES[0],
        "object": "banana",
        "color": "yellow",
        "symbol0": "A",
        "symbol1": "B",
        "choice0": "blue",
        "choice1": "yellow",
    }

    output = model.new_trace(input_sample)

    print(f"Prompt:\n{output['raw_input']}\n")
    print(f"Answer position: {output['answer_position']}")
    print(f"Answer: {output['answer']}")
    print(f"Raw output: {output['raw_output']}")

    # Verify output
    assert output["answer_position"] == 1, "Yellow should be at position 1"
    assert output["answer"] == "B", "Answer should be B"
    assert output["raw_output"] == " B", "Raw output should have leading space + answer"
    assert "banana" in output["raw_input"]
    assert "yellow" in output["raw_input"]

    print("✓ Forward execution correct")
    print("✓ Test 6 passed\n")


def test_multiple_forward_executions():
    """Test multiple forward executions for consistency."""
    print("=== Test 7: Multiple Forward Executions ===")

    print("Testing 10 random samples:\n")

    for i in range(10):
        # sample_answerable_question() returns a fully computed CausalTrace
        trace = sample_answerable_question()

        # Verify consistency
        assert "raw_input" in trace
        assert "raw_output" in trace
        assert "answer_position" in trace
        assert "answer" in trace

        # Verify answer matches answer_position
        expected_symbol = trace[f"symbol{trace['answer_position']}"]
        assert trace["answer"] == expected_symbol, (
            "Answer should match symbol at answer_position"
        )

        print(f"Sample {i + 1}: {trace['object']} is {trace['color']}")
        print(f"  Choices: {trace['choice0']}, {trace['choice1']}")
        print(f"  Answer at position {trace['answer_position']}: {trace['answer']}")

    print("\n✓ All executions consistent")
    print("✓ Test 7 passed\n")


def test_edge_case_duplicate_colors():
    """Test edge case where object color appears in multiple choice positions."""
    print("=== Test 8: Edge Case - Duplicate Colors ===")

    model = positional_causal_model

    input_sample = {
        "template": TEMPLATES[0],
        "object": "snow",
        "color": "white",
        "symbol0": "X",
        "symbol1": "Y",
        "choice0": "white",
        "choice1": "white",
    }

    output = model.new_trace(input_sample)

    print("Object: snow")
    print("Color: white")
    print("Choices: white, white")
    print(f"Answer position: {output['answer_position']}")
    print(f"Answer: {output['answer']}")

    # Should return first match
    assert output["answer_position"] == 0, "Should return first match"
    assert output["answer"] == "X", "Should return symbol at position 0"

    print("✓ Handles duplicate colors correctly")
    print("✓ Test 8 passed\n")


def test_constants():
    """Test that constants are correctly defined."""
    print("=== Test 9: Constants ===")

    print(f"Number of objects: {len(OBJECTS)}")
    print(f"Number of colors: {len(COLORS)}")
    print(f"Number of choices: {NUM_CHOICES}")
    print(f"Alphabet length: {len(ALPHABET)}")
    print(f"Number of templates: {len(TEMPLATES)}")

    # Verify we have enough colors for choices
    assert len(COLORS) >= NUM_CHOICES, "Should have at least NUM_CHOICES colors"

    # Verify we have enough symbols
    assert len(ALPHABET) >= NUM_CHOICES, "Should have at least NUM_CHOICES symbols"

    # Verify color extraction worked
    assert "yellow" in COLORS
    assert "green" in COLORS
    assert "blue" in COLORS

    print("✓ Constants properly defined")
    print("✓ Test 9 passed\n")


def main():
    """Run all tests."""
    print("Testing MCQA Causal Models")
    print("=" * 70)
    print()

    try:
        test_causal_model_structure()
        test_sample_answerable_question()
        test_fill_template_mechanism()
        test_answer_position_mechanism()
        test_answer_retrieval_mechanism()
        test_forward_execution()
        test_multiple_forward_executions()
        test_edge_case_duplicate_colors()
        test_constants()

        print("\n" + "=" * 70)
        print("🎉 All causal model tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
