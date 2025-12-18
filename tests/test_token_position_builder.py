"""
Tests for the declarative token position builder system.
"""

import pytest
from unittest.mock import Mock
from causalab.neural.token_position_builder import build_token_position_factories


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    pipeline = Mock()
    pipeline.tokenizer = Mock()
    pipeline.tokenizer.pad_token_id = 0

    # Mock tokenization: "The sum of {x} and {y} is "
    # Tokens: ["The", " sum", " of", " ", x_value, " and", " ", y_value, " is", " "]
    # Positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    def mock_load(
        input_dict,
        add_special_tokens=False,
        return_offsets_mapping=False,
        no_padding=False,
    ):
        raw_input = input_dict["raw_input"]
        # Simplified tokenization for testing
        # Just split on spaces and assign sequential IDs
        tokens = raw_input.split()
        token_ids = list(range(len(tokens)))
        result = {"input_ids": [token_ids]}

        if return_offsets_mapping:
            # Create fake character offsets
            offsets = []
            pos = 0
            for token in tokens:
                start = pos
                end = pos + len(token)
                offsets.append((start, end))
                pos = end + 1  # +1 for space
            result["offset_mapping"] = [offsets]

        return result

    pipeline.load = mock_load

    def mock_decode(token_ids):
        # Mock decoding - just return placeholder
        if isinstance(token_ids, list):
            return f"<token_{token_ids[0]}>"
        return f"<token_{token_ids}>"

    pipeline.tokenizer.decode = mock_decode

    return pipeline


def test_fixed_position_last_token(mock_pipeline):
    """Test fixed position: last token."""
    template = "The sum of {x} and {y} is "
    specs = {"last": {"type": "index", "position": -1}}

    factories = build_token_position_factories(specs, template)
    token_pos = factories["last"](mock_pipeline)

    # Test with sample input
    input_sample = {"raw_input": "The sum of 5 and 7 is ", "x": 5, "y": 7}

    result = token_pos.index(input_sample)
    # Should return the last token position
    assert isinstance(result, list)
    assert len(result) == 1


def test_fixed_position_first_token(mock_pipeline):
    """Test fixed position: first token."""
    template = "The sum of {x} and {y} is "
    specs = {"first": {"type": "index", "position": 0}}

    factories = build_token_position_factories(specs, template)
    token_pos = factories["first"](mock_pipeline)

    input_sample = {"raw_input": "The sum of 5 and 7 is ", "x": 5, "y": 7}

    result = token_pos.index(input_sample)
    assert result == [0]


def test_variable_position(mock_pipeline):
    """Test variable position: finding where a variable appears."""
    template = "The sum of {x} and {y} is "
    specs = {"x": {"type": "variable", "name": "x"}}

    factories = build_token_position_factories(specs, template)
    token_pos = factories["x"](mock_pipeline)

    input_sample = {"raw_input": "The sum of 5 and 7 is ", "x": 5, "y": 7}

    result = token_pos.index(input_sample)
    assert isinstance(result, list)
    assert len(result) > 0


def test_variable_not_in_template():
    """Test that referencing a non-existent variable raises error."""
    template = "The sum of {x} and {y} is "
    specs = {
        "z": {"type": "variable", "name": "z"}  # z not in template!
    }

    with pytest.raises(ValueError, match="not found in template"):
        build_token_position_factories(specs, template)


def test_scoped_index_last_token_of_variable(mock_pipeline):
    """Test scoped indexing: last token within a variable."""
    template = "The answer is {answer}"
    specs = {
        "last_of_answer": {
            "type": "index",
            "position": -1,
            "scope": {"variable": "answer"},
        }
    }

    factories = build_token_position_factories(specs, template)
    token_pos = factories["last_of_answer"](mock_pipeline)

    # Use a multi-word answer to test multi-token variables
    # With space-splitting: ["The", "answer", "is", "New", "York"]
    input_sample = {"raw_input": "The answer is New York", "answer": "New York"}

    result = token_pos.index(input_sample)
    # Should return the last token of the variable's tokenization
    # "New York" splits into tokens [3, 4], so last token is [4]
    assert result == [4]


def test_relative_position_after_variable(mock_pipeline):
    """Test relative positioning: token after a variable."""
    template = "The sum of {x} and {y} is "
    specs = {
        "after_x": {"type": "index", "position": +1, "relative_to": {"variable": "x"}}
    }

    factories = build_token_position_factories(specs, template)
    token_pos = factories["after_x"](mock_pipeline)

    # With space-splitting: ["The", "sum", "of", "5", "and", "7", "is", ""]
    input_sample = {"raw_input": "The sum of 5 and 7 is ", "x": 5, "y": 7}

    result = token_pos.index(input_sample)
    # x=5 is at position [3], so +1 relative gives [4]
    assert result == [4]


def test_multiple_specs():
    """Test building multiple token positions at once."""
    template = "The sum of {x} and {y} is "
    specs = {
        "last": {"type": "index", "position": -1},
        "first": {"type": "index", "position": 0},
        "x": {"type": "variable", "name": "x"},
        "y": {"type": "variable", "name": "y"},
    }

    factories = build_token_position_factories(specs, template)

    assert len(factories) == 4
    assert "last" in factories
    assert "first" in factories
    assert "x" in factories
    assert "y" in factories


def test_dynamic_position_basic(mock_pipeline):
    """Test dynamic position spec using a function."""
    template = "The answer is {option_a} or {option_b}"

    # Dynamic spec that chooses between option_a and option_b
    specs = {
        "correct_answer": lambda setting: {
            "type": "variable",
            "name": "option_a" if setting["correct"] == "a" else "option_b",
        }
    }

    factories = build_token_position_factories(specs, template)
    token_pos = factories["correct_answer"](mock_pipeline)

    # Test when correct = "a"
    # With space-splitting: ["The", "answer", "is", "yes", "or", "no"]
    input_sample_a = {
        "raw_input": "The answer is yes or no",
        "option_a": "yes",
        "option_b": "no",
        "correct": "a",
    }
    result_a = token_pos.index(input_sample_a)
    assert result_a == [3]  # "yes" is at position [3]

    # Test when correct = "b"
    input_sample_b = {
        "raw_input": "The answer is yes or no",
        "option_a": "yes",
        "option_b": "no",
        "correct": "b",
    }
    result_b = token_pos.index(input_sample_b)
    assert result_b == [5]  # "no" is at position [5]


def test_dynamic_position_with_conditional_index(mock_pipeline):
    """Test dynamic position spec that returns different index types."""
    template = "The answer is {answer}"

    # Dynamic spec that returns first or last token based on a flag
    specs = {
        "target": lambda setting: {
            "type": "index",
            "position": -1 if setting["use_last"] else 0,
        }
    }

    factories = build_token_position_factories(specs, template)
    token_pos = factories["target"](mock_pipeline)

    # Test with use_last = True
    input_sample_last = {
        "raw_input": "The answer is correct",
        "answer": "correct",
        "use_last": True,
    }
    result_last = token_pos.index(input_sample_last)
    # Should return last token
    tokens = input_sample_last["raw_input"].split()
    assert result_last == [len(tokens) - 1]

    # Test with use_last = False
    input_sample_first = {
        "raw_input": "The answer is correct",
        "answer": "correct",
        "use_last": False,
    }
    result_first = token_pos.index(input_sample_first)
    # Should return first token
    assert result_first == [0]


def test_dynamic_position_mixed_with_static(mock_pipeline):
    """Test that dynamic and static specs can be mixed together."""
    template = "The {x} equals {y}"

    specs = {
        "last": {"type": "index", "position": -1},  # Static
        "x": {"type": "variable", "name": "x"},  # Static
        "target": lambda setting: {  # Dynamic
            "type": "variable",
            "name": "x" if setting["target_x"] else "y",
        },
    }

    factories = build_token_position_factories(specs, template)

    # Verify all three factories exist
    assert len(factories) == 3
    assert "last" in factories
    assert "x" in factories
    assert "target" in factories

    # Test the dynamic one
    token_pos = factories["target"](mock_pipeline)

    # With space-splitting: ["The", "5", "equals", "10"]
    input_sample = {"raw_input": "The 5 equals 10", "x": 5, "y": 10, "target_x": True}
    result = token_pos.index(input_sample)
    assert result == [1]  # x=5 is at position [1]
