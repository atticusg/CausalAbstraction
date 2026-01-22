"""Tests for core data structures: EntityBindingTaskConfig, EntityGroup, BindingMatrix."""

import pytest
from causalab.tasks.entity_binding.config import (
    EntityGroup,
    BindingMatrix,
    create_sample_love_config,
    create_sample_action_config,
)


def test_entity_group():
    """Test that EntityGroup correctly stores and retrieves entities."""
    # Create a group with person and food
    group = EntityGroup(["Pete", "jam"], group_index=0)

    assert group.group_index == 0
    assert group.get_entity(0) == "Pete"
    assert group.get_entity(1) == "jam"
    assert group.get_entity(2) is None  # Out of bounds
    assert len(group.entities) == 2


def test_binding_matrix():
    """Test that BindingMatrix correctly manages multiple entity groups."""
    # Create two entity groups
    group1 = EntityGroup(["Pete", "jam"], 0)
    group2 = EntityGroup(["Ann", "pie"], 1)

    # Create binding matrix
    matrix = BindingMatrix([group1, group2], max_groups=3, max_entities_per_group=2)

    # Test retrieval
    assert matrix.get_entity(0, 0) == "Pete"
    assert matrix.get_entity(0, 1) == "jam"
    assert matrix.get_entity(1, 0) == "Ann"
    assert matrix.get_entity(1, 1) == "pie"
    assert matrix.get_entity(2, 0) is None  # Inactive group

    # Test dimensions
    assert matrix.active_groups == 2
    assert matrix.get_entities_per_group() == 2


def test_love_config():
    """Test the sample love configuration."""
    config = create_sample_love_config()

    assert config.max_groups == 3
    assert config.max_entities_per_group == 2
    assert config.entity_roles == {0: "person", 1: "food"}
    assert len(config.entity_pools[0]) == 6  # 6 people
    assert len(config.entity_pools[1]) == 6  # 6 foods
    assert config.statement_template == "{e0} loves {e1}"

    # Test prompt wrapper fields
    assert hasattr(config, "prompt_prefix")
    assert hasattr(config, "prompt_suffix")
    assert hasattr(config, "statement_question_separator")


def test_action_config():
    """Test the sample action configuration."""
    config = create_sample_action_config()

    assert config.max_groups == 3
    assert config.max_entities_per_group == 3
    assert config.entity_roles == {0: "person", 1: "object", 2: "location"}
    assert len(config.entity_pools[0]) == 8  # 8 people
    assert len(config.entity_pools[1]) == 8  # 8 objects
    assert len(config.entity_pools[2]) == 8  # 8 locations
    assert config.statement_template == "{e0} put {e1} in the {e2}"


def test_prompt_wrapper_defaults():
    """Test that prompt wrapper fields have sensible defaults."""
    config = create_sample_love_config()

    # Sample configs come with instruction-tuned defaults
    assert (
        config.prompt_prefix
        == "We will ask a question about the following sentences.\n\n"
    )
    assert config.prompt_suffix == "\nAnswer:"
    assert config.statement_question_separator == " "  # Default from dataclass


def test_prompt_wrapper_customization():
    """Test that prompt wrapper can be customized."""
    config = create_sample_love_config()

    # Customize wrapper
    config.prompt_prefix = "Instructions:\n\n"
    config.prompt_suffix = "\nAnswer:"
    config.statement_question_separator = "\n\n"

    assert config.prompt_prefix == "Instructions:\n\n"
    assert config.prompt_suffix == "\nAnswer:"
    assert config.statement_question_separator == "\n\n"


def test_binding_matrix_with_none_entities():
    """Test that BindingMatrix handles None entities correctly."""
    # Create groups with some None entities
    group1 = EntityGroup(["Pete", "jam"], 0)
    group2 = EntityGroup([None, None], 1)

    matrix = BindingMatrix([group1, group2], max_groups=3, max_entities_per_group=2)

    assert matrix.get_entity(0, 0) == "Pete"
    assert matrix.get_entity(1, 0) is None
    assert matrix.get_entity(1, 1) is None
    assert matrix.active_groups == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
