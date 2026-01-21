"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Tests for template processing with prompt wrapper functionality.
"""

import pytest
from causalab.tasks.entity_binding.config import (
    create_sample_love_config,
    create_sample_action_config,
    EntityGroup,
    BindingMatrix
)
from causalab.tasks.entity_binding.templates import TemplateProcessor


def test_statement_template_filling():
    """Test basic statement template filling."""
    config = create_sample_love_config()
    processor = TemplateProcessor(config)

    group1 = EntityGroup(["Pete", "jam"], 0)
    group2 = EntityGroup(["Ann", "pie"], 1)
    matrix = BindingMatrix([group1, group2], config.max_groups, config.max_entities_per_group)

    statement = processor.fill_statement_template(matrix)

    # Should produce: "Pete loves jam, and Ann loves pie."
    assert "Pete loves jam" in statement
    assert "Ann loves pie" in statement
    assert statement.endswith(".")


def test_question_template_selection():
    """Test selecting appropriate question templates."""
    config = create_sample_love_config()
    processor = TemplateProcessor(config)

    # Query person, answer food
    template1 = processor.select_question_template((0,), 1)
    assert "What does" in template1

    # Query food, answer person
    template2 = processor.select_question_template((1,), 0)
    assert "Who loves" in template2


def test_question_template_filling():
    """Test filling question templates with entities."""
    config = create_sample_love_config()
    processor = TemplateProcessor(config)

    group = EntityGroup(["Pete", "jam"], 0)
    matrix = BindingMatrix([group], config.max_groups, config.max_entities_per_group)

    # Query person -> food
    template = processor.select_question_template((0,), 1)
    question = processor.fill_question_template(template, 0, (0,), matrix)

    assert "Pete" in question
    assert "love" in question


def test_full_prompt_generation_without_wrapper():
    """Test generating complete prompts without wrapper."""
    config = create_sample_love_config()
    processor = TemplateProcessor(config)

    group1 = EntityGroup(["Pete", "jam"], 0)
    group2 = EntityGroup(["Ann", "pie"], 1)
    matrix = BindingMatrix([group1, group2], config.max_groups, config.max_entities_per_group)

    prompt, answer = processor.generate_full_prompt(matrix, 0, (0,), 1)

    # Should have statement and question
    assert "Pete loves jam" in prompt
    assert "Ann loves pie" in prompt
    assert "What does Pete love?" in prompt
    assert answer == "jam"


def test_full_prompt_generation_with_wrapper():
    """Test generating prompts with instruction wrapper."""
    config = create_sample_love_config()
    config.prompt_prefix = "We will ask a question about the following sentences.\n\n"
    config.statement_question_separator = "\n\n"
    config.prompt_suffix = "\nAnswer:"

    processor = TemplateProcessor(config)

    group1 = EntityGroup(["Pete", "jam"], 0)
    group2 = EntityGroup(["Ann", "pie"], 1)
    matrix = BindingMatrix([group1, group2], config.max_groups, config.max_entities_per_group)

    prompt, answer = processor.generate_full_prompt(matrix, 0, (0,), 1)

    # Should have prefix
    assert prompt.startswith("We will ask a question")

    # Should have suffix
    assert prompt.endswith("Answer:")

    # Should have separator between statement and question
    assert "\n\n" in prompt

    # Should still have correct content
    assert "Pete loves jam" in prompt
    assert "What does Pete love?" in prompt
    assert answer == "jam"


def test_action_template_processing():
    """Test template processing with 3-entity action task."""
    config = create_sample_action_config()
    processor = TemplateProcessor(config)

    group = EntityGroup(["Pete", "jam", "cup"], 0)
    matrix = BindingMatrix([group], config.max_groups, config.max_entities_per_group)

    statement = processor.fill_statement_template(matrix)

    assert "Pete put jam in the cup" in statement


def test_action_question_templates():
    """Test various action question templates."""
    config = create_sample_action_config()
    processor = TemplateProcessor(config)

    group = EntityGroup(["Pete", "jam", "cup"], 0)
    matrix = BindingMatrix([group], config.max_groups, config.max_entities_per_group)

    # Query person -> object
    template1 = processor.select_question_template((0,), 1)
    question1 = processor.fill_question_template(template1, 0, (0,), matrix)
    assert "Pete" in question1
    assert "put somewhere" in question1

    # Query object -> location
    template2 = processor.select_question_template((1,), 2)
    question2 = processor.fill_question_template(template2, 0, (1,), matrix)
    assert "jam" in question2
    assert "was" in question2 and "put" in question2


def test_wrapper_with_action_task():
    """Test prompt wrapper with action task."""
    config = create_sample_action_config()
    config.prompt_prefix = "Instructions:\n\n"
    config.statement_question_separator = "\n\n"
    config.prompt_suffix = "\nAnswer:"

    processor = TemplateProcessor(config)

    group1 = EntityGroup(["Pete", "jam", "cup"], 0)
    group2 = EntityGroup(["Ann", "book", "box"], 1)
    matrix = BindingMatrix([group1, group2], config.max_groups, config.max_entities_per_group)

    prompt, answer = processor.generate_full_prompt(matrix, 0, (0,), 1)

    # Should have wrapper
    assert prompt.startswith("Instructions:")
    assert prompt.endswith("Answer:")

    # Should have content
    assert "Pete put jam in the cup" in prompt
    assert "Ann put book in the box" in prompt
    assert answer == "jam"


def test_empty_wrapper_produces_simple_format():
    """Test that empty wrapper produces original simple format."""
    config = create_sample_love_config()
    # Defaults are empty
    processor = TemplateProcessor(config)

    group = EntityGroup(["Pete", "jam"], 0)
    matrix = BindingMatrix([group], config.max_groups, config.max_entities_per_group)

    prompt, answer = processor.generate_full_prompt(matrix, 0, (0,), 1)

    # Should be simple format: "Statement Question"
    assert not prompt.startswith("We will ask")
    assert not prompt.endswith("Answer:")
    assert " " in prompt  # Default separator


def test_multiple_groups_statement_conjunction():
    """Test that multiple groups are properly conjoined with delimiters."""
    config = create_sample_love_config()
    processor = TemplateProcessor(config)

    group1 = EntityGroup(["Pete", "jam"], 0)
    group2 = EntityGroup(["Ann", "pie"], 1)
    group3 = EntityGroup(["Bob", "cake"], 2)
    matrix = BindingMatrix([group1, group2, group3], config.max_groups, config.max_entities_per_group)

    statement = processor.fill_statement_template(matrix)

    # Should have all three statements with proper conjunction
    assert "Pete loves jam" in statement
    assert "Ann loves pie" in statement
    assert "Bob loves cake" in statement
    # Should use "and" for final conjunction
    assert ", and " in statement


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
