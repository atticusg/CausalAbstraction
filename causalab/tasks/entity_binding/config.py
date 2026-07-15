"""
Core data structures for Entity Binding tasks.

This module provides the fundamental data structures needed to represent
entity binding tasks with arbitrary numbers of entity groups and entities per group.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _build_conjoined_template(
    template: str,
    num_repetitions: int,
    delimiters: List[str],
    group_prefix: str = "g",
    capitalize_first: bool = True,
) -> str:
    """
    Build a conjoined template with unique variable names for each repetition.

    Takes "{e0} loves {e1}" and creates:
    "{g0_e0} loves {g0_e1}, {g1_e0} loves {g1_e1}."

    Args:
        template: Template string with {variable} placeholders
        num_repetitions: Number of times to repeat the template
        delimiters: List of delimiters of length num_repetitions (last one after final statement)
        group_prefix: Prefix for group index (default "g")
        capitalize_first: Whether to capitalize first letter of result

    Returns:
        A new template string with unique variable names per repetition.
    """
    if num_repetitions == 0:
        return ""

    if len(delimiters) != num_repetitions:
        raise ValueError(
            f"Expected {num_repetitions} delimiters, got {len(delimiters)}"
        )

    # Find all {variable} placeholders in the template
    var_pattern = re.compile(r"\{(\w+)\}")

    result_parts = []
    for rep_idx in range(num_repetitions):
        # Replace variable names with group-qualified names
        def replace_var(m, rep=rep_idx):
            var_name = m.group(1)
            return "{" + f"{group_prefix}{rep}_{var_name}" + "}"

        qualified = var_pattern.sub(replace_var, template)
        result_parts.append(qualified)
        result_parts.append(delimiters[rep_idx])

    result = "".join(result_parts)

    if capitalize_first and result:
        result = result[0].upper() + result[1:]

    return result


def _expand_delimiters(delimiters: List[str], num_statements: int) -> List[str]:
    """
    Expand FILL-style delimiter spec to a list of the correct length.

    Args:
        delimiters: Delimiter spec with "FILL" marker, e.g. [", ", "FILL", ", and ", "."]
        num_statements: Number of statements to join

    Returns:
        List of delimiters of length num_statements
    """
    fill_index = delimiters.index("FILL")
    filler = delimiters[fill_index - 1]
    result = delimiters[: fill_index - 1] + delimiters[fill_index + 1 :]

    while len(result) < num_statements:
        result.insert(fill_index - 1, filler)

    if len(result) > num_statements:
        result = result[-num_statements:]

    # For 2 statements, strip the comma from ", and " to get " and "
    if num_statements == 2 and len(result) > 0 and ", and" in result[0]:
        result[0] = result[0].lstrip(",")

    return result


@dataclass
class EntityBindingTaskConfig:
    """
    Configuration for an entity binding task.

    Defines the structure including:
    - Maximum dimensions (groups and entities per group)
    - Entity pools for each role position
    - Templates for generating text
    - Prompt formatting
    """

    max_groups: int
    max_entities_per_group: int
    entity_roles: Dict[int, str]
    entity_pools: Dict[int, List[str]]
    statement_template: str
    question_templates: Dict[Tuple[Tuple[int, ...], int], str]
    delimiters: List[str]
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    statement_question_separator: str = " "
    fixed_query_indices: Optional[Tuple[int, ...]] = None
    fixed_answer_index: Optional[int] = None

    def build_mega_template(
        self,
        active_groups: int,
        query_indices: Tuple[int, ...],
        answer_index: int,
    ) -> str:
        """
        Build the full template string combining statement and question.

        Variable names in the result:
        - Statement entities: g0_e0, g0_e1, g1_e0, g1_e1, ...
        - Question entities: entity role names (person, food, etc.)
        """
        delimiters = _expand_delimiters(self.delimiters, active_groups)
        statement_template_str = _build_conjoined_template(
            self.statement_template, active_groups, delimiters
        )
        question_template_str = self.question_templates.get(
            (query_indices, answer_index), ""
        )
        body = f"{statement_template_str}{self.statement_question_separator}{question_template_str}"
        return f"{self.prompt_prefix}{body}{self.prompt_suffix}"

    def fill_template(self, mega_template: str, values: Dict[str, str]) -> str:
        """Fill in a mega template with entity values."""
        return mega_template.format(**values)


def create_sample_love_config() -> EntityBindingTaskConfig:
    """
    Create a sample configuration for the 'love' task (Pete loves jam, Ann loves pie).

    This is a simple 2-entity-per-group task with people and foods.
    """
    return EntityBindingTaskConfig(
        max_groups=2,
        max_entities_per_group=2,
        entity_roles={0: "person", 1: "food"},
        entity_pools={
            0: ["Pete", "Ann", "Tim", "Bob", "Sue", "Kate"],
            1: ["jam", "pie", "cake", "bread", "soup", "tea"],
        },
        statement_template="{e0} loves {e1}",
        delimiters=[", ", "FILL", ", and ", "."],
        question_templates={
            ((0,), 1): "What does {person} love?",
            ((1,), 0): "Who loves {food}?",
        },
        prompt_prefix="We will ask a question about the following sentences.\n\n",
        prompt_suffix="\nAnswer: ",
    )


# Token and generation budget constants
MAX_TASK_TOKENS = 128
MAX_NEW_TOKENS = 4
