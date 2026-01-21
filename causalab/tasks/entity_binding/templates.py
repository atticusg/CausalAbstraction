"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Template processing system for entity binding tasks.

This module handles the conversion from abstract entity bindings to concrete text.
It manages both statement templates (facts) and question templates.
"""

from typing import Tuple
import random
from .config import EntityBindingTaskConfig, BindingMatrix
from causalab.causal.causal_utils import statement_conjunction_function


class TemplateProcessor:
    """
    Processes templates to generate text from entity bindings.

    This class handles the core logic of filling templates with entities
    to create both factual statements and questions.
    """

    def __init__(self, config: EntityBindingTaskConfig):
        self.config = config

    def fill_statement_template(self, binding_matrix: BindingMatrix) -> str:
        """
        Fill the statement template with entities from the binding matrix.

        Args:
            binding_matrix: The entity bindings to use

        Returns:
            A filled statement like "Pete loves jam, Ann loves pie."
        """

        # Fill the template using string formatting
        try:
            filled_statements = []
            for e in range(binding_matrix.get_active_groups()):
                entity_dict = {
                    f"entity_e{i}": binding_matrix.get_entity(e, i)
                    for i in range(binding_matrix.get_entities_per_group())
                }
                filled_statements.append(
                    self.config.statement_template.format(**entity_dict)
                )
            return statement_conjunction_function(
                filled_statements, self.config.delimiters
            )
        except KeyError as e:
            raise ValueError(
                f"Template requires entity key {e} that's not in binding matrix"
            )

    def select_question_template(
        self, query_indices: Tuple[int, ...], answer_index: int
    ) -> str:
        """
        Select the appropriate question template based on query pattern.

        Args:
            query_indices: Tuple of entity indices that appear in the question
            answer_index: Index of the entity that should be the answer

        Returns:
            The question template string

        Raises:
            KeyError: If no template exists for this query pattern
        """
        key = (query_indices, answer_index)
        if key not in self.config.question_templates:
            raise KeyError(f"No question template for query pattern {key}")
        return self.config.question_templates[key]

    def fill_question_template(
        self,
        template: str,
        query_group: int,
        query_indices: Tuple[int, ...],
        binding_matrix: BindingMatrix,
    ) -> str:
        """
        Fill a question template with the queried entities.

        Args:
            template: The question template to fill
            query_group: Which group contains the query/answer entities
            query_indices: Which entity positions are mentioned in the question
            binding_matrix: The entity bindings

        Returns:
            A filled question like "What does Pete love?"
        """
        # Build dictionary of entities to substitute into template
        substitution_dict = {}

        # Add query entities
        for i, entity_idx in enumerate(query_indices):
            entity = binding_matrix.get_entity(query_group, entity_idx)
            if entity is None:
                raise ValueError(
                    f"Query entity at group {query_group}, index {entity_idx} is None"
                )
            substitution_dict["query_entity"] = (
                entity  # Simple case: single query entity
            )

        # For more complex templates, we might need additional context
        # Add all entities from the query group for flexible template access
        for e in range(binding_matrix.get_entities_per_group()):
            entity = binding_matrix.get_entity(query_group, e)
            role_name = self.config.entity_roles.get(e, f"entity{e}")
            if entity is not None:
                substitution_dict[role_name] = entity

        return template.format(**substitution_dict)

    def generate_full_prompt(
        self,
        binding_matrix: BindingMatrix,
        query_group: int,
        query_indices: Tuple[int, ...],
        answer_index: int,
    ) -> Tuple[str, str]:
        """
        Generate a complete prompt with statement + question, and the expected answer.

        Args:
            binding_matrix: The entity bindings
            query_group: Which group contains the query/answer
            query_indices: Entity positions mentioned in question
            answer_index: Position of the answer entity

        Returns:
            (full_prompt, expected_answer) tuple
        """
        # Generate the factual statement
        statement = self.fill_statement_template(binding_matrix)

        # Select and fill the question template
        question_template = self.select_question_template(query_indices, answer_index)
        question = self.fill_question_template(
            question_template, query_group, query_indices, binding_matrix
        )

        # Get the expected answer
        expected_answer = binding_matrix.get_entity(query_group, answer_index)
        if expected_answer is None:
            raise ValueError(
                f"Answer entity at group {query_group}, index {answer_index} is None"
            )

        # Combine statement and question with configurable separator
        separator = self.config.statement_question_separator
        core_prompt = f"{statement}{separator}{question}"

        # Apply prefix and suffix
        full_prompt = (
            f"{self.config.prompt_prefix}{core_prompt}{self.config.prompt_suffix}"
        )

        return full_prompt, expected_answer


def create_sample_binding_matrix(
    config: EntityBindingTaskConfig, num_groups: int
) -> BindingMatrix:
    """
    Create a sample binding matrix for testing.

    Args:
        config: Task configuration
        num_groups: Number of groups to create

    Returns:
        A BindingMatrix with randomly sampled entities
    """
    groups = []
    entities_per_group = config.max_entities_per_group

    for g in range(num_groups):
        entities = []
        for e in range(entities_per_group):
            if e in config.entity_pools:
                entity = random.choice(config.entity_pools[e])
                entities.append(entity)
            else:
                entities.append(f"entity_{g}_{e}")  # Fallback for missing pools
        groups.append(EntityGroup(entities, g))

    return BindingMatrix(groups, config.max_groups, config.max_entities_per_group)


# Import EntityGroup here to avoid circular imports
from .config import EntityGroup
