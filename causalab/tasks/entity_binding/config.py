"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Core data structures for Entity Binding tasks.

This module provides the fundamental data structures needed to represent
entity binding tasks with arbitrary numbers of entity groups and entities per group.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class EntityBindingTaskConfig:
    """
    Configuration for an entity binding task.

    This defines the structure of the task including:
    - Maximum dimensions (groups and entities per group)
    - Entity pools for each role position
    - Templates for generating text
    - Prompt formatting (prefix/suffix for instruction tuning)
    """

    max_groups: int  # Maximum number of entity groups (k)
    max_entities_per_group: int  # Maximum entities per group (d)
    entity_roles: Dict[int, str]  # {0: "person", 1: "food", 2: "location"}
    entity_pools: Dict[int, List[str]]  # {0: ["Pete", "Ann"], 1: ["jam", "pie"]}
    statement_template: str  # Template for the factual statements
    question_templates: Dict[
        Tuple[Tuple[int, ...], int], str
    ]  # Question templates by (query_indices, answer_index)
    delimiters: List[str]  # Delimiters used in statement conjunction
    prompt_prefix: str = ""  # Text to prepend before the prompt
    prompt_suffix: str = ""  # Text to append after the prompt
    statement_question_separator: str = " "  # Separator between statement and question
    fixed_query_indices: Optional[Tuple[int, ...]] = (
        None  # If set, always use these query indices
    )


class EntityGroup:
    """
    Represents one binding group G_i = (entity_0, entity_1, ..., entity_m).

    In the example "Pete loves jam", this would be EntityGroup(["Pete", "jam"], 0)
    """

    def __init__(self, entities: List[str], group_index: int):
        self.entities = entities
        self.group_index = group_index

    def get_entity(self, entity_index: int) -> Optional[str]:
        """Get entity at position entity_index, or None if index is out of bounds."""
        if 0 <= entity_index < len(self.entities):
            return self.entities[entity_index]
        return None

    def __repr__(self):
        return f"EntityGroup({self.entities}, group_{self.group_index})"


class BindingMatrix:
    """
    Represents the full binding matrix G with all entity groups.

    This is the core data structure that holds all the entity bindings
    for a particular instance of the task.
    """

    def __init__(
        self, groups: List[EntityGroup], max_groups: int, max_entities_per_group: int
    ):
        self.groups = groups
        self.active_groups = len(groups)
        self.max_groups = max_groups
        self.max_entities_per_group = max_entities_per_group

    def get_entity(self, group_idx: int, entity_idx: int) -> Optional[str]:
        """
        Get G[group_idx][entity_idx], return None if inactive/out of bounds.

        This handles the case where we have fewer active groups or entities
        than the maximum allowed by the configuration.
        """
        if group_idx < self.active_groups and group_idx < len(self.groups):
            return self.groups[group_idx].get_entity(entity_idx)
        return None

    def get_active_groups(self) -> int:
        """Get the maximum number of active groups."""
        return self.active_groups

    def get_entities_per_group(self) -> int:
        """Get the number of entities in the first active group (assumes all groups have same size)."""
        if self.groups:
            return len(self.groups[0].entities)
        return 0

    def __repr__(self):
        return f"BindingMatrix({self.groups}, active={self.active_groups})"


def create_sample_love_config() -> EntityBindingTaskConfig:
    """
    Create a sample configuration for the "love" task (Pete loves jam, Ann loves pie).

    This is a simple 2-entity-per-group task with people and foods.
    """
    return EntityBindingTaskConfig(
        max_groups=3,  # Support up to 3 groups
        max_entities_per_group=2,  # 2 entities per group (person, food)
        entity_roles={0: "person", 1: "food"},
        entity_pools={
            0: ["Pete", "Ann", "Tim", "Bob", "Sue", "Kate"],
            1: ["jam", "pie", "cake", "bread", "soup", "tea"],
        },
        statement_template="{entity_e0} loves {entity_e1}",
        delimiters=[", ", "FILL", ", and ", "."],
        question_templates={
            # Query person (index 0), answer food (index 1)
            ((0,), 1): "What does {query_entity} love?",
            # Query food (index 1), answer person (index 0)
            ((1,), 0): "Who loves {query_entity}?",
        },
    )


def create_sample_action_config() -> EntityBindingTaskConfig:
    """
    Create a sample configuration for action tasks (Pete put jam in the cup).

    This is a 3-entity-per-group task with person, object, location.
    """
    return EntityBindingTaskConfig(
        max_groups=3,  # Support up to 3 action groups
        max_entities_per_group=3,  # 3 entities per group (person, object, location)
        entity_roles={0: "person", 1: "object", 2: "location"},
        entity_pools={
            0: ["Pete", "Ann", "Bob", "Sue", "Tim", "Kate", "Dan", "Lily"],
            1: ["jam", "water", "book", "coin", "pen", "key", "phone", "watch"],
            2: ["cup", "box", "table", "shelf", "drawer", "bag", "pocket", "basket"],
        },
        statement_template="{entity_e0} put {entity_e1} in the {entity_e2}",
        delimiters=[", ", "FILL", ", and ", "."],
        question_templates={
            # === SINGLE ENTITY QUERIES ===
            # Query person (0), answer object (1) - only mention the person
            ((0,), 1): "What did {person} put somewhere?",
            # Query person (0), answer location (2) - only mention the person
            ((0,), 2): "Where did {person} put something?",
            # Query object (1), answer person (0) - only mention the object
            ((1,), 0): "Who put {object} somewhere?",
            # Query object (1), answer location (2) - only mention the object
            ((1,), 2): "Where was {object} put?",
            # Query location (2), answer person (0) - only mention the location
            ((2,), 0): "Who put something in the {location}?",
            # Query location (2), answer object (1) - only mention the location
            ((2,), 1): "What was put in the {location}?",
            # === TWO ENTITY QUERIES ===
            # Query person+object (0,1), answer location (2) - mention both person and object
            ((0, 1), 2): "Where did {person} put {object}?",
            # Query person+location (0,2), answer object (1) - mention person and location
            ((0, 2), 1): "What did {person} put in the {location}?",
            # Query object+location (1,2), answer person (0) - mention object and location
            ((1, 2), 0): "Who put {object} in the {location}?",
        },
    )
