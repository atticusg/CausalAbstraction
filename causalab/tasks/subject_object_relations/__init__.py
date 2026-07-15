"""subject_object_relations: factory task over 35 LRE subject→object relations.

Factory task — use ``SubjectObjectRelationsConfig(relation=<name>)`` (or
``task.relation=<name>`` in a runner config) to select the relation. Relation
content is bundled as model-agnostic JSON under ``data/relations/`` (built by
``data/build_relations.py``).
"""

from .config import (
    SubjectObjectRelationsConfig,
    load_manifest,
    relation_names,
)
from .causal_models import create_causal_model
from .counterfactuals import COUNTERFACTUAL_GENERATORS, generate_dataset
from .token_positions import create_token_positions

__all__ = [
    "SubjectObjectRelationsConfig",
    "load_manifest",
    "relation_names",
    "create_causal_model",
    "generate_dataset",
    "COUNTERFACTUAL_GENERATORS",
    "create_token_positions",
]
