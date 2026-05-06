"""Criteria registry — maps criterion names to score modules.

Each module exposes a self-contained ``compute_score(...) -> dict[str, float]``
with a standardized signature.  The orchestrator calls ``compute_score`` once per
(task, featurizer, criterion).  Adding a new criterion = one score file + one entry
here.
"""

from __future__ import annotations

from types import ModuleType

from causalab.methods.scores import (
    coherence,
    distance_from_behavior_manifold,
)

CRITERIA_REGISTRY: dict[str, ModuleType] = {
    "coherence": coherence,
    "distance_from_behavior_manifold": distance_from_behavior_manifold,
}
