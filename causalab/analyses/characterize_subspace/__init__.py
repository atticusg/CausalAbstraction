"""Subspace characterization analysis.

Reproduces the geometry of a phase-1-supplied subspace on its own dataset,
collects broad-corpus webtext activations, and uses an LLM judge to derive
an independent hypothesis about what semantic axis the subspace tracks.
The judge call that derives the hypothesis never sees the user-supplied
significance description; a separate reconcile call compares the two. See
the module docstrings in :mod:`schemas`, :mod:`prompts`, and :mod:`judge`
for the invariants that enforce that.

The output is a refined-hypothesis bundle consumed downstream.
"""

from causalab.analyses.characterize_subspace.judge import (
    derive_hypothesis,
    reconcile_hypotheses,
)
from causalab.analyses.characterize_subspace.schemas import (
    JudgeHypothesis,
    PeakRecord,
    ProjectionStats,
    QuantileBin,
    ReconciliationIteration,
    ReconciliationResult,
    Significance,
    Span,
    Step1Summary,
    Verdict,
    WebtextEvidence,
)
from causalab.analyses.characterize_subspace.subspace_builder import (
    build_block_sae_artifact,
    build_subspace_artifact,
    resolve_cluster_feature_ids,
    resolve_subspace_artifact,
)

ANALYSIS_NAME = "characterize_subspace"

__all__ = [
    "ANALYSIS_NAME",
    "JudgeHypothesis",
    "PeakRecord",
    "ProjectionStats",
    "QuantileBin",
    "ReconciliationIteration",
    "ReconciliationResult",
    "Significance",
    "Span",
    "Step1Summary",
    "Verdict",
    "WebtextEvidence",
    "build_block_sae_artifact",
    "build_subspace_artifact",
    "derive_hypothesis",
    "reconcile_hypotheses",
    "resolve_cluster_feature_ids",
    "resolve_subspace_artifact",
]
