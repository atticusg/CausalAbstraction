"""The causalab-owned workflow runner (docs/workflow_protocol.md §8).

The document model (parse/validate/schedule/digest) lives in
:mod:`causalab.protocol.workflow` — backend- and torch-free; this package
executes loaded workflows: protocol steps through backend routing over
the run-tree artifact overlay, ``select`` reductions, the closed plot
vocabulary, save publication, and the run manifest.
"""

from causalab.workflow.runner import OverlayArtifacts, WorkflowRunResult, run_workflow

__all__ = ["OverlayArtifacts", "WorkflowRunResult", "run_workflow"]
