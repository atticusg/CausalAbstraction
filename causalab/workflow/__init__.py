"""The causalab-owned workflow runner (docs/workflow_protocol.md §8).

The document model (parse/validate/schedule/digest) lives in
:mod:`causalab.workflow.document` — backend- and torch-free; this package
executes loaded workflows: protocol steps through backend routing over the
run-tree artifact overlay, script steps by resolving their inputs and calling
``main(inputs, outputs)``, then the per-step ``_step.json`` records and the
``workflow.json`` run manifest.

There is no publication step: the run tree *is* the publication (§0).
"""

from causalab.workflow.runner import (
    OverlayArtifacts,
    WorkflowRunResult,
    run_workflow,
)

__all__ = ["OverlayArtifacts", "WorkflowRunResult", "run_workflow"]
