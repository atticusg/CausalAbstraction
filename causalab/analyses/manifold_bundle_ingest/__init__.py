"""Hydra analysis: ingest a manifold-SAE-autointerp community into characterize inputs.

The entry point is :func:`causalab.analyses.manifold_bundle_ingest.main.main`
(``main(cfg)``), dispatched by the runner when a config carries
``_name_: manifold_bundle_ingest``. It reads
``cfg.manifold_bundle_ingest.{bundle, community_id}`` and is **task-less and
model-less** — a data-preparation step that converts one community of a
manifold-SAE bundle into the ``characterize_subspace`` input set (rotation
safetensors + ``step1_dataset.json`` + ``subspace_manifest.json``), with no
hand-editing (issue #265). The pure logic lives in :mod:`.ingest`.
"""

from causalab.analyses.manifold_bundle_ingest.ingest import (
    build_characterize_inputs,
    extract_manifold_exemplars,
    find_community_record,
)
from causalab.analyses.manifold_bundle_ingest.main import ANALYSIS_NAME

# Note: the ``main`` *function* is intentionally not re-exported here — doing so
# would shadow the ``main`` *submodule* the runner imports (``mod.main``).

__all__ = [
    "ANALYSIS_NAME",
    "build_characterize_inputs",
    "extract_manifold_exemplars",
    "find_community_record",
]
