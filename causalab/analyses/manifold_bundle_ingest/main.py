"""Hydra entry point: ingest a manifold-SAE bundle community into characterize inputs.

manifold_bundle_ingest answers: *which explore-subspace inputs does a given
manifold-SAE-autointerp community correspond to?* — a **task-less, model-less**
data-preparation analysis that reads one community's record from a bundle and
writes the rotation safetensors, exemplar ``step1_dataset.json``, and a
``subspace_manifest.json`` (issue #265). Its outputs are the prerequisites for
``characterize_subspace``: point that analysis's ``subspace.artifact`` /
``step1_dataset`` / ``significance`` at the manifest this step emits.

Reads ``cfg.manifold_bundle_ingest.{bundle, community_id}`` only — no ``cfg.task``
(its inputs are a bundle directory, not a runner-generated task dataset) and no
``cfg.model`` (it queries no model; it is a format adapter). The heavy lifting
lives in :mod:`.ingest`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from omegaconf import DictConfig

from causalab.analyses.manifold_bundle_ingest.ingest import build_characterize_inputs

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "manifold_bundle_ingest"


def _resolve_output_dir(analysis_cfg: DictConfig) -> str:
    out_dir = analysis_cfg.get("_output_dir") or os.path.join(
        os.getcwd(), ANALYSIS_NAME
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the bundle-ingestion step and return the emitted manifest."""
    acfg = cfg[ANALYSIS_NAME]
    bundle = acfg.get("bundle")
    if not bundle:
        raise ValueError(
            "manifold_bundle_ingest.bundle is required "
            "(path to the manifold-SAE-autointerp bundle dir)."
        )
    community_id = acfg.get("community_id")
    if community_id is None:
        raise ValueError(
            "manifold_bundle_ingest.community_id is required (the `comm` field)."
        )

    out_dir = _resolve_output_dir(acfg)
    manifest = build_characterize_inputs(
        bundle_dir=str(bundle),
        community_id=int(community_id),
        out_dir=out_dir,
    )
    logger.info(
        "manifold_bundle_ingest wrote characterize inputs for community %s to %s",
        community_id,
        out_dir,
    )
    return {"output_dir": out_dir, "manifest": manifest}
