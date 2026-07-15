"""Identity-aware caching for the ``output_manifold`` belief-distribution artifact.

The per-example output distributions (``per_example_output_dists.safetensors``)
are expensive to collect — a full model forward pass per training example — so
``output_manifold`` caches them under a shared ``experiment_root`` and reuses
them across TPS configs. The original cache validated only the last tensor
dimension (``n_classes + 1``), so a debug pass (``n_train=16``) and a full pass
(``n_train=64``) sharing one root silently reused the small debug cache and then
crashed downstream in ``load_natural_distributions`` (see GH #220).

This module makes reuse safe by:

1. Recording a comprehensive **identity** alongside the artifact (everything that
   determines its content — task config, seed, distribution columns, and the
   model/prompt settings — but *not* device/attn impl, so a cache can be reused
   across devices) in a ``.meta.json`` sidecar via ``save_tensors_with_meta``.
2. Reusing the cache only when the current run's identity matches the sidecar and
   the loaded tensor passes structural sanity checks (row count, column count,
   rows summing to ~1).
3. Never overwriting a differing run: the displaced canonical artifacts are moved
   to a timestamped ``archive/<ts>/`` folder before recollection.

The canonical fixed-path artifact is preserved so downstream consumers
(``path_steering``, ``pullback`` via ``methods/pullback/geodesic.py``) keep
reading ``per_example_output_dists.safetensors`` unchanged.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import shutil
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.io.artifacts import load_tensor_results
from causalab.io.plots.figure_format import ALLOWED_FIGURE_FORMATS
from causalab.runner.helpers import _task_config_for_metadata

logger = logging.getLogger(__name__)

# Stem (no extension) for the belief-distribution artifact pair written via
# ``save_tensors_with_meta`` → ``per_example_output_dists.{safetensors,meta.json}``.
BELIEF_DISTS_STEM = "per_example_output_dists"

# Canonical belief-cache members in an output_manifold dir, in archive order.
# The dists + its sidecar, the derived Hellinger PCA + its sidecar, and the
# interactive 3D viz. All are regenerated from the dists, so must not survive a
# recollection of different data.
_CANONICAL_MEMBERS = (
    f"{BELIEF_DISTS_STEM}.safetensors",
    f"{BELIEF_DISTS_STEM}.meta.json",
    "hellinger_pca.safetensors",
    "hellinger_pca.meta.json",
    "hellinger_pca_3d.html",
)

# The 2D Hellinger-PCA scatter, whose extension follows the figure_format config
# (png by default, pdf opt-in). Matched by stem so whichever format is present
# gets archived alongside the canonical members above.
_CANONICAL_VIZ_STEMS = ("hellinger_pca_2d",)


def build_belief_identity(cfg: DictConfig, task: Any) -> dict:
    """Comprehensive, JSON-safe identity for a belief-distribution artifact.

    Captures everything that determines the artifact's content so two runs reuse
    the same cache only when they would produce identical distributions. Device
    and attention implementation are intentionally excluded — they may perturb
    numerics but not semantics, and excluding them lets a cache be reused across
    devices.
    """
    task_config = _task_config_for_metadata(
        OmegaConf.to_container(cfg.task, resolve=True)
    )
    return {
        "task_config": task_config,
        "seed": cfg.seed,
        "intervention_variable": task.intervention_variable,
        "intervention_values": [str(v) for v in task.intervention_values],
        "model": {
            "name": cfg.model.name,
            "dtype": cfg.model.get("dtype"),
            "chat_template": cfg.model.get("chat_template", False),
            "chat_answer_directive": cfg.model.get("chat_answer_directive"),
        },
    }


def belief_cache_status(
    out_root: str,
    identity: dict,
    expected_rows: int,
    expected_dim: int,
    *,
    row_sum_atol: float = 1e-3,
) -> tuple[bool, str]:
    """Decide whether the cached belief-distribution artifact may be reused.

    Returns ``(reusable, reason)``. Reuse requires that the artifact exists, is a
    2-D tensor of shape ``(expected_rows, expected_dim)`` whose rows sum to ~1,
    and — when a metadata sidecar is present — that its recorded identity equals
    ``identity``. A pre-fix artifact with no sidecar (``legacy``) is accepted on
    structural sanity alone, which still recollects a stale smaller-``n`` cache
    (the GH #220 failure).
    """
    dists_path = os.path.join(out_root, f"{BELIEF_DISTS_STEM}.safetensors")
    if not os.path.exists(dists_path):
        return False, "no cached dists artifact"

    saved = load_tensor_results(out_root, f"{BELIEF_DISTS_STEM}.safetensors")["dists"]
    if saved.ndim != 2:
        return False, f"cached dists is not 2-D (shape {tuple(saved.shape)})"
    if saved.shape[0] != expected_rows:
        return False, f"row count {saved.shape[0]} != expected {expected_rows}"
    if saved.shape[-1] != expected_dim:
        return False, f"last-dim {saved.shape[-1]} != expected {expected_dim}"

    row_sums = saved.float().sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=row_sum_atol):
        return False, "rows do not sum to ~1 (not valid distributions)"

    meta_path = os.path.join(out_root, f"{BELIEF_DISTS_STEM}.meta.json")
    if not os.path.exists(meta_path):
        return True, "legacy artifact (no metadata) passed structural sanity"

    try:
        with open(meta_path) as f:
            stored_identity = json.load(f).get("identity")
    except (json.JSONDecodeError, OSError) as e:
        return False, f"could not read metadata sidecar ({e})"

    if stored_identity != identity:
        return False, "identity mismatch vs metadata sidecar"
    return True, "identity + structural checks passed"


def archive_belief_artifacts(out_root: str) -> str | None:
    """Move the existing canonical belief-cache set into ``archive/<ts>/``.

    ``<ts>`` is the displaced dists sidecar's ``created_at`` when available (so the
    archive is named for when the displaced artifact was produced), otherwise the
    current UTC time. Returns the archive directory, or ``None`` when there is
    nothing to archive (a fresh run).
    """
    present = [
        m for m in _CANONICAL_MEMBERS if os.path.exists(os.path.join(out_root, m))
    ]
    present += [
        f"{stem}.{fmt}"
        for stem in _CANONICAL_VIZ_STEMS
        for fmt in sorted(ALLOWED_FIGURE_FORMATS)
        if os.path.exists(os.path.join(out_root, f"{stem}.{fmt}"))
    ]
    if not present:
        return None

    base_dest = os.path.join(out_root, "archive", _displaced_timestamp(out_root))
    dest = base_dest
    suffix = 1
    while os.path.exists(dest):
        dest = f"{base_dest}__{suffix}"
        suffix += 1
    os.makedirs(dest, exist_ok=True)

    for member in present:
        shutil.move(os.path.join(out_root, member), os.path.join(dest, member))
    logger.info("Archived %d displaced belief artifact(s) to %s", len(present), dest)
    return dest


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (for the ``created_at`` sidecar field)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _displaced_timestamp(out_root: str) -> str:
    meta_path = os.path.join(out_root, f"{BELIEF_DISTS_STEM}.meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                created = json.load(f).get("created_at")
            if created:
                return _sanitize_ts(str(created))
        except (json.JSONDecodeError, OSError):
            pass
    return _sanitize_ts(utc_now_iso())


def _sanitize_ts(ts: str) -> str:
    """Collapse non-alphanumeric runs to ``-`` so a timestamp is filesystem-safe."""
    return re.sub(r"[^0-9A-Za-z]+", "-", ts).strip("-")
