"""Producer side of the subspace artifact the analysis consumes.

:func:`loading.load_subspace` reads a ``.safetensors`` rotation that some
phase-1 pipeline produced. This module builds such an artifact directly from
an SAE checkpoint plus a feature cluster, so an SAE cluster (the common
phase-1 source) can be ingested without a hand-rolled preprocessing script.

Layering: path I/O and artifact layout live here (analysis layer); the pure
decoder-directions -> orthonormal-basis math lives in
:func:`causalab.methods.sae.decoder_subspace` (docs/CODEBASE.md §3 invariant 4).

CLI::

    uv run python -m causalab.analyses.characterize_subspace.subspace_builder \\
        --sae-checkpoint /path/to/sae.pt \\
        --clusters /path/to/clustered_sae_latent_semantic_labels.json \\
        --cluster-id 950 \\
        --out /path/to/subspace.safetensors
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
from collections.abc import Mapping
from typing import Any

from safetensors.torch import save_file

from causalab.io.sae_checkpoints import load_block_sae_frame, read_sae_decoder
from causalab.methods.sae import decoder_subspace

logger = logging.getLogger(__name__)

# read_sae_decoder is re-exported (it now lives in causalab.io.sae_checkpoints,
# the sanctioned home for reading foreign SAE checkpoints; #319) so existing
# `from ...subspace_builder import read_sae_decoder` callers keep working.
__all__ = [
    "build_block_sae_artifact",
    "build_subspace_artifact",
    "read_sae_decoder",
    "resolve_cluster_feature_ids",
    "resolve_subspace_artifact",
]

_SOURCE_REQUIRED_KEYS = ("sae_checkpoint", "clusters_path", "cluster_id")


def resolve_cluster_feature_ids(clusters_path: str, cluster_id: str | int) -> list[int]:
    """Return the sorted, de-duplicated feature ids for a cluster.

    Reads a ``clustered_sae_latent_semantic_labels.json``-style mapping of
    ``{cluster_id: [{"feature_id"|"feature": int, ...}, ...]}``. Accepts a
    string or int ``cluster_id`` (keys are strings on disk).
    """
    with open(clusters_path, "r", encoding="utf-8") as fh:
        clusters = json.load(fh)
    key = str(cluster_id)
    if key not in clusters:
        raise KeyError(
            f"Cluster {key!r} not found in {clusters_path}. "
            f"Known clusters (first 20): {sorted(clusters)[:20]}"
        )
    feats: set[int] = set()
    for entry in clusters[key]:
        fid = entry.get("feature_id", entry.get("feature"))
        if fid is not None:
            feats.add(int(fid))
    if not feats:
        raise ValueError(f"Cluster {key!r} in {clusters_path} has no feature ids.")
    return sorted(feats)


def build_block_sae_artifact(
    *,
    block_sae_checkpoint: str,
    block_id: int,
    out_path: str,
) -> dict[str, Any]:
    """Build a ``.safetensors`` subspace artifact from one block of a block SAE.

    A block/Grassmannian SAE feature is *already* a K-dim orthonormal Stiefel
    subspace, so (unlike :func:`build_subspace_artifact`) there is no clustering
    or QR — :func:`causalab.io.sae_checkpoints.load_block_sae_frame` returns the
    block's frame directly. Writes it under the ``rotation_matrix`` key (the key
    :func:`loading.load_subspace` and ``subspace/fixed.py`` both read) and returns
    a provenance dict carrying the loader's metadata.
    """
    frame, block_meta = load_block_sae_frame(str(block_sae_checkpoint), int(block_id))
    d_model, k = int(frame.shape[0]), int(frame.shape[1])

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({"rotation_matrix": frame}, str(out))

    provenance = {
        "subspace_artifact": str(out),
        "block_sae_checkpoint": str(block_sae_checkpoint),
        "block_id": int(block_id),
        "d_model": d_model,
        "k_features": k,
        "orthonormalized": True,  # block frames are orthonormal Stiefel frames
        "block_sae_meta": block_meta,
    }
    logger.info(
        "Wrote block-SAE subspace artifact %s: (d_model=%d, k=%d) from block_id=%d of %s.",
        out,
        d_model,
        k,
        int(block_id),
        block_sae_checkpoint,
    )
    return provenance


def build_subspace_artifact(
    *,
    sae_checkpoint: str,
    clusters_path: str,
    cluster_id: str | int,
    out_path: str,
    orthonormalize: bool = True,
) -> dict[str, Any]:
    """Build a ``.safetensors`` subspace artifact from an SAE feature cluster.

    Resolves the cluster's feature ids, reads the SAE decoder, builds a
    ``(d_model, k)`` basis via :func:`decoder_subspace`, and writes it under
    the ``rotation_matrix`` key (the first key :func:`loading.load_subspace`
    looks for). Returns a provenance dict suitable for a manifest.
    """
    feature_ids = resolve_cluster_feature_ids(clusters_path, cluster_id)
    decoder, d_model_hint = read_sae_decoder(sae_checkpoint)
    rotation = decoder_subspace(
        decoder, feature_ids, d_model=d_model_hint, orthonormalize=orthonormalize
    )
    d_model, k = int(rotation.shape[0]), int(rotation.shape[1])

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({"rotation_matrix": rotation}, str(out))

    provenance = {
        "subspace_artifact": str(out),
        "sae_checkpoint": sae_checkpoint,
        "clusters_path": clusters_path,
        "cluster_id": str(cluster_id),
        "feature_ids": feature_ids,
        "d_model": d_model,
        "k_features": k,
        "orthonormalized": orthonormalize,
    }
    logger.info(
        "Wrote subspace artifact %s: (d_model=%d, k=%d) from cluster %s (%d features).",
        out,
        d_model,
        k,
        cluster_id,
        len(feature_ids),
    )
    return provenance


def resolve_subspace_artifact(
    *,
    artifact: str | None,
    source: Mapping[str, Any] | None,
    out_dir: str,
    artifact_name: str = "subspace.safetensors",
) -> tuple[str, dict[str, Any] | None]:
    """Return a path to the subspace ``.safetensors``, building it if needed.

    Exactly one of ``artifact`` (a ready file) or ``source`` must be provided.
    When ``source`` is given the artifact is built under ``out_dir`` and the
    build provenance is returned alongside the path (``None`` when a ready
    ``artifact`` was supplied). ``source`` dispatches on its kind:

    - ``{block_sae_checkpoint, block_id}`` — one block of a block/Grassmannian
      SAE (built via :func:`build_block_sae_artifact`).
    - ``{sae_checkpoint, clusters_path, cluster_id[, orthonormalize]}`` — a
      vanilla SAE decoder cluster (built via :func:`build_subspace_artifact`).
    """
    has_artifact = bool(artifact)
    has_source = source is not None and len(source) > 0
    if has_artifact and has_source:
        raise ValueError(
            "Provide either subspace.artifact or subspace.source, not both."
        )
    if not has_artifact and not has_source:
        raise ValueError("One of subspace.artifact or subspace.source is required.")
    if has_artifact:
        return str(artifact), None

    assert source is not None  # narrowed by has_source
    out_path = os.path.join(out_dir, artifact_name)

    # Block/Grassmannian SAE source: one block = one K-dim Stiefel subspace.
    # `is None`, not falsy: block_id=0 is a legitimate block index.
    if source.get("block_sae_checkpoint") is not None:
        if not str(source["block_sae_checkpoint"]).strip():
            raise ValueError("subspace.source.block_sae_checkpoint is empty.")
        if source.get("block_id") is None:
            raise ValueError(
                "subspace.source.block_sae_checkpoint requires block_id "
                "(the index into the SAE's n_groups axis)."
            )
        provenance = build_block_sae_artifact(
            block_sae_checkpoint=str(source["block_sae_checkpoint"]),
            block_id=int(source["block_id"]),
            out_path=out_path,
        )
        return out_path, provenance

    # Otherwise: vanilla SAE decoder-cluster source.
    # `is None`, not falsy: cluster_id=0 (int) is a legitimate cluster key.
    missing = [k for k in _SOURCE_REQUIRED_KEYS if source.get(k) is None]
    if missing:
        raise ValueError(
            f"subspace.source is missing required keys: {missing}. Provide a "
            "vanilla-cluster source {sae_checkpoint, clusters_path, cluster_id} "
            "or a block source {block_sae_checkpoint, block_id}."
        )
    empty_paths = [
        k for k in ("sae_checkpoint", "clusters_path") if not str(source[k]).strip()
    ]
    if empty_paths:
        raise ValueError(f"subspace.source has empty path values: {empty_paths}.")
    provenance = build_subspace_artifact(
        sae_checkpoint=str(source["sae_checkpoint"]),
        clusters_path=str(source["clusters_path"]),
        cluster_id=source["cluster_id"],
        out_path=out_path,
        orthonormalize=bool(source.get("orthonormalize", True)),
    )
    return out_path, provenance


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m causalab.analyses.characterize_subspace.subspace_builder",
        description=(
            "Build a (d_model, k) subspace .safetensors from an SAE checkpoint "
            "and a feature cluster, for the characterize_subspace analysis to ingest."
        ),
    )
    parser.add_argument("--sae-checkpoint", required=True, help="Path to the SAE .pt.")
    parser.add_argument(
        "--clusters",
        required=True,
        help="Path to clustered_sae_latent_semantic_labels.json.",
    )
    parser.add_argument(
        "--cluster-id", required=True, help="Cluster id to build the subspace from."
    )
    parser.add_argument("--out", required=True, help="Output .safetensors path.")
    parser.add_argument(
        "--no-orthonormalize",
        action="store_true",
        help="Stack raw decoder directions instead of an orthonormal QR basis.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    provenance = build_subspace_artifact(
        sae_checkpoint=args.sae_checkpoint,
        clusters_path=args.clusters,
        cluster_id=args.cluster_id,
        out_path=args.out,
        orthonormalize=not args.no_orthonormalize,
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
