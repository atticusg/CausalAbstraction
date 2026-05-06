"""Load a subspace featurizer onto an interchange target."""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


def load_subspace_onto_target(
    target,
    subspace_dir: str,
    method: str,
    k_features: int,
    layer: int | None = None,
) -> None:
    """Load a subspace featurizer (PCA rotation or DAS checkpoint) onto a target.

    Args:
        target: Interchange target whose first unit gets the featurizer.
        subspace_dir: Path to the subspace output directory.
        method: ``"pca"`` or ``"das"``.
        k_features: Number of subspace dimensions (used for DAS).
        layer: Explicit layer override for DAS featurizer loading.
            When *None*, the layer is read from ``metadata.json``.
    """
    unit = target.flatten()[0]

    if method == "pca":
        from safetensors.torch import load_file
        from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer

        rotation_path = os.path.join(subspace_dir, "rotation.safetensors")
        if os.path.exists(rotation_path):
            data = load_file(rotation_path)
            rotation = data["rotation_matrix"]
            feat = SubspaceFeaturizer(
                rotation_subspace=rotation, trainable=False, id="PCA"
            )
            unit.set_featurizer(feat)
            logger.info("Loaded PCA rotation from %s", rotation_path)
        else:
            logger.warning("No PCA rotation found at %s", rotation_path)
    elif method == "das":
        from causalab.analyses.activation_manifold.loading import load_featurizer

        das_dir = os.path.join(subspace_dir, "das")
        if os.path.isdir(das_dir):
            pos_id = unit.id.split(".")[-1]
            if layer is None:
                meta_path = os.path.join(subspace_dir, "metadata.json")
                if os.path.isfile(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    layer = meta.get("layer", 0)
                else:
                    layer = 0
            load_featurizer(das_dir, target, layer, pos_id)
            logger.info("Loaded DAS featurizer from %s", das_dir)
