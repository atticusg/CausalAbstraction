"""Load a subspace featurizer onto a site spec — functionally."""

from __future__ import annotations

import json
import logging
import os

from causalab.neural.specs import SiteSpec

logger = logging.getLogger(__name__)


def load_subspace_onto_spec(
    spec: SiteSpec,
    subspace_dir: str,
    method: str,
    k_features: int,
    layer: int | None = None,
) -> SiteSpec:
    """Load a subspace featurizer (PCA rotation or DAS checkpoint) onto a spec.

    Constructive successor of the retired mutating loader (WU5, #507): returns
    a new :class:`~causalab.neural.specs.SiteSpec` carrying the loaded
    featurizer (the input spec is frozen and unchanged). When no artifact is
    found the spec is returned as-is, with a warning — matching the legacy
    fall-through.

    The **pca** branch keeps its artifact contract unchanged: it reads
    ``rotation.safetensors``'s ``rotation_matrix`` tensor directly and wraps
    it in a frozen ``SubspaceFeaturizer``. The **das** branch reads the
    trained per-cell bundle (WU1 spec bundle or legacy format) via
    :func:`causalab.analyses.activation_manifold.loading.load_featurizer`.

    Args:
        spec: Site spec whose featurizer to replace (positions/key unchanged).
        subspace_dir: Path to the subspace output directory.
        method: ``"pca"`` or ``"das"``.
        k_features: Number of subspace dimensions (unused; kept for signature
            stability with config-driven callers).
        layer: Explicit layer override for DAS featurizer loading.
            When *None*, the layer is read from ``metadata.json``.
    """
    del k_features  # config-threaded; the artifacts carry their own shapes

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
            logger.info("Loaded PCA rotation from %s", rotation_path)
            return spec.with_featurizer(feat)
        logger.warning("No PCA rotation found at %s", rotation_path)
        return spec
    elif method == "das":
        from causalab.analyses.activation_manifold.loading import load_featurizer

        das_dir = os.path.join(subspace_dir, "das")
        if os.path.isdir(das_dir):
            # The bundle path is models/<layer>__<pos_id>; the position name
            # comes structurally from the spec's own resolver (the retired
            # loader parsed it out of the unit id string).
            pos_id = getattr(spec.positions, "id", None)
            if pos_id is None:
                raise ValueError(
                    f"spec {spec.key!r} has no named position resolver; the DAS "
                    "subspace bundle is keyed by (layer, position name)."
                )
            if layer is None:
                meta_path = os.path.join(subspace_dir, "metadata.json")
                if os.path.isfile(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    layer = int(meta.get("layer", 0) or 0)
                else:
                    layer = 0
            featurizer, feature_ids = load_featurizer(das_dir, layer, pos_id)
            updated = spec.with_featurizer(featurizer)
            if feature_ids is not None:
                updated = updated.with_feature_ids(feature_ids)
            logger.info("Loaded DAS featurizer from %s", das_dir)
            return updated
        return spec
    # Legacy fall-through: methods without a loadable featurizer artifact
    # (e.g. dbm/boundless) leave the spec unchanged.
    logger.warning("No loadable subspace featurizer for method %r", method)
    return spec
