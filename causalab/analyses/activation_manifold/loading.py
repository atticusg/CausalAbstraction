"""Featurizer loading.

Loads trained featurizer checkpoints from disk — constructively, via
:func:`causalab.neural.specs.load_site_specs` (WU5, #507). The legacy path
mutated a caller-prebuilt target in place; the constructive loader returns
the stored featurizer (and feature ids), and callers apply it functionally —
``spec.with_featurizer(...)`` / ``spec.with_feature_ids(...)``.

Reads both bundle formats: the WU1 spec bundle (``sites.json`` +
``featurizers.safetensors``/``.meta.json``, written by
:func:`causalab.io.artifacts.save_training_artifacts` and
:func:`causalab.neural.specs.save_site_specs`) and the legacy
``units_metadata.json`` bundle, written by the retired target ``save``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

from causalab.neural.featurizer import Featurizer
from causalab.neural.specs import SiteSpec, load_site_specs

logger = logging.getLogger(__name__)


def load_featurizer(
    featurizer_path: str | None,
    layer: int,
    pos_id: str,
) -> tuple[Featurizer, Optional[Sequence[int]]]:
    """Load one cell's stored featurizer (and feature ids) from disk.

    Reads the single-spec bundle at ``<featurizer_path>/models/<layer>__<pos_id>``
    (both WU1 and legacy formats, via ``load_site_specs``) and returns
    ``(featurizer, feature_ids)``. For identity (``featurizer_path=None``) or a
    missing bundle directory, returns ``(Featurizer(), None)``.

    The load is constructive: no target/spec is mutated. Apply the result
    functionally, e.g.::

        featurizer, feature_ids = load_featurizer(path, layer, pos_id)
        spec = spec.with_featurizer(featurizer).with_feature_ids(feature_ids)

    Positions are deliberately not rebound here (``token_positions=None``):
    callers already hold specs whose positions came from the task config, and
    only the feature-space state is loaded.
    """
    if featurizer_path is None:
        return Featurizer(), None

    key_str = f"{layer}__{pos_id}"
    unit_dir = os.path.join(featurizer_path, "models", key_str)

    if not os.path.exists(unit_dir):
        logger.warning(
            f"Featurizer dir not found: {unit_dir}, falling back to identity"
        )
        return Featurizer(), None

    specs = load_site_specs(unit_dir)
    if len(specs) != 1:
        raise ValueError(
            f"Expected a single-spec bundle at {unit_dir}, found {len(specs)} "
            "records; per-cell featurizer bundles store exactly one site."
        )
    loaded = specs[0]
    return loaded.fsite.featurizer, loaded.fsite.feature_ids


def apply_loaded_featurizer(
    spec: SiteSpec,
    featurizer_path: str | None,
    layer: int,
    pos_id: str,
) -> tuple[SiteSpec, Featurizer]:
    """Functional convenience over :func:`load_featurizer`.

    Returns ``(updated_spec, featurizer)`` where ``updated_spec`` carries the
    loaded featurizer (and, when stored, feature ids); the input ``spec`` is
    unchanged (frozen).
    """
    featurizer, feature_ids = load_featurizer(featurizer_path, layer, pos_id)
    updated = spec.with_featurizer(featurizer)
    if feature_ids is not None:
        updated = updated.with_feature_ids(feature_ids)
    return updated, featurizer
