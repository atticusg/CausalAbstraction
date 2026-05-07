"""Featurizer loading.

Loads trained featurizer checkpoints from disk into InterchangeTarget objects.
"""

from __future__ import annotations

import logging
import os

from causalab.neural.featurizer import Featurizer
from causalab.neural.units import InterchangeTarget

logger = logging.getLogger(__name__)


def load_featurizer(
    featurizer_path: str | None,
    interchange_target: InterchangeTarget,
    layer: int,
    pos_id: str,
) -> Featurizer:
    """Load featurizer from disk into the interchange_target, return the featurizer.

    For identity (featurizer_path=None), returns a default Identity featurizer.
    """
    if featurizer_path is None:
        return Featurizer()

    key_str = f"{layer}__{pos_id}"
    unit_dir = os.path.join(featurizer_path, "models", key_str)

    if not os.path.exists(unit_dir):
        logger.warning(
            f"Featurizer dir not found: {unit_dir}, falling back to identity"
        )
        return Featurizer()

    interchange_target.load(unit_dir)

    units = interchange_target.flatten()
    if units:
        return units[0].featurizer
    return Featurizer()
