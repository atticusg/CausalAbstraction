"""Multidimensional scaling utilities for distance matrix visualization."""

from __future__ import annotations

import numpy as np


def mds_embed(
    D: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
    n_init: int = 20,
) -> np.ndarray:
    """Embed a pairwise distance matrix into low-dimensional coordinates via MDS.

    Args:
        D: (N, N) symmetric distance matrix.
        n_components: Target dimensionality (2 or 3).
        random_state: Random seed for reproducibility.
        n_init: Number of random restarts.

    Returns:
        (N, n_components) embedded coordinates.
    """
    from sklearn.manifold import MDS

    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        normalized_stress="auto",
        random_state=random_state,
        n_init=n_init,
    )
    return mds.fit_transform(D)
