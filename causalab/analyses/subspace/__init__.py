"""Subspace discovery methods.

Each function finds a low-dimensional subspace that encodes a causal variable,
sets the featurizer on the interchange target, and returns the projected features.

Grid-mode functions scan a (layer x token_position) grid and return per-cell
scores suitable for heatmap visualization.
"""

from causalab.analyses.subspace.pca import find_pca_subspace
from causalab.analyses.subspace.das import find_das_subspace
from causalab.analyses.subspace.dbm import find_dbm_subspace
from causalab.analyses.subspace.boundless import find_boundless_subspace
from causalab.analyses.subspace.loading import load_subspace_onto_target
from causalab.analyses.subspace.grid import (
    run_das_grid,
    run_pca_grid,
    run_dbm_grid,
    run_boundless_grid,
)

__all__ = [
    "find_pca_subspace",
    "find_das_subspace",
    "find_dbm_subspace",
    "find_boundless_subspace",
    "load_subspace_onto_target",
    "run_das_grid",
    "run_pca_grid",
    "run_dbm_grid",
    "run_boundless_grid",
]
