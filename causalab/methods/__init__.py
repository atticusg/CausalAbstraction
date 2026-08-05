"""causalab.methods — compositions over the intervention surface in ``neural/``.

Importing this package registers every concrete ``Featurizer`` subclass so
that ``causalab.neural.featurizer.Featurizer.from_dict`` can dispatch to them
via ``Featurizer.__subclasses__()``.
"""

from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.methods.standardize import StandardizeFeaturizer
from causalab.methods.spline.featurizer import (
    ManifoldFeaturizer,
    ManifoldProjectFeaturizer,
)
from causalab.methods.sae import SAEFeaturizer
from causalab.methods.umap import UMAPFeaturizer

__all__ = [
    "SubspaceFeaturizer",
    "StandardizeFeaturizer",
    "ManifoldFeaturizer",
    "ManifoldProjectFeaturizer",
    "SAEFeaturizer",
    "UMAPFeaturizer",
]
