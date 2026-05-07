"""causalab.methods — compositions over the pyvene surface in ``neural/``.

Importing this package registers every concrete ``Featurizer`` subclass so
that ``causalab.neural.featurizer.Featurizer.from_dict`` can dispatch to them
via ``Featurizer.__subclasses__()``.
"""

from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer  # noqa: F401
from causalab.methods.standardize import StandardizeFeaturizer  # noqa: F401
from causalab.methods.spline.featurizer import (  # noqa: F401
    ManifoldFeaturizer,
    ManifoldProjectFeaturizer,
)
from causalab.methods.sae import SAEFeaturizer  # noqa: F401
from causalab.methods.umap import UMAPFeaturizer  # noqa: F401
