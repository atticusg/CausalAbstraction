"""The reference protocol backend: native pytorch hooks (spec §8).

The only subpackage of ``causalab/neural`` after the protocol refactor:
site resolution over raw module hooks, position resolution against the
padded batch frame, the closed mechanism set, featurizers with the
error-term contract, metric lowering, the train loop, and artifact
stamping. Everything enters through :class:`PytorchHooksBackend`.
"""

from causalab.neural.pytorch_hooks.backend import PytorchHooksBackend
from causalab.neural.pytorch_hooks.encoding import (
    EncodedBatch,
    encode,
    resolve_position,
)
from causalab.neural.pytorch_hooks.executor import PointExecutor
from causalab.neural.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.pytorch_hooks.sites import ResolvedSite, resolve_site

__all__ = [
    "EncodedBatch",
    "ModelBundle",
    "PointExecutor",
    "PytorchHooksBackend",
    "ResolvedSite",
    "encode",
    "load_model",
    "resolve_position",
    "resolve_site",
]
