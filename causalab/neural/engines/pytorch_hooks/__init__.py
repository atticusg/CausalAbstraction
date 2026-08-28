"""The reference protocol engine: native pytorch hooks (spec §8).

Site resolution over raw module hooks, position resolution against the
padded batch frame, the closed mechanism set, featurizers with the
error-term contract, metric lowering, the train loop, and artifact
stamping. Everything enters through :class:`PytorchHooksEngine`.
"""

from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.shared.encoding import (
    EncodedBatch,
    encode,
    resolve_position,
)
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.sites import ResolvedSite, resolve_site

__all__ = [
    "EncodedBatch",
    "ModelBundle",
    "PointExecutor",
    "PytorchHooksEngine",
    "ResolvedSite",
    "encode",
    "load_model",
    "resolve_position",
    "resolve_site",
]
