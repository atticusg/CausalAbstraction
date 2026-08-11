"""
featurizer.py
=============
Base featurizer infrastructure: the ``Featurizer`` wrapper, identity modules,
composed featurizers, and intervention factory helpers.

All concrete featurizer implementations live in ``causalab.methods`` and
override the (de)serialisation hooks defined here. This module must not
import from ``causalab.methods`` at module scope — the base dispatches to
subclasses via ``Featurizer.__subclasses__()`` and lazy-imports
``causalab.methods`` only as a fallback to trigger subclass registration.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

# Type alias for composed errors: list of per-stage errors
ComposedError = list[Tensor | None]


def _iter_all_subclasses(cls: type) -> "list[type]":
    """Recursively walk ``cls.__subclasses__()``."""
    found: list[type] = []
    stack = list(cls.__subclasses__())
    while stack:
        sub = stack.pop()
        found.append(sub)
        stack.extend(sub.__subclasses__())
    return found


def _find_subclass_for(featurizer_module_class_name: str) -> "type[Featurizer] | None":
    """Return the Featurizer subclass whose FEATURIZER_MODULE_CLASS_NAME matches.

    On first miss, triggers ``import causalab.methods`` so subclass modules are
    registered via import side-effects, then retries.
    """
    for _attempt in range(2):
        for sub in _iter_all_subclasses(Featurizer):
            if sub.FEATURIZER_MODULE_CLASS_NAME == featurizer_module_class_name:
                return sub
        # Lazy-trigger subclass registration via methods package imports.
        try:
            importlib.import_module("causalab.methods")
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
#  Basic identity featurizers                                                 #
# --------------------------------------------------------------------------- #
class IdentityFeaturizerModule(torch.nn.Module):
    """A no-op featurizer: *x -> (x, None)*."""

    def forward(self, x: Tensor) -> tuple[Tensor, None]:
        return x, None


class IdentityInverseFeaturizerModule(torch.nn.Module):
    """Inverse of :class:`IdentityFeaturizerModule`."""

    def forward(self, x: Tensor, error: None) -> Tensor:
        return x


# --------------------------------------------------------------------------- #
#  High-level Featurizer wrapper                                              #
# --------------------------------------------------------------------------- #
class Featurizer:
    """Container object holding paired featurizer and inverse modules.

    Parameters
    ----------
    featurizer :
        A `torch.nn.Module` mapping **x -> (features, error)**.
    inverse_featurizer :
        A `torch.nn.Module` mapping **(features, error) -> x_hat**.
    n_features :
        Dimensionality of the feature space.  **Required** when you intend to
        build a *mask* intervention; optional otherwise.
    id :
        Human-readable identifier used by `__str__` methods of the generated
        interventions.
    """

    # Declared for type checker - set by subclasses or dynamically
    _trainable: bool | None = None
    _trainable_das: bool | None = None
    _trainable_flow: bool | None = None
    _manifold: Any = None
    fitted_radius: float | None = None

    # Subclasses set this to the ``featurizer.__class__.__name__`` value that
    # should dispatch to them during ``from_dict``.
    FEATURIZER_MODULE_CLASS_NAME: str | None = None

    # --------------------------------------------------------------------- #
    #  Construction / public accessors                                      #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        featurizer: torch.nn.Module | None = None,
        inverse_featurizer: torch.nn.Module | None = None,
        *,
        n_features: int | None = None,
        id: str = "null",
        tie_masks: bool = False,
    ) -> None:
        self.featurizer = (
            featurizer if featurizer is not None else IdentityFeaturizerModule()
        )
        self.inverse_featurizer = (
            inverse_featurizer
            if inverse_featurizer is not None
            else IdentityInverseFeaturizerModule()
        )
        self.n_features = n_features
        self.id = id
        self.tie_masks = tie_masks

    # ------------------------- Convenience I/O --------------------------- #
    def is_trivial(self) -> bool:
        """Return True if this is an identity featurizer with no learned weights.

        Trivial featurizers don't need to be serialized - they can be
        reconstructed from just knowing they're identity.

        Uses the id="null" convention: identity featurizers have id="null",
        while learned featurizers have descriptive ids like "subspace", "sae".
        """
        return self.id == "null"

    def featurize(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        return self.featurizer(x)

    def inverse_featurize(self, x: Tensor, error: Tensor | None) -> Tensor:
        return self.inverse_featurizer(x, error)

    # -------------------- Composition operator -------------------------- #
    def __rshift__(self, other: "Featurizer") -> "ComposedFeaturizer":
        """Compose featurizers: self >> other means self first, then other.

        Returns a ComposedFeaturizer that chains the stages with per-stage
        error preservation. Flattens nested compositions for associativity.

        Example:
            das = SubspaceFeaturizer(rotation_subspace=rot)
            standardize = StandardizeFeaturizer(mean, std)
            manifold_feat = ManifoldFeaturizer(manifold)
            composed = das >> standardize >> manifold_feat
        """
        # Flatten existing compositions for associativity
        self_stages = self.stages if isinstance(self, ComposedFeaturizer) else [self]
        other_stages = (
            other.stages if isinstance(other, ComposedFeaturizer) else [other]
        )
        return ComposedFeaturizer(self_stages + other_stages)

    # --------------------------------------------------------------------- #
    #  (De)serialisation helpers                                            #
    # --------------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any] | None:
        """Serialize to dict. Trivial featurizers return None.

        Concrete subclasses in ``causalab.methods`` override this. The base
        returns None for trivial (id="null") featurizers, serializes named
        identity-module featurizers (e.g. mask interventions), and raises for
        any non-trivial featurizer that hasn't overridden this method.
        """
        if self.is_trivial():
            return None
        if isinstance(self.featurizer, IdentityFeaturizerModule):
            return {
                "model_info": {
                    "featurizer_class": "IdentityFeaturizerModule",
                    "inverse_featurizer_class": "IdentityInverseFeaturizerModule",
                    "n_features": self.n_features,
                    "featurizer_id": self.id,
                    "tie_masks": self.tie_masks,
                },
                "featurizer_state_dict": self.featurizer.state_dict(),
                "inverse_state_dict": self.inverse_featurizer.state_dict(),
            }
        raise NotImplementedError(
            f"{type(self).__name__}.to_dict() is not implemented. "
            "Concrete featurizer subclasses must override to_dict()."
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Featurizer":
        """Reconstruct Featurizer from dict by dispatching to the matching subclass."""
        model_info = data["model_info"]
        featurizer_class = model_info["featurizer_class"]

        if featurizer_class == "IdentityFeaturizerModule":
            featurizer = IdentityFeaturizerModule()
            inverse = IdentityInverseFeaturizerModule()
            return cls(
                featurizer,
                inverse,
                n_features=model_info["n_features"],
                id=model_info.get("featurizer_id", "null"),
                tie_masks=model_info.get("tie_masks", False),
            )
        if featurizer_class == "ComposedFeaturizer":
            stages = [cls.from_dict(stage_dict) for stage_dict in data["stages"]]
            return ComposedFeaturizer(
                stages,
                id=model_info.get("featurizer_id"),
            )

        subclass = _find_subclass_for(featurizer_class)
        if subclass is None:
            raise ValueError(f"Unknown featurizer class '{featurizer_class}'.")
        return subclass.from_dict(data)


# --------------------------------------------------------------------------- #
#  Composed Featurizer                                                         #
# --------------------------------------------------------------------------- #
class ComposedFeaturizerModule(torch.nn.Module):
    """Forward module: chains stages, collects per-stage errors."""

    def __init__(self, stages: list[torch.nn.Module]) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> tuple[Tensor, ComposedError]:
        errors: ComposedError = []
        for stage in self.stages:
            x, error = stage(x)
            errors.append(error)
        return x, errors


class ComposedInverseFeaturizerModule(torch.nn.Module):
    """Inverse module: reverses chain, passes each error to its stage."""

    def __init__(self, stages: list[torch.nn.Module]) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor, errors: ComposedError) -> Tensor:
        for stage, error in zip(reversed(self.stages), reversed(errors)):
            x = stage(x, error)
        return x


class ComposedFeaturizer(Featurizer):
    """Chain of featurizers with per-stage error preservation.

    Error type: list[Tensor | None] - one entry per stage.
    Bijective stages contribute None, lossy stages contribute their error.
    Perfect reconstruction requires all errors.

    Example:
        das = SubspaceFeaturizer(rotation_subspace=rot)
        standardize = StandardizeFeaturizer(mean, std)
        manifold_feat = ManifoldFeaturizer(manifold)

        composed = das >> standardize >> manifold_feat
        features, errors = composed.featurize(x)  # errors: [das_err, None, None]
        x_rec = composed.inverse_featurize(features, errors)  # perfect reconstruction
    """

    def __init__(
        self,
        stages: list[Featurizer],
        *,
        id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.stages = stages
        super().__init__(
            featurizer=ComposedFeaturizerModule([s.featurizer for s in stages]),
            inverse_featurizer=ComposedInverseFeaturizerModule(
                [s.inverse_featurizer for s in stages]
            ),
            n_features=stages[-1].n_features if stages else None,
            id=id or " >> ".join(s.id for s in stages),
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any] | None:
        """Serialize composed featurizer as list of stage serializations."""
        # Serialize each stage
        stage_dicts = []
        for stage in self.stages:
            stage_dict = stage.to_dict()
            if stage_dict is None:
                # If any stage can't be serialized (e.g., SAE), we can't serialize
                return None
            stage_dicts.append(stage_dict)

        return {
            "model_info": {
                "featurizer_class": "ComposedFeaturizer",
                "n_features": self.n_features,
                "featurizer_id": self.id,
            },
            "stages": stage_dicts,
        }

    def featurize(  # type: ignore[override]
        self, x: Tensor
    ) -> tuple[Tensor, ComposedError]:
        """Forward pass returning features and list of per-stage errors."""
        return self.featurizer(x)

    def inverse_featurize(  # type: ignore[override]
        self,
        x: Tensor,
        error: ComposedError,  # noqa: N803
    ) -> Tensor:
        """Inverse pass using per-stage errors for reconstruction."""
        return self.inverse_featurizer(x, error)
