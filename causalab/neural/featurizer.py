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
import pyvene as pv  # type: ignore[import-untyped]

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
    # should dispatch to them during ``from_dict`` / ``load_modules``.
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

    # -------------------- Intervention builders -------------------------- #
    def get_interchange_intervention(self) -> type[pv.TrainableIntervention]:
        if not hasattr(self, "_interchange_intervention"):
            self._interchange_intervention = build_feature_interchange_intervention(
                self.featurizer, self.inverse_featurizer, self.id
            )
        return self._interchange_intervention

    def get_collect_intervention(self) -> type[pv.CollectIntervention]:
        if not hasattr(self, "_collect_intervention"):
            self._collect_intervention = build_feature_collect_intervention(
                self.featurizer, self.id
            )
        return self._collect_intervention

    def get_mask_intervention(self) -> type[pv.TrainableIntervention]:
        if self.n_features is None:
            raise ValueError(
                "`n_features` must be provided on the Featurizer "
                "to construct a mask intervention."
            )
        if not hasattr(self, "_mask_intervention"):
            self._mask_intervention = build_feature_mask_intervention(
                self.featurizer,
                self.inverse_featurizer,
                self.n_features,
                self.id,
                self.tie_masks,
            )
        return self._mask_intervention

    def get_steering_intervention(self) -> type[pv.TrainableIntervention]:
        """Get intervention class for steering in this feature space."""
        if not hasattr(self, "_steering_intervention"):
            self._steering_intervention = build_feature_steering_intervention(
                self.featurizer, self.inverse_featurizer, self.id
            )
        return self._steering_intervention

    def get_replace_intervention(self) -> type[pv.TrainableIntervention]:
        """Get intervention class for feature replacement in this feature space."""
        if not hasattr(self, "_replace_intervention"):
            self._replace_intervention = build_feature_replace_intervention(
                self.featurizer, self.inverse_featurizer, self.id
            )
        return self._replace_intervention

    def get_interpolation_intervention(self) -> type[pv.TrainableIntervention]:
        """Get intervention class for arbitrary interpolation in this feature space."""
        if not hasattr(self, "_interpolation_intervention"):
            self._interpolation_intervention = build_feature_interpolation_intervention(
                self.featurizer, self.inverse_featurizer, self.id
            )
        return self._interpolation_intervention

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

    def save_modules(self, path: str) -> tuple[str | None, str | None]:
        """Serialise featurizer & inverse to ``<path>_{featurizer, inverse_featurizer}``.

        The base implementation handles identity/trivial featurizers. Concrete
        subclasses in ``causalab.methods`` override this when they need extra
        config (e.g. rotation matrix for Subspace).
        """
        featurizer_class = self.featurizer.__class__.__name__
        inverse_featurizer_class = self.inverse_featurizer.__class__.__name__

        model_info = {
            "featurizer_class": featurizer_class,
            "inverse_featurizer_class": inverse_featurizer_class,
            "n_features": self.n_features,
            "featurizer_id": self.id,
            "tie_masks": self.tie_masks,
            "additional_config": {},
        }
        torch.save(
            {"model_info": model_info, "state_dict": self.featurizer.state_dict()},
            f"{path}_featurizer",
        )
        torch.save(
            {
                "model_info": model_info,
                "state_dict": self.inverse_featurizer.state_dict(),
            },
            f"{path}_inverse_featurizer",
        )
        return f"{path}_featurizer", f"{path}_inverse_featurizer"

    @classmethod
    def load_modules(cls, path: str) -> Featurizer:
        """Inverse of :meth:`save_modules` — dispatches to subclass by module-class name."""
        featurizer_data = torch.load(f"{path}_featurizer")
        inverse_data = torch.load(f"{path}_inverse_featurizer")

        model_info = featurizer_data["model_info"]
        featurizer_class = model_info["featurizer_class"]

        if featurizer_class == "IdentityFeaturizerModule":
            featurizer = IdentityFeaturizerModule()
            inverse = IdentityInverseFeaturizerModule()
            featurizer.load_state_dict(featurizer_data["state_dict"])
            inverse.load_state_dict(inverse_data["state_dict"])
            return cls(
                featurizer,
                inverse,
                n_features=model_info["n_features"],
                id=model_info.get("featurizer_id", "null"),
                tie_masks=model_info.get("tie_masks", False),
            )

        subclass = _find_subclass_for(featurizer_class)
        if subclass is None:
            raise ValueError(f"Unknown featurizer class '{featurizer_class}'.")
        return subclass.load_modules(path)


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


# --------------------------------------------------------------------------- #
#  Intervention factory helpers                                               #
# --------------------------------------------------------------------------- #
# NOTE: These functions return *classes*, not instances. This is intentional.
#
# Pyvene's IntervenableConfig expects intervention_type to be a class that it
# will instantiate itself (passing its own kwargs like embed_dim, etc.). We
# can't pass an instance because pyvene tries to call the constructor.
#
# The problem: our interventions need custom config (featurizer, inverse,
# n_features) that pyvene doesn't know about. Solution: dynamically create a
# class that captures these values in a closure, so the __init__ pyvene calls
# doesn't need them as arguments.
#
# This is a "type crime" - the dynamic class creation makes static typing
# essentially impossible. The pyright: ignore comments in interchange.py are
# the pragmatic consequence.
# --------------------------------------------------------------------------- #
def build_feature_interchange_intervention(
    featurizer: torch.nn.Module,
    inverse_featurizer: torch.nn.Module,
    featurizer_id: str,
) -> type[pv.TrainableIntervention]:
    """Return a class implementing PyVENE's TrainableIntervention."""

    class FeatureInterchangeIntervention(
        pv.TrainableIntervention, pv.DistributedRepresentationIntervention
    ):
        """Swap features between *base* and *source* in the featurized space."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._featurizer = featurizer
            self._inverse = inverse_featurizer

        def forward(self, base, source, subspaces=None):
            f_base, base_err = self._featurizer(base)
            f_src, _ = self._featurizer(source)

            if subspaces is None or _subspace_is_all_none(subspaces):
                f_out = f_src
            else:
                f_out = pv.models.intervention_utils._do_intervention_by_swap(
                    f_base,
                    f_src,
                    "interchange",
                    self.interchange_dim,
                    subspaces,
                    subspace_partition=self.subspace_partition,
                    use_fast=self.use_fast,
                )
            return self._inverse(f_out, base_err).to(base.dtype)

        def __str__(self):
            return f"FeatureInterchangeIntervention(id={featurizer_id})"

    return FeatureInterchangeIntervention


def build_feature_interpolation_intervention(
    featurizer: torch.nn.Module,
    inverse_featurizer: torch.nn.Module,
    featurizer_id: str,
) -> type[pv.TrainableIntervention]:
    """Return a class implementing an arbitrary interpolation intervention in feature space.

    The interpolation function and its keyword parameters are set via
    set_interpolation() after instantiation (same pattern as set_temperature
    in FeatureMaskIntervention). pyvene's forward signature is fixed to
    (base, source, subspaces=None) -- extra params cannot be passed there.

    Example:
        def linear_interp(f_base, f_src, alpha):
            return (1 - alpha) * f_base + alpha * f_src

        intervention.set_interpolation(linear_interp, alpha=0.5)
    """

    class FeatureInterpolateIntervention(
        pv.TrainableIntervention, pv.DistributedRepresentationIntervention
    ):
        """Patch an interpolation of base and source features into the model."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._featurizer = featurizer
            self._inverse_featurizer = inverse_featurizer
            self._interpolation_function = None
            self._interpolation_params: dict[str, Any] = {}

        def set_interpolation(self, fn: Any, **params: Any) -> None:
            self._interpolation_function = fn
            self._interpolation_params = params

        def forward(
            self, base: torch.Tensor, source: torch.Tensor, subspaces: Any = None
        ):
            if self._interpolation_function is None:
                raise ValueError(
                    "Interpolation function not set. Call set_interpolation() first."
                )
            f_base, base_err = self._featurizer(base)
            f_src, _ = self._featurizer(source)
            f_out = self._interpolation_function(
                f_base=f_base, f_src=f_src, **self._interpolation_params
            )
            return self._inverse_featurizer(f_out, base_err).to(base.dtype)

        def __str__(self) -> str:
            return f"FeatureInterpolateIntervention(id={featurizer_id})"

    return FeatureInterpolateIntervention


def build_feature_collect_intervention(
    featurizer: torch.nn.Module, featurizer_id: str
) -> type[pv.CollectIntervention]:
    """Return a `CollectIntervention` operating in feature space."""

    class FeatureCollectIntervention(pv.CollectIntervention):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._featurizer = featurizer

        def forward(self, base, source=None, subspaces=None):
            f_base, _ = self._featurizer(base)
            return pv.models.intervention_utils._do_intervention_by_swap(
                f_base,
                source,
                "collect",
                self.interchange_dim,
                subspaces,
                subspace_partition=self.subspace_partition,
                use_fast=self.use_fast,
            )

        def __str__(self):
            return f"FeatureCollectIntervention(id={featurizer_id})"

    return FeatureCollectIntervention


def build_feature_steering_intervention(
    featurizer: torch.nn.Module,
    inverse_featurizer: torch.nn.Module,
    featurizer_id: str,
) -> type[pv.TrainableIntervention]:
    """Return a class implementing steering intervention in feature space.

    The intervention:
    1. Featurizes base activation to get (base_features, base_error)
    2. Adds steering vector to base_features: steered = base_features + steering_vector
    3. Reconstructs via inverse_featurizer(steered, base_error)

    Args:
        featurizer: Module mapping activation -> (features, error)
        inverse_featurizer: Module mapping (features, error) -> activation
        featurizer_id: Human-readable identifier

    Returns:
        A pyvene intervention class

    Note:
        The steering vector is passed via pyvene's `source` parameter during
        the forward pass. Unlike interchange interventions, this source is
        a pre-computed tensor, not activations from a model forward pass.
    """

    class FeatureSteeringIntervention(
        pv.TrainableIntervention, pv.DistributedRepresentationIntervention
    ):
        """Add steering vectors to base activations in the featurized space."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._featurizer = featurizer
            self._inverse = inverse_featurizer

        def forward(self, base, source, subspaces=None):
            # source is the steering vector (already in feature space)
            steering_vector = source

            # Featurize base to get features and error term
            base_features, base_error = self._featurizer(base)

            # Add steering vector to base features
            steered_features = base_features + steering_vector.to(base_features.dtype)

            # Reconstruct, preserving orthogonal component
            return self._inverse(steered_features, base_error).to(base.dtype)

        def __str__(self):
            return f"FeatureSteeringIntervention(id={featurizer_id})"

    return FeatureSteeringIntervention


def build_feature_replace_intervention(
    featurizer: torch.nn.Module,
    inverse_featurizer: torch.nn.Module,
    featurizer_id: str,
) -> type[pv.TrainableIntervention]:
    """Return a class implementing feature replacement in feature space.

    Unlike steering (add), this REPLACES base features with the source vector.
    The error term (orthogonal component) is still preserved from base.

    Order of operations:
    1. Run featurizer on base to get (features, error)
    2. Replace features with the replacement vector (ignore base features)
    3. Reconstruct with replacement features + original error

    Args:
        featurizer: Module mapping activation -> (features, error)
        inverse_featurizer: Module mapping (features, error) -> activation
        featurizer_id: Human-readable identifier

    Returns:
        A pyvene intervention class

    Note:
        When replacement_vector = 0, this zeros out the feature contribution
        while preserving the orthogonal component (reconstruction error).
        This is useful for ablation studies measuring feature importance.
    """

    class FeatureReplaceIntervention(
        pv.TrainableIntervention, pv.DistributedRepresentationIntervention
    ):
        """Replace base features with source vector in the featurized space."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._featurizer = featurizer
            self._inverse = inverse_featurizer

        def forward(
            self, base: torch.Tensor, source: torch.Tensor, subspaces: Any = None
        ):
            replacement_vector = source

            # 1. Run featurizer on base to get error term
            _, base_error = self._featurizer(base)

            # 2. Replace features entirely with the replacement vector
            replaced_features = replacement_vector.to(base.dtype)

            # 3. Reconstruct with replacement features but original error
            return self._inverse(replaced_features, base_error)

        def __str__(self):
            return f"FeatureReplaceIntervention(id={featurizer_id})"

    return FeatureReplaceIntervention


def build_feature_mask_intervention(
    featurizer: torch.nn.Module,
    inverse_featurizer: torch.nn.Module,
    n_features: int,
    featurizer_id: str,
    tie_masks: bool = False,
) -> type[pv.TrainableIntervention]:
    """Return a trainable mask intervention.

    Args:
        featurizer: The featurizer module
        inverse_featurizer: The inverse featurizer module
        n_features: Number of features in the featurized space
        featurizer_id: Human-readable identifier for the intervention
        tie_masks: If True, use a single scalar mask weight for all features.
                  If False, use per-feature mask weights (default behavior).
    """

    class FeatureMaskIntervention(pv.TrainableIntervention):
        """Differential-binary masking in the featurized space."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._featurizer = featurizer
            self._inverse = inverse_featurizer
            self._tie_masks = tie_masks
            self._featurizer_id = featurizer_id

            # Learnable parameters
            if tie_masks:
                # Single scalar mask weight for all features
                self.mask = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
            else:
                # Per-feature mask weights
                self.mask = torch.nn.Parameter(
                    torch.zeros(n_features), requires_grad=True
                )
            self.temperature: Tensor | None = None  # must be set by user

        # -------------------- API helpers -------------------- #
        def get_temperature(self) -> Tensor:
            if self.temperature is None:
                raise ValueError("Temperature has not been set.")
            return self.temperature

        def set_temperature(self, temp: float | Tensor) -> None:
            self.temperature = torch.as_tensor(temp, dtype=self.mask.dtype).to(
                self.mask.device
            )

        # ------------------------- forward ------------------- #
        def forward(  # type: ignore[override]
            self, base: Tensor, source: Tensor, subspaces: Any = None
        ) -> Tensor:
            if self.temperature is None:
                raise ValueError("Cannot run forward without a temperature.")

            f_base, base_err = self._featurizer(base)
            f_src, _ = self._featurizer(source)

            # Align devices / dtypes
            mask = self.mask.to(f_base.device)
            temp = self.temperature.to(f_base.device)

            f_base = f_base.to(mask.dtype)
            f_src = f_src.to(mask.dtype)

            if self.training:
                gate = torch.sigmoid(mask / temp)
            else:
                gate = (torch.sigmoid(mask) > 0.5).float()

            f_out = (1.0 - gate) * f_base + gate * f_src
            return self._inverse(f_out.to(base.dtype), base_err).to(base.dtype)

        # ---------------- Sparsity regulariser --------------- #
        def get_sparsity_loss(self) -> Tensor:
            if self.temperature is None:
                raise ValueError("Temperature has not been set.")
            gate = torch.sigmoid(self.mask / self.temperature)
            return torch.norm(gate, p=1)

        def __str__(self):
            tied_str = ",tied" if self._tie_masks else ""
            return f"FeatureMaskIntervention(id={self._featurizer_id}{tied_str})"

    return FeatureMaskIntervention


# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #
def _subspace_is_all_none(subspaces: list[Any] | None) -> bool:
    """Return ``True`` if *every* element of *subspaces* is ``None``."""
    if subspaces is None:
        return True
    if not subspaces:  # Empty list
        return False
    return all(
        inner is None or (inner and all(elem is None for elem in inner))
        for inner in subspaces
    )
