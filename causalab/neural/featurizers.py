"""
featurizers.py
==============
Utility classes for defining *invertible* feature spaces on top of a model's
hidden-state tensors, together with intervention helpers that operate inside
those spaces.

Key ideas
---------

* **Featurizer** – a lightweight wrapper holding:
    • a forward `featurizer` module that maps a tensor **x → (f, error)**
      where *error* is the reconstruction residual (useful for lossy
      featurizers such as sparse auto-encoders);
    • an `inverse_featurizer` that re-assembles the original space
      **(f, error) → x̂**.

* **Interventions** – three higher-order factory functions build PyVENE
  interventions that work in the featurized space:
    - *interchange*
    - *collect*
    - *mask* (differential binary masking)

All public classes / functions below carry PEP-257-style doc-strings.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import pyvene as pv  # type: ignore[import-untyped]


# --------------------------------------------------------------------------- #
#  Basic identity featurizers                                                 #
# --------------------------------------------------------------------------- #
class IdentityFeaturizerModule(torch.nn.Module):
    """A no-op featurizer: *x → (x, None)*."""

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
        A `torch.nn.Module` mapping **x → (features, error)**.
    inverse_featurizer :
        A `torch.nn.Module` mapping **(features, error) → x̂**.
    n_features :
        Dimensionality of the feature space.  **Required** when you intend to
        build a *mask* intervention; optional otherwise.
    id :
        Human-readable identifier used by `__str__` methods of the generated
        interventions.
    """

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

    # --------------------------------------------------------------------- #
    #  (De)serialisation helpers                                            #
    # --------------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any] | None:
        """Return serializable dict representation, or None if trivial/SAE.

        Returns None for:
        - Identity featurizers (trivial, can be reconstructed)
        - SAE featurizers (loaded from sae_lens, not serializable)

        For DAS/Subspace featurizers, returns the learned rotation matrix.
        """
        featurizer_class = self.featurizer.__class__.__name__

        if featurizer_class == "SAEFeaturizerModule":
            return None

        if self.is_trivial():
            return None

        inverse_featurizer_class = self.inverse_featurizer.__class__.__name__

        # Extra config needed for Subspace featurizers
        additional_config: dict[str, Any] = {}
        if featurizer_class == "SubspaceFeaturizerModule":
            additional_config["rotation_matrix"] = (
                self.featurizer.rotate.weight.detach().clone()  # type: ignore[union-attr]
            )
            additional_config["requires_grad"] = (
                self.featurizer.rotate.weight.requires_grad  # type: ignore[union-attr]
            )

        return {
            "model_info": {
                "featurizer_class": featurizer_class,
                "inverse_featurizer_class": inverse_featurizer_class,
                "n_features": self.n_features,
                "featurizer_id": self.id,
                "additional_config": additional_config,
            },
            "featurizer_state_dict": self.featurizer.state_dict(),
            "inverse_state_dict": self.inverse_featurizer.state_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Featurizer":
        """Reconstruct Featurizer from dict (inverse of to_dict)."""
        model_info = data["model_info"]
        featurizer_class = model_info["featurizer_class"]

        if featurizer_class == "SubspaceFeaturizerModule":
            rot = model_info["additional_config"]["rotation_matrix"]
            requires_grad = model_info["additional_config"]["requires_grad"]

            in_dim, out_dim = rot.shape
            rotate_layer = pv.models.layers.LowRankRotateLayer(  # type: ignore[attr-defined]
                in_dim, out_dim, init_orth=False
            )
            rotate_layer.weight.data.copy_(rot)
            rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
            rotate_layer.requires_grad_(requires_grad)

            featurizer = SubspaceFeaturizerModule(rotate_layer)
            inverse = SubspaceInverseFeaturizerModule(rotate_layer)
        elif featurizer_class == "IdentityFeaturizerModule":
            featurizer = IdentityFeaturizerModule()
            inverse = IdentityInverseFeaturizerModule()
        else:
            raise ValueError(f"Unknown featurizer class '{featurizer_class}'.")

        featurizer.load_state_dict(data["featurizer_state_dict"])
        inverse.load_state_dict(data["inverse_state_dict"])

        return cls(
            featurizer,
            inverse,
            n_features=model_info["n_features"],
            id=model_info.get("featurizer_id", "loaded"),
        )

    def save_modules(self, path: str) -> tuple[str | None, str | None]:
        """Serialise featurizer & inverse to `<path>_{featurizer, inverse}`.

        Notes
        -----
        * **SAE featurizers** are *not* serialisable: a
          :class:`NotImplementedError` is raised.
        * Existing files will be *silently overwritten*.
        """
        featurizer_class = self.featurizer.__class__.__name__

        if featurizer_class == "SAEFeaturizerModule":
            # SAE featurizers are to be loaded from sae_lens
            return None, None

        inverse_featurizer_class = self.inverse_featurizer.__class__.__name__

        # Extra config needed for Subspace featurizers
        additional_config: dict[str, Any] = {}
        if featurizer_class == "SubspaceFeaturizerModule":
            additional_config["rotation_matrix"] = (
                self.featurizer.rotate.weight.detach().clone()
            )
            additional_config["requires_grad"] = (
                self.featurizer.rotate.weight.requires_grad
            )

        model_info = {
            "featurizer_class": featurizer_class,
            "inverse_featurizer_class": inverse_featurizer_class,
            "n_features": self.n_features,
            "featurizer_id": self.id,
            "additional_config": additional_config,
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
        """Inverse of :meth:`save_modules`.

        Returns
        -------
        Featurizer
            A *new* instance with reconstructed modules and metadata.
        """
        featurizer_data = torch.load(f"{path}_featurizer")
        inverse_data = torch.load(f"{path}_inverse_featurizer")

        model_info = featurizer_data["model_info"]
        featurizer_class = model_info["featurizer_class"]

        if featurizer_class == "SubspaceFeaturizerModule":
            rot = model_info["additional_config"]["rotation_matrix"]
            requires_grad = model_info["additional_config"]["requires_grad"]

            # Re-build a parametrised orthogonal layer with identical shape.
            in_dim, out_dim = rot.shape
            rotate_layer = pv.models.layers.LowRankRotateLayer(
                in_dim, out_dim, init_orth=False
            )
            rotate_layer.weight.data.copy_(rot)
            rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
            rotate_layer.requires_grad_(requires_grad)

            featurizer = SubspaceFeaturizerModule(rotate_layer)
            inverse = SubspaceInverseFeaturizerModule(rotate_layer)

            # Sanity-check weight shape
            assert featurizer.rotate.weight.shape == rot.shape, (
                "Rotation-matrix shape mismatch after deserialisation."
            )
        elif featurizer_class == "IdentityFeaturizerModule":
            featurizer = IdentityFeaturizerModule()
            inverse = IdentityInverseFeaturizerModule()
        else:
            raise ValueError(f"Unknown featurizer class '{featurizer_class}'.")

        featurizer.load_state_dict(featurizer_data["state_dict"])
        inverse.load_state_dict(inverse_data["state_dict"])

        return cls(
            featurizer,
            inverse,
            n_features=model_info["n_features"],
            id=model_info.get("featurizer_id", "loaded"),
        )


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

        def __init__(self, **kwargs):
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
        def forward(self, base, source, subspaces=None):
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
#  Concrete featurizer implementations                                        #
# --------------------------------------------------------------------------- #
class SubspaceFeaturizerModule(torch.nn.Module):
    """Linear projector onto an orthogonal *rotation* sub-space."""

    def __init__(self, rotate_layer: pv.models.layers.LowRankRotateLayer) -> None:
        super().__init__()
        self.rotate = rotate_layer

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r = self.rotate.weight.T  # (out, in)ᵀ
        f = x.to(r.dtype) @ r.T
        error = x - (f @ r).to(x.dtype)
        return f, error


class SubspaceInverseFeaturizerModule(torch.nn.Module):
    """Inverse of :class:`SubspaceFeaturizerModule`."""

    def __init__(self, rotate_layer: pv.models.layers.LowRankRotateLayer) -> None:
        super().__init__()
        self.rotate = rotate_layer

    def forward(self, f: Tensor, error: Tensor) -> Tensor:
        r = self.rotate.weight.T
        return (f.to(r.dtype) @ r).to(f.dtype) + error.to(f.dtype)


class SubspaceFeaturizer(Featurizer):
    """Orthogonal linear sub-space featurizer."""

    def __init__(
        self,
        *,
        shape: tuple[int, int] | None = None,
        rotation_subspace: Tensor | None = None,
        trainable: bool = True,
        id: str = "subspace",
        **kwargs: Any,
    ) -> None:
        assert shape is not None or rotation_subspace is not None, (
            "Provide either `shape` or `rotation_subspace`."
        )

        if shape is not None:
            rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=True)
        else:
            shape = rotation_subspace.shape
            rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=False)
            rotate.weight.data.copy_(rotation_subspace)

        rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
        rotate.requires_grad_(trainable)

        super().__init__(
            SubspaceFeaturizerModule(rotate),
            SubspaceInverseFeaturizerModule(rotate),
            n_features=rotate.weight.shape[1],
            id=id,
            **kwargs,
        )


class SAEFeaturizerModule(torch.nn.Module):
    """Wrapper around a *Sparse Autoencoder*'s encode() / decode() pair."""

    def __init__(self, sae: Any) -> None:
        super().__init__()
        self.sae = sae

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.sae.encode(x.to(self.sae.dtype))
        error = x - self.sae.decode(features).to(x.dtype)
        return features.to(x.dtype), error


class SAEInverseFeaturizerModule(torch.nn.Module):
    """Inverse for :class:`SAEFeaturizerModule`."""

    def __init__(self, sae: Any) -> None:
        super().__init__()
        self.sae = sae

    def forward(self, features: Tensor, error: Tensor) -> Tensor:
        return self.sae.decode(features.to(self.sae.dtype)).to(
            features.dtype
        ) + error.to(features.dtype)


# currently unused but not dead code - usage to come
class SAEFeaturizer(Featurizer):
    """Featurizer backed by a pre-trained sparse auto-encoder.

    Notes
    -----
    Serialisation is *disabled* for SAE featurizers – saving will raise
    ``NotImplementedError``.
    """

    def __init__(self, sae: Any, *, trainable: bool = False, **kwargs: Any) -> None:
        sae.requires_grad_(trainable)
        super().__init__(
            SAEFeaturizerModule(sae),
            SAEInverseFeaturizerModule(sae),
            n_features=sae.cfg.to_dict()["d_sae"],
            id="sae",
            **kwargs,
        )


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


# currently unused but not dead code - usage to come
def build_SVD_featurizers(
    model_units: list[Any],
    svd_results: dict[str, dict[str, Any]],
    trainable: bool = False,
    featurizer_id: str = "SVD",
) -> None:
    """
    Build and set SVD featurizers on model units from pre-computed SVD results.

    This function takes SVD results from compute_svd() and creates SubspaceFeaturizer
    instances for each model unit, setting them in place. The featurizers use the
    SVD rotation matrices to project activations into the principal component space.

    Args:
        model_units (List[AtomicModelUnit]): List of model units to set featurizers on.
                                             Each unit's ID must exist in svd_results.
        svd_results (Dict[str, Dict]): Dictionary from compute_svd() mapping unit IDs
                                       to SVD results. Each result must contain a
                                       "rotation" key with a torch tensor.
        trainable (bool): Whether the featurizers should be trainable (default: False).
                         If True, the rotation matrices will have requires_grad=True.
        featurizer_id (str): Identifier string for the featurizers (default: "SVD").
                            Useful for distinguishing between different featurizer types.

    Returns:
        None: Modifies model_units in place by calling set_featurizer() and
              set_feature_indices(None) on each unit.

    Raises:
        ValueError: If a model unit's ID is not found in svd_results.

    Example:
        >>> from causalab.experiments.pyvene_core import collect_features
        >>> from causalab.neural.collect import compute_svd
        >>> from causalab.neural.featurizers import build_SVD_featurizers
        >>>
        >>> # Collect features
        >>> features_dict = collect_features(dataset, pipeline, model_units)
        >>>
        >>> # Compute SVD
        >>> svd_results = compute_svd(features_dict, n_components=10, normalize=True)
        >>>
        >>> # Build and set featurizers
        >>> build_SVD_featurizers(model_units, svd_results, trainable=False)
        >>>
        >>> # Now all model_units have SVD featurizers set
    """
    for model_unit in model_units:
        # Check that SVD results exist for this unit
        if model_unit.id not in svd_results:
            raise ValueError(
                f"Model unit '{model_unit.id}' not found in svd_results. "
                f"Available IDs: {list(svd_results.keys())}"
            )

        # Get SVD results for this unit
        result = svd_results[model_unit.id]
        rotation = result["rotation"]

        # Create SubspaceFeaturizer with the rotation matrix
        featurizer = SubspaceFeaturizer(
            rotation_subspace=rotation,
            trainable=trainable,
            id=featurizer_id,
        )

        # Set featurizer on the model unit
        model_unit.set_featurizer(featurizer)

        # Use all components (no feature selection)
        model_unit.set_feature_indices(None)
