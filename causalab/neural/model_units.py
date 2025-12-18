"""
model_units.py
==============
Abstractions for locating intervention points and features inside a transformer
model. An **AtomicModelUnit** specifies *where* in the network (layer, component
type, indices) to intervene and *how* to access a particular representation space
(via a Featurizer).

This file introduces:

* `ComponentIndexer` – a callable that yields dynamic indices, e.g., token
  positions in a transformer.
* `AtomicModelUnit` – specifies location (layer, component_type, indices) and
  feature space (Featurizer + optional feature-subset).
* `InterchangeTarget` – container for grouped AtomicModelUnits that share
  counterfactual inputs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Iterator

from causalab.neural.featurizers import Featurizer

# Version identifier for model unit serialization format
MODEL_UNIT_VERSION = "2.0"


class ComponentIndexer:
    """Callable wrapper that returns location indices for a given *input*.
    This is used to specify the *where* in the model to intervene, e.g.,
    the *input* might be a batch of tokenized text with indices that are
    the positions of the tokens to be intervened upon.
    """

    def __init__(self, indexer: Callable[..., list[int]], id: str = "null") -> None:
        """
        Parameters
        ----------
        indexer :
            A function `input -> List[int]` returning the indices.
        id :
            Human-readable identifier for diagnostics / printing.
        """
        self.indexer = indexer
        self.id = id

    # ------------------------------------------------------------------ #
    def index(
        self, input: Any, batch: bool = False, is_original: bool | None = None
    ) -> list[int] | list[list[int]]:
        """Return indices for *input* by delegating to wrapped function.

        Parameters
        ----------
        input :
            The input to index.
        batch : bool
            Whether to process a batch of inputs.
        is_original : bool or None
            Whether this is an original input (True) or counterfactual (False).
            Only passed to the indexer function if not None.
        """
        if batch:
            return [self._call_indexer(i, is_original) for i in input]
        return self._call_indexer(input, is_original)

    def _call_indexer(self, input: Any, is_original: bool | None) -> list[int]:
        """Call the indexer function with is_original parameter only if it's not None."""
        if is_original is not None:
            try:
                # Try calling with is_original parameter
                return self.indexer(input, is_original=is_original)
            except TypeError:
                # Fallback for indexers that don't accept is_original
                return self.indexer(input)
        else:
            # is_original is None, don't pass it
            return self.indexer(input)

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return f"ComponentIndexer(id='{self.id}')"


class InterchangeTarget:
    """A container for lists of lists of AtomicModelUnit instances.

    This class wraps a List[List[AtomicModelUnit]] structure and provides:
    1. List-like indexing and iteration
    2. Delegation of AtomicModelUnit methods to all contained units

    The nested structure represents:
    - Outer list: Groups of units sharing the same counterfactual input
    - Inner list: Individual model units to be intervened upon

    When calling AtomicModelUnit methods (save, load, set_featurizer, etc.),
    they are automatically applied to all units in the structure.
    """

    def __init__(self, model_units_list: list[list[AtomicModelUnit]]) -> None:
        """
        Parameters
        ----------
        model_units_list :
            Nested list structure of AtomicModelUnit instances.
        """
        self.model_units_list = model_units_list

    # -------------------- List-like interface ------------------------ #
    def __getitem__(self, index: int) -> list[AtomicModelUnit]:
        """Support indexing: target[0] returns first group."""
        return self.model_units_list[index]

    def __setitem__(self, index: int, value: list[AtomicModelUnit]) -> None:
        """Support assignment: target[0] = new_group."""
        self.model_units_list[index] = value

    def __len__(self) -> int:
        """Return number of groups."""
        return len(self.model_units_list)

    def __iter__(self) -> Iterator[list[AtomicModelUnit]]:
        """Support iteration: for group in target: ..."""
        return iter(self.model_units_list)

    def __repr__(self) -> str:
        total_units = sum(len(group) for group in self.model_units_list)
        return (
            f"InterchangeTarget("
            f"{len(self.model_units_list)} groups, "
            f"{total_units} total units)"
        )

    # -------------------- Utility helpers ---------------------------- #
    def flatten(self) -> list[AtomicModelUnit]:
        """Return a flat list of all units across all groups."""
        return [unit for group in self.model_units_list for unit in group]

    def nest_to_match(self, flat_list: list[Any]) -> list[list[Any]]:
        """Reshape a flat list to match this InterchangeTarget's nested structure.

        Takes an arbitrary flat list and reshapes it to have the same nested
        structure as this InterchangeTarget (same group sizes).

        Args:
            flat_list: Flat list of items to reshape

        Returns:
            Nested list with the same structure as self.model_units_list

        Raises:
            ValueError: If flat_list length doesn't match total number of units

        Example:
            >>> target = InterchangeTarget([[unit1, unit2], [unit3, unit4, unit5]])
            >>> features = [feat1, feat2, feat3, feat4, feat5]
            >>> nested = target.nest_to_match(features)
            >>> # Returns: [[feat1, feat2], [feat3, feat4, feat5]]
        """
        # Calculate expected length
        expected_len = sum(len(group) for group in self.model_units_list)
        if len(flat_list) != expected_len:
            raise ValueError(
                f"Length of flat_list ({len(flat_list)}) must match "
                f"total number of units ({expected_len})"
            )

        # Reshape according to group sizes
        nested_list = []
        start = 0
        for group in self.model_units_list:
            group_size = len(group)
            nested_list.append(flat_list[start : start + group_size])
            start += group_size

        return nested_list

    # -------------------- Delegated AtomicModelUnit methods ---------- #
    def save(self, parent_dir: str) -> list[str]:
        """Save all units to subdirectories.

        Args:
            parent_dir: Parent directory for saving all units

        Returns:
            List of paths to created unit directories
        """
        paths = []
        for unit in self.flatten():
            unit_dir = unit.save(parent_dir)
            paths.append(unit_dir)
        return paths

    def load(self, parent_dir: str, ignore_version_mismatch: bool = False) -> None:
        """Load all units from their subdirectories.

        Assumes each unit's subdirectory is named after its ID:
        {parent_dir}/{unit.id}/

        Args:
            parent_dir: Parent directory containing unit subdirectories
            ignore_version_mismatch: If True, skip version validation for all units.
                If False (default), raise an exception when loading from an
                incompatible version.

        Returns:
            None (modifies all units in place)
        """
        for unit in self.flatten():
            unit_dir = os.path.join(parent_dir, unit.id)
            unit.load(unit_dir, ignore_version_mismatch=ignore_version_mismatch)

    def set_featurizer(self, featurizer: Featurizer) -> None:
        """Set the same featurizer on all units.

        Args:
            featurizer: Featurizer to set on all units
        """
        for unit in self.flatten():
            unit.set_featurizer(featurizer)

    def set_feature_indices(self, indices: list[int] | None) -> None:
        """Set the same feature indices on all units.

        Args:
            indices: Feature indices to set on all units
        """
        for unit in self.flatten():
            unit.set_feature_indices(indices)

    def get_feature_indices(self) -> dict[str, list[int] | None]:
        """Get feature indices for all units as a dictionary.

        Returns:
            Dictionary mapping unit IDs to their feature indices
        """
        return {unit.id: unit.get_feature_indices() for unit in self.flatten()}


class AtomicModelUnit:
    """A basic unit of intervention specifying location and feature space.

    This is the basic unit of intervention in this library.
    It specifies a *location* in the model (layer, component_type, indices)
    and a *feature space* (Featurizer) to be used for intervention.
    The `feature_indices` are an optional subset of the features
    returned by the featurizer.  If not specified, all features
    are used.

    The `shape` is reserved for downstream sub-space
    creation helpers.  The `id` is a human-readable identifier
    for the unit, used for diagnostics and printing.
    """

    def __init__(
        self,
        layer: int,
        component_type: str,
        indices_func: ComponentIndexer | list[int],
        unit: str = "pos",
        featurizer: Featurizer | None = None,
        feature_indices: list[int] | None = None,
        shape: tuple[int, ...] | None = None,
        *,
        id: str = "null",
    ) -> None:
        """
        Parameters
        ----------
        layer :
            Layer number in the *IntervenableModel*.
        component_type :
            E.g. 'block_input', 'mlp', 'head_attention_value_output', etc.
        indices_func :
            Either a `ComponentIndexer` **or** a static list of indices.
        unit :
            String describing the dimension being indexed (e.g. 'pos' or
            'h.pos' for attention head · token position).
        featurizer :
            A `Featurizer` (defaults to identity featurizer).
        feature_indices :
            Optional subset of indices inside the featurizer's feature vector.
            Will be bounds-checked against `featurizer.n_features`.
        shape :
            Reserved for downstream sub-space creation helpers.
        id :
            Diagnostic identifier.
        """
        self.id = id
        self.layer = layer
        self.component_type = component_type
        self.unit = unit

        # Normalise to an indexer
        if isinstance(indices_func, list):
            constant = indices_func
            self._static_indices = True

            def _const(_):
                return constant

            self._indices_func = ComponentIndexer(_const, id=f"constant_{constant}")
        else:
            self._static_indices = False
            self._indices_func = indices_func

        self.featurizer = featurizer or Featurizer()
        self.shape = shape

        # Bounds-check feature indices
        self.feature_indices = None
        if feature_indices is not None:
            self.set_feature_indices(feature_indices)

    # ------------------------ Feature helpers -------------------------- #
    def get_shape(self) -> tuple[int, ...] | None:
        return self.shape

    def get_index_id(self) -> str:
        """Return the identifier of the indexer function."""
        return self._indices_func.id

    def index_component(
        self, input: Any, **kwargs: Any
    ) -> list[int] | list[list[int]] | list[list[list[int]]]:
        """Return indices for *input* by delegating to wrapped function."""
        return self._indices_func.index(input, **kwargs)

    def get_feature_indices(self) -> list[int] | None:
        return self.feature_indices

    def set_feature_indices(self, feature_indices: list[int] | None) -> None:
        """Assign `feature_indices` after validating bounds."""
        if (
            self.featurizer.n_features is not None
            and feature_indices is not None
            and len(feature_indices) > 0
            and max(feature_indices) >= self.featurizer.n_features
        ):
            raise ValueError(
                f"Feature index {max(feature_indices)} exceeds "
                f"featurizer dimensionality {self.featurizer.n_features}"
            )
        self.feature_indices = feature_indices

    def set_featurizer(self, featurizer: Featurizer) -> None:
        """Swap in a new featurizer (re-checking feature bounds)."""
        self.featurizer = featurizer
        if self.feature_indices is not None:
            self.set_feature_indices(self.feature_indices)

    # ------------------------ PyVENE helpers -------------------------- #
    def is_static(self) -> bool:
        """Return whether this unit uses static indices."""
        return self._static_indices

    def create_intervention_config(
        self, group_key: str | int, intervention_type: str
    ) -> dict[str, Any]:
        """Return PyVENE config dict for this unit + featurizer."""
        config: dict[str, Any] = {
            "component": self.component_type,
            "unit": self.unit,
            "layer": self.layer,
            "group_key": group_key,
        }
        if intervention_type == "interchange":
            config["intervention_type"] = self.featurizer.get_interchange_intervention()
        elif intervention_type == "collect":
            config["intervention_type"] = self.featurizer.get_collect_intervention()
        elif intervention_type == "mask":
            config["intervention_type"] = self.featurizer.get_mask_intervention()
        else:
            raise ValueError(f"Unknown intervention type '{intervention_type}'.")

        return config

    def __repr__(self) -> str:
        return f"AtomicModelUnit(id='{self.id}')"

    # ---------------- Utility & misc ----------------------------------- #
    def set_layer(self, layer: int) -> None:
        """Set the layer number."""
        self.layer = layer

    def get_layer(self) -> int:
        """Get the layer number."""
        return self.layer

    # ---------------- Serialization ------------------------------------ #
    def save(self, parent_dir: str) -> str:
        """Save this model unit to a new subdirectory.

        Creates a directory structure:
            {parent_dir}/{self.id}/
                ├── metadata.json          # Unit configuration and indices
                ├── featurizer             # Featurizer module state
                └── inverse_featurizer     # Inverse featurizer module state

        Args:
            parent_dir: Parent directory for saving

        Returns:
            Path to the created unit directory
        """
        # Create unit-specific directory
        unit_dir = os.path.join(parent_dir, self.id)
        os.makedirs(unit_dir, exist_ok=True)

        # Prepare metadata
        metadata = {
            "id": self.id,
            "feature_indices": (
                [int(i) for i in self.feature_indices]
                if self.feature_indices is not None
                else None
            ),
            "layer": self.layer,
            "component_type": self.component_type,
            "unit": self.unit,
            "index_id": self.get_index_id(),
            "featurizer_info": {
                "id": self.featurizer.id,
                "n_features": self.featurizer.n_features,
            },
            "shape": self.shape if self.shape is None else list(self.shape),
            "version": MODEL_UNIT_VERSION,
        }

        # Save metadata
        metadata_path = os.path.join(unit_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save featurizer modules using standard names
        filename = os.path.join(unit_dir, "model")
        self.featurizer.save_modules(filename)

        return unit_dir

    def load(self, unit_dir: str, ignore_version_mismatch: bool = False) -> None:
        """Load model unit from a directory.

        Expects directory structure:
            {unit_dir}/
                ├── metadata.json
                ├── model_featurizer
                └── model_inverse_featurizer

        Args:
            unit_dir: Directory containing saved unit
            ignore_version_mismatch: If True, skip version validation. If False
                (default), raise an exception when loading from an incompatible version.

        Returns:
            None (modifies the object in place)

        Raises:
            ValueError: If the saved unit version doesn't match "2.0" and
                ignore_version_mismatch is False.
        """
        # Load metadata
        metadata_path = os.path.join(unit_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Validate version compatibility
        version = metadata.get("version")
        if version != MODEL_UNIT_VERSION and not ignore_version_mismatch:
            raise ValueError(
                f"Loading unit from version {version}, current version is {MODEL_UNIT_VERSION}. "
                "Some features may not work correctly. "
                "Set ignore_version_mismatch=True to load anyway."
            )

        # Load featurizer modules
        model_path = os.path.join(unit_dir, "model")
        featurizer_path = model_path + "_featurizer"
        inverse_featurizer_path = model_path + "_inverse_featurizer"

        if not os.path.exists(featurizer_path) or not os.path.exists(
            inverse_featurizer_path
        ):
            raise FileNotFoundError(
                f"Featurizer files not found in {unit_dir}. "
                f"Expected {featurizer_path} and {inverse_featurizer_path}"
            )

        self.set_featurizer(Featurizer.load_modules(model_path))

        # Load feature indices from metadata
        indices = metadata.get("feature_indices")
        self.set_feature_indices(indices)

        # Update shape if present
        if "shape" in metadata and metadata["shape"] is not None:
            self.shape = tuple(metadata["shape"])
