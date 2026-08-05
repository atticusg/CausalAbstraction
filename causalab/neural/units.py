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

import inspect
import json
import os
from typing import Any, Callable, Iterator

import torch

from causalab.neural.featurizer import Featurizer

# Version identifier for model unit serialization format
MODEL_UNIT_VERSION = "2.0"


def _indexer_accepts_is_original(indexer: Callable[..., Any]) -> bool:
    """Whether ``indexer`` can be called with an ``is_original=...`` keyword.

    Decided once from the signature — never by catching a ``TypeError`` at call
    time. Catching the ``TypeError`` conflated "the indexer does not take the
    flag" with "the indexer took the flag but a bug inside it raised", silently
    re-running such an indexer *without* the flag; for a paired position that
    returns base positions on a counterfactual read — a wrong-position
    intervention instead of a crash (#430).

    Returns ``True`` when the signature has a ``**kwargs`` catch-all, or a
    parameter named ``is_original`` that is reachable by keyword
    (``POSITIONAL_OR_KEYWORD`` or ``KEYWORD_ONLY``). A *positional-only*
    ``is_original`` cannot be satisfied by the keyword call, so it counts as
    unsupported. Callables whose signature is not introspectable (some C
    builtins) are treated as unsupported.
    """
    try:
        params = inspect.signature(indexer).parameters
    except (TypeError, ValueError):
        return False
    for param in params.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "is_original" and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


# Sentinel for `index_component`'s `attention_mask` default. Lets us
# distinguish "caller omitted the kwarg" (an error for batched-multi calls)
# from "caller explicitly passed None" (intentional opt-out, leave indices
# in each example's unpadded frame). See AtomicModelUnit.index_component.
_ATTENTION_MASK_UNSET: Any = object()


def _shift_row(
    positions: list[int], shift: int, padded_len: int, ex_idx: int
) -> list[int]:
    """Add ``shift`` to every position, bounds-checked against ``padded_len``."""
    shifted = [p + shift for p in positions]
    oob = [p for p in shifted if p < 0 or p >= padded_len]
    if oob:
        raise ValueError(
            f"Position {oob} is out of bounds for padded length {padded_len} "
            f"(example {ex_idx}, per-row shift {shift}). Token-position indices "
            "must fall within the sequence; an out-of-bounds index would "
            "otherwise reach the model's activation as a silent wrong-token "
            "read. Common causes: a position computed for a differently-"
            "shaped input (e.g. a base position reused on a shorter "
            "counterfactual, #176), or — when `pipeline.max_length` is set — "
            "an indexer returning positions already in the padded frame so the "
            "shift pushes them past the end (use `pipeline.max_length=None`)."
        )
    return shifted


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
        # Detected once, at construction, from the signature — see
        # `_indexer_accepts_is_original`. Dispatch reads this flag instead of
        # probing the indexer with a try/except, so a `TypeError` raised inside
        # the indexer propagates as the real bug it is (#430).
        self._accepts_is_original = _indexer_accepts_is_original(indexer)

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
        """Call the wrapped indexer, threading ``is_original`` iff it accepts it.

        Whether the indexer accepts ``is_original`` was decided once at
        construction from its signature (``self._accepts_is_original``); this
        never catches a ``TypeError`` to decide. So a ``TypeError`` raised
        *inside* an ``is_original``-accepting indexer propagates like any other
        bug, instead of being swallowed and the indexer silently re-invoked
        without the flag — which for a paired position would read base positions
        on a counterfactual pass (#430).
        """
        if is_original is not None and self._accepts_is_original:
            return self.indexer(input, is_original=is_original)
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
    def save(self, parent_dir: str) -> str:
        """Save all units to a directory.

        Saves to consolidated format with up to 3 files:
        - units_metadata.json (all unit metadata)
        - featurizers.safetensors / featurizers.meta.json (only if there are
          non-trivial featurizers; written via the safetensors+meta convention)

        Args:
            parent_dir: Directory for saving

        Returns:
            Path to the parent directory
        """
        os.makedirs(parent_dir, exist_ok=True)

        # Consolidated format: 1-2 files total
        # Collect all metadata
        all_metadata: dict[str, dict[str, Any]] = {}
        for unit in self.flatten():
            all_metadata[unit.id] = {
                "id": unit.id,
                "feature_indices": (
                    [int(i) for i in unit.feature_indices]
                    if unit.feature_indices is not None
                    else None
                ),
                "layer": unit.layer,
                "component_type": unit.component_type,
                "unit": unit.unit,
                "index_id": unit.get_index_id(),
                "featurizer_info": {
                    "id": unit.featurizer.id,
                    "n_features": unit.featurizer.n_features,
                    "is_trivial": unit.featurizer.is_trivial(),
                },
                "shape": unit.shape if unit.shape is None else list(unit.shape),
                "version": MODEL_UNIT_VERSION,
            }

        # Save metadata to single JSON file
        metadata_path = os.path.join(parent_dir, "units_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(all_metadata, f, indent=2)

        # Save non-trivial featurizers to single file (if any)
        featurizers: dict[str, dict[str, Any]] = {}
        for unit in self.flatten():
            featurizer_data = unit.featurizer.to_dict()
            if featurizer_data is not None:
                featurizers[unit.id] = featurizer_data

        if featurizers:
            from causalab.io.nested_artifacts import save_nested

            save_nested(featurizers, parent_dir, "featurizers")

        return parent_dir

    def load(self, parent_dir: str, ignore_version_mismatch: bool = False) -> None:
        """Load all units from a directory.

        Loads from consolidated format (units_metadata.json + featurizers.safetensors/.meta.json).

        Args:
            parent_dir: Directory containing saved units
            ignore_version_mismatch: If True, skip version validation.

        Returns:
            None (modifies all units in place)
        """
        metadata_path = os.path.join(parent_dir, "units_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"units_metadata.json not found in {parent_dir}. "
                "Expected consolidated format."
            )

        with open(metadata_path, "r") as f:
            all_metadata = json.load(f)

        # Load featurizers if present
        featurizers_path = os.path.join(parent_dir, "featurizers.safetensors")
        featurizers: dict[str, Any] = {}
        if os.path.exists(featurizers_path):
            from causalab.io.nested_artifacts import load_nested

            featurizers, _ = load_nested(parent_dir, "featurizers")

        # Apply to units
        for unit in self.flatten():
            if unit.id not in all_metadata:
                raise ValueError(f"Unit {unit.id} not found in saved metadata")

            metadata = all_metadata[unit.id]

            # Validate version
            version = metadata.get("version")
            if version != MODEL_UNIT_VERSION and not ignore_version_mismatch:
                raise ValueError(
                    f"Loading unit from version {version}, "
                    f"current version is {MODEL_UNIT_VERSION}."
                )

            # Load featurizer if non-trivial
            if unit.id in featurizers:
                unit.set_featurizer(Featurizer.from_dict(featurizers[unit.id]))

            # Load feature indices
            unit.set_feature_indices(metadata.get("feature_indices"))

            # Update shape
            if metadata.get("shape") is not None:
                unit.shape = tuple(metadata["shape"])

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
            Layer number in the model.
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
        self,
        input: Any,
        *,
        batch: bool = False,
        attention_mask: Any = _ATTENTION_MASK_UNSET,
        **kwargs: Any,
    ) -> list[int] | list[list[int]] | list[list[list[int]]]:
        """Return indices for *input*, optionally shifted into the padded-batch frame.

        Indexers compute positions in each example's *unpadded* tokenization
        frame. When ``attention_mask`` is provided (the output of
        ``pipeline.load(...)["attention_mask"]``), those positions are rewritten
        into the padded-batch frame: ``argmax(attention_mask, dim=1)`` gives the
        per-row offset (``0`` for right-padding, ``padded_len - unpadded_len``
        for left-padding). Without this shift, downstream consumers that index
        into the padded tensor land on the wrong absolute token for rows
        shorter than the batch max.

        Required for batched-multi calls. ``batch=True`` with ``len(input) > 1``
        means ``pipeline.load`` may pad shorter rows — the only configuration
        where the unpadded/padded frame mismatch is harmful — so this method
        raises ``ValueError`` if ``attention_mask`` is omitted in that case.
        Pass ``attention_mask=None`` explicitly to opt out (e.g. when you
        intentionally want indices in each example's unpadded frame, or for
        offline / diagnostic usage that won't index into a padded tensor).
        For ``batch=False`` or batches of one, the kwarg is genuinely optional
        — there's no padding to correct for.

        Assumes contiguous padding (a prefix of zeros for left-pad or a suffix
        for right-pad). Interior zeros in ``attention_mask`` are not supported.
        """
        attention_mask = self._resolve_attention_mask(input, batch, attention_mask)
        raw = self._indices_func.index(input, batch=batch, **kwargs)
        if attention_mask is None:
            return raw
        return self._apply_padding_shift(raw, attention_mask, batch=batch)

    def resolve_positions(
        self,
        raw_inputs: Any,
        *,
        attention_mask: Any = _ATTENTION_MASK_UNSET,
        is_original: bool | None = None,
    ) -> list[list[int]]:
        """Per-example token positions for a *batch*, in the padded-batch frame.

        Where :meth:`index_component` returns whatever nested structure the
        backbone's gather wanted — a bare position axis here, ``[head_axis,
        position_axis]`` on :class:`~causalab.neural.LM_units.AttentionHead` —
        this always returns just the positions. The head, when a unit has one, is
        a separate scalar (:meth:`head_index`), because the engine indexes the
        head and position axes independently.
        """
        return self.index_component(  # type: ignore[return-value]
            raw_inputs,
            batch=True,
            attention_mask=attention_mask,
            is_original=is_original,
        )

    def head_index(self) -> int | None:
        """The attention head this unit selects, or ``None`` for whole-width units."""
        return None

    def tensorized_index_groups(self, unit_indices: Any) -> list[Any]:
        """Split a unit's per-batch index structure into the groups that are
        stacked *separately* into index tensors before the gather/scatter.

        A plain single-axis ``pos`` unit's whole structure is stacked by one
        ``torch.tensor`` call, so it is a single group. Multi-axis units (e.g.
        :class:`AttentionHead`, whose structure is ``[head_axis, position_axis]``
        and which indexes the head and position axes separately) override this
        to return one group per axis. The ragged-span guard checks
        each group independently so the head and position axes — which
        legitimately differ in width — are never compared against each other.
        """
        return [unit_indices]

    @staticmethod
    def _resolve_attention_mask(
        input: Any, batch: bool, attention_mask: Any
    ) -> torch.Tensor | None:
        """Translate the sentinel default; raise if a batched-multi call omitted the mask.

        See :meth:`index_component` for the contract.
        """
        if attention_mask is not _ATTENTION_MASK_UNSET:
            return attention_mask
        if batch and isinstance(input, (list, tuple)) and len(input) > 1:
            raise ValueError(
                "Batched `index_component` with multiple examples requires "
                "`attention_mask` — pass `pipeline.load(...)['attention_mask']` "
                "so unpadded-frame positions get shifted into the padded-batch "
                "frame. If you intentionally want indices in each example's "
                "unpadded frame (e.g. offline export or symbolic diagnostics), "
                "pass `attention_mask=None` explicitly."
            )
        return None

    def _apply_padding_shift(
        self,
        indices: Any,
        attention_mask: torch.Tensor,
        *,
        batch: bool,
    ) -> Any:
        """Shift single-axis position indices into the padded-batch frame.

        Subclasses with multi-axis returns (e.g. :class:`AttentionHead`,
        which returns ``[head_axis, position_axis]``) override this to shift
        only the position axis.

        The bounds check in ``_shift_row`` runs even when no shift is needed
        (no left-padding, e.g. a ``batch_size=1`` call): a position computed
        for a differently-shaped input — such as a base position reused on a
        shorter counterfactual (#176) — would otherwise read the wrong token
        silently. Raising here turns that into a catchable ``ValueError``.
        """
        per_row_shift = torch.argmax(attention_mask.int(), dim=1).tolist()
        padded_len = int(attention_mask.shape[1])

        if not batch:
            return _shift_row(indices, per_row_shift[0], padded_len, ex_idx=0)

        return [
            _shift_row(row, per_row_shift[i], padded_len, ex_idx=i)
            for i, row in enumerate(indices)
        ]

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

    # ------------------------ Index helpers --------------------------- #
    def is_static(self) -> bool:
        """Return whether this unit uses static indices."""
        return self._static_indices

    def __repr__(self) -> str:
        return f"AtomicModelUnit(id='{self.id}')"

    # ---------------- Utility & misc ----------------------------------- #
    def set_layer(self, layer: int) -> None:
        """Set the layer number."""
        self.layer = layer

    def get_layer(self) -> int:
        """Get the layer number."""
        return self.layer
