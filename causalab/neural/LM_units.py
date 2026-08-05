"""
LM_units.py
===========
Helpers that bind the *core* component / featurizer abstractions from
`model_units.py` to language-model pipelines.  They let you refer to:

* A **ResidualStream** slice: hidden state of one or more token positions.
* An **AttentionHead** value: output for a single attention head.

All helpers inherit from :class:`model_units.AtomicModelUnit`, so they carry
the full featurizer + feature indexing machinery.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch

from causalab.neural.units import (
    _ATTENTION_MASK_UNSET,
    AtomicModelUnit,
    ComponentIndexer,
    Featurizer,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  LLM-specific AtomicModelUnits                                              #
# --------------------------------------------------------------------------- #
class ResidualStream(AtomicModelUnit):
    """Residual-stream slice at *layer* for given token position(s)."""

    def __init__(
        self,
        layer: int,
        token_indices: list[int] | ComponentIndexer,
        *,
        featurizer: Featurizer | None = None,
        shape: tuple[int, ...] | None = None,
        feature_indices: list[int] | None = None,
        target_output: bool = False,
    ) -> None:
        component_type = "block_output" if target_output else "block_input"
        self.token_indices = token_indices
        tok_id = (
            token_indices.id
            if isinstance(token_indices, ComponentIndexer)
            else token_indices
        )
        # Include component_type in the UID to distinguish between block_input (embeddings/layer -1)
        # and block_output (normal layers) when they target the same layer number
        uid = f"ResidualStream(Layer-{layer},{component_type},Token-{tok_id})"

        unit = "pos"

        super().__init__(
            layer=layer,
            component_type=component_type,
            indices_func=token_indices,
            unit=unit,
            featurizer=featurizer or Featurizer(),
            feature_indices=feature_indices,
            shape=shape,
            id=uid,
        )

    @classmethod
    def load_modules(
        cls, base_name: str, dir: str, token_positions: list[ComponentIndexer]
    ) -> ResidualStream:
        # TODO: the `+ 6` here and below assumes a single-char delimiter after
        # "Layer" / "Token" (id schema is "Layer-N,...,Token-id"). Any change
        # to delimiter width will silently break parsing.
        # Extract layer number plus one additonal
        # character after "Layer" for the _ or :
        layer_start = base_name.find("Layer") + 6
        layer_end = base_name.find(",", layer_start)
        layer = int(base_name[layer_start:layer_end])

        # Extract component_type ("block_input" or "block_output") sitting
        # between the layer field and the Token field, then map back to the
        # constructor's `target_output` flag. Mirrors MLP.load_modules's
        # `location` parsing.
        component_type_start = layer_end + 1
        component_type_end = base_name.find(",", component_type_start)
        component_type = base_name[component_type_start:component_type_end]
        target_output = component_type == "block_output"

        # Extract token position plus one additional
        # character after "Token" for the _ or :
        token_start = base_name.find("Token") + 6
        token_end = base_name.find(")", token_start)
        tok_id = base_name[token_start:token_end]
        # Find the element of the list with a .id that matches tok_id
        token_indices = next((tp for tp in token_positions if tp.id == tok_id), None)
        if token_indices is None:
            raise ValueError(
                f"Token position with id '{tok_id}' not found in provided list."
            )

        base_path = os.path.join(dir, base_name)
        indices_path = base_path + "_indices"

        # Load the featurizer
        featurizer = Featurizer.load_modules(base_path)

        # Load indices if they exist
        feature_indices = None
        try:
            with open(indices_path, "r") as f:
                feature_indices = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load indices for {base_name}: {e}")

        return cls(
            layer=layer,
            token_indices=token_indices,
            featurizer=featurizer,
            feature_indices=feature_indices,
            target_output=target_output,
        )


class MLP(AtomicModelUnit):
    """MLP slice at *layer* for given token position(s)."""

    VALID_LOCATIONS = ("mlp_input", "mlp_output", "mlp_activation")

    def __init__(
        self,
        layer: int,
        token_indices: list[int] | ComponentIndexer,
        *,
        featurizer: Featurizer | None = None,
        shape: tuple[int, ...] | None = None,
        feature_indices: list[int] | None = None,
        location: str = "mlp_output",
    ) -> None:
        if location not in self.VALID_LOCATIONS:
            raise ValueError(
                f"Invalid location '{location}'. Must be one of {self.VALID_LOCATIONS}"
            )

        self.token_indices = token_indices
        tok_id = (
            token_indices.id
            if isinstance(token_indices, ComponentIndexer)
            else token_indices
        )
        uid = f"MLP(Layer-{layer},{location},Token-{tok_id})"

        unit = "pos"

        super().__init__(
            layer=layer,
            component_type=location,
            indices_func=token_indices,
            unit=unit,
            featurizer=featurizer or Featurizer(),
            feature_indices=feature_indices,
            shape=shape,
            id=uid,
        )

    @classmethod
    def load_modules(
        cls, base_name: str, dir: str, token_positions: list[ComponentIndexer]
    ) -> MLP:
        # Extract layer number
        layer_start = base_name.find("Layer") + 6
        layer_end = base_name.find(",", layer_start)
        layer = int(base_name[layer_start:layer_end])

        # Extract location (mlp_input, mlp_output, or mlp_activation)
        location_start = layer_end + 1
        location_end = base_name.find(",", location_start)
        location = base_name[location_start:location_end]

        # Extract token position
        token_start = base_name.find("Token") + 6
        token_end = base_name.find(")", token_start)
        tok_id = base_name[token_start:token_end]

        # Find the element of the list with a .id that matches tok_id
        token_indices = next((tp for tp in token_positions if tp.id == tok_id), None)
        if token_indices is None:
            raise ValueError(
                f"Token position with id '{tok_id}' not found in provided list."
            )

        base_path = os.path.join(dir, base_name)
        indices_path = base_path + "_indices"

        # Load the featurizer
        featurizer = Featurizer.load_modules(base_path)

        # Load indices if they exist
        feature_indices = None
        try:
            with open(indices_path, "r") as f:
                feature_indices = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load indices for {base_name}: {e}")

        return cls(
            layer=layer,
            token_indices=token_indices,
            featurizer=featurizer,
            location=location,
            feature_indices=feature_indices,
        )


class AttentionOutput(AtomicModelUnit):
    """Whole attention-block output at *layer* (all heads), for path-patching restoration.

    Distinct from :class:`AttentionHead` (a single head's value stream): this targets
    the ``attention_output`` component — the full attention sublayer output (all
    heads jointly) that is written back into the residual stream. The merged output is
    already ``hidden_size``-wide, so this unit is GQA-correct without per-head handling.

    Two uses:

    - **Layer-level ablation** (``replace``): zero- or mean-replace the whole sublayer
      output to measure its behavioral contribution. See
      :func:`~causalab.neural.activations.targets.build_attention_output_targets` and the
      ``ablation`` analysis (``component_type=attention_output``).
    - **Path patching** (*restorer*): freeze its activation to the clean-base value so a
      perturbed sender's contribution cannot reach downstream components through this
      layer's attention. Never restore ``block_output`` instead — that bundles the
      residual stream carrying the sender's direct contribution and would erase the very
      path being measured (see ``docs/PATH_PATCHING.md`` §4).
    """

    def __init__(
        self,
        layer: int,
        token_indices: list[int] | ComponentIndexer,
        *,
        featurizer: Featurizer | None = None,
        shape: tuple[int, ...] | None = None,
        feature_indices: list[int] | None = None,
    ) -> None:
        component_type = "attention_output"  # whole-block attention output
        self.token_indices = token_indices
        tok_id = (
            token_indices.id
            if isinstance(token_indices, ComponentIndexer)
            else token_indices
        )
        uid = f"AttentionOutput(Layer-{layer},{component_type},Token-{tok_id})"

        unit = "pos"

        super().__init__(
            layer=layer,
            component_type=component_type,
            indices_func=token_indices,
            unit=unit,
            featurizer=featurizer or Featurizer(),
            feature_indices=feature_indices,
            shape=shape,
            id=uid,
        )

    @classmethod
    def load_modules(
        cls, base_name: str, dir: str, token_positions: list[ComponentIndexer]
    ) -> AttentionOutput:
        # UID schema: "AttentionOutput(Layer-N,attention_output,Token-id)". The
        # `+ 6` offsets assume a single-char delimiter after "Layer"/"Token",
        # matching MLP.load_modules. component_type is fixed, so only layer and
        # token need parsing.
        layer_start = base_name.find("Layer") + 6
        layer_end = base_name.find(",", layer_start)
        layer = int(base_name[layer_start:layer_end])

        token_start = base_name.find("Token") + 6
        token_end = base_name.find(")", token_start)
        tok_id = base_name[token_start:token_end]
        token_indices = next((tp for tp in token_positions if tp.id == tok_id), None)
        if token_indices is None:
            raise ValueError(
                f"Token position with id '{tok_id}' not found in provided list."
            )

        base_path = os.path.join(dir, base_name)
        indices_path = base_path + "_indices"

        featurizer = Featurizer.load_modules(base_path)

        feature_indices = None
        try:
            with open(indices_path, "r") as f:
                feature_indices = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load indices for {base_name}: {e}")

        return cls(
            layer=layer,
            token_indices=token_indices,
            featurizer=featurizer,
            feature_indices=feature_indices,
        )


class AttentionHead(AtomicModelUnit):
    """Attention-head value stream at (*layer*, *head*) for token position(s)."""

    def __init__(
        self,
        layer: int,
        head: int,
        token_indices: list[int] | ComponentIndexer,
        *,
        featurizer: Featurizer | None = None,
        shape: tuple[int, ...] | None = None,
        feature_indices: list[int] | None = None,
        target_output: bool = True,
    ) -> None:
        self.head = head
        component_type = (
            "head_attention_value_output"
            if target_output
            else "head_attention_value_input"
        )

        tok_id = (
            token_indices.id
            if isinstance(token_indices, ComponentIndexer)
            else token_indices
        )
        uid = f"AttentionHead(Layer-{layer},Head-{head},Token-{tok_id})"

        unit = "h.pos"

        super().__init__(
            layer=layer,
            component_type=component_type,
            indices_func=token_indices,
            unit=unit,
            featurizer=featurizer or Featurizer(),
            feature_indices=feature_indices,
            shape=shape,
            id=uid,
        )

    @classmethod
    def load_modules(
        cls,
        base_name: str,
        submission_folder_path: str,
        token_positions: list[ComponentIndexer],
    ) -> AttentionHead:
        """Load AttentionHead from a base name and submission folder path."""
        # TODO: the `+ 6` magic numbers below assume a single-char delimiter
        # after "Layer" / "Head" / "Token". The id schema is
        # "AttentionHead(Layer-N,Head-H,Token-id)" — any change to delimiter
        # width will silently break parsing.
        # Check if the base name starts with "AttentionHead"
        # Extract layer number plus
        layer_start = base_name.find("Layer") + 6
        layer_end = base_name.find(",", layer_start)
        layer = int(base_name[layer_start:layer_end])

        # Extract head number
        head_start = base_name.find(",Head") + 6
        head_end = base_name.find(",", head_start)
        head = int(base_name[head_start:head_end])

        # Extract token position id; look up the matching ComponentIndexer in
        # the supplied list. Mirrors ResidualStream / MLP.load_modules.
        token_start = base_name.find("Token") + 6
        token_end = base_name.find(")", token_start)
        tok_id = base_name[token_start:token_end]
        token_indices = next((tp for tp in token_positions if tp.id == tok_id), None)
        if token_indices is None:
            raise ValueError(
                f"Token position with id '{tok_id}' not found in provided list."
            )

        base_path = os.path.join(submission_folder_path, base_name)
        indices_path = base_path + "_indices"

        # Load the featurizer
        featurizer = Featurizer.load_modules(base_path)

        # Load indices if they exist
        feature_indices = None
        try:
            with open(indices_path, "r") as f:
                feature_indices = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load indices for {base_name}: {e}")

        return cls(
            layer=layer,
            head=head,
            token_indices=token_indices,
            featurizer=featurizer,
            feature_indices=feature_indices,
        )

    # ------------------------------------------------------------------ #

    def index_component(
        self,
        input: Any,
        *,
        batch: bool = False,
        attention_mask: Any = _ATTENTION_MASK_UNSET,
        **kwargs: Any,
    ) -> list[Any]:
        """Return ``[head_axis, position_axis]`` for *input*.

        See :meth:`AtomicModelUnit.index_component` for the ``attention_mask``
        contract (required for batched-multi calls; pass ``None`` explicitly to
        opt out). When a mask is supplied, the position axis is shifted into the
        padded-batch frame; the head axis is left untouched.
        """
        attention_mask = self._resolve_attention_mask(input, batch, attention_mask)
        if batch:
            raw = [
                [[self.head]] * len(input),
                [self._indices_func.index(x, **kwargs) for x in input],
            ]
        else:
            raw = [[[self.head]], [self._indices_func.index(input, **kwargs)]]
        if attention_mask is None:
            return raw
        return self._apply_padding_shift(raw, attention_mask, batch=batch)

    def _apply_padding_shift(
        self,
        indices: Any,
        attention_mask: torch.Tensor,
        *,
        batch: bool,
    ) -> Any:
        """Shift only the position axis; leave the head axis untouched.

        Note: ``index_component`` wraps the position axis as a list-of-rows even
        when ``batch=False`` (``[[positions]]``), so the shift always routes
        through the base ``batch=True`` code path.
        """
        del batch  # position axis is always list-of-rows shaped
        *prefix_axes, position_axis = list(indices)
        shifted = super()._apply_padding_shift(
            position_axis, attention_mask, batch=True
        )
        return [*prefix_axes, shifted]

    def tensorized_index_groups(self, unit_indices: Any) -> list[Any]:
        """An ``AttentionHead``'s structure is ``[head_axis, position_axis]``;
        each axis is stacked into its own index tensor, so each is its own
        tensorized group. See :meth:`AtomicModelUnit.tensorized_index_groups`.
        """
        return list(unit_indices)

    def resolve_positions(
        self,
        raw_inputs: Any,
        *,
        attention_mask: Any = _ATTENTION_MASK_UNSET,
        is_original: bool | None = None,
    ) -> list[list[int]]:
        """Just the position axis of ``[head_axis, position_axis]``."""
        return self.index_component(
            raw_inputs,
            batch=True,
            attention_mask=attention_mask,
            is_original=is_original,
        )[1]

    def head_index(self) -> int:
        return self.head


class AttentionHeadValue(AttentionHead):
    """A single head's *value vector* (``head_value_output``) at (*layer*, *head*).

    The per-head value activation ``v_h = W_V^h · x`` *before* attention weighting —
    distinct from :class:`AttentionHead` (``head_attention_value_output``: the head's
    attention-weighted, output-projected contribution to the residual, used as a
    path-patching *sender*).

    Path patching uses this as the *receiver* realizing the **value input of head h**
    edge (the TransformerLens / Goldowsky-Dill "v" receiver): collecting ``v_h`` under
    the path-restricted forward and re-injecting it isolates the information reaching
    head ``h`` through its value path, while its attention pattern (from the clean q/k)
    and the other heads are left untouched. Note the model exposes no per-head value
    *input* — ``v_proj`` reads the whole residual — so the value vector is the
    granularity at which this edge is realizable.

    Under grouped-query attention the value vector is **shared** across a KV group, so
    this unit is addressed in KV-head space and re-injecting it reaches every query head
    in the group (the per-query-head value path does not physically exist); the
    query→KV-group mapping lives in
    :func:`causalab.methods.path_patching.targets.build_receiver_unit`.

    Reuses :class:`AttentionHead`'s ``h.pos`` indexing wholesale; only the
    component and the diagnostic id differ.
    """

    def __init__(
        self,
        layer: int,
        head: int,
        token_indices: list[int] | ComponentIndexer,
        *,
        featurizer: Featurizer | None = None,
        shape: tuple[int, ...] | None = None,
        feature_indices: list[int] | None = None,
    ) -> None:
        super().__init__(
            layer,
            head,
            token_indices,
            featurizer=featurizer,
            shape=shape,
            feature_indices=feature_indices,
            target_output=True,
        )
        # Retarget from head_attention_value_output (the sender component) to the
        # per-head value vector. component_type/id are plain attributes on
        # AtomicModelUnit, read lazily when the unit is resolved to a site, so
        # resetting them post-init is enough — indexing is inherited.
        self.component_type = "head_value_output"
        tok_id = (
            token_indices.id
            if isinstance(token_indices, ComponentIndexer)
            else token_indices
        )
        self.id = f"AttentionHeadValue(Layer-{layer},Head-{head},Token-{tok_id})"


class AttentionHeadQuery(AttentionHead):
    """A single head's *query vector* (``head_query_output``) at (*layer*, *head*).

    The per-head query activation ``q_h = W_Q^h · x`` — the exact query-path mirror
    of :class:`AttentionHeadValue`'s value-path receiver. Path patching uses this as
    the *receiver* realizing the **query input of head h** edge: collecting ``q_h``
    under the path-restricted forward and re-injecting it isolates the information
    reaching head ``h`` through its *query* (which determines where the head looks),
    leaving its keys, values, and the other heads untouched.

    As for the value vector, there is no per-head query *input* to read, so the
    query vector ``q_h`` (post-projection, pre-attention) is the granularity at
    which this edge is realizable. Unlike the value vector, the query
    vector is **per-query-head** even under grouped-query attention (only k/v are
    shared across a KV group), so this unit is addressed directly in query-head space
    with no query→KV-group remap — contrast :class:`AttentionHeadValue`, whose KV-head
    mapping lives in
    :func:`causalab.methods.path_patching.targets.build_receiver_unit`.

    Reuses :class:`AttentionHead`'s ``h.pos`` indexing wholesale; only the
    component and the diagnostic id differ.
    """

    def __init__(
        self,
        layer: int,
        head: int,
        token_indices: list[int] | ComponentIndexer,
        *,
        featurizer: Featurizer | None = None,
        shape: tuple[int, ...] | None = None,
        feature_indices: list[int] | None = None,
    ) -> None:
        super().__init__(
            layer,
            head,
            token_indices,
            featurizer=featurizer,
            shape=shape,
            feature_indices=feature_indices,
            target_output=True,
        )
        # Retarget from head_attention_value_output (the sender component) to the
        # per-head query vector. component_type/id are plain attributes on
        # AtomicModelUnit, read lazily when the unit is resolved to a site, so
        # resetting them post-init is enough — indexing is inherited.
        self.component_type = "head_query_output"
        tok_id = (
            token_indices.id
            if isinstance(token_indices, ComponentIndexer)
            else token_indices
        )
        self.id = f"AttentionHeadQuery(Layer-{layer},Head-{head},Token-{tok_id})"
