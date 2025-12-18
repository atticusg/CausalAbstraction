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

from causalab.neural.model_units import (
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
        # Extract layer number plus one additonal
        # character after "Layer" for the _ or :
        layer_start = base_name.find("Layer") + 6
        layer_end = base_name.find(",", layer_start)
        layer = int(base_name[layer_start:layer_end])

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
        token_positions: list[int] | ComponentIndexer,
    ) -> AttentionHead:
        """Load AttentionHead from a base name and submission folder path."""
        # Check if the base name starts with "AttentionHead"
        # Extract layer number plus
        layer_start = base_name.find("Layer") + 6
        layer_end = base_name.find(",", layer_start)
        layer = int(base_name[layer_start:layer_end])

        # Extract head number
        head_start = base_name.find(",Head") + 6
        head_end = base_name.find(",", head_start)
        head = int(base_name[head_start:head_end])

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
            token_indices=token_positions,
            featurizer=featurizer,
            feature_indices=feature_indices,
        )

    # ------------------------------------------------------------------ #

    def index_component(
        self, input: Any, batch: bool = False, **kwargs: Any
    ) -> list[Any]:
        """Return indices for *input* by delegating to wrapped function."""
        if batch:
            return [
                [[self.head]] * len(input),
                [self._indices_func.index(x, **kwargs) for x in input],
            ]
        return [[[self.head]], [self._indices_func.index(input, **kwargs)]]
