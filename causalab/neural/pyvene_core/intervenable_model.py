"""
intervenable_model.py
=====================
Utilities for creating and managing pyvene IntervenableModel instances.

This module provides helper functions for creating intervention models
and managing their lifecycle, including proper memory cleanup.
"""

from __future__ import annotations

import gc

import torch
import pyvene as pv  # type: ignore[import-untyped]

from causalab.neural.pipeline import Pipeline
from causalab.neural.model_units import AtomicModelUnit, InterchangeTarget


def prepare_intervenable_model(
    pipeline: Pipeline,
    model_units: InterchangeTarget | list[AtomicModelUnit],
    intervention_type: str = "interchange",
) -> pv.IntervenableModel:
    """
    Prepare an intervenable model for specified model units and intervention type.

    Creates a pyvene IntervenableModel configured for the specified intervention type
    and model units. Handles both static and dynamic index configurations. The intervention
    configs are linked across groups, meaning those components share a counterfactual input.

    Args:
        pipeline: The pipeline containing the base model
        model_units: Either an InterchangeTarget (nested structure with groups) or a flat
                    list of AtomicModelUnit instances. If a flat list is provided, all units
                    will be placed in a single group.
        intervention_type: The type of intervention to use ("interchange", "collect", or "mask")

    Returns:
        intervenable_model: The prepared intervenable model on the pipeline's device
    """
    # Auto-wrap if needed
    if isinstance(model_units, list):
        # Flat list - wrap in single group
        interchange_target = InterchangeTarget([model_units])
    else:
        # Already an InterchangeTarget
        interchange_target = model_units

    # Check if all model units have static indices
    # If all indices are static, we can use a more efficient model
    static = True
    for group in interchange_target:
        for model_unit in group:
            if not model_unit.is_static():
                static = False

    # Create intervention configs for all model units
    configs = []
    for i, group in enumerate(interchange_target):
        for model_unit in group:
            config = model_unit.create_intervention_config(i, intervention_type)
            configs.append(config)

    # Create the intervenable model with the collected configs
    intervention_config = pv.IntervenableConfig(configs)
    intervenable_model = pv.IntervenableModel(
        intervention_config, model=pipeline.model, use_fast=static
    )
    intervenable_model.set_device(pipeline.model.device)

    return intervenable_model


def delete_intervenable_model(intervenable_model: pv.IntervenableModel) -> None:
    """
    Delete the intervenable model and clear CUDA memory.

    This function properly cleans up an intervenable model by moving it to CPU first,
    then deleting it and clearing all CUDA caches to prevent memory leaks.

    Args:
        intervenable_model: The pyvene intervenable model to be deleted
    """
    intervenable_model.set_device("cpu", set_model=False)
    del intervenable_model
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
