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
from causalab.neural.units import AtomicModelUnit, InterchangeTarget


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
        intervention_type: The type of intervention to use ("interchange", "collect", "mask", "add", "replace", or "interpolation")

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
    if hasattr(pipeline.model, "hf_device_map"):
        # Model is sharded across GPUs via device_map; accelerate blocks .to() on it.
        # Move each intervention to the device of the layer it hooks into so that
        # torch operations (e.g. gather) see tensors on the same device.
        device_map = pipeline.model.hf_device_map
        for key, intervention in intervenable_model.interventions.items():
            intervention.to(_device_for_key(key, device_map))
        # pyvene's gather/scatter internals call get_device() to place index tensors,
        # which on a sharded model returns the embedding GPU rather than the layer's
        # actual device. Returning None makes gather_neurons fall back to the layer
        # tensor's device, which is what we want.
        intervenable_model.get_device = lambda: None  # type: ignore[method-assign]
    else:
        intervenable_model.set_device(pipeline.model.device)

    return intervenable_model


def _device_for_key(key: str, hf_device_map: dict) -> str:
    """Resolve the GPU device for a pyvene intervention key.

    pyvene keys look like ``"model.layers.77#0"``. We strip the group-index
    suffix and walk up the dotted path until we find a match in hf_device_map.
    """
    path = key.split("#")[0]
    while path:
        if path in hf_device_map:
            return hf_device_map[path]
        path = path.rsplit(".", 1)[0] if "." in path else ""
    return next(iter(hf_device_map.values()))


def device_for_layer(pipeline: Pipeline, layer: int) -> torch.device:
    """Resolve the device a given transformer layer lives on.

    For models loaded with ``device_map="auto"``, different layers can live
    on different GPUs. Tensors that participate in operations at that layer
    (steering vectors, featurizers, etc.) must be on the same device. For
    single-device models this returns ``model.device``.
    """
    if hasattr(pipeline.model, "hf_device_map"):
        device_map = pipeline.model.hf_device_map
        key = f"model.layers.{layer}"
        if key in device_map:
            return torch.device(device_map[key])
        # Fallback: look up nearest ancestor in the map
        return torch.device(_device_for_key(key, device_map))
    return pipeline.model.device


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
