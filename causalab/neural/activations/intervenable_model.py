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

# pyvene 0.1.8 realizes per-head components (``head_attention_value_output`` and
# the ``head_{value,query}_output`` receivers) at the ``hidden_size //
# num_attention_heads`` width — it infers the head dimension from the head
# *count*, not ``config.head_dim``. On a model whose ``head_dim`` is decoupled
# from that ratio (e.g. Qwen3-4B: ``head_dim=128`` vs ``hidden // n_head = 80``)
# the per-head accessor returns a wrong-width vector *silently*, so every
# collect / interchange / steer / mask over a head unit is corrupted. The
# path-patching receiver builder already guards its value-receiver path
# (``methods/path_patching/targets.py``); the guard below is the general one at
# the single chokepoint every per-head operation flows through. Tracked in #386.
_HEAD_COMPONENTS = frozenset(
    {
        "head_attention_value_output",
        "head_attention_value_input",
        "head_value_output",
        "head_query_output",
    }
)


def _assert_head_dim_supported(
    pipeline: Pipeline, units: list[AtomicModelUnit]
) -> None:
    """Raise if any per-head unit targets a model with a decoupled ``head_dim``.

    Fails loudly here instead of letting pyvene return a wrong-width per-head
    activation (see the module note above; #386). No-op for non-head units and
    for the common coupled case (``head_dim == hidden // n_head``).
    """
    head_units = [
        u for u in units if getattr(u, "component_type", None) in _HEAD_COMPONENTS
    ]
    if not head_units:
        return
    config = pipeline.model.config
    coupled = config.hidden_size // config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or coupled
    if head_dim == coupled:
        return
    components = sorted({u.component_type for u in head_units})
    raise NotImplementedError(
        f"Per-head component(s) {components} target a model with a decoupled "
        f"head_dim (head_dim={head_dim} != hidden_size // num_attention_heads="
        f"{coupled}). pyvene 0.1.8 slices per-head value/query vectors at the "
        f"hidden // n_head width (inferred from the head count, not config."
        f"head_dim), so on such models (e.g. Qwen3) the per-head activation comes "
        f"back the wrong width — silently. Use a residual / mlp_input target "
        f"instead, or a model with head_dim == hidden // n_head. Tracked in #386."
    )


def prepare_intervenable_model(
    pipeline: Pipeline,
    model_units: InterchangeTarget | list[AtomicModelUnit],
    intervention_type: str = "interchange",
    type_by_unit: dict[str, str] | None = None,
    noise_seed: int = 0,
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
        intervention_type: The type of intervention to use ("interchange", "collect", "mask", "add", "replace", "interpolation", or "noise")
        type_by_unit: Optional ``unit_id -> intervention_type`` map. When given, each
            unit uses its mapped type instead of the single ``intervention_type`` —
            this is how a *mixed* model is built (e.g. ``noise`` on the corrupted
            entry units and ``replace`` on the restored site for causal tracing).
            Every unit in ``model_units`` must have an entry.
        noise_seed: Seed for ``noise`` interventions (reproducible per (site, seed)).

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

    # Fail loudly before pyvene silently mis-slices a per-head unit on a
    # decoupled-head_dim model (#386).
    _assert_head_dim_supported(pipeline, interchange_target.flatten())

    # Check if all model units have static indices
    # If all indices are static, we can use a more efficient model
    static = True
    for group in interchange_target:
        for model_unit in group:
            if not model_unit.is_static():
                static = False

    # Create intervention configs for all model units. A per-unit type map (if
    # given) lets one model mix intervention types across sites.
    configs = []
    for i, group in enumerate(interchange_target):
        for model_unit in group:
            unit_type = (
                type_by_unit[model_unit.id]
                if type_by_unit is not None
                else intervention_type
            )
            config = model_unit.create_intervention_config(
                i, unit_type, noise_seed=noise_seed
            )
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


def prepare_mixed_intervenable_model(
    pipeline: Pipeline,
    interchange_target: InterchangeTarget,
    collect_units: list[AtomicModelUnit],
) -> pv.IntervenableModel:
    """Build an IntervenableModel that mixes *interchange* and *collect* roles.

    Path patching's PASS 1 needs a single forward in which the sender + restorers
    apply an interchange intervention while a *receiver* is read out (collected) —
    after the restorers have written their base values. pyvene supports this: each
    intervention carries its own ``group_key`` and ``intervention_type``, so we emit
    interchange configs for every unit in ``interchange_target`` (group_key = group
    index) and collect configs for each ``collect_unit`` (group_key continuing past
    the interchange groups). The caller drives it with
    ``sources = [<one per interchange group>, *([None] * n_collect)]`` — the collect
    groups consume no source and read the base.

    **Collect-order contract.** Configs are appended in ``collect_units`` order, and
    pyvene's ``sorted_keys`` is config-add order (no ``intervenables_sort_fn``), so the
    collected activations (``result[0][1]``, iterated over ``sorted_keys`` keeping only
    ``CollectIntervention`` outputs) come back in ``collect_units`` order. Receiver-set
    path patching relies on this: it injects ``v*`` in the *same* order in PASS 2, with
    no reordering. Do **not** re-sort ``configs`` here without revisiting that contract
    (and the order-invariance test that guards it).

    Mirrors :func:`prepare_intervenable_model`'s device handling and ``use_fast``
    rule (static only when *every* unit has static indices).
    """
    # Guard the interchange + collect units against a decoupled-head_dim model
    # before pyvene silently mis-slices a per-head unit (#386).
    _assert_head_dim_supported(
        pipeline, [*interchange_target.flatten(), *collect_units]
    )

    static = all(
        unit.is_static() for group in interchange_target for unit in group
    ) and all(unit.is_static() for unit in collect_units)

    configs = []
    for group_index, group in enumerate(interchange_target):
        for model_unit in group:
            configs.append(
                model_unit.create_intervention_config(group_index, "interchange")
            )
    # Collect groups follow the interchange groups; each receiver gets its own key.
    next_key = len(list(interchange_target))
    for offset, receiver in enumerate(collect_units):
        configs.append(
            receiver.create_intervention_config(next_key + offset, "collect")
        )

    intervenable_model = pv.IntervenableModel(
        pv.IntervenableConfig(configs), model=pipeline.model, use_fast=static
    )
    if hasattr(pipeline.model, "hf_device_map"):
        device_map = pipeline.model.hf_device_map
        for key, intervention in intervenable_model.interventions.items():
            intervention.to(_device_for_key(key, device_map))
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


def reset_noise_interventions(intervenable_model: pv.IntervenableModel) -> None:
    """Restart the RNG stream of every ``noise`` intervention on the model.

    A reused intervenable model carries its noise generators' state across
    forward calls. When the same model is iterated over independent groups of
    examples (the length buckets of one causal-trace cell), each group must see
    the noise stream from its seed — exactly what a fresh per-group model gave.
    Calling this at each group boundary reproduces that, without rebuilding the
    pyvene hooks. Interventions without a ``reset_noise_rng`` (replace, steer,
    …) are left untouched.
    """
    for intervention in intervenable_model.interventions.values():
        reset = getattr(intervention, "reset_noise_rng", None)
        if callable(reset):
            reset()


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
