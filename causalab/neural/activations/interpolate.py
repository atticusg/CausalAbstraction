"""
interpolate.py
==============
Core utilities for running interpolation intervention experiments.

This module provides functions for running interventions on neural networks using
the pyvene library. It focuses on interpolation interventions where activations
are computed as an arbitrary function of base and source featurized activations:

    new_act = inverse_featurizer(f(f_base, f_src, **params), base_err)

The canonical use case is linear interpolation:

    def linear_interp(f_base, f_src, alpha):
        return (1 - alpha) * f_base + alpha * f_src

At alpha=1 this reduces to interchange; at alpha=0 it is the identity.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
from tqdm import tqdm
from pyvene import IntervenableModel  # type: ignore[import-untyped]

from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from causalab.neural.pipeline import Pipeline
from causalab.neural.units import InterchangeTarget
from causalab.neural.featurizer import Featurizer
from causalab.neural.activations.intervenable_model import (
    prepare_intervenable_model,
    delete_intervenable_model,
)
from causalab.neural.activations.interchange_mode import prepare_intervenable_inputs
from causalab.neural.activations.data_utils import (
    convert_to_top_k,
    move_outputs_to_cpu,
)

logger = logging.getLogger(__name__)


def set_interventions_interpolation(
    intervenable_model: IntervenableModel,
    fn: Callable[..., torch.Tensor],
    **params: Any,
) -> None:
    """Push an interpolation function and its parameters onto all intervention instances.

    Args:
        intervenable_model: The pyvene IntervenableModel whose interventions to configure.
        fn: Callable with signature (f_base, f_src, **params) -> Tensor.
        **params: Keyword arguments forwarded to fn on each call.
    """
    for v in intervenable_model.interventions.values():
        intervention: Any = v[0] if isinstance(v, tuple) else v
        if hasattr(intervention, "set_interpolation"):
            intervention.set_interpolation(fn, **params)


def batched_interpolation_intervention(
    pipeline: Pipeline,
    intervenable_model: IntervenableModel,
    examples: list[CounterfactualExample] | list[LabeledCounterfactualExample],
    interchange_target: InterchangeTarget,
    fn: Callable[..., torch.Tensor],
    params: dict[str, Any],
    output_scores: bool | int = True,
) -> dict[str, Any]:
    """Perform interpolation interventions on a batch of examples.

    Args:
        pipeline: The pipeline containing the model.
        intervenable_model: PyVENE model with preset intervention locations.
        examples: List of counterfactual examples.
        interchange_target: InterchangeTarget containing model components to intervene on.
        fn: Interpolation function with signature (f_base, f_src, **params) -> Tensor.
        params: Keyword arguments forwarded to fn.
        output_scores: Whether to include scores in output dictionary (default: True).

    Returns:
        dict: Dictionary with 'sequences' and optionally 'scores' keys.
    """
    batched_base, batched_counterfactuals, inv_locations, feature_indices = (
        prepare_intervenable_inputs(pipeline, examples, interchange_target)
    )

    set_interventions_interpolation(intervenable_model, fn, **params)

    gen_kwargs = {"output_scores": output_scores}
    output = pipeline.intervenable_generate(
        intervenable_model,
        batched_base,
        batched_counterfactuals,
        inv_locations,
        feature_indices,
        **gen_kwargs,
    )

    for batched in [batched_base] + batched_counterfactuals:
        for k, v in batched.items():
            batched[k] = v.cpu()

    return output


def run_interpolation_interventions(
    pipeline: Pipeline,
    counterfactual_dataset: list[CounterfactualExample]
    | list[LabeledCounterfactualExample],
    interchange_target: InterchangeTarget,
    fn: Callable[..., torch.Tensor],
    params: dict[str, Any],
    batch_size: int = 32,
    output_scores: bool | int = True,
) -> dict[str, list[Any]]:
    """Run interpolation interventions on a full counterfactual dataset in batches.

    The intervention patches an arbitrary function of the base and source
    featurized activations:

        new_act = inverse_featurizer(fn(f_base, f_src, **params), base_err)

    Args:
        pipeline: The pipeline containing the model.
        counterfactual_dataset: List of counterfactual examples.
        interchange_target: InterchangeTarget containing model components to intervene on.
        fn: Interpolation function with signature (f_base, f_src, **params) -> Tensor.
        params: Keyword arguments forwarded to fn on each call.
        batch_size: Number of examples to process in each batch.
        output_scores: Controls score output format:
            - False: No scores
            - True: Full vocabulary scores (on CPU)
            - int (e.g., 10): Top-k scores (on CPU, memory efficient)

    Returns:
        dict: Dictionary with 'sequences' (on CPU) and optionally 'scores' keys
              (on CPU, in top-k format if int was provided).
    """
    intervenable_model = prepare_intervenable_model(
        pipeline, interchange_target, intervention_type="interpolation"
    )

    all_outputs = []

    for start in tqdm(
        range(0, len(counterfactual_dataset), batch_size),
        desc="Processing batches",
        disable=not logger.isEnabledFor(logging.DEBUG),
        leave=False,
    ):
        examples = counterfactual_dataset[start : start + batch_size]
        with torch.no_grad():
            output_dict = batched_interpolation_intervention(
                pipeline,
                intervenable_model,
                examples,
                interchange_target,
                fn=fn,
                params=params,
                output_scores=output_scores,
            )
            all_outputs.append(output_dict)

    delete_intervenable_model(intervenable_model)

    if not isinstance(output_scores, bool) and output_scores > 0:
        all_outputs = convert_to_top_k(all_outputs, pipeline, k=output_scores)

    all_outputs = move_outputs_to_cpu(all_outputs)

    all_outputs = {
        k: [output[k] for output in all_outputs] for k in all_outputs[0].keys()
    }

    return all_outputs


def sweep_interpolation_interventions(
    pipeline: Pipeline,
    counterfactual_dataset: list[CounterfactualExample]
    | list[LabeledCounterfactualExample],
    interchange_target: InterchangeTarget,
    configs: dict[str, tuple[Featurizer, Callable[..., torch.Tensor], dict[str, Any]]],
    batch_size: int = 32,
    output_scores: bool | int = True,
) -> dict[str, dict[str, list[Any]]]:
    """Run interpolation interventions for multiple (featurizer, fn, params) configurations.

    For each named configuration, sets the featurizer on the interchange target and
    calls run_interpolation_interventions. This avoids reconstructing the intervenable
    model from scratch in user code and centralises the featurizer-swap logic.

    Args:
        pipeline: The pipeline containing the model.
        counterfactual_dataset: List of counterfactual examples.
        interchange_target: InterchangeTarget whose featurizer is swapped per config.
        configs: Mapping from config name to (featurizer, fn, params) tuple.
        batch_size: Number of examples to process in each batch.
        output_scores: Controls score output format (same semantics as
            run_interpolation_interventions).

    Returns:
        dict mapping each config name to the result dict from
        run_interpolation_interventions (keys: 'sequences', optionally 'scores').
    """
    results: dict[str, dict[str, list[Any]]] = {}
    for name, (featurizer, fn, params) in configs.items():
        interchange_target.set_featurizer(featurizer)
        results[name] = run_interpolation_interventions(
            pipeline,
            counterfactual_dataset,
            interchange_target,
            fn,
            params,
            batch_size,
            output_scores,
        )
    return results
