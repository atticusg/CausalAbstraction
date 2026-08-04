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

from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from causalab.neural.pipeline import Pipeline
from causalab.neural.units import InterchangeTarget
from causalab.neural.featurizer import Featurizer
from causalab.neural.activations.engine import (
    build_interventions,
    build_plans,
    generate_with_interventions,
)
from causalab.neural.activations.interchange_mode import (
    collect_group_sources,
    prepare_interchange_batch,
)
from causalab.neural.activations.data_utils import (
    convert_to_top_k,
    move_outputs_to_cpu,
)

logger = logging.getLogger(__name__)


def set_interventions_interpolation(
    interventions: Any,
    fn: Callable[..., torch.Tensor],
    **params: Any,
) -> None:
    """Push an interpolation function and its parameters onto every intervention.

    The function cannot ride in the intervention's ``forward`` signature (which is
    fixed to base/source/feature-indices), so it is set on the object first — the
    same pattern a mask intervention uses for its temperature.

    Args:
        interventions: The interventions to configure.
        fn: Callable with signature (f_base, f_src, **params) -> Tensor.
        **params: Keyword arguments forwarded to fn on each call.
    """
    for intervention in interventions:
        if hasattr(intervention, "set_interpolation"):
            intervention.set_interpolation(fn, **params)


def batched_interpolation_intervention(
    pipeline: Pipeline,
    examples: list[CounterfactualExample] | list[LabeledCounterfactualExample],
    interchange_target: InterchangeTarget,
    fn: Callable[..., torch.Tensor],
    params: dict[str, Any],
    output_scores: bool | int = True,
    interventions: Any = None,
) -> dict[str, Any]:
    """Perform interpolation interventions on a batch of examples.

    Args:
        pipeline: The pipeline containing the model.
        examples: List of counterfactual examples.
        interchange_target: InterchangeTarget containing model components to intervene on.
        fn: Interpolation function with signature (f_base, f_src, **params) -> Tensor.
        params: Keyword arguments forwarded to fn.
        output_scores: Whether to include scores in output dictionary (default: True).
        interventions: The run's interventions, one per flattened unit; built if
            not supplied.

    Returns:
        dict: Dictionary with 'sequences' and optionally 'scores' keys.
    """
    batch = prepare_interchange_batch(pipeline, examples, interchange_target)
    # Raw, like interchange: the interpolation featurizes the source itself.
    sources = collect_group_sources(pipeline, batch)
    if interventions is None:
        interventions = build_interventions(batch.units, "interpolation")
    set_interventions_interpolation(interventions, fn, **params)
    plans = build_plans(
        batch.units,
        batch.base_positions,
        "interpolation",
        sources=sources,
        feature_indices=batch.feature_indices,
        interventions=interventions,
    )
    result = generate_with_interventions(
        pipeline, batch.base_encoding, plans, output_scores=bool(output_scores)
    )
    return pipeline.format_generation(result, batch.base_encoding, output_scores)


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
    interventions = build_interventions(interchange_target.flatten(), "interpolation")

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
                examples,
                interchange_target,
                fn=fn,
                params=params,
                output_scores=output_scores,
                interventions=interventions,
            )
            all_outputs.append(output_dict)

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
