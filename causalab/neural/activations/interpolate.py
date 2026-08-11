"""Interpolation interventions — activations patched as an arbitrary function
of the base and source featurized activations:

    new_act = inverse_featurizer(fn(f_base, f_src, **params), base_err)

The canonical use case is linear interpolation:

    def linear_interp(f_base, f_src, alpha):
        return (1 - alpha) * f_base + alpha * f_src

At alpha=1 this reduces to interchange; at alpha=0 it is the identity.

SH2 (#411): the runner is a thin delegation onto the nnsight batched-execution
engine (``EditSpec(mode="interpolate")`` over
:func:`causalab.neural.dataset.run_intervened_generation`), exactly like the
other public wrappers; ``fn`` is applied in-trace via ``FeaturizedSite.edit``
with keyword arguments ``f_base`` / ``f_src``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch

from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from causalab.neural.pipeline import GenerationResult, Pipeline
from causalab.neural.specs import SiteSpec

__all__ = ["run_interpolation_interventions"]


def run_interpolation_interventions(
    pipeline: Pipeline,
    counterfactual_dataset: list[CounterfactualExample]
    | list[LabeledCounterfactualExample],
    groups: Sequence[Sequence[SiteSpec]],
    fn: Callable[..., torch.Tensor] | None = None,
    params: dict[str, Any] | None = None,
    batch_size: int = 32,
    output_scores: bool | int = True,
    gen_kwargs: dict[str, Any] | None = None,
) -> GenerationResult:
    """Run interpolation interventions on a full counterfactual dataset in batches.

    Args:
        pipeline: The pipeline containing the model.
        counterfactual_dataset: List of counterfactual examples.
        groups: Nested spec groups (``Sequence[Sequence[SiteSpec]]``) — the
            outer index ``g`` picks the counterfactual input feeding the
            group's source reads (the grouping contract on
            :func:`~causalab.neural.dataset.run_intervened_generation`).
        fn: Interpolation function with signature ``(f_base, f_src, **params)``
            (called with ``f_base`` / ``f_src`` as keywords).
        params: Keyword arguments forwarded to fn on each call.
        batch_size: Number of examples to process in each batch.
        output_scores: Controls score output format:
            - False: No scores
            - True: Full vocabulary scores (on CPU)
            - int (e.g., 10): Top-k scores (on CPU, memory efficient)
        gen_kwargs: Extra HF ``generate`` kwargs forwarded to
            :func:`~causalab.neural.dataset.run_intervened_generation`
            (e.g. ``{"min_new_tokens": N}`` — the escape hatch its
            ragged-scores refusal names).

    Returns:
        One flat :class:`~causalab.neural.pipeline.GenerationResult` over the
        whole dataset (EU5b, #487; scores on CPU, ``scores_top_k`` set instead
        of ``scores`` when an int was provided).
    """
    if fn is None:
        raise TypeError("run_interpolation_interventions requires fn")

    from causalab.neural.dataset import run_intervened_generation
    from causalab.neural.pipeline import compress_scores_top_k
    from causalab.neural.specs import EditSpec

    edit_groups = [
        [
            EditSpec(
                site,
                mode="interpolate",
                interpolate_fn=fn,
                interpolate_params=params or {},
            )
            for site in group
        ]
        for group in groups
    ]
    result = run_intervened_generation(
        pipeline,  # pyright: ignore[reportArgumentType]  # LMPipeline in practice
        counterfactual_dataset,
        edit_groups,
        batch_size=batch_size,
        output_scores=bool(output_scores),
        **(gen_kwargs or {}),
    )
    if not isinstance(output_scores, bool) and output_scores > 0:
        result = compress_scores_top_k(result, pipeline, k=output_scores)
    return result
