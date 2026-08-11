"""Interchange-mode execution over counterfactual datasets.

The public wrapper :func:`run_interchange_interventions` lowers a
counterfactual dataset onto the nnsight batched-execution engine
(:func:`causalab.neural.dataset.run_intervened_generation`) — one
tokenization per batch side, one early-stopped source collect per
counterfactual group (on ``source_pipeline``'s model when cross-model,
SH2 #411), ONE generate trace per base batch with prefill edits.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.neural.pipeline import GenerationResult, Pipeline, compress_scores_top_k
from causalab.neural.specs import SiteSpec

logger = logging.getLogger(__name__)


def run_interchange_interventions(
    pipeline: Pipeline,
    counterfactual_dataset: list[CounterfactualExample],
    groups: Sequence[Sequence[SiteSpec]],
    batch_size: int = 32,
    output_scores: bool | int = True,
    source_pipeline: Pipeline | None = None,
    gen_kwargs: dict[str, Any] | None = None,
) -> GenerationResult:
    """
    Run interchange interventions on a full counterfactual dataset in batches.

    This function:
    1. Prepares an intervenable model configured for interchange interventions
    2. Processes the dataset in batches, applying interventions to each batch
    3. Converts scores to top-k format if requested (for memory efficiency)
    4. Returns ONE flat :class:`~causalab.neural.pipeline.GenerationResult`
       over the whole dataset

    Args:
        pipeline: Target pipeline where interventions are applied
        counterfactual_dataset: List of counterfactual examples
        groups: Nested spec groups (``Sequence[Sequence[SiteSpec]]``) — the
            outer index ``g`` picks the counterfactual input feeding the
            group's source reads (``example["counterfactual_inputs"][g]``;
            the grouping contract on
            :func:`~causalab.neural.dataset.run_intervened_generation`).
        batch_size: Number of examples to process in each batch
        output_scores: Controls score output format:
            - False: No scores
            - True: Full vocabulary scores (on CPU)
            - int (e.g., 10): Top-k scores (on CPU, memory efficient)
        source_pipeline: If provided, collect activations from this pipeline instead
            of the target pipeline. Enables cross-model patching where activations
            from source_pipeline are patched into pipeline (the target).
        gen_kwargs: Extra HF ``generate`` kwargs forwarded to
            :func:`~causalab.neural.dataset.run_intervened_generation`
            (e.g. ``{"min_new_tokens": N}`` — the escape hatch its
            ragged-scores refusal names).

    Returns:
        One flat :class:`~causalab.neural.pipeline.GenerationResult` over the
        whole dataset (EU5b, #487): ``sequences`` ``(n_examples,
        max_new_tokens)``, ``strings`` one entry per example, and per-step
        ``scores`` (or ``scores_top_k`` for an integer ``output_scores``) —
        the internal batch split never appears in the shape. The io/artifact
        boundary converts via
        :meth:`~causalab.neural.pipeline.GenerationResult.to_raw_results`.
    """
    # PL3 (#405): rerouted onto the nnsight batched-execution engine — one
    # tokenization per batch side with batch-first position resolution, one
    # early-stopped source collect per counterfactual group, ONE generate
    # trace per base batch (prefill edits, pyvene's split-forward layout).
    # Cross-model patching rides the same path (SH2, #411): the collect
    # forwards run on source_pipeline's model with its own tokenization.
    from causalab.neural.dataset import run_intervened_generation
    from causalab.neural.specs import EditSpec

    edit_groups = [
        [EditSpec(site, mode="interchange") for site in group] for group in groups
    ]
    result = run_intervened_generation(
        pipeline,  # pyright: ignore[reportArgumentType]  # LMPipeline in practice
        counterfactual_dataset,
        edit_groups,
        batch_size=batch_size,
        output_scores=bool(output_scores),
        source_pipeline=source_pipeline,  # pyright: ignore[reportArgumentType]
        **(gen_kwargs or {}),
    )
    if not isinstance(output_scores, bool) and output_scores > 0:
        result = compress_scores_top_k(result, pipeline, k=output_scores)
    return result
