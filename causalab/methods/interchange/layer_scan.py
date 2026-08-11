"""Layer scan functions — distribution-level interchange interventions."""

import logging
from typing import Dict, Any, List, Callable

import torch
from tqdm import tqdm

from causalab.neural.activations.site_grids import SiteGrid
from causalab.neural.pipeline import GenerationResult, LMPipeline
from causalab.neural.activations.interchange_mode import run_interchange_interventions
from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.metric import (
    InterchangeMetric,
    score_intervention_outputs,
    _normalize_var_indices,  # pyright: ignore[reportPrivateUsage]
    _logits_to_class_probs,  # pyright: ignore[reportPrivateUsage]
)

logger = logging.getLogger(__name__)


def run_layer_scan(
    grid: SiteGrid,
    dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    batch_size: int,
    metric: InterchangeMetric,
    output_scores: bool | int = True,
    causal_model: CausalModel | None = None,
    original_outputs: List[Dict[str, Any]] | None = None,
    source_pipeline: LMPipeline | None = None,
    gen_kwargs: Dict[str, Any] | None = None,
) -> Dict[tuple[Any, ...], float]:
    """Run interchange interventions across grid cells and score with a metric.

    Unlike ``run_interchange_custom_score_heatmap`` this function:
    - Accepts an in-memory dataset (no file path)
    - Forwards ``output_scores`` to ``run_interchange_interventions``
      so metrics can access logits/scores
    - Returns the raw scores dict without forced visualization

    This makes it suitable for logit-level metrics (KL divergence, JS
    divergence, top-k overlap, etc.) and for use as a building block in
    task-specific pipelines that add their own visualization.

    Args:
        grid: :data:`~causalab.neural.activations.site_grids.SiteGrid` — dict
            mapping cell keys (e.g. ``(layer, pos_id)``) to nested
            ``SiteSpec`` groups (WU2 builder output).
        dataset: Counterfactual examples (in-memory).
        pipeline: Target LMPipeline.
        batch_size: Batch size for interventions.
        metric: InterchangeMetric whose ``fn`` receives the full per-example
            output dict (including ``"scores"`` when ``output_scores`` is
            truthy).
        output_scores: Forwarded to ``run_interchange_interventions``. Use
            ``True`` (full-vocab scores) when the metric reads scores; an
            ``int`` compresses to top-k, which the scorer refuses — here
            *before* the scan runs a single generation (the refusal is
            statically known, so it fires ahead of the loop rather than
            after the whole sweep).
        causal_model: Required when ``metric.needs_causal_expected``.
        original_outputs: Required when ``metric.needs_original_output``.
        source_pipeline: Optional source pipeline for cross-model patching.
        gen_kwargs: Extra HF ``generate`` kwargs forwarded to
            ``run_interchange_interventions`` (e.g. ``{"min_new_tokens": N}``
            — the escape hatch the engine's ragged-scores refusal names).

    Returns:
        Dict mapping grid keys to mean metric scores.
    """
    if not isinstance(output_scores, bool) and output_scores > 0:
        # Fail fast: score_intervention_outputs refuses top-k-compressed
        # scores unconditionally, so an int output_scores is a statically
        # known poison combination — refuse it before any generation work
        # (mirrors the scorer's wording, metric.py).
        raise ValueError(
            f"cannot score a layer scan run with output_scores={output_scores}: "
            "an int compresses to top-k scores (scores_top_k), but metrics "
            "consume full-vocabulary per-step tensors. Generate with "
            "output_scores=True (not an int) for metric scoring."
        )
    results: Dict[tuple[Any, ...], GenerationResult] = {}

    for key, groups in tqdm(
        grid.items(),
        desc="Layer scan",
        total=len(grid),
    ):
        results[key] = run_interchange_interventions(
            pipeline=pipeline,
            counterfactual_dataset=dataset,
            groups=groups,
            batch_size=batch_size,
            output_scores=output_scores,
            source_pipeline=source_pipeline,
            gen_kwargs=gen_kwargs,
        )

    return score_intervention_outputs(
        results=results,
        dataset=dataset,
        metric=metric,
        causal_model=causal_model,
        original_outputs=original_outputs,
    )


# ---------------------------------------------------------------------------
# Feature collection with caching
# ---------------------------------------------------------------------------


def collect_all_features_cached(
    grid: SiteGrid,
    dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    batch_size: int,
    output_dir: str | None,
) -> Dict[tuple[Any, ...], torch.Tensor]:
    """Collect features for all grid cells in a single forward pass, with per-layer caching.

    Checks the cache for each cell first.  Only cells with cache misses are
    collected, and those are collected together in one ``collect_features`` call
    so the model forward passes are shared across layers.

    Cache path layout is owned by ``causalab.io.artifacts``; this function only
    decides which keys are hot or cold.
    """
    from causalab.neural.activations.collect import collect_features
    from causalab.io.artifacts import load_cached_features, save_cached_features

    result: Dict[tuple[Any, ...], torch.Tensor] = {}
    uncached_keys: list[tuple[Any, ...]] = []

    # 1. Load what we can from cache
    for key in grid:
        layer, pos_id = key
        if output_dir is not None:
            cached = load_cached_features(output_dir, layer, pos_id, len(dataset))
            if cached is not None:
                result[key] = cached
                continue
        uncached_keys.append(key)

    if not uncached_keys:
        return result

    # 2. Collect all uncached cells in one pass
    all_specs = []
    spec_key_to_cell: dict[str, tuple[Any, ...]] = {}
    for key in uncached_keys:
        specs = [spec for group in grid[key] for spec in group]
        for spec in specs:
            spec_key_to_cell[spec.key] = key
        all_specs.extend(specs)

    logger.info(
        "Collecting features for %d targets in a single pass...", len(uncached_keys)
    )
    features_dict = collect_features(
        dataset,
        pipeline,
        all_specs,
        batch_size=batch_size,
    )
    # collect_output_logits is False (default), so the return is dict[str, Tensor]
    assert isinstance(features_dict, dict)

    # 3. Map back to keys and cache
    for spec_key, features in features_dict.items():
        key = spec_key_to_cell[spec_key]
        result[key] = features
        if output_dir is not None:
            layer, pos_id = key
            save_cached_features(output_dir, layer, pos_id, features)

    return result


# ---------------------------------------------------------------------------
# Centroid patching
# ---------------------------------------------------------------------------


def run_centroid_layer_scan(
    grid: SiteGrid,
    dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    batch_size: int,
    score_token_ids: list[int] | list[list[int]],
    n_classes: int,
    example_to_class: Any,
    ref_dists: torch.Tensor,
    score_token_index: int = 0,
    n_steer: int = 50,
    output_dir: str | None = None,
    precomputed_features: Dict[tuple[Any, ...], torch.Tensor] | None = None,
    comparison_fn: Callable | None = None,
    return_patched_dists: bool = False,
    source_pipeline: LMPipeline | None = None,
    gen_kwargs: Dict[str, Any] | None = None,
) -> (
    Dict[tuple[Any, ...], float]
    | tuple[Dict[tuple[Any, ...], float], Dict[tuple[Any, ...], torch.Tensor]]
):
    """Centroid interchange: compute per-class centroid activations, patch each
    centroid into test examples, and compare output to class-average distribution.

    For each class (node):
    1. Average all training activations for that class -> centroid
    2. Patch centroid into all test examples
    3. comparison_fn(ref_dists[class], patched_output)

    Supports multi-token classes via ``list[list[int]]`` — joint probabilities
    are computed by multiplying across generation steps.

    Args:
        precomputed_features: If provided, skip feature collection and use these
            directly. Keys must match ``grid`` keys.
        comparison_fn: Distribution comparison function ``(N, C), (N, C) -> (N,)``.
        return_patched_dists: If True, also return per-layer mean patched
            distributions as ``(n_classes, n_score_tokens)`` tensors.
            Defaults to ``kl_divergence``.
        source_pipeline: If provided, activations (centroids) are collected from
            this pipeline instead of ``pipeline``.  The centroids are then patched
            into ``pipeline`` (the target).  Enables cross-model patching.
        gen_kwargs: Extra HF ``generate`` kwargs forwarded to
            ``run_intervened_generation`` (e.g. ``{"min_new_tokens": N}`` —
            the escape hatch its ragged-scores refusal names).

    Returns dict mapping grid keys to mean score across classes.
    """
    if comparison_fn is None:
        raise ValueError(
            "comparison_fn is required for centroid_layer_scan. "
            "Ensure the intervention_metric resolves to a distribution comparison."
        )
    cmp = comparison_fn
    from causalab.neural.dataset import run_intervened_generation
    from causalab.neural.specs import EditSpec

    token_seqs = _normalize_var_indices(score_token_ids)
    n_steps = max(len(seq) for seq in token_seqs)

    # Use a subset as base inputs for steering
    steer_subset = dataset[:n_steer]
    base_examples = [
        {"input": ex["input"], "counterfactual_inputs": [ex["input"]]}
        for ex in steer_subset
    ]

    # Collect all features upfront in a single forward pass (skip if precomputed).
    # When source_pipeline is provided, collect from the source model so that
    # centroids represent that model's representation of the variable.
    feature_pipeline = source_pipeline if source_pipeline is not None else pipeline
    if precomputed_features is not None:
        all_features = precomputed_features
    else:
        all_features = collect_all_features_cached(
            grid,
            dataset,
            feature_pipeline,
            batch_size,
            output_dir,
        )

    # Precompute class assignments (same for all cells)
    example_classes = [example_to_class(ex) for ex in dataset]
    class_counts = torch.zeros(n_classes)
    for cls in example_classes:
        class_counts[cls] += 1

    scores: Dict[tuple[Any, ...], float] = {}
    patched_dists: Dict[tuple[Any, ...], torch.Tensor] = {}

    for key, cell_groups in tqdm(
        grid.items(),
        desc="Centroid layer scan",
        total=len(grid),
    ):
        features = all_features[key]

        # Centroids live in each site's FEATURE space (e.g. PCA), where the
        # replace-mode edit writes; averaging there avoids noise from discarded
        # dimensions biasing the mean. The engine re-featurizes the base at the
        # write and keeps *base's* error term (the ST3 contract) — the class-
        # average error the old raw-space roundtrip embedded was likewise
        # dropped when the intervention re-featurized its source.
        specs = [spec for group in cell_groups for spec in group]
        featurizer = specs[0].fsite.featurizer if specs else None
        if featurizer is not None and hasattr(featurizer, "featurizer"):
            # Features may already be in feature space (e.g. PCA-projected)
            # if the collect intervention applied the featurizer during collection.
            already_projected = (
                featurizer.n_features is not None
                and features.shape[-1] == featurizer.n_features
            )
            if already_projected:
                projected = features
            else:
                with torch.no_grad():
                    projected, _ = featurizer.featurizer(features)
        else:
            # No featurizer — raw space is the (identity) feature space
            projected = features

        centroids = torch.zeros(n_classes, projected.shape[1])
        for i, cls in enumerate(example_classes):
            centroids[cls] += projected[i]
        for c in range(n_classes):
            if class_counts[c] > 0:
                centroids[c] /= class_counts[c]

        n_score_tokens = len(token_seqs)
        patched_accum = (
            torch.zeros(n_classes, n_score_tokens) if return_patched_dists else None
        )
        patched_counts = torch.zeros(n_classes) if return_patched_dists else None

        kl_per_class: list[float] = []
        for cls in range(n_classes):
            if class_counts[cls] == 0:
                continue

            # Patch this class's centroid into every base example: one
            # replace-mode edit per site (a broadcast feature-space vector
            # needs no counterfactual forward), batched by the engine.
            groups = [
                [
                    EditSpec(spec, mode="replace", vector=centroids[cls])
                    for spec in group
                ]
                for group in cell_groups
            ]
            # The engine returns ONE flat GenerationResult; its per-step
            # scores are (n_examples, vocab), so the retired per-batch score
            # loop collapses to flat indexing (EU5b, #487). Ragged internal
            # batches (one batch early-EOSing at fewer steps than another)
            # refuse loudly inside run_intervened_generation rather than
            # silently dropping the short batches the way the legacy loop
            # did — the error message names the escape hatches
            # (batch_size >= len(dataset), or min_new_tokens=max_new_tokens).
            output = run_intervened_generation(
                pipeline,
                base_examples,  # pyright: ignore[reportArgumentType]  # constructed dicts conform to CounterfactualExample TypedDict at runtime
                groups,
                batch_size=batch_size,
                output_scores=True,
                **(gen_kwargs or {}),
            )

            cls_kls: list[float] = []
            step_scores = output.scores or []
            if len(step_scores) > score_token_index:
                logits_per_step = [
                    step_scores[score_token_index + k]
                    for k in range(n_steps)
                    if score_token_index + k < len(step_scores)
                ]
                probs = _logits_to_class_probs(logits_per_step, token_seqs).cpu()
                n_examples = probs.shape[0]

                # Compare ref vs patched for this class
                ref = ref_dists[cls].unsqueeze(0).expand(n_examples, -1)
                per_example = cmp(ref, probs)
                cls_kls = per_example.tolist()

                # Accumulate patched distributions (full vocab softmax for heatmaps)
                if patched_accum is not None:
                    assert patched_counts is not None  # set together with patched_accum
                    probs_fvs = _logits_to_class_probs(
                        logits_per_step, token_seqs, full_vocab_softmax=True
                    ).cpu()
                    patched_accum[cls] += probs_fvs.sum(dim=0)
                    patched_counts[cls] += n_examples
            if cls_kls:
                kl_per_class.append(sum(cls_kls) / len(cls_kls))

        scores[key] = (
            sum(kl_per_class) / len(kl_per_class) if kl_per_class else float("nan")
        )

        if patched_accum is not None:
            assert patched_counts is not None  # set together with patched_accum
            # Average across examples
            for c in range(n_classes):
                if patched_counts[c] > 0:
                    patched_accum[c] /= patched_counts[c]
            patched_dists[key] = patched_accum

    if return_patched_dists:
        return scores, patched_dists
    return scores
