"""Layer scan functions — distribution-level interchange interventions."""

import logging
import math
from typing import Dict, Any, List, Callable

import torch
from tqdm import tqdm

from causalab.neural.pipeline import LMPipeline
from causalab.neural.units import InterchangeTarget
from causalab.neural.activations.interchange_mode import run_interchange_interventions
from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.metric import (
    InterchangeMetric,
    score_intervention_outputs,
    _normalize_var_indices,
    _logits_to_class_probs,
)

logger = logging.getLogger(__name__)


def run_layer_scan(
    interchange_targets: Dict[tuple[Any, ...], InterchangeTarget],
    dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    batch_size: int,
    metric: InterchangeMetric,
    output_scores: bool | int = True,
    causal_model: CausalModel | None = None,
    original_outputs: List[Dict[str, Any]] | None = None,
    source_pipeline: LMPipeline | None = None,
) -> Dict[tuple[Any, ...], float]:
    """Run interchange interventions across targets and score with a metric.

    Unlike ``run_interchange_custom_score_heatmap`` this function:
    - Accepts an in-memory dataset (no file path)
    - Forwards ``output_scores`` to ``run_interchange_interventions``
      so metrics can access logits/scores
    - Returns the raw scores dict without forced visualization

    This makes it suitable for logit-level metrics (KL divergence, JS
    divergence, top-k overlap, etc.) and for use as a building block in
    task-specific pipelines that add their own visualization.

    Args:
        interchange_targets: Dict mapping keys (e.g. ``(layer, pos_id)``)
            to InterchangeTarget objects.
        dataset: Counterfactual examples (in-memory).
        pipeline: Target LMPipeline.
        batch_size: Batch size for interventions.
        metric: InterchangeMetric whose ``fn`` receives the full per-example
            output dict (including ``"scores"`` when ``output_scores`` is
            truthy).
        output_scores: Forwarded to ``run_interchange_interventions``.
            ``True`` returns full vocab scores, an ``int`` returns top-k.
        causal_model: Required when ``metric.needs_causal_expected``.
        original_outputs: Required when ``metric.needs_original_output``.
        source_pipeline: Optional source pipeline for cross-model patching.

    Returns:
        Dict mapping target keys to mean metric scores.
    """
    raw_results: Dict[tuple[Any, ...], Dict[str, Any]] = {}

    for key, target in tqdm(
        interchange_targets.items(),
        desc="Layer scan",
        total=len(interchange_targets),
    ):
        raw_results[key] = run_interchange_interventions(
            pipeline=pipeline,
            counterfactual_dataset=dataset,
            interchange_target=target,
            batch_size=batch_size,
            output_scores=output_scores,
            source_pipeline=source_pipeline,
        )

    return score_intervention_outputs(
        raw_results=raw_results,
        dataset=dataset,
        metric=metric,
        causal_model=causal_model,
        original_outputs=original_outputs,
    )


# ---------------------------------------------------------------------------
# Feature collection with caching
# ---------------------------------------------------------------------------


def collect_all_features_cached(
    interchange_targets: Dict[tuple[Any, ...], InterchangeTarget],
    dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    batch_size: int,
    output_dir: str | None,
) -> Dict[tuple[Any, ...], torch.Tensor]:
    """Collect features for all targets in a single forward pass, with per-layer caching.

    Checks the cache for each target first.  Only targets with cache misses are
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
    for key in interchange_targets:
        layer, pos_id = key
        if output_dir is not None:
            cached = load_cached_features(output_dir, layer, pos_id, len(dataset))
            if cached is not None:
                result[key] = cached
                continue
        uncached_keys.append(key)

    if not uncached_keys:
        return result

    # 2. Collect all uncached targets in one pass
    all_units = []
    unit_id_to_key: dict[str, tuple[Any, ...]] = {}
    for key in uncached_keys:
        target = interchange_targets[key]
        units = target.flatten()
        for u in units:
            unit_id_to_key[u.id] = key
        all_units.extend(units)

    logger.info(
        "Collecting features for %d targets in a single pass...", len(uncached_keys)
    )
    features_dict = collect_features(
        dataset=dataset,
        pipeline=pipeline,
        model_units=all_units,
        batch_size=batch_size,
    )

    # 3. Map back to keys and cache
    for unit_id, features in features_dict.items():
        key = unit_id_to_key[unit_id]
        result[key] = features
        if output_dir is not None:
            layer, pos_id = key
            save_cached_features(output_dir, layer, pos_id, features)

    return result


# ---------------------------------------------------------------------------
# Pairwise patching with per-example comparison
# ---------------------------------------------------------------------------


def run_pairwise_layer_scan(
    interchange_targets: Dict[tuple[Any, ...], InterchangeTarget],
    dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    batch_size: int,
    score_token_ids: list[int] | list[list[int]],
    score_token_index: int = 0,
    causal_model: CausalModel | None = None,
    source_pipeline: LMPipeline | None = None,
    output_dir: str | None = None,
    comparison_fn: Callable | None = None,
) -> Dict[tuple[Any, ...], float]:
    """Pairwise interchange: patch each counterfactual's activation and compare
    to that specific counterfactual's own output (not the class average).

    For each (base, cf) pair:
    1. Run cf through the model -> get cf's output distribution
    2. Patch cf's activation into base -> get patched output distribution
    3. comparison_fn(cf_output, patched_output)

    Supports multi-token classes via ``list[list[int]]`` — joint probabilities
    are computed by multiplying across generation steps.

    Args:
        comparison_fn: Distribution comparison function ``(N, C), (N, C) -> (N,)``.
            Defaults to ``kl_divergence``.

    Returns dict mapping target keys to mean score across examples.
    """
    if comparison_fn is None:
        raise ValueError(
            "comparison_fn is required for pairwise_layer_scan. "
            "Ensure the intervention_metric resolves to a distribution comparison."
        )
    cmp = comparison_fn

    token_seqs = _normalize_var_indices(score_token_ids)
    n_steps = max(len(seq) for seq in token_seqs)

    # Step 1: Collect counterfactual output distributions
    logger.info("Collecting counterfactual output distributions...")
    cf_inputs = [ex["counterfactual_inputs"][0] for ex in dataset]
    cf_probs_list: list[torch.Tensor] = []
    n_batches = math.ceil(len(cf_inputs) / batch_size)
    for bi in range(n_batches):
        batch = cf_inputs[bi * batch_size : (bi + 1) * batch_size]
        result = pipeline.generate(batch)
        gen_scores = result["scores"]
        logits_per_step = [
            gen_scores[score_token_index + k]
            for k in range(n_steps)
            if score_token_index + k < len(gen_scores)
        ]
        batch_probs = _logits_to_class_probs(logits_per_step, token_seqs)
        cf_probs_list.append(batch_probs.cpu())
    cf_probs = torch.cat(cf_probs_list, dim=0)  # (N, n_classes)

    # Collect and cache subspace features if output_dir is set
    if output_dir is not None:
        collect_all_features_cached(
            interchange_targets,
            dataset,
            pipeline,
            batch_size,
            output_dir,
        )

    # Step 2: Run interchange interventions and compare per-example
    scores: Dict[tuple[Any, ...], float] = {}

    for key, target in tqdm(
        interchange_targets.items(),
        desc="Pairwise layer scan",
        total=len(interchange_targets),
    ):
        raw = run_interchange_interventions(
            pipeline=pipeline,
            counterfactual_dataset=dataset,
            interchange_target=target,
            batch_size=batch_size,
            output_scores=True,
            source_pipeline=source_pipeline,
        )

        # Flatten scores across batches
        raw_scores = raw.get("scores")
        if raw_scores is None or not raw_scores:
            scores[key] = float("nan")
            continue

        n_tokens = len(raw_scores[0])
        flat_scores = [
            torch.cat([batch_scores[t] for batch_scores in raw_scores], dim=0)
            for t in range(n_tokens)
        ]

        logits_per_step = [
            flat_scores[score_token_index + k]
            for k in range(n_steps)
            if score_token_index + k < len(flat_scores)
        ]
        patched_probs = _logits_to_class_probs(logits_per_step, token_seqs).cpu()

        # Compare cf vs patched per example
        scores_per_example = cmp(cf_probs, patched_probs)
        scores[key] = scores_per_example.mean().item()

    return scores


# ---------------------------------------------------------------------------
# Centroid patching
# ---------------------------------------------------------------------------


def run_centroid_layer_scan(
    interchange_targets: Dict[tuple[Any, ...], InterchangeTarget],
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
            directly. Keys must match ``interchange_targets`` keys.
        comparison_fn: Distribution comparison function ``(N, C), (N, C) -> (N,)``.
        return_patched_dists: If True, also return per-layer mean patched
            distributions as ``(n_classes, n_score_tokens)`` tensors.
            Defaults to ``kl_divergence``.
        source_pipeline: If provided, activations (centroids) are collected from
            this pipeline instead of ``pipeline``.  The centroids are then patched
            into ``pipeline`` (the target).  Enables cross-model patching.

    Returns dict mapping target keys to mean score across classes.
    """
    if comparison_fn is None:
        raise ValueError(
            "comparison_fn is required for centroid_layer_scan. "
            "Ensure the intervention_metric resolves to a distribution comparison."
        )
    cmp = comparison_fn
    from causalab.neural.activations.interchange_mode import (
        prepare_intervenable_model,
        delete_intervenable_model,
        prepare_intervenable_inputs,
    )

    from causalab.neural.activations.intervenable_model import device_for_layer

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
            interchange_targets,
            dataset,
            feature_pipeline,
            batch_size,
            output_dir,
        )

    # Precompute class assignments (same for all targets)
    example_classes = [example_to_class(ex) for ex in dataset]
    class_counts = torch.zeros(n_classes)
    for cls in example_classes:
        class_counts[cls] += 1

    n_steer_batches = math.ceil(len(base_examples) / batch_size)

    scores: Dict[tuple[Any, ...], float] = {}
    patched_dists: Dict[tuple[Any, ...], torch.Tensor] = {}

    for key, target in tqdm(
        interchange_targets.items(),
        desc="Centroid layer scan",
        total=len(interchange_targets),
    ):
        features = all_features[key]

        # Project features through featurizer before averaging, so centroids
        # are computed in the subspace (e.g. PCA) rather than raw activation
        # space.  This avoids noise from discarded dimensions biasing the mean.
        # Then inverse-project back to raw space for pyvene patching.
        units = target.flatten()
        featurizer = units[0].featurizer if units else None
        if featurizer is not None and hasattr(featurizer, "featurizer"):
            # Features may already be in feature space (e.g. PCA-projected)
            # if the collect intervention applied the featurizer during collection.
            already_projected = (
                featurizer.n_features is not None
                and features.shape[-1] == featurizer.n_features
            )
            if already_projected:
                projected = features
                error = None
            else:
                with torch.no_grad():
                    projected, error = featurizer.featurizer(features)

            # Average in projected space
            centroids_proj = torch.zeros(n_classes, projected.shape[1])
            error_accum = (
                torch.zeros(n_classes, error.shape[1]) if error is not None else None
            )
            for i, cls in enumerate(example_classes):
                centroids_proj[cls] += projected[i]
                if error_accum is not None:
                    error_accum[cls] += error[i]
            for c in range(n_classes):
                if class_counts[c] > 0:
                    centroids_proj[c] /= class_counts[c]
                    if error_accum is not None:
                        error_accum[c] /= class_counts[c]

            # Inverse-project back to raw activation space
            with torch.no_grad():
                centroids = featurizer.inverse_featurizer(
                    centroids_proj,
                    error_accum,
                )
        else:
            # No featurizer — average directly in raw space
            centroids = torch.zeros(n_classes, features.shape[1])
            for i, cls in enumerate(example_classes):
                centroids[cls] += features[i]
            for c in range(n_classes):
                if class_counts[c] > 0:
                    centroids[c] /= class_counts[c]

        # For each class, patch its centroid into base examples and score
        intervenable_model = prepare_intervenable_model(pipeline, target)
        # Centroids are scattered into the target layer's tensor — they must
        # live on that layer's device (which may differ across GPUs for
        # sharded models).
        target_layers = {u.layer for u in target.flatten()}
        device = device_for_layer(pipeline, next(iter(target_layers)))

        n_score_tokens = len(token_seqs)
        patched_accum = (
            torch.zeros(n_classes, n_score_tokens) if return_patched_dists else None
        )
        patched_counts = torch.zeros(n_classes) if return_patched_dists else None

        kl_per_class: list[float] = []
        for cls in range(n_classes):
            if class_counts[cls] == 0:
                continue

            centroid = centroids[cls]  # (hidden_dim,)

            # Patch centroid into all base examples
            cls_kls: list[float] = []
            for bi in range(n_steer_batches):
                batch = base_examples[bi * batch_size : (bi + 1) * batch_size]
                bs = len(batch)

                batched_base, _, inv_locations, feature_indices = (
                    prepare_intervenable_inputs(pipeline, batch, target)
                )
                # pyvene expects (batch, seq, dim) — add seq dim
                source_repr = [centroid.view(1, 1, -1).expand(bs, 1, -1).to(device)]

                gen_kwargs = {"output_scores": True}
                output = pipeline.intervenable_generate(
                    intervenable_model,
                    batched_base,
                    None,
                    inv_locations,
                    feature_indices,
                    source_representations=source_repr,
                    **gen_kwargs,
                )

                out_scores = output.get("scores", [])
                if len(out_scores) <= score_token_index:
                    continue
                logits_per_step = [
                    out_scores[score_token_index + k]
                    for k in range(n_steps)
                    if score_token_index + k < len(out_scores)
                ]
                probs = _logits_to_class_probs(logits_per_step, token_seqs).cpu()

                # Compare ref vs patched for this class
                ref = ref_dists[cls].unsqueeze(0).expand(bs, -1)
                per_example = cmp(ref, probs)
                cls_kls.extend(per_example.tolist())

                # Accumulate patched distributions (full vocab softmax for heatmaps)
                if patched_accum is not None:
                    probs_fvs = _logits_to_class_probs(
                        logits_per_step, token_seqs, full_vocab_softmax=True
                    ).cpu()
                    patched_accum[cls] += probs_fvs.sum(dim=0)
                    patched_counts[cls] += bs

            if cls_kls:
                kl_per_class.append(sum(cls_kls) / len(cls_kls))

        delete_intervenable_model(intervenable_model)

        scores[key] = (
            sum(kl_per_class) / len(kl_per_class) if kl_per_class else float("nan")
        )

        if patched_accum is not None:
            # Average across examples
            for c in range(n_classes):
                if patched_counts[c] > 0:
                    patched_accum[c] /= patched_counts[c]
            patched_dists[key] = patched_accum

    if return_patched_dists:
        return scores, patched_dists
    return scores
