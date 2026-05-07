"""Pullback analysis: geodesic belief paths and embedding optimization.

Extracted from ``causalab.runner.run_exp.run_pullback`` for standalone use.
"""

from __future__ import annotations

import json
import logging
import os
import random as _random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from causalab.runner.helpers import (
    resolve_task,
    build_targets_for_grid,
    _task_config_for_metadata,
    generate_datasets,
)
from causalab.io.pipelines import (
    load_pipeline,
    find_subspace_dirs,
    find_activation_manifold_dirs,
    load_subspace_metadata,
)
from causalab.neural.activations.intervenable_model import device_for_layer
from causalab.io.nested_artifacts import load_nested, save_nested
from causalab.io.plots.figure_format import resolve_figure_format_from_analysis
from causalab.io.plots.plot_utils import resolve_task_colormap

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "pullback"


def _log_cache_misses(label: str, missing: list, max_items: int = 20) -> None:
    """Loud warning for missing cache entries under replot_only=True; never raises."""
    if not missing:
        return
    head = missing[:max_items]
    suffix = "" if len(missing) <= max_items else f" (+{len(missing) - max_items} more)"
    logger.error(
        "replot_only: %d %s entries missing from cache%s: %s",
        len(missing),
        label,
        suffix,
        head,
    )
    logger.error("  → these will be skipped in visualization.")


def _normalize_k_opt(k_opt_cfg) -> list[tuple[str, int | None]]:
    """Normalize embedding_optim.k_opt into an ordered list of (label, value) pairs.

    Accepts a scalar (`None` or `int`) or a list mixing both. Labels: ``None`` →
    ``"full"``, ``int k`` → ``f"k{k}"``. Duplicates collapsed, order preserved.
    """
    from omegaconf import ListConfig

    if isinstance(k_opt_cfg, (list, ListConfig)):
        items = list(k_opt_cfg)
    else:
        items = [k_opt_cfg]
    out: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for v in items:
        vi = None if v is None else int(v)
        label = "full" if vi is None else f"k{vi}"
        if label in seen:
            continue
        seen.add(label)
        out.append((label, vi))
    if not out:
        out.append(("full", None))
    return out


def _coerce_results_by_kopt(
    raw: dict, default_label: str = "full"
) -> dict[str, dict[tuple[int, int], dict]]:
    """Normalize an on-disk optimization cache to ``{k_opt_label: {pair: result}}``.

    Old caches are flat ``{pair: result}``; transparently lifts them under
    ``default_label`` with a one-line warning so prior runs keep working.
    """
    if not raw:
        return {}
    sample_key = next(iter(raw))
    if isinstance(sample_key, tuple):
        logger.warning(
            "optimization_results.pt is in legacy flat format — migrating in-memory "
            "to nested {%r: {pair: result}}; re-run without replot_only to rewrite.",
            default_label,
        )
        return {default_label: dict(raw)}
    return {str(label): dict(per) for label, per in raw.items()}


def _attach_recapitulation_metrics_and_aggregate(
    results: dict[tuple[int, int], dict],
    concept_names: list[str],
    label: str,
    compute_aggregate: bool,
) -> dict | None:
    """Attach path-recapitulation metrics to each pair (in-place) + aggregate.

    Three parameterization-invariant scalars per (path family, pair):
    ``r_squared`` (shape recapitulation), ``mean_dist_from_geometric``
    (k-space distance to v_geo polyline), ``arc_length_ratio`` (coverage /
    detour). Computed for ``optimized`` and ``linear`` against ``v_geometric_k``.

    Args:
        results: per-pair results for one k_opt label. Each entry must have
            ``v_geometric_k``; optionally ``v_optimized_k`` / ``v_linear_k``.
            Mutated to add ``recapitulation_optimized`` / ``recapitulation_linear``.
        concept_names: index → name for per-pair report keys.
        label: k_opt label, threaded into the summary + log.
        compute_aggregate: when False (cherry-picked ``selected_pairs``), skip
            the aggregate summary + paired t-test (biased on a non-random subset).

    Returns:
        Aggregate summary dict, or ``None`` when ``compute_aggregate=False``
        or no pair has both optimized and linear metrics.
    """
    from causalab.methods.pullback.optimization import path_recapitulation_metrics

    for (ci, cj), res in results.items():
        v_geo = res.get("v_geometric_k")
        if v_geo is None:
            continue
        if "v_optimized_k" in res:
            res["recapitulation_optimized"] = path_recapitulation_metrics(
                res["v_optimized_k"],
                v_geo,
            )
        if "v_linear_k" in res:
            res["recapitulation_linear"] = path_recapitulation_metrics(
                res["v_linear_k"],
                v_geo,
            )

    if not compute_aggregate:
        return None

    pair_keys = [
        p
        for p in results
        if "recapitulation_optimized" in results[p]
        and "recapitulation_linear" in results[p]
    ]
    n = len(pair_keys)
    if n == 0:
        return None

    metric_names = ("r_squared", "mean_dist_from_geometric", "arc_length_ratio")
    opt_vals = {
        m: torch.tensor([results[p]["recapitulation_optimized"][m] for p in pair_keys])
        for m in metric_names
    }
    lin_vals = {
        m: torch.tensor([results[p]["recapitulation_linear"][m] for p in pair_keys])
        for m in metric_names
    }

    def _agg(t: Tensor) -> dict[str, float]:
        finite = t[torch.isfinite(t)]
        n_f = finite.numel()
        return {
            "mean": float(finite.mean()) if n_f > 0 else float("nan"),
            "se": (
                float(finite.std(unbiased=True) / (n_f**0.5))
                if n_f > 1
                else float("nan")
            ),
        }

    summary: dict = {
        "n_pairs": n,
        "k_opt_label": label,
        "optimized": {m: _agg(opt_vals[m]) for m in metric_names},
        "linear": {m: _agg(lin_vals[m]) for m in metric_names},
        "per_pair": {
            f"{concept_names[ci]}->{concept_names[cj]}": {
                "optimized": results[(ci, cj)]["recapitulation_optimized"],
                "linear": results[(ci, cj)]["recapitulation_linear"],
            }
            for ci, cj in pair_keys
        },
    }

    if n >= 2:
        from scipy import stats as _stats

        t_r2, p_r2 = _stats.ttest_rel(
            opt_vals["r_squared"].numpy(),
            lin_vals["r_squared"].numpy(),
        )
        summary["paired_t_test_r_squared_optimized_vs_linear"] = {
            "t_statistic": float(t_r2),
            "p_value": float(p_r2),
            "mean_diff": float((opt_vals["r_squared"] - lin_vals["r_squared"]).mean()),
        }

    logger.info(
        "  [%s] path recapitulation  "
        "opt R²=%.3f dist=%.3f arc=%.3f  "
        "lin R²=%.3f dist=%.3f arc=%.3f  (n=%d)",
        label,
        summary["optimized"]["r_squared"]["mean"],
        summary["optimized"]["mean_dist_from_geometric"]["mean"],
        summary["optimized"]["arc_length_ratio"]["mean"],
        summary["linear"]["r_squared"]["mean"],
        summary["linear"]["mean_dist_from_geometric"]["mean"],
        summary["linear"]["arc_length_ratio"]["mean"],
        n,
    )

    return summary


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the pullback analysis: geodesic belief paths + embedding optimization.

    Caching model:
      - replot_only=False (default): full recompute. Geodesic and embedding-optim
        results are computed for ``selected_pair_indices`` and merged into the
        on-disk caches (``geodesic_paths.pt``, ``optimization_results.pt``).
        Untouched pair entries from prior runs are preserved.
      - replot_only=True: skip the expensive optimization. Caches are loaded
        and ``path_recapitulation_*.json`` is recomputed from cached paths
        (so iterating on metric definitions doesn't require re-running LBFGS),
        then visualization runs. Missing pair entries are logged loudly.
    """
    from safetensors.torch import load_file as load_safetensors

    from causalab.methods.pullback.geodesic import (
        compute_manifold_trace_paths,
        load_natural_distributions,
    )
    from causalab.methods.metric import tokenize_variable_values

    analysis = cfg[ANALYSIS_NAME]
    root = cfg.experiment_root
    figure_fmt = resolve_figure_format_from_analysis(analysis)

    # 1. Discover subspace x activation_manifold combinations
    ss_filter = analysis.subspace
    m_filter = analysis.activation_manifold
    if ss_filter is not None:
        ss_list = [ss_filter]
    else:
        ss_list = find_subspace_dirs(root)
    combos: list[tuple[str, str]] = []
    for ss in ss_list:
        if m_filter is not None:
            combos.append((ss, m_filter))
        else:
            for m in find_activation_manifold_dirs(root, ss):
                combos.append((ss, m))
    if not combos:
        raise ValueError(
            f"No activation_manifold results found in {root}. "
            "Run subspace and activation_manifold analyses first."
        )
    logger.info("Pullback: found %d subspace/activation_manifold combos", len(combos))

    replot_only = bool(analysis.get("replot_only", False))
    if replot_only:
        if analysis.selected_pairs is None:
            logger.info(
                "replot_only=True (no selected_pairs) — loading all cached "
                "pairs; metrics recomputed and visualization re-rendered."
            )
        else:
            logger.info(
                "replot_only=True — loading from cache for %d named pair(s); "
                "missing entries will be logged and skipped.",
                len(analysis.selected_pairs),
            )

    # Use the first combo for the output dir (single-combo case).
    ss_sub, m_sub = combos[0]
    out_dir = Path(root) / "pullback" / ss_sub / m_sub
    tv = cfg.task.get("target_variable")
    if tv:
        out_dir = out_dir / tv
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualization"
    vis_dir.mkdir(exist_ok=True)

    belief_path_cfg = analysis.belief_path

    # ── Resolve task + pair selection (lightweight — no model) ────────────
    task, _task_cfg_raw = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )
    concept_values = list(task.intervention_values)
    concept_names = [str(v) for v in concept_values]
    W = len(concept_names)
    value_to_idx = {v: i for i, v in enumerate(concept_values)}

    all_pull_pairs = [(i, j) for i in range(W) for j in range(i + 1, W)]
    max_pairs_cfg = analysis.get("max_pairs")

    if analysis.selected_pairs is not None:
        # Cherry-picked subset — compute only the named pairs (under replot_only,
        # load only those from cache). Aggregate metrics are skipped at write
        # time; warn here so the reason is visible at the start of the run.
        named_pairs = [
            (value_to_idx[a], value_to_idx[b]) for a, b in analysis.selected_pairs
        ]
        selected_pair_indices = sorted(set(named_pairs))
        if not replot_only:
            logger.warning(
                "selected_pairs is set — computing only the %d named pair(s); "
                "aggregate path_recapitulation_*.json will be skipped "
                "(cherry-picked subset cannot yield unbiased stats).",
                len(selected_pair_indices),
            )
    elif replot_only:
        # No selected_pairs under replot_only → load whatever is in the cache.
        # selected_pair_indices left empty so the missing-cache log is a no-op.
        selected_pair_indices = []
    elif max_pairs_cfg is None or len(all_pull_pairs) <= int(max_pairs_cfg):
        selected_pair_indices = sorted(all_pull_pairs)
    else:
        from causalab.analyses.path_steering.main import _sample_pairs

        selected_pair_indices = sorted(
            _sample_pairs(all_pull_pairs, int(max_pairs_cfg), cfg.seed)
        )

    logger.info("Pullback: %d pair(s) selected", len(selected_pair_indices))

    # ── Cache file paths ──────────────────────────────────────────────────
    # Caches use the safetensors+meta convention; pass a (dir, stem) pair to
    # ``save_nested`` / ``load_nested``. The on-disk artifacts are
    # ``geodesic_paths.{safetensors,meta.json}`` and
    # ``optimization_results.{safetensors,meta.json}``.
    geo_stem = "geodesic_paths"
    opt_cache_stem = "optimization_results"
    geo_safetensors = out_dir / f"{geo_stem}.safetensors"
    opt_cache_safetensors = out_dir / f"{opt_cache_stem}.safetensors"

    # ── Load model (always — replot_only still needs it for activation-3D viz) ──
    pipeline = load_pipeline(
        model_name=cfg.model.name,
        task=task,
        max_new_tokens=cfg.task.max_new_tokens,
        device=cfg.model.device,
        dtype=cfg.model.get("dtype"),
        eager_attn=cfg.model.get("eager_attn"),
    )
    model = pipeline.model
    tokenizer = pipeline.tokenizer

    # var_indices is only consumed by collect_grid_distributions in the compute branch.
    var_indices: Any = None
    if not replot_only:
        concept_token_ids = tokenize_variable_values(
            tokenizer,
            concept_values,
            task.result_token_pattern,
        )
        if isinstance(concept_token_ids, torch.Tensor) and concept_token_ids.dim() == 1:
            var_indices = concept_token_ids
        elif isinstance(concept_token_ids, list) and all(
            isinstance(ids, list) and len(ids) >= 1 for ids in concept_token_ids
        ):
            var_indices = concept_token_ids
        else:
            raise NotImplementedError(
                "Multi-token output tasks not yet supported in pullback"
            )

    steered_variable = task.intervention_variable
    causal_model = task.causal_model

    # Generate balanced samples (only when computing).
    filtered_samples: list = []
    pair_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    if not replot_only:
        _random.seed(cfg.seed)
        for ci, cj in selected_pair_indices:
            base_val = concept_values[ci]
            cf_val = concept_values[cj]
            for _ in range(analysis.n_prompts):
                base_trace = causal_model.sample_input(
                    filter_func=lambda t, v=base_val: t[steered_variable] == v
                )
                cf_trace = causal_model.sample_input(
                    filter_func=lambda t, v=cf_val: t[steered_variable] == v
                )
                n = len(filtered_samples)
                filtered_samples.append(
                    {
                        "input": base_trace,
                        "counterfactual_inputs": [cf_trace],
                    }
                )
                pair_groups[(ci, cj)].append(n)
        logger.info(
            "Generated %d balanced samples across %d pairs",
            len(filtered_samples),
            len(pair_groups),
        )

    # ── Class-level natural distributions (always — needed for Hellinger viz) ──
    _bm_train_ds, _ = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        balanced=cfg.task.get("balanced", False),
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
    )
    _bm_class_assignments = torch.tensor(
        [task.intervention_value_index(ex) for ex in _bm_train_ds],
        dtype=torch.long,
    )
    natural_dists, all_class_dists_WW1 = load_natural_distributions(
        root,
        W,
        class_assignments=_bm_class_assignments,
    )
    P_start_centroid_WW1 = all_class_dists_WW1
    P_end_centroid_WW1 = all_class_dists_WW1

    # Auto-discover output manifold checkpoint.
    output_manifold_ckpt = getattr(belief_path_cfg, "output_manifold_ckpt", None)
    if output_manifold_ckpt is None:
        bm_dir = os.path.join(root, "output_manifold")
        if os.path.isdir(bm_dir):
            bm_subs = [
                d for d in os.listdir(bm_dir) if os.path.isdir(os.path.join(bm_dir, d))
            ]
            if bm_subs:
                parts = [bm_dir, bm_subs[0]]
                if tv:
                    parts.append(tv)
                parts += ["manifold_spline", "ckpt_final.safetensors"]
                output_manifold_ckpt = os.path.join(*parts)

    # Load output manifold object — pullback requires it unconditionally.
    if output_manifold_ckpt is None or not os.path.exists(output_manifold_ckpt):
        raise ValueError(
            "Pullback requires an output_manifold checkpoint. "
            "Run the output_manifold analysis first, or set "
            "belief_path.output_manifold_ckpt explicitly."
        )
    from causalab.methods.spline.belief_fit import load_output_manifold

    bm_sub = os.path.relpath(
        os.path.dirname(os.path.dirname(output_manifold_ckpt)),
        os.path.join(root, "output_manifold"),
    )
    belief_manifold_obj, _ = load_output_manifold(root, bm_sub)
    logger.info("Pullback: loaded output manifold")

    belief_n_steps = belief_path_cfg.n_steps

    # ── Geodesic step ─────────────────────────────────────────────────────
    geodesic_paths: dict
    t_values: torch.Tensor
    if replot_only:
        if geo_safetensors.exists():
            geodesic_paths, _ = load_nested(str(out_dir), geo_stem)
        else:
            geodesic_paths = {}
        if not geodesic_paths:
            logger.error("replot_only: no geodesic cache found at %s", geo_safetensors)
        missing_geo = [
            (concept_names[ci], concept_names[cj])
            for (ci, cj) in selected_pair_indices
            if (ci, cj) not in geodesic_paths
        ]
        _log_cache_misses("geodesic (start, end)", missing_geo)
        t_values = torch.linspace(0, 1, belief_n_steps)
    else:
        geodesic_paths, t_values = compute_manifold_trace_paths(
            selected_pair_indices=selected_pair_indices,
            belief_manifold=belief_manifold_obj,
            n_steps=belief_n_steps,
        )
        # Merge with prior cache (preserves untouched pair entries) and save.
        if geo_safetensors.exists():
            cached_geo, _ = load_nested(str(out_dir), geo_stem)
            cached_geo.update(geodesic_paths)
            geodesic_paths = cached_geo
        save_nested(geodesic_paths, str(out_dir), geo_stem)

    # ── Visualize belief targets (works in both modes) ────────────────────
    vis_flags = analysis.visualization
    task_colormap = resolve_task_colormap(cfg.task, "rainbow")

    if vis_flags.belief_trajectories:
        from causalab.analyses.pullback.visualization import plot_belief_paths

        belief_results = {
            (ci, cj): {
                "base_class": ci,
                "cf_class": cj,
                "p_target_AW1": geodesic_paths[(ci, cj)],
            }
            for (ci, cj) in selected_pair_indices
            if (ci, cj) in geodesic_paths
        }
        if belief_results:
            plot_belief_paths(
                belief_results,
                concept_names,
                output_dir=str(vis_dir / "belief_paths"),
                colormap=task_colormap,
                figure_format=figure_fmt,
            )

    # ── Embedding optimization step ───────────────────────────────────────
    k_opt_list = _normalize_k_opt(analysis.embedding_optim.get("k_opt"))
    logger.info(
        "Pullback: k_opt sweep over %d value(s): %s",
        len(k_opt_list),
        [label for label, _ in k_opt_list],
    )
    results_by_kopt: dict[str, dict[tuple[int, int], dict]] = {}
    summary_by_kopt: dict[str, dict] = {}

    ss_meta = load_subspace_metadata(root, ss_sub, target_variable=tv)
    layer = ss_meta.get("layer")
    k_features = ss_meta.get("k_features")
    if layer is None or k_features is None:
        logger.warning(
            "Skipping pullback embedding step: missing layer or k_features in metadata for %s",
            ss_sub,
        )
    else:
        subspace_out = os.path.join(root, "subspace", ss_sub)
        if tv:
            subspace_out = os.path.join(subspace_out, tv)
        manifold_out = os.path.join(root, "activation_manifold", ss_sub, m_sub)
        if tv:
            manifold_out = os.path.join(manifold_out, tv)

        from causalab.analyses.activation_manifold.loading import load_featurizer
        from causalab.io.counterfactuals import load_counterfactual_examples

        training_features = load_safetensors(
            os.path.join(subspace_out, "features", "training_features.safetensors")
        )["features"]
        train_dataset_3d = load_counterfactual_examples(
            os.path.join(subspace_out, "train_dataset.json"),
            task.causal_model,
        )

        device = device_for_layer(pipeline, layer)
        _tp_name = ss_meta.get("token_position")
        targets, _tp_list = build_targets_for_grid(
            pipeline,
            task,
            [layer],
            [_tp_name] if _tp_name else None,
        )
        steered_tp = _tp_list[0]
        interchange_target = next(iter(targets.values()))

        featurizer = load_featurizer(
            manifold_out, interchange_target, layer, steered_tp.id
        )
        manifold_obj = featurizer.stages[-1].featurizer.manifold.to(device)
        std_stage = featurizer.stages[-2].featurizer
        manifold_mean = std_stage._mean
        manifold_std = std_stage._std

        if replot_only:
            if opt_cache_safetensors.exists():
                raw_cache, _ = load_nested(str(out_dir), opt_cache_stem)
            else:
                raw_cache = {}
            results_by_kopt = _coerce_results_by_kopt(raw_cache)
            if not results_by_kopt:
                logger.error(
                    "replot_only: no embedding-optim cache found at %s",
                    opt_cache_safetensors,
                )
            for label, _ in k_opt_list:
                per_label = results_by_kopt.get(label, {})
                missing_opt = [
                    (concept_names[ci], concept_names[cj])
                    for (ci, cj) in selected_pair_indices
                    if (ci, cj) not in per_label
                ]
                _log_cache_misses(
                    f"embedding-optim[{label}] (start, end)",
                    missing_opt,
                )
                # Recompute metrics on cached paths so iterating on metric
                # definitions doesn't require re-running LBFGS.
                summary = _attach_recapitulation_metrics_and_aggregate(
                    per_label,
                    concept_names,
                    label,
                    compute_aggregate=(analysis.selected_pairs is None),
                )
                if summary is not None:
                    summary_by_kopt[label] = summary
        else:
            from causalab.methods.pullback.optimization import (
                run_pair_optimization,
            )
            from causalab.analyses.activation_manifold.utils import (
                _compute_intrinsic_ranges,
            )
            from causalab.analyses.path_steering.path_mode import (
                _build_geodesic_path,
                _build_linear_path_kd,
            )
            from causalab.methods.steer.collect import collect_grid_distributions
            from causalab.analyses.subspace import load_subspace_onto_target

            def _append_other_bin(distributions: torch.Tensor) -> torch.Tensor:
                """Append (1 - sum) bin so concept probs form a proper simplex."""
                other = (1.0 - distributions.sum(dim=-1, keepdim=True)).clamp(min=0.0)
                return torch.cat([distributions, other], dim=-1)

            manifold_ranges = _compute_intrinsic_ranges(
                training_features,
                manifold_obj,
                manifold_mean,
                manifold_std,
            )

            n_classes = len(concept_names)
            intrinsic_centroids = {
                c: manifold_obj.control_points[c] for c in range(n_classes)
            }

            emb_n_steps = analysis.embedding_optim.n_steps
            precomputed_comparisons: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

            # --- Pass 1: full featurizer → geometric comparison distributions ---
            for ci, cj in selected_pair_indices:
                pair_samples = [filtered_samples[n] for n in pair_groups[(ci, cj)]]
                pair_comps: dict[str, torch.Tensor] = {}

                geo_grid = _build_geodesic_path(
                    intrinsic_centroids[ci],
                    intrinsic_centroids[cj],
                    emb_n_steps,
                    manifold_obj,
                )
                with torch.no_grad():
                    geo_probs = collect_grid_distributions(
                        pipeline=pipeline,
                        grid_points=geo_grid,
                        interchange_target=interchange_target,
                        filtered_samples=pair_samples,
                        var_indices=var_indices,
                        n_base_samples=len(pair_samples),
                        average=False,
                        full_vocab_softmax=True,
                    )
                pair_comps["geo_probs_raw"] = geo_probs
                pair_comps["geo_probs_AW1"] = _append_other_bin(geo_probs.mean(dim=1))
                with torch.no_grad():
                    geo_decoded = manifold_obj.decode(geo_grid.to(device))
                pair_comps["v_geometric_k"] = (
                    geo_decoded * (manifold_std.to(device) + 1e-6)
                    + manifold_mean.to(device)
                ).cpu()

                precomputed_comparisons[(ci, cj)] = pair_comps

            logger.info(
                "Collected geometric distributions for %d pairs (full featurizer)",
                len(precomputed_comparisons),
            )

            # --- Pass 2: PCA featurizer → linear comparison + optimization ---
            load_subspace_onto_target(
                interchange_target,
                subspace_out,
                ss_meta.get("method", "pca"),
                k_features,
            )

            centroid_k = {}
            for c in range(n_classes):
                mask = torch.tensor(
                    [
                        task.intervention_value_index(ex) == c
                        for ex in train_dataset_3d[: training_features.shape[0]]
                    ]
                )
                if mask.any():
                    centroid_k[c] = training_features[mask].mean(dim=0)

            k = training_features.shape[1]

            for ci, cj in selected_pair_indices:
                pair_samples = [filtered_samples[n] for n in pair_groups[(ci, cj)]]
                lin_grid = _build_linear_path_kd(
                    centroid_k[ci],
                    centroid_k[cj],
                    emb_n_steps,
                )
                with torch.no_grad():
                    lin_probs = collect_grid_distributions(
                        pipeline=pipeline,
                        grid_points=lin_grid,
                        interchange_target=interchange_target,
                        filtered_samples=pair_samples,
                        var_indices=var_indices,
                        n_base_samples=len(pair_samples),
                        average=False,
                        full_vocab_softmax=True,
                    )
                precomputed_comparisons[(ci, cj)]["lin_probs_raw"] = lin_probs
                precomputed_comparisons[(ci, cj)]["lin_probs_AW1"] = _append_other_bin(
                    lin_probs.mean(dim=1)
                )
                precomputed_comparisons[(ci, cj)]["v_linear_k"] = lin_grid.cpu()

            logger.info(
                "Collected linear distributions for %d pairs (PCA featurizer)",
                len(precomputed_comparisons),
            )

            # FeatureInterpolateIntervention preserves base_err — needed for differentiable optim.
            from causalab.neural.activations.intervenable_model import (
                prepare_intervenable_model,
                delete_intervenable_model,
            )

            intervenable_model = prepare_intervenable_model(
                pipeline, interchange_target, intervention_type="interpolation"
            )
            intervenable_model.disable_model_gradients()
            intervenable_model.eval()

            if emb_n_steps != belief_n_steps:
                emb_t_values = torch.linspace(0, 1, emb_n_steps)
            else:
                emb_t_values = t_values

            def _resample_path(
                path_AW: torch.Tensor, src_t: torch.Tensor, dst_t: torch.Tensor
            ) -> torch.Tensor:
                """Linearly interpolate a (A, W) path from src_t grid to dst_t grid."""
                idx = torch.searchsorted(src_t, dst_t).clamp(1, len(src_t) - 1)
                frac = (
                    (dst_t - src_t[idx - 1]) / (src_t[idx] - src_t[idx - 1] + 1e-12)
                ).unsqueeze(1)
                return path_AW[idx - 1] + frac * (path_AW[idx] - path_AW[idx - 1])

            if emb_n_steps != belief_n_steps:
                resampled_paths = {
                    pair: _resample_path(p, t_values, emb_t_values)
                    for pair, p in geodesic_paths.items()
                }
            else:
                resampled_paths = dict(geodesic_paths)

            # Per-k_opt optimization loop. Geo/lin/geodesic precomputations
            # are k_opt-independent and reused across iterations; only the
            # optimized trajectory differs per k_opt slice.
            for label, k_opt_value in k_opt_list:
                logger.info(
                    "Pullback: optimizing for k_opt=%s (label %s)",
                    "full" if k_opt_value is None else str(k_opt_value),
                    label,
                )
                _emb_dict = OmegaConf.to_container(
                    analysis.embedding_optim, resolve=True
                )
                _emb_dict["k_opt"] = k_opt_value
                optimizer_cfg_for_label = OmegaConf.create(_emb_dict)

                results = run_pair_optimization(
                    selected_pair_indices=selected_pair_indices,
                    pair_groups=pair_groups,
                    filtered_samples=filtered_samples,
                    pipeline=pipeline,
                    intervenable_model=intervenable_model,
                    interchange_target=interchange_target,
                    k=k,
                    var_indices=var_indices,
                    geodesic_paths=resampled_paths,
                    centroid_k=centroid_k,
                    P_start_centroid_WW1=P_start_centroid_WW1,
                    P_end_centroid_WW1=P_end_centroid_WW1,
                    t_values=emb_t_values,
                    optimizer_cfg=optimizer_cfg_for_label,
                    device=device,
                    concept_names=concept_names,
                    sample_embeddings_k=training_features,
                    base_metric="hellinger",
                    skip_optimization=getattr(analysis, "skip_optimization", False),
                    precomputed_comparisons=precomputed_comparisons,
                )

                summary = _attach_recapitulation_metrics_and_aggregate(
                    results,
                    concept_names,
                    label,
                    compute_aggregate=(analysis.selected_pairs is None),
                )
                if summary is not None:
                    summary_by_kopt[label] = summary
                results_by_kopt[label] = results

            delete_intervenable_model(intervenable_model)

        # Persist metrics + cache. Runs in both branches so replot_only also
        # rewrites path_recapitulation_*.json from cached paths.
        if summary_by_kopt:
            metrics_dir = out_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            for label, summary in summary_by_kopt.items():
                out_path = metrics_dir / f"path_recapitulation_{label}.json"
                with open(out_path, "w") as f:
                    json.dump(summary, f, indent=2)

        if results_by_kopt:
            # Merge with any existing on-disk labels so a sweep at a new k_opt
            # doesn't clobber prior runs at other k_opt values.
            if opt_cache_safetensors.exists():
                prior, _ = load_nested(str(out_dir), opt_cache_stem)
                existing = _coerce_results_by_kopt(prior)
            else:
                existing = {}
            existing.update(results_by_kopt)
            save_nested(existing, str(out_dir), opt_cache_stem)
            results_by_kopt = existing

        any_results = any(per for per in results_by_kopt.values())

        # ── Belief-vs-actual plots from optimization results ──
        if vis_flags.belief_trajectories and any_results:
            from causalab.analyses.pullback.visualization import plot_belief_paths

            plot_belief_paths(
                results_by_kopt,
                concept_names,
                output_dir=str(vis_dir / "belief_paths"),
                colormap=task_colormap,
                figure_format=figure_fmt,
            )

        # ── Activation-space 3D plots ──
        if vis_flags.embedding_3d and any_results:
            from causalab.analyses.pullback.visualization import plot_activation_paths

            iv = task.intervention_variable
            param_dict_viz = {}
            if iv and train_dataset_3d:
                param_dict_viz = {
                    iv: np.array(
                        [
                            float(task.intervention_value_index(ex))
                            for ex in train_dataset_3d
                        ]
                    ),
                }

            plot_activation_paths(
                results_by_kopt,
                training_features,
                concept_names,
                output_dir=str(vis_dir / "activation_paths"),
                manifold_obj=manifold_obj,
                manifold_mean=manifold_mean,
                manifold_std=manifold_std,
                colormap=task_colormap,
                intervention_variable=task.intervention_variable,
                param_dict=param_dict_viz,
                variable_values=[str(v) for v in task.intervention_values],
            )

    # ── Hellinger PCA overlay ─────────────────────────────────────────────
    # `belief_paths_for_plot` is the geodesic cache filtered to the requested
    # pair set; the viz function renders one red trace per pair. Gated on
    # `plot_belief_target` so it can be turned off without removing the cache.
    plot_belief_target = bool(getattr(vis_flags, "plot_belief_target", True))
    belief_paths_for_plot = None
    if plot_belief_target:
        belief_paths_for_plot = {
            pair: tensor
            for pair, tensor in geodesic_paths.items()
            if pair in set(selected_pair_indices)
        }
        if not belief_paths_for_plot:
            belief_paths_for_plot = None

    any_results = any(per for per in results_by_kopt.values())
    if vis_flags.hellinger_pca_3d and (any_results or belief_paths_for_plot):
        from causalab.analyses.pullback.visualization import (
            plot_belief_paths_hellinger_pca,
        )

        plot_belief_paths_hellinger_pca(
            results_by_kopt,
            natural_dists,
            concept_names,
            output_dir=str(vis_dir / "belief_space"),
            colormap=task_colormap,
            intervention_variable=task.intervention_variable,
            belief_manifold=belief_manifold_obj,
            train_dataset=_bm_train_ds,
            belief_paths=belief_paths_for_plot,
        )
        logger.info("Saved Hellinger PCA overlay plots to %s", vis_dir)

    # ── Save metadata ─────────────────────────────────────────────────────
    metadata = {
        "analysis": "pullback",
        "subspace": analysis.subspace,
        "activation_manifold": analysis.activation_manifold,
        "replot_only": replot_only,
        "n_prompts": analysis.n_prompts,
        "n_pairs": len(selected_pair_indices),
        "belief_path": OmegaConf.to_container(analysis.belief_path, resolve=True),
        "embedding_optim": OmegaConf.to_container(
            analysis.embedding_optim, resolve=True
        ),
        "model": cfg.model.id,
        "task": cfg.task.name,
        "task_config": _task_config_for_metadata(
            OmegaConf.to_container(cfg.task, resolve=True)
        ),
        "seed": cfg.seed,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Pullback analysis complete. Output in %s", out_dir)
    return results_by_kopt
