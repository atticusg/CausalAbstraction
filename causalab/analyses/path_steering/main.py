"""Path steering analysis: compute distortion metrics for subspace/manifold combinations."""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any

import numpy as np
import torch
from torch import Tensor
from omegaconf import DictConfig, OmegaConf

from causalab.runner.helpers import (
    _task_config_for_metadata,
    resolve_task,
    generate_datasets,
    build_targets_for_grid,
)
from causalab.io.pipelines import (
    load_pipeline,
    load_lite_pipeline,
    find_subspace_dirs,
    find_activation_manifold_dirs,
    load_subspace_metadata,
    load_activation_manifold_metadata,
)
from causalab.io.artifacts import (
    load_tensor_results,
    load_tensors_with_meta,
)
from causalab.io.sklearn_pca import load_pca
from causalab.io.plots.figure_format import (
    resolve_figure_format_from_analysis,
)
from causalab.tasks.loader import load_task_counterfactuals

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "path_steering"


def _sample_pairs(
    candidates: list[tuple[int, int]],
    n_to_sample: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Uniform random sampling without replacement, deterministic via seed."""
    if n_to_sample <= 0 or not candidates:
        return []
    import random as _random

    rng = _random.Random(seed)
    return rng.sample(candidates, min(n_to_sample, len(candidates)))


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


def _compute_and_save_isometry(
    pm,
    featurizer,
    belief_manifold_eval,
    isometry_cfg,
    out_dir: str,
    criteria_results: dict,
) -> None:
    """Compute isometry score for one path mode and persist artifacts.

    Side effects: writes to ``criteria_results`` and to
    ``{out_dir}/criteria/isometry/{pm.label}/``. Inputs are read from cached
    manifolds; no model forward passes, path interventions, or class centroids
    are needed.
    """
    from causalab.methods.scores.isometry import (
        compute_isometry_from_manifolds,
        _save_isometry_artifacts,
    )

    n_arc_steps = isometry_cfg.get("n_arc_steps", 150)
    n_interior_per_pair = int(isometry_cfg.get("n_interior_per_pair", 0))
    manifold_iso = featurizer.stages[-1].featurizer.manifold
    std_stage = featurizer.stages[-2].featurizer
    iso_mean = std_stage._mean
    iso_std = std_stage._std

    if belief_manifold_eval is None:
        logger.warning(
            "Isometry [%s] skipped: no output manifold loaded "
            "(run output_manifold analysis first).",
            pm.label,
        )
        return

    # `linear_subspace` collapses to `linear` for the isometry metric: with
    # manifold-derived vertices the lift PCA→residual is an isometry, so the
    # two produce identical numbers. The pm.label is still used for artifact
    # paths so existing on-disk layouts are preserved.
    iso_path_mode = "geometric" if pm.label == "geometric" else "linear"

    iso_metrics, D_X, D_Y, iso_grid, iso_grid_belief = compute_isometry_from_manifolds(
        activation_manifold=manifold_iso,
        activation_mean=iso_mean,
        activation_std=iso_std,
        belief_manifold=belief_manifold_eval,
        n_arc_steps=n_arc_steps,
        path_mode=iso_path_mode,
        n_interior_per_pair=n_interior_per_pair,
    )
    result_key = f"isometry/{pm.label}"
    criteria_results[result_key] = iso_metrics
    logger.info(
        "  %s: r=%.4f over %d pairs",
        result_key,
        iso_metrics["pearson_r"],
        iso_metrics["n_pairs"],
    )
    iso_out = os.path.join(out_dir, "criteria", "isometry", pm.label)
    _save_isometry_artifacts(
        iso_metrics,
        D_X,
        D_Y,
        iso_grid,
        iso_out,
        grid_points_belief=iso_grid_belief,
        metadata={
            "n_arc_steps": n_arc_steps,
            "n_centroids": iso_metrics.get(
                "n_centroids",
                int(iso_grid.shape[0]),
            ),
            "n_interior_per_pair": n_interior_per_pair,
            "path_mode": pm.label,
            "periodic_dims": list(getattr(manifold_iso, "periodic_dims", []) or []),
            "periods": list(
                manifold_iso.periods if hasattr(manifold_iso, "periods") else []
            ),
            "belief_periodic_dims": list(
                getattr(belief_manifold_eval, "periodic_dims", []) or []
            ),
            "belief_periods": list(
                belief_manifold_eval.periods
                if hasattr(belief_manifold_eval, "periods")
                else []
            ),
        },
    )


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the path steering analysis: compute distortion metrics."""
    from causalab.analyses.path_steering.registry import CRITERIA_REGISTRY
    from causalab.analyses.path_steering.path_mode import resolve_path_modes
    from causalab.neural.featurizer import ComposedFeaturizer

    analysis = cfg[ANALYSIS_NAME]
    figure_fmt = resolve_figure_format_from_analysis(analysis)
    root = cfg.experiment_root
    metrics = list(analysis.eval_criteria)
    visualizations = list(analysis.get("visualizations", []))
    replot_only = bool(analysis.get("replot_only", False))
    recompute_isometry = bool(analysis.get("recompute_isometry", False))
    if replot_only and recompute_isometry:
        raise ValueError("replot_only and recompute_isometry are mutually exclusive.")
    if replot_only and analysis.selected_pairs is None:
        raise ValueError("replot_only=True requires `selected_pairs` to be set.")
    if replot_only:
        logger.info(
            "replot_only=True — skipping model load, criteria, and extras; "
            "rendering plots from cached pair_distributions for %d named pair(s) only.",
            len(analysis.selected_pairs),
        )
    if recompute_isometry:
        logger.info(
            "recompute_isometry=True — skipping model load, path interventions, "
            "and other criteria; recomputing only isometry from cached manifolds."
        )
    # Discover subspace x manifold combinations
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

    # Load task + model (once)
    task, task_cfg_raw = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )
    if replot_only or recompute_isometry:
        pipeline = load_lite_pipeline(
            model_name=cfg.model.name,
            max_new_tokens=cfg.task.max_new_tokens,
        )
    else:
        pipeline = load_pipeline(
            model_name=cfg.model.name,
            task=task,
            max_new_tokens=cfg.task.max_new_tokens,
            device=cfg.model.device,
            dtype=cfg.model.get("dtype"),
            eager_attn=cfg.model.get("eager_attn"),
        )

    # Build graph-edge metadata for visualization on graph_walk tasks
    if cfg.task.name == "graph_walk":
        from causalab.tasks.graph_walk.graphs import build_graph

        _graph = build_graph(
            cfg.task.graph_type,
            cfg.task.graph_size,
            cfg.task.get("graph_size_2"),
        )
        _graph_edges = [
            (i, j)
            for i, neighbors in _graph.adjacency.items()
            for j in neighbors
            if j > i
        ]
        _graph_edge_node_coords = {
            i: {task.intervention_variable: float(i)} for i in range(_graph.n_nodes)
        }

    if task.intervention_variable is None:
        logger.warning("No intervention_variable on task; skipping evaluation")
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {}

    # task_cfg is the Hydra task config — metrics read from it directly
    task_cfg = cfg.task

    # Per-criterion configs resolved via Hydra defaults
    criteria_cfgs = {name: analysis.get(name, OmegaConf.create({})) for name in metrics}

    results = {}

    for ss_sub, m_sub in combos:
        logger.info("=== Evaluating %s / %s ===", ss_sub, m_sub)

        tv = cfg.task.get("target_variable")
        ss_meta = load_subspace_metadata(root, ss_sub, target_variable=tv)
        layer = ss_meta.get("layer")

        is_linear_baseline = m_sub == "linear"

        if not is_linear_baseline:
            m_meta = load_activation_manifold_metadata(
                root, ss_sub, m_sub, target_variable=tv
            )
            layer = layer or m_meta.get("layer")

        if layer is None:
            logger.warning("Skipping %s/%s: no layer in metadata", ss_sub, m_sub)
            continue

        # Build interchange target — prefer subspace token_position, fall back to manifold
        _tp_name = ss_meta.get("token_position")
        if not _tp_name and not is_linear_baseline:
            _tp_name = m_meta.get("token_position")
        targets, _tp_list = build_targets_for_grid(
            pipeline,
            task,
            [layer],
            [_tp_name] if _tp_name else None,
        )
        token_pos = _tp_list[0]
        interchange_target = next(iter(targets.values()))

        if is_linear_baseline:
            # Linear baseline: load only the subspace featurizer (no manifold)
            from causalab.analyses.subspace import load_subspace_onto_target

            ss_method = ss_meta.get("method", "pca")
            k_features = ss_meta.get("k_features")
            subspace_out_dir = os.path.join(root, "subspace", ss_sub)
            if tv:
                subspace_out_dir = os.path.join(subspace_out_dir, tv)
            load_subspace_onto_target(
                interchange_target, subspace_out_dir, ss_method, k_features
            )
            featurizer = interchange_target.flatten()[0].featurizer
            logger.info(
                "Linear baseline: loaded subspace featurizer only (no manifold)"
            )
        else:
            # Load featurizer from manifold output
            from causalab.analyses.activation_manifold.loading import load_featurizer

            manifold_out = os.path.join(root, "activation_manifold", ss_sub, m_sub)
            if tv:
                manifold_out = os.path.join(manifold_out, tv)
            featurizer = load_featurizer(
                manifold_out, interchange_target, layer, token_pos.id
            )

        encode_mode = getattr(analysis, "encode_mode", "nearest_centroid")
        if encode_mode != "nearest_centroid":
            for s in getattr(featurizer, "stages", []):
                manifold = (
                    getattr(s.featurizer, "manifold", None)
                    if hasattr(s, "featurizer")
                    else None
                )
                if manifold is not None and hasattr(manifold, "encode_mode"):
                    manifold.encode_mode = encode_mode
                    logger.info("Manifold encode_mode: %s", encode_mode)
                    break

        # Generate eval samples
        cf_mod = load_task_counterfactuals(task.name)
        filtered_samples = cf_mod.generate_dataset(
            task.causal_model,
            analysis.n_eval_samples,
            cfg.seed + 100,
        )

        tv = cfg.task.get("target_variable")
        out_dir = os.path.join(root, "path_steering", ss_sub, m_sub)
        if tv:
            out_dir = os.path.join(out_dir, tv)
        os.makedirs(out_dir, exist_ok=True)

        # Resolve path modes once per (subspace, manifold) combo
        path_modes_cfg = list(
            OmegaConf.to_container(analysis.get("path_modes"), resolve=True)
        )
        path_modes = resolve_path_modes(
            path_modes_cfg=path_modes_cfg,
            composed_featurizer=featurizer
            if isinstance(featurizer, ComposedFeaturizer)
            else None,
        )

        # --- Collect centroid-pair distributions per path mode ---
        from causalab.methods.metric import tokenize_variable_values
        import itertools

        values = task.intervention_values
        var_indices = tokenize_variable_values(
            pipeline.tokenizer,
            values,
            task.result_token_pattern,
        )
        num_steps = analysis.num_steps_along_path
        oversteer_frac = float(
            OmegaConf.select(analysis, "oversteer.frac", default=0.0) or 0.0
        )
        oversteer_steps = (
            int(OmegaConf.select(analysis, "oversteer.num_steps", default=0) or 0)
            if oversteer_frac > 0
            else 0
        )
        n_total_steps = num_steps + oversteer_steps
        n_prompts = min(analysis.n_prompts, len(filtered_samples))
        batch_size = analysis.batch_size
        distance_function = task_cfg.get("distance_function", "hellinger")

        # Generate pair indices (same for all path modes)
        n_values = len(values)
        value_strs = [str(v) for v in values]

        def _match_value(query: str) -> int:
            """Find index of a value by exact or fuzzy match."""
            q = str(query)
            if q in value_strs:
                return value_strs.index(q)
            # Try parsing as tuple and matching numerically
            q_stripped = q.strip("() ")
            try:
                q_nums = tuple(float(x) for x in q_stripped.split(","))
                for i, v in enumerate(values):
                    if isinstance(v, (tuple, list)) and len(v) == len(q_nums):
                        if all(abs(float(a) - b) < 1e-6 for a, b in zip(v, q_nums)):
                            return i
            except (ValueError, TypeError):
                pass
            raise ValueError(f"No match for {q!r} in values")

        # Pair selection.
        # - null (default): unbiased random sample governed by `max_pairs`. All
        #   aggregate criteria summaries are written.
        # - list of [start, end] string pairs: compute ONLY these pairs. Aggregate
        #   criteria writes (results_summary.csv, paired_ttest_*.json,
        #   criteria/*/metrics.json, isometry compute) are skipped because a
        #   cherry-picked subset cannot yield unbiased stats. A warning is logged
        #   at run start. Mirrors `causalab/analyses/pullback/main.py:177-208`.
        all_pairs = list(itertools.combinations(range(n_values), 2))
        max_pairs = analysis.get("max_pairs")

        if analysis.selected_pairs is not None:
            named_pairs = [
                (_match_value(a), _match_value(b)) for a, b in analysis.selected_pairs
            ]
            pairs = sorted(set(named_pairs))
            if not replot_only:
                logger.warning(
                    "selected_pairs is set — computing only the %d named pair(s); "
                    "aggregate criteria summaries (results_summary.csv, "
                    "paired_ttest_*.json, criteria/*/metrics.json, isometry) will "
                    "be skipped (cherry-picked subset cannot yield unbiased stats).",
                    len(pairs),
                )
        elif replot_only:
            raise ValueError("replot_only=True requires `selected_pairs` to be set.")
        elif max_pairs is None or len(all_pairs) <= int(max_pairs):
            pairs = sorted(all_pairs)
        else:
            pairs = sorted(_sample_pairs(all_pairs, int(max_pairs), cfg.seed))

        logger.info(
            "Path steering: %d pair(s) selected (%d total possible)",
            len(pairs),
            len(all_pairs),
        )

        # Select n_prompts base samples
        eval_samples = filtered_samples[:n_prompts]

        criteria_results = {}
        viz_results = {}

        # Load saved features for centroid computation and random baselines
        from safetensors.torch import load_file as _load_file

        subspace_out_dir = os.path.join(root, "subspace", ss_sub)
        if tv:
            subspace_out_dir = os.path.join(subspace_out_dir, tv)
        ss_meta = load_subspace_metadata(root, ss_sub, target_variable=tv)
        # For grid-mode subspaces, features are stored per cell under layer_x_pos/
        _ss_mode = ss_meta.get("mode", "single")
        _ss_method = ss_meta.get("method", "pca")
        if _ss_mode == "grid" and _ss_method == "pca" and _tp_name:
            _feat_dir = os.path.join(
                subspace_out_dir, "layer_x_pos", f"L{layer}_{_tp_name}", "features"
            )
        else:
            _feat_dir = os.path.join(subspace_out_dir, "features")
        pca_path = os.path.join(_feat_dir, "training_features.safetensors")
        raw_path = os.path.join(_feat_dir, "raw_features.safetensors")
        pca_features = _load_file(pca_path)["features"]  # (N, k)
        _use_linear = any(getattr(pm, "label", None) == "linear" for pm in path_modes)
        _raw_features_missing = not os.path.exists(raw_path)
        if _use_linear and not _raw_features_missing:
            raw_features = _load_file(raw_path)["features"]  # (N, D)
        elif _use_linear and _raw_features_missing:
            logger.warning(
                "raw_features.safetensors not found at %s; "
                "skipping linear path mode (re-run subspace analysis to regenerate)",
                raw_path,
            )
            # Remove linear path mode to avoid shape mismatch
            path_modes = [
                pm for pm in path_modes if getattr(pm, "label", None) != "linear"
            ]
            raw_features = pca_features  # placeholder, unused
        else:
            raw_features = pca_features  # placeholder, never used for non-linear modes

        # Separate path-based metrics (coherence, distance_from_behavior_manifold).
        # Isometry is handled separately above, not through the registry.
        path_based_metrics = []
        if not replot_only:
            for name in metrics:
                if name == "isometry":
                    continue
                if name not in CRITERIA_REGISTRY:
                    logger.warning("Unknown criterion: %s", name)
                    continue
                mod = CRITERIA_REGISTRY[name]
                if getattr(mod, "SUPPORTS_PATH_MODES", False):
                    path_based_metrics.append((name, mod))

        # Load output manifold if available (for distance_from_behavior_manifold criterion + isometry)
        belief_manifold_eval = None
        belief_manifold_dir = os.path.join(root, "output_manifold")
        if os.path.isdir(belief_manifold_dir):
            bm_subs = [
                d
                for d in os.listdir(belief_manifold_dir)
                if os.path.isdir(os.path.join(belief_manifold_dir, d))
            ]
            if bm_subs:
                from causalab.methods.spline.belief_fit import (
                    load_output_manifold,
                )

                _bm_sub = os.path.join(bm_subs[0], tv) if tv else bm_subs[0]
                belief_manifold_eval, _ = load_output_manifold(root, _bm_sub)
                logger.info("Loaded output manifold '%s' for evaluation", _bm_sub)

        # Compute centroids from saved features (no forward passes).
        # Load saved dataset from subspace (ensures row alignment with features)
        saved_ds_path = os.path.join(subspace_out_dir, "train_dataset.json")
        if os.path.exists(saved_ds_path):
            from causalab.io.counterfactuals import load_counterfactual_examples

            _train_ds = load_counterfactual_examples(saved_ds_path, task.causal_model)
        else:
            _train_ds, _ = generate_datasets(
                task,
                n_train=ss_meta.get("n_train", task_cfg.get("n_train", 1000)),
                n_test=0,
                seed=ss_meta.get("seed", cfg.seed),
                enumerate_all=cfg.task.enumerate_all,
                resample_variable=cfg.task.get("resample_variable", "all"),
            )

        # Compute centroids in PCA and raw space
        from causalab.neural.pipeline import resolve_device

        device = torch.device(resolve_device(cfg.model.device))
        n_values = len(values)
        pca_centroids = torch.zeros(n_values, pca_features.shape[1], device=device)
        raw_centroids = torch.zeros(n_values, raw_features.shape[1], device=device)
        centroid_mask = torch.zeros(n_values, dtype=torch.bool)
        counts = torch.zeros(n_values)
        for i, ex in enumerate(_train_ds[: pca_features.shape[0]]):
            ci = task.intervention_value_index(ex)
            pca_centroids[ci] += pca_features[i].to(device)
            raw_centroids[ci] += raw_features[i].to(device)
            counts[ci] += 1
        for ci in range(n_values):
            if counts[ci] > 0:
                pca_centroids[ci] /= counts[ci]
                raw_centroids[ci] /= counts[ci]
                centroid_mask[ci] = True

        # Derive intrinsic (spline) centroids from PCA centroids
        manifold_obj = None
        feat_mean = feat_std = None
        # Determine whether this featurizer pipeline carries a manifold stage.
        has_manifold = len(featurizer.stages) >= 2 and hasattr(
            featurizer.stages[-1].featurizer, "manifold"
        )
        if has_manifold:
            manifold_obj = featurizer.stages[-1].featurizer.manifold.to(device)
            std_stage = featurizer.stages[-2].featurizer
            feat_mean = std_stage._mean.to(device)
            feat_std = std_stage._std.to(device)
            standardized = (pca_centroids - feat_mean) / (feat_std + 1e-6)
            spline_centroids, _ = manifold_obj.encode(standardized)
        else:
            logger.warning(
                "No manifold attached to featurizer (path_mode requires manifold); "
                "falling back to PCA centroids."
            )
            spline_centroids = pca_centroids

        from causalab.analyses.path_steering.path_mode import (
            _build_geodesic_path,
            _build_linear_path_kd,
        )

        for pm in path_modes:
            logger.info("=== Path mode: %s ===", pm.label)

            if recompute_isometry:
                # Lightweight: skip path interventions and all other criteria.
                # Uses cached manifolds + centroids; no model forward passes.
                if "isometry" in metrics:
                    try:
                        _compute_and_save_isometry(
                            pm=pm,
                            featurizer=featurizer,
                            belief_manifold_eval=belief_manifold_eval,
                            isometry_cfg=criteria_cfgs.get(
                                "isometry", OmegaConf.create({})
                            ),
                            out_dir=out_dir,
                            criteria_results=criteria_results,
                        )
                    except Exception as e:
                        logger.warning(
                            "recompute_isometry [%s] failed: %s",
                            pm.label,
                            e,
                            exc_info=True,
                        )
                continue

            # Swap featurizer if this mode has an override
            if pm.featurizer_override is not None:
                for u in interchange_target.flatten():
                    u.set_featurizer(pm.featurizer_override)

            try:
                # Select centroids in the appropriate space for this path mode
                centroids = pm.select_centroids(
                    intrinsic=spline_centroids,
                    raw=raw_centroids,
                    pca=pca_centroids,
                )

                # Optional extras-side state, populated only if n_extra > 0 below.
                vertices_belief: list[Tensor] | None = None
                sampled_pairs: list[tuple[int, int]] = []

                # Try loading cached distributions
                from safetensors.torch import load_file as _sf_load, save_file

                paths_dir = os.path.join(out_dir, "paths", pm.label)
                cached_path = os.path.join(paths_dir, "pair_distributions.safetensors")

                # Try loading from cache, filtering to requested pairs
                pair_distributions = None
                pair_grid_points = None
                if os.path.exists(cached_path):
                    _cached = _sf_load(cached_path)
                    _cached_dists = _cached["pair_distributions"]
                    _cached_gp = _cached.get("pair_grid_points", None)
                    if _cached_dists.shape[1] != n_total_steps:
                        logger.warning(
                            "Cached path step count %d != expected %d (oversteer changed?) — recomputing",
                            _cached_dists.shape[1],
                            n_total_steps,
                        )
                    else:
                        _pairs_json_path = os.path.join(paths_dir, "pairs.json")
                        if os.path.exists(_pairs_json_path):
                            with open(_pairs_json_path) as f:
                                _cached_pairs = [
                                    tuple(p) for p in json.load(f)["pairs"]
                                ]
                            # Find which requested pairs exist in cache
                            _matched = [
                                _cached_pairs.index(p)
                                for p in pairs
                                if p in _cached_pairs
                            ]
                            if len(_matched) == len(pairs):
                                pair_distributions = _cached_dists[_matched]
                                pair_grid_points = (
                                    _cached_gp[_matched]
                                    if _cached_gp is not None
                                    else None
                                )
                                computed_pairs = pairs
                                logger.info("Loaded %d pairs from cache", len(pairs))
                            elif replot_only and _matched:
                                # Partial cache: plot only the pairs we have.
                                pair_distributions = _cached_dists[_matched]
                                pair_grid_points = (
                                    _cached_gp[_matched]
                                    if _cached_gp is not None
                                    else None
                                )
                                _requested = list(pairs)
                                computed_pairs = [
                                    p for p in _requested if p in _cached_pairs
                                ]
                                _missing = [
                                    (value_strs[ci], value_strs[cj])
                                    for (ci, cj) in _requested
                                    if (ci, cj) not in _cached_pairs
                                ]
                                _log_cache_misses(
                                    f"pair_distributions[{pm.label}]",
                                    _missing,
                                )
                                pairs = computed_pairs
                                logger.info(
                                    "replot_only=True: loaded %d/%d requested pairs from cache",
                                    len(computed_pairs),
                                    len(_requested),
                                )
                            elif replot_only and not _matched:
                                _missing = [
                                    (value_strs[ci], value_strs[cj])
                                    for (ci, cj) in pairs
                                ]
                                _log_cache_misses(
                                    f"pair_distributions[{pm.label}]",
                                    _missing,
                                )

                # Reconstruct grid points if missing
                if pair_distributions is not None and pair_grid_points is None:
                    import torch as _torch

                    _gp_list = []
                    for si, ei in pairs:
                        gp = pm.build_path(
                            centroids[si],
                            centroids[ei],
                            num_steps,
                            manifold_obj=manifold_obj,
                            oversteer_frac=oversteer_frac,
                            oversteer_steps=oversteer_steps,
                        )
                        _gp_list.append(gp.cpu())
                    pair_grid_points = _torch.stack(_gp_list)
                    logger.info("Reconstructed grid points: %s", pair_grid_points.shape)
                if pair_distributions is None and replot_only:
                    logger.warning(
                        "replot_only=True but no cached pair_distributions for "
                        "%s at %s — skipping this path mode.",
                        pm.label,
                        cached_path,
                    )
                    continue
                if pair_distributions is None:
                    # Collect distributions for requested pairs
                    from tqdm import tqdm

                    pair_dists_list = []
                    pair_grid_points_list = []
                    computed_pairs = []
                    for pi, (si, ei) in enumerate(
                        tqdm(pairs, desc=f"Collecting {pm.label} paths")
                    ):
                        if not centroid_mask[si] or not centroid_mask[ei]:
                            logger.warning(
                                "Skipping pair (%d, %d): missing centroid", si, ei
                            )
                            continue
                        grid_points = pm.build_path(
                            centroids[si],
                            centroids[ei],
                            num_steps,
                            manifold_obj=manifold_obj,
                            oversteer_frac=oversteer_frac,
                            oversteer_steps=oversteer_steps,
                        )
                        from causalab.methods.steer.collect import (
                            collect_grid_distributions,
                        )

                        # (num_steps, n_prompts, W)
                        probs = collect_grid_distributions(
                            pipeline=pipeline,
                            grid_points=grid_points,
                            interchange_target=interchange_target,
                            filtered_samples=eval_samples,
                            var_indices=var_indices,
                            batch_size=batch_size,
                            n_base_samples=n_prompts,
                            average=False,
                            full_vocab_softmax=True,
                        )
                        pair_dists_list.append(probs)
                        pair_grid_points_list.append(grid_points.cpu())
                        computed_pairs.append((si, ei))

                    # (n_pairs, num_steps, n_prompts, W)
                    pair_distributions = torch.stack(pair_dists_list)
                    # (n_pairs, num_steps, d)
                    pair_grid_points = torch.stack(pair_grid_points_list)
                    logger.info(
                        "Collected pair distributions: %s",
                        pair_distributions.shape,
                    )

                    # Save paths — merge into existing cache if selected_pairs
                    os.makedirs(paths_dir, exist_ok=True)
                    _existing_compatible = False
                    if analysis.selected_pairs is not None and os.path.exists(
                        cached_path
                    ):
                        _existing = _sf_load(cached_path)
                        # Only merge if cached step count matches the freshly computed one;
                        # otherwise the stale cache is invalid (e.g. oversteer/num_steps changed)
                        # and we overwrite it.
                        _existing_compatible = (
                            _existing["pair_distributions"].shape[1]
                            == pair_distributions.shape[1]
                        )
                    if _existing_compatible:
                        # Merge new pairs into existing cache
                        _pairs_json_path = os.path.join(paths_dir, "pairs.json")
                        with open(_pairs_json_path) as f:
                            _existing_pairs = [tuple(p) for p in json.load(f)["pairs"]]
                        _all_pairs = list(_existing_pairs)
                        _all_dists = [_existing["pair_distributions"]]
                        # Only use existing grid_points if it matches the pair count
                        _existing_gp = _existing.get("pair_grid_points")
                        _gp_valid = (
                            _existing_gp is not None
                            and _existing_gp.shape[0] == len(_existing_pairs)
                            and pair_grid_points is not None
                        )
                        _all_gp = [_existing_gp] if _gp_valid else []
                        for pi, p in enumerate(pairs):
                            if p not in _existing_pairs:
                                _all_pairs.append(p)
                                _all_dists.append(pair_distributions[pi : pi + 1])
                                if _gp_valid:
                                    _all_gp.append(pair_grid_points[pi : pi + 1])
                        merged_dists = torch.cat(_all_dists, dim=0).float()
                        save_dict = {"pair_distributions": merged_dists}
                        if _all_gp:
                            save_dict["pair_grid_points"] = torch.cat(
                                _all_gp, dim=0
                            ).float()
                        save_file(save_dict, cached_path)
                        with open(_pairs_json_path, "w") as f:
                            json.dump(
                                {
                                    "pairs": _all_pairs,
                                    "values": [str(v) for v in values],
                                    "n_normal_steps": num_steps,
                                },
                                f,
                                indent=2,
                            )
                        logger.info(
                            "Merged %d new pairs into cache (%d total)",
                            len(pairs) - len(_existing_pairs),
                            len(_all_pairs),
                        )
                    else:
                        save_file(
                            {
                                "pair_distributions": pair_distributions.float(),
                                "pair_grid_points": pair_grid_points.float(),
                            },
                            cached_path,
                        )
                        with open(os.path.join(paths_dir, "pairs.json"), "w") as f:
                            json.dump(
                                {
                                    "pairs": computed_pairs,
                                    "values": [str(v) for v in values],
                                    "n_normal_steps": num_steps,
                                },
                                f,
                                indent=2,
                            )

                # Collect for belief-space path plot
                if "_belief_space_dists" not in locals():
                    _belief_space_dists = {}
                _belief_space_dists[pm.label] = pair_distributions

                # Stash per-mode grid points and distributions
                if pm.label == "geometric" and pair_grid_points is not None:
                    _geo_pair_grid_points = pair_grid_points
                    _geo_pair_distributions = pair_distributions
                if pm.label == "linear" and pair_grid_points is not None:
                    _lin_pair_grid_points = pair_grid_points
                    _lin_pair_distributions = pair_distributions

                # Extra pairs: sample N additional cross-path pairs (not on any
                # existing i↔j geodesic) using the same (i, j, f) interior
                # construction as isometry's dense grid.
                n_extra = int(analysis.get("n_extra_pairs", 0))
                if replot_only:
                    n_extra = 0
                if n_extra > 0:
                    import random as _random

                    K_iso = (
                        int(
                            criteria_cfgs.get("isometry", OmegaConf.create({})).get(
                                "n_interior_per_pair", 0
                            )
                        )
                        or 4
                    )

                    # Build vertex set in the space this mode operates in. The
                    # logical vertex layout (W centroids + K interior points
                    # along each i↔j pair) is identical across modes, so the
                    # shared rng seed below picks the same logical pairs —
                    # required for paired t-tests across modes. Interior
                    # points come from pm.build_path so geometric modes get
                    # per-dim periodic shortest-arc handling for free.
                    cents = centroids
                    W_extras = len(values)
                    # Vertices in this mode's space (activation-side; existing).
                    vertices = [(cents[i], frozenset({i})) for i in range(W_extras)]
                    # Parallel vertex set in belief u-space, by isometry: each
                    # interior vertex at fraction f along the activation-side
                    # i↔j geodesic maps to the same fraction f along the belief
                    # u-space i↔j geodesic. Same K_iso interior points so the
                    # alpha-grid lines up.
                    if belief_manifold_eval is not None:
                        belief_cp = belief_manifold_eval.control_points
                        vertices_belief = [belief_cp[i] for i in range(W_extras)]
                    for i in range(W_extras):
                        for j in range(i + 1, W_extras):
                            interior = pm.build_path(
                                cents[i],
                                cents[j],
                                K_iso + 2,
                                manifold_obj=manifold_obj,
                            )[1 : K_iso + 1]
                            for k in range(K_iso):
                                vertices.append((interior[k], frozenset({i, j})))
                            if vertices_belief is not None:
                                belief_interior = _build_geodesic_path(
                                    belief_cp[i],
                                    belief_cp[j],
                                    K_iso + 2,
                                    belief_manifold_eval,
                                )[1 : K_iso + 1]
                                for k in range(K_iso):
                                    vertices_belief.append(belief_interior[k])

                    # Filter pairs to those whose support union has >= 3 distinct
                    # centroid indices (i.e., not on any single i↔j geodesic).
                    candidate_pairs = []
                    for ai in range(len(vertices)):
                        for bi in range(ai + 1, len(vertices)):
                            sa, sb = vertices[ai][1], vertices[bi][1]
                            if len(sa | sb) >= 3:
                                candidate_pairs.append((ai, bi))

                    # Shared seed across modes so each mode samples the same
                    # logical (vertex-index) pairs — paired t-test over modes
                    # then compares the corresponding paths between the same
                    # logical endpoints.
                    rng = _random.Random(cfg.seed)
                    n_sample = min(n_extra, len(candidate_pairs))
                    sampled_pairs = rng.sample(candidate_pairs, n_sample)

                    # Cache: extras are deterministic in (seed, n_extra, K, num_steps,
                    # n_prompts, V). If those match what's on disk, reuse.
                    extras_cache = os.path.join(
                        paths_dir,
                        "pair_distributions_extra.safetensors",
                    )
                    extras_meta_path = os.path.join(paths_dir, "extras_meta.json")
                    cache_meta = {
                        "seed": int(cfg.seed),
                        "n_extra": int(n_extra),
                        "K": int(K_iso),
                        "num_steps": int(num_steps),
                        "n_prompts": int(n_prompts),
                        "n_vertices": int(len(vertices)),
                        "sampled_pairs": [list(p) for p in sampled_pairs],
                    }
                    extra_pair_distributions = None
                    if os.path.exists(extras_cache) and os.path.exists(
                        extras_meta_path
                    ):
                        with open(extras_meta_path) as _f:
                            saved_meta = json.load(_f)
                        if saved_meta == cache_meta:
                            _cached_extras = _sf_load(extras_cache)
                            extra_pair_distributions = _cached_extras[
                                "pair_distributions"
                            ]
                            logger.info(
                                "Loaded %d cached extra pairs for %s",
                                extra_pair_distributions.shape[0],
                                pm.label,
                            )

                    if extra_pair_distributions is None:
                        from causalab.methods.steer.collect import (
                            collect_grid_distributions,
                        )

                        extra_dists_list = []
                        for ai, bi in sampled_pairs:
                            coord_a, _ = vertices[ai]
                            coord_b, _ = vertices[bi]
                            grid_points = pm.build_path(
                                coord_a,
                                coord_b,
                                num_steps,
                                manifold_obj=manifold_obj,
                                oversteer_frac=oversteer_frac,
                                oversteer_steps=oversteer_steps,
                            )
                            probs = collect_grid_distributions(
                                pipeline=pipeline,
                                grid_points=grid_points,
                                interchange_target=interchange_target,
                                filtered_samples=eval_samples,
                                var_indices=var_indices,
                                batch_size=batch_size,
                                n_base_samples=n_prompts,
                                average=False,
                                full_vocab_softmax=True,
                            )
                            extra_dists_list.append(probs)
                        if extra_dists_list:
                            extra_pair_distributions = torch.stack(extra_dists_list)
                            os.makedirs(paths_dir, exist_ok=True)
                            save_file(
                                {
                                    "pair_distributions": extra_pair_distributions.float()
                                },
                                extras_cache,
                            )
                            with open(extras_meta_path, "w") as _f:
                                json.dump(cache_meta, _f, indent=2)

                    if extra_pair_distributions is not None:
                        pair_distributions = torch.cat(
                            [pair_distributions, extra_pair_distributions],
                            dim=0,
                        )
                        logger.info(
                            "Added %d extra cross-path pairs to %s (total %d pairs)",
                            extra_pair_distributions.shape[0],
                            pm.label,
                            pair_distributions.shape[0],
                        )

                # Per-pair belief-space endpoints for ALL paths (centroid pairs
                # + extras), enabling the matched-fraction geodesic reference
                # for every row. Centroid endpoints come from
                # belief_manifold.control_points; extras endpoints come from
                # vertices_belief, which was built parallel to activation-side
                # vertices using the activation↔belief manifold isometry.
                path_belief_endpoints = None
                if belief_manifold_eval is not None:
                    belief_cp = belief_manifold_eval.control_points
                    eps_list = [
                        torch.stack([belief_cp[i].cpu(), belief_cp[j].cpu()])
                        for i, j in pairs
                    ]
                    if vertices_belief is not None:
                        for ai, bi in sampled_pairs:
                            eps_list.append(
                                torch.stack(
                                    [
                                        vertices_belief[ai].cpu(),
                                        vertices_belief[bi].cpu(),
                                    ]
                                )
                            )
                    path_belief_endpoints = torch.stack(eps_list)

                # Aggregate criteria are skipped under cherry-pick (selected_pairs).
                # `pairs` IS the set of pairs we computed on; bias is prevented by
                # omitting writes entirely rather than masking after the fact.
                if analysis.selected_pairs is None:
                    for metric_name, metric_mod in path_based_metrics:
                        result_key = f"{metric_name}/{pm.label}"
                        criteria_dir = os.path.join(out_dir, "criteria")
                        try:
                            score = metric_mod.compute_score(
                                pair_distributions=pair_distributions,
                                output_dir=criteria_dir,
                                path_mode_label=pm.label,
                                belief_manifold=belief_manifold_eval,
                                # Per-pair belief-space endpoints — metrics that
                                # need a matched-fraction geodesic reference
                                # consume this; others ignore via **kwargs.
                                path_belief_endpoints=path_belief_endpoints,
                            )
                            criteria_results[result_key] = score
                            logger.info("  %s: %s", result_key, score)
                        except Exception as e:
                            logger.warning("  %s failed: %s", result_key, e)
                            criteria_results[result_key] = metric_mod.NAN_RESULT

                # Isometry: activation-side vs belief-manifold geometry under this path mode
                if (
                    "isometry" in metrics
                    and not replot_only
                    and analysis.selected_pairs is None
                ):
                    _compute_and_save_isometry(
                        pm=pm,
                        featurizer=featurizer,
                        belief_manifold_eval=belief_manifold_eval,
                        isometry_cfg=criteria_cfgs.get(
                            "isometry", OmegaConf.create({})
                        ),
                        out_dir=out_dir,
                        criteria_results=criteria_results,
                    )

                # Visualize all pairs from saved distributions
                from causalab.analyses.path_steering.path_visualization import (
                    plot_saved_pair_distributions,
                )

                score_labels = (
                    [str(v) for v in task.output_token_values]
                    if task.output_token_values
                    else [str(v) for v in values]
                )
                _path_viz_cfg = analysis.get("path_visualization", {})
                _colored_concepts = _path_viz_cfg.get(
                    "colored_concepts_in_legend", None
                )
                if _colored_concepts is not None:
                    _colored_concepts = list(_colored_concepts)
                _figsize = _path_viz_cfg.get("figsize", None)
                if _figsize is not None:
                    _figsize = tuple(_figsize)
                _font_scale = float(_path_viz_cfg.get("font_scale", 1.0))
                plot_saved_pair_distributions(
                    pair_distributions=pair_distributions,
                    pairs=computed_pairs,
                    value_labels=[str(v) for v in values],
                    output_dir=os.path.join(out_dir, "vis", "paths", pm.label),
                    path_mode_label=pm.label,
                    score_labels=score_labels,
                    colormap=task_cfg.get("colormap", "rainbow"),
                    output_token_values=task.output_token_values
                    or task.intervention_values,
                    per_pair_snap_indices=None,
                    color_by_dim=task_cfg.get("color_by_dim", 0),
                    figure_format=figure_fmt,
                    colored_concepts_in_legend=_colored_concepts,
                    figsize=_figsize,
                    font_scale=_font_scale,
                )

            finally:
                # Restore original featurizer
                if pm.featurizer_override is not None:
                    for u in interchange_target.flatten():
                        u.set_featurizer(featurizer)

        # Paired t-tests between path modes for the path-based criteria.
        # Compares per-pair scores under each path mode (same pair indices,
        # so paired test is appropriate). Saves samples + differences.
        # Skipped under cherry-pick (selected_pairs) — the aggregate is biased.
        # Reductions metrics may emit alongside the primary `per_pair_scores.pt`.
        # Empty suffix is the primary score (always present); other suffixes are
        # opt-in per metric (skipped silently when absent — see os.path.exists).
        # Each metric writes per-pair scores under its artifact dir as
        # `<stem>.safetensors` + `<stem>.meta.json` via save_tensors_with_meta.
        # Empty suffix is the primary score (always present); other suffixes
        # are opt-in per metric (skipped silently when absent).
        REDUCTION_STEMS = {
            "": "per_pair_scores",
            "_worst": "per_pair_scores_worst",  # coherence only
            "_geodesic": "per_pair_scores_geodesic",  # distance only
        }
        mode_labels = [pm.label for pm in path_modes]
        if len(mode_labels) >= 2 and analysis.selected_pairs is None:
            import itertools as _it
            from scipy import stats as _stats

            for metric_name, _metric_mod in path_based_metrics:
                for suffix, stem in REDUCTION_STEMS.items():
                    for mode_a, mode_b in _it.combinations(mode_labels, 2):
                        dir_a = os.path.join(out_dir, "criteria", metric_name, mode_a)
                        dir_b = os.path.join(out_dir, "criteria", metric_name, mode_b)
                        scores_path_a = os.path.join(dir_a, f"{stem}.safetensors")
                        scores_path_b = os.path.join(dir_b, f"{stem}.safetensors")
                        if not (
                            os.path.exists(scores_path_a)
                            and os.path.exists(scores_path_b)
                        ):
                            continue
                        a = load_tensors_with_meta(dir_a, stem)[0]["value"].numpy()
                        b = load_tensors_with_meta(dir_b, stem)[0]["value"].numpy()
                        if a.shape != b.shape or a.size < 2:
                            continue
                        t_stat, p_val = _stats.ttest_rel(a, b)
                        ttest_result = {
                            "test": "paired_t",
                            "metric": metric_name,
                            "reduction": suffix.lstrip("_") or "mean",
                            "mode_a": mode_a,
                            "mode_b": mode_b,
                            "n": int(a.size),
                            "mean_a": float(a.mean()),
                            "mean_b": float(b.mean()),
                            "mean_diff": float((a - b).mean()),
                            "t_statistic": float(t_stat),
                            "p_value": float(p_val),
                            "samples_a": a.tolist(),
                            "samples_b": b.tolist(),
                            "differences": (a - b).tolist(),
                        }
                        out_path = os.path.join(
                            out_dir,
                            "criteria",
                            metric_name,
                            f"paired_ttest_{mode_a}_vs_{mode_b}{suffix}.json",
                        )
                        with open(out_path, "w") as f:
                            json.dump(ttest_result, f, indent=2)
                        logger.info(
                            "  paired t-test %s%s [%s vs %s]: mean_diff=%.4f, t=%.3f, p=%.4g (n=%d)",
                            metric_name,
                            suffix,
                            mode_a,
                            mode_b,
                            ttest_result["mean_diff"],
                            ttest_result["t_statistic"],
                            ttest_result["p_value"],
                            ttest_result["n"],
                        )

        # Belief-space path visualization (MDS + Nystrom)
        if "_belief_space_dists" in locals() and _belief_space_dists:
            bm_dir = os.path.join(root, "output_manifold")
            nd_path = os.path.join(bm_dir, "per_example_output_dists.safetensors")
            pca_st_path = os.path.join(bm_dir, "hellinger_pca.safetensors")
            if os.path.exists(nd_path) and os.path.exists(pca_st_path):
                try:
                    from causalab.analyses.path_steering.path_visualization import (
                        plot_paths_in_belief_space,
                    )

                    nd = load_tensor_results(
                        bm_dir, "per_example_output_dists.safetensors"
                    )["dists"]
                    hellinger_pca = load_pca(bm_dir, "hellinger_pca")
                    belief_dir = os.path.join(out_dir, "vis", "belief_space")
                    # Regenerate train_dataset matching output_manifold's params to
                    # provide TRUE class labels (row-aligned with natural_dists).
                    _bm_train_ds, _ = generate_datasets(
                        task,
                        n_train=cfg.task.n_train,
                        n_test=cfg.task.n_test,
                        seed=cfg.seed,
                        balanced=cfg.task.get("balanced", False),
                        enumerate_all=cfg.task.enumerate_all,
                        resample_variable=cfg.task.get("resample_variable", "all"),
                    )
                    plot_paths_in_belief_space(
                        natural_dists=nd,
                        hellinger_pca=hellinger_pca,
                        all_pair_distributions=_belief_space_dists,
                        pairs=pairs,
                        value_labels=[str(v) for v in values],
                        output_dir=belief_dir,
                        path_colors=OmegaConf.to_container(
                            analysis.get("path_visualization", {}).get(
                                "path_colors", {}
                            ),
                            resolve=True,
                        )
                        or None,
                        variable_values=[str(v) for v in values],
                        colormap=task_cfg.get("colormap", "rainbow"),
                        edges=_graph_edges if "_graph_edges" in locals() else None,
                        edge_node_coords=_graph_edge_node_coords
                        if "_graph_edge_node_coords" in locals()
                        else None,
                        intervention_variable=task.intervention_variable,
                        train_dataset=_bm_train_ds,
                        belief_manifold=belief_manifold_eval,
                    )
                except Exception as e:
                    logger.warning(
                        "Belief-space path visualization failed: %s", e, exc_info=True
                    )
            else:
                logger.info(
                    "Skipping belief-space paths (missing output_manifold artifacts)"
                )

        # Visualizations
        for viz_name in visualizations:
            if viz_name == "path_visualization":
                if is_linear_baseline:
                    logger.info("Skipping %s for linear baseline", viz_name)
                    continue
                viz_cfg = analysis.get("path_visualization", {})
                try:
                    from causalab.io.plots.plot_3d_interactive import (
                        plot_3d,
                        PathTrace,
                    )
                    from causalab.analyses.activation_manifold.utils import (
                        _compute_intrinsic_ranges,
                    )

                    n_steps_3d = analysis.num_steps_along_path
                    pc = viz_cfg.get("path_colors", {})
                    pca_components_viz = viz_cfg.get("pca_components", None)
                    if pca_components_viz is not None:
                        pca_components_viz = list(pca_components_viz)

                    # Compute intrinsic ranges for manifold mesh rendering
                    manifold_ranges = None
                    if (
                        manifold_obj is not None
                        and feat_mean is not None
                        and feat_std is not None
                    ):
                        manifold_ranges = _compute_intrinsic_ranges(
                            pca_features,
                            manifold_obj,
                            feat_mean,
                            feat_std,
                        )

                    path_3d_dir = os.path.join(out_dir, "vis", "paths", "3d_paths")
                    os.makedirs(path_3d_dir, exist_ok=True)

                    for si, ei in pairs:
                        sv, ev = str(values[si]), str(values[ei])
                        path_traces = []

                        # Geometric path (intrinsic -> decode -> PCA space)
                        if manifold_obj is not None:
                            geo_path = _build_geodesic_path(
                                spline_centroids[si],
                                spline_centroids[ei],
                                n_steps_3d,
                                manifold_obj,
                            )
                            with torch.no_grad():
                                decoded = manifold_obj.decode(
                                    geo_path.to(device), r=None
                                ).to(feat_mean.device)
                                decoded = decoded * (feat_std + 1e-6) + feat_mean
                            path_traces.append(
                                PathTrace(
                                    points=decoded.cpu(),
                                    name="geometric",
                                    color=pc.get("geometric", "black"),
                                    is_intrinsic=False,
                                )
                            )

                        # Linear path (in PCA space)
                        lin_path = _build_linear_path_kd(
                            pca_centroids[si],
                            pca_centroids[ei],
                            n_steps_3d,
                        )
                        path_traces.append(
                            PathTrace(
                                points=lin_path.cpu(),
                                name="linear",
                                color=pc.get("linear", "darkgray"),
                                is_intrinsic=False,
                            )
                        )

                        # For 2D+ tasks, provide per-dimension params so the
                        # dropdown colors by row/column rather than flat index.
                        import numpy as _np

                        n_feat = pca_features.shape[0]
                        _edge_coords_viz = (
                            _graph_edge_node_coords
                            if "_graph_edge_node_coords" in locals()
                            else None
                        )
                        if (
                            values
                            and isinstance(values[0], (tuple, list))
                            and len(values[0]) > 1
                        ):
                            coord_names = getattr(task, "coordinate_names", None)
                            n_dims = len(values[0])
                            param_dict_viz = {}
                            dim_labels = []
                            for d in range(n_dims):
                                dim_label = (
                                    coord_names[d]
                                    if coord_names and d < len(coord_names)
                                    else f"dim{d}"
                                )
                                dim_labels.append(dim_label)
                                param_dict_viz[dim_label] = _np.array(
                                    [
                                        float(
                                            values[task.intervention_value_index(ex)][d]
                                        )
                                        for ex in _train_ds[:n_feat]
                                    ],
                                    dtype=float,
                                )
                            # Rebuild edge_node_coords with per-dim keys
                            if _edge_coords_viz is not None:
                                _edge_coords_viz = {}
                                for node_id in range(len(values)):
                                    _edge_coords_viz[node_id] = {
                                        dim_labels[d]: float(values[node_id][d])
                                        for d in range(n_dims)
                                    }
                        else:
                            param_dict_viz = {
                                task.intervention_variable or "class": _np.array(
                                    [
                                        task.intervention_value_index(ex)
                                        for ex in _train_ds[:n_feat]
                                    ],
                                    dtype=float,
                                ),
                            }
                        plot_3d(
                            features=pca_features.cpu()
                            if hasattr(pca_features, "cpu")
                            else pca_features,
                            output_path=os.path.join(path_3d_dir, f"{sv}_{ev}.html"),
                            param_dict=param_dict_viz,
                            intervention_variable=task.intervention_variable,
                            variable_values=[str(v) for v in values],
                            colormap=task_cfg.get("colormap", "rainbow"),
                            manifold_obj=manifold_obj,
                            mean=feat_mean.cpu() if feat_mean is not None else None,
                            std=feat_std.cpu() if feat_std is not None else None,
                            ranges=manifold_ranges,
                            paths=path_traces,
                            edges=_graph_edges if "_graph_edges" in locals() else None,
                            edge_node_coords=_edge_coords_viz,
                            pca_components=pca_components_viz,
                        )
                        if (
                            figure_fmt
                            and figure_fmt != "html"
                            and manifold_obj is not None
                        ):
                            from causalab.io.plots.pca_scatter import (
                                plot_manifold_3d_static,
                            )

                            plot_manifold_3d_static(
                                features=pca_features.cpu()
                                if hasattr(pca_features, "cpu")
                                else pca_features,
                                manifold_obj=manifold_obj,
                                mean=feat_mean.cpu() if feat_mean is not None else None,
                                std=feat_std.cpu() if feat_std is not None else None,
                                ranges=manifold_ranges,
                                output_path=os.path.join(
                                    path_3d_dir, f"{sv}_{ev}.{figure_fmt}"
                                ),
                                param_dict=param_dict_viz,
                                title=f"Steering: {sv} to {ev}",
                                intervention_variable=task.intervention_variable,
                                colormap=task_cfg.get("colormap", "rainbow"),
                                variable_values=[str(v) for v in values],
                                figure_format=figure_fmt,
                                paths=path_traces,
                            )
                    logger.info("Saved %d 3D path plots to %s", len(pairs), path_3d_dir)
                except Exception as e:
                    logger.warning("3D path visualization failed: %s", e, exc_info=True)
            elif viz_name == "isometry_visualization":
                from causalab.methods.scores.isometry import visualize_isometry

                iso_root = os.path.join(out_dir, "criteria", "isometry")
                iso_mode_dirs = []
                if os.path.isdir(iso_root):
                    for mode_name in sorted(os.listdir(iso_root)):
                        mode_dir = os.path.join(iso_root, mode_name)
                        if os.path.isdir(mode_dir) and os.path.exists(
                            os.path.join(mode_dir, "tensors.safetensors")
                        ):
                            iso_mode_dirs.append((mode_name, mode_dir))
                if iso_mode_dirs:
                    iso_viz_cfg = analysis.get(
                        "isometry_visualization", OmegaConf.create({})
                    )
                    iso_task_cfg = task_cfg.get("isometry", OmegaConf.create({}))
                    iso_distance_fn = iso_task_cfg.get(
                        "distance_function",
                        task_cfg.get("distance_function", "hellinger"),
                    )
                    iso_grid_range = iso_task_cfg.get("grid_range", None)
                    for mode_name, mode_dir in iso_mode_dirs:
                        try:
                            visualize_isometry(
                                artifact_dir=mode_dir,
                                viz_cfg=iso_viz_cfg,
                                distance_function=iso_distance_fn,
                                variable_values=[str(v) for v in values],
                                grid_range=list(iso_grid_range)
                                if iso_grid_range is not None
                                else None,
                                output_dir=os.path.join(
                                    out_dir, "vis", "isometry", mode_name
                                ),
                                colormap=task_cfg.get("colormap", "Viridis"),
                                edges=_graph_edges
                                if "_graph_edges" in locals()
                                else None,
                            )
                            logger.info(
                                "Isometry visualization saved for %s", mode_name
                            )
                        except Exception as e:
                            logger.warning(
                                "Isometry visualization failed for %s: %s",
                                mode_name,
                                e,
                                exc_info=True,
                            )
                else:
                    logger.info("Skipping isometry_visualization: no artifacts found")
            elif viz_name == "dual_manifold":
                if is_linear_baseline:
                    logger.info("Skipping %s for linear baseline", viz_name)
                    continue
                if (
                    "_geo_pair_grid_points" not in locals()
                    or "_geo_pair_distributions" not in locals()
                ):
                    logger.info("Skipping dual_manifold: no geometric path data")
                    continue
                bm_dir_dm = os.path.join(root, "output_manifold")
                nd_path = os.path.join(
                    bm_dir_dm, "per_example_output_dists.safetensors"
                )
                pca_st_path_bel = os.path.join(bm_dir_dm, "hellinger_pca.safetensors")
                if not (os.path.exists(nd_path) and os.path.exists(pca_st_path_bel)):
                    logger.info(
                        "Skipping dual_manifold: missing output_manifold artifacts"
                    )
                    continue
                try:
                    from causalab.io.plots.dual_manifold import (
                        DualManifoldData,
                        save_dual_manifold_html,
                    )
                    from causalab.analyses.path_steering.path_visualization import (
                        _is_2d_spatial,
                    )
                    from safetensors.torch import load_file as _sf_load_bel

                    nd = _sf_load_bel(nd_path)["dists"]
                    _hellinger_pca = load_pca(bm_dir_dm, "hellinger_pca")
                    n_feat = pca_features.shape[0]
                    _feat_classes = np.array(
                        [
                            task.intervention_value_index(ex)
                            for ex in _train_ds[:n_feat]
                        ],
                        dtype=int,
                    )
                    # TRUE class labels for natural_dists rows: regenerate
                    # output_manifold's train_dataset (deterministic via seed).
                    _bm_train_ds_dm, _ = generate_datasets(
                        task,
                        n_train=cfg.task.n_train,
                        n_test=cfg.task.n_test,
                        seed=cfg.seed,
                        balanced=cfg.task.get("balanced", False),
                        enumerate_all=cfg.task.enumerate_all,
                        resample_variable=cfg.task.get("resample_variable", "all"),
                    )
                    _bel_classes_true = np.array(
                        [
                            task.intervention_value_index(ex)
                            for ex in _bm_train_ds_dm[: nd.shape[0]]
                        ],
                        dtype=int,
                    )
                    _lin_gp = (
                        _lin_pair_grid_points
                        if "_lin_pair_grid_points" in locals()
                        else None
                    )
                    _lin_pd = (
                        _lin_pair_distributions
                        if "_lin_pair_distributions" in locals()
                        else None
                    )
                    _subspace_feat = featurizer.stages[0].featurizer
                    data = DualManifoldData.from_evaluate_artifacts(
                        geo_grid_points=_geo_pair_grid_points,
                        geo_distributions=_geo_pair_distributions,
                        lin_grid_points=_lin_gp,
                        lin_distributions=_lin_pd,
                        pairs=pairs,
                        manifold_obj=manifold_obj,
                        feat_mean=feat_mean,
                        feat_std=feat_std,
                        pca_features=pca_features,
                        subspace_featurizer=_subspace_feat,
                        feature_classes=_feat_classes,
                        belief_manifold=belief_manifold_eval,
                        hellinger_pca=_hellinger_pca,
                        natural_dists=nd,
                        class_labels=[str(v) for v in values],
                        output_token_values=task.output_token_values
                        or task.intervention_values,
                        edges=_graph_edges if "_graph_edges" in locals() else None,
                        n_normal_steps=num_steps if oversteer_frac > 0 else None,
                        bel_class_assignments_true=_bel_classes_true,
                    )
                    _cmap = task_cfg.get("colormap", "rainbow")
                    dual_dir = os.path.join(out_dir, "vis")
                    os.makedirs(dual_dir, exist_ok=True)
                    _otv = task.output_token_values or task.intervention_values
                    _is_grid = _otv is not None and _is_2d_spatial(_otv)
                    _cbd = task_cfg.get("color_by_dim", 1)
                    if _is_grid:
                        save_dual_manifold_html(
                            data,
                            os.path.join(dual_dir, "dual_manifold.html"),
                            colormap=_cmap,
                            dist_mode="grid_flow",
                            color_by_dim=_cbd,
                        )
                        save_dual_manifold_html(
                            data,
                            os.path.join(dual_dir, "dual_manifold_gridview.html"),
                            colormap=_cmap,
                            dist_mode="grid_flow_dual",
                            color_by_dim=_cbd,
                        )
                    else:
                        save_dual_manifold_html(
                            data,
                            os.path.join(dual_dir, "dual_manifold.html"),
                            colormap=_cmap,
                            dist_mode="lines",
                        )
                        save_dual_manifold_html(
                            data,
                            os.path.join(dual_dir, "dual_manifold_bars.html"),
                            colormap=_cmap,
                            dist_mode="bars",
                        )
                    logger.info("Saved dual manifold viewers to %s", dual_dir)
                except Exception as e:
                    logger.warning(
                        "Dual manifold visualization failed: %s", e, exc_info=True
                    )
            else:
                logger.warning("Unknown visualization: %s", viz_name)

        # Save metadata. Under recompute_isometry, merge our isometry-only
        # results with the existing on-disk criteria so non-isometry entries
        # from prior runs aren't clobbered.
        if recompute_isometry:
            existing_meta_path = os.path.join(out_dir, "metadata.json")
            if os.path.exists(existing_meta_path):
                with open(existing_meta_path) as f:
                    _existing = json.load(f)
                _merged = dict(_existing.get("criteria", {}))
                _merged.update(criteria_results)
                criteria_results = _merged

        eval_meta = {
            "analysis": "path_steering",
            "subspace": ss_sub,
            "activation_manifold": m_sub,
            "model": cfg.model.name,
            "task": cfg.task.name,
            "task_config": _task_config_for_metadata(
                OmegaConf.to_container(cfg.task, resolve=True)
            ),
            "criteria": criteria_results,
            "visualizations": list(viz_results.keys()),
            "path_modes": OmegaConf.to_container(
                analysis.get("path_modes", ["geometric"]), resolve=True
            ),
            "n_eval_samples": analysis.n_eval_samples,
            "seed": cfg.seed,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(eval_meta, f, indent=2)

        # CSV summary — skipped under cherry-pick (selected_pairs) since the
        # aggregate values would be biased by the cherry-picked subset.
        if analysis.selected_pairs is None:
            with open(
                os.path.join(out_dir, "results_summary.csv"), "w", newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["criterion", "value"])
                for name, val in criteria_results.items():
                    if isinstance(val, dict):
                        for k, v in val.items():
                            writer.writerow([f"{name}/{k}", v])
                    else:
                        writer.writerow([name, val])

        results[(ss_sub, m_sub)] = {
            "output_dir": out_dir,
            "criteria": criteria_results,
            "visualizations": viz_results,
        }

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Evaluate analysis complete.")
    return results
