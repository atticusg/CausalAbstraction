"""Subspace analysis: find a k-dimensional subspace for the causal variable.

Supports two modes:
- **Single-cell** (default): operates on one (layer, token_position) resolved
  from config or auto-discovered from ``locate/`` results.
- **Grid mode** (``analysis.layers`` is non-null): scans a layer x
  token_position grid, produces per-cell score heatmaps, and optionally
  runs detailed single-cell analysis on the best cell.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping, cast

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.runner.helpers import (
    resolve_task,
    prepare_datasets,
    build_targets_for_grid,
    resolve_intervention_metric,
    _task_config_for_metadata,  # pyright: ignore[reportPrivateUsage]
)
from causalab.io.pipelines import load_pipeline, load_locate_result
from causalab.io.plots.figure_format import (
    path_with_figure_format,
    resolve_figure_format_from_analysis,
)


# --------------------------------------------------------------------------- #
# Component-type dispatch                                                     #
# --------------------------------------------------------------------------- #


_VALID_COMPONENT_TYPES = ("residual_stream", "attention_head", "mlp")


def _build_targets_for_component(
    pipeline,
    task,
    component_type: str,
    layers: list[int],
    *,
    token_positions: list[str] | None,
    token_position: str | None,
    heads: list[int] | None,
):
    """Dispatch to the right spec-grid builder based on component_type.

    Returns (targets, axes_meta) where ``targets`` is a spec grid
    (:mod:`causalab.neural.activations.site_grids`) and axes_meta records the
    grid axes used for downstream plotting/metadata. Score heatmaps are
    residual-stream-only; attention_head and mlp build a single fused grid
    whose feature-count heatmap is rendered later by ``plot_feature_counts``.
    """
    if component_type == "residual_stream":
        targets, _tp_objs = build_targets_for_grid(
            pipeline,
            task,
            layers,
            token_positions,
        )
        token_position_ids = [tp.id for tp in _tp_objs]
        axes_meta = {
            "kind": "residual_stream",
            "layers": layers,
            "token_position_ids": token_position_ids,
        }
        return targets, axes_meta

    if component_type == "attention_head":
        from causalab.neural.activations.site_grids import build_attention_head_sites

        token_position_lookup = task.create_token_positions(pipeline)
        if token_position is None:
            raise ValueError(
                "subspace.token_position (singular) must be set for "
                "component_type='attention_head'."
            )
        if token_position not in token_position_lookup:
            raise ValueError(
                f"Unknown token_position {token_position!r} for task "
                f"{task.name!r}. Available: {sorted(token_position_lookup)}"
            )
        tp = token_position_lookup[token_position]

        if heads is None:
            cfg_obj = pipeline.model.config
            num_heads = (
                getattr(cfg_obj, "num_attention_heads", None)
                or getattr(cfg_obj, "n_head", None)
                or getattr(cfg_obj, "num_heads", None)
            )
            if num_heads is None:
                raise ValueError(
                    "Could not determine number of attention heads from model config."
                )
            heads = list(range(int(num_heads)))

        targets = build_attention_head_sites(
            pipeline=pipeline,
            layers=layers,
            heads=list(heads),
            token_position=tp,
            mode="one_target_all_units",
        )
        axes_meta = {
            "kind": "attention_head",
            "layers": layers,
            "heads": list(heads),
            "token_position_id": tp.id,
        }
        return targets, axes_meta

    if component_type == "mlp":
        from causalab.neural.activations.site_grids import build_mlp_sites

        token_position_lookup = task.create_token_positions(pipeline)
        if token_positions is None:
            tp_list = list(token_position_lookup.values())
        else:
            missing = [n for n in token_positions if n not in token_position_lookup]
            if missing:
                raise ValueError(
                    f"Unknown token positions {missing} for task {task.name!r}. "
                    f"Available: {sorted(token_position_lookup)}"
                )
            tp_list = [token_position_lookup[n] for n in token_positions]

        targets = build_mlp_sites(
            pipeline=pipeline,
            layers=layers,
            token_positions=tp_list,
            mode="one_target_all_units",
        )
        axes_meta = {
            "kind": "mlp",
            "layers": layers,
            "token_position_ids": [tp.id for tp in tp_list],
        }
        return targets, axes_meta

    raise ValueError(
        f"Unknown component_type: {component_type!r}. "
        f"Expected one of {_VALID_COMPONENT_TYPES}."
    )


def _save_feature_count_heatmap(
    train_result: dict,
    targets,
    output_dir: str,
    title: str,
    figure_format: str,
    n_features: int,
) -> None:
    """Save the Boundless-DAS feature-count heatmap.

    Aggregates ``feature_indices`` (keyed by opaque ``spec.key``) across all
    grid cells and joins each key's (component, layer, head/position)
    structurally from the grid's own specs
    (:func:`causalab.io.plots.grid_cells.cells_from_site_grid`) — the same
    call covers residual / attention / mlp. Passes the average test score as
    the displayed accuracy.
    """
    from causalab.io.plots.feature_masks import plot_feature_counts
    from causalab.io.plots.grid_cells import cells_from_site_grid

    feature_indices: dict = {}
    for res in train_result["results_by_key"].values():
        feature_indices.update(res.get("feature_indices", {}) or {})

    if not feature_indices:
        logger.info("No feature_indices to plot; skipping feature_counts heatmap.")
        return

    avg_score = train_result.get("avg_test_score", 0.0)

    save_path = path_with_figure_format(
        os.path.join(output_dir, "heatmaps", "feature_counts"),
        figure_format,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        component_type, cells = cells_from_site_grid(
            targets, feature_indices, n_features
        )
        plot_feature_counts(
            component_type,
            cells,
            scores=float(avg_score),
            title=title,
            save_path=save_path,
            figure_format=figure_format,
        )
    except Exception as e:
        logger.warning("feature_counts heatmap render failed: %s", e, exc_info=True)


logger = logging.getLogger(__name__)

ANALYSIS_NAME = "subspace"

# task.* keys read with no safe default: ``intervention_metric`` (direct
# attribute access) and ``colormap`` (the ``${task.colormap}`` interpolation in
# subspace.yaml). Validated up front by the runner (#264).
REQUIRED_TASK_KEYS = ("colormap", "intervention_metric")


# --------------------------------------------------------------------------- #
# Grid-mode helpers                                                           #
# --------------------------------------------------------------------------- #


def _save_grid_results(
    scores: dict[tuple, float],
    layers: list[int],
    token_position_ids: list,
    output_dir: str,
    title: str,
    figure_format: str = "png",
    train_scores: dict[tuple, float] | None = None,
    prescan_block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save grid results.json and heatmaps.  Returns the results dict."""
    from causalab.io.plots.score_heatmap import plot_residual_stream_heatmap

    # max() overloads can't infer dict.get's key-fn signature here; use lambda
    best_key = max(scores, key=lambda k: scores[k]) if scores else None

    results_data: dict[str, Any] = {
        "best_cell": (
            {"layer": best_key[0], "token_position": best_key[1]}
            if best_key is not None
            else None
        ),
        "best_layer": best_key[0] if best_key is not None else None,
        "scores_per_cell": {
            f"{layer}|{pos_id}": score for (layer, pos_id), score in scores.items()
        },
    }
    if prescan_block is not None:
        # Approx and exact side by side wherever both exist (#456).
        results_data["prescan"] = {
            "approx_scores_per_cell": {
                f"{layer}|{pos_id}": score
                for (layer, pos_id), score in prescan_block[
                    "approx_scores_per_cell"
                ].items()
            },
            "survivors": [
                f"{layer}|{pos_id}" for layer, pos_id in prescan_block["survivors"]
            ],
            "top_k": prescan_block["top_k"],
            "n_examples": prescan_block["n_examples"],
            "exact_and_approx": {
                f"{layer}|{pos_id}": pair
                for (layer, pos_id), pair in prescan_block["exact_and_approx"].items()
            },
            "agreement_at_k": prescan_block["agreement_at_k"],
            "rank_correlation": prescan_block["rank_correlation"],
            "abs_rank_correlation": prescan_block["abs_rank_correlation"],
        }
    with open(os.path.join(output_dir, "grid_results.json"), "w") as f:
        json.dump(results_data, f, indent=2)

    heatmaps_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)

    try:
        plot_residual_stream_heatmap(
            scores=scores,
            layers=layers,
            token_position_ids=token_position_ids,
            title=f"{title} (test)" if train_scores else title,
            save_path=path_with_figure_format(
                os.path.join(heatmaps_dir, "test"),
                figure_format,
            ),
            figure_format=figure_format,
        )
    except Exception as e:
        logger.warning("Test heatmap render failed: %s", e)

    if train_scores:
        try:
            plot_residual_stream_heatmap(
                scores=train_scores,
                layers=layers,
                token_position_ids=token_position_ids,
                title=f"{title} (train)",
                save_path=path_with_figure_format(
                    os.path.join(heatmaps_dir, "train"),
                    figure_format,
                ),
                figure_format=figure_format,
            )
        except Exception as e:
            logger.warning("Train heatmap render failed: %s", e)

    if prescan_block is not None and prescan_block["approx_scores_per_cell"]:
        approx = prescan_block["approx_scores_per_cell"]
        try:
            bound = max(abs(v) for v in approx.values()) or 1.0
            plot_residual_stream_heatmap(
                scores=approx,
                layers=layers,
                token_position_ids=token_position_ids,
                title=f"{title} — attribution pre-scan (approx)",
                save_path=path_with_figure_format(
                    os.path.join(heatmaps_dir, "prescan"),
                    figure_format,
                ),
                figure_format=figure_format,
                cmap="seismic",
                vmin=-bound,
                vmax=bound,
                cbar_label="approx Δ logit-diff",
            )
        except Exception as e:
            logger.warning("Prescan heatmap render failed: %s", e)

    if best_key is not None:
        logger.info(
            "Grid best cell: (layer=%s, position=%s) score=%.4f",
            best_key[0],
            best_key[1],
            scores[best_key],
        )

    return results_data


def _save_per_layer_x_pos_scatters(
    features_by_key: dict[tuple, Any],
    train_dataset: list,
    task,
    out_dir: str,
    figure_format: str,
    *,
    colormap: str | None,
) -> None:
    """Save 2D and 3D scatter plots for every (layer, position) in the grid."""
    from causalab.analyses.subspace._visualization import save_features_visualization

    embeddings = task.causal_model.embeddings or None
    intervention_variable = task.intervention_variable
    variable_values = (
        [str(v) for v in task.intervention_values] if task.intervention_values else None
    )

    layer_x_pos_dir = os.path.join(out_dir, "layer_x_pos")
    for (layer, pos_id), features in features_by_key.items():
        cell_dir = os.path.join(layer_x_pos_dir, f"L{layer}_{pos_id}")
        try:
            save_features_visualization(
                features,
                train_dataset,
                cell_dir,
                intervention_variable,
                embeddings,
                colormap=colormap,
                variable_values=variable_values,
                figure_format=figure_format,
            )
        except Exception as e:
            logger.warning(
                "Scatter for (layer=%s, pos=%s) failed: %s",
                layer,
                pos_id,
                e,
            )

    logger.info(
        "Saved per-layer_x_pos scatters for %d positions to %s",
        len(features_by_key),
        layer_x_pos_dir,
    )


def _save_pca_per_cell_artifacts(
    svd_results_by_cell: dict[tuple, dict],
    features_by_key: dict[tuple, Any],
    out_dir: str,
) -> None:
    """Save rotation matrix and projected features for every PCA grid cell.

    Writes into ``layer_x_pos/L{layer}_{pos}/`` alongside the scatter
    visualizations that ``_save_per_layer_x_pos_scatters`` already creates.
    """
    from safetensors.torch import save_file

    layer_x_pos_dir = os.path.join(out_dir, "layer_x_pos")
    saved = 0
    for (layer, pos_id), svd in svd_results_by_cell.items():
        cell_dir = os.path.join(layer_x_pos_dir, f"L{layer}_{pos_id}")
        os.makedirs(cell_dir, exist_ok=True)

        # rotation.safetensors
        rotation = svd["rotation"].contiguous()
        var_ratios = svd["explained_variance_ratio"]
        save_file(
            {
                "rotation_matrix": rotation,
                "explained_variance_ratio": torch.tensor(var_ratios),
            },
            os.path.join(cell_dir, "rotation.safetensors"),
        )

        # features/training_features.safetensors
        features = features_by_key.get((layer, pos_id))
        if features is not None:
            feat_dir = os.path.join(cell_dir, "features")
            os.makedirs(feat_dir, exist_ok=True)
            save_file(
                {"features": features.contiguous()},
                os.path.join(feat_dir, "training_features.safetensors"),
            )
        saved += 1

    logger.info(
        "Saved per-cell PCA artifacts for %d cells to %s",
        saved,
        layer_x_pos_dir,
    )


def _run_grid(
    cfg: DictConfig,
    pipeline,
    task,
    train_dataset: list,
    test_dataset: list,
    out_dir: str,
    method: str,
    k_features: int,
    batch_size: int,
    intervention_metric,
    figure_format: str,
    *,
    layers: list[int],
    token_positions: list[str] | None,
    das_training_epoch: int,
    das_lr: float,
    dbm_training_epoch: int,
    dbm_lr: float,
    dbm_regularization_coefficient: float,
    dbm_tie_masks: bool,
    boundless_training_epoch: int,
    boundless_lr: float,
    boundless_regularization_coefficient: float,
    colormap: str | None,
    component_type: str = "residual_stream",
    heads: list[int] | None = None,
    token_position: str | None = None,
    prescan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a grid scan across the component-appropriate axes.

    ``prescan`` (``{"enabled", "top_k", "n_examples"}``) is the optional
    attribution-patching fail-fast gate (#456) for the trained grids
    (``das`` / ``dbm`` / ``boundless``, residual stream): a one-backward
    gradient × Δactivation pre-scan over the training pairs prunes the grid
    to its top-k cells before any training runs; approximate and exact
    scores are reported together in ``grid_results.json``.
    """
    from causalab.analyses.subspace.grid import (
        run_das_grid,
        run_pca_grid,
        run_dbm_grid,
        run_boundless_grid,
    )
    from causalab.methods.trained_subspace.train import save_train_results

    layers = list(layers)
    position_names = list(token_positions) if token_positions is not None else None

    targets, axes_meta = _build_targets_for_component(
        pipeline,
        task,
        component_type,
        layers,
        token_positions=position_names,
        token_position=token_position,
        heads=list(heads) if heads is not None else None,
    )
    token_position_ids: list[Any] = list(axes_meta.get("token_position_ids", []) or [])

    # Optional attribution-patching fail-fast gate (CAP3, #456): approximate
    # each cell's interchange effect with one forward+backward per batch and
    # train only the top-k surviving cells.
    prescan_cfg = dict(prescan) if prescan is not None else {}
    prescan_block: dict[str, Any] | None = None
    if prescan_cfg.get("enabled"):
        if method not in ("das", "dbm", "boundless"):
            raise ValueError(
                "subspace.prescan gates the trained grids (das / dbm / "
                f"boundless), not method={method!r}. Disable the prescan."
            )
        if component_type != "residual_stream":
            raise ValueError(
                "subspace.prescan requires component_type: residual_stream — "
                "non-residual components run as a single joint target, so "
                "there is no per-cell grid to prune."
            )
        from causalab.methods.interchange import (
            counterfactual_logit_diff_ids,
            run_attribution_prescan,
            select_top_k,
        )

        n_examples = prescan_cfg.get("n_examples")
        subset = train_dataset if not n_examples else train_dataset[: int(n_examples)]
        pair_ids = counterfactual_logit_diff_ids(pipeline, subset, task.causal_model)
        approx_scores = run_attribution_prescan(
            targets, subset, pipeline, batch_size, pair_ids
        )
        top_k = int(prescan_cfg.get("top_k", 10))
        # Rank by |approx|: the linearization's sign is unreliable at early
        # layers, but the magnitude separates live cells from dead ones (see
        # causalab/methods/interchange/attribution.py).
        survivors = select_top_k(approx_scores, top_k, by_abs=True)
        logger.info(
            "Attribution pre-scan: %s trains on %d/%d cells (top_k=%d over %d pairs)",
            method,
            len(survivors),
            len(targets),
            top_k,
            len(subset),
        )
        targets = {key: targets[key] for key in survivors}
        prescan_block = {
            "approx_scores_per_cell": approx_scores,
            "survivors": survivors,
            "top_k": top_k,
            "n_examples": len(subset),
        }

    log_dir = os.path.join(out_dir, "logs")

    if method == "das":
        grid_result = run_das_grid(
            targets,
            train_dataset,
            test_dataset,
            pipeline,
            task.causal_model,
            k_features,
            batch_size,
            metric=intervention_metric,
            loss_config={
                "training_epoch": das_training_epoch,
                "init_lr": das_lr,
            },
            log_dir=log_dir,
        )
        test_scores = grid_result["test_scores"]
        train_scores = grid_result["train_scores"]
        title = f"DAS k={k_features}"
        save_train_results(grid_result["train_result"], out_dir)

    elif method == "dbm":
        grid_result = run_dbm_grid(
            targets,
            train_dataset,
            test_dataset,
            pipeline,
            task.causal_model,
            k_features,
            batch_size,
            metric=intervention_metric,
            dbm_config={
                "training_epoch": dbm_training_epoch,
                "init_lr": dbm_lr,
                "tie_masks": dbm_tie_masks,
                "masking": {
                    "regularization_coefficient": dbm_regularization_coefficient,
                },
            },
            log_dir=log_dir,
        )
        test_scores = grid_result["test_scores"]
        train_scores = grid_result["train_scores"]
        title = f"DBM k={k_features}"
        save_train_results(grid_result["train_result"], out_dir)

    elif method == "pca":
        grid_result = run_pca_grid(
            targets,
            train_dataset,
            pipeline,
            k_features,
            batch_size,
        )
        test_scores = grid_result["scores"]
        train_scores = None
        title = f"PCA explained variance (top {k_features})"

    elif method == "boundless":
        grid_result = run_boundless_grid(
            targets,
            train_dataset,
            test_dataset,
            pipeline,
            task.causal_model,
            k_features,
            batch_size,
            metric=intervention_metric,
            boundless_config={
                "training_epoch": boundless_training_epoch,
                "init_lr": boundless_lr,
                "masking": {
                    "regularization_coefficient": boundless_regularization_coefficient,
                },
            },
            log_dir=log_dir,
        )
        test_scores = grid_result["test_scores"]
        train_scores = grid_result["train_scores"]
        title = f"Boundless DAS k={k_features}"
        save_train_results(grid_result["train_result"], out_dir)

    else:
        raise ValueError(f"Unknown subspace method: {method}")

    if prescan_block is not None:
        from causalab.methods.interchange import (
            spearman_rank_correlation,
            top_k_agreement,
        )

        approx_scores = prescan_block["approx_scores_per_cell"]
        prescan_block["exact_and_approx"] = {
            key: {"approx": approx_scores[key], "exact": exact}
            for key, exact in test_scores.items()
        }
        prescan_block["agreement_at_k"] = top_k_agreement(
            approx_scores, test_scores, prescan_block["top_k"], by_abs=True
        )
        prescan_block["rank_correlation"] = spearman_rank_correlation(
            approx_scores, test_scores
        )
        prescan_block["abs_rank_correlation"] = spearman_rank_correlation(
            {key: abs(value) for key, value in approx_scores.items()},
            test_scores,
        )

    if component_type == "residual_stream":
        grid_results_data = _save_grid_results(
            scores=test_scores,
            layers=layers,
            token_position_ids=token_position_ids,
            output_dir=out_dir,
            title=title,
            figure_format=figure_format,
            train_scores=train_scores,
            prescan_block=prescan_block,
        )
    else:
        # Non-residual components run as one_target_all_units → single-cell
        # scoring; the residual-stream score heatmap doesn't apply, but we
        # still persist grid_results.json with the aggregate score(s).
        grid_results_data = {
            "scores_per_cell": {
                str(k[0] if isinstance(k, tuple) and len(k) == 1 else k): v
                for k, v in test_scores.items()
            },
            "best_cell": None,
            "component_type": component_type,
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "grid_results.json"), "w") as f:
            json.dump(grid_results_data, f, indent=2)

    # Mask-based methods: save feature-count heatmap (component type detected
    # structurally from the grid specs). For DBM with tie_masks=True this renders
    # as a binary on/off heatmap per unit; for boundless / DBM with
    # tie_masks=False it shows the per-feature-dim count selected.
    if method in ("dbm", "boundless"):
        _save_feature_count_heatmap(
            train_result=grid_result["train_result"],
            targets=targets,
            output_dir=out_dir,
            title=f"{method.upper()} feature counts (k={k_features})",
            figure_format=figure_format,
            n_features=k_features,
        )

    # Save per-cell artifacts (PCA only — scatters + rotation/features)
    if method == "pca" and "features_by_key" in grid_result:
        _save_per_layer_x_pos_scatters(
            grid_result["features_by_key"],
            train_dataset,
            task,
            out_dir,
            figure_format,
            colormap=colormap,
        )
        _save_pca_per_cell_artifacts(
            grid_result["svd_results_by_cell"],
            grid_result["features_by_key"],
            out_dir,
        )

    # Save shared training dataset at grid root (row-aligned with features)
    from causalab.io.counterfactuals import save_counterfactual_examples

    save_counterfactual_examples(
        train_dataset,
        os.path.join(out_dir, "train_dataset.json"),
    )

    # Metadata
    best_key = (
        max(test_scores, key=test_scores.get)
        if test_scores and component_type == "residual_stream"
        else None
    )
    seed = cfg.seed
    metadata: dict[str, Any] = {
        "analysis": "subspace",
        "mode": "grid",
        "method": method,
        "component_type": component_type,
        "k_features": k_features,
        "layers": layers,
        "token_positions": [str(p) for p in token_position_ids],
        "best_cell": (
            {"layer": best_key[0], "token_position": str(best_key[1])}
            if best_key and len(best_key) >= 2
            else None
        ),
        "model": cfg.model.name,
        "task": cfg.task.name,
        "task_config": _task_config_for_metadata(
            cast(dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True))
        ),
        "n_train": cfg.task.n_train,
        "seed": seed,
    }
    if component_type == "attention_head":
        metadata["heads"] = axes_meta.get("heads")
        metadata["token_position"] = axes_meta.get("token_position_id")
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    result: dict[str, Any] = {
        "method": method,
        "mode": "grid",
        "component_type": component_type,
        "k_features": k_features,
        "layers": layers,
        "grid_results": grid_results_data,
        "output_dir": out_dir,
        "metadata": metadata,
    }
    return result


def _run_best_cell_detail(  # pyright: ignore[reportUnusedFunction]
    cfg: DictConfig,
    method: str,
    sites,
    train_dataset: list,
    test_dataset: list,
    pipeline,
    task,
    k_features: int,
    batch_size: int,
    out_dir: str,
    intervention_metric,
    figure_format: str,
    *,
    colormap: str | None,
    das_training_epoch: int,
    das_lr: float,
) -> dict[str, Any] | None:
    """Run single-cell analysis on the best grid cell for scatter plots."""
    from causalab.analyses.subspace import find_pca_subspace, find_das_subspace

    best_dir = os.path.join(out_dir, "best_cell")
    os.makedirs(best_dir, exist_ok=True)

    embeddings = task.causal_model.embeddings or None
    intervention_variable = task.intervention_variable
    variable_values = (
        [str(v) for v in task.intervention_values] if task.intervention_values else None
    )

    try:
        if method == "pca":
            sub = find_pca_subspace(
                sites,
                train_dataset,
                pipeline,
                k_features,
                batch_size,
                best_dir,
                intervention_variable=intervention_variable,
                embeddings=embeddings,
                colormap=colormap,
                variable_values=variable_values,
                figure_format=figure_format,
            )
            return {
                "explained_variance_ratio": sub["explained_variance_ratio"],
                "features_shape": list(sub["features"].shape),
            }
        elif method == "das":
            sub = find_das_subspace(
                sites,
                train_dataset,
                test_dataset,
                pipeline,
                task.causal_model,
                k_features,
                batch_size,
                best_dir,
                metric=intervention_metric,
                loss_config={
                    "training_epoch": das_training_epoch,
                    "init_lr": das_lr,
                },
                intervention_variable=intervention_variable,
                embeddings=embeddings,
                colormap=colormap,
                variable_values=variable_values,
                figure_format=figure_format,
            )
            return {"features_shape": list(sub["features"].shape)}
    except Exception as e:
        logger.warning("Best-cell detail failed: %s", e, exc_info=True)
    return None


# --------------------------------------------------------------------------- #
# Single-cell mode (existing logic)                                           #
# --------------------------------------------------------------------------- #


def _run_single_cell(
    cfg: DictConfig,
    pipeline,
    task,
    train_dataset: list,
    test_dataset: list,
    layer: int,
    out_dir: str,
    method: str,
    k_features: int,
    batch_size: int,
    intervention_metric,
    figure_format: str,
    *,
    token_positions: list[str] | None,
    colormap: str | None,
    vis_dims: list[int] | None,
    detailed_hover: bool,
    max_hover_chars: int,
    das_training_epoch: int,
    das_lr: float,
    dbm_training_epoch: int = 20,
    dbm_lr: float = 0.001,
    dbm_regularization_coefficient: float = 0.1,
    dbm_tie_masks: bool = False,
    fixed_cfg: Any = None,
    position_names_override: list[str] | None = None,
) -> dict[str, Any]:
    """Run subspace analysis on a single (layer, token_position) cell."""
    from causalab.analyses.subspace import (
        find_pca_subspace,
        find_fixed_subspace,
        find_das_subspace,
        find_dbm_subspace,
        find_boundless_subspace,
        resolve_fixed_rotation,
        score_fixed_subspace,
        orient_rotation,
    )

    position_names = list(token_positions) if token_positions else None
    if position_names_override is not None:
        position_names = position_names_override
    if position_names is None:
        # No explicit position: resolve from the task rather than assuming a
        # hard-coded name (e.g. the old `last_token` default, which many tasks
        # do not define — they expose `last`). A single declared position is
        # unambiguous (the common case); multiple positions are ambiguous for a
        # single cell, so list them and ask the caller to pick one.
        available = list(task.create_token_positions(pipeline))
        if len(available) == 1:
            position_names = available
        else:
            raise ValueError(
                f"token_positions is unset and task {task.name!r} declares "
                f"{len(available)} token positions {sorted(available)}; a single "
                "cell needs exactly one. Set subspace.token_positions: [<name>] to "
                "pick one, or leave subspace.layers unset to auto-resolve "
                "(layer, token_position) from locate/ results."
            )
    targets, _tp_list = build_targets_for_grid(pipeline, task, [layer], position_names)
    sites = next(iter(targets.values()))

    result: dict[str, Any] = {
        "method": method,
        "k_features": k_features,
        "layer": layer,
        "token_position": _tp_list[0].id,
    }

    embeddings = task.causal_model.embeddings or None
    intervention_variable = task.intervention_variable
    variable_values = (
        [str(v) for v in task.intervention_values] if task.intervention_values else None
    )

    if method == "pca":
        sub = find_pca_subspace(
            sites,
            train_dataset,
            pipeline,
            k_features,
            batch_size,
            out_dir,
            intervention_variable=intervention_variable,
            embeddings=embeddings,
            colormap=colormap,
            vis_dims=vis_dims,
            variable_values=variable_values,
            detailed_hover=detailed_hover,
            max_hover_chars=max_hover_chars,
            figure_format=figure_format,
        )
        result.update(
            {
                "explained_variance_ratio": sub["explained_variance_ratio"],
                "features_shape": list(sub["features"].shape),
            }
        )
    elif method == "das":
        sub = find_das_subspace(
            sites,
            train_dataset,
            test_dataset,
            pipeline,
            task.causal_model,
            k_features,
            batch_size,
            out_dir,
            metric=intervention_metric,
            loss_config={
                "training_epoch": das_training_epoch,
                "init_lr": das_lr,
            },
            intervention_variable=intervention_variable,
            embeddings=embeddings,
            colormap=colormap,
            variable_values=variable_values,
            detailed_hover=detailed_hover,
            max_hover_chars=max_hover_chars,
            figure_format=figure_format,
        )
        result.update(
            {
                "das_result": sub["das_result"],
                "features_shape": list(sub["features"].shape),
            }
        )
    elif method == "dbm":
        sub = find_dbm_subspace(
            sites,
            train_dataset,
            test_dataset,
            pipeline,
            task.causal_model,
            k_features,
            batch_size,
            out_dir,
            metric=intervention_metric,
            dbm_config={
                "training_epoch": dbm_training_epoch,
                "init_lr": dbm_lr,
                "tie_masks": dbm_tie_masks,
                "masking": {
                    "regularization_coefficient": dbm_regularization_coefficient,
                },
            },
            figure_format=figure_format,
        )
        result.update({"dbm_result": sub["dbm_result"]})
    elif method == "boundless":
        sub = find_boundless_subspace(
            sites,
            train_dataset,
            test_dataset,
            pipeline,
            task.causal_model,
            k_features,
            batch_size,
            out_dir,
            metric=intervention_metric,
        )
        result.update({"boundless_result": sub["boundless_result"]})
    elif method == "fixed":
        rotation, provenance = resolve_fixed_rotation(fixed_cfg, out_dir=out_dir)
        # Auto-transpose / validate orientation against the model's d_model so a
        # (k, d_model) artifact resolves the same as a (d_model, k) one (#255).
        # k_features (config) disambiguates a square rotation (its orientation is
        # ambiguous even with d_model known).
        rotation = orient_rotation(
            rotation,
            d_model=int(pipeline.model.config.hidden_size),
            k_features_hint=k_features,
        )
        if int(rotation.shape[1]) != k_features:
            raise ValueError(
                f"subspace.k_features={k_features} does not match the given "
                f"rotation's k={int(rotation.shape[1])}. Set k_features to the "
                "rotation's column count."
            )
        sub = find_fixed_subspace(
            sites,
            train_dataset,
            pipeline,
            rotation,
            batch_size,
            out_dir,
            intervention_variable=intervention_variable,
            embeddings=embeddings,
            colormap=colormap,
            vis_dims=vis_dims,
            variable_values=variable_values,
            detailed_hover=detailed_hover,
            max_hover_chars=max_hover_chars,
            figure_format=figure_format,
            k_features_hint=k_features,
        )
        result.update(
            {
                "features_shape": list(sub["features"].shape),
                "fixed_provenance": provenance,
            }
        )
        # Optionally score the threaded subspace's interchange-IIA / mediation at
        # this pinned cell (subspace.fixed.score). find_fixed_subspace RETURNS the
        # spec carrying the frozen featurizer (functional attach), so thread that
        # returned spec in — the scan is then RESTRICTED to the given rotation;
        # score_fixed_subspace rebuilds a featurizer-free control for the
        # full-cell IIA. Retires the session-local fixed_subspace_mediation (#262).
        if fixed_cfg is not None and bool(fixed_cfg.get("score", False)):
            score_dataset = test_dataset if test_dataset else train_dataset
            scores = score_fixed_subspace(
                sites=[[sub["spec"]]],
                cell_key=next(iter(targets)),
                layer=layer,
                position_name=position_names[0],
                pipeline=pipeline,
                task=task,
                dataset=score_dataset,
                batch_size=batch_size,
                intervention_metric=cfg.task.intervention_metric,
            )
            result.update(scores)
            # extract_values pins float leaves of files named exactly
            # ``results.json`` (tests/end_to_end/_helpers/golden.py:318), so the
            # IIA scores must land there to be value-gated by the golden.
            with open(os.path.join(out_dir, "results.json"), "w") as f:
                json.dump(scores, f, indent=2)
    else:
        raise ValueError(f"Unknown subspace method: {method}")

    return result


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #


def _dispatch_grid_or_single(
    cfg: DictConfig,
    analysis: DictConfig,
    pipeline,
    task,
    train_dataset: list,
    test_dataset: list,
    out_dir: str,
    intervention_metric,
) -> dict[str, Any]:
    """Run one subspace pass (grid or single-cell) for a fully resolved analysis cfg.

    Extracted from ``main()`` so it can be called once per mode when
    ``analysis.modes`` is set, or once for the single-mode path.
    """
    method = analysis.method
    k_features = analysis.k_features
    batch_size = analysis.batch_size
    fixed_cfg = analysis.get("fixed", {})
    if method == "fixed":
        _fixed_layers = analysis.get("layers", None)
        if not _fixed_layers or len(list(_fixed_layers)) != 1:
            raise ValueError(
                "method: fixed requires a single explicit layer "
                "(subspace.layers: [L]) — a given rotation is tied to one "
                "(layer, token_position). Grid scan and locate-auto resolution "
                "are not supported for fixed subspaces."
            )
    seed = cfg.seed
    figure_fmt = resolve_figure_format_from_analysis(analysis)

    vis_cfg = analysis.get("visualization", {})
    colormap = vis_cfg.get("colormap", None)
    vis_dims = (
        list(vis_cfg.vis_dims) if vis_cfg.get("vis_dims", None) is not None else None
    )
    detailed_hover = vis_cfg.get("detailed_hover", False)
    max_hover_chars = vis_cfg.get("max_hover_chars", 50)
    token_positions = analysis.get("token_positions")
    component_type = analysis.get("component_type", "residual_stream")
    heads = analysis.get("heads", None)
    token_position = analysis.get("token_position", None)
    das_training_epoch = analysis.das.training_epoch
    das_lr = analysis.das.lr
    dbm_cfg = analysis.get("dbm", {})
    dbm_training_epoch = dbm_cfg.get("training_epoch", 20)
    dbm_lr = dbm_cfg.get("lr", 0.001)
    dbm_regularization_coefficient = dbm_cfg.get("regularization_coefficient", 0.1)
    dbm_tie_masks = bool(dbm_cfg.get("tie_masks", False))
    boundless_cfg = analysis.get("boundless", {})
    boundless_training_epoch = boundless_cfg.get("training_epoch", 20)
    boundless_lr = boundless_cfg.get("lr", 0.001)
    boundless_regularization_coefficient = boundless_cfg.get(
        "regularization_coefficient", 0.1
    )

    layers_cfg = analysis.get("layers", None)
    # For non-residual components in grid mode, "no layers specified" means
    # "scan all hidden layers" (one_target_all_units across the full grid).
    # The locate/ auto-resolve path is residual-stream-specific and would
    # collapse to a single cell, which is not the intent for component DBM.
    if layers_cfg is None and component_type != "residual_stream":
        layers_cfg = list(range(pipeline.model.config.num_hidden_layers))

    if layers_cfg is not None:
        layers = [int(lyr) for lyr in layers_cfg]
        if len(layers) == 1:
            # Single-cell mode
            layer = layers[0]
            result = _run_single_cell(
                cfg,
                pipeline,
                task,
                train_dataset,
                test_dataset,
                layer,
                out_dir,
                method,
                k_features,
                batch_size,
                intervention_metric,
                figure_fmt,
                token_positions=token_positions,
                colormap=colormap,
                vis_dims=vis_dims,
                detailed_hover=detailed_hover,
                max_hover_chars=max_hover_chars,
                das_training_epoch=das_training_epoch,
                das_lr=das_lr,
                dbm_training_epoch=dbm_training_epoch,
                dbm_lr=dbm_lr,
                dbm_regularization_coefficient=dbm_regularization_coefficient,
                dbm_tie_masks=dbm_tie_masks,
                fixed_cfg=fixed_cfg,
            )

            # Save metadata for single-cell mode. For method=fixed the on-disk
            # artifact is a PCA-style rotation matrix, so record method="pca"
            # (artifact-type semantics) — this is what load_subspace_onto_spec
            # branches on, so downstream consumers need no fixed-specific code.
            # The fixed origin is preserved under discovery/fixed_source.
            # _run_single_cell resolves the position (explicit name, the task's
            # sole position, or an override), so record the resolved id rather
            # than the raw config, which is null under the default.
            _tp_names = list(analysis.get("token_positions") or [])
            metadata = {
                "analysis": "subspace",
                "mode": "single",
                "method": "pca" if method == "fixed" else method,
                "k_features": k_features,
                "layer": layer,
                "token_position": result.get("token_position")
                or (str(_tp_names[0]) if _tp_names else None),
                "model": cfg.model.name,
                "task": cfg.task.name,
                "task_config": _task_config_for_metadata(
                    cast(dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True))
                ),
                "n_train": cfg.task.n_train,
                "seed": seed,
            }
            if method == "fixed":
                metadata["discovery"] = "fixed"
                metadata["fixed_source"] = result.get("fixed_provenance")
            with open(os.path.join(out_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            result["output_dir"] = out_dir
            result["metadata"] = metadata
        else:
            # Grid mode
            result = _run_grid(
                cfg,
                pipeline,
                task,
                train_dataset,
                test_dataset,
                out_dir,
                method,
                k_features,
                batch_size,
                intervention_metric,
                figure_fmt,
                layers=layers,
                token_positions=token_positions,
                das_training_epoch=das_training_epoch,
                das_lr=das_lr,
                dbm_training_epoch=dbm_training_epoch,
                dbm_lr=dbm_lr,
                dbm_regularization_coefficient=dbm_regularization_coefficient,
                dbm_tie_masks=dbm_tie_masks,
                boundless_training_epoch=boundless_training_epoch,
                boundless_lr=boundless_lr,
                boundless_regularization_coefficient=boundless_regularization_coefficient,
                colormap=colormap,
                component_type=component_type,
                heads=list(heads) if heads is not None else None,
                token_position=token_position,
                prescan=cast(
                    Mapping[str, Any],
                    OmegaConf.to_container(analysis.prescan, resolve=True),
                )
                if analysis.get("prescan") is not None
                else None,
            )
    else:
        # Auto-resolve from locate/
        locate_result = load_locate_result(cfg.experiment_root)
        if not locate_result:
            raise ValueError(
                "No locate/ results found and no layers specified. "
                "Either run the locate analysis first or pass analysis.layers=[<int>]."
            )
        best_cell = locate_result.get("best_cell") or {}
        layer = best_cell.get("layer") or locate_result.get("best_layer")
        token_position_from_locate = best_cell.get("token_position")
        if not token_position_from_locate:
            raise ValueError(
                "locate/ results do not contain best_cell.token_position. "
                "Please set subspace.token_positions explicitly."
            )
        if layer is None:
            raise ValueError(
                "locate/ results do not contain best_cell.layer or best_layer. "
                "Please set subspace.layers explicitly."
            )
        logger.info(
            "Auto-resolved layer=%d, token_position=%s from locate/ results",
            layer,
            token_position_from_locate,
        )

        result = _run_single_cell(
            cfg,
            pipeline,
            task,
            train_dataset,
            test_dataset,
            layer,
            out_dir,
            method,
            k_features,
            batch_size,
            intervention_metric,
            figure_fmt,
            token_positions=token_positions,
            colormap=colormap,
            vis_dims=vis_dims,
            detailed_hover=detailed_hover,
            max_hover_chars=max_hover_chars,
            das_training_epoch=das_training_epoch,
            das_lr=das_lr,
            dbm_training_epoch=dbm_training_epoch,
            dbm_lr=dbm_lr,
            dbm_regularization_coefficient=dbm_regularization_coefficient,
            dbm_tie_masks=dbm_tie_masks,
            position_names_override=[token_position_from_locate],
        )

        # Save metadata for single-cell mode
        metadata = {
            "analysis": "subspace",
            "mode": "single",
            "method": method,
            "k_features": k_features,
            "layer": layer,
            "token_position": token_position_from_locate,
            "model": cfg.model.name,
            "task": cfg.task.name,
            "task_config": _task_config_for_metadata(
                cast(dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True))
            ),
            "n_train": cfg.task.n_train,
            "seed": seed,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        result["output_dir"] = out_dir
        result["metadata"] = metadata

    return result


def _mode_out_dir(
    base_out_dir: str, mode_analysis: DictConfig, mode_name: str, tv: str | None
) -> str:
    """Per-mode output dir that re-resolves ``_subdir`` from the mode's overrides.

    ``base_out_dir`` bakes in the analysis-level ``_subdir`` (resolved once,
    e.g. ``pca_k64``). A mode that overrides a ``_subdir``-feeding field such as
    ``k_features`` would otherwise collide with sibling modes under that same
    dir — both ``modes: [{k_features: 16}, {k_features: 64}]`` entries landed in
    ``subspace/pca_k64/`` and the k=16 result was overwritten (#171, sub-issue
    C6). Recompute the leaf subdir from the merged per-mode config instead.

    When a mode does not change ``_subdir`` (e.g. the component-mode ``name:``
    pattern), ``mode_analysis._subdir == basename(base_out_dir)``, so this
    returns ``base_out_dir/mode_name[/tv]`` — identical to the prior behavior.
    """
    subspace_root = os.path.dirname(base_out_dir)
    out_dir = os.path.join(subspace_root, str(mode_analysis._subdir), mode_name)
    if tv:
        out_dir = os.path.join(out_dir, tv)
    return out_dir


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the subspace analysis: find a k-dimensional subspace for the causal variable.

    Supports a single-mode pass (default) or a multi-mode sweep when
    ``cfg.subspace.modes`` is set. Each mode entry is merged onto the
    analysis config and writes its artifacts to a per-mode subdirectory.
    """
    analysis = cfg[ANALYSIS_NAME]

    base_out_dir = analysis._output_dir
    tv = cfg.task.get("target_variable")
    os.makedirs(base_out_dir, exist_ok=True)

    # Shared loads (task, datasets, pipeline) — happen once across modes.
    task, _task_cfg_raw = resolve_task(
        task_name=cfg.task.name,
        task_config=cast(
            dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True)
        ),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )
    # The intervention-success metric is the task's own checker (#167) — no
    # lenient-containment default. ``_comparison_fn`` (distribution) is unused
    # here; subspace methods score via ``intervention_metric``.
    string_metric, _comparison_fn = resolve_intervention_metric(
        cfg.task.intervention_metric, checker=task.checker
    )
    intervention_metric = string_metric
    pipeline = load_pipeline(
        model_name=cfg.model.name,
        task=task,
        max_new_tokens=cfg.task.max_new_tokens,
        device=cfg.model.device,
        dtype=cfg.model.get("dtype"),
        eager_attn=cfg.model.get("eager_attn"),
        use_chat_template=cfg.model.get("chat_template", False),
        chat_answer_directive=cfg.model.get("chat_answer_directive"),
    )
    # Correct-only filtering is a REQUIRED, explicit choice (no silent default):
    # a trained subspace fit on prompts the model gets wrong cannot recover a real
    # subspace (the causal signal isn't there to align to). Declare
    # `subspace.filter_correct` in the runner config — there is no fallback.
    if OmegaConf.is_missing(analysis, "filter_correct"):
        raise ValueError(
            "subspace.filter_correct is required (no default). Set it true "
            "(correct-only — trained subspaces need it; matches the paper's "
            "sampling) or false (e.g. tiny-random smoke, where the model solves "
            "nothing so filtering would empty the dataset)."
        )
    train_dataset, test_dataset = prepare_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
        filter_correct=analysis.filter_correct,
        pipeline=pipeline,
        metric=task.checker,
        filter_batch_size=analysis.batch_size,
    )

    try:
        modes = analysis.get("modes", None)
        if modes:
            results_by_mode: dict[str, Any] = {}
            for mode in modes:
                mode_dict = cast(
                    dict[str, Any],
                    OmegaConf.to_container(mode, resolve=True)
                    if isinstance(mode, DictConfig)
                    else dict(mode),
                )
                mode_name = (
                    mode_dict.get("name") or mode_dict.get("component_type") or "mode"
                )
                # Per-mode analysis cfg = analysis defaults + mode overrides.
                # OmegaConf.merge returns ListConfig | DictConfig; merging two
                # DictConfigs always yields a DictConfig here.
                mode_analysis = cast(
                    DictConfig,
                    OmegaConf.merge(
                        analysis,
                        OmegaConf.create(
                            {k: v for k, v in mode_dict.items() if k != "name"}
                        ),
                    ),
                )
                # Per-mode output dir: re-resolve _subdir from the mode's
                # overrides so modes differing by k/method don't collide, then
                # mode_name, then target_variable.
                mode_out_dir = _mode_out_dir(base_out_dir, mode_analysis, mode_name, tv)
                os.makedirs(mode_out_dir, exist_ok=True)
                logger.info(
                    "=== subspace mode: %s (component_type=%s) ===",
                    mode_name,
                    mode_analysis.get("component_type", "residual_stream"),
                )
                results_by_mode[mode_name] = _dispatch_grid_or_single(
                    cfg,
                    mode_analysis,
                    pipeline,
                    task,
                    train_dataset,
                    test_dataset,
                    mode_out_dir,
                    intervention_metric,
                )
            result = {
                "modes": results_by_mode,
                "output_dir": base_out_dir,
            }
        else:
            out_dir = os.path.join(base_out_dir, tv) if tv else base_out_dir
            os.makedirs(out_dir, exist_ok=True)
            result = _dispatch_grid_or_single(
                cfg,
                analysis,
                pipeline,
                task,
                train_dataset,
                test_dataset,
                out_dir,
                intervention_metric,
            )
    finally:
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Subspace analysis complete. Output in %s", base_out_dir)
    return result
