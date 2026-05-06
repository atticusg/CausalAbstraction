"""Locate analysis: identify which (layer, token_position) cell encodes a causal variable.

Scans a (layer × token_position) grid via interchange or DBM binary masking,
saves per-variable heatmaps, and reports the best cell per variable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.runner.helpers import (
    resolve_task,
    generate_datasets,
    get_output_token_ids,
    resolve_intervention_metric,
    _task_config_for_metadata,
)
from causalab.io.pipelines import load_pipeline
from causalab.io.plots.plot_utils import resolve_task_colormap
from causalab.io.plots.score_heatmap import plot_residual_stream_heatmap
from causalab.io.plots.figure_format import (
    path_with_figure_format,
    resolve_figure_format_from_analysis,
)

from causalab.methods.interchange.tracing import run_residual_stream_tracing
from causalab.neural.token_positions import get_list_of_each_token
from causalab.io.plots.string_heatmap import (
    build_token_labels,
    plot_single_pair_trace_heatmap,
)

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "locate"
HANDLES_MULTI_VARIABLE = True


def _resolve_target_variables(cfg: DictConfig) -> list[str | None]:
    """Read ``task.target_variables`` (plural) with fallback to ``task.target_variable``.

    Returns a non-empty list so the caller can always iterate. A ``[None]``
    result means "use whatever the task module exports by default".
    """
    plural = cfg.task.get("target_variables")
    if plural:
        return list(plural)
    singular = cfg.task.get("target_variable")
    return [singular]


def _run_scan_for_variable(
    cfg: DictConfig,
    pipeline,
    target_variable: str | None,
    var_out_dir: str,
    string_metric,
    comparison_fn,
    *,
    layers,
    token_positions,
    method: str,
    mode: str,
    batch_size: int,
    n_steer: int,
    figure_format: str = "pdf",
    source_pipeline=None,
    dbm_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the task for one variable and run the configured scan."""
    task, _task_cfg_raw = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=target_variable,
        seed=cfg.seed,
    )
    train_dataset, test_dataset = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
    )

    if layers is None:
        n_layers = pipeline.model.config.num_hidden_layers
        layers = list(range(n_layers))
    else:
        layers = list(layers)

    position_names = list(token_positions) if token_positions is not None else None

    score_token_ids, _n_score_tokens = get_output_token_ids(task, pipeline)
    n_classes = (
        len(task.intervention_values) if task.intervention_variable else _n_score_tokens
    )

    if method == "interchange":
        from causalab.analyses.locate import run_interchange_scan

        return run_interchange_scan(
            pipeline=pipeline,
            task=task,
            layers=layers,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            mode=mode,
            score_token_ids=score_token_ids,
            n_classes=n_classes,
            batch_size=batch_size,
            n_steer=n_steer,
            out_dir=var_out_dir,
            position_names=position_names,
            comparison_fn=comparison_fn,
            experiment_root=cfg.experiment_root,
            colormap=resolve_task_colormap(cfg.task, None),
            figure_format=figure_format,
            source_pipeline=source_pipeline,
        )
    elif method == "dbm_binary":
        from causalab.analyses.locate import run_dbm_binary_scan

        return run_dbm_binary_scan(
            pipeline=pipeline,
            task=task,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            layers=layers,
            batch_size=batch_size,
            out_dir=var_out_dir,
            metric=string_metric,
            position_names=position_names,
            dbm_cfg=dbm_cfg,
            figure_format=figure_format,
        )
    else:
        raise ValueError(f"Unknown locate method: {method}")


def _save_variable_results(
    result: dict[str, Any],
    var_out_dir: str,
    layers: list[int],
    variable_label: str,
    colormap: str | None,
    figure_format: str = "pdf",
) -> dict[str, Any]:
    """Persist per-variable results: JSON scores, heatmap figure."""
    scores_per_cell = result.get("scores_per_cell", {})
    scores_per_layer = result.get("scores_per_layer", {})
    token_position_ids = result.get("token_position_ids", [])

    best_cell = None
    if scores_per_cell:
        best_cell = max(scores_per_cell, key=lambda k: scores_per_cell[k])

    results_data = {
        "best_cell": (
            {"layer": best_cell[0], "token_position": best_cell[1]}
            if best_cell is not None
            else None
        ),
        "best_layer": best_cell[0] if best_cell is not None else None,
        "scores_per_cell": {
            f"{layer}|{pos_id}": score
            for (layer, pos_id), score in scores_per_cell.items()
        },
        "scores_per_layer": {str(k): v for k, v in scores_per_layer.items()},
        "base_accuracy": result.get("base_accuracy"),
        "token_position_ids": [str(p) for p in token_position_ids],
    }
    with open(os.path.join(var_out_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)

    if scores_per_cell:
        try:
            plot_residual_stream_heatmap(
                scores=scores_per_cell,
                layers=layers,
                token_position_ids=token_position_ids,
                title=f"Locate: {variable_label}",
                save_path=path_with_figure_format(
                    os.path.join(var_out_dir, "heatmap.pdf"),
                    figure_format,
                ),
                figure_format=figure_format,
            )
        except Exception as e:
            logger.warning("Heatmap render failed for %s: %s", variable_label, e)

    if best_cell is not None:
        logger.info(
            "%s: best cell = (layer=%s, position=%s), score=%.4f",
            variable_label,
            best_cell[0],
            best_cell[1],
            scores_per_cell[best_cell],
        )
    return results_data


def _run_single_pair_trace(
    cfg: DictConfig,
    pipeline,
    layers: list[int],
    out_dir: str,
    figure_format: str = "pdf",
    source_pipeline=None,
) -> None:
    """Run a single-pair residual stream trace and save the artifact.

    Picks one counterfactual pair from the first target variable's dataset,
    patches every (layer, token) cell, and saves the per-cell output tokens
    as ``single_pair_trace.json`` alongside a frequency-colored heatmap.
    """
    target_variable = _resolve_target_variables(cfg)[0]
    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=target_variable,
        seed=cfg.seed,
    )
    train_dataset, _ = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
    )

    example = train_dataset[0]
    prompt = example["input"]["raw_input"]
    cf_prompt = example["counterfactual_inputs"][0]["raw_input"]

    # Per-token positions for pedagogical granularity
    token_positions = get_list_of_each_token(prompt, pipeline)

    result = run_residual_stream_tracing(
        pipeline=pipeline,
        prompt=prompt,
        counterfactual_prompt=cf_prompt,
        token_positions=token_positions,
        layers=layers,
        verbose=False,
        source_pipeline=source_pipeline,
    )

    # Build self-contained JSON (no pipeline needed to plot later)
    token_labels = build_token_labels(pipeline, prompt, token_positions)
    token_position_ids = [tp.id for tp in token_positions]

    cells: dict[str, dict[str, str]] = {}
    for (layer, pos_id), res in result["intervention_results"].items():
        key = f"{layer}|{pos_id}"
        output_str = res["string"][0].strip() if res.get("string") else ""
        cells[key] = {"output": output_str}

    trace_data = {
        "prompt": prompt,
        "counterfactual_prompt": cf_prompt,
        "layers": layers,
        "token_position_ids": token_position_ids,
        "token_labels": token_labels,
        "cells": cells,
    }

    trace_path = os.path.join(out_dir, "single_pair_trace.json")
    with open(trace_path, "w") as f:
        json.dump(trace_data, f, indent=2)
    logger.info("Single-pair trace saved to %s", trace_path)

    plot_single_pair_trace_heatmap(
        trace_data=trace_data,
        title="Single-Pair Trace",
        save_path=os.path.join(out_dir, "single_pair_trace_heatmap"),
        figure_format=figure_format,
    )


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the locate analysis over a (layer × token_position) grid.

    Loops over ``task.target_variables`` (or ``task.target_variable`` fallback),
    nesting per-variable output under ``{analysis._output_dir}/{variable}/``.

    Cross-model patching is enabled by setting ``analysis.source_model`` to a
    model name (different from ``cfg.model.name``).  When set, activations are
    collected from the source model and patched into the target model.  When
    ``null`` (the default), standard single-model patching is used.
    """
    analysis = cfg[ANALYSIS_NAME]
    figure_fmt = resolve_figure_format_from_analysis(analysis)
    string_metric, comparison_fn = resolve_intervention_metric(
        cfg.task.intervention_metric
    )

    out_dir = analysis._output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Validate target variables before loading the pipeline so errors are caught early.
    target_variables = _resolve_target_variables(cfg)
    if None in target_variables:
        # Probe the task to list available variables. resolve_task handles
        # factory tasks (which require task_cfg); pass a placeholder
        # target_variable since we discard the loaded task immediately.
        _probe, _ = resolve_task(
            task_name=cfg.task.name,
            task_config=OmegaConf.to_container(cfg.task, resolve=True),
            target_variable="__probe__",
            seed=cfg.seed,
        )
        available = [
            v
            for v in _probe.causal_model.values
            if v not in ("raw_input", "raw_output")
        ]
        raise ValueError(
            f"task.target_variable (or task.target_variables) must be set. "
            f"Available variables for task '{cfg.task.name}': {available}"
        )

    # Load the (target) pipeline once and reuse across variables.
    placeholder_task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=target_variables[0],
        seed=cfg.seed,
    )
    pipeline = load_pipeline(
        model_name=cfg.model.name,
        task=placeholder_task,
        max_new_tokens=cfg.task.max_new_tokens,
        device=cfg.model.device,
        dtype=cfg.model.get("dtype"),
        eager_attn=cfg.model.get("eager_attn"),
    )

    # Optionally load a source pipeline for cross-model patching.
    source_model_name = analysis.get("source_model")
    source_pipeline = None
    if source_model_name:
        logger.info(
            "Cross-model locate: collecting activations from source_model=%s, "
            "patching into target_model=%s",
            source_model_name,
            cfg.model.name,
        )
        source_pipeline = load_pipeline(
            model_name=source_model_name,
            task=placeholder_task,
            max_new_tokens=cfg.task.max_new_tokens,
            device=cfg.model.device,
            dtype=cfg.model.get("dtype"),
            eager_attn=cfg.model.get("eager_attn"),
        )

    logger.info("Locate scan over variables: %s", target_variables)

    layers = analysis.layers
    if layers is None:
        layers = list(range(pipeline.model.config.num_hidden_layers))
    else:
        layers = list(layers)

    all_results: dict[str, Any] = {}
    for target_variable in target_variables:
        label = target_variable
        var_out_dir = os.path.join(out_dir, label)
        os.makedirs(var_out_dir, exist_ok=True)

        logger.info("=== Locate: target_variable=%s ===", label)
        result = _run_scan_for_variable(
            cfg=cfg,
            pipeline=pipeline,
            target_variable=target_variable,
            var_out_dir=var_out_dir,
            string_metric=string_metric,
            comparison_fn=comparison_fn,
            layers=analysis.layers,
            token_positions=analysis.get("token_positions"),
            method=analysis.method,
            mode=analysis.mode,
            batch_size=analysis.batch_size,
            n_steer=analysis.n_steer,
            figure_format=figure_fmt,
            source_pipeline=source_pipeline,
            dbm_cfg=OmegaConf.to_container(analysis.dbm, resolve=True)
            if analysis.get("dbm") is not None
            else None,
        )
        results_data = _save_variable_results(
            result=result,
            var_out_dir=var_out_dir,
            layers=layers,
            variable_label=label,
            colormap=resolve_task_colormap(cfg.task),
            figure_format=figure_fmt,
        )
        all_results[label] = results_data

    # Top-level results.json for back-compat with ``load_locate_result`` —
    # mirrors the first variable's best_layer / best_cell so downstream
    # analyses (subspace, activation_manifold) can resolve a default layer.
    if all_results:
        first_label = next(iter(all_results))
        top_results = dict(all_results[first_label])
        top_results["default_variable"] = first_label
        top_results["per_variable"] = {
            label: {
                "best_cell": data.get("best_cell"),
                "best_layer": data.get("best_layer"),
            }
            for label, data in all_results.items()
        }
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(top_results, f, indent=2)

    # Optional single-pair trace for pedagogical visualization
    if analysis.get("visualization", {}).get("single_pair_trace", True):
        try:
            _run_single_pair_trace(
                cfg, pipeline, layers, out_dir, figure_fmt, source_pipeline
            )
        except Exception as e:
            logger.warning("Single-pair trace failed: %s", e)

    metadata = {
        "analysis": "locate",
        "method": analysis.method,
        "mode": analysis.mode,
        "model": cfg.model.name,
        "source_model": source_model_name,
        "task": cfg.task.name,
        "task_config": _task_config_for_metadata(
            OmegaConf.to_container(cfg.task, resolve=True)
        ),
        "layers": layers,
        "token_positions": (
            list(analysis.token_positions)
            if analysis.get("token_positions") is not None
            else None
        ),
        "target_variables": [v for v in target_variables],
        "n_train": cfg.task.n_train,
        "n_test": cfg.task.n_test,
        "seed": cfg.seed,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    del pipeline
    if source_pipeline is not None:
        del source_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Locate analysis complete. Output in %s", out_dir)
    return {
        "output_dir": out_dir,
        "per_variable_results": all_results,
        "metadata": metadata,
    }
