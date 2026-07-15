"""Ablation analysis: zero/mean-ablate a component grid (attention heads, whole
attention-sublayer outputs, MLPs, or residual-stream positions), measure the
behavioral accuracy drop, and render a (layer × head) or (layer × position) drop
heatmap.

For each grid cell the component's output is replaced (zeros for zero-ablation,
the corpus-mean activation for mean-ablation) across the configured span, the
model generates, and accuracy is graded against the task's ``raw_output``. The
reported drop is ``base_accuracy − ablated_accuracy``. Explicit unit *combos* can
be ablated jointly. Ablation is behavioral (one accuracy per cell), not per
causal-variable, so unlike ``locate`` there is no ``HANDLES_MULTI_VARIABLE``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.analyses.ablation.spans import resolve_span
from causalab.io.pipelines import load_pipeline
from causalab.io.plots.figure_format import (
    path_with_figure_format,
    resolve_figure_format_from_analysis,
)
from causalab.io.plots.score_heatmap import (
    plot_attention_head_heatmap,
    plot_residual_stream_heatmap,
)
from causalab.methods.ablation import (
    make_mean_vectors,
    make_zero_vectors,
    run_ablation_combo,
    run_ablation_scan,
)
from causalab.methods.metric import compute_base_accuracy, make_causal_metric
from causalab.neural.activations.targets import (
    build_attention_head_targets,
    build_attention_output_targets,
    build_mlp_targets,
    build_residual_stream_targets,
)
from causalab.neural.units import InterchangeTarget
from causalab.runner.helpers import (
    generate_datasets,
    resolve_task,
    _task_config_for_metadata,  # pyright: ignore[reportPrivateUsage]
)

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "ablation"

VALID_COMPONENT_TYPES = ("attention_head", "attention_output", "mlp", "residual")
VALID_MODES = ("zero", "mean")


def _resolve_target_variable(cfg: DictConfig) -> str:
    """Single target variable for the (behavioral) ablation grade."""
    variable = cfg.task.get("target_variable")
    if variable is None:
        plural = cfg.task.get("target_variables")
        variable = plural[0] if plural else None
    if variable is None:
        raise ValueError(
            "task.target_variable (or task.target_variables[0]) must be set for "
            "the ablation analysis."
        )
    return variable


def _build_grid_targets(
    pipeline, component_type: str, layers: list[int], heads: list[int], span
) -> dict[tuple[Any, ...], InterchangeTarget]:
    """One single-unit target per grid cell, dispatched on component type.

    ``attention_head`` is keyed ``(layer, head)``; ``attention_output``, ``mlp``
    and ``residual`` are keyed ``(layer, span.id)`` (a layer × span grid).
    """
    if component_type == "attention_head":
        return build_attention_head_targets(pipeline, layers, heads, span)
    if component_type == "attention_output":
        return build_attention_output_targets(pipeline, layers, [span])
    if component_type == "residual":
        return build_residual_stream_targets(pipeline, layers, [span])
    return build_mlp_targets(pipeline, layers, [span])


def _combo_units(pipeline, component_type: str, combo: list[Any], span) -> list[Any]:
    """Materialize the units for one combo entry.

    ``attention_head`` combos are ``[[layer, head], ...]``; ``attention_output`` /
    ``mlp`` / ``residual`` combos are ``[layer, ...]``. Units are built via the
    same target builders (one cell at a time) so they carry identical
    featurizers/shapes to the grid units.
    """
    units: list[Any] = []
    if component_type == "attention_head":
        for layer, head in combo:
            target = build_attention_head_targets(pipeline, [layer], [head], span)
            units.extend(target[(layer, head)].flatten())
    else:
        for layer in combo:
            target = _build_grid_targets(pipeline, component_type, [layer], [], span)
            units.extend(target[(layer, span.id)].flatten())
    return units


def _build_reference_vectors(
    mode: str,
    pipeline,
    dataset,
    units: list[Any],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Zero or corpus-mean reference vector per unit id, built once for all units.

    Units are grouped one-per-group into a combined target so a single
    ``make_*`` call covers the whole grid (plus any combo units).
    """
    combined_target = InterchangeTarget([[u] for u in units])
    if mode == "zero":
        return make_zero_vectors(combined_target)
    return make_mean_vectors(pipeline, dataset, combined_target, batch_size=batch_size)


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the ablation analysis over a component grid and optional combos."""
    analysis = cfg[ANALYSIS_NAME]
    figure_fmt = resolve_figure_format_from_analysis(analysis)

    component_type = analysis.component_type
    if component_type not in VALID_COMPONENT_TYPES:
        raise ValueError(
            f"ablation.component_type must be one of {VALID_COMPONENT_TYPES}, "
            f"got {component_type!r}"
        )
    mode = analysis.mode
    if mode not in VALID_MODES:
        raise ValueError(f"ablation.mode must be one of {VALID_MODES}, got {mode!r}")

    # `_subdir` (component_type_mode) does not encode the span, so two runs
    # differing only in `span` would share an output dir and overwrite each other.
    # `span` is "all" or a list of position names; a list has no clean Hydra
    # interpolation form (it stringifies to "['last_token']"), so fold a sanitized
    # span token into the output dir here, mirroring the runtime span id ("all" or
    # "name1+name2") and keeping the flat one-segment layout the siblings use.
    span_cfg = analysis.span
    span_token = (
        span_cfg if isinstance(span_cfg, str) else "+".join(str(s) for s in span_cfg)
    )
    span_token = re.sub(r"[^0-9A-Za-z._+-]+", "_", span_token) or "span"
    out_dir = f"{analysis._output_dir}_{span_token}"
    os.makedirs(out_dir, exist_ok=True)

    # --- task, pipeline, data -------------------------------------------------
    target_variable = _resolve_target_variable(cfg)
    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=cast(
            dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True)
        ),
        target_variable=target_variable,
        seed=cfg.seed,
    )
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
    train_dataset, test_dataset = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
    )

    base = compute_base_accuracy(
        test_dataset, pipeline, task.checker, batch_size=analysis.batch_size
    )
    base_accuracy = base["accuracy"]
    logger.info("Ablation base accuracy: %.4f", base_accuracy)

    # --- grid + span ----------------------------------------------------------
    config = pipeline.model.config
    layers = (
        list(analysis.layers)
        if analysis.get("layers") is not None
        else list(range(config.num_hidden_layers))
    )
    heads = (
        list(analysis.heads)
        if analysis.get("heads") is not None
        else list(range(config.num_attention_heads))
    )
    span = resolve_span(analysis.span, task, pipeline, test_dataset)

    targets = _build_grid_targets(pipeline, component_type, layers, heads, span)

    # --- combos (optional) ----------------------------------------------------
    raw_combos = analysis.get("combos")
    combos: list[Any] = (
        list(cast(list, OmegaConf.to_container(raw_combos, resolve=True)))
        if raw_combos is not None
        else []
    )
    combo_unit_sets = [
        _combo_units(pipeline, component_type, combo, span) for combo in combos
    ]

    # --- reference vectors (one build covering grid + combo units) -----------
    grid_units = [u for target in targets.values() for u in target.flatten()]
    all_units = {u.id: u for u in grid_units}
    for units in combo_unit_sets:
        for u in units:
            all_units[u.id] = u
    vectors = _build_reference_vectors(
        mode, pipeline, train_dataset, list(all_units.values()), analysis.batch_size
    )

    # --- scan -----------------------------------------------------------------
    metric = make_causal_metric(task.checker)  # task checker vs raw_output label (#167)
    ablated_accuracy = run_ablation_scan(
        targets,
        test_dataset,
        pipeline,
        vectors,
        metric=metric,
        batch_size=analysis.batch_size,
        causal_model=task.causal_model,
    )
    drop_grid = {key: base_accuracy - acc for key, acc in ablated_accuracy.items()}

    # --- combos ---------------------------------------------------------------
    combo_results = []
    for combo, units in zip(combos, combo_unit_sets):
        combo_acc = run_ablation_combo(
            units,
            test_dataset,
            pipeline,
            vectors,
            metric=metric,
            batch_size=analysis.batch_size,
            causal_model=task.causal_model,
        )
        combo_results.append(
            {
                "units": combo,
                "ablated_accuracy": combo_acc,
                "drop": base_accuracy - combo_acc,
            }
        )

    # --- complement / keep-span (optional, sufficiency) -----------------------
    # Ablate every grid layer *except* the kept span, jointly: how much behavior
    # survives on just those layers. The grid is keyed (layer, head) / (layer,
    # pos_id), so key[0] is the layer.
    raw_keep = analysis.get("complement_keep")
    complement_result: dict[str, Any] | None = None
    if raw_keep is not None:
        keep_set = {int(k) for k in cast(list, OmegaConf.to_container(raw_keep))}
        complement_units = [
            u
            for key, target in targets.items()
            if int(key[0]) not in keep_set
            for u in target.flatten()
        ]
        if not complement_units:
            logger.warning(
                "complement_keep=%s leaves no grid layers to ablate; skipping.",
                sorted(keep_set),
            )
        else:
            comp_acc = run_ablation_combo(
                complement_units,
                test_dataset,
                pipeline,
                vectors,
                metric=metric,
                batch_size=analysis.batch_size,
                causal_model=task.causal_model,
            )
            complement_result = {
                "kept_layers": sorted(keep_set),
                "ablated_layers": sorted(
                    {int(key[0]) for key in targets if int(key[0]) not in keep_set}
                ),
                "ablated_accuracy": comp_acc,
                "drop": base_accuracy - comp_acc,
            }

    # --- save -----------------------------------------------------------------
    _save_results(
        out_dir=out_dir,
        component_type=component_type,
        mode=mode,
        base_accuracy=base_accuracy,
        drop_grid=drop_grid,
        combo_results=combo_results,
        top_k=analysis.get("top_k", 20),
        layers=layers,
        heads=heads,
        span_id=span.id,
        figure_format=figure_fmt,
        complement_result=complement_result,
    )
    metadata = {
        "analysis": ANALYSIS_NAME,
        "component_type": component_type,
        "mode": mode,
        # span may be a string ("all") or a ListConfig of names — normalize to
        # a JSON-safe value.
        "span": (
            OmegaConf.to_container(analysis.span, resolve=True)
            if OmegaConf.is_config(analysis.span)
            else analysis.span
        ),
        "span_id": span.id,
        "model": cfg.model.name,
        "task": cfg.task.name,
        "task_config": _task_config_for_metadata(
            cast(dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True))
        ),
        "layers": layers,
        "heads": heads if component_type == "attention_head" else None,
        "complement_keep": (
            complement_result["kept_layers"] if complement_result else None
        ),
        "target_variable": target_variable,
        "n_train": cfg.task.n_train,
        "n_test": cfg.task.n_test,
        "seed": cfg.seed,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Ablation analysis complete. Output in %s", out_dir)
    return {
        "output_dir": out_dir,
        "base_accuracy": base_accuracy,
        "drop_grid": {f"{k[0]}|{k[1]}": v for k, v in drop_grid.items()},
        "combos": combo_results,
        "complement": complement_result,
        "metadata": metadata,
    }


def _save_results(
    *,
    out_dir: str,
    component_type: str,
    mode: str,
    base_accuracy: float,
    drop_grid: dict[tuple[Any, ...], float],
    combo_results: list[dict[str, Any]],
    top_k: int,
    layers: list[int],
    heads: list[int],
    span_id: Any,
    figure_format: str,
    complement_result: dict[str, Any] | None = None,
) -> None:
    """Persist results.json and the drop heatmap."""
    # Keys are (layer, head) for attention heads or (layer, pos_id) for MLP.
    top_cells = sorted(drop_grid.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    results_data = {
        "component_type": component_type,
        "mode": mode,
        "base_accuracy": base_accuracy,
        "drop_grid": {f"{k[0]}|{k[1]}": v for k, v in drop_grid.items()},
        "top_k_cells": [{"cell": f"{k[0]}|{k[1]}", "drop": v} for k, v in top_cells],
        "combos": combo_results,
        "complement": complement_result,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)

    # We plot the accuracy *drop*, a signed quantity: ablation usually hurts
    # accuracy (positive drop) but can also help it (negative drop). The shipped
    # create_heatmap defaults to a sequential viridis map clamped to [0, 1], so a
    # negative drop would clamp to the floor color — visually indistinguishable
    # from a true zero drop, hiding the sign. When any cell is negative, switch to
    # a diverging map with symmetric bounds centered at 0 so improvements read as
    # a distinct color; otherwise keep the [0, 1] sequential scale. The colorbar
    # is labelled "Accuracy drop" rather than the generic "Score".
    title = f"Ablation accuracy drop ({mode}, {component_type})"
    finite_drops = [
        v for v in drop_grid.values() if v is not None and not math.isnan(v)
    ]
    if finite_drops and min(finite_drops) < 0.0:
        bound = max(abs(min(finite_drops)), abs(max(finite_drops)))
        cmap, vmin, vmax = "coolwarm", -bound, bound
    else:
        cmap, vmin, vmax = "viridis", 0.0, 1.0
    # path_with_figure_format swaps the extension to match figure_format; the
    # literal here is just the stem placeholder.
    save_path = path_with_figure_format(
        os.path.join(out_dir, "heatmap.png"), figure_format
    )
    try:
        if component_type == "attention_head":
            plot_attention_head_heatmap(
                scores=drop_grid,
                layers=layers,
                heads=heads,
                title=title,
                save_path=save_path,
                figure_format=figure_format,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_label="Accuracy drop",
            )
        else:
            plot_residual_stream_heatmap(
                scores=drop_grid,
                layers=layers,
                token_position_ids=[span_id],
                title=title,
                save_path=save_path,
                figure_format=figure_format,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_label="Accuracy drop",
            )
    except Exception as e:  # noqa: BLE001 — a render failure shouldn't lose results
        logger.warning("Ablation heatmap render failed: %s", e)
