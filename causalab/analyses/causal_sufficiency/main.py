"""Causal sufficiency analysis: ROME-style causal tracing.

Corrupt the residual stream where the information enters (the embedding layer
over a configured span — ``zero``, ``mean`` or seeded ``noise`` scaled to ``3σ``
of the subject embeddings, ROME-style), establishing a broken-behaviour floor,
then **restore one clean site at a time** (optionally a centered ``window`` of
consecutive layers, ROME's severed traces) over a grid of attention heads /
attention-sublayer outputs / MLPs / residual-stream positions and measure how
much of the behaviour recovers. The reported quantity per cell is
``recovery = restored_metric − corrupted_floor`` (optionally normalised to the
``clean_ceiling − floor`` band), rendered as a (layer × head) or (layer ×
position) recovery heatmap.

Corruption and restoration are applied together in a single forward pass (see
:mod:`causalab.methods.causal_tracing.run`): ``zero``/``mean`` corruption and the
restore are ``replace`` interventions; ``noise`` corruption uses the dynamic
per-token noise intervention (so it spans the whole multi-token subject) mixed
with the ``replace`` restore. The recovery metric is the softmax probability of
the answer token (``prob``), the logit difference between a correct and a
distractor token (``logit_diff``), or a single answer token's logit (``logit``).
Tracing is behavioural (one score per cell), not per causal-variable, so there is
no ``HANDLES_MULTI_VARIABLE``.
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
from causalab.methods.causal_tracing import (
    VALID_CORRUPTIONS,
    collect_clean_vectors,
    corruption_intervention_type,
    make_corruption_vectors,
    run_causal_trace_scan,
    run_corrupted_floor,
)
from causalab.methods.metric import (
    InterchangeMetric,
    compute_base_outputs,
    make_logit_diff_metric,
    make_logit_metric,
    make_prob_metric,
    score_intervention_outputs,
)
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

ANALYSIS_NAME = "causal_sufficiency"

VALID_COMPONENT_TYPES = ("attention_head", "attention_output", "mlp", "residual")
VALID_METRICS = ("logit_diff", "logit", "prob")


def _build_grid_targets(
    pipeline, component_type: str, layers: list[int], heads: list[int], span
) -> dict[tuple[Any, ...], InterchangeTarget]:
    """One single-unit restore target per grid cell, dispatched on component type.

    ``attention_head`` is keyed ``(layer, head)``; ``attention_output``, ``mlp``
    and ``residual`` are keyed ``(layer, span.id)`` (a layer × span grid). Mirrors
    the ablation grid so the heatmap keys line up.
    """
    if component_type == "attention_head":
        return build_attention_head_targets(pipeline, layers, heads, span)
    if component_type == "attention_output":
        return build_attention_output_targets(pipeline, layers, [span])
    if component_type == "residual":
        return build_residual_stream_targets(pipeline, layers, [span])
    return build_mlp_targets(pipeline, layers, [span])


def _window_layers(center: int, window: int, n_layers: int) -> list[int]:
    """The ``window`` consecutive layers centered on ``center``, clamped to range.

    ``window=1`` returns ``[center]`` (a single state); ``window=10`` returns the
    ROME severed-trace band ``[center-4 … center+5]`` (clamped at the ends).
    """
    half = (window - 1) // 2
    lo = max(0, center - half)
    hi = min(n_layers - 1, center + (window - 1 - half))
    return list(range(lo, hi + 1))


def _windowed_swept_targets(
    pipeline,
    component_type: str,
    layers: list[int],
    heads: list[int],
    span,
    window: int,
    n_layers: int,
) -> dict[tuple[Any, ...], InterchangeTarget]:
    """Restore grid where each cell restores a window of layers at one site.

    Keyed by *center* layer (``(center, head)`` or ``(center, span.id)``) so the
    heatmap axes are unchanged; the cell's target collects the units across the
    windowed layers, which the scan restores jointly. ``window=1`` reduces to the
    single-unit grid of :func:`_build_grid_targets`.
    """
    targets: dict[tuple[Any, ...], InterchangeTarget] = {}
    for center in layers:
        w_layers = _window_layers(center, window, n_layers)
        if component_type == "attention_head":
            for head in heads:
                sub = build_attention_head_targets(pipeline, w_layers, [head], span)
                units = [u for t in sub.values() for u in t.flatten()]
                targets[(center, head)] = InterchangeTarget([units])
        else:
            sub = _build_grid_targets(pipeline, component_type, w_layers, [], span)
            units = [u for t in sub.values() for u in t.flatten()]
            targets[(center, span.id)] = InterchangeTarget([units])
    return targets


def _build_metric(metric_cfg: DictConfig, pipeline, dataset) -> InterchangeMetric:
    """Logit-difference (correct vs distractor) or single-token logit metric.

    Built with ``relative_to_base=False`` so each cell's score is the *raw*
    patched logit(-difference); recovery (relative to the corrupted floor) is
    computed by the analysis, not the metric.
    """
    kind = metric_cfg.kind

    def _require(name: str) -> str:
        value = metric_cfg.get(name)
        if value is None:
            raise ValueError(
                f"metric.kind={kind!r} requires metric.{name} to be set to a task "
                "variable (the answer/contrast token to read); it is null."
            )
        return value

    if kind == "logit_diff":
        correct = _require("correct_variable")
        distractor = _require("distractor_variable")
        return make_logit_diff_metric(
            pipeline,
            dataset,
            correct_of=lambda ex: ex["input"][correct],
            distractor_of=lambda ex: ex["input"][distractor],
            relative_to_base=False,
        )
    if kind in ("logit", "prob"):
        answer = _require("answer_variable")
        builder = make_logit_metric if kind == "logit" else make_prob_metric
        return builder(
            pipeline,
            dataset,
            answer_of=lambda ex: ex["input"][answer],
            relative_to_base=False,
        )
    raise ValueError(f"metric.kind must be one of {VALID_METRICS}, got {kind!r}")


def _score_clean_ceiling(
    base_outputs: list[dict[str, Any]],
    dataset,
    metric: InterchangeMetric,
    causal_model,
) -> float:
    """Score the un-intervened (clean) run under ``metric`` — the recovery ceiling.

    Wraps ``compute_base_outputs`` into the single-synthetic-batch shape
    ``score_intervention_outputs`` expects (``string``/``scores`` nested one level
    by "batch"), so the clean run is scored by exactly the same metric as each
    intervened cell.
    """
    n = len(base_outputs)
    strings = [o["string"] for o in base_outputs]
    raw: dict[str, Any] = {"string": [strings]}
    if base_outputs and base_outputs[0].get("scores"):
        n_tokens = len(base_outputs[0]["scores"])
        raw["scores"] = [
            [
                torch.stack([base_outputs[i]["scores"][t] for i in range(n)], dim=0)
                for t in range(n_tokens)
            ]
        ]
    return score_intervention_outputs(
        raw_results={("clean",): raw},
        dataset=dataset,
        metric=metric,
        causal_model=causal_model,
        original_outputs=None,
    )[("clean",)]


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the causal-sufficiency (causal-tracing) analysis over a restore grid."""
    analysis = cfg[ANALYSIS_NAME]
    figure_fmt = resolve_figure_format_from_analysis(analysis)

    corruption_cfg = analysis.corruption
    restore_cfg = analysis.restore
    kind = corruption_cfg.kind
    if kind not in VALID_CORRUPTIONS:
        raise ValueError(
            f"corruption.kind must be one of {VALID_CORRUPTIONS}, got {kind!r}"
        )
    component_type = restore_cfg.component_type
    if component_type not in VALID_COMPONENT_TYPES:
        raise ValueError(
            f"restore.component_type must be one of {VALID_COMPONENT_TYPES}, "
            f"got {component_type!r}"
        )
    normalize = analysis.get("normalize", "recovery")
    if normalize not in ("recovery", "fraction"):
        raise ValueError(
            f"normalize must be 'recovery' or 'fraction', got {normalize!r}"
        )

    # Fold the restore span into the output dir so runs differing only in span
    # don't collide (a list span has no clean Hydra-interpolation form); mirrors
    # the ablation analysis's span folding.
    span_cfg = restore_cfg.span
    span_token = (
        span_cfg if isinstance(span_cfg, str) else "+".join(str(s) for s in span_cfg)
    )
    span_token = re.sub(r"[^0-9A-Za-z._+-]+", "_", span_token) or "span"
    out_dir = f"{analysis._output_dir}_{span_token}"
    os.makedirs(out_dir, exist_ok=True)

    # --- task, pipeline, data -------------------------------------------------
    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=cast(
            dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True)
        ),
        target_variable=cfg.task.get("target_variable"),
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

    config = pipeline.model.config
    layers = (
        list(restore_cfg.layers)
        if restore_cfg.get("layers") is not None
        else list(range(config.num_hidden_layers))
    )
    heads = (
        list(restore_cfg.heads)
        if restore_cfg.get("heads") is not None
        else list(range(config.num_attention_heads))
    )

    # --- corruption: entry site + reference vectors ---------------------------
    corruption_span = resolve_span(corruption_cfg.span, task, pipeline, test_dataset)
    corruption_layer = corruption_cfg.get("layer", -1)
    entry_target = build_residual_stream_targets(
        pipeline, [corruption_layer], [corruption_span]
    )[(corruption_layer, corruption_span.id)]
    entry_units = entry_target.flatten()
    entry_type = corruption_intervention_type(kind)
    noise_seed = corruption_cfg.get("noise_seed", 0)
    corruption_vectors = make_corruption_vectors(
        kind,
        pipeline,
        train_dataset,
        entry_target,
        noise_scale=corruption_cfg.get("noise_scale", 3.0),
        noise_seed=noise_seed,
        batch_size=analysis.batch_size,
    )

    # --- restore grid + clean (restoration) vectors ---------------------------
    # `window` restores a centered band of consecutive layers at each site (ROME
    # uses ~10 for the MLP/attention severed traces); window=1 restores one state.
    window = restore_cfg.get("window", 1)
    restore_span = resolve_span(restore_cfg.span, task, pipeline, test_dataset)
    swept_targets = _windowed_swept_targets(
        pipeline,
        component_type,
        layers,
        heads,
        restore_span,
        window,
        config.num_hidden_layers,
    )
    # Union of every (windowed) restored unit — one collection pass covers them all.
    swept_units = list(
        {u.id: u for t in swept_targets.values() for u in t.flatten()}.values()
    )
    clean_vectors = collect_clean_vectors(
        pipeline, test_dataset, swept_units, batch_size=analysis.batch_size
    )

    # --- metric + baselines ---------------------------------------------------
    metric = _build_metric(analysis.metric, pipeline, test_dataset)
    base_outputs = compute_base_outputs(
        test_dataset, pipeline, batch_size=analysis.batch_size
    )
    clean_ceiling = _score_clean_ceiling(
        base_outputs, test_dataset, metric, task.causal_model
    )
    corrupted_floor = run_corrupted_floor(
        pipeline,
        test_dataset,
        entry_units,
        corruption_vectors,
        metric=metric,
        entry_type=entry_type,
        noise_seed=noise_seed,
        batch_size=analysis.batch_size,
        causal_model=task.causal_model,
        original_outputs=base_outputs if metric.needs_original_output else None,
    )
    logger.info(
        "Causal tracing: clean ceiling=%.4f, corrupted floor=%.4f",
        clean_ceiling,
        corrupted_floor,
    )

    # --- restore sweep --------------------------------------------------------
    restored = run_causal_trace_scan(
        swept_targets,
        test_dataset,
        pipeline,
        entry_units=entry_units,
        corruption_vectors=corruption_vectors,
        clean_vectors=clean_vectors,
        metric=metric,
        entry_type=entry_type,
        noise_seed=noise_seed,
        batch_size=analysis.batch_size,
        causal_model=task.causal_model,
        original_outputs=base_outputs if metric.needs_original_output else None,
    )

    band = clean_ceiling - corrupted_floor
    recovery_grid: dict[tuple[Any, ...], float] = {}
    for key, value in restored.items():
        raw_recovery = value - corrupted_floor
        if normalize == "fraction":
            recovery_grid[key] = raw_recovery / band if band != 0 else 0.0
        else:
            recovery_grid[key] = raw_recovery

    # --- save -----------------------------------------------------------------
    _save_results(
        out_dir=out_dir,
        kind=kind,
        component_type=component_type,
        normalize=normalize,
        clean_ceiling=clean_ceiling,
        corrupted_floor=corrupted_floor,
        recovery_grid=recovery_grid,
        top_k=analysis.get("top_k", 20),
        layers=layers,
        heads=heads,
        span_id=restore_span.id,
        figure_format=figure_fmt,
    )
    metadata = {
        "analysis": ANALYSIS_NAME,
        "corruption_kind": kind,
        "corruption_layer": corruption_layer,
        "corruption_span": (
            OmegaConf.to_container(corruption_cfg.span, resolve=True)
            if OmegaConf.is_config(corruption_cfg.span)
            else corruption_cfg.span
        ),
        "noise_scale": corruption_cfg.get("noise_scale", 3.0)
        if kind == "noise"
        else None,
        "noise_seed": noise_seed if kind == "noise" else None,
        "restore_component_type": component_type,
        "restore_window": window,
        "restore_span": (
            OmegaConf.to_container(restore_cfg.span, resolve=True)
            if OmegaConf.is_config(restore_cfg.span)
            else restore_cfg.span
        ),
        "restore_span_id": restore_span.id,
        "normalize": normalize,
        "metric": OmegaConf.to_container(analysis.metric, resolve=True),
        "clean_ceiling": clean_ceiling,
        "corrupted_floor": corrupted_floor,
        "model": cfg.model.name,
        "task": cfg.task.name,
        "task_config": _task_config_for_metadata(
            cast(dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True))
        ),
        "layers": layers,
        "heads": heads if component_type == "attention_head" else None,
        "n_train": cfg.task.n_train,
        "n_test": cfg.task.n_test,
        "seed": cfg.seed,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Causal-sufficiency analysis complete. Output in %s", out_dir)
    return {
        "output_dir": out_dir,
        "clean_ceiling": clean_ceiling,
        "corrupted_floor": corrupted_floor,
        "recovery_grid": {f"{k[0]}|{k[1]}": v for k, v in recovery_grid.items()},
        "metadata": metadata,
    }


def _save_results(
    *,
    out_dir: str,
    kind: str,
    component_type: str,
    normalize: str,
    clean_ceiling: float,
    corrupted_floor: float,
    recovery_grid: dict[tuple[Any, ...], float],
    top_k: int,
    layers: list[int],
    heads: list[int],
    span_id: Any,
    figure_format: str,
) -> None:
    """Persist results.json and the recovery heatmap."""
    top_cells = sorted(recovery_grid.items(), key=lambda kv: kv[1], reverse=True)[
        :top_k
    ]
    results_data = {
        "corruption_kind": kind,
        "component_type": component_type,
        "normalize": normalize,
        "clean_ceiling": clean_ceiling,
        "corrupted_floor": corrupted_floor,
        "recovery_grid": {f"{k[0]}|{k[1]}": v for k, v in recovery_grid.items()},
        "top_k_cells": [
            {"cell": f"{k[0]}|{k[1]}", "recovery": v} for k, v in top_cells
        ],
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)

    # Recovery is signed: a mediating site recovers behaviour (positive), but a
    # site can also push past the clean ceiling or below the floor. When any cell
    # is negative use a diverging map centred at 0; otherwise the sequential
    # [0, max] scale. Fraction-normalised recovery sits in ~[0, 1].
    title = f"Causal tracing recovery ({kind}, {component_type})"
    finite = [v for v in recovery_grid.values() if v is not None and not math.isnan(v)]
    if finite and min(finite) < 0.0:
        bound = max(abs(min(finite)), abs(max(finite)))
        cmap, vmin, vmax = "coolwarm", -bound, bound
    elif normalize == "fraction":
        cmap, vmin, vmax = "viridis", 0.0, 1.0
    else:
        cmap, vmin, vmax = "viridis", 0.0, (max(finite) if finite else 1.0)

    save_path = path_with_figure_format(
        os.path.join(out_dir, "heatmap.png"), figure_format
    )
    try:
        if component_type == "attention_head":
            plot_attention_head_heatmap(
                scores=recovery_grid,
                layers=layers,
                heads=heads,
                title=title,
                save_path=save_path,
                figure_format=figure_format,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_label="Recovery",
            )
        else:
            plot_residual_stream_heatmap(
                scores=recovery_grid,
                layers=layers,
                token_position_ids=[span_id],
                title=title,
                save_path=save_path,
                figure_format=figure_format,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_label="Recovery",
            )
    except Exception as e:  # noqa: BLE001 — a render failure shouldn't lose results
        logger.warning("Causal-tracing heatmap render failed: %s", e)
