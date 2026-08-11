"""Single-pair residual-stream trace → self-contained artifact (analysis layer).

Owns the artifact-directory layout for the ``locate`` analysis's single-pair
trace: runs ``run_single_pair_trace`` (the method), assembles a self-contained
``single_pair_trace.json`` (re-plottable without a pipeline), and renders the
frequency-colored heatmap. Both the ``locate`` analysis (``main.py``) and the
``explore-behavior`` skill template call this so the layout lives in one
place rather than being re-implemented per caller (docs/CODEBASE.md invariant 4).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from causalab.io.plots.string_heatmap import (
    build_token_labels,
    plot_single_pair_trace_heatmap,
)
from causalab.methods.interchange.single_pair import run_single_pair_trace
from causalab.neural.pipeline import GenerationResult, LMPipeline
from causalab.neural.token_positions import TokenPosition

logger = logging.getLogger(__name__)


def _cell_output(res: GenerationResult) -> str:
    """One intervention cell's decoded output, stripped.

    Each cell is a single-example run, so its flat
    :class:`~causalab.neural.pipeline.GenerationResult` carries exactly one
    string (EU5b, #487 — the legacy nested-list normalization is gone).
    """
    return res.strings[0].strip() if res.strings else ""


def save_single_pair_trace(
    pipeline: LMPipeline,
    prompt: str,
    counterfactual_prompt: str,
    token_positions: list[TokenPosition],
    layers: list[int] | None,
    out_dir: str,
    figure_format: str = "png",
    *,
    title: str = "Single-Pair Trace",
    extra_fields: dict[str, Any] | None = None,
    source_pipeline: LMPipeline | None = None,
) -> dict[str, Any]:
    """Trace one base/CF pair across every (layer, token) cell and save it.

    Runs ``run_single_pair_trace`` then writes a self-contained
    ``single_pair_trace.json`` (no pipeline needed to re-plot) plus a
    frequency-colored heatmap under ``out_dir``. ``extra_fields`` are merged
    into the JSON (e.g. ``token`` / ``input_idx`` / clean outputs supplied by a
    manifest-driven caller). Returns the assembled ``trace_data`` dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    result = run_single_pair_trace(
        pipeline=pipeline,
        prompt=prompt,
        counterfactual_prompt=counterfactual_prompt,
        token_positions=token_positions,
        layers=layers,
        verbose=False,
        source_pipeline=source_pipeline,
    )

    # Positions the counterfactual can't supply are dropped inside
    # run_single_pair_trace (#176); use the effective list it returns so the
    # heatmap axis matches the traced cells exactly.
    effective_positions = result["token_positions"]
    token_labels = build_token_labels(pipeline, prompt, effective_positions)
    cells: dict[str, dict[str, str]] = {}
    for (layer, pos_id), res in result["intervention_results"].items():
        cells[f"{layer}|{pos_id}"] = {"output": _cell_output(res)}

    trace_data: dict[str, Any] = {
        "prompt": prompt,
        "counterfactual_prompt": counterfactual_prompt,
        "layers": result["metadata"]["layers_used"],
        "token_position_ids": [tp.id for tp in effective_positions],
        "token_labels": token_labels,
        "cells": cells,
    }
    if extra_fields:
        trace_data.update(extra_fields)

    trace_path = os.path.join(out_dir, "single_pair_trace.json")
    with open(trace_path, "w") as f:
        json.dump(trace_data, f, indent=2)
    logger.info("Single-pair trace saved to %s", trace_path)

    plot_single_pair_trace_heatmap(
        trace_data=trace_data,
        title=title,
        save_path=os.path.join(out_dir, "single_pair_trace_heatmap"),
        figure_format=figure_format,
    )
    return trace_data
