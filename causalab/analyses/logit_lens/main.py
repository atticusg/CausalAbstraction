"""Logit lens analysis: read the model's per-layer next-token predictions.

Runs the logit lens (``causalab/methods/logit_lens.py``) over a task's test set
and produces two views:

1. **Dataset-level** — for every residual-stream layer at the *last* token,
   the top-k predicted tokens per example plus, when the task exposes answer
   tokens, the probability mass on those answer tokens by layer (the standard
   "at which depth does the answer emerge?" curve).
2. **Single-example heatmap** — a (layer × token-position) grid of the top-1
   predicted token for one representative prompt, rendered with the shared
   ``string_heatmap`` renderer (no logit-lens-specific plot code).

This analysis owns the research question, dataset loading, and artifact layout;
the projection math lives in the method.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.analyses.logit_lens.prompts import render_logit_lens_heatmap
from causalab.io.pipelines import load_pipeline
from causalab.io.plots.figure_format import resolve_figure_format_from_analysis
from causalab.methods.logit_lens import compute_logit_lens, save_logit_lens_results
from causalab.neural.token_positions import (
    TokenPosition,
    get_last_token_index,
)
from causalab.runner.helpers import (
    prepare_datasets,
    get_output_token_ids,
    resolve_task,
    _task_config_for_metadata,  # pyright: ignore[reportPrivateUsage]
)

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "logit_lens"


def _flatten_answer_token_ids(score_token_ids: Any) -> list[int] | None:
    """Flatten ``get_output_token_ids`` output (list[list[int]]) to a flat list.

    Returns None when the task exposes no answer tokens, which disables the
    optional answer-mass track.
    """
    if not score_token_ids:
        return None
    flat: list[int] = []
    for group in score_token_ids:
        if isinstance(group, int):
            flat.append(group)
        else:
            flat.extend(int(t) for t in group)
    return flat or None


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the logit lens analysis for one target variable."""
    analysis = cfg[ANALYSIS_NAME]
    figure_fmt = resolve_figure_format_from_analysis(analysis)

    out_dir = analysis._output_dir
    os.makedirs(out_dir, exist_ok=True)

    target_variable = (
        cfg.task.get("target_variable")
        or (cfg.task.get("target_variables") or [None])[0]
    )
    if target_variable is None:
        raise ValueError(
            "logit_lens requires task.target_variable (or task.target_variables). "
            "It selects the answer tokens used for the answer-mass track."
        )

    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=cast(
            dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True)
        ),
        target_variable=target_variable,
        seed=cfg.seed,
    )
    _train_dataset, test_dataset = prepare_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
        filter_correct=False,
    )
    if not test_dataset:
        raise ValueError("logit_lens: test dataset is empty; nothing to analyze.")

    pipeline = load_pipeline(
        model_name=cfg.model.name,
        task=task,
        max_new_tokens=cfg.task.max_new_tokens,
        device=cfg.model.device,
        dtype=cfg.model.get("dtype"),
        eager_attn=cfg.model.get("eager_attn"),
    )

    n_layers = pipeline.model.config.num_hidden_layers
    layers = (
        list(analysis.layers)
        if analysis.get("layers") is not None
        else list(range(n_layers))
    )
    top_k = analysis.top_k
    batch_size = analysis.batch_size
    apply_final_norm = analysis.apply_final_norm

    score_token_ids, _ = get_output_token_ids(task, pipeline)
    answer_token_ids = _flatten_answer_token_ids(score_token_ids)

    # ---- 1. Dataset-level lens at the last token across all layers ----------
    last_pos = TokenPosition(
        indexer=functools.partial(get_last_token_index, pipeline=pipeline),
        pipeline=pipeline,
        id="last",
    )
    result = compute_logit_lens(
        test_dataset,
        pipeline,
        layers=layers,
        token_positions=[last_pos],
        top_k=top_k,
        batch_size=batch_size,
        target_token_ids=answer_token_ids,
        apply_final_norm=apply_final_norm,
    )
    paths = save_logit_lens_results(result, out_dir)
    logger.info("Logit lens dataset-level results saved under %s", out_dir)

    # ---- 2. Single-example (layer × token) heatmap --------------------------
    if analysis.get("visualization", {}).get("heatmap", True):
        try:
            _render_example_heatmap(
                pipeline=pipeline,
                example=test_dataset[0],
                layers=layers,
                top_k=top_k,
                apply_final_norm=apply_final_norm,
                batch_size=batch_size,
                out_dir=out_dir,
                figure_format=figure_fmt,
            )
        except Exception as exc:  # visualization is best-effort
            logger.warning("Logit lens heatmap render failed: %s", exc)

    metadata = {
        "analysis": ANALYSIS_NAME,
        "model": cfg.model.name,
        "task": cfg.task.name,
        "task_config": _task_config_for_metadata(
            cast(dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True))
        ),
        "target_variable": target_variable,
        "layers": layers,
        "top_k": top_k,
        "apply_final_norm": apply_final_norm,
        "has_answer_track": answer_token_ids is not None,
        "n_test": cfg.task.n_test,
        "seed": cfg.seed,
    }
    # Run-level metadata is analysis-owned; write it to a distinct file so it
    # never clobbers the method's intrinsic ``metadata.json`` (written by
    # ``save_logit_lens_results``). See docs/CODEBASE.md invariant 4.
    analysis_metadata_path = os.path.join(out_dir, "analysis_metadata.json")
    with open(analysis_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    paths["analysis_metadata_path"] = analysis_metadata_path

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Logit lens analysis complete. Output in %s", out_dir)
    return {"output_dir": out_dir, "artifact_paths": paths, "metadata": metadata}


def _render_example_heatmap(
    pipeline: Any,
    example: Any,
    layers: list[int],
    top_k: int,
    apply_final_norm: bool,
    batch_size: int,
    out_dir: str,
    figure_format: str,
) -> None:
    """Render a (layer × token) top-1-token heatmap for a single prompt.

    Prepends layer -1 (embedding output) so the grid shows predictions from the
    very bottom of the stack upward. Reuses the shared string-heatmap renderer
    by wrapping each cell's top-1 token in the ``{"string": [tok]}`` shape it
    already understands.
    """
    render_logit_lens_heatmap(
        pipeline,
        example["input"]["raw_input"],
        layers=list(layers),
        top_k=top_k,
        batch_size=batch_size,
        apply_final_norm=apply_final_norm,
        out_dir=out_dir,
        figure_format=figure_format,
        save_results=False,
    )
    logger.info("Logit lens heatmap saved under %s", out_dir)
