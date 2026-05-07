"""Attention pattern analysis: extract, visualize, and characterize attention heads.

Scans a configurable (layer x head) grid, computes per-head statistics
(entropy, max attention, self-attention, previous-token), optionally computes
average attention by semantic token type, and saves all visualizations and
numeric results.

All artifacts are saved to ``{experiment_root}/attention_pattern/``.
"""

from __future__ import annotations

import logging
import os
import random
from collections import Counter
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.io.artifacts import save_experiment_metadata, save_json_results
from causalab.io.pipelines import load_pipeline
from causalab.io.plots.attention_pattern import (
    plot_attention_heatmap,
    plot_attention_statistics,
    plot_layer_head_attention_grid,
    plot_token_type_attention_heatmap,
)
from causalab.io.plots.figure_format import resolve_figure_format_from_analysis
from causalab.methods.attention_pattern_analysis import (
    analyze_attention_statistics,
    compute_average_attention,
    compute_average_attention_by_token_type,
    get_attention_patterns,
)
from causalab.runner.helpers import _task_config_for_metadata, resolve_task
from causalab.tasks.loader import load_task_counterfactuals

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "attention_pattern"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_correct_examples(
    pipeline,
    task,
    n_target: int,
    max_attempts: int,
    seed: int,
) -> list[dict]:
    """Sample from the task and keep only examples the model predicts correctly."""
    cf_mod = load_task_counterfactuals(task.name)
    sampler = getattr(cf_mod, "sample_answerable_question", None) or getattr(
        cf_mod, "sample_input", None
    )
    if sampler is None:
        raise ValueError(
            f"Task {task.name!r} has no sample_answerable_question or sample_input"
        )

    rng_state = random.getstate()
    random.seed(seed)
    correct: list[dict] = []
    attempts = 0
    try:
        while len(correct) < n_target and attempts < max_attempts:
            attempts += 1
            example = sampler()
            pred = pipeline.generate([example])
            pred_str = pipeline.dump(pred["sequences"]).strip()
            expected = str(example["raw_output"]).strip()
            if expected in pred_str or pred_str in expected:
                correct.append(example)
    finally:
        random.setstate(rng_state)

    logger.info(
        "Generated %d correct examples from %d attempts", len(correct), attempts
    )
    return correct


def _decode_tokens(pipeline, prompt: dict) -> list[str]:
    """Decode non-pad token strings from a prompt for axis labels."""
    encoded = pipeline.load([prompt])
    token_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    tokens = []
    for tid, mask in zip(token_ids, attention_mask):
        if mask == 1:
            tok = pipeline.tokenizer.convert_ids_to_tokens([tid.item()])[0]
            tok = tok.replace("\u2581", "_").replace("\u0120", "_")
            tokens.append(tok)
    return tokens


def _resolve_layer_head_pairs(
    cfg_layers: list[int] | None,
    cfg_heads: list[int] | None,
    num_layers: int,
    num_heads: int,
) -> list[tuple[int, int]]:
    """Build the (layer, head) pairs to scan from config."""
    layers = list(range(num_layers)) if cfg_layers is None else list(cfg_layers)
    heads = list(range(num_heads)) if cfg_heads is None else list(cfg_heads)
    return [(l, h) for l in layers for h in heads]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run attention pattern analysis over a (layer x head) grid.

    For each scanned pair, extracts attention patterns from correctly-predicted
    examples, computes statistics, and saves heatmaps and numeric results.
    """
    analysis = cfg[ANALYSIS_NAME]
    figure_fmt = resolve_figure_format_from_analysis(analysis)
    out_dir = analysis._output_dir
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    # --- Resolve task and load pipeline ---
    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
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
    )

    # --- Generate correct examples ---
    examples = _generate_correct_examples(
        pipeline=pipeline,
        task=task,
        n_target=analysis.n_examples,
        max_attempts=analysis.max_attempts,
        seed=cfg.seed,
    )
    if not examples:
        raise RuntimeError(
            "No correct examples found; the model may be unable to solve this task."
        )

    # --- Resolve token positions (for token-type analysis) ---
    token_positions_lookup = task.create_token_positions(pipeline)
    source_positions = None
    target_positions = None
    if analysis.source_token_types and analysis.target_token_types:
        source_names = list(analysis.source_token_types)
        target_names = list(analysis.target_token_types)
        available = sorted(token_positions_lookup.keys())
        for name in source_names + target_names:
            if name not in token_positions_lookup:
                raise ValueError(
                    f"Token position {name!r} not found. Available: {available}"
                )
        source_positions = [token_positions_lookup[n] for n in source_names]
        target_positions = [token_positions_lookup[n] for n in target_names]

    # --- Decode tokens for visualization (from first example) ---
    tokens = _decode_tokens(pipeline, examples[0])
    pad_token = pipeline.tokenizer.pad_token

    # --- Resolve (layer, head) pairs ---
    num_layers = pipeline.model.config.num_hidden_layers
    num_heads = pipeline.model.config.num_attention_heads
    pairs = _resolve_layer_head_pairs(
        cfg_layers=list(analysis.layers) if analysis.layers is not None else None,
        cfg_heads=list(analysis.heads) if analysis.heads is not None else None,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    logger.info(
        "Scanning %d (layer, head) pairs across %d examples",
        len(pairs),
        len(examples),
    )

    # --- Main scan loop ---
    all_statistics: dict[str, dict] = {}
    all_token_type_results: dict[str, dict] = {}
    # Cache first-example results per pair for the comparison grid
    first_example_cache: dict[tuple[int, int], dict] = {}

    for layer, head in pairs:
        pair_key = f"L{layer}_H{head}"
        logger.info("Processing %s ...", pair_key)

        results = get_attention_patterns(
            pipeline=pipeline,
            layer=layer,
            head=head,
            prompts=examples,
            token_positions=None,
        )
        first_example_cache[(layer, head)] = results[0]

        # --- Per-head statistics ---
        stats = analyze_attention_statistics(results)
        all_statistics[pair_key] = stats

        # --- Per-head visualizations ---
        vis_cfg = analysis.visualization
        pair_vis_dir = os.path.join(vis_dir, pair_key)
        os.makedirs(pair_vis_dir, exist_ok=True)

        # 1. Single-example heatmap (first example)
        if vis_cfg.per_head_heatmaps:
            try:
                plot_attention_heatmap(
                    attention_pattern=results[0]["attention_pattern"],
                    title=f"Attention - Layer {layer}, Head {head}",
                    tokens=tokens,
                    save_path=os.path.join(pair_vis_dir, f"heatmap.{figure_fmt}"),
                    ignore_first_token=analysis.ignore_first_token,
                    pad_token=pad_token,
                    figure_format=figure_fmt,
                )
            except Exception as e:
                logger.warning("Heatmap failed for %s: %s", pair_key, e)

        # 2. Average attention heatmap (same-length examples only)
        if vis_cfg.average_heatmaps:
            try:
                lengths = [r["seq_len"] for r in results]
                length_counts = Counter(lengths)
                most_common_len, count = length_counts.most_common(1)[0]
                if count >= 2:
                    filtered = [r for r in results if r["seq_len"] == most_common_len]
                    avg_result = compute_average_attention(
                        filtered,
                        ignore_first_token=analysis.ignore_first_token,
                    )
                    avg_tokens = (
                        tokens[1:] if avg_result["ignored_first_token"] else tokens
                    )
                    plot_attention_heatmap(
                        attention_pattern=avg_result["average_pattern"],
                        title=f"Avg Attention - L{layer} H{head} (n={avg_result['num_samples']})",
                        tokens=avg_tokens,
                        save_path=os.path.join(
                            pair_vis_dir, f"avg_heatmap.{figure_fmt}"
                        ),
                        ignore_first_token=False,  # already handled
                        pad_token=pad_token,
                        figure_format=figure_fmt,
                    )
            except Exception as e:
                logger.warning("Average heatmap failed for %s: %s", pair_key, e)

        # 3. Statistics bar chart
        if vis_cfg.statistics_charts:
            try:
                plot_attention_statistics(
                    statistics=stats,
                    title=f"Statistics - L{layer} H{head}",
                    save_path=os.path.join(pair_vis_dir, f"statistics.{figure_fmt}"),
                    figure_format=figure_fmt,
                )
            except Exception as e:
                logger.warning("Statistics chart failed for %s: %s", pair_key, e)

        # 4. Token-type attention heatmap
        if vis_cfg.token_type_heatmaps and source_positions and target_positions:
            try:
                tt_result = compute_average_attention_by_token_type(
                    attention_results=results,
                    source_positions=source_positions,
                    target_positions=target_positions,
                )
                all_token_type_results[pair_key] = {
                    "attention_matrix": tt_result["attention_matrix"].tolist(),
                    "std_matrix": tt_result["std_matrix"].tolist(),
                    "source_ids": tt_result["source_ids"],
                    "target_ids": tt_result["target_ids"],
                    "num_samples": tt_result["num_samples"],
                }
                plot_token_type_attention_heatmap(
                    attention_matrix=tt_result["attention_matrix"],
                    source_ids=tt_result["source_ids"],
                    target_ids=tt_result["target_ids"],
                    title=f"Token-Type Attention - L{layer} H{head} (n={tt_result['num_samples']})",
                    save_path=os.path.join(
                        pair_vis_dir, f"token_type_heatmap.{figure_fmt}"
                    ),
                    std_matrix=tt_result["std_matrix"],
                    figure_format=figure_fmt,
                )
            except Exception as e:
                logger.warning("Token-type heatmap failed for %s: %s", pair_key, e)

    # --- Per-layer head comparison grids ---
    if analysis.visualization.head_comparison_grid:
        scanned_layers = sorted(set(l for l, _ in pairs))
        for layer in scanned_layers:
            layer_heads = [h for l, h in pairs if l == layer]
            if len(layer_heads) < 2:
                continue
            grid_results = [first_example_cache[(layer, h)] for h in layer_heads]
            try:
                plot_layer_head_attention_grid(
                    attention_results=grid_results,
                    title=f"Head Comparison - Layer {layer}",
                    tokens=tokens,
                    save_path=os.path.join(vis_dir, f"layer_{layer}_grid.{figure_fmt}"),
                    ignore_first_token=analysis.ignore_first_token,
                    pad_token=pad_token,
                    max_tokens=analysis.max_tokens_display,
                    figure_format=figure_fmt,
                )
            except Exception as e:
                logger.warning("Head comparison grid failed for layer %d: %s", layer, e)

    # --- Save numeric results ---
    save_json_results(
        {
            "statistics_per_head": all_statistics,
            "n_examples": len(examples),
            "pairs_scanned": [list(p) for p in pairs],
        },
        out_dir,
        "results.json",
    )
    if all_token_type_results:
        save_json_results(all_token_type_results, out_dir, "token_type_results.json")

    # --- Metadata ---
    metadata = {
        "analysis": "attention_pattern",
        "model": cfg.model.name,
        "task": cfg.task.name,
        "task_config": _task_config_for_metadata(
            OmegaConf.to_container(cfg.task, resolve=True)
        ),
        "n_examples": len(examples),
        "pairs_scanned": [[l, h] for l, h in pairs],
        "source_token_types": (
            list(analysis.source_token_types) if analysis.source_token_types else None
        ),
        "target_token_types": (
            list(analysis.target_token_types) if analysis.target_token_types else None
        ),
        "seed": cfg.seed,
    }
    save_experiment_metadata(metadata, out_dir)

    # --- Cleanup ---
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Attention pattern analysis complete. Output in %s", out_dir)
    return {"output_dir": out_dir, "metadata": metadata}
