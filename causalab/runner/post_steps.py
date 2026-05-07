"""Post-pipeline visualization steps.

Runs cross-analysis visualizations after the analysis chain completes.
Each entry in the ``post:`` config list dispatches to a handler that loads
results from prior analysis steps and generates aggregate plots.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from causalab.io.plots.figure_format import path_with_figure_format

logger = logging.getLogger(__name__)

_VIZ_TYPES: dict[str, Any] = {}


def _register(name: str):
    def decorator(fn):
        _VIZ_TYPES[name] = fn
        return fn

    return decorator


def run_post_steps(
    post_configs: list,
    target_variables: list[str],
    experiment_root: str,
    figure_format: str = "pdf",
) -> dict[str, Any]:
    """Run post-pipeline visualization steps.

    Parameters
    ----------
    post_configs:
        List of post-step dicts, each with at least a ``type`` key.
    target_variables:
        Variable names used to locate per-variable result directories.
    experiment_root:
        Root artifact directory for the experiment.
    figure_format:
        Output figure format (``"pdf"`` or ``"png"``).
    """
    out_dir = os.path.join(experiment_root, "post")
    os.makedirs(out_dir, exist_ok=True)

    results: dict[str, Any] = {}
    for viz_cfg in post_configs:
        viz_type = viz_cfg["type"] if isinstance(viz_cfg, dict) else viz_cfg.type
        if viz_type not in _VIZ_TYPES:
            raise ValueError(
                f"Unknown post-step type: {viz_type!r}. Available: {sorted(_VIZ_TYPES)}"
            )
        results[viz_type] = _VIZ_TYPES[viz_type](
            viz_cfg,
            target_variables,
            experiment_root,
            out_dir,
            figure_format,
        )

    logger.info("Post-pipeline steps complete. Output in %s", out_dir)
    return results


@_register("variable_localization_heatmap")
def _run_variable_localization_heatmap(
    viz_cfg,
    target_variables: list[str],
    experiment_root: str,
    out_dir: str,
    figure_format: str,
) -> dict[str, Any]:
    """Load locate results for each variable and plot combined heatmap."""
    from causalab.io.plots.score_heatmap import (
        plot_variable_localization_heatmap,
    )

    source_step = _cfg_get(viz_cfg, "source_step", "locate")
    source_method = _cfg_get(viz_cfg, "source_method", "interchange")

    scores_by_variable: dict[str, dict[str, float]] = {}
    var_best_position: dict[str, str | None] = {}
    reference_token_position_ids: list[str] = []
    for var in target_variables:
        results_path = os.path.join(
            experiment_root, source_step, source_method, var, "results.json"
        )
        if not os.path.exists(results_path):
            logger.warning("Missing results for variable %s: %s", var, results_path)
            continue
        with open(results_path) as f:
            data = json.load(f)
        scores_by_variable[var] = data["scores_per_layer"]
        best_cell = data.get("best_cell")
        var_best_position[var] = best_cell["token_position"] if best_cell else None
        if not reference_token_position_ids and data.get("token_position_ids"):
            reference_token_position_ids = data["token_position_ids"]

    # Sort variables by their best-cell token position (left-to-right in input)
    def _pos_sort_key(var: str) -> int:
        pos = var_best_position.get(var)
        if pos is None or pos not in reference_token_position_ids:
            return len(reference_token_position_ids)
        return reference_token_position_ids.index(pos)

    sorted_vars = sorted(scores_by_variable, key=_pos_sort_key)
    scores_by_variable = {v: scores_by_variable[v] for v in sorted_vars}

    if not scores_by_variable:
        raise FileNotFoundError(
            f"No locate results found under {experiment_root}/{source_step}/{source_method}/"
        )

    title = _cfg_get(viz_cfg, "title", None) or "Variable Localization Heatmap"
    save_path = path_with_figure_format(
        os.path.join(out_dir, "variable_localization_heatmap.png"),
        figure_format,
    )

    plot_variable_localization_heatmap(
        scores_by_variable=scores_by_variable,
        title=title,
        save_path=save_path,
        figure_format=figure_format,
    )

    logger.info("Saved variable localization heatmap to %s", save_path)
    return {"save_path": save_path, "variables": list(scores_by_variable.keys())}


def _cfg_get(cfg, key: str, default=None):
    """Get a value from a dict or DictConfig."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return cfg.get(key, default)
