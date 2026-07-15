"""Prompt-driven logit-lens views (raw strings instead of a task dataset).

The shipped analysis (``main.py``) reads its inputs from a task; the
``explore-behavior`` skill needs the same (layer × token) top-1 heatmap and
per-cell top-k JSON for ad-hoc prompt strings. Both share the per-prompt core
here so the ``input_{i:02d}/`` artifact layout and the top-1 cell-wrapping live
in the analysis layer only, not duplicated in a skill template
(docs/CODEBASE.md invariant 4).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from causalab.io.plots.string_heatmap import (
    plot_residual_stream_intervention_heatmap,
)
from causalab.methods.logit_lens import compute_logit_lens, save_logit_lens_results
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import get_list_of_each_token


def render_logit_lens_heatmap(
    pipeline: LMPipeline,
    prompt: str,
    *,
    layers: List[int],
    top_k: int,
    batch_size: int,
    apply_final_norm: bool,
    out_dir: str,
    figure_format: str,
    save_results: bool = False,
) -> Dict[str, Any]:
    """Compute + render a (layer × token) top-1-token logit-lens heatmap.

    ``layers`` are the transformer layers to show; layer -1 (embedding output)
    is always prepended so the grid reads from the bottom of the stack upward.
    When ``save_results`` is set, the per-cell top-k tokens are also persisted
    via ``save_logit_lens_results`` (the standard method save helper). Returns
    the ``compute_logit_lens`` result dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    token_positions = get_list_of_each_token(prompt, pipeline)
    heatmap_layers = [-1] + list(layers)
    example = {"input": {"raw_input": prompt}, "counterfactual_inputs": []}

    result = compute_logit_lens(
        [example],
        pipeline,
        layers=heatmap_layers,
        token_positions=token_positions,
        top_k=top_k,
        batch_size=batch_size,
        apply_final_norm=apply_final_norm,
    )
    if save_results:
        save_logit_lens_results(result, out_dir)

    # Reuse the shared string-heatmap renderer by wrapping each cell's top-1
    # token in the ``{"string": [tok]}`` shape it already understands.
    intervention_results = {
        key: {"string": [payload["tokens"][0][0]]}
        for key, payload in result["top_k_by_unit"].items()
    }
    plot_residual_stream_intervention_heatmap(
        intervention_results=intervention_results,
        prompt=prompt,
        layers=heatmap_layers,
        token_positions=token_positions,
        pipeline=pipeline,
        title="Logit Lens: top-1 token by layer and position",
        save_path=os.path.join(out_dir, "logit_lens_heatmap"),
        show_scores=False,
        color_by_frequency=True,
        figure_format=figure_format,
    )
    return result


def run_logit_lens_on_prompts(
    pipeline: LMPipeline,
    prompts: List[str],
    out_root: str,
    *,
    top_k: int = 10,
    batch_size: int = 16,
    apply_final_norm: bool = True,
    figure_format: str = "png",
) -> List[str]:
    """Run the prompt-driven logit lens over several prompts under ``out_root``.

    Each prompt gets its own ``input_{i:02d}/`` directory containing the
    per-cell top-k JSON (``save_logit_lens_results``) plus a top-1 heatmap.
    Returns the list of per-prompt output directories.
    """
    n_layers = pipeline.model.config.num_hidden_layers
    layers = list(range(n_layers))
    out_dirs: List[str] = []
    for i, prompt in enumerate(prompts):
        out_dir = os.path.join(out_root, f"input_{i:02d}")
        render_logit_lens_heatmap(
            pipeline,
            prompt,
            layers=layers,
            top_k=top_k,
            batch_size=batch_size,
            apply_final_norm=apply_final_norm,
            out_dir=out_dir,
            figure_format=figure_format,
            save_results=True,
        )
        out_dirs.append(out_dir)
    return out_dirs
