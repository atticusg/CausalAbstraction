"""``logit_lens`` mode: logit lens over several inputs, all tokens x all layers.

For each prompt in ``cfg.exploration.logit_lens.inputs`` (a JSON list of prompt
strings) produces a (layer x token-position) top-1-token heatmap plus the
per-cell top-k token JSON, under ``out_dir`` — a thin caller of the shipped
``run_logit_lens_on_prompts`` entry point (which owns the ``input_NN/`` layout +
compute/save/plot). Mirrors the heatmap path of
``causalab/analyses/logit_lens/main.py`` but takes raw prompt strings instead of
a task.
"""

from __future__ import annotations

import json

from omegaconf import DictConfig

from causalab.analyses.logit_lens.prompts import run_logit_lens_on_prompts


def run(pipeline, acfg: DictConfig, out_dir: str) -> list[str]:
    with open(acfg.inputs) as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(
            "exploration.logit_lens.inputs must point to a non-empty JSON list "
            "of prompt strings"
        )

    out_dirs = run_logit_lens_on_prompts(
        pipeline,
        prompts,
        out_dir,
        top_k=int(acfg.top_k),
        batch_size=int(acfg.batch_size),
        figure_format=acfg.figure_format,
    )
    for i, (prompt, d) in enumerate(zip(prompts, out_dirs)):
        print(f"[logit_lens] input {i}: {prompt!r} -> {d}")
    return list(out_dirs)
