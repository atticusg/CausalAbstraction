"""Exploratory evidence about what algorithm a model runs on a task.

exploration answers: *what does the model do at the tokens that matter?* — a
task-less analysis that runs one of four exploratory **modes** over raw,
hand-authored inputs (not a runner-generated task dataset), selected by
``cfg.exploration.mode``:

* ``probe``      — greedy-decode a batch of prompts; report each output.
* ``logit_lens`` — logit lens (all tokens x all layers) over several prompts.
* ``pair``       — one base/counterfactual interchange trace (one manifest row).
* ``pca``        — centered PCA of the residual stream at each critical token.
* ``knockout``   — zero/mean-ablate attention heads + MLP layer-bands; report the
                   behavioral drop per head and per MLP band width.

Each mode reads its inputs from ``cfg.exploration.<mode>.*`` and the model from
``cfg.model``, writing under ``cfg.exploration._output_dir`` (``.../exploration/<mode>``).
The ``pair`` and ``pca`` modes fan out one unit per task (per pair / per token)
via a Hydra override on ``exploration.pair.index`` / ``exploration.pca.tokens``.
"""

from __future__ import annotations

import os
from typing import Any

from omegaconf import DictConfig

from causalab.analyses.exploration._pipeline import build_pipeline

ANALYSIS_NAME = "exploration"
_MODES = ("probe", "logit_lens", "pair", "pca", "knockout")


def _resolve_output_dir(analysis_cfg: DictConfig) -> str:
    out_dir = analysis_cfg.get("_output_dir") or os.path.join(
        os.getcwd(), ANALYSIS_NAME
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _build_pipeline(cfg: DictConfig, max_new_tokens: int):
    """Build a plain LMPipeline from ``cfg.model`` (task-less; no generation
    contract beyond greedy decode)."""
    model = cfg.model
    return build_pipeline(
        model.name,
        max_new_tokens=max_new_tokens,
        device=model.get("device"),
        dtype=model.get("dtype"),
    )


def main(cfg: DictConfig) -> Any:
    acfg = cfg[ANALYSIS_NAME]
    mode = acfg.mode
    if mode not in _MODES:
        raise ValueError(f"exploration.mode must be one of {_MODES}; got {mode!r}")
    out_dir = _resolve_output_dir(acfg)

    if mode == "probe":
        from causalab.analyses.exploration.probe_prompts import run as run_mode

        pipeline = _build_pipeline(cfg, int(acfg.probe.get("max_new_tokens", 3)))
        return run_mode(pipeline, acfg.probe, out_dir)

    if mode == "logit_lens":
        from causalab.analyses.exploration.logit_lens_inputs import run as run_mode

        pipeline = _build_pipeline(cfg, 3)
        return run_mode(pipeline, acfg.logit_lens, out_dir)

    if mode == "pair":
        from causalab.analyses.exploration.pair_trace import run as run_mode

        pipeline = _build_pipeline(cfg, int(acfg.pair.get("max_new_tokens", 3)))
        return run_mode(pipeline, acfg.pair, out_dir)

    if mode == "pca":
        # forward-only collection; max_new_tokens is immaterial.
        from causalab.analyses.exploration.pca_critical_tokens import run as run_mode

        pipeline = _build_pipeline(cfg, 1)
        return run_mode(pipeline, acfg.pca, out_dir)

    # mode == "knockout" — ablate components, grade the generated output vs. base.
    from causalab.analyses.exploration.knockout_components import run as run_mode

    pipeline = _build_pipeline(cfg, int(acfg.knockout.get("max_new_tokens", 3)))
    return run_mode(pipeline, acfg.knockout, out_dir)
