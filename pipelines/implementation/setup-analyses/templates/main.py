"""{{ANALYSIS_NAME}}: {{RESEARCH_QUESTION}}

{{LONGER_PURPOSE_PARAGRAPH_FROM_SPEC}}

This is a session-local *analysis* (research-question wrapper) — see docs/CODEBASE.md §3.
Layering rules respected by this module:
  - depends on causalab/{neural,methods,io,causal,tasks,runner.helpers}, never on
    causalab/analyses/ (peer modules) or causalab/runner/run_exp internals
  - all disk I/O routes through causalab.io.* primitives (invariant 3)
  - no hyperparameter defaults inline — every knob comes from `cfg.{{ANALYSIS_NAME}}.<knob>`
    or `cfg.task.<knob>` per invariants 5 and 11
  - `cfg.experiment_root` is the single source of truth for output paths (invariant 7)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from omegaconf import DictConfig, OmegaConf

from causalab.io.pipelines import load_pipeline
from causalab.runner.helpers import (
    generate_datasets,
    resolve_task,
)

# {{IMPORTS_FROM_SPEC_SECTION_3}}
# Each callable listed in set_up_analysis.md §3 — replace this comment with concrete imports.
# Examples:
#   from causalab.methods.metric import compute_reference_distributions
#   from methods.{{METHOD_NAME}} import {{METHOD_NAME}}     # session-local

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "{{ANALYSIS_NAME}}"


def _locate_analysis_cfg(cfg: DictConfig) -> DictConfig:
    """Return this analysis's config slice from the full run config.

    The runner hands ``main`` the *whole* cfg and dispatches each step by the
    ``_name_`` sentinel inside its slice, not by the top-level key (see
    ``causalab.runner.run_exp._iter_analysis_steps``). By default ``analysis.yaml``
    declares ``# @package {{ANALYSIS_NAME}}``, so the slice sits at
    ``cfg.{{ANALYSIS_NAME}}`` and the fast path resolves it directly. If you mount
    the config under a *different* package (``# @package <other>`` — the
    CONFIG_KEY != ANALYSIS_NAME case, e.g. a slice at ``cfg.subspace`` whose
    ``_name_`` is still ``{{ANALYSIS_NAME}}``), the slice is located by its
    ``_name_`` exactly as the runner does — so ``cfg[ANALYSIS_NAME]`` is never
    assumed.
    """
    fast = cfg.get(ANALYSIS_NAME)
    if isinstance(fast, DictConfig) and fast.get("_name_") == ANALYSIS_NAME:
        return fast
    for key in cfg:
        value = cfg[key]
        if isinstance(value, DictConfig) and value.get("_name_") == ANALYSIS_NAME:
            return value
    raise KeyError(
        f"No config slice with _name_={ANALYSIS_NAME!r} found in cfg. Ensure "
        f"analysis.yaml sets `_name_: {ANALYSIS_NAME}` and is on the runner's "
        f"defaults list."
    )


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the {{ANALYSIS_NAME}} analysis.

    All artifacts are saved under
    ``cfg.experiment_root/{{ANALYSIS_NAME}}/{cfg.{{ANALYSIS_NAME}}._subdir}/``.
    """
    analysis = _locate_analysis_cfg(cfg)
    out_dir = analysis._output_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- Load task ---
    task, _task_cfg = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )

    # --- Build datasets (sizes/balance/enumeration come from cfg.task per invariant 12) ---
    train_dataset, test_dataset = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        balanced=cfg.task.get("balanced", False),
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
    )

    # --- Load LM ---
    # `task` is the 2nd positional of load_pipeline (it runs task.validate on the
    # pipeline); `device` lives at cfg.model.device (default "auto", resolved
    # downstream), not at a top-level cfg.device.
    pipeline = load_pipeline(
        model_name=cfg.model.name,
        task=task,
        max_new_tokens=cfg.task.get("max_new_tokens", 1),
        device=cfg.model.device,
    )

    # --- Run methods listed in set_up_analysis.md §3 ---
    # TODO: implement. Each step below is a placeholder taken from the spec.
    # Reach into cfg.{{ANALYSIS_NAME}}.<knob> for every hyperparameter — never hardcode.
    #
    #   results = {{METHOD_NAME}}(
    #       activations=...,
    #       layer=analysis.layer,
    #       head=analysis.head,
    #   )
    raise NotImplementedError(
        "{{ANALYSIS_NAME}} not yet implemented. "
        "See set_up_analysis.md alongside this file for the spec."
    )

    # --- Persist outputs (every analysis writes metadata + named result files) ---
    # Import the writers you use from causalab.io.artifacts. Mind the arg order:
    # the payload comes FIRST, the output dir SECOND.
    #   save_json_results(results, out_dir, "results.json")
    #   save_tensor_results({"<name>": <tensor>}, out_dir, "<name>.safetensors")
    #   metadata = {
    #       "analysis": ANALYSIS_NAME,
    #       "model": cfg.model.name,
    #       "task": cfg.task.name,
    #       "seed": cfg.seed,
    #   }
    #   save_experiment_metadata(metadata, out_dir)   # dict FIRST, then out_dir
    # return results
