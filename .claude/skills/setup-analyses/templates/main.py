"""{{ANALYSIS_NAME}}: {{RESEARCH_QUESTION}}

{{LONGER_PURPOSE_PARAGRAPH_FROM_SPEC}}

This is a session-local *analysis* (research-question wrapper) — see ARCHITECTURE.md §3.
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


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the {{ANALYSIS_NAME}} analysis.

    All artifacts are saved under
    ``cfg.experiment_root/{{ANALYSIS_NAME}}/{cfg.{{ANALYSIS_NAME}}._subdir}/``.
    """
    analysis = cfg[ANALYSIS_NAME]
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
    pipeline = load_pipeline(
        model_name=cfg.model.name,
        max_new_tokens=cfg.task.get("max_new_tokens", 1),
        device=cfg.get("device", "cuda"),
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
    # save_json_results(results, out_dir, "results.json")
    # save_tensor_results(<tensor>, out_dir, "<name>.safetensors")
    # save_experiment_metadata(out_dir, cfg)
    # return results
