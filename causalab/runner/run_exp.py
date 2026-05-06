"""Unified experiment entry point.

Each runner config pulls one or more analysis defaults at the top level via
``- analysis/<name>`` entries in its defaults list.  Each analysis YAML
declares ``# @package <name>``, so its body is mounted at ``cfg.<name>``.
Execution order follows the order of those entries in the defaults list,
recovered at runtime via OmegaConf insertion order.

Usage::

    # Single-step runner
    uv run python -m causalab.runner.run_exp --config-name baseline_demo

    # Multi-step pipeline
    uv run python -m causalab.runner.run_exp --config-name he_pipeline

    # Introspect
    uv run python -m causalab.runner.run_exp --config-name baseline_demo --cfg job
"""

from __future__ import annotations

import gc
import importlib
import logging
import os
from collections.abc import Iterator

import sys as _sys
import matplotlib as _matplotlib

if "matplotlib.pyplot" not in _sys.modules:
    _matplotlib.use("Agg")
del _sys, _matplotlib

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

logger = logging.getLogger(__name__)


def _load_analysis(analysis_name: str):
    """Dynamically load the analysis module.

    Tries the shipped ``causalab.analyses.<name>.main`` namespace first; on
    ``ImportError`` falls back to a session-local ``analyses.<name>.main`` —
    importable when ``${SESSION_DIR}/code/`` is on ``PYTHONPATH`` (set by
    ``scripts/run_exp.sh`` whenever ``--experiment-root`` lives under
    ``agent_logs/``). See ``.claude/skills/research-session/CONVENTIONS.md``.

    Returns the module so callers can check for module-level flags
    (e.g. ``HANDLES_MULTI_VARIABLE``).  The entry point is ``mod.main``.
    """
    try:
        return importlib.import_module(f"causalab.analyses.{analysis_name}.main")
    except ImportError:
        # Session-local fallback only when running under a research session (avoids PYTHONPATH shadowing).
        if os.environ.get("CAUSALAB_SESSION_CODE"):
            try:
                return importlib.import_module(f"analyses.{analysis_name}.main")
            except ImportError as exc:
                raise ImportError(
                    f"Could not import analysis {analysis_name!r}. Tried "
                    f"causalab.analyses.{analysis_name}.main and "
                    f"analyses.{analysis_name}.main. For session-local analyses, "
                    f"ensure ${{SESSION_DIR}}/code/ is on PYTHONPATH (scripts/run_exp.sh "
                    f"sets this automatically when --experiment-root lives under agent_logs/)."
                ) from exc
        raise


def _resolve_target_variables(cfg: DictConfig) -> list[str | None]:
    """Return the list of target variables, or [None] for module default."""
    target_variables = cfg.task.get("target_variables", None)
    if target_variables:
        return list(target_variables)
    singular = cfg.task.get("target_variable", None)
    if singular:
        return [singular]
    return [None]


def _run_analysis_for_variables(
    cfg: DictConfig,
    analysis_fn,
    analysis_mod,
    analysis_name: str,
    target_variables: list[str | None],
    base_root: str,
) -> None:
    """Run a single analysis across target variables.

    Analyses that set ``HANDLES_MULTI_VARIABLE = True`` handle their own
    variable loop (e.g. locate) and are called once without iteration.
    """
    if getattr(analysis_mod, "HANDLES_MULTI_VARIABLE", False):
        analysis_fn(cfg)
        return

    for tv in target_variables:
        OmegaConf.update(cfg, "task.target_variable", tv, force_add=True)

        logger.info(
            "Running analysis: %s | target_variable: %s",
            analysis_name,
            tv or "(module default)",
        )
        logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

        analysis_fn(cfg)


def _release_gpu_memory() -> None:
    """Collect unreferenced objects and flush PyTorch's GPU memory cache.

    Analyses load a fresh pipeline per step and let it go out of scope on
    return. Python's GC may not collect it immediately, leaving the weights
    on the GPU when the next step tries to load. Forcing collection here
    ensures the memory is actually free before the next analysis starts.
    """
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _iter_analysis_steps(cfg: DictConfig) -> Iterator[tuple[str, DictConfig]]:
    """Yield (step_name, step_cfg) in defaults-list order.

    A top-level cfg key is treated as an analysis step iff its value is a
    DictConfig containing ``_name_``. Order follows OmegaConf insertion order,
    which mirrors the runner's defaults list.
    """
    for key in cfg:
        value = cfg[key]
        if isinstance(value, DictConfig) and "_name_" in value:
            yield key, value


def _run_steps(cfg: DictConfig) -> None:
    """Iterate analysis steps in defaults-list order, then run post-steps."""
    target_variables = _resolve_target_variables(cfg)
    base_root = cfg.experiment_root

    steps = list(_iter_analysis_steps(cfg))
    if not steps:
        raise ValueError(
            "No analysis steps found. Add `- analysis/<name>` entries to the "
            "runner's defaults list."
        )

    last_step_cfg: DictConfig | None = None
    for step_name, step_cfg in steps:
        analysis_name = step_cfg._name_

        logger.info("=== Step: %s (%s) ===", step_name, analysis_name)

        analysis_mod = _load_analysis(analysis_name)
        mod_name = getattr(analysis_mod, "ANALYSIS_NAME", None)
        if mod_name != analysis_name:
            raise RuntimeError(
                f"Module {analysis_mod.__name__} declares ANALYSIS_NAME="
                f"{mod_name!r} but cfg slice has _name_={analysis_name!r}"
            )
        _run_analysis_for_variables(
            cfg,
            analysis_mod.main,
            analysis_mod,
            analysis_name,
            target_variables,
            base_root,
        )
        _release_gpu_memory()
        last_step_cfg = step_cfg

    # Post-pipeline visualization steps
    if cfg.get("post"):
        from causalab.runner.post_steps import run_post_steps

        figure_format = "pdf"
        if last_step_cfg is not None and "visualization" in last_step_cfg:
            figure_format = last_step_cfg.visualization.get("figure_format", "pdf")

        run_post_steps(
            list(cfg.post),
            [v for v in target_variables if v is not None],
            base_root,
            figure_format,
        )


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    # Insert task variant into experiment_root when set
    from causalab.io.configs import apply_experiment_root_variant

    apply_experiment_root_variant(cfg)

    # --- Dispatch ---
    _run_steps(cfg)


if __name__ == "__main__":
    main()
