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
import shutil
from collections.abc import Iterator

import sys as _sys
import matplotlib as _matplotlib

if "matplotlib.pyplot" not in _sys.modules:
    _matplotlib.use("Agg")
del _sys, _matplotlib

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

logger = logging.getLogger(__name__)

# The only leading-underscore directive keys the runner recognizes in an
# analysis step slice: ``_name_`` (dispatch) plus ``_subdir``/``_output_dir``
# (output routing). Any other ``_..._`` key is a config-author mistake — see
# sub-issue C2 of #171.
_RESERVED_DIRECTIVES = frozenset({"_name_", "_subdir", "_output_dir"})


def _load_analysis(analysis_name: str):
    """Dynamically load the analysis module.

    Tries the shipped ``causalab.analyses.<name>.main`` namespace first; on
    ``ImportError`` falls back to a session-local ``analyses.<name>.main`` when
    ``CAUSALAB_SESSION_CODE`` is set (the shipped namespace always wins). See
    ``causalab/runner/README.md`` "Session-local code injection".

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
                    f"analyses.{analysis_name}.main. For session-local analyses, see "
                    f"causalab/runner/README.md 'Session-local code injection'."
                ) from exc
        raise


def _resolve_target_variables(cfg: DictConfig) -> list[str | None]:
    """Return the list of target variables, or [None] for module default.

    Task-less analyses (e.g. ``characterize_subspace``) carry no ``cfg.task``;
    the struct-safe ``cfg.get`` avoids a ``ConfigAttributeError`` and falls back
    to the module default.
    """
    task = cfg.get("task")
    if task is None:
        return [None]
    target_variables = task.get("target_variables", None)
    if target_variables:
        return list(target_variables)
    singular = task.get("target_variable", None)
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
            # OmegaConf returns keys as DictKeyType (Enum | str); top-level keys
            # are always strings at runtime.
            yield str(key), value


def _check_known_directives(step_name: str, step_cfg: DictConfig) -> None:
    """Reject unrecognized ``_..._`` directive keys in an analysis step slice.

    The runner honors exactly three leading-underscore directives — ``_name_``
    (dispatch) plus ``_subdir``/``_output_dir`` (output routing). A hand-added
    key such as ``_runner_target_`` is otherwise silently ignored, so the
    author's intent is lost without any signal (#171, sub-issue C2). Fail fast
    instead, mirroring the ``_name_``-mismatch ``RuntimeError`` in ``_run_steps``.
    """
    unknown = sorted(
        str(key)
        for key in step_cfg
        if str(key).startswith("_") and str(key) not in _RESERVED_DIRECTIVES
    )
    if unknown:
        raise ValueError(
            f"Step {step_name!r} has unrecognized directive key(s) {unknown}. "
            f"The runner only honors {sorted(_RESERVED_DIRECTIVES)}; remove or "
            f"rename these keys (dispatch is by the bare `_name_:` field)."
        )


def _step_output_dir(step_cfg: DictConfig, analysis_name: str, base_root: str) -> str:
    """Best-effort resolution of the directory a step writes its artifacts to.

    Analyses that declare ``_output_dir`` in their config slice own the exact
    path (possibly nested per ``_subdir``); the rest follow the convention
    ``${experiment_root}/<analysis_name>`` (e.g. ``baseline``). Used only to
    clean up a crashed step's partial output, so a wrong guess degrades to
    "nothing cleaned", never to touching the wrong tree.
    """
    explicit = step_cfg.get("_output_dir")
    if explicit:
        return str(explicit)
    return os.path.join(base_root, analysis_name)


def _clean_failed_step_output(
    step_name: str, analysis_name: str, out_dir: str, pre_existed: bool
) -> None:
    """Remove a crashed step's partial output dir so it isn't mistaken for a
    complete result (#269 I1).

    A half-written dir left in place is silently picked up by the artifact
    viewer / downstream consumers (which discover artifacts by globbing,
    including dot-prefixed entries) as if the step had finished. Only a dir
    *created during this run* is removed; a dir that pre-existed the step (e.g.
    good output from an earlier run that this step partially overwrote before
    crashing) is preserved and flagged instead — destroying prior results is
    worse than a stale-partial warning. The traceback that triggered this is on
    its way up the stack (and into the run log) regardless.
    """
    if not pre_existed and os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
        logger.error(
            "Step %r (%s) crashed; removed its partial output dir %s so it "
            "won't be picked up as a complete result.",
            step_name,
            analysis_name,
            out_dir,
        )
    elif pre_existed:
        logger.error(
            "Step %r (%s) crashed; its output dir %s pre-existed this run and "
            "may now hold partial/overwritten artifacts — inspect or delete it "
            "before trusting downstream viewer / interpret output.",
            step_name,
            analysis_name,
            out_dir,
        )
    else:
        logger.error(
            "Step %r (%s) crashed before writing any output dir (%s).",
            step_name,
            analysis_name,
            out_dir,
        )


def _collect_required_task_keys(
    steps: list[tuple[str, DictConfig]],
) -> list[tuple[str, tuple[str, ...]]]:
    """Pair each requested analysis with its declared ``REQUIRED_TASK_KEYS``.

    Loads each step's module (``importlib`` caches it, so the dispatch loop's
    later load is free) and reads its module-level ``REQUIRED_TASK_KEYS`` tuple
    — the same module-flag pattern as ``ANALYSIS_NAME`` / ``HANDLES_MULTI_VARIABLE``.
    Analyses that declare none (e.g. task-less ``characterize_subspace``)
    contribute nothing, so validation stays a no-op for them.
    """
    requirements: list[tuple[str, tuple[str, ...]]] = []
    for _step_name, step_cfg in steps:
        analysis_name = str(step_cfg._name_)
        mod = _load_analysis(analysis_name)
        keys = tuple(getattr(mod, "REQUIRED_TASK_KEYS", ()))
        if keys:
            requirements.append((analysis_name, keys))
    return requirements


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

    # Fail fast: a factory task missing required keys otherwise crashes one key
    # at a time, deep in a run after the model is loaded. Validate the whole task
    # config up front against the union of keys the requested analyses declare,
    # listing every gap at once (#264). ``cfg.get`` is struct-safe for the
    # task-less runner case (#219), where the requirements list is empty anyway.
    from causalab.runner.helpers import validate_task_config

    validate_task_config(cfg.get("task"), _collect_required_task_keys(steps))

    last_step_cfg: DictConfig | None = None
    for step_name, step_cfg in steps:
        analysis_name = step_cfg._name_
        _check_known_directives(step_name, step_cfg)

        logger.info("=== Step: %s (%s) ===", step_name, analysis_name)

        analysis_mod = _load_analysis(analysis_name)
        mod_name = getattr(analysis_mod, "ANALYSIS_NAME", None)
        if mod_name != analysis_name:
            raise RuntimeError(
                f"Module {analysis_mod.__name__} declares ANALYSIS_NAME="
                f"{mod_name!r} but cfg slice has _name_={analysis_name!r}"
            )

        out_dir = _step_output_dir(step_cfg, analysis_name, base_root)
        pre_existed = os.path.isdir(out_dir)
        try:
            _run_analysis_for_variables(
                cfg,
                analysis_mod.main,
                analysis_mod,
                analysis_name,
                target_variables,
                base_root,
            )
        except Exception:
            _clean_failed_step_output(step_name, analysis_name, out_dir, pre_existed)
            raise
        _release_gpu_memory()
        last_step_cfg = step_cfg

    # Post-pipeline visualization steps
    if cfg.get("post"):
        from causalab.runner.post_steps import run_post_steps
        from causalab.io.plots.figure_format import (
            resolve_figure_format_from_analysis,
        )

        figure_format = resolve_figure_format_from_analysis(last_step_cfg)

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

    # Load a project-local .env (e.g. OPENROUTER_API_KEY for the
    # subspace-characterization judge) so runs need no manual export or `source`
    # — including SLURM jobs, which read the same shared-FS .env. See #221.
    from causalab.runner.env import load_project_dotenv

    load_project_dotenv()

    # Insert task variant into experiment_root when set
    from causalab.io.configs import apply_experiment_root_variant

    apply_experiment_root_variant(cfg)

    # --- Dispatch ---
    _run_steps(cfg)


if __name__ == "__main__":
    main()
