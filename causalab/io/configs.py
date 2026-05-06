"""Runner config save/load helpers.

Provides functions for notebook and runner workflows:
- ``save_runner_config``: persist a YAML override string to the runner
  configs directory so Hydra can pick it up.
- ``load_runner_config``: compose the full Hydra config for a runner
  preset and return it as an OmegaConf DictConfig.

Runner configs are primary Hydra configs (selected via ``--config-name runners/<group>/<name>``)
that include ``base`` for shared globals and declare task/model directly.
Analysis defaults are loaded via Hydra defaults list composition —
each runner config declares ``- /analysis/<name>`` in its defaults list,
and each analysis YAML carries ``# @package <name>`` so its body lands
at ``cfg.<name>``.

Demo notebooks under ``demos/`` route their configs through these helpers,
which place them under ``configs/runners/demos/`` to keep the configs tree
organized.
"""

from __future__ import annotations

import os

import causalab

# Subdirectory (relative to ``configs/``) where notebook-authored runner
# configs land. Notebooks call ``save_runner_config(yaml, "<name>")`` and the
# file is written to ``configs/<DEMOS_SUBDIR>/<name>.yaml``.
DEMOS_SUBDIR = "runners/demos"


def _configs_dir() -> str:
    return os.path.join(os.path.dirname(causalab.__file__), "configs")


def save_runner_config(yaml_str: str, config_name: str) -> str:
    """Write a YAML override string to ``configs/runners/demos/{config_name}.yaml``.

    Returns the absolute path of the written file.
    """
    path = os.path.join(_configs_dir(), DEMOS_SUBDIR, f"{config_name}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(yaml_str)
    return path


def apply_experiment_root_variant(cfg) -> None:
    """If ``cfg.task.variant`` is set, append it to ``cfg.experiment_root``.

    Mutates ``cfg`` in place. Shared between the runner entrypoint and
    notebooks so both resolve paths identically. Appending preserves any
    caller-supplied override (e.g. a session-scoped root from research-mode
    workflows); rebuilding the path here would silently clobber it.
    """
    from omegaconf import OmegaConf

    variant = cfg.task.get("variant")
    if variant:
        OmegaConf.update(
            cfg,
            "experiment_root",
            f"{cfg.experiment_root}/{variant}",
        )


def load_runner_config(config_name: str):
    """Compose the full Hydra config for runner *config_name*.

    Uses Hydra's ``compose`` API so the returned ``DictConfig`` contains
    every default merged in — exactly what ``run_exp.py`` would see.
    Runner configs live under ``configs/runners/<group>/<name>.yaml``;
    notebook-authored configs land under ``runners/demos/`` and are
    composed via ``config_name=runners/demos/<name>``.

    Accepts either a bare name (``"residual_stream_tracing_demo"``) — which
    is auto-prefixed with ``runners/demos/`` to match
    :func:`save_runner_config` — or a path containing ``/`` (e.g.
    ``"runners/weekdays/weekdays_8b_pipeline"``), used verbatim.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    configs_dir = _configs_dir()
    if "/" not in config_name:
        config_name = f"{DEMOS_SUBDIR}/{config_name}"

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(config_name=config_name)
    apply_experiment_root_variant(cfg)
    return cfg
