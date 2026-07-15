"""Canonical enumeration + Hydra-compose for end-to-end runner configs.

Two trees feed the end-to-end gate:

* Production runners under ``causalab/configs/runners/`` — composed via
  :func:`causalab.io.configs.load_runner_config` (untouched, normal Hydra
  searchpath). The compose-only test exercises these.
* Test-side runners under ``tests/end_to_end/configs/{smoke,golden}/`` —
  composed via :func:`load_e2e_runner_config`, which initializes Hydra on
  the test-side primary config dir and adds ``causalab/configs/`` to
  ``hydra.searchpath`` so shared ``/base`` / ``/task`` / ``/analysis``
  defaults still resolve.

This module is imported by tests under ``tests/end_to_end/``.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

# Repo root: tests/end_to_end/_helpers/enumeration.py → repo root is 3 up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Production configs tree — home to non-baseline runner groups (``age/``,
# ``demos/``, ``mcqa/``, …) plus shared ``/base``, ``/task``,
# ``/analysis``, ``/model`` groups.
MAIN_CONFIGS_DIR = _REPO_ROOT / "causalab" / "configs"
DEFAULT_CONFIGS_ROOT = MAIN_CONFIGS_DIR  # backwards-compat alias

# Test-side runner configs (smoke + golden runner groups) live here.
# The primary config dir for Hydra composition when loading these.
E2E_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def enumerate_runner_configs(configs_root: Path = DEFAULT_CONFIGS_ROOT) -> list[str]:
    """Return Hydra-style names for every runner YAML under ``configs/runners/``.

    Names are returned in the form ``runners/<group>/<name>`` (no
    ``.yaml`` suffix), suitable for passing to
    :func:`causalab.io.configs.load_runner_config`. After the e2e
    consolidation the former ``runners/baselines/`` group lives under
    ``tests/end_to_end/configs/smoke/`` — use
    :func:`enumerate_e2e_configs` for those.
    """
    runners_dir = configs_root / "runners"
    if not runners_dir.is_dir():
        return []
    names: list[str] = []
    for yaml_path in sorted(runners_dir.rglob("*.yaml")):
        rel = yaml_path.relative_to(configs_root).with_suffix("")
        names.append(rel.as_posix())
    return names


def enumerate_e2e_configs(group: str) -> list[str]:
    """Return ``<group>/<name>`` for every YAML under ``tests/end_to_end/configs/<group>/``.

    ``group`` is ``"smoke"`` (tiny-random, existence-only; CPU) or
    ``"golden"`` (chat-coherent, value pins + determinism; GPU). Returned
    names are suitable for passing to :func:`load_e2e_runner_config`.
    """
    group_dir = E2E_CONFIGS_DIR / group
    if not group_dir.is_dir():
        return []
    return sorted(f"{group}/{p.stem}" for p in group_dir.glob("*.yaml"))


def load_e2e_runner_config(name: str) -> DictConfig:
    """Compose a test-side runner cfg from ``tests/end_to_end/configs/``.

    ``name`` is ``<group>/<stem>`` (e.g. ``"smoke/age"`` or
    ``"golden/mcqa"``). The primary config dir is
    :data:`E2E_CONFIGS_DIR`; the production tree
    (:data:`MAIN_CONFIGS_DIR`) is added to ``hydra.searchpath`` so shared
    ``/base`` / ``/task: <name>`` / ``/analysis/<name>`` / ``/model``
    defaults resolve without duplication. Test-local model variants
    (``chat-coherent``, ``tiny-random``) live under
    ``tests/end_to_end/configs/model/`` and take precedence over the
    production model group.

    Returned ``DictConfig`` is fully composed; callers set per-test
    fields (``experiment_root``, ``seed``) before running.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(E2E_CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name=name,
            overrides=[f"hydra.searchpath=[file://{MAIN_CONFIGS_DIR}]"],
        )
    # Same post-compose hook the production loader applies; keeps
    # ``task.variant`` handling consistent with real runs.
    from causalab.io.configs import apply_experiment_root_variant

    apply_experiment_root_variant(cfg)
    return cfg
