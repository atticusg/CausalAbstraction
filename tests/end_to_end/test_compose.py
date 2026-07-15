"""Compose-only + dispatch-shape gates for every end-to-end runner config.

Two parametrized tests:

1. ``test_runner_config_composes`` — for every YAML under
   ``tests/end_to_end/configs/{smoke,golden}/``, compose it via
   Hydra and assert no exception. Catches Hydra defaults-list breakage,
   missing ``_target_`` qualnames, and broken interpolations the moment
   an e2e runner config is touched. No deeper assertions — the smoke
   tier covers actual execution; this test only verifies that the
   composition path is healthy.

2. ``test_runner_dispatch_shape`` — for every covered runner config,
   iterate analysis steps via
   :func:`causalab.runner.run_exp._iter_analysis_steps` and assert that
   each ``_name_`` resolves to an importable
   ``causalab.analyses.<name>.main`` module. If
   ``_iter_analysis_steps`` is renamed or its dispatch shape changes,
   this test fails the day that drift happens.

Scope is **deliberately narrow**: this gate covers only the
end-to-end test fixtures under ``tests/end_to_end/configs/``. The
production runner configs under ``causalab/configs/runners/`` are
research artifacts whose composability is the researcher's problem,
not the test gate's. Composition uses
:func:`tests.end_to_end._helpers.enumeration.load_e2e_runner_config`,
which adds ``causalab/configs/`` to ``hydra.searchpath`` so shared
``/base`` / ``/task`` / ``/analysis`` defaults still resolve.

Default tier: ``unit`` (declared via module-scope ``pytestmark``).

Wall-time budget: <5 s on CPU, no GPU, no model loads. Hydra global
state is cleared between calls by the loader, so the parametrize loop
is safe.
"""

from __future__ import annotations

import importlib

import pytest
from omegaconf import DictConfig

from causalab.runner.run_exp import (
    _iter_analysis_steps,  # pyright: ignore[reportPrivateUsage]
)
from tests.end_to_end._helpers.enumeration import (
    enumerate_e2e_configs,
    load_e2e_runner_config,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Parametrize sources
# ---------------------------------------------------------------------------

# Enumerate at import time so failures point at the offending config name
# (via the parametrize id) rather than a generic collection error.
_E2E_SMOKE_CONFIGS: list[str] = enumerate_e2e_configs("smoke")
_E2E_GOLDEN_CONFIGS: list[str] = enumerate_e2e_configs("golden")

# All configs the compose gate covers. Parametrize IDs are stable (the
# ``<group>/<stem>`` name itself) so external tooling can cross-reference
# junit by ``test_function[<id>]``.
_ALL_RUNNER_CONFIGS: list[str] = _E2E_SMOKE_CONFIGS + _E2E_GOLDEN_CONFIGS

# Empty parametrize lists silently pass — pytest collects zero cases and the
# file goes green with nothing exercised. Convert that into a loud collection
# error so the gate fails the day enumeration drifts.
if not _E2E_SMOKE_CONFIGS:
    raise RuntimeError("no smoke configs found under tests/end_to_end/configs/smoke/")
if not _E2E_GOLDEN_CONFIGS:
    raise RuntimeError("no golden configs found under tests/end_to_end/configs/golden/")


# ---------------------------------------------------------------------------
# Compose-only test — every runner config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", _ALL_RUNNER_CONFIGS, ids=_ALL_RUNNER_CONFIGS)
def test_runner_config_composes(config_name: str) -> None:
    """Compose ``config_name`` via Hydra and assert no exception.

    A failure here means one of:
      - a ``defaults:`` entry references a nonexistent group/file;
      - a ``_target_`` qualname has a typo;
      - an OmegaConf interpolation is malformed;
      - the YAML itself has a syntax error.

    None of those should land on ``main``; this test makes them visible at
    PR time.
    """
    cfg = load_e2e_runner_config(config_name)
    # Hydra returns a DictConfig on success and raises on failure, so the
    # real contract is "no exception". The isinstance check pins the return
    # type — converts a no-op assertion into one with teeth.
    assert isinstance(cfg, DictConfig), (
        f"load_e2e_runner_config({config_name!r}) returned "
        f"{type(cfg).__name__}, expected DictConfig"
    )


# ---------------------------------------------------------------------------
# Dispatch-shape sanity test — every runner config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", _ALL_RUNNER_CONFIGS, ids=_ALL_RUNNER_CONFIGS)
def test_runner_dispatch_shape(config_name: str) -> None:
    """Every analysis step's ``_name_`` resolves to an importable module.

    The ``_name_`` → ``causalab.analyses.<name>.main`` dispatch shape is
    the one ``run_exp.py``'s analysis loader resolves. If
    :func:`causalab.runner.run_exp._iter_analysis_steps` ever changes
    shape (different key, different module template, registry pattern,
    …), this test fails.

    This test exercises the shipped (``causalab.analyses.<name>.main``)
    dispatch path. The session-local fallback in
    :func:`causalab.runner.run_exp._load_analysis` (when
    ``CAUSALAB_SESSION_CODE`` is set, falling back to ``analyses.<name>.main``
    without the ``causalab.`` prefix) is not exercised here — those are
    session-scoped and not shipped on disk.
    """
    cfg = load_e2e_runner_config(config_name)
    steps = list(_iter_analysis_steps(cfg))
    assert steps, (
        f"runner {config_name!r} yielded no analysis steps; "
        f"either its defaults list lacks `- analysis/<name>` entries, or "
        f"`_iter_analysis_steps` no longer recognises the dispatch shape."
    )
    for step_key, step_cfg in steps:
        name = step_cfg.get("_name_")
        assert isinstance(name, str) and name, (
            f"step {step_key!r} in {config_name!r} has no `_name_` string; "
            f"`_iter_analysis_steps` should not have yielded it."
        )
        module_qualname = f"causalab.analyses.{name}.main"
        try:
            importlib.import_module(module_qualname)
        except ImportError as exc:  # pragma: no cover — failure path
            pytest.fail(
                f"step {step_key!r} in {config_name!r} declares "
                f"_name_={name!r}, but `import {module_qualname}` failed: {exc}. "
                f"Either the analysis module is missing or the dispatch shape "
                f"in run_exp.py has drifted."
            )
