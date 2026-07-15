"""Regression tests for required-`task.*`-key validation (#264).

Factory tasks (resolved by name with no shipped ``configs/task/<name>.yaml``)
can silently omit keys that shipped task configs always set —
``intervention_metric`` (direct attribute access) and ``colormap`` /
``colormap2`` (``${task.*}`` interpolations baked into the analysis configs).
Each used to crash one at a time, deep in a run after the model was loaded.
``validate_task_config`` now collects the union of keys the requested analyses
declare (via each analysis module's ``REQUIRED_TASK_KEYS``) and fails fast with
every missing one listed at once. Ties to #263, whose task-setup template
emits a task config carrying these keys.
"""

import pytest
from omegaconf import OmegaConf

from causalab.runner import run_exp
from causalab.runner.helpers import validate_task_config

pytestmark = pytest.mark.unit


def _struct_task(d):
    """Build a struct-mode ``task`` DictConfig, matching what Hydra hands the runner."""
    cfg = OmegaConf.create(d)
    OmegaConf.set_struct(cfg, True)
    return cfg


def test_two_missing_keys_reported_in_one_error():
    """Acceptance: a task missing two required keys reports BOTH in a single error."""
    task_cfg = _struct_task({"name": "factory_task", "n_train": 10})
    requirements = [("activation_manifold", ("colormap", "intervention_metric"))]

    with pytest.raises(ValueError) as exc:
        validate_task_config(task_cfg, requirements)

    msg = str(exc.value)
    assert "task.colormap" in msg
    assert "task.intervention_metric" in msg
    # One raised error carries both — not one-at-a-time.
    assert msg.count("required by") == 2


def test_error_names_every_analysis_needing_a_key():
    """A key required by several analyses lists all of them; all gaps reported once."""
    task_cfg = _struct_task({"name": "factory_task"})
    requirements = [
        ("activation_manifold", ("colormap", "intervention_metric")),
        ("subspace", ("colormap", "intervention_metric")),
        ("path_steering", ("colormap", "colormap2")),
    ]

    with pytest.raises(ValueError) as exc:
        validate_task_config(task_cfg, requirements)

    msg = str(exc.value)
    for key in ("task.colormap", "task.colormap2", "task.intervention_metric"):
        assert key in msg
    # `colormap` is required by all three; the message attributes it to each.
    colormap_line = next(
        line for line in msg.splitlines() if line.strip().startswith("- task.colormap ")
    )
    for analysis in ("activation_manifold", "subspace", "path_steering"):
        assert analysis in colormap_line


def test_present_but_null_key_is_valid():
    """Membership, not truthiness: an explicit ``null`` (e.g. ``colormap2: null``) passes.

    Shipped configs and the #263 task-setup template emit ``colormap2: null``;
    the key is present so the ``${task.colormap2}`` interpolation resolves. Only
    *absence* breaks it, so validation must accept a null value.
    """
    task_cfg = _struct_task(
        {"colormap": "rainbow", "colormap2": None, "intervention_metric": "kl"}
    )
    requirements = [
        ("activation_manifold", ("colormap", "intervention_metric")),
        ("path_steering", ("colormap", "colormap2")),
    ]
    validate_task_config(task_cfg, requirements)  # no raise


def test_no_requirements_is_noop_even_without_task():
    """Task-less analyses declare no keys -> validation is a no-op (#219 stays green)."""
    validate_task_config(None, [])  # no raise


def test_complete_task_config_passes():
    """A task config carrying every required key validates cleanly."""
    task_cfg = _struct_task(
        {
            "name": "natural_domains_arithmetic",
            "colormap": "rainbow",
            "colormap2": None,
            "intervention_metric": "kl",
        }
    )
    requirements = [
        ("activation_manifold", ("colormap", "intervention_metric")),
        ("locate", ("intervention_metric",)),
    ]
    validate_task_config(task_cfg, requirements)  # no raise


# ---------------------------------------------------------------------------
# Runner wiring: _run_steps validates before running any step
# ---------------------------------------------------------------------------


def _struct_cfg(d):
    cfg = OmegaConf.create(d)
    OmegaConf.set_struct(cfg, True)
    return cfg


def test_run_steps_fails_fast_before_any_analysis_runs(monkeypatch):
    """A missing required key aborts in ``_run_steps`` before the analysis runs.

    ``_collect_required_task_keys`` reads ``REQUIRED_TASK_KEYS`` off the loaded
    module (same flag pattern as ``ANALYSIS_NAME``); a stub module supplies it so
    the test stays torch-free, mirroring ``test_taskless_analysis.py``.
    """
    cfg = _struct_cfg(
        {
            "experiment_root": "/out",
            "task": {"name": "factory_task"},  # missing colormap + intervention_metric
            "step": {"_name_": "fake"},
        }
    )

    ran = {"main": False}

    class FakeMod:
        ANALYSIS_NAME = "fake"
        REQUIRED_TASK_KEYS = ("colormap", "intervention_metric")

        @staticmethod
        def main(_c):
            ran["main"] = True

    monkeypatch.setattr(run_exp, "_load_analysis", lambda name: FakeMod)
    monkeypatch.setattr(run_exp, "_release_gpu_memory", lambda: None)

    with pytest.raises(ValueError) as exc:
        run_exp._run_steps(cfg)

    assert "task.colormap" in str(exc.value)
    assert "task.intervention_metric" in str(exc.value)
    assert ran["main"] is False, "validation must abort before any analysis runs"
