"""Regression tests for task-less analyses in the runner (#219).

A runner that mounts only an ``analysis/`` + ``model`` with no ``/task`` (e.g.
``characterize_subspace``, whose data is the step-1 JSON rather than a
causal-model task) used to crash in ``run_exp.main()`` before the analysis ever
loaded: both ``apply_experiment_root_variant`` and ``_resolve_target_variables``
dereferenced ``cfg.task.*`` with attribute access, which raises
``ConfigAttributeError`` under Hydra's struct mode when ``task`` is absent.

These guard the struct-safe accessors and assert a task-less runner reaches
completion with its ``experiment_root`` left un-prefixed (no leaked variant
segment). ``main()`` is ``@hydra.main``-decorated and can't be called with a
cfg directly, so the end-to-end test mirrors its two-line body
(``apply_experiment_root_variant`` then ``_run_steps``) with a stub analysis,
the same boundary-stubbing spirit as ``test_run_exp_dispatch.py``.
"""

from omegaconf import OmegaConf

from causalab.io.configs import apply_experiment_root_variant
from causalab.runner import run_exp

import pytest

pytestmark = pytest.mark.unit


def _struct_cfg(d):
    """Build a struct-mode DictConfig, matching what Hydra hands ``main()``."""
    cfg = OmegaConf.create(d)
    OmegaConf.set_struct(cfg, True)
    return cfg


def test_apply_experiment_root_variant_no_task():
    """No ``task`` key -> no-op, ``experiment_root`` unchanged (no crash)."""
    cfg = _struct_cfg({"experiment_root": "/out"})
    apply_experiment_root_variant(cfg)
    assert cfg.experiment_root == "/out"


def test_apply_experiment_root_variant_with_variant():
    """``task.variant`` still appends to the root (behavior preserved)."""
    cfg = _struct_cfg({"experiment_root": "/out", "task": {"variant": "weekdays"}})
    apply_experiment_root_variant(cfg)
    assert cfg.experiment_root == "/out/weekdays"


def test_resolve_target_variables_no_task():
    """No ``task`` key -> module-default ``[None]`` (no crash)."""
    cfg = _struct_cfg({"experiment_root": "/out"})
    assert run_exp._resolve_target_variables(cfg) == [None]


def test_taskless_runner_runs_to_completion(tmp_path, monkeypatch):
    """A task-less runner reaches completion at the un-prefixed root (#219).

    Acceptance criteria 1-3: ``main()``-equivalent path runs without a task to
    completion, no dummy-task stub, and no variant segment appended.
    """
    root = str(tmp_path / "out")
    cfg = _struct_cfg({"experiment_root": root, "step": {"_name_": "fake"}})

    seen = {}

    class FakeMod:
        ANALYSIS_NAME = "fake"

        @staticmethod
        def main(c):
            seen["root"] = c.experiment_root
            (tmp_path / "ran.sentinel").write_text("ok")

    monkeypatch.setattr(run_exp, "_load_analysis", lambda name: FakeMod)
    monkeypatch.setattr(run_exp, "_release_gpu_memory", lambda: None)

    # Mirror main()'s body — both calls must tolerate the missing task.
    apply_experiment_root_variant(cfg)
    run_exp._run_steps(cfg)

    assert seen["root"] == root, (
        "experiment_root must stay un-prefixed (no variant leak)"
    )
    assert (tmp_path / "ran.sentinel").exists(), "analysis step never ran"
