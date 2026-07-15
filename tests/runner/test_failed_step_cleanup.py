"""Failed-step output cleanup in the runner (issue #269, I1).

A crashed analysis step used to leave its half-written output dir behind, where
the artifact viewer then picked it up as if the step
had finished. ``_run_steps`` now removes a dir *created by the failing run*
(preserving a pre-existing one, which it only flags) before re-raising. These
tests pin that behaviour with a stub analysis that creates its dir then raises —
no model / GPU.
"""

from __future__ import annotations

import os
import types

import pytest
from omegaconf import OmegaConf

from causalab.runner import run_exp
from causalab.runner.run_exp import (  # pyright: ignore[reportPrivateUsage]
    _clean_failed_step_output,
    _run_steps,
    _step_output_dir,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _step_output_dir
# ---------------------------------------------------------------------------


def test_step_output_dir_uses_explicit_output_dir():
    step_cfg = OmegaConf.create(
        {"_name_": "subspace", "_output_dir": "/x/subspace/pca_k64"}
    )
    assert _step_output_dir(step_cfg, "subspace", "/x") == "/x/subspace/pca_k64"


def test_step_output_dir_falls_back_to_convention():
    step_cfg = OmegaConf.create({"_name_": "baseline"})
    assert (
        _step_output_dir(step_cfg, "baseline", "/x/artifacts")
        == "/x/artifacts/baseline"
    )


# ---------------------------------------------------------------------------
# _clean_failed_step_output
# ---------------------------------------------------------------------------


def test_cleanup_removes_newly_created_dir(tmp_path):
    out_dir = tmp_path / "baseline"
    out_dir.mkdir()
    (out_dir / "partial.json").write_text("{}")
    _clean_failed_step_output("baseline", "baseline", str(out_dir), pre_existed=False)
    assert not out_dir.exists()
    # Removal, not in-tree quarantine: nothing left for the viewer to glob.
    assert not (tmp_path / ".failed").exists()


def test_cleanup_preserves_preexisting_dir(tmp_path):
    out_dir = tmp_path / "baseline"
    out_dir.mkdir()
    (out_dir / "good.json").write_text("{}")
    _clean_failed_step_output("baseline", "baseline", str(out_dir), pre_existed=True)
    assert out_dir.exists()
    assert (out_dir / "good.json").exists()


def test_cleanup_noop_when_no_dir(tmp_path):
    out_dir = tmp_path / "never_made"
    # Must not raise even though the dir was never created.
    _clean_failed_step_output("baseline", "baseline", str(out_dir), pre_existed=False)
    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# _run_steps integration (stubbed analysis)
# ---------------------------------------------------------------------------


def _failing_analysis(out_dir: str):
    """Build a stub analysis module whose main() creates ``out_dir`` then raises."""

    def main(cfg):  # noqa: ANN001
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "partial.json"), "w") as fh:
            fh.write("{}")
        raise RuntimeError("kaboom")

    return types.SimpleNamespace(
        ANALYSIS_NAME="faily", main=main, __name__="stub_faily"
    )


def test_run_steps_cleans_partial_dir_on_crash(tmp_path, monkeypatch):
    """A crashing step's freshly-created output dir is removed before the error
    propagates, so nothing partial is left for the viewer."""
    exp_root = tmp_path / "artifacts"
    exp_root.mkdir()
    out_dir = exp_root / "faily"

    cfg = OmegaConf.create(
        {"experiment_root": str(exp_root), "faily": {"_name_": "faily"}}
    )
    monkeypatch.setattr(
        run_exp, "_load_analysis", lambda name: _failing_analysis(str(out_dir))
    )

    with pytest.raises(RuntimeError, match="kaboom"):
        _run_steps(cfg)

    assert not out_dir.exists(), "partial output dir should have been removed"
    assert not (exp_root / ".failed").exists()


def test_run_steps_preserves_preexisting_dir_on_crash(tmp_path, monkeypatch):
    """A dir that pre-existed the run is preserved (only flagged), not destroyed."""
    exp_root = tmp_path / "artifacts"
    out_dir = exp_root / "faily"
    out_dir.mkdir(parents=True)
    (out_dir / "good.json").write_text('{"prior": true}')

    cfg = OmegaConf.create(
        {"experiment_root": str(exp_root), "faily": {"_name_": "faily"}}
    )
    monkeypatch.setattr(
        run_exp, "_load_analysis", lambda name: _failing_analysis(str(out_dir))
    )

    with pytest.raises(RuntimeError, match="kaboom"):
        _run_steps(cfg)

    assert out_dir.exists(), "pre-existing dir must not be destroyed on crash"
    assert (out_dir / "good.json").exists()
