"""Regression test for `subspace method: fixed` (#218).

Threads a **given** rotation end-to-end (subspace[method:fixed] →
activation_manifold → path_steering) on the tiny-random stub and asserts:

1. The canonical subspace bundle lands under ``subspace/fixed_k{K}/{tv}/`` with
   the keys downstream reads — crucially ``features/raw_features.safetensors``,
   which ``path_steering`` needs for the **linear** path mode.
2. ``path_steering`` ran the **linear** path mode (the ``paths/linear/`` dir is
   produced) — i.e. it was NOT silently skipped. When ``raw_features`` is
   missing, ``path_steering`` drops the linear mode entirely
   (``path_steering/main.py``), so the directory's existence is the direct
   "not skipped" signal.

This is smoke-tier (tiny-random, CPU): the linear-vs-geometric *numbers* need a
coherent model, but "did the linear plumbing fire" is model-agnostic. The fixed
rotation is generated in ``tmp_path`` and injected via a Hydra override, so this
is a bespoke test rather than an auto-collected ``configs/smoke/`` runner (which
can't inject a tmp path). See docs/TESTS.md "Smoke vs golden split".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from causalab.runner.run_exp import _run_steps
from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME, tiny_random_model
from tests.end_to_end._helpers.enumeration import load_e2e_runner_config

pytestmark = pytest.mark.smoke

K_FEATURES = 8


@pytest.fixture(scope="session")
def _prewarm_tiny_model():
    """Populate the HF on-disk cache before the runner loads the stub."""
    tiny_random_model()


def _make_rotation(d_model: int, k: int, path: Path) -> None:
    """Write a deterministic orthonormal ``(d_model, k)`` rotation .safetensors."""
    gen = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(d_model, k, generator=gen))
    save_file({"rotation_matrix": q.contiguous()}, str(path))


def test_fixed_subspace_threads_linear_path_mode(
    tmp_path: Path, _prewarm_tiny_model
) -> None:
    from transformers import AutoConfig

    d_model = int(AutoConfig.from_pretrained(TINY_RANDOM_MODEL_NAME).hidden_size)
    rotation_path = tmp_path / "rotation.safetensors"
    _make_rotation(d_model, K_FEATURES, rotation_path)

    cfg = load_e2e_runner_config("fixed_subspace_chain")
    OmegaConf.update(cfg, "experiment_root", str(tmp_path), force_add=True)
    OmegaConf.update(cfg, "subspace.fixed.artifact", str(rotation_path), force_add=True)

    _run_steps(cfg)

    # 1. Canonical bundle under subspace/fixed_k{K}/{tv}/.
    ss_dir = tmp_path / "subspace" / f"fixed_k{K_FEATURES}" / "result"
    for rel in (
        "rotation.safetensors",
        "train_dataset.json",
        "metadata.json",
        "features/training_features.safetensors",
        "features/raw_features.safetensors",
    ):
        assert (ss_dir / rel).exists(), f"missing fixed-subspace artifact: {rel}"

    meta = json.loads((ss_dir / "metadata.json").read_text())
    assert meta["mode"] == "single"
    assert meta["method"] == "pca"  # artifact-type semantics (rotation matrix)
    assert meta["k_features"] == K_FEATURES
    assert meta["discovery"] == "fixed"

    # 2. path_steering ran the linear path mode (not skipped). Its absence would
    #    mean raw_features wasn't threaded and the geodesic-vs-linear comparison
    #    was silently dropped.
    paths_dir = (
        tmp_path
        / "path_steering"
        / f"fixed_k{K_FEATURES}"
        / "L1_last_token"
        / "spline_s0.0"
        / "result"
        / "paths"
    )
    assert (paths_dir / "linear" / "pair_distributions.safetensors").exists(), (
        "linear path mode was skipped — raw_features.safetensors not threaded"
    )
    assert (paths_dir / "geometric" / "pair_distributions.safetensors").exists(), (
        "geometric path mode did not run"
    )


def test_fixed_subspace_score_writes_results(
    tmp_path: Path, _prewarm_tiny_model
) -> None:
    """``subspace.fixed.score: true`` scores the threaded rotation at the cell.

    Plumbing-tier (tiny-random, CPU): the score *value* is meaningless on random-init
    weights — the golden (``weekdays_fixed_subspace``, chat-coherent) pins the value.
    Here we assert only that the score *fired* and wrote a well-formed ``results.json``
    with ``subspace_score`` / ``full_cell_score`` / ``score_ratio`` (the same
    ``run_layer_scan`` plumbing ``locate`` uses). Its absence would mean the ``score``
    knob silently no-op'd. See docs/TESTS.md "Smoke vs golden split".
    """
    from transformers import AutoConfig

    d_model = int(AutoConfig.from_pretrained(TINY_RANDOM_MODEL_NAME).hidden_size)
    rotation_path = tmp_path / "rotation.safetensors"
    _make_rotation(d_model, K_FEATURES, rotation_path)

    cfg = load_e2e_runner_config("fixed_subspace_only")
    OmegaConf.update(cfg, "experiment_root", str(tmp_path), force_add=True)
    OmegaConf.update(cfg, "subspace.fixed.artifact", str(rotation_path), force_add=True)
    OmegaConf.update(cfg, "subspace.fixed.score", True, force_add=True)

    _run_steps(cfg)

    results_path = (
        tmp_path / "subspace" / f"fixed_k{K_FEATURES}" / "result" / "results.json"
    )
    assert results_path.exists(), (
        "fixed.score=true did not write results.json — the scan was skipped"
    )
    scores = json.loads(results_path.read_text())
    for key in ("subspace_score", "full_cell_score", "score_ratio"):
        assert key in scores, f"results.json missing {key}"
    # The score's units are the task's intervention_metric (weekdays = kl), so the
    # value is a divergence (>= 0, unbounded above) rather than a [0, 1] accuracy.
    # The *value* is garbage on tiny-random — only finiteness / non-negativity is
    # asserted here (the golden pins the real numbers). score_ratio may be NaN
    # (zero-denominator guard); the key must still be present.
    for key in ("subspace_score", "full_cell_score"):
        val = scores[key]
        assert isinstance(val, (int, float)) and math.isfinite(val), (
            f"{key} is not a finite number: {val!r}"
        )
        assert val >= 0.0, f"{key}={val} is negative"


def test_nonorthonormal_rotation_is_orthonormalized(
    tmp_path: Path, _prewarm_tiny_model
) -> None:
    """A non-orthonormal given rotation must not produce diverging frames.

    SubspaceFeaturizer orthogonalizes its rotation, so if the saved features used
    the raw (non-orthonormal) basis they'd disagree with the featurizer the
    manifold/steering analyses rebuild. The fix orthonormalizes once: the saved
    rotation must be orthonormal and the saved projected features must equal
    ``raw @ saved_rotation`` (the same frame the featurizer reproduces).
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig

    d_model = int(AutoConfig.from_pretrained(TINY_RANDOM_MODEL_NAME).hidden_size)
    # Deliberately non-orthonormal: scaled + shifted Gaussian columns.
    gen = torch.Generator().manual_seed(1)
    raw_rotation = torch.randn(d_model, K_FEATURES, generator=gen) * 2.0 + 1.0
    gram = raw_rotation.T @ raw_rotation
    assert not torch.allclose(gram, torch.eye(K_FEATURES), atol=1e-3), (
        "fixture rotation should be non-orthonormal"
    )
    rotation_path = tmp_path / "rotation.safetensors"
    save_file({"rotation_matrix": raw_rotation.contiguous()}, str(rotation_path))

    cfg = load_e2e_runner_config("fixed_subspace_only")
    OmegaConf.update(cfg, "experiment_root", str(tmp_path), force_add=True)
    OmegaConf.update(cfg, "subspace.fixed.artifact", str(rotation_path), force_add=True)
    _run_steps(cfg)

    ss_dir = tmp_path / "subspace" / f"fixed_k{K_FEATURES}" / "result"
    saved_rot = load_file(str(ss_dir / "rotation.safetensors"))["rotation_matrix"]
    saved_feats = load_file(str(ss_dir / "features/training_features.safetensors"))[
        "features"
    ]
    saved_raw = load_file(str(ss_dir / "features/raw_features.safetensors"))["features"]

    # Saved rotation was orthonormalized (not the raw non-orthonormal input).
    saved_gram = saved_rot.float().T @ saved_rot.float()
    assert torch.allclose(saved_gram, torch.eye(K_FEATURES), atol=1e-4), (
        "saved rotation_matrix is not orthonormal — featurizer/manifold frames diverge"
    )
    # Saved features use the SAME (orthonormal) rotation the featurizer rebuilds.
    expected = saved_raw.float() @ saved_rot.float()
    assert torch.allclose(saved_feats.float(), expected, atol=1e-4), (
        "saved training_features do not match raw @ saved_rotation"
    )


def test_fixed_inputs_are_mutually_exclusive(tmp_path: Path) -> None:
    """feature_ids / artifact / source must not be combined (no silent winner)."""
    from causalab.analyses.subspace import resolve_fixed_rotation

    rotation_path = tmp_path / "rotation.safetensors"
    _make_rotation(8, K_FEATURES, rotation_path)

    with pytest.raises(ValueError, match="exactly one of feature_ids"):
        resolve_fixed_rotation(
            {"artifact": str(rotation_path), "feature_ids": [1, 2, 3]},
            out_dir=str(tmp_path),
        )

    with pytest.raises(ValueError, match="one of feature_ids / artifact / source"):
        resolve_fixed_rotation({}, out_dir=str(tmp_path))
