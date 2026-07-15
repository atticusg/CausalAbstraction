"""Golden PCA pin for the `exploration` analysis on a coherent model.

Runs exploration ``pca`` mode on ``chat-coherent`` (Qwen3-4B-Instruct-2507) over a
fixed, in-code set of arithmetic prompts and pins the residual-stream PCA's
``explained_variance_ratio`` at the ``=`` token across layers. The ``=`` token's
residual encodes the computed sum, so it carries real cross-input variance on a
coherent model (a fixed ``index`` would land on a constant chat-template token and
give degenerate PCA). EVR is the sign/permutation/rotation-invariant PCA signal —
the raw ``projections``/``rotation`` tensors flip sign run-to-run, so they are NOT
pinned directly; a looser sign-invariant ``|rotation|`` aggregate is checked too.

This is the GPU ``golden`` tier: the numbers need a coherent model (tiny-random
weights give degenerate logits). Like ``test_fixed_subspace.py`` it is a bespoke
golden — inputs are built on the fly and injected to ``tmp_path``, so it can't be an
auto-collected ``configs/golden/`` runner, and EVR lives in a ``.meta.json`` sidecar
the auto golden harness doesn't capture. The expected values live in
``test_exploration_pca_expected.json`` beside this file; regenerate them with
``EXPLORATION_PCA_UPDATE=1`` on a GPU box, review the diff, and commit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from causalab.io.artifacts import load_tensors_with_meta
from causalab.runner.run_exp import _run_steps
from tests.end_to_end._helpers.enumeration import load_e2e_runner_config

pytestmark = pytest.mark.golden

_EXPECTED = Path(__file__).with_name("test_exploration_pca_expected.json")
# EVR is a ratio in [0,1]; tolerance absorbs GPU BLAS / bf16 drift in the upstream
# activations. The |rotation| aggregate is a coarser sign-invariant direction check.
_EVR_TOL = 2e-3
_ABS_TOL = 2e-2


def _build_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Write a fixed, well-conditioned set of arithmetic prompts + the `=` slot.

    Varying both operands spreads the computed sums (1..15), so the residual
    stream at `=` has real variance structure and the top PCs are well-separated.
    """
    prompts = [f"{a} + {b} =" for a in (1, 3, 5, 7, 9) for b in (0, 2, 4, 6)]
    inputs = tmp_path / "pca_inputs.json"
    inputs.write_text(json.dumps(prompts))
    essential = tmp_path / "essential_tokens.json"
    essential.write_text(json.dumps([{"label": "eq", "text": "="}]))
    return inputs, essential


def _actual_values(token_dir: Path) -> dict:
    """Read EVR + a sign-invariant |rotation| aggregate from each saved layer."""
    out: dict[str, dict] = {}
    for meta_path in sorted(token_dir.glob("L*.meta.json")):
        stem = meta_path.name[: -len(".meta.json")]
        tensors, meta = load_tensors_with_meta(str(token_dir), stem)
        rot = tensors["rotation"].float().abs()
        out[stem] = {
            "explained_variance_ratio": [
                float(x) for x in meta["explained_variance_ratio"]
            ],
            "rotation_abs_mean": float(rot.mean().item()),
            "rotation_abs_std": float(rot.std().item()),
        }
    return out


def test_exploration_pca_evr_golden(tmp_path: Path) -> None:
    inputs, essential = _build_inputs(tmp_path)

    cfg = load_e2e_runner_config("exploration_pca")
    OmegaConf.update(cfg, "experiment_root", str(tmp_path / "art"), force_add=True)
    OmegaConf.update(cfg, "exploration.pca.inputs", str(inputs), force_add=True)
    OmegaConf.update(
        cfg, "exploration.pca.essential_tokens", str(essential), force_add=True
    )

    _run_steps(cfg)

    token_dir = tmp_path / "art" / "exploration" / "pca" / "eq"
    actual = _actual_values(token_dir)
    assert actual, "no PCA layers were produced under exploration/pca/eq"

    if os.environ.get("EXPLORATION_PCA_UPDATE"):
        _EXPECTED.write_text(json.dumps(actual, indent=2) + "\n")
        pytest.skip(
            f"bootstrapped {_EXPECTED.name} ({sorted(actual)}); review and commit"
        )

    assert _EXPECTED.exists(), (
        f"missing {_EXPECTED.name}; regenerate with EXPLORATION_PCA_UPDATE=1 on a GPU box"
    )
    expected = json.loads(_EXPECTED.read_text())
    assert set(actual) == set(expected), (
        f"layer set drift: got {sorted(actual)}, expected {sorted(expected)}"
    )
    for layer, exp in expected.items():
        got = actual[layer]
        ev_e, ev_g = exp["explained_variance_ratio"], got["explained_variance_ratio"]
        assert len(ev_e) == len(ev_g), (
            f"{layer}: EVR length {len(ev_g)} != expected {len(ev_e)}"
        )
        for i, (e, g) in enumerate(zip(ev_e, ev_g)):
            assert abs(e - g) <= _EVR_TOL, (
                f"{layer} EVR[{i}]: got {g!r}, expected {e!r} (tol {_EVR_TOL})"
            )
        for key, tol in (
            ("rotation_abs_mean", _ABS_TOL),
            ("rotation_abs_std", _ABS_TOL),
        ):
            assert abs(exp[key] - got[key]) <= tol, (
                f"{layer} {key}: got {got[key]!r}, expected {exp[key]!r} (tol {tol})"
            )
