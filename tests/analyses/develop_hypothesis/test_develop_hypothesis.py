"""Tests for the develop_hypothesis analysis (`main(cfg)`).

Exercises the task-less Hydra analysis end-to-end on CPU: it loads the
session-local ``models.py`` / ``counterfactuals.py`` scaffolds named by
``cfg.develop_hypothesis.hypotheses_dir``, injects the null/all references, runs
the engine, and writes ``distinguishability.json`` under ``_output_dir``. The
causal model is symbolic (no neural model), so the distinguishability rates are
fully deterministic at a fixed seed — this is a ``numerical_unit`` value-pin:
develop_hypothesis loads no model, so a coherent-model golden does not apply; the
ground truth is generated on the fly from the inline scaffold and compared to
``expected_distinguishability.json`` beside this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from causalab.analyses.develop_hypothesis import main as dh_main

pytestmark = pytest.mark.numerical_unit

_EXPECTED = Path(__file__).with_name("expected_distinguishability.json")

_MODELS_PY = """
from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var

_D = [0, 1, 2, 3]
_values = {"X": _D, "Y": _D, "total": list(range(7)), "raw_input": None, "raw_output": None}
_mech = {
    "X": input_var(_D),
    "Y": input_var(_D),
    "total": Mechanism(parents=["X", "Y"], compute=lambda t: t["X"] + t["Y"]),
    "raw_input": Mechanism(parents=["X", "Y"], compute=lambda t: f"{t['X']}+{t['Y']}="),
    "raw_output": Mechanism(parents=["total"], compute=lambda t: str(t["total"])),
}
m = CausalModel(_mech, _values, id="add")
MODELS = {"add": m}
DEFAULT_MODEL = "add"
HYPOTHESES = {"total": ("add", ["total"])}
TARGETS = ["total"]
"""

_CF_PY = """
import random

def _ex(xb, yb, xc, yc):
    return {"input": {"X": xb, "Y": yb}, "counterfactual_inputs": [{"X": xc, "Y": yc}]}

def _draw(rng):
    return rng.randint(0, 3), rng.randint(0, 3), rng.randint(0, 3), rng.randint(0, 3)

def make_datasets(model, n=10, seed=0):
    rng = random.Random(seed)
    return {"wide": [_ex(*_draw(rng)) for _ in range(n)]}

def random_pairs(model, n, seed=0):
    rng = random.Random(seed + 1)
    return [_ex(*_draw(rng)) for _ in range(n)]

DATASET_ROLES = {"wide": {"width": "wide", "split": "train"}}
"""


def test_main_loads_scaffolds_and_writes_report(tmp_path, capsys):
    hdir = tmp_path / "hypotheses"
    hdir.mkdir()
    (hdir / "models.py").write_text(_MODELS_PY)
    (hdir / "counterfactuals.py").write_text(_CF_PY)
    out_dir = tmp_path / "artifacts" / "develop_hypothesis" / "n8"

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "experiment_root": str(tmp_path / "artifacts"),
            "develop_hypothesis": {
                "_name_": "develop_hypothesis",
                "_output_dir": str(out_dir),
                "hypotheses_dir": str(hdir),
                "n": 8,
                "random_n": 20,
                "confound_threshold": 0.2,
            },
        }
    )

    try:
        report = dh_main.main(cfg)
    finally:
        # The analysis imports the scaffolds under canonical names; drop them so
        # the import doesn't leak into other tests in the session.
        sys.modules.pop("models", None)
        sys.modules.pop("counterfactuals", None)
        resolved = str(hdir.resolve())
        if resolved in sys.path:
            sys.path.remove(resolved)

    out = out_dir / "distinguishability.json"
    assert out.exists()
    saved = json.loads(out.read_text())

    # Structural invariants.
    assert set(saved["hypotheses"]) == {"total", "null", "all"}
    assert saved["targets"] == ["total"]
    assert saved["dataset_roles"] == {"wide": {"width": "wide", "split": "train"}}
    assert report["targets"] == ["total"]
    assert "wide" in capsys.readouterr().out

    # Value pins (exact — the symbolic distinguishability engine is deterministic
    # at this seed/scaffold). ``total`` ≡ ``all`` are always-confounded because
    # ``total`` fully determines ``raw_output``; ``null`` is its own singleton.
    # Regenerate the ground truth by rerunning the analysis on the inline scaffold
    # (seed=0, n=8, random_n=20) if the engine changes.
    expected = json.loads(_EXPECTED.read_text())
    assert saved["datasets"]["wide"]["size"] == expected["datasets"]["wide"]["size"]
    assert (
        saved["datasets"]["wide"]["per_target"]["total"]
        == expected["datasets"]["wide"]["per_target"]["total"]
    )
    assert saved["always_confounded"] == expected["always_confounded"]
    assert saved["singletons"] == expected["singletons"]
