"""Tiny-random end-to-end smoke tests for the exploration analysis.

Runs each ``mode`` through ``main(cfg)`` on the random-weight Llama stub (CPU).
Weights are random, so these only assert "the wiring runs and the expected
artifacts appear" — no numerical claims. Pure-logic tests live in the sibling
``test_exploration_cli.py`` on the ``unit`` tier.
"""

from __future__ import annotations

import json
import math

import pytest
from omegaconf import OmegaConf

from causalab.analyses.exploration import main as exp_main
from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME

pytestmark = pytest.mark.smoke


def _cfg(tmp_path, mode: str, block: dict) -> OmegaConf:
    out_dir = tmp_path / "art" / "exploration" / mode
    return OmegaConf.create(
        {
            "experiment_root": str(tmp_path / "art"),
            "model": {
                "name": TINY_RANDOM_MODEL_NAME,
                "device": "cpu",
                "dtype": "float32",
            },
            "exploration": {
                "_name_": "exploration",
                "mode": mode,
                "_output_dir": str(out_dir),
                mode: block,
            },
        }
    )


def test_probe_mode_end_to_end(tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(["2 + 2 =", "7 + 1 ="]))
    cfg = _cfg(
        tmp_path, "probe", {"prompts": str(prompts), "out": None, "max_new_tokens": 3}
    )
    exp_main.main(cfg)
    out = tmp_path / "art" / "exploration" / "probe" / "outputs.json"
    assert out.exists()
    assert len(json.loads(out.read_text())) == 2


def test_logit_lens_mode_end_to_end(tmp_path):
    inputs = tmp_path / "in.json"
    inputs.write_text(json.dumps(["2 + 2 =", "7 + 1 ="]))
    cfg = _cfg(
        tmp_path,
        "logit_lens",
        {"inputs": str(inputs), "top_k": 10, "batch_size": 16, "figure_format": "png"},
    )
    exp_main.main(cfg)
    out_dir = tmp_path / "art" / "exploration" / "logit_lens"
    assert out_dir.exists()
    assert any(out_dir.iterdir())


def test_pair_mode_end_to_end(tmp_path):
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "token": "op",
                "input_idx": 0,
                "base": "2 + 2 =",
                "counterfactual": "2 + 3 =",
            }
        )
        + "\n"
    )
    cfg = _cfg(
        tmp_path,
        "pair",
        {
            "manifest": str(manifest),
            "index": 0,
            "figure_format": "png",
            "max_new_tokens": 3,
        },
    )
    exp_main.main(cfg)
    trace = (
        tmp_path
        / "art"
        / "exploration"
        / "pair"
        / "op"
        / "input_0"
        / "single_pair_trace.json"
    )
    assert trace.exists()


def test_knockout_mode_end_to_end(tmp_path):
    inputs = tmp_path / "knock_in.json"
    inputs.write_text(json.dumps(["2 + 2 =", "7 + 1 ="]))
    cfg = _cfg(
        tmp_path,
        "knockout",
        {
            "inputs": str(inputs),
            # span=all exercises multi-position MLP mean (the collect slice path).
            "span": "all",
            "essential_tokens": None,
            "components": ["attention_head", "mlp"],
            "ablation_modes": ["zero", "mean"],
            # tiny model has 2 layers: width 1 -> 2 cells, width 2 -> 1 band, 3+ skipped.
            "mlp_widths": [1, 2, 3],
            "head_band_widths": [1, 2, 3],
            "complement": True,  # exercise the sufficiency (keep-band) sweep too
            "layers": None,
            "heads": None,
            "batch_size": 16,
            "max_new_tokens": 3,
        },
    )
    out = exp_main.main(cfg)
    knock_dir = tmp_path / "art" / "exploration" / "knockout"
    assert (knock_dir / "metadata.json").exists()

    for ablation_mode in ("zero", "mean"):
        results = json.loads((knock_dir / ablation_mode / "results.json").read_text())
        cfg_model = out["results"][ablation_mode]
        assert results == cfg_model  # returned payload matches the persisted file

        # Both metrics are scored from one set of generations and described for
        # the web app's metric selector.
        assert set(results["metrics"]) == {"match_drop", "logit_diff"}

        assert results["complement"] is True

        heads = results["components"]["attention_head"]
        n_layers = len(heads["layers"])
        n_heads = len(heads["heads"])
        assert set(heads["metrics"]) == {"match_drop", "logit_diff"}
        for name in ("match_drop", "logit_diff"):
            grid = heads["metrics"][name]["drop_grid"]
            assert len(grid) == n_layers * n_heads
            assert all(math.isfinite(d) for d in grid.values())
            # Whole-sublayer bands: width 1 = one cell per layer (all heads in it),
            # width 2 = one sliding band. width 3 > 2 layers, skipped.
            h_widths = heads["metrics"][name]["widths"]
            assert set(h_widths) == {"1", "2"}
            assert len(h_widths["1"]) == n_layers
            assert len(h_widths["2"]) == n_layers - 1
        # match_drop is a fraction in [0, 1]; logit_diff is an unbounded logit gap.
        match_grid = heads["metrics"]["match_drop"]["drop_grid"]
        assert all(0.0 <= d <= 1.0 for d in match_grid.values())

        mlp = results["components"]["mlp"]
        assert set(mlp["metrics"]) == {"match_drop", "logit_diff"}
        for name in ("match_drop", "logit_diff"):
            widths = mlp["metrics"][name]["widths"]
            assert set(widths) == {"1", "2"}  # width 3 > 2 layers, skipped
            assert len(widths["1"]) == n_layers  # per-layer scan
            assert len(widths["2"]) == n_layers - 1  # sliding bands of 2
            assert all(math.isfinite(d) for s in widths.values() for d in s.values())
        match_widths = mlp["metrics"]["match_drop"]["widths"]
        assert all(0.0 <= d <= 1.0 for s in match_widths.values() for d in s.values())

        # complement=True adds the sufficiency sweep for both families. With 2
        # layers: width-1 complement keeps one layer / ablates the other (2 cells);
        # width-2 complement is the full band -> empty, so it carries no cells.
        for family in (heads, mlp):
            for name in ("match_drop", "logit_diff"):
                comp = family["metrics"][name]["complement_widths"]
                assert set(comp) == {"1", "2"}
                assert len(comp["1"]) == n_layers
                assert comp["2"] == {}  # full-width band has no complement
                assert all(math.isfinite(d) for d in comp["1"].values())


def test_pca_mode_end_to_end(tmp_path):
    rows = [
        {"input": f"{a} + {b} =", "positions": {"op": 0}}
        for a, b in [(2, 3), (4, 1), (5, 5), (1, 7), (3, 2), (6, 0)]
    ]
    inputs = tmp_path / "pca_inputs.json"
    inputs.write_text(json.dumps(rows))
    essential = tmp_path / "essential.json"
    essential.write_text(json.dumps([{"label": "op", "index": 0}]))
    cfg = _cfg(
        tmp_path,
        "pca",
        {
            "inputs": str(inputs),
            "essential_tokens": str(essential),
            "labels": None,
            "n_components": 2,
            "layers": None,
            "tokens": None,
            "batch_size": 16,
        },
    )
    exp_main.main(cfg)
    out_dir = tmp_path / "art" / "exploration" / "pca"
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "op" / "L0.safetensors").exists()
