"""Unit tests for the exploration analysis helpers (pure logic, no model load).

The tiny-random end-to-end checks (one per mode) live in
``test_exploration_smoke.py``.
"""

from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from causalab.analyses.exploration import main as exp_main
from causalab.analyses.exploration._pipeline import DTYPE_MAP
from causalab.analyses.exploration.pair_trace import _read_manifest_row
from causalab.analyses.exploration.pca_critical_tokens import _load_inputs

pytestmark = pytest.mark.unit


def test_dtype_map_values_are_torch_attrs():
    import torch

    assert set(DTYPE_MAP) == {"bfloat16", "float16", "float32"}
    for value in DTYPE_MAP.values():
        assert hasattr(torch, value)


def test_main_rejects_unknown_mode(tmp_path):
    cfg = OmegaConf.create(
        {"exploration": {"mode": "bogus", "_output_dir": str(tmp_path)}}
    )
    with pytest.raises(ValueError, match="exploration.mode must be one of"):
        exp_main.main(cfg)


# --- pair mode: manifest reader ---


def test_read_manifest_row_selects_row_and_skips_blank_lines(tmp_path):
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        '{"base": "a", "counterfactual": "b"}\n'
        "\n"
        "   \n"
        '{"base": "c", "counterfactual": "d"}\n'
    )
    assert _read_manifest_row(str(manifest), 0)["base"] == "a"
    assert _read_manifest_row(str(manifest), 1)["base"] == "c"


def test_read_manifest_row_out_of_range_raises(tmp_path):
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text('{"base": "a", "counterfactual": "b"}\n')
    with pytest.raises(IndexError, match="out of range"):
        _read_manifest_row(str(manifest), 5)


# --- pca mode: inputs loader ---


def test_load_inputs_bare_strings(tmp_path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps(["2 + 2 =", "3 + 4 ="]))
    prompts, positions = _load_inputs(str(p))
    assert prompts == ["2 + 2 =", "3 + 4 ="]
    assert positions == [{}, {}]


def test_load_inputs_objects_with_positions(tmp_path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps([{"input": "2 + 2 =", "positions": {"op": 1}}]))
    prompts, positions = _load_inputs(str(p))
    assert prompts == ["2 + 2 ="]
    assert positions == [{"op": 1}]


def test_load_inputs_rejects_empty(tmp_path):
    p = tmp_path / "in.json"
    p.write_text("[]")
    with pytest.raises(ValueError, match="non-empty"):
        _load_inputs(str(p))


def test_load_inputs_rejects_bad_element(tmp_path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps([123]))
    with pytest.raises(ValueError, match="string or an object"):
        _load_inputs(str(p))
