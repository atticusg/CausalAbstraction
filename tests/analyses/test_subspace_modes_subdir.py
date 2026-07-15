"""Regression test for subspace per-mode output routing (#171, sub-issue C6).

``base_out_dir = analysis._output_dir`` resolves the analysis-level ``_subdir``
(``${.method}_k${.k_features}``) once, so a ``modes:`` sweep that overrode only
``k_features`` wrote every entry to ``subspace/pca_k64/...`` — the k=16 result
was overwritten by the k=64 run. ``_mode_out_dir`` now re-resolves ``_subdir``
from each mode's merged config, while staying backward-compatible with the
component-mode ``name:`` pattern (which doesn't touch ``_subdir``).
"""

import os
from typing import cast

import pytest
from omegaconf import DictConfig, OmegaConf

from causalab.analyses.subspace.main import (
    _mode_out_dir,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.unit


def _analysis():
    """A subspace slice mirroring the shipped config, attached so `_output_dir`
    resolves `${experiment_root}`."""
    cfg = OmegaConf.create(
        {
            "experiment_root": "/exp",
            "subspace": {
                "_name_": "subspace",
                "_subdir": "${.method}_k${.k_features}",
                "_output_dir": "${experiment_root}/subspace/${._subdir}",
                "method": "pca",
                "k_features": 64,
                "component_type": "residual_stream",
            },
        }
    )
    return cfg.subspace


def _merge_mode(analysis, mode_dict):
    """Replicate main()'s per-mode merge + mode_name derivation."""
    mode_name = str(mode_dict.get("name") or mode_dict.get("component_type") or "mode")
    mode_analysis = cast(
        DictConfig,
        OmegaConf.merge(
            analysis,
            OmegaConf.create({k: v for k, v in mode_dict.items() if k != "name"}),
        ),
    )
    return mode_analysis, mode_name


def test_k_sweep_modes_get_distinct_dirs():
    """The C6 fix: modes differing only by k_features no longer collide."""
    analysis = _analysis()
    base = analysis._output_dir  # /exp/subspace/pca_k64
    dirs = []
    for mode_dict in ({"k_features": 16}, {"k_features": 64}):
        ma, mn = _merge_mode(analysis, mode_dict)
        dirs.append(_mode_out_dir(base, ma, mn, "answer"))
    assert dirs[0] != dirs[1]
    assert dirs[0] == os.path.join("/exp", "subspace", "pca_k16", "mode", "answer")
    assert dirs[1] == os.path.join("/exp", "subspace", "pca_k64", "mode", "answer")


def test_named_component_modes_backward_compatible():
    """Component modes (same k, distinct `name:`) keep `base_out_dir/name[/tv]`."""
    analysis = _analysis()
    base = analysis._output_dir  # /exp/subspace/pca_k64
    out = {}
    for mode_dict in (
        {"name": "dbm_attention", "component_type": "attention_head"},
        {"name": "dbm_mlp", "component_type": "mlp"},
    ):
        ma, mn = _merge_mode(analysis, mode_dict)
        out[mn] = _mode_out_dir(base, ma, mn, "answer")
    assert out["dbm_attention"] == os.path.join(base, "dbm_attention", "answer")
    assert out["dbm_mlp"] == os.path.join(base, "dbm_mlp", "answer")


def test_no_target_variable_omits_tv_segment():
    """`tv=None` drops the trailing target-variable segment."""
    analysis = _analysis()
    base = analysis._output_dir
    ma, mn = _merge_mode(analysis, {"name": "solo"})
    assert _mode_out_dir(base, ma, mn, None) == os.path.join(base, "solo")
