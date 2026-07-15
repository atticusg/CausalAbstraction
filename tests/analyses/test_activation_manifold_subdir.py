"""Regression test for activation_manifold output-path routing (#171, sub-issue C5).

``main.py`` used to build the method subdir as ``f"{method}_s{smoothness}"``
directly, so a CLI override of ``activation_manifold._subdir=...`` changed the
resolved ``_subdir`` but not the actual write path — a silent no-op. The path
now flows through ``_manifold_subdir``, which reads ``analysis._subdir``.
"""

import pytest
from omegaconf import OmegaConf

from causalab.analyses.activation_manifold.main import (
    _manifold_subdir,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.unit


def _analysis(**overrides):
    """A minimal activation_manifold slice mirroring the shipped config."""
    base = {
        "_name_": "activation_manifold",
        "_subdir": "${.method}_s${.smoothness}",
        "method": "spline",
        "smoothness": 0.0,
        "embedding_shuffle_seed": None,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_default_matches_method_smoothness():
    """Default behavior is unchanged: subdir == resolved `${.method}_s${.smoothness}`."""
    assert _manifold_subdir(_analysis()) == "spline_s0.0"


def test_subdir_override_is_honored():
    """The C5 fix: a `_subdir=` override now routes output (was a no-op)."""
    assert _manifold_subdir(_analysis(_subdir="custom_run")) == "custom_run"


def test_shuffle_seed_suffix_preserved():
    """`embedding_shuffle_seed` still appends a `_shuf{seed}` suffix."""
    assert _manifold_subdir(_analysis(embedding_shuffle_seed=7)) == "spline_s0.0_shuf7"
