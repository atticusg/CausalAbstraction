"""Regression test for runner directive validation (#171, sub-issue C2).

A config author who adds an unrecognized ``_..._`` directive (e.g. a
``_runner_target_`` key, thinking the runner needs an explicit dispatch path)
used to have it silently ignored — the runner dispatches by the bare ``_name_``
field alone. The dropped intent produced no signal. ``_check_known_directives``
now fails fast, mirroring the ``_name_``-mismatch ``RuntimeError`` in the same
dispatch loop.
"""

import pytest
from omegaconf import OmegaConf

from causalab.runner.run_exp import (
    _RESERVED_DIRECTIVES,  # pyright: ignore[reportPrivateUsage]
    _check_known_directives,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.unit


def test_unknown_directive_raises():
    """An unrecognized ``_..._`` key is rejected, and the message names it."""
    step_cfg = OmegaConf.create(
        {
            "_name_": "subspace",
            "_subdir": "pca_k64",
            "_output_dir": "/exp/subspace/pca_k64",
            "_runner_target_": "analyses.or_slant_diagnostics.main.main",
            "method": "pca",
        }
    )
    with pytest.raises(ValueError) as exc:
        _check_known_directives("subspace", step_cfg)
    assert "_runner_target_" in str(exc.value)


def test_clean_slice_passes():
    """The three recognized directives (plus ordinary fields) do not raise."""
    step_cfg = OmegaConf.create(
        {
            "_name_": "subspace",
            "_subdir": "pca_k64",
            "_output_dir": "/exp/subspace/pca_k64",
            "method": "pca",
            "k_features": 64,
        }
    )
    _check_known_directives("subspace", step_cfg)  # no raise


def test_reserved_set_is_the_documented_three():
    """Guards docs/CODEBASE.md §2: exactly these three directives are honored."""
    assert _RESERVED_DIRECTIVES == {"_name_", "_subdir", "_output_dir"}
