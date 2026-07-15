"""Analysis-level component dispatch for the ``ablation`` analysis.

``causalab/analyses/ablation/main.py`` maps a runner config's
``component_type`` string to the right target builder in two places:
``_build_grid_targets`` (the scan grid) and ``_combo_units`` (explicit joint
sets / the complement path). The builder layer is unit-tested in
``tests/neural/activations/test_targets.py``; this module pins the *dispatch* —
that ``"attention_output"`` and ``"residual"`` reach their builders and produce
units keyed the way the scan / save path expects. A mis-dispatch here would make
``component_type=attention_output`` silently fall through to MLP units.
"""

from __future__ import annotations

import pytest

from causalab.analyses.ablation.main import (
    VALID_COMPONENT_TYPES,
    _build_grid_targets,
    _combo_units,
)
from causalab.neural.activations.targets import detect_component_type_from_targets
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition, get_last_token_index

pytestmark = pytest.mark.unit


def _last_token(pipeline: LMPipeline) -> TokenPosition:
    return TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline),
        pipeline,
        id="last_token",
    )


class TestValidComponentTypes:
    def test_new_component_types_are_registered(self) -> None:
        # The widened surface from feedback item 4: attention_output + residual
        # join the original head/mlp grid.
        assert set(VALID_COMPONENT_TYPES) == {
            "attention_head",
            "attention_output",
            "mlp",
            "residual",
        }


class TestBuildGridTargetsDispatch:
    """``_build_grid_targets`` routes each config string to the right builder."""

    @pytest.mark.parametrize(
        "component_type,detected",
        [
            ("attention_output", "attention_output"),
            ("residual", "residual_stream"),
            ("mlp", "mlp"),
        ],
    )
    def test_layer_by_span_grid_keyed_and_detected(
        self, mock_tiny_lm: LMPipeline, component_type: str, detected: str
    ) -> None:
        layers = [0, 1]
        span = _last_token(mock_tiny_lm)
        targets = _build_grid_targets(mock_tiny_lm, component_type, layers, [], span)
        # Layer × span grid keyed (layer, span.id) — the shape _save_results plots.
        assert set(targets.keys()) == {(0, "last_token"), (1, "last_token")}
        assert detect_component_type_from_targets(targets) == detected

    def test_attention_head_still_keyed_by_layer_head(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        span = _last_token(mock_tiny_lm)
        targets = _build_grid_targets(mock_tiny_lm, "attention_head", [0], [0, 1], span)
        assert set(targets.keys()) == {(0, 0), (0, 1)}
        assert detect_component_type_from_targets(targets) == "attention_head"


class TestComboUnitsDispatch:
    """``_combo_units`` materializes joint-ablation units for ``[layer, ...]``
    combos of the non-head component types (the path used by ``combos`` and the
    ``complement_keep`` sufficiency check)."""

    @pytest.mark.parametrize("component_type", ["attention_output", "residual", "mlp"])
    def test_layer_combo_yields_one_unit_per_layer(
        self, mock_tiny_lm: LMPipeline, component_type: str
    ) -> None:
        span = _last_token(mock_tiny_lm)
        units = _combo_units(mock_tiny_lm, component_type, [0, 1], span)
        assert len(units) == 2
        # Units carry distinct ids per layer (so reference vectors don't collide).
        assert len({u.id for u in units}) == 2

    def test_attention_head_combo_uses_layer_head_pairs(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        span = _last_token(mock_tiny_lm)
        units = _combo_units(mock_tiny_lm, "attention_head", [[0, 0], [0, 1]], span)
        assert len(units) == 2
        assert len({u.id for u in units}) == 2
