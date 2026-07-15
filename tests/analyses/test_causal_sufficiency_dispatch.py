"""Analysis-level dispatch for the ``causal_sufficiency`` analysis.

``causalab/analyses/causal_sufficiency/main.py`` maps a runner config's
``restore.component_type`` string to the right target builder in
``_build_grid_targets`` (the restore grid). The builder layer is unit-tested in
``tests/neural/activations/test_targets.py``; this module pins the *dispatch* —
that each component string reaches its builder and produces units keyed the way
the scan / save path expects — plus the registered corruption kinds and
component types.
"""

from __future__ import annotations

import pytest

from causalab.analyses.causal_sufficiency.main import (
    VALID_COMPONENT_TYPES,
    VALID_METRICS,
    _build_grid_targets,
    _window_layers,
    _windowed_swept_targets,
)
from causalab.methods.causal_tracing import VALID_CORRUPTIONS
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


class TestRegisteredSurface:
    def test_corruption_kinds(self) -> None:
        assert set(VALID_CORRUPTIONS) == {"zero", "mean", "noise"}

    def test_component_types(self) -> None:
        assert set(VALID_COMPONENT_TYPES) == {
            "attention_head",
            "attention_output",
            "mlp",
            "residual",
        }

    def test_metric_kinds(self) -> None:
        assert set(VALID_METRICS) == {"logit_diff", "logit", "prob"}


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
        assert set(targets.keys()) == {(0, "last_token"), (1, "last_token")}
        assert detect_component_type_from_targets(targets) == detected

    def test_attention_head_keyed_by_layer_head(self, mock_tiny_lm: LMPipeline) -> None:
        span = _last_token(mock_tiny_lm)
        targets = _build_grid_targets(
            mock_tiny_lm, "attention_head", [0, 1], [0, 1], span
        )
        assert set(targets.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert detect_component_type_from_targets(targets) == "attention_head"


class TestWindowing:
    """``window`` widens each cell to a centered band of consecutive layers."""

    @pytest.mark.parametrize(
        "center,window,n_layers,expected",
        [
            (3, 1, 12, [3]),  # single state
            (5, 3, 12, [4, 5, 6]),  # centered band
            (0, 5, 12, [0, 1, 2]),  # clamped at the low end
            (11, 5, 12, [9, 10, 11]),  # clamped at the high end
        ],
    )
    def test_window_layers(
        self, center: int, window: int, n_layers: int, expected: list[int]
    ) -> None:
        assert _window_layers(center, window, n_layers) == expected

    def test_window_keys_by_center_and_collects_window_units(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        span = _last_token(mock_tiny_lm)
        n_layers = mock_tiny_lm.model.config.num_hidden_layers
        assert n_layers >= 2  # window=2 at center 0 needs a layer 1 to include
        targets = _windowed_swept_targets(
            mock_tiny_lm, "residual", [0, 1], [], span, window=2, n_layers=n_layers
        )
        # Keyed by center layer (heatmap axis unchanged).
        assert set(targets.keys()) == {(0, "last_token"), (1, "last_token")}
        # A window=2 cell at center 0 restores 2 layers' worth of units
        # ([0, 1]); the count matches _window_layers under clamping.
        assert len(targets[(0, "last_token")].flatten()) == len(
            _window_layers(0, 2, n_layers)
        )
        assert len(targets[(0, "last_token")].flatten()) == 2

    def test_window_1_is_single_unit_per_cell(self, mock_tiny_lm: LMPipeline) -> None:
        span = _last_token(mock_tiny_lm)
        n_layers = mock_tiny_lm.model.config.num_hidden_layers
        targets = _windowed_swept_targets(
            mock_tiny_lm, "residual", [0, 1], [], span, window=1, n_layers=n_layers
        )
        for target in targets.values():
            assert len(target.flatten()) == 1
