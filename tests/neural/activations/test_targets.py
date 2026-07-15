"""Direct tests for ``causalab.neural.activations.targets``.

This module is the canonical builder layer between an
:class:`~causalab.neural.pipeline.LMPipeline` plus task metadata and the
``Dict[Tuple, InterchangeTarget]`` dicts that
:mod:`causalab.methods.interchange` runs interventions against.
``build_residual_stream_targets``, ``build_mlp_targets``, and
``build_attention_head_targets`` turn ``(layers, token_positions, mode)`` into
the keyed dicts consumed by ``runner.helpers.build_targets_for_grid`` and from
there by the ``locate``, ``subspace``, ``activation_manifold``,
``path_steering``, and ``pullback`` analyses.

The inverse helpers ``detect_component_type_from_targets`` /
``extract_grid_dimensions_from_targets`` recover the grid axes plotted by
``io.plots.score_heatmap``. If any of these return wrong keys or mis-shaped
units, interchange interventions register on the wrong sites and every
downstream score heatmap is silently mislabelled — these tests pin that
contract.

The happy-path tests run against the real tiny Llama stub via the session
``tiny_pipeline`` fixture in :mod:`tests.neural.conftest`. A small parametrized
``_FakeConfig`` is kept only to exercise the head-size / feature-size config
fallback branches (``n_head`` / ``num_attention_heads`` / ``num_heads``,
``n_inner`` / ``intermediate_size`` / ``hidden_size*4``) that the tiny
Llama config cannot reach because it always carries ``head_dim``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from causalab.neural.activations import (
    build_attention_head_targets,
    build_attention_output_targets,
    build_mlp_targets,
    build_residual_stream_targets,
    detect_component_type_from_targets,
    extract_grid_dimensions_from_targets,
)
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget


# --------------------------------------------------------------------------- #
#  Local helpers                                                              #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).float()


@pytest.fixture(scope="module")
def last_pos(tiny_pipeline: LMPipeline) -> TokenPosition:
    """One :class:`TokenPosition` keyed ``"last"``, returning index 0."""
    return TokenPosition(lambda _: [0], tiny_pipeline, id="last")


@pytest.fixture(scope="module")
def two_positions(tiny_pipeline: LMPipeline) -> list[TokenPosition]:
    """Two :class:`TokenPosition` objects keyed ``"pos_a"`` / ``"pos_b"``."""
    pos_a = TokenPosition(lambda _: [0], tiny_pipeline, id="pos_a")
    pos_b = TokenPosition(lambda _: [1], tiny_pipeline, id="pos_b")
    return [pos_a, pos_b]


def _fake_pipeline(**config_attrs: Any) -> LMPipeline:
    """Build a minimal :class:`LMPipeline` whose ``model.config`` has exactly
    ``config_attrs`` set.

    This is the *only* place we sidestep ``tiny_pipeline``: tiny-Llama's config
    always carries ``head_dim`` and ``intermediate_size``, so it can never
    exercise the ``n_head`` / ``num_attention_heads`` / ``num_heads`` / missing-all
    branches in :func:`build_attention_head_targets`, nor the ``n_inner`` /
    ``hidden_size*4`` fallback branches in :func:`build_mlp_targets`.
    """
    pipeline = LMPipeline.__new__(LMPipeline)
    pipeline.model = SimpleNamespace(
        config=SimpleNamespace(**config_attrs),
        device="cpu",
        dtype=torch.float32,
    )
    pipeline.tokenizer = SimpleNamespace(pad_token_id=0, padding_side="right")
    return pipeline


def _fake_token_position(pipeline: LMPipeline, *, id: str) -> TokenPosition:
    return TokenPosition(lambda _: [0], pipeline, id=id)


# =========================================================================== #
#  build_residual_stream_targets                                              #
# =========================================================================== #
class TestBuildResidualStreamTargetsUnit:
    """``build_residual_stream_targets`` keys (layer, pos.id) → ``InterchangeTarget``.

    These targets feed ``runner.helpers.build_targets_for_grid`` which the
    ``locate``, ``subspace``, ``activation_manifold``, and ``pullback``
    analyses consume; wrong keys here mis-route every downstream
    score-heatmap cell.
    """

    pytestmark = pytest.mark.unit

    def test_returns_dict_of_interchange_targets(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
            mode="one_target_per_unit",
        )
        assert isinstance(targets, dict)
        for v in targets.values():
            assert isinstance(v, InterchangeTarget)

    def test_per_unit_keys_are_layer_position_tuples(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        assert len(targets) == len(layers) * len(two_positions)
        for key in targets:
            assert isinstance(key, tuple) and len(key) == 2
            layer, pos_id = key
            assert layer in layers
            assert pos_id in {"pos_a", "pos_b"}

    def test_all_units_mode_collapses_to_single_key(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            token_positions=two_positions,
            mode="one_target_all_units",
        )
        assert list(targets.keys()) == [("all",)]

    def test_per_layer_mode_collapses_positions_per_layer(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_layer",
        )
        assert set(targets.keys()) == {(layer,) for layer in layers}

    def test_layer_minus_one_rewrites_to_zero_block_input(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        """``layer=-1`` always maps to ``layer=0, target_output=False``.

        This is the embedding/block_input rewrite documented in the source
        docstring; the override is intentional and silent — pin it.
        """
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=[-1],
            token_positions=[last_pos],
            mode="one_target_per_unit",
            target_output=True,  # caller asks for output …
        )
        unit = next(iter(targets.values())).flatten()[0]
        # … but the rewrite forces block_input at layer 0.
        assert unit.layer == 0
        assert unit.component_type == "block_input"

    def test_layer_minus_one_key_keeps_caller_layer(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        """The dict key keeps ``layer=-1`` even though the unit's ``.layer`` is 0.

        Downstream heatmap code reads the *key*, so the caller-facing axis is
        preserved while the unit's internal layer is rewritten to 0.
        """
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=[-1],
            token_positions=[last_pos],
            mode="one_target_per_unit",
        )
        assert list(targets.keys()) == [(-1, "last")]

    def test_invalid_mode_raises_value_error(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            build_residual_stream_targets(
                pipeline=tiny_pipeline,
                layers=[0],
                token_positions=[last_pos],
                mode="not_a_valid_mode",
            )


class TestBuildResidualStreamTargetsProperty:
    """Cardinality and round-trip invariants on residual-stream targets."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize(
        "mode,expected_n",
        [
            ("one_target_per_unit", 4),  # 2 layers * 2 positions
            ("one_target_per_layer", 2),  # 2 layers
            ("one_target_all_units", 1),  # collapsed
        ],
    )
    def test_cardinality_invariant_across_modes(
        self,
        tiny_pipeline: LMPipeline,
        two_positions: list[TokenPosition],
        mode: str,
        expected_n: int,
    ) -> None:
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            token_positions=two_positions,
            mode=mode,
        )
        assert len(targets) == expected_n

    def test_detect_extract_round_trip_recovers_layers(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        component_type = detect_component_type_from_targets(targets)
        dims = extract_grid_dimensions_from_targets(component_type, targets)
        assert component_type == "residual_stream"
        assert dims["layers"] == layers

    def test_determinism_across_repeated_calls(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        """Building twice with the same args yields the same key set."""
        a = build_residual_stream_targets(
            pipeline=tiny_pipeline, layers=[0, 1], token_positions=[last_pos]
        )
        b = build_residual_stream_targets(
            pipeline=tiny_pipeline, layers=[0, 1], token_positions=[last_pos]
        )
        assert set(a.keys()) == set(b.keys())


# =========================================================================== #
#  build_mlp_targets                                                          #
# =========================================================================== #
class TestBuildMlpTargetsUnit:
    """``build_mlp_targets`` keys (layer, pos.id) → MLP-unit ``InterchangeTarget``.

    The ``location`` argument selects between ``mlp_input`` / ``mlp_output``
    (feature_size = hidden_size) and ``mlp_activation`` (feature_size from
    ``n_inner`` / ``intermediate_size`` / ``hidden_size*4`` fallback chain).
    Wrong shape here makes the featurizer downstream allocate a mis-sized
    subspace.
    """

    pytestmark = pytest.mark.unit

    def test_returns_dict_of_interchange_targets(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        targets = build_mlp_targets(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
            mode="one_target_per_unit",
        )
        assert isinstance(targets, dict)
        for v in targets.values():
            assert isinstance(v, InterchangeTarget)

    def test_per_unit_keys_are_layer_position_tuples(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        targets = build_mlp_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        assert len(targets) == len(layers) * len(two_positions)
        for layer, pos_id in targets:
            assert layer in layers
            assert pos_id in {"pos_a", "pos_b"}

    def test_mlp_output_uses_hidden_size(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        targets = build_mlp_targets(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
            location="mlp_output",
        )
        unit = next(iter(targets.values())).flatten()[0]
        assert unit.shape == (tiny_pipeline.model.config.hidden_size,)

    def test_mlp_activation_prefers_n_inner_when_set(self) -> None:
        """If ``config.n_inner`` is present and non-None it wins over
        ``intermediate_size``."""
        pipeline = _fake_pipeline(hidden_size=16, n_inner=42, intermediate_size=64)
        pos = _fake_token_position(pipeline, id="last")
        targets = build_mlp_targets(
            pipeline=pipeline,
            layers=[0],
            token_positions=[pos],
            location="mlp_activation",
        )
        unit = next(iter(targets.values())).flatten()[0]
        assert unit.shape == (42,)

    def test_mlp_activation_falls_back_to_intermediate_size(self) -> None:
        """When ``n_inner`` is absent (e.g. Llama), ``intermediate_size`` wins."""
        pipeline = _fake_pipeline(hidden_size=16, intermediate_size=64)
        pos = _fake_token_position(pipeline, id="last")
        targets = build_mlp_targets(
            pipeline=pipeline,
            layers=[0],
            token_positions=[pos],
            location="mlp_activation",
        )
        unit = next(iter(targets.values())).flatten()[0]
        assert unit.shape == (64,)

    def test_mlp_activation_falls_back_to_hidden_size_times_four(self) -> None:
        """Final fallback when neither ``n_inner`` nor ``intermediate_size``
        are on the config."""
        pipeline = _fake_pipeline(hidden_size=16)
        pos = _fake_token_position(pipeline, id="last")
        targets = build_mlp_targets(
            pipeline=pipeline,
            layers=[0],
            token_positions=[pos],
            location="mlp_activation",
        )
        unit = next(iter(targets.values())).flatten()[0]
        assert unit.shape == (64,)

    def test_mlp_activation_with_none_n_inner_falls_back_to_intermediate(
        self,
    ) -> None:
        """Spec: ``n_inner`` *present but None* should not be picked — fall
        through to ``intermediate_size``."""
        pipeline = _fake_pipeline(hidden_size=16, n_inner=None, intermediate_size=64)
        pos = _fake_token_position(pipeline, id="last")
        targets = build_mlp_targets(
            pipeline=pipeline,
            layers=[0],
            token_positions=[pos],
            location="mlp_activation",
        )
        unit = next(iter(targets.values())).flatten()[0]
        assert unit.shape == (64,)

    def test_invalid_mode_raises_value_error(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            build_mlp_targets(
                pipeline=tiny_pipeline,
                layers=[0],
                token_positions=[last_pos],
                mode="not_a_valid_mode",
            )


class TestBuildMlpTargetsProperty:
    """Cardinality + round-trip invariants on MLP targets."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize(
        "mode,expected_n",
        [
            ("one_target_per_unit", 4),
            ("one_target_per_layer", 2),
            ("one_target_all_units", 1),
        ],
    )
    def test_cardinality_invariant_across_modes(
        self,
        tiny_pipeline: LMPipeline,
        two_positions: list[TokenPosition],
        mode: str,
        expected_n: int,
    ) -> None:
        targets = build_mlp_targets(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            token_positions=two_positions,
            mode=mode,
        )
        assert len(targets) == expected_n

    def test_detect_extract_round_trip_recovers_layers(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        targets = build_mlp_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        component_type = detect_component_type_from_targets(targets)
        dims = extract_grid_dimensions_from_targets(component_type, targets)
        assert component_type == "mlp"
        assert dims["layers"] == layers


# =========================================================================== #
#  build_attention_output_targets                                             #
# =========================================================================== #
class TestBuildAttentionOutputTargetsUnit:
    """``build_attention_output_targets`` keys (layer, pos.id) → whole-sublayer
    ``InterchangeTarget``.

    Mirrors :func:`build_mlp_targets` in shape and grouping but targets pyvene's
    ``attention_output`` component (all heads jointly). The merged output is
    ``hidden_size``-wide, so the unit shape must be ``(hidden_size,)`` regardless
    of head count — getting this wrong mis-sizes the ablation reference vector.
    """

    pytestmark = pytest.mark.unit

    def test_returns_dict_of_interchange_targets(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        targets = build_attention_output_targets(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
            mode="one_target_per_unit",
        )
        assert isinstance(targets, dict)
        for v in targets.values():
            assert isinstance(v, InterchangeTarget)

    def test_per_unit_keys_are_layer_position_tuples(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        targets = build_attention_output_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        assert len(targets) == len(layers) * len(two_positions)
        for layer, pos_id in targets:
            assert layer in layers
            assert pos_id in {"pos_a", "pos_b"}

    def test_unit_shape_is_hidden_size(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        targets = build_attention_output_targets(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
        )
        unit = next(iter(targets.values())).flatten()[0]
        assert unit.shape == (tiny_pipeline.model.config.hidden_size,)
        assert unit.component_type == "attention_output"

    def test_invalid_mode_raises_value_error(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            build_attention_output_targets(
                pipeline=tiny_pipeline,
                layers=[0],
                token_positions=[last_pos],
                mode="not_a_valid_mode",
            )


class TestBuildAttentionOutputTargetsProperty:
    """Cardinality + detect/extract round-trip on attention-output targets."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize(
        "mode,expected_n",
        [
            ("one_target_per_unit", 4),  # 2 layers * 2 positions
            ("one_target_per_layer", 2),
            ("one_target_all_units", 1),
        ],
    )
    def test_cardinality_invariant_across_modes(
        self,
        tiny_pipeline: LMPipeline,
        two_positions: list[TokenPosition],
        mode: str,
        expected_n: int,
    ) -> None:
        targets = build_attention_output_targets(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            token_positions=two_positions,
            mode=mode,
        )
        assert len(targets) == expected_n

    def test_detect_extract_round_trip_recovers_layers(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        targets = build_attention_output_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        component_type = detect_component_type_from_targets(targets)
        dims = extract_grid_dimensions_from_targets(component_type, targets)
        assert component_type == "attention_output"
        assert dims["layers"] == layers


# =========================================================================== #
#  build_attention_head_targets                                               #
# =========================================================================== #
class TestBuildAttentionHeadTargetsUnit:
    """``build_attention_head_targets`` keys (layer, head) → ``InterchangeTarget``.

    Head-size resolution walks the config in this priority order:
    ``head_dim`` (Llama-style) → ``n_head`` → ``num_attention_heads`` →
    ``num_heads``; missing all four raises ``ValueError``. Getting this
    wrong allocates the wrong-sized AttentionHead featurizer.
    """

    pytestmark = pytest.mark.unit

    def test_returns_dict_of_interchange_targets(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        targets = build_attention_head_targets(
            pipeline=tiny_pipeline,
            layers=[0],
            heads=[0, 1],
            token_position=last_pos,
            mode="one_target_per_unit",
        )
        assert isinstance(targets, dict)
        for v in targets.values():
            assert isinstance(v, InterchangeTarget)

    def test_per_unit_keys_are_layer_head_tuples(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        layers = [0, 1]
        heads = [0, 1, 2]
        targets = build_attention_head_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            heads=heads,
            token_position=last_pos,
            mode="one_target_per_unit",
        )
        assert len(targets) == len(layers) * len(heads)
        for layer, head in targets:
            assert layer in layers
            assert head in heads

    def test_all_units_mode_collapses_to_single_key(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        targets = build_attention_head_targets(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            heads=[0, 1],
            token_position=last_pos,
            mode="one_target_all_units",
        )
        assert list(targets.keys()) == [("all",)]

    def test_head_dim_attr_wins(self) -> None:
        """When ``config.head_dim`` is present it short-circuits the
        ``hidden_size // num_heads`` arithmetic."""
        pipeline = _fake_pipeline(hidden_size=64, head_dim=7, n_head=4)
        pos = _fake_token_position(pipeline, id="last")
        targets = build_attention_head_targets(
            pipeline=pipeline,
            layers=[0],
            heads=[0],
            token_position=pos,
        )
        unit = next(iter(targets.values())).flatten()[0]
        # head_dim=7 wins over hidden_size//n_head = 16.
        assert unit.shape == (7,)

    @pytest.mark.parametrize(
        "attr_name",
        ["n_head", "num_attention_heads", "num_heads"],
    )
    def test_head_count_attr_chain(self, attr_name: str) -> None:
        """In the no-``head_dim`` branch, head_size = hidden_size // n_heads
        and the n_heads attribute name is resolved by the documented
        priority chain."""
        pipeline = _fake_pipeline(hidden_size=32, **{attr_name: 4})
        pos = _fake_token_position(pipeline, id="last")
        targets = build_attention_head_targets(
            pipeline=pipeline,
            layers=[0],
            heads=[0],
            token_position=pos,
        )
        unit = next(iter(targets.values())).flatten()[0]
        assert unit.shape == (8,)  # 32 // 4

    def test_missing_all_head_attrs_raises_value_error(self) -> None:
        pipeline = _fake_pipeline(hidden_size=32)  # no head_dim, no n_head etc.
        pos = _fake_token_position(pipeline, id="last")
        with pytest.raises(ValueError, match="Could not determine number of heads"):
            build_attention_head_targets(
                pipeline=pipeline,
                layers=[0],
                heads=[0],
                token_position=pos,
            )

    def test_invalid_mode_raises_value_error(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            build_attention_head_targets(
                pipeline=tiny_pipeline,
                layers=[0],
                heads=[0],
                token_position=last_pos,
                mode="not_a_valid_mode",
            )


class TestBuildAttentionHeadTargetsProperty:
    """Cardinality + round-trip invariants on attention-head targets."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize(
        "mode,expected_n",
        [
            ("one_target_per_unit", 6),  # 2 layers * 3 heads
            ("one_target_per_layer", 2),
            ("one_target_all_units", 1),
        ],
    )
    def test_cardinality_invariant_across_modes(
        self,
        tiny_pipeline: LMPipeline,
        last_pos: TokenPosition,
        mode: str,
        expected_n: int,
    ) -> None:
        targets = build_attention_head_targets(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            heads=[0, 1, 2],
            token_position=last_pos,
            mode=mode,
        )
        assert len(targets) == expected_n

    def test_detect_extract_round_trip_recovers_layers_and_heads(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        layers = [0, 1]
        heads = [0, 1, 2]
        targets = build_attention_head_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            heads=heads,
            token_position=last_pos,
            mode="one_target_per_unit",
        )
        component_type = detect_component_type_from_targets(targets)
        dims = extract_grid_dimensions_from_targets(component_type, targets)
        assert component_type == "attention_head"
        assert dims["layers"] == layers
        assert dims["heads"] == heads


# =========================================================================== #
#  detect_component_type_from_targets                                          #
# =========================================================================== #
class TestDetectComponentTypeFromTargetsUnit:
    """``detect_component_type_from_targets`` dispatches downstream
    ``extract_grid_dimensions_from_targets`` calls — mis-detection routes the
    plot to the wrong axis layout."""

    pytestmark = pytest.mark.unit

    def test_detects_attention_head(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        targets = build_attention_head_targets(
            pipeline=tiny_pipeline,
            layers=[0],
            heads=[0],
            token_position=last_pos,
        )
        assert detect_component_type_from_targets(targets) == "attention_head"

    def test_detects_residual_stream(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
        )
        assert detect_component_type_from_targets(targets) == "residual_stream"

    def test_detects_mlp(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        targets = build_mlp_targets(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
        )
        assert detect_component_type_from_targets(targets) == "mlp"

    def test_empty_dict_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            detect_component_type_from_targets({})


class TestDetectComponentTypeFromTargetsProperty:
    """Invariance: the detected component_type does not depend on grouping
    ``mode``."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize(
        "mode",
        ["one_target_per_unit", "one_target_per_layer", "one_target_all_units"],
    )
    def test_residual_stream_detection_invariant_under_mode(
        self,
        tiny_pipeline: LMPipeline,
        two_positions: list[TokenPosition],
        mode: str,
    ) -> None:
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            token_positions=two_positions,
            mode=mode,
        )
        assert detect_component_type_from_targets(targets) == "residual_stream"

    @pytest.mark.parametrize(
        "mode",
        ["one_target_per_unit", "one_target_per_layer", "one_target_all_units"],
    )
    def test_mlp_detection_invariant_under_mode(
        self,
        tiny_pipeline: LMPipeline,
        two_positions: list[TokenPosition],
        mode: str,
    ) -> None:
        targets = build_mlp_targets(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            token_positions=two_positions,
            mode=mode,
        )
        assert detect_component_type_from_targets(targets) == "mlp"

    @pytest.mark.parametrize(
        "mode",
        ["one_target_per_unit", "one_target_per_layer", "one_target_all_units"],
    )
    def test_attention_head_detection_invariant_under_mode(
        self,
        tiny_pipeline: LMPipeline,
        last_pos: TokenPosition,
        mode: str,
    ) -> None:
        targets = build_attention_head_targets(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            heads=[0, 1],
            token_position=last_pos,
            mode=mode,
        )
        assert detect_component_type_from_targets(targets) == "attention_head"


# =========================================================================== #
#  extract_grid_dimensions_from_targets                                       #
# =========================================================================== #
class TestExtractGridDimensionsFromTargetsUnit:
    """``extract_grid_dimensions_from_targets`` recovers the (layers, heads)
    or (layers, token_position_ids) axes consumed by
    :func:`causalab.io.plots.score_heatmap`."""

    pytestmark = pytest.mark.unit

    def test_attention_head_axes_are_sorted(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        """For ``attention_head`` keys, the dict is a sorted-set over the
        ``(layer, head)`` tuples."""
        # Build with unsorted layers/heads to confirm the function sorts.
        layers = [1, 0]
        heads = [2, 0, 1]
        targets = build_attention_head_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            heads=heads,
            token_position=last_pos,
            mode="one_target_per_unit",
        )
        dims = extract_grid_dimensions_from_targets("attention_head", targets)
        assert dims["layers"] == [0, 1]
        assert dims["heads"] == [0, 1, 2]

    def test_residual_stream_preserves_position_insertion_order(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """For residual-stream/MLP targets, token_position_ids keep first-seen
        insertion order — ``score_heatmap`` relies on this axis order."""
        pos_b = TokenPosition(lambda _: [1], tiny_pipeline, id="zzz_pos_b")
        pos_a = TokenPosition(lambda _: [0], tiny_pipeline, id="aaa_pos_a")
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=[0],
            # Pass alphabetically-late id first; round-trip must preserve order
            # (not sort).
            token_positions=[pos_b, pos_a],
            mode="one_target_per_unit",
        )
        dims = extract_grid_dimensions_from_targets("residual_stream", targets)
        assert dims["token_position_ids"] == ["zzz_pos_b", "aaa_pos_a"]

    def test_mlp_axis_names_are_layers_and_token_position_ids(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        targets = build_mlp_targets(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        dims = extract_grid_dimensions_from_targets("mlp", targets)
        assert set(dims.keys()) == {"layers", "token_position_ids"}

    def test_raises_on_unknown_component_type(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        """Unknown ``component_type`` must raise ``ValueError`` instead of
        silently falling through to the position-id path (which would
        mislabel heatmap axes)."""
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
            mode="one_target_per_unit",
        )
        with pytest.raises(ValueError, match="Unknown component_type"):
            extract_grid_dimensions_from_targets("not_a_real_component", targets)


class TestExtractGridDimensionsFromTargetsProperty:
    """Round-trip invariants: ``extract(detect(build(...)), build(...))``
    recovers the input layer / head / position-id lists across all three
    builders."""

    pytestmark = pytest.mark.property

    def test_round_trip_residual_stream_recovers_inputs(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        targets = build_residual_stream_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        component_type = detect_component_type_from_targets(targets)
        dims = extract_grid_dimensions_from_targets(component_type, targets)
        assert dims["layers"] == layers
        # Insertion order from the two_positions fixture.
        assert dims["token_position_ids"] == ["pos_a", "pos_b"]

    def test_round_trip_mlp_recovers_inputs(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        targets = build_mlp_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        component_type = detect_component_type_from_targets(targets)
        dims = extract_grid_dimensions_from_targets(component_type, targets)
        assert dims["layers"] == layers
        assert dims["token_position_ids"] == ["pos_a", "pos_b"]

    def test_round_trip_attention_head_recovers_inputs(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        layers = [0, 1]
        heads = [0, 1, 2]
        targets = build_attention_head_targets(
            pipeline=tiny_pipeline,
            layers=layers,
            heads=heads,
            token_position=last_pos,
            mode="one_target_per_unit",
        )
        component_type = detect_component_type_from_targets(targets)
        dims = extract_grid_dimensions_from_targets(component_type, targets)
        assert dims["layers"] == layers
        assert dims["heads"] == heads
