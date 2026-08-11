"""Direct tests for ``causalab.neural.activations.site_grids`` (WU2, #504).

The spec-native successor of the legacy ``targets.py`` builders (deleted in
the WU6 sweep, #508): ``build_residual_stream_sites`` / ``build_mlp_sites`` /
``build_attention_output_sites`` / ``build_attention_head_sites`` turn
``(layers, positions/heads, mode)`` into ``dict[key_tuple, list[list[SiteSpec]]]``
grids, and ``grid_component`` / ``extract_grid_dimensions_from_targets``
recover the axes plotted by ``io.plots.score_heatmap``.

These tests pin the direct contract: key tuples, grouping shapes, config
width probing, the ``layer=-1`` embeddings rewrite, spec-key uniqueness.
The legacy-equivalence gate that compared every builder × grouping mode
cell-by-cell against the legacy ``targets.py`` output went with the legacy
module in the WU6 sweep — the WU4/WU5 caller migrations swapped builders
against that proven-identical grid while both surfaces existed.

The happy-path tests run against the real tiny Llama stub via the session
``tiny_pipeline`` fixture in :mod:`tests.neural.conftest`. A small
``_fake_pipeline`` is kept only to exercise the head-size / feature-size
config fallback branches (``n_head`` / ``num_attention_heads`` / ``num_heads``,
``n_inner`` / ``intermediate_size`` / ``hidden_size*4``) that the tiny Llama
config cannot reach because it always carries ``head_dim``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from causalab.neural.activations.site_grids import (
    build_attention_head_sites,
    build_attention_output_sites,
    build_mlp_sites,
    build_residual_stream_sites,
    extract_grid_dimensions_from_targets,
    grid_component,
)
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.head_view import HeadSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec
from causalab.neural.token_positions import TokenPosition

_MODES = ["one_target_per_unit", "one_target_per_layer", "one_target_all_units"]


# --------------------------------------------------------------------------- #
#  Local helpers                                                              #
# --------------------------------------------------------------------------- #
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
    ``config_attrs`` set — the only place we sidestep ``tiny_pipeline``, to
    reach the config-probing fallback branches (see the module docstring)."""
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


def _all_specs(grid: dict[tuple[Any, ...], list[list[SiteSpec]]]) -> list[SiteSpec]:
    return [spec for groups in grid.values() for group in groups for spec in group]


def _single_spec(grid: dict[tuple[Any, ...], list[list[SiteSpec]]]) -> SiteSpec:
    return next(iter(grid.values()))[0][0]


# =========================================================================== #
#  build_residual_stream_sites                                                #
# =========================================================================== #
class TestBuildResidualStreamSitesUnit:
    """``build_residual_stream_sites`` keys (layer, pos.id) → ``[[SiteSpec]]``.

    Same key/grouping contract as the legacy builder — wrong keys here
    mis-route every downstream score-heatmap cell.
    """

    pytestmark = pytest.mark.unit

    def test_returns_dict_of_spec_groups(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
            mode="one_target_per_unit",
        )
        assert isinstance(grid, dict)
        for groups in grid.values():
            assert isinstance(groups, list)
            for group in groups:
                assert isinstance(group, list)
                for spec in group:
                    assert isinstance(spec, SiteSpec)

    def test_per_unit_keys_are_layer_position_tuples(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        assert len(grid) == len(layers) * len(two_positions)
        for key in grid:
            assert isinstance(key, tuple) and len(key) == 2
            layer, pos_id = key
            assert layer in layers
            assert pos_id in {"pos_a", "pos_b"}

    def test_all_units_mode_collapses_to_single_key(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            token_positions=two_positions,
            mode="one_target_all_units",
        )
        assert list(grid.keys()) == [("all",)]
        # Single fused group holding every spec.
        assert len(grid[("all",)]) == 1
        assert len(grid[("all",)][0]) == 4

    def test_per_layer_mode_collapses_positions_per_layer(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_layer",
        )
        assert set(grid.keys()) == {(layer,) for layer in layers}
        for groups in grid.values():
            assert len(groups) == 1  # one group per layer …
            assert len(groups[0]) == 2  # … holding both positions

    def test_spec_fields(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        """Component, layer, width, and positions land on the spec itself."""
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=[1],
            token_positions=[last_pos],
        )
        spec = _single_spec(grid)
        assert spec.fsite.site == Site("block_output", 1)
        assert spec.width == tiny_pipeline.model.config.hidden_size
        assert spec.positions is last_pos

    def test_target_output_false_targets_block_input(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=[1],
            token_positions=[last_pos],
            target_output=False,
        )
        assert _single_spec(grid).fsite.site == Site("block_input", 1)

    def test_layer_minus_one_rewrites_to_zero_block_input(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        """``layer=-1`` always maps to ``Site("block_input", 0)``.

        The embedding rewrite from the legacy builder: the engine's ``Site``
        requires ``layer >= 0``, so ``-1`` lives only in the key tuple. The
        override is intentional and silent even when the caller asks for
        output — pin it.
        """
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=[-1],
            token_positions=[last_pos],
            mode="one_target_per_unit",
            target_output=True,  # caller asks for output …
        )
        spec = _single_spec(grid)
        # … but the rewrite forces block_input at layer 0.
        assert spec.fsite.site == Site("block_input", 0)

    def test_layer_minus_one_key_keeps_caller_layer(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        """The dict key keeps ``layer=-1`` even though the engine site sits at
        layer 0 — downstream heatmap code reads the *key*."""
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=[-1],
            token_positions=[last_pos],
            mode="one_target_per_unit",
        )
        assert list(grid.keys()) == [(-1, "last")]

    def test_duplicate_position_ids_raise(self, tiny_pipeline: LMPipeline) -> None:
        """Two positions sharing an ``id`` collide grid keys (and spec keys) —
        refused loudly instead of the legacy silent overwrite."""
        pos_1 = TokenPosition(lambda _: [0], tiny_pipeline, id="same")
        pos_2 = TokenPosition(lambda _: [1], tiny_pipeline, id="same")
        with pytest.raises(ValueError, match="duplicate grid keys"):
            build_residual_stream_sites(
                pipeline=tiny_pipeline,
                layers=[0],
                token_positions=[pos_1, pos_2],
                mode="one_target_per_unit",
            )

    @pytest.mark.parametrize("mode", ["one_target_all_units", "one_target_per_layer"])
    def test_duplicate_cells_raise_in_fused_modes(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition, mode: str
    ) -> None:
        """The uniqueness guard runs before mode dispatch: fused modes (where
        legacy silently double-ran duplicate units in one target) refuse
        duplicate cells just like the per-unit mode — here via a repeated
        layer entry."""
        with pytest.raises(ValueError, match="duplicate grid keys"):
            build_residual_stream_sites(
                pipeline=tiny_pipeline,
                layers=[0, 0],
                token_positions=[last_pos],
                mode=mode,
            )

    def test_invalid_mode_raises_value_error(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            build_residual_stream_sites(
                pipeline=tiny_pipeline,
                layers=[0],
                token_positions=[last_pos],
                mode="not_a_valid_mode",
            )


# =========================================================================== #
#  build_mlp_sites                                                            #
# =========================================================================== #
class TestBuildMlpSitesUnit:
    """``build_mlp_sites`` keys (layer, pos.id) → MLP-site ``[[SiteSpec]]``.

    ``location`` selects ``mlp_input`` / ``mlp_output`` (width = hidden_size)
    vs ``mlp_activation`` (width from the ``n_inner`` / ``intermediate_size``
    / ``hidden_size*4`` fallback chain). Wrong width mis-sizes every
    downstream DAS/DBM rotation.
    """

    pytestmark = pytest.mark.unit

    def test_per_unit_keys_are_layer_position_tuples(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        grid = build_mlp_sites(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        assert len(grid) == len(layers) * len(two_positions)
        for layer, pos_id in grid:
            assert layer in layers
            assert pos_id in {"pos_a", "pos_b"}

    def test_mlp_output_uses_hidden_size(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        grid = build_mlp_sites(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
            location="mlp_output",
        )
        spec = _single_spec(grid)
        assert spec.fsite.site == Site("mlp_output", 0)
        assert spec.width == tiny_pipeline.model.config.hidden_size

    def test_mlp_activation_prefers_n_inner_when_set(self) -> None:
        """If ``config.n_inner`` is present and non-None it wins over
        ``intermediate_size``."""
        pipeline = _fake_pipeline(hidden_size=16, n_inner=42, intermediate_size=64)
        pos = _fake_token_position(pipeline, id="last")
        grid = build_mlp_sites(
            pipeline=pipeline,
            layers=[0],
            token_positions=[pos],
            location="mlp_activation",
        )
        assert _single_spec(grid).width == 42

    def test_mlp_activation_falls_back_to_intermediate_size(self) -> None:
        """When ``n_inner`` is absent (e.g. Llama), ``intermediate_size`` wins."""
        pipeline = _fake_pipeline(hidden_size=16, intermediate_size=64)
        pos = _fake_token_position(pipeline, id="last")
        grid = build_mlp_sites(
            pipeline=pipeline,
            layers=[0],
            token_positions=[pos],
            location="mlp_activation",
        )
        assert _single_spec(grid).width == 64

    def test_mlp_activation_falls_back_to_hidden_size_times_four(self) -> None:
        """Final fallback when neither ``n_inner`` nor ``intermediate_size``
        are on the config."""
        pipeline = _fake_pipeline(hidden_size=16)
        pos = _fake_token_position(pipeline, id="last")
        grid = build_mlp_sites(
            pipeline=pipeline,
            layers=[0],
            token_positions=[pos],
            location="mlp_activation",
        )
        assert _single_spec(grid).width == 64

    def test_mlp_activation_with_none_n_inner_falls_back_to_intermediate(
        self,
    ) -> None:
        """``n_inner`` *present but None* is not picked — fall through to
        ``intermediate_size``."""
        pipeline = _fake_pipeline(hidden_size=16, n_inner=None, intermediate_size=64)
        pos = _fake_token_position(pipeline, id="last")
        grid = build_mlp_sites(
            pipeline=pipeline,
            layers=[0],
            token_positions=[pos],
            location="mlp_activation",
        )
        assert _single_spec(grid).width == 64

    def test_unknown_location_is_refused_at_build_time(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        """A location outside the engine's ``Site`` vocabulary fails at build
        time (the legacy builder deferred this to pyvene execution)."""
        with pytest.raises(ValueError, match="unknown component"):
            build_mlp_sites(
                pipeline=tiny_pipeline,
                layers=[0],
                token_positions=[last_pos],
                location="mlp_bogus",
            )

    def test_invalid_mode_raises_value_error(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            build_mlp_sites(
                pipeline=tiny_pipeline,
                layers=[0],
                token_positions=[last_pos],
                mode="not_a_valid_mode",
            )


# =========================================================================== #
#  build_attention_output_sites                                               #
# =========================================================================== #
class TestBuildAttentionOutputSitesUnit:
    """``build_attention_output_sites`` keys (layer, pos.id) → whole-sublayer
    ``[[SiteSpec]]`` — ``hidden_size``-wide regardless of head count."""

    pytestmark = pytest.mark.unit

    def test_per_unit_keys_are_layer_position_tuples(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        grid = build_attention_output_sites(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        assert len(grid) == len(layers) * len(two_positions)
        for layer, pos_id in grid:
            assert layer in layers
            assert pos_id in {"pos_a", "pos_b"}

    def test_spec_is_attention_output_at_hidden_size(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        grid = build_attention_output_sites(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
        )
        spec = _single_spec(grid)
        assert spec.fsite.site == Site("attention_output", 0)
        assert spec.width == tiny_pipeline.model.config.hidden_size

    def test_invalid_mode_raises_value_error(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            build_attention_output_sites(
                pipeline=tiny_pipeline,
                layers=[0],
                token_positions=[last_pos],
                mode="not_a_valid_mode",
            )


# =========================================================================== #
#  build_attention_head_sites                                                 #
# =========================================================================== #
class TestBuildAttentionHeadSitesUnit:
    """``build_attention_head_sites`` keys (layer, head) → per-head
    ``[[SiteSpec]]`` on ``HeadSite(kind="attention_value")``.

    Head-size resolution walks the config in the legacy priority order:
    ``head_dim`` → ``n_head`` → ``num_attention_heads`` → ``num_heads``;
    missing all four raises ``ValueError``.
    """

    pytestmark = pytest.mark.unit

    def test_per_unit_keys_are_layer_head_tuples(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        layers = [0, 1]
        heads = [0, 1, 2]
        grid = build_attention_head_sites(
            pipeline=tiny_pipeline,
            layers=layers,
            heads=heads,
            token_position=last_pos,
            mode="one_target_per_unit",
        )
        assert len(grid) == len(layers) * len(heads)
        for layer, head in grid:
            assert layer in layers
            assert head in heads

    def test_all_units_mode_collapses_to_single_key(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        grid = build_attention_head_sites(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            heads=[0, 1],
            token_position=last_pos,
            mode="one_target_all_units",
        )
        assert list(grid.keys()) == [("all",)]

    def test_per_layer_mode_groups_all_heads_of_a_layer(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        """The knockout shape: one group per layer holding all its heads."""
        grid = build_attention_head_sites(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            heads=[0, 1, 2],
            token_position=last_pos,
            mode="one_target_per_layer",
        )
        assert set(grid.keys()) == {(0,), (1,)}
        for groups in grid.values():
            assert len(groups) == 1
            assert [s.fsite.site.head for s in groups[0]] == [0, 1, 2]

    def test_spec_site_is_attention_value_head_site(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        """The sender component the legacy ``AttentionHead`` unit targeted
        (pyvene ``head_attention_value_output``) — kind ``attention_value``."""
        grid = build_attention_head_sites(
            pipeline=tiny_pipeline,
            layers=[1],
            heads=[2],
            token_position=last_pos,
        )
        spec = _single_spec(grid)
        assert spec.fsite.site == HeadSite(kind="attention_value", layer=1, head=2)
        assert spec.positions is last_pos

    def test_head_dim_attr_wins(self) -> None:
        """When ``config.head_dim`` is present it short-circuits the
        ``hidden_size // num_heads`` arithmetic."""
        pipeline = _fake_pipeline(hidden_size=64, head_dim=7, n_head=4)
        pos = _fake_token_position(pipeline, id="last")
        grid = build_attention_head_sites(
            pipeline=pipeline,
            layers=[0],
            heads=[0],
            token_position=pos,
        )
        # head_dim=7 wins over hidden_size//n_head = 16.
        assert _single_spec(grid).width == 7

    @pytest.mark.parametrize(
        "attr_name",
        ["n_head", "num_attention_heads", "num_heads"],
    )
    def test_head_count_attr_chain(self, attr_name: str) -> None:
        """In the no-``head_dim`` branch, width = hidden_size // n_heads and
        the n_heads attribute is resolved by the documented priority chain."""
        pipeline = _fake_pipeline(hidden_size=32, **{attr_name: 4})
        pos = _fake_token_position(pipeline, id="last")
        grid = build_attention_head_sites(
            pipeline=pipeline,
            layers=[0],
            heads=[0],
            token_position=pos,
        )
        assert _single_spec(grid).width == 8  # 32 // 4

    def test_missing_all_head_attrs_raises_value_error(self) -> None:
        pipeline = _fake_pipeline(hidden_size=32)  # no head_dim, no n_head etc.
        pos = _fake_token_position(pipeline, id="last")
        with pytest.raises(ValueError, match="Could not determine number of heads"):
            build_attention_head_sites(
                pipeline=pipeline,
                layers=[0],
                heads=[0],
                token_position=pos,
            )

    def test_invalid_mode_raises_value_error(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            build_attention_head_sites(
                pipeline=tiny_pipeline,
                layers=[0],
                heads=[0],
                token_position=last_pos,
                mode="not_a_valid_mode",
            )


# =========================================================================== #
#  grid_component                                                             #
# =========================================================================== #
class TestGridComponentUnit:
    """``grid_component`` dispatches downstream axis extraction — structural
    (``site.component`` / ``HeadSite``), never id-string matching."""

    pytestmark = pytest.mark.unit

    def test_detects_residual_stream(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline, layers=[0], token_positions=[last_pos]
        )
        assert grid_component(grid) == "residual_stream"

    def test_detects_block_input_as_residual_stream(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=[-1],
            token_positions=[last_pos],
        )
        assert grid_component(grid) == "residual_stream"

    @pytest.mark.parametrize("location", ["mlp_input", "mlp_activation", "mlp_output"])
    def test_detects_mlp_at_every_location(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition, location: str
    ) -> None:
        grid = build_mlp_sites(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
            location=location,
        )
        assert grid_component(grid) == "mlp"

    def test_detects_attention_output(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        grid = build_attention_output_sites(
            pipeline=tiny_pipeline, layers=[0], token_positions=[last_pos]
        )
        assert grid_component(grid) == "attention_output"

    def test_detects_attention_head(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        grid = build_attention_head_sites(
            pipeline=tiny_pipeline,
            layers=[0],
            heads=[0],
            token_position=last_pos,
        )
        assert grid_component(grid) == "attention_head"

    @pytest.mark.parametrize("kind", ["value", "query", "attention_value"])
    def test_any_head_site_kind_is_attention_head(self, kind: str) -> None:
        """Structural parity with the legacy ``"AttentionHead" in unit_id``
        substring match, which also caught the value/query receiver units."""
        spec = SiteSpec(
            fsite=FeaturizedSite(HeadSite(kind=kind, layer=0, head=0)),
            positions=None,
            key="k",
        )
        assert grid_component({(0, 0): [[spec]]}) == "attention_head"

    def test_empty_dict_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            grid_component({})

    def test_empty_grid_value_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="no specs"):
            grid_component({("all",): [[]]})

    def test_embeddings_site_has_no_grid_component(self) -> None:
        """No builder emits ``embeddings``; it has no legacy counterpart and
        no heatmap axis layout — refused, mirroring the legacy unknown-id
        error."""
        spec = SiteSpec(
            fsite=FeaturizedSite(Site("embeddings", 0)), positions=None, key="e"
        )
        with pytest.raises(ValueError, match="Unknown grid component"):
            grid_component({(0, "last"): [[spec]]})


class TestGridComponentProperty:
    """Invariance: the detected component does not depend on grouping mode."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("mode", _MODES)
    def test_residual_stream_detection_invariant_under_mode(
        self,
        tiny_pipeline: LMPipeline,
        two_positions: list[TokenPosition],
        mode: str,
    ) -> None:
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            token_positions=two_positions,
            mode=mode,
        )
        assert grid_component(grid) == "residual_stream"

    @pytest.mark.parametrize("mode", _MODES)
    def test_attention_head_detection_invariant_under_mode(
        self,
        tiny_pipeline: LMPipeline,
        last_pos: TokenPosition,
        mode: str,
    ) -> None:
        grid = build_attention_head_sites(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            heads=[0, 1],
            token_position=last_pos,
            mode=mode,
        )
        assert grid_component(grid) == "attention_head"


# =========================================================================== #
#  extract_grid_dimensions_from_targets                                       #
# =========================================================================== #
class TestExtractGridDimensionsUnit:
    """``extract_grid_dimensions_from_targets`` recovers the (layers, heads)
    or (layers, token_position_ids) axes consumed by
    :func:`causalab.io.plots.score_heatmap` — ported as-is from the legacy
    module (it reads only dict keys)."""

    pytestmark = pytest.mark.unit

    def test_attention_head_axes_are_sorted(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        # Build with unsorted layers/heads to confirm the function sorts.
        grid = build_attention_head_sites(
            pipeline=tiny_pipeline,
            layers=[1, 0],
            heads=[2, 0, 1],
            token_position=last_pos,
            mode="one_target_per_unit",
        )
        dims = extract_grid_dimensions_from_targets("attention_head", grid)
        assert dims["layers"] == [0, 1]
        assert dims["heads"] == [0, 1, 2]

    def test_residual_stream_preserves_position_insertion_order(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """token_position_ids keep first-seen insertion order — the
        ``score_heatmap`` axis-order contract."""
        pos_b = TokenPosition(lambda _: [1], tiny_pipeline, id="zzz_pos_b")
        pos_a = TokenPosition(lambda _: [0], tiny_pipeline, id="aaa_pos_a")
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=[0],
            # Pass alphabetically-late id first; round-trip must preserve
            # order (not sort).
            token_positions=[pos_b, pos_a],
            mode="one_target_per_unit",
        )
        dims = extract_grid_dimensions_from_targets("residual_stream", grid)
        assert dims["token_position_ids"] == ["zzz_pos_b", "aaa_pos_a"]

    def test_mlp_axis_names_are_layers_and_token_position_ids(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        grid = build_mlp_sites(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        dims = extract_grid_dimensions_from_targets("mlp", grid)
        assert set(dims.keys()) == {"layers", "token_position_ids"}

    def test_raises_on_unknown_component_type(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=[0],
            token_positions=[last_pos],
            mode="one_target_per_unit",
        )
        with pytest.raises(ValueError, match="Unknown component_type"):
            extract_grid_dimensions_from_targets("not_a_real_component", grid)


# =========================================================================== #
#  Cardinality + round-trip properties                                        #
# =========================================================================== #
class TestGridBuilderProperty:
    """Cardinality, round-trip, key-uniqueness, and determinism invariants
    across all four builders."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize(
        "mode,expected_n",
        [
            ("one_target_per_unit", 4),  # 2 layers * 2 positions
            ("one_target_per_layer", 2),  # 2 layers
            ("one_target_all_units", 1),  # collapsed
        ],
    )
    @pytest.mark.parametrize(
        "builder",
        [
            build_residual_stream_sites,
            build_mlp_sites,
            build_attention_output_sites,
        ],
    )
    def test_cardinality_invariant_across_modes(
        self,
        tiny_pipeline: LMPipeline,
        two_positions: list[TokenPosition],
        builder: Any,
        mode: str,
        expected_n: int,
    ) -> None:
        grid = builder(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            token_positions=two_positions,
            mode=mode,
        )
        assert len(grid) == expected_n
        # Grouping never drops or duplicates a spec.
        assert len(_all_specs(grid)) == 4

    @pytest.mark.parametrize(
        "mode,expected_n",
        [
            ("one_target_per_unit", 6),  # 2 layers * 3 heads
            ("one_target_per_layer", 2),
            ("one_target_all_units", 1),
        ],
    )
    def test_attention_head_cardinality_invariant_across_modes(
        self,
        tiny_pipeline: LMPipeline,
        last_pos: TokenPosition,
        mode: str,
        expected_n: int,
    ) -> None:
        grid = build_attention_head_sites(
            pipeline=tiny_pipeline,
            layers=[0, 1],
            heads=[0, 1, 2],
            token_position=last_pos,
            mode=mode,
        )
        assert len(grid) == expected_n
        assert len(_all_specs(grid)) == 6

    @pytest.mark.parametrize("mode", _MODES)
    def test_spec_keys_unique_across_grid(
        self,
        tiny_pipeline: LMPipeline,
        two_positions: list[TokenPosition],
        mode: str,
    ) -> None:
        """Every generated spec key is unique across the grid, in every mode —
        they feed collect-result dicts (WU3)."""
        for grid in (
            build_residual_stream_sites(
                pipeline=tiny_pipeline,
                layers=[-1, 0, 1],
                token_positions=two_positions,
                mode=mode,
            ),
            build_attention_head_sites(
                pipeline=tiny_pipeline,
                layers=[0, 1],
                heads=[0, 1, 2],
                token_position=two_positions[0],
                mode=mode,
            ),
        ):
            keys = [spec.key for spec in _all_specs(grid)]
            assert len(set(keys)) == len(keys)

    def test_detect_extract_round_trip_recovers_layers(
        self, tiny_pipeline: LMPipeline, two_positions: list[TokenPosition]
    ) -> None:
        layers = [0, 1]
        grid = build_residual_stream_sites(
            pipeline=tiny_pipeline,
            layers=layers,
            token_positions=two_positions,
            mode="one_target_per_unit",
        )
        component = grid_component(grid)
        dims = extract_grid_dimensions_from_targets(component, grid)
        assert component == "residual_stream"
        assert dims["layers"] == layers
        assert dims["token_position_ids"] == ["pos_a", "pos_b"]

    def test_detect_extract_round_trip_recovers_layers_and_heads(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        layers = [0, 1]
        heads = [0, 1, 2]
        grid = build_attention_head_sites(
            pipeline=tiny_pipeline,
            layers=layers,
            heads=heads,
            token_position=last_pos,
            mode="one_target_per_unit",
        )
        component = grid_component(grid)
        dims = extract_grid_dimensions_from_targets(component, grid)
        assert component == "attention_head"
        assert dims["layers"] == layers
        assert dims["heads"] == heads

    def test_determinism_across_repeated_calls(
        self, tiny_pipeline: LMPipeline, last_pos: TokenPosition
    ) -> None:
        """Building twice with the same args yields the same keys and specs
        (specs are frozen values; only the identity featurizer objects and
        the shared position resolver distinguish instances)."""
        a = build_residual_stream_sites(
            pipeline=tiny_pipeline, layers=[0, 1], token_positions=[last_pos]
        )
        b = build_residual_stream_sites(
            pipeline=tiny_pipeline, layers=[0, 1], token_positions=[last_pos]
        )
        assert list(a.keys()) == list(b.keys())
        for spec_a, spec_b in zip(_all_specs(a), _all_specs(b)):
            assert spec_a.key == spec_b.key
            assert spec_a.fsite.site == spec_b.fsite.site
            assert spec_a.width == spec_b.width
            assert spec_a.positions is spec_b.positions
