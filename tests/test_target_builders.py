"""
Tests for the InterchangeTarget builder functions in causalab.neural.activations.

Verifies that:
- All builder functions are importable from causalab.neural.activations
- Builders return dicts with tuple keys
- Key shapes match the expected mode contract
- detect_component_type_from_targets and extract_grid_dimensions_from_targets
  round-trip correctly.
"""

import pytest
import torch

from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget
from causalab.neural.activations import (
    build_residual_stream_targets,
    build_attention_head_targets,
    build_mlp_targets,
    detect_component_type_from_targets,
    extract_grid_dimensions_from_targets,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


class _MockConfig:
    """Minimal model config that satisfies all three builders."""

    name_or_path = "mock_model"
    num_hidden_layers = 4
    hidden_size = 32
    n_head = 4  # used by build_attention_head_targets
    # n_inner absent → mlp_activation falls back to hidden_size * 4


class _MockModel:
    def __init__(self):
        self.config = _MockConfig()
        self.device = "cpu"
        self.dtype = torch.float32

    def to(self, device=None, dtype=None):
        return self


@pytest.fixture(scope="module")
def mock_pipeline():
    """Create a mock LMPipeline without loading real model weights."""
    pipeline = LMPipeline.__new__(LMPipeline)
    pipeline.model = _MockModel()

    class _MockTokenizer:
        pad_token_id = 0
        padding_side = "right"

    pipeline.tokenizer = _MockTokenizer()
    return pipeline


@pytest.fixture(scope="module")
def last_token_position(mock_pipeline):
    """A simple TokenPosition that always returns token index 0."""
    return TokenPosition(lambda _: [0], mock_pipeline, id="last")


@pytest.fixture(scope="module")
def two_token_positions(mock_pipeline):
    """Two TokenPositions for multi-position builder calls."""
    pos_a = TokenPosition(lambda _: [0], mock_pipeline, id="pos_a")
    pos_b = TokenPosition(lambda _: [1], mock_pipeline, id="pos_b")
    return [pos_a, pos_b]


# --------------------------------------------------------------------------- #
# 1. Import smoke tests                                                        #
# --------------------------------------------------------------------------- #


def test_import_build_residual_stream_targets():
    from causalab.neural.activations import build_residual_stream_targets  # noqa: F401


def test_import_build_attention_head_targets():
    from causalab.neural.activations import build_attention_head_targets  # noqa: F401


def test_import_build_mlp_targets():
    from causalab.neural.activations import build_mlp_targets  # noqa: F401


def test_import_detect_component_type():
    from causalab.neural.activations import detect_component_type_from_targets  # noqa: F401


def test_import_extract_grid_dimensions():
    from causalab.neural.activations import extract_grid_dimensions_from_targets  # noqa: F401


# --------------------------------------------------------------------------- #
# 2. build_residual_stream_targets                                             #
# --------------------------------------------------------------------------- #


def test_residual_stream_returns_dict(mock_pipeline, last_token_position):
    targets = build_residual_stream_targets(
        pipeline=mock_pipeline,
        layers=[0],
        token_positions=[last_token_position],
        mode="one_target_per_unit",
    )
    assert isinstance(targets, dict)


def test_residual_stream_values_are_interchange_targets(
    mock_pipeline, last_token_position
):
    targets = build_residual_stream_targets(
        pipeline=mock_pipeline,
        layers=[0],
        token_positions=[last_token_position],
        mode="one_target_per_unit",
    )
    for v in targets.values():
        assert isinstance(v, InterchangeTarget)


def test_residual_stream_per_unit_keys_are_tuples(mock_pipeline, last_token_position):
    targets = build_residual_stream_targets(
        pipeline=mock_pipeline,
        layers=[0],
        token_positions=[last_token_position],
        mode="one_target_per_unit",
    )
    for k in targets.keys():
        assert isinstance(k, tuple)


def test_residual_stream_per_unit_key_shape(mock_pipeline, two_token_positions):
    """Keys should be (layer, position_id) tuples."""
    layers = [0]
    targets = build_residual_stream_targets(
        pipeline=mock_pipeline,
        layers=layers,
        token_positions=two_token_positions,
        mode="one_target_per_unit",
    )
    # Expect one key per (layer, position) combination
    assert len(targets) == len(layers) * len(two_token_positions)
    for key in targets.keys():
        assert len(key) == 2
        layer, pos_id = key
        assert layer in layers
        assert pos_id in ["pos_a", "pos_b"]


def test_residual_stream_all_units_key(mock_pipeline, two_token_positions):
    """mode='one_target_all_units' should produce a single key ('all',)."""
    targets = build_residual_stream_targets(
        pipeline=mock_pipeline,
        layers=[0],
        token_positions=two_token_positions,
        mode="one_target_all_units",
    )
    assert list(targets.keys()) == [("all",)]


def test_residual_stream_per_layer_key(mock_pipeline, two_token_positions):
    """mode='one_target_per_layer' should produce one key per layer."""
    layers = [0]
    targets = build_residual_stream_targets(
        pipeline=mock_pipeline,
        layers=layers,
        token_positions=two_token_positions,
        mode="one_target_per_layer",
    )
    assert len(targets) == len(layers)
    assert (0,) in targets


# --------------------------------------------------------------------------- #
# 3. build_mlp_targets                                                         #
# --------------------------------------------------------------------------- #


def test_mlp_returns_dict(mock_pipeline, last_token_position):
    targets = build_mlp_targets(
        pipeline=mock_pipeline,
        layers=[0],
        token_positions=[last_token_position],
        mode="one_target_per_unit",
    )
    assert isinstance(targets, dict)


def test_mlp_per_unit_key_shape(mock_pipeline, two_token_positions):
    """Keys should be (layer, position_id) tuples."""
    layers = [0]
    targets = build_mlp_targets(
        pipeline=mock_pipeline,
        layers=layers,
        token_positions=two_token_positions,
        mode="one_target_per_unit",
    )
    assert len(targets) == len(layers) * len(two_token_positions)
    for key in targets.keys():
        layer, pos_id = key
        assert layer in layers
        assert pos_id in ["pos_a", "pos_b"]


def test_mlp_values_are_interchange_targets(mock_pipeline, last_token_position):
    targets = build_mlp_targets(
        pipeline=mock_pipeline,
        layers=[0],
        token_positions=[last_token_position],
    )
    for v in targets.values():
        assert isinstance(v, InterchangeTarget)


# --------------------------------------------------------------------------- #
# 4. build_attention_head_targets                                              #
# --------------------------------------------------------------------------- #


def test_attention_head_returns_dict(mock_pipeline, last_token_position):
    targets = build_attention_head_targets(
        pipeline=mock_pipeline,
        layers=[0],
        heads=[0, 1],
        token_position=last_token_position,
        mode="one_target_per_unit",
    )
    assert isinstance(targets, dict)


def test_attention_head_per_unit_key_shape(mock_pipeline, last_token_position):
    """Keys should be (layer, head) tuples."""
    layers = [0]
    heads = [0, 1, 2]
    targets = build_attention_head_targets(
        pipeline=mock_pipeline,
        layers=layers,
        heads=heads,
        token_position=last_token_position,
        mode="one_target_per_unit",
    )
    assert len(targets) == len(layers) * len(heads)
    for key in targets.keys():
        layer, head = key
        assert layer in layers
        assert head in heads


def test_attention_head_values_are_interchange_targets(
    mock_pipeline, last_token_position
):
    targets = build_attention_head_targets(
        pipeline=mock_pipeline,
        layers=[0],
        heads=[0],
        token_position=last_token_position,
    )
    for v in targets.values():
        assert isinstance(v, InterchangeTarget)


def test_attention_head_all_units_key(mock_pipeline, last_token_position):
    targets = build_attention_head_targets(
        pipeline=mock_pipeline,
        layers=[0],
        heads=[0, 1],
        token_position=last_token_position,
        mode="one_target_all_units",
    )
    assert list(targets.keys()) == [("all",)]


# --------------------------------------------------------------------------- #
# 5. detect_component_type_from_targets                                        #
# --------------------------------------------------------------------------- #


def test_detect_attention_head(mock_pipeline, last_token_position):
    targets = build_attention_head_targets(
        pipeline=mock_pipeline,
        layers=[0],
        heads=[0],
        token_position=last_token_position,
    )
    assert detect_component_type_from_targets(targets) == "attention_head"


def test_detect_residual_stream(mock_pipeline, last_token_position):
    targets = build_residual_stream_targets(
        pipeline=mock_pipeline,
        layers=[0],
        token_positions=[last_token_position],
    )
    assert detect_component_type_from_targets(targets) == "residual_stream"


def test_detect_mlp(mock_pipeline, last_token_position):
    targets = build_mlp_targets(
        pipeline=mock_pipeline,
        layers=[0],
        token_positions=[last_token_position],
    )
    assert detect_component_type_from_targets(targets) == "mlp"


def test_detect_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        detect_component_type_from_targets({})


# --------------------------------------------------------------------------- #
# 6. extract_grid_dimensions_from_targets                                      #
# --------------------------------------------------------------------------- #


def test_extract_grid_attention_head(mock_pipeline, last_token_position):
    layers = [0]
    heads = [0, 1, 2]
    targets = build_attention_head_targets(
        pipeline=mock_pipeline,
        layers=layers,
        heads=heads,
        token_position=last_token_position,
        mode="one_target_per_unit",
    )
    dims = extract_grid_dimensions_from_targets("attention_head", targets)
    assert dims["layers"] == layers
    assert dims["heads"] == heads


def test_extract_grid_residual_stream(mock_pipeline, two_token_positions):
    layers = [0]
    targets = build_residual_stream_targets(
        pipeline=mock_pipeline,
        layers=layers,
        token_positions=two_token_positions,
        mode="one_target_per_unit",
    )
    dims = extract_grid_dimensions_from_targets("residual_stream", targets)
    assert dims["layers"] == layers
    assert set(dims["token_position_ids"]) == {"pos_a", "pos_b"}


def test_extract_grid_mlp(mock_pipeline, two_token_positions):
    layers = [0]
    targets = build_mlp_targets(
        pipeline=mock_pipeline,
        layers=layers,
        token_positions=two_token_positions,
        mode="one_target_per_unit",
    )
    dims = extract_grid_dimensions_from_targets("mlp", targets)
    assert dims["layers"] == layers
    assert set(dims["token_position_ids"]) == {"pos_a", "pos_b"}
