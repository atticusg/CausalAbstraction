"""Tests for the path-patching analysis' receiver resolution.

The two-pass runner itself is exercised end-to-end in
``tests/methods/path_patching`` (on the real tiny pipeline). These pin the
analysis-layer glue: that ``path_patching.receiver`` config maps to the right
:class:`ReceiverSpec`, and that the output-dir/figure tag disambiguates receivers.
The full numeric run is golden-only (it needs a model+tokenizer with single-token
answers, e.g. GPT-2 on IOI — see docs/TESTS.md "Smoke vs golden split").
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from causalab.analyses.path_patching.main import (
    _build_receiver_spec,
    _reachable_senders,
    _receiver_tag,
    _resolve_token_position,
)
from causalab.methods.path_patching.targets import OUTPUT, ReceiverSpec
from causalab.neural.activations.targets import build_attention_head_targets
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition, get_last_token_index

pytestmark = pytest.mark.unit


def _analysis(receiver: dict | None, **extra) -> OmegaConf:
    body: dict = {"layers": [0], "heads": [0], **extra}
    if receiver is not None:
        body["receiver"] = receiver
    return OmegaConf.create(body)


def _pos(pipeline: LMPipeline) -> TokenPosition:
    """A last-token read position, the default the runner passes in."""
    return TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline), pipeline, id="last_token"
    )


def _sender_grid(pipeline: LMPipeline):
    """A (layer x head) sender grid over both layers of the tiny model + last-token pos."""
    pos = TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline), pipeline, id="last_token"
    )
    targets = build_attention_head_targets(pipeline, [0, 1], [0, 1], pos)
    return {key: t.flatten()[0] for key, t in targets.items()}, pos


class TestBuildReceiverSpec:
    def test_missing_block_defaults_to_output(self, mock_tiny_lm: LMPipeline) -> None:
        assert _build_receiver_spec(_analysis(None), _pos(mock_tiny_lm)) is OUTPUT

    def test_kind_output_is_output(self, mock_tiny_lm: LMPipeline) -> None:
        assert (
            _build_receiver_spec(_analysis({"kind": "output"}), _pos(mock_tiny_lm))
            is OUTPUT
        )

    def test_head_value_input(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _pos(mock_tiny_lm)
        spec = _build_receiver_spec(
            _analysis({"kind": "head_value_input", "layer": 1, "head": 0}), pos
        )
        assert spec.kind == "head_value_input"
        assert spec.layer == 1 and spec.head == 0
        assert spec.token_position is pos  # the passed-in read position

    def test_head_query_input(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _pos(mock_tiny_lm)
        spec = _build_receiver_spec(
            _analysis({"kind": "head_query_input", "layer": 1, "head": 0}), pos
        )
        assert spec.kind == "head_query_input"
        assert spec.layer == 1 and spec.head == 0
        assert spec.token_position is pos

    def test_head_query_input_requires_head(self, mock_tiny_lm: LMPipeline) -> None:
        with pytest.raises(ValueError, match="requires head"):
            _build_receiver_spec(
                _analysis({"kind": "head_query_input", "layer": 1}), _pos(mock_tiny_lm)
            )

    def test_mlp_input(self, mock_tiny_lm: LMPipeline) -> None:
        spec = _build_receiver_spec(
            _analysis({"kind": "mlp_input", "layer": 1}), _pos(mock_tiny_lm)
        )
        assert spec.kind == "mlp_input" and spec.layer == 1

    def test_residual_point(self, mock_tiny_lm: LMPipeline) -> None:
        spec = _build_receiver_spec(
            _analysis(
                {"kind": "residual", "layer": 1, "residual_point": "block_input"}
            ),
            _pos(mock_tiny_lm),
        )
        assert spec.kind == "residual" and spec.residual_point == "block_input"

    def test_unknown_kind_raises(self, mock_tiny_lm: LMPipeline) -> None:
        with pytest.raises(ValueError, match="receiver.kind must be one of"):
            _build_receiver_spec(_analysis({"kind": "bogus"}), _pos(mock_tiny_lm))


class TestReachableSenders:
    def test_output_returns_grid_unchanged(self, mock_tiny_lm: LMPipeline) -> None:
        senders, _ = _sender_grid(mock_tiny_lm)
        out = _reachable_senders(mock_tiny_lm, senders, OUTPUT, "output")
        assert out is senders

    def test_drops_downstream_layers(self, mock_tiny_lm: LMPipeline) -> None:
        # head_value_input at layer 1 reads before attention 1, so only layer-0 senders
        # reach it; layer-1 senders are downstream and must be dropped.
        senders, pos = _sender_grid(mock_tiny_lm)
        rcv = ReceiverSpec(kind="head_value_input", layer=1, head=0, token_position=pos)
        out = _reachable_senders(mock_tiny_lm, senders, rcv, _receiver_tag(rcv))
        assert {key[0] for key in out} == {0}
        assert len(out) < len(senders)

    def test_empty_grid_raises(self, mock_tiny_lm: LMPipeline) -> None:
        # A value receiver at layer 0 is reachable by no sender in the grid.
        senders, pos = _sender_grid(mock_tiny_lm)
        rcv = ReceiverSpec(kind="head_value_input", layer=0, head=0, token_position=pos)
        with pytest.raises(ValueError, match="No sender in the grid can reach"):
            _reachable_senders(mock_tiny_lm, senders, rcv, _receiver_tag(rcv))


class TestReceiverTag:
    def test_tags(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _pos(mock_tiny_lm)

        def mk(r: dict | None):
            return _build_receiver_spec(_analysis(r), pos)

        assert _receiver_tag(mk(None)) == "output"
        assert (
            _receiver_tag(mk({"kind": "head_value_input", "layer": 1, "head": 0}))
            == "head_value_input_L1H0"
        )
        assert (
            _receiver_tag(mk({"kind": "head_query_input", "layer": 1, "head": 0}))
            == "head_query_input_L1H0"
        )
        assert _receiver_tag(mk({"kind": "mlp_input", "layer": 2})) == "mlp_input_L2"
        assert (
            _receiver_tag(mk({"kind": "residual", "layer": 3}))
            == "residual_block_output_L3"
        )

    def test_non_default_position_in_tag(self, mock_tiny_lm: LMPipeline) -> None:
        # A non-last-token read position is appended so sweeps at S2 vs last_token
        # land in distinct output subdirs.
        s2 = TokenPosition(lambda inp: [0], mock_tiny_lm, id="name_C")
        spec = _build_receiver_spec(
            _analysis({"kind": "head_query_input", "layer": 1, "head": 0}), s2
        )
        assert _receiver_tag(spec) == "head_query_input_L1H0_name_C"


class TestResolveTokenPosition:
    class _StubTask:
        name = "IOI"

        def create_token_positions(self, pipeline):
            return {"name_C": TokenPosition(lambda inp: [0], pipeline, id="name_C")}

    def test_defaults_to_last_token(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _resolve_token_position(_analysis(None), self._StubTask(), mock_tiny_lm)
        assert pos.id == "last_token"

    def test_named_position_resolves(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _resolve_token_position(
            _analysis(None, token_position="name_C"), self._StubTask(), mock_tiny_lm
        )
        assert pos.id == "name_C"

    def test_unknown_position_raises(self, mock_tiny_lm: LMPipeline) -> None:
        with pytest.raises(ValueError, match="not a position of task"):
            _resolve_token_position(
                _analysis(None, token_position="nope"), self._StubTask(), mock_tiny_lm
            )
