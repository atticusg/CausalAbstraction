"""Tests for path-patching target construction (``methods.path_patching.targets``).

The restorer set *is* the estimand definition, so these pin its composition: only
``attention_output`` / ``mlp_output`` (never ``block_output``), the sender
excluded, the right components per ``restore`` selection, and the right layer
range above the sender. Property tier — shape/membership contracts on the tiny
model's config, no generation.
"""

from __future__ import annotations

import pytest

from causalab.methods.path_patching.targets import (
    OUTPUT,
    ReceiverSpec,
    build_path_patching_target,
    build_receiver_unit,
    build_restorer_set,
    sender_reaches_receiver,
)
from causalab.neural.LM_units import AttentionHead
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition, get_last_token_index


def _last_token(pipeline: LMPipeline) -> TokenPosition:
    return TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline), pipeline, id="last_token"
    )


def _pos(pipeline: LMPipeline, id: str) -> TokenPosition:
    return TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline), pipeline, id=id
    )


def _sender(pipeline: LMPipeline, layer: int, head: int = 0) -> AttentionHead:
    hd = pipeline.model.config.hidden_size // pipeline.model.config.num_attention_heads
    return AttentionHead(
        layer, head, _last_token(pipeline), target_output=True, shape=(hd,)
    )


class TestRestorerSet:
    pytestmark = pytest.mark.property

    def test_never_includes_block_output_or_sender(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        sender = _sender(mock_tiny_lm, 0)
        restorers = build_restorer_set(mock_tiny_lm, sender)
        comps = {u.component_type for u in restorers}
        assert "block_output" not in comps
        assert comps <= {"attention_output", "mlp_output"}
        assert sender.id not in {u.id for u in restorers}

    def test_full_restore_covers_attention_and_mlp_above_sender(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        n_layers = mock_tiny_lm.model.config.num_hidden_layers
        sender = _sender(mock_tiny_lm, 0)
        restorers = build_restorer_set(
            mock_tiny_lm, sender, restore=("attention", "mlp")
        )
        layers_attn = sorted(
            u.layer for u in restorers if u.component_type == "attention_output"
        )
        layers_mlp = sorted(
            u.layer for u in restorers if u.component_type == "mlp_output"
        )
        # attention frozen for every layer strictly above the sender
        assert layers_attn == list(range(1, n_layers))
        # mlp frozen for the sender's own layer (attn-head sender) and all above
        assert layers_mlp == list(range(0, n_layers))

    def test_attention_only_omits_all_mlp(self, mock_tiny_lm: LMPipeline) -> None:
        sender = _sender(mock_tiny_lm, 0)
        restorers = build_restorer_set(mock_tiny_lm, sender, restore=("attention",))
        assert all(u.component_type == "attention_output" for u in restorers)

    def test_mlp_only_omits_all_attention(self, mock_tiny_lm: LMPipeline) -> None:
        sender = _sender(mock_tiny_lm, 0)
        restorers = build_restorer_set(mock_tiny_lm, sender, restore=("mlp",))
        assert all(u.component_type == "mlp_output" for u in restorers)
        # still includes the sender-layer MLP (attn-head sender)
        assert any(u.layer == 0 for u in restorers)

    def test_top_layer_sender_full_is_sender_mlp_only(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """A sender in the top layer has no layers above it, so the only restorer
        is its own layer's MLP output."""
        n_layers = mock_tiny_lm.model.config.num_hidden_layers
        sender = _sender(mock_tiny_lm, n_layers - 1)
        restorers = build_restorer_set(
            mock_tiny_lm, sender, restore=("attention", "mlp")
        )
        assert [u.component_type for u in restorers] == ["mlp_output"]
        assert restorers[0].layer == n_layers - 1

    def test_restorers_share_sender_read_position(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        sender = _sender(mock_tiny_lm, 0)
        restorers = build_restorer_set(mock_tiny_lm, sender)
        assert all("Token-last_token" in u.id for u in restorers)

    def test_empty_restore_raises(self, mock_tiny_lm: LMPipeline) -> None:
        with pytest.raises(ValueError, match="at least one component family"):
            build_restorer_set(mock_tiny_lm, _sender(mock_tiny_lm, 0), restore=())

    def test_invalid_restore_family_raises(self, mock_tiny_lm: LMPipeline) -> None:
        with pytest.raises(ValueError, match="not valid"):
            build_restorer_set(
                mock_tiny_lm, _sender(mock_tiny_lm, 0), restore=("attention", "qkv")
            )


class TestReceiverSpec:
    pytestmark = pytest.mark.property

    def test_output_rejects_location_fields(self, mock_tiny_lm: LMPipeline) -> None:
        with pytest.raises(ValueError, match="takes no layer"):
            ReceiverSpec(kind="output", layer=1)

    def test_internal_requires_layer_and_position(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        with pytest.raises(ValueError, match="requires layer and token_position"):
            ReceiverSpec(kind="mlp_input", token_position=_last_token(mock_tiny_lm))
        with pytest.raises(ValueError, match="requires layer and token_position"):
            ReceiverSpec(kind="mlp_input", layer=1)

    def test_head_value_requires_head(self, mock_tiny_lm: LMPipeline) -> None:
        with pytest.raises(ValueError, match="requires head"):
            ReceiverSpec(
                kind="head_value_input",
                layer=1,
                token_position=_last_token(mock_tiny_lm),
            )

    def test_head_query_requires_head(self, mock_tiny_lm: LMPipeline) -> None:
        with pytest.raises(ValueError, match="requires head"):
            ReceiverSpec(
                kind="head_query_input",
                layer=1,
                token_position=_last_token(mock_tiny_lm),
            )

    def test_non_head_receiver_rejects_head(self, mock_tiny_lm: LMPipeline) -> None:
        with pytest.raises(ValueError, match="does not take a head"):
            ReceiverSpec(
                kind="mlp_input",
                layer=1,
                head=0,
                token_position=_last_token(mock_tiny_lm),
            )


class TestReceiverUnit:
    pytestmark = pytest.mark.property

    def test_output_has_no_unit(self, mock_tiny_lm: LMPipeline) -> None:
        assert build_receiver_unit(mock_tiny_lm, OUTPUT) is None

    def test_head_value_input_targets_head_value_output(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        unit = build_receiver_unit(
            mock_tiny_lm,
            ReceiverSpec(kind="head_value_input", layer=1, head=0, token_position=pos),
        )
        assert unit is not None
        # The per-head value vector — the only realizable "v of head h" read point.
        assert unit.component_type == "head_value_output"
        assert unit.unit == "h.pos"

    def test_head_value_input_guards_out_of_range_query_head(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        # A head index at num_attention_heads is out of the query-head range.
        n_head = mock_tiny_lm.model.config.num_attention_heads
        pos = _last_token(mock_tiny_lm)
        with pytest.raises(ValueError, match="out of range"):
            build_receiver_unit(
                mock_tiny_lm,
                ReceiverSpec(
                    kind="head_value_input", layer=1, head=n_head, token_position=pos
                ),
            )

    def test_head_query_input_targets_head_query_output(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        unit = build_receiver_unit(
            mock_tiny_lm,
            ReceiverSpec(kind="head_query_input", layer=1, head=0, token_position=pos),
        )
        assert unit is not None
        # The per-head query vector — the only realizable "q of head h" read point.
        assert unit.component_type == "head_query_output"
        assert unit.unit == "h.pos"

    def test_head_query_input_guards_out_of_range_head(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        n_head = mock_tiny_lm.model.config.num_attention_heads
        pos = _last_token(mock_tiny_lm)
        with pytest.raises(ValueError, match="out of range"):
            build_receiver_unit(
                mock_tiny_lm,
                ReceiverSpec(
                    kind="head_query_input", layer=1, head=n_head, token_position=pos
                ),
            )

    def test_gqa_query_head_is_per_query_head(self, gqa_tiny_lm: LMPipeline) -> None:
        # Unlike the value vector (shared per KV group), the query vector is
        # per-query-head even under GQA, so head_query_input addresses the query
        # head directly with no KV-group remap.
        cfg = gqa_tiny_lm.model.config
        n_head, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        assert n_kv < n_head  # fixture really is grouped-query
        pos = _last_token(gqa_tiny_lm)
        for qh in range(n_head):
            unit = build_receiver_unit(
                gqa_tiny_lm,
                ReceiverSpec(
                    kind="head_query_input", layer=1, head=qh, token_position=pos
                ),
            )
            assert unit is not None
            assert unit.component_type == "head_query_output"
            assert unit.head == qh  # the query head itself, not a KV-group index

    def test_mlp_input_targets_mlp_input(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _last_token(mock_tiny_lm)
        unit = build_receiver_unit(
            mock_tiny_lm, ReceiverSpec(kind="mlp_input", layer=1, token_position=pos)
        )
        assert unit is not None and unit.component_type == "mlp_input"

    def test_gqa_query_head_maps_to_kv_group(self, gqa_tiny_lm: LMPipeline) -> None:
        # On GQA the value vector is shared per KV group, so a query head maps to
        # `head // (n_head // n_kv)`. The AttentionHeadValue unit must address that
        # KV-head index (head_value_output is in KV-head space).
        cfg = gqa_tiny_lm.model.config
        n_head, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        assert n_kv < n_head  # fixture really is grouped-query
        group = n_head // n_kv
        pos = _last_token(gqa_tiny_lm)
        for qh in range(n_head):
            unit = build_receiver_unit(
                gqa_tiny_lm,
                ReceiverSpec(
                    kind="head_value_input", layer=1, head=qh, token_position=pos
                ),
            )
            assert unit is not None
            assert unit.component_type == "head_value_output"
            assert unit.head == qh // group  # KV-group index, not the query head

    def test_decoupled_head_dim_value_receiver_uses_config_head_dim(
        self, gqa_decoupled_head_dim_lm: LMPipeline
    ) -> None:
        """The value receiver's width follows ``config.head_dim``, not
        ``hidden // n_head``.

        Getting this wrong is silent in the common coupled case and fatal when the
        two differ: PASS 1 would collect a ``hidden // n_head``-wide vector that
        cannot be injected into the true ``head_dim``-wide slot in PASS 2. The
        decoupled config was unsupported outright before the nnsight migration
        (#386); this pins that it now resolves to the right width.
        """
        cfg = gqa_decoupled_head_dim_lm.model.config
        ratio = cfg.hidden_size // cfg.num_attention_heads
        assert cfg.head_dim != ratio  # the fixture is genuinely decoupled
        pos = _last_token(gqa_decoupled_head_dim_lm)
        unit = build_receiver_unit(
            gqa_decoupled_head_dim_lm,
            ReceiverSpec(kind="head_value_input", layer=1, head=0, token_position=pos),
        )
        assert unit is not None
        assert unit.component_type == "head_value_output"
        assert unit.shape == (cfg.head_dim,)

    def test_decoupled_head_dim_query_receiver_uses_config_head_dim(
        self, gqa_decoupled_head_dim_lm: LMPipeline
    ) -> None:
        cfg = gqa_decoupled_head_dim_lm.model.config
        assert cfg.head_dim != cfg.hidden_size // cfg.num_attention_heads
        pos = _last_token(gqa_decoupled_head_dim_lm)
        unit = build_receiver_unit(
            gqa_decoupled_head_dim_lm,
            ReceiverSpec(kind="head_query_input", layer=1, head=1, token_position=pos),
        )
        assert unit is not None
        assert unit.component_type == "head_query_output"
        assert unit.shape == (cfg.head_dim,)
        assert unit.head == 1  # queries are per-query-head: no KV-group remap

    def test_residual_point_selects_block_component(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        out = build_receiver_unit(
            mock_tiny_lm, ReceiverSpec(kind="residual", layer=1, token_position=pos)
        )
        inp = build_receiver_unit(
            mock_tiny_lm,
            ReceiverSpec(
                kind="residual",
                layer=1,
                token_position=pos,
                residual_point="block_input",
            ),
        )
        assert out is not None and out.component_type == "block_output"
        assert inp is not None and inp.component_type == "block_input"


class TestInternalRestorerRange:
    """The restorer range must stop at the receiver's read point, with the right
    mid-block membership. On the 2-layer tiny model, a sender at layer 0 and a
    receiver at layer 1 makes the attn(1)/mlp(1) membership the discriminating
    test across receiver kinds (full restore)."""

    pytestmark = pytest.mark.property

    def _layers(self, restorers, component_type):
        return sorted(u.layer for u in restorers if u.component_type == component_type)

    def test_head_value_input_excludes_its_layer_attention(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        sender = _sender(mock_tiny_lm, 0)
        rcv = ReceiverSpec(kind="head_value_input", layer=1, head=0, token_position=pos)
        restorers = build_restorer_set(mock_tiny_lm, sender, rcv)
        # Reads the residual entering attention layer 1, so layer-1 attention is
        # NOT a restorer; only the sender-layer MLP is between sender and receiver.
        assert self._layers(restorers, "attention_output") == []
        assert self._layers(restorers, "mlp_output") == [0]

    def test_mlp_input_includes_its_layer_attention_not_mlp(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        sender = _sender(mock_tiny_lm, 0)
        rcv = ReceiverSpec(kind="mlp_input", layer=1, token_position=pos)
        restorers = build_restorer_set(mock_tiny_lm, sender, rcv)
        # Reads after layer-1 attention but before layer-1 MLP.
        assert self._layers(restorers, "attention_output") == [1]
        assert self._layers(restorers, "mlp_output") == [0]

    def test_residual_block_output_includes_full_layer(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        sender = _sender(mock_tiny_lm, 0)
        rcv = ReceiverSpec(kind="residual", layer=1, token_position=pos)
        restorers = build_restorer_set(mock_tiny_lm, sender, rcv)
        # block_output at layer 1 reads after the whole block 1.
        assert self._layers(restorers, "attention_output") == [1]
        assert self._layers(restorers, "mlp_output") == [0, 1]

    def test_residual_block_input_excludes_its_layer(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        sender = _sender(mock_tiny_lm, 0)
        rcv = ReceiverSpec(
            kind="residual", layer=1, token_position=pos, residual_point="block_input"
        )
        restorers = build_restorer_set(mock_tiny_lm, sender, rcv)
        assert self._layers(restorers, "attention_output") == []
        assert self._layers(restorers, "mlp_output") == [0]

    def test_internal_never_includes_block_output(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        sender = _sender(mock_tiny_lm, 0)
        rcv = ReceiverSpec(kind="residual", layer=1, token_position=pos)
        comps = {
            u.component_type for u in build_restorer_set(mock_tiny_lm, sender, rcv)
        }
        assert comps <= {"attention_output", "mlp_output"}

    def test_restorers_freeze_at_receiver_position(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        # Sender and receiver read at distinct positions; restorers must use the
        # receiver's (the residual at a position is written only at that position).
        sender = AttentionHead(
            0,
            0,
            _pos(mock_tiny_lm, "sender_pos"),
            target_output=True,
            shape=(
                mock_tiny_lm.model.config.hidden_size
                // mock_tiny_lm.model.config.num_attention_heads,
            ),
        )
        rcv = ReceiverSpec(
            kind="mlp_input", layer=1, token_position=_pos(mock_tiny_lm, "recv_pos")
        )
        restorers = build_restorer_set(mock_tiny_lm, sender, rcv)
        assert restorers and all("Token-recv_pos" in u.id for u in restorers)


class TestSenderReachesReceiver:
    pytestmark = pytest.mark.property

    def test_output_always_reachable(self, mock_tiny_lm: LMPipeline) -> None:
        n_layers = mock_tiny_lm.model.config.num_hidden_layers
        # Even a top-layer sender reaches the output (it reads past the whole stack).
        assert sender_reaches_receiver(
            mock_tiny_lm, _sender(mock_tiny_lm, n_layers - 1), OUTPUT
        )

    def test_upstream_sender_reaches_internal(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _last_token(mock_tiny_lm)
        rcv = ReceiverSpec(kind="mlp_input", layer=1, token_position=pos)
        assert sender_reaches_receiver(mock_tiny_lm, _sender(mock_tiny_lm, 0), rcv)

    def test_downstream_sender_does_not_reach(self, mock_tiny_lm: LMPipeline) -> None:
        # Sender at layer 1 (write depth 2) vs mlp_input at layer 0 (read depth 1).
        pos = _last_token(mock_tiny_lm)
        rcv = ReceiverSpec(kind="mlp_input", layer=0, token_position=pos)
        assert not sender_reaches_receiver(mock_tiny_lm, _sender(mock_tiny_lm, 1), rcv)

    def test_same_layer_head_value_does_not_reach(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        # An attn sender at L writes at depth 2L; a value-input at L reads at 2L (before
        # attention L computes), so the same-layer head does not reach it.
        pos = _last_token(mock_tiny_lm)
        rcv = ReceiverSpec(kind="head_value_input", layer=1, head=0, token_position=pos)
        assert not sender_reaches_receiver(mock_tiny_lm, _sender(mock_tiny_lm, 1), rcv)


class TestBuildTarget:
    pytestmark = pytest.mark.property

    def test_two_groups_sender_then_restorers(self, mock_tiny_lm: LMPipeline) -> None:
        sender = _sender(mock_tiny_lm, 0)
        target = build_path_patching_target(mock_tiny_lm, sender)
        assert len(target) == 2
        assert [u.id for u in target[0]] == [sender.id]
        assert sender.id not in {u.id for u in target[1]}
        assert len(target[1]) >= 1

    def test_no_restorers_collapses_to_single_group(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """A top-layer sender under attention-only restore has no restorers, so the
        target collapses to the single sender group (not a degenerate empty
        second group). This pins the runner/builder agreement the empty case
        used to break — both route through ``group_path_patching_target``."""
        n_layers = mock_tiny_lm.model.config.num_hidden_layers
        sender = _sender(mock_tiny_lm, n_layers - 1)
        restorers = build_restorer_set(mock_tiny_lm, sender, restore=("attention",))
        assert restorers == []
        target = build_path_patching_target(
            mock_tiny_lm, sender, restore=("attention",)
        )
        assert len(target) == 1
        assert [u.id for u in target[0]] == [sender.id]
