"""Tests for path-patching target construction (``methods.path_patching.targets``).

The restorer set *is* the estimand definition, so these pin its composition: only
``attention_output`` / ``mlp_output`` (never ``block_output``), the sender
excluded, the right components per ``restore`` selection, and the right layer
range above the sender. Property tier — shape/membership contracts on the tiny
model's config, no generation. Freeze *positions* are the plan builder's job and
are pinned in ``test_plans.py``.
"""

from __future__ import annotations

import pytest

from causalab.methods.path_patching.targets import (
    OUTPUT,
    ReceiverSpec,
    build_receiver_site,
    build_restorer_sites,
    sender_reaches_receiver,
)
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.head_view import HeadSite
from causalab.neural.specs import SiteSpec
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import Site
from causalab.neural.token_positions import TokenPosition, get_last_token_index


def _last_token(pipeline: LMPipeline) -> TokenPosition:
    return TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline), pipeline, id="last_token"
    )


def _sender(pipeline: LMPipeline, layer: int, head: int = 0) -> SiteSpec:
    hd = pipeline.model.config.hidden_size // pipeline.model.config.num_attention_heads
    return SiteSpec(
        fsite=FeaturizedSite(HeadSite("attention_value", layer, head)),
        positions=_last_token(pipeline),
        key=f"AttentionHead.L{layer}.H{head}.last_token",
        width=hd,
    )


class TestRestorerSet:
    pytestmark = pytest.mark.property

    def test_never_includes_block_output_or_sender(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        sender = _sender(mock_tiny_lm, 0)
        restorers = build_restorer_sites(mock_tiny_lm, sender)
        comps = {s.component for s in restorers}
        assert "block_output" not in comps
        assert comps <= {"attention_output", "mlp_output"}
        # the sender is an attention *head*; whole-attention restorers never sit
        # at a depth that could include its own write (layer-0 attention).
        assert not any(
            s.component == "attention_output" and s.layer == 0 for s in restorers
        )

    def test_full_restore_covers_attention_and_mlp_above_sender(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        n_layers = mock_tiny_lm.model.config.num_hidden_layers
        sender = _sender(mock_tiny_lm, 0)
        restorers = build_restorer_sites(
            mock_tiny_lm, sender, restore=("attention", "mlp")
        )
        layers_attn = sorted(
            s.layer for s in restorers if s.component == "attention_output"
        )
        layers_mlp = sorted(s.layer for s in restorers if s.component == "mlp_output")
        # attention frozen for every layer strictly above the sender
        assert layers_attn == list(range(1, n_layers))
        # mlp frozen for the sender's own layer (attn-head sender) and all above
        assert layers_mlp == list(range(0, n_layers))

    def test_attention_only_omits_all_mlp(self, mock_tiny_lm: LMPipeline) -> None:
        sender = _sender(mock_tiny_lm, 0)
        restorers = build_restorer_sites(mock_tiny_lm, sender, restore=("attention",))
        assert all(s.component == "attention_output" for s in restorers)

    def test_mlp_only_omits_all_attention(self, mock_tiny_lm: LMPipeline) -> None:
        sender = _sender(mock_tiny_lm, 0)
        restorers = build_restorer_sites(mock_tiny_lm, sender, restore=("mlp",))
        assert all(s.component == "mlp_output" for s in restorers)
        # still includes the sender-layer MLP (attn-head sender)
        assert any(s.layer == 0 for s in restorers)

    def test_top_layer_sender_full_is_sender_mlp_only(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """A sender in the top layer has no layers above it, so the only restorer
        is its own layer's MLP output."""
        n_layers = mock_tiny_lm.model.config.num_hidden_layers
        sender = _sender(mock_tiny_lm, n_layers - 1)
        restorers = build_restorer_sites(
            mock_tiny_lm, sender, restore=("attention", "mlp")
        )
        assert [s.component for s in restorers] == ["mlp_output"]
        assert restorers[0].layer == n_layers - 1

    def test_empty_restore_raises(self, mock_tiny_lm: LMPipeline) -> None:
        with pytest.raises(ValueError, match="at least one component family"):
            build_restorer_sites(mock_tiny_lm, _sender(mock_tiny_lm, 0), restore=())

    def test_invalid_restore_family_raises(self, mock_tiny_lm: LMPipeline) -> None:
        with pytest.raises(ValueError, match="not valid"):
            build_restorer_sites(
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


class TestReceiverSite:
    pytestmark = pytest.mark.property

    def test_output_has_no_site(self, mock_tiny_lm: LMPipeline) -> None:
        assert build_receiver_site(mock_tiny_lm, OUTPUT) is None

    def test_head_value_input_targets_value_head_site(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        site = build_receiver_site(
            mock_tiny_lm,
            ReceiverSpec(kind="head_value_input", layer=1, head=0, token_position=pos),
        )
        # The per-head value vector — the realizable "v of head h" read point.
        assert site == HeadSite("value", 1, 0)

    def test_head_value_input_guards_out_of_range_query_head(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        # A head index at num_attention_heads is out of the query-head range.
        n_head = mock_tiny_lm.model.config.num_attention_heads
        pos = _last_token(mock_tiny_lm)
        with pytest.raises(ValueError, match="out of range"):
            build_receiver_site(
                mock_tiny_lm,
                ReceiverSpec(
                    kind="head_value_input", layer=1, head=n_head, token_position=pos
                ),
            )

    def test_head_query_input_targets_query_head_site(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        site = build_receiver_site(
            mock_tiny_lm,
            ReceiverSpec(kind="head_query_input", layer=1, head=0, token_position=pos),
        )
        # The per-head (pre-RoPE) query vector — the "q of head h" read point.
        assert site == HeadSite("query", 1, 0)

    def test_head_query_input_guards_out_of_range_head(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        n_head = mock_tiny_lm.model.config.num_attention_heads
        pos = _last_token(mock_tiny_lm)
        with pytest.raises(ValueError, match="out of range"):
            build_receiver_site(
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
            site = build_receiver_site(
                gqa_tiny_lm,
                ReceiverSpec(
                    kind="head_query_input", layer=1, head=qh, token_position=pos
                ),
            )
            assert site == HeadSite("query", 1, qh)  # the query head itself

    def test_mlp_input_targets_mlp_input(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _last_token(mock_tiny_lm)
        site = build_receiver_site(
            mock_tiny_lm, ReceiverSpec(kind="mlp_input", layer=1, token_position=pos)
        )
        assert site == Site("mlp_input", 1)

    def test_gqa_query_head_maps_to_kv_group(self, gqa_tiny_lm: LMPipeline) -> None:
        # On GQA the value vector is shared per KV group, so a query head maps to
        # `head // (n_head // n_kv)`. The value HeadSite must address that KV-head
        # index (HeadSite("value", ...) is in KV-head space).
        cfg = gqa_tiny_lm.model.config
        n_head, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        assert n_kv < n_head  # fixture really is grouped-query
        group = n_head // n_kv
        pos = _last_token(gqa_tiny_lm)
        for qh in range(n_head):
            site = build_receiver_site(
                gqa_tiny_lm,
                ReceiverSpec(
                    kind="head_value_input", layer=1, head=qh, token_position=pos
                ),
            )
            assert site == HeadSite("value", 1, qh // group)  # KV-group index

    def test_decoupled_head_dim_value_receiver_supported(
        self, gqa_decoupled_head_dim_lm: LMPipeline
    ) -> None:
        # pyvene 0.1.8 sliced value vectors at hidden // n_head, so decoupled
        # head_dim (Qwen3) was refused. HeadSite honours config.head_dim, so the
        # value receiver now resolves (the finite end-to-end scan is pinned in
        # test_run.py).
        cfg = gqa_decoupled_head_dim_lm.model.config
        assert cfg.head_dim != cfg.hidden_size // cfg.num_attention_heads
        pos = _last_token(gqa_decoupled_head_dim_lm)
        site = build_receiver_site(
            gqa_decoupled_head_dim_lm,
            ReceiverSpec(kind="head_value_input", layer=1, head=0, token_position=pos),
        )
        assert site == HeadSite("value", 1, 0)

    def test_residual_point_selects_block_component(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        out = build_receiver_site(
            mock_tiny_lm, ReceiverSpec(kind="residual", layer=1, token_position=pos)
        )
        inp = build_receiver_site(
            mock_tiny_lm,
            ReceiverSpec(
                kind="residual",
                layer=1,
                token_position=pos,
                residual_point="block_input",
            ),
        )
        assert out == Site("block_output", 1)
        assert inp == Site("block_input", 1)


class TestInternalRestorerRange:
    """The restorer range must stop at the receiver's read point, with the right
    mid-block membership. On the 2-layer tiny model, a sender at layer 0 and a
    receiver at layer 1 makes the attn(1)/mlp(1) membership the discriminating
    test across receiver kinds (full restore)."""

    pytestmark = pytest.mark.property

    def _layers(self, restorers, component):
        return sorted(s.layer for s in restorers if s.component == component)

    def test_head_value_input_excludes_its_layer_attention(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        sender = _sender(mock_tiny_lm, 0)
        rcv = ReceiverSpec(kind="head_value_input", layer=1, head=0, token_position=pos)
        restorers = build_restorer_sites(mock_tiny_lm, sender, rcv)
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
        restorers = build_restorer_sites(mock_tiny_lm, sender, rcv)
        # Reads after layer-1 attention but before layer-1 MLP.
        assert self._layers(restorers, "attention_output") == [1]
        assert self._layers(restorers, "mlp_output") == [0]

    def test_residual_block_output_includes_full_layer(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        sender = _sender(mock_tiny_lm, 0)
        rcv = ReceiverSpec(kind="residual", layer=1, token_position=pos)
        restorers = build_restorer_sites(mock_tiny_lm, sender, rcv)
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
        restorers = build_restorer_sites(mock_tiny_lm, sender, rcv)
        assert self._layers(restorers, "attention_output") == []
        assert self._layers(restorers, "mlp_output") == [0]

    def test_internal_never_includes_block_output(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        sender = _sender(mock_tiny_lm, 0)
        rcv = ReceiverSpec(kind="residual", layer=1, token_position=pos)
        comps = {s.component for s in build_restorer_sites(mock_tiny_lm, sender, rcv)}
        assert comps <= {"attention_output", "mlp_output"}


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
