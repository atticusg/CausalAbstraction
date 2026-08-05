"""Tests for receiver-*set* path patching (``run_path_patching_set`` /
``run_path_patching_set_scan``) — patching one sender into several receivers at
once (the IOI Fig. 4 / Fig. 5 case).

The tiny-random model has no IOI behaviour, so these pin the two correctness
invariants the feature rests on rather than circuit identity:

* **Degeneracy** — a 1-element set reproduces the single-receiver run exactly.
* **Order-invariance** — the collected ``v*`` order matches the injected order, so
  a 2-element set scores identically regardless of receiver order (the runtime
  guard on the collect-order = config-add-order contract).

plus the structural guards: one group per receiver, the deepest-receiver
restorer/reachability rule, and the set preconditions (shared token position,
internal-only).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.metric import InterchangeMetric
from causalab.methods.path_patching.run import (
    _validate_receiver_set,
    run_path_patching_scan,
    run_path_patching_set,
    run_path_patching_set_scan,
)
from causalab.methods.path_patching.targets import (
    OUTPUT,
    ReceiverSpec,
    build_receiver_unit,
    build_restorer_set,
    deepest_receiver,
    sender_reaches_any,
)
from causalab.neural.activations.engine import build_interventions
from causalab.neural.activations.targets import build_attention_head_targets
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition, get_last_token_index


def _trace(text: str) -> CausalTrace:
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _dataset() -> list[dict[str, Any]]:
    return [
        {
            "input": _trace("hello world"),
            "counterfactual_inputs": [_trace("blue green")],
        },
        {"input": _trace("red sky"), "counterfactual_inputs": [_trace("cat dog")]},
    ]


def _last_token(pipeline: LMPipeline) -> TokenPosition:
    return TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline), pipeline, id="last_token"
    )


def _first_logit_metric() -> InterchangeMetric:
    def fn(
        intervention_output: dict[str, Any],
        _expected: dict[str, Any],
        _original: dict[str, Any],
    ) -> float:
        idx = intervention_output["example_idx"]
        return float(intervention_output["scores"][0][idx][0])

    return InterchangeMetric(
        fn=fn,
        needs_causal_expected=False,
        needs_original_output=False,
        needs_scores=True,
    )


def _sender(pipeline: LMPipeline, layer: int, head: int) -> Any:
    pos = _last_token(pipeline)
    return build_attention_head_targets(pipeline, [layer], [head], pos)[
        (layer, head)
    ].flatten()[0]


class TestDegeneracy:
    """A 1-element receiver set must reproduce the single-receiver run exactly: the
    set engine reduces to today's code when ``len(receivers) == 1``."""

    pytestmark = pytest.mark.numerical_unit

    def test_one_element_set_matches_single_receiver(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        receiver = ReceiverSpec(
            kind="head_value_input", layer=1, head=0, token_position=pos
        )
        senders = {(0, 0): _sender(mock_tiny_lm, 0, 0)}

        single = run_path_patching_scan(
            mock_tiny_lm,
            _dataset(),
            senders,
            metric=_first_logit_metric(),
            receiver=receiver,
            restore=("attention",),
            batch_size=2,
        )
        as_set = run_path_patching_set_scan(
            mock_tiny_lm,
            _dataset(),
            senders,
            receiver_specs=[receiver],
            metric=_first_logit_metric(),
            restore=("attention",),
            batch_size=2,
        )
        assert single.keys() == as_set.keys()
        for key in single:
            assert single[key] == pytest.approx(as_set[key], abs=1e-6)


class TestOrderInvariance:
    """The same receiver list drives PASS-1 collect and PASS-2 inject, so the
    collected ``v*`` order matches the injected order regardless of how the set is
    listed: ``{A, B}`` and ``{B, A}`` must score identically. This is the runtime
    guard on the collect-order ↔ sorted-keys contract."""

    pytestmark = pytest.mark.property

    def test_set_score_independent_of_receiver_order(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        a = ReceiverSpec(kind="head_value_input", layer=1, head=0, token_position=pos)
        b = ReceiverSpec(kind="head_value_input", layer=1, head=1, token_position=pos)
        senders = {(0, 0): _sender(mock_tiny_lm, 0, 0)}

        ab = run_path_patching_set_scan(
            mock_tiny_lm,
            _dataset(),
            senders,
            receiver_specs=[a, b],
            metric=_first_logit_metric(),
            restore=("attention",),
            batch_size=2,
        )
        ba = run_path_patching_set_scan(
            mock_tiny_lm,
            _dataset(),
            senders,
            receiver_specs=[b, a],
            metric=_first_logit_metric(),
            restore=("attention",),
            batch_size=2,
        )
        for key in ab:
            assert ab[key] == pytest.approx(ba[key], abs=1e-6)


class TestPass2Grouping:
    """PASS 2 puts each receiver in its own group, so the model has one
    intervention group per receiver — the batch validation needs one source
    representation per group, and PASS 2 injects one v* per receiver."""

    pytestmark = pytest.mark.unit

    def test_one_intervention_per_receiver(self, mock_tiny_lm: LMPipeline) -> None:
        """Each receiver gets its own intervention, so PASS 2 injects one v* per
        receiver rather than sharing one across them."""
        pos = _last_token(mock_tiny_lm)
        receivers = [
            build_receiver_unit(
                mock_tiny_lm,
                ReceiverSpec(
                    kind="head_value_input", layer=1, head=h, token_position=pos
                ),
            )
            for h in (0, 1)
        ]
        interventions = build_interventions(receivers, "interchange")
        assert len(interventions) == len(receivers)
        assert len({id(i) for i in interventions}) == len(receivers)


class TestDeepestReceiver:
    """The restorer set and reachability gate for a set use the deepest receiver
    (max read depth); reaching it ⇔ reaching at least one member."""

    pytestmark = pytest.mark.unit

    def test_deepest_is_max_read_depth(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _last_token(mock_tiny_lm)
        shallow = ReceiverSpec(
            kind="head_query_input", layer=0, head=0, token_position=pos
        )
        deep = ReceiverSpec(
            kind="head_query_input", layer=1, head=0, token_position=pos
        )
        assert deepest_receiver(mock_tiny_lm, [shallow, deep]) is deep
        assert deepest_receiver(mock_tiny_lm, [deep, shallow]) is deep

    def test_restorers_built_against_deepest(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _last_token(mock_tiny_lm)
        specs = [
            ReceiverSpec(kind="head_query_input", layer=0, head=0, token_position=pos),
            ReceiverSpec(kind="head_query_input", layer=1, head=0, token_position=pos),
        ]
        sender = _sender(mock_tiny_lm, 0, 0)
        deepest = deepest_receiver(mock_tiny_lm, specs)
        # The union restorer set is exactly build_restorer_set against the deepest
        # member — the contract run_path_patching_set_scan relies on.
        expected = build_restorer_set(
            mock_tiny_lm, sender, deepest, restore=("attention",)
        )
        against_last = build_restorer_set(
            mock_tiny_lm, sender, specs[-1], restore=("attention",)
        )
        assert [u.id for u in expected] == [u.id for u in against_last]

    def test_sender_reaches_any_matches_deepest(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _last_token(mock_tiny_lm)
        n_layers = mock_tiny_lm.model.config.num_hidden_layers
        specs = [
            ReceiverSpec(
                kind="head_query_input", layer=n_layers - 1, head=0, token_position=pos
            )
        ]
        upstream = _sender(mock_tiny_lm, 0, 0)
        # A sender at the deepest receiver's own layer reads before it (value/query
        # are projected from the entering residual), so it does NOT reach the set.
        at_receiver = _sender(mock_tiny_lm, n_layers - 1, 0)
        assert sender_reaches_any(mock_tiny_lm, upstream, specs) is True
        assert sender_reaches_any(mock_tiny_lm, at_receiver, specs) is False


class TestScanReachability:
    """A sender that reaches no receiver scores nan (rather than raising as the
    single-receiver scan does), so disconnected layers drop out and the grid stays
    rectangular."""

    pytestmark = pytest.mark.property

    def test_unreachable_sender_scores_nan(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _last_token(mock_tiny_lm)
        n_layers = mock_tiny_lm.model.config.num_hidden_layers
        specs = [
            ReceiverSpec(
                kind="head_query_input", layer=n_layers - 1, head=0, token_position=pos
            )
        ]
        # Sender at the receiver's own layer: structurally unreachable.
        senders = {(n_layers - 1, 0): _sender(mock_tiny_lm, n_layers - 1, 0)}
        scores = run_path_patching_set_scan(
            mock_tiny_lm,
            _dataset(),
            senders,
            receiver_specs=specs,
            metric=_first_logit_metric(),
            restore=("attention",),
            batch_size=2,
        )
        assert all(torch.isnan(torch.tensor(v)) for v in scores.values()), scores


class TestSetPreconditions:
    """A receiver set must be internal-only and share one token position."""

    pytestmark = pytest.mark.unit

    def test_mixed_token_position_raises(self, mock_tiny_lm: LMPipeline) -> None:
        end = _last_token(mock_tiny_lm)
        other = TokenPosition(lambda inp: [0], mock_tiny_lm, id="first_token")
        specs = [
            ReceiverSpec(kind="head_value_input", layer=1, head=0, token_position=end),
            ReceiverSpec(
                kind="head_value_input", layer=1, head=1, token_position=other
            ),
        ]
        with pytest.raises(ValueError, match="share one token position"):
            _validate_receiver_set(specs)

    def test_output_in_set_raises(self, mock_tiny_lm: LMPipeline) -> None:
        with pytest.raises(ValueError, match="internal-receiver only"):
            _validate_receiver_set([OUTPUT])

    def test_empty_set_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _validate_receiver_set([])

    def test_single_sender_reaches_none_raises(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _last_token(mock_tiny_lm)
        n_layers = mock_tiny_lm.model.config.num_hidden_layers
        specs = [
            ReceiverSpec(
                kind="head_query_input", layer=n_layers - 1, head=0, token_position=pos
            )
        ]
        sender = _sender(mock_tiny_lm, n_layers - 1, 0)
        with pytest.raises(ValueError, match="reaches no receiver"):
            run_path_patching_set(mock_tiny_lm, _dataset(), sender, specs)

    def test_value_only_distinct_position_objects_ok(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """Two value-equal-but-distinct last_token positions share one read point, so
        the position guard (which dedupes by ``id``) must NOT trip."""
        a = _last_token(mock_tiny_lm)
        b = _last_token(mock_tiny_lm)  # distinct object, same id="last_token"
        assert a is not b
        _validate_receiver_set(
            [
                ReceiverSpec(
                    kind="head_value_input", layer=1, head=0, token_position=a
                ),
                ReceiverSpec(
                    kind="head_value_input", layer=1, head=1, token_position=b
                ),
            ]
        )

    def test_gqa_value_set_collision_raises(self, gqa_tiny_lm: LMPipeline) -> None:
        """On a grouped-query model, two query heads in the same KV group remap to the
        same value unit. A head_value_input set over them would collapse to one
        collect/inject site, so building the set must raise rather than misbehave."""
        cfg = gqa_tiny_lm.model.config
        group = cfg.num_attention_heads // cfg.num_key_value_heads
        assert group >= 2  # heads 0 and 1 share KV group 0
        pos = _last_token(gqa_tiny_lm)
        specs = [
            ReceiverSpec(kind="head_value_input", layer=1, head=0, token_position=pos),
            ReceiverSpec(kind="head_value_input", layer=1, head=1, token_position=pos),
        ]
        with pytest.raises(ValueError, match="same model unit"):
            run_path_patching_set(
                gqa_tiny_lm, _dataset(), _sender(gqa_tiny_lm, 0, 0), specs
            )
