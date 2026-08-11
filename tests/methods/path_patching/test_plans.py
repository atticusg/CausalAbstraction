"""Tests for the path-patching plan builder (``methods.path_patching.plans``).

Structure pins on the built :class:`~causalab.neural.plan.Plan`: which named
inputs exist per regime, one receiver-inject ``EditOp`` per receiver on the
``final`` input (the explicit dataflow that replaced pyvene's ``sorted_keys``
collect-order contract), the minimal pass structure the staged compiler
schedules, and the position wiring (sender at its own base/source positions,
restorers and receivers frozen at the receiver's). Mostly property tier;
``TestEdgePlanNumerics`` pins the executed plans' numbers (self-patch
identity / cross-patch motion) at the ``numerical_unit`` tier — full raw-hook
parity on genuine counterfactuals is pinned by the hook oracles and
``TestLeftPadGenerateParity`` in this package.
"""

from __future__ import annotations

import pytest
import torch

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.path_patching.plans import build_edge_plan
from causalab.methods.path_patching.targets import (
    OUTPUT,
    ReceiverSpec,
    build_receiver_site,
)
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.head_view import HeadSite
from causalab.neural.specs import SiteSpec
from causalab.neural.pipeline import LMPipeline
from causalab.neural.plan import EditOp, Plan, run_plan
from causalab.neural.positions import resolve_positions
from causalab.neural.staged import lower_staged
from causalab.neural.token_positions import TokenPosition, get_last_token_index


def _trace(text: str) -> CausalTrace:
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _dataset() -> list[CounterfactualExample]:
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


def _sender(pipeline: LMPipeline, layer: int = 0, head: int = 0) -> SiteSpec:
    hd = pipeline.model.config.hidden_size // pipeline.model.config.num_attention_heads
    return SiteSpec(
        fsite=FeaturizedSite(HeadSite("attention_value", layer, head)),
        positions=_last_token(pipeline),
        key=f"AttentionHead.L{layer}.H{head}.last_token",
        width=hd,
    )


def _stage_of(program, key) -> int:
    for i, stage in enumerate(program.stages):
        if any(key in group for group in stage):
            return i
    raise AssertionError(f"{key!r} not scheduled")


class TestPlanStructure:
    pytestmark = pytest.mark.property

    def test_output_receiver_saves_base_logits(self, mock_tiny_lm: LMPipeline) -> None:
        plan, key = build_edge_plan(mock_tiny_lm, _dataset(), _sender(mock_tiny_lm), [])
        assert key == "base"
        assert set(plan.inputs) == {"source", "base", "clean"}
        assert plan.save_logits == ("base",)
        assert all(op.input in ("base",) for op in plan.ops)

    def test_no_restorers_drops_clean_input(self, mock_tiny_lm: LMPipeline) -> None:
        plan, _ = build_edge_plan(
            mock_tiny_lm, _dataset(), _sender(mock_tiny_lm), [], restorer_sites=[]
        )
        assert set(plan.inputs) == {"source", "base"}
        assert len(plan.ops) == 1  # the sender interchange only

    def test_one_final_edit_per_receiver(self, mock_tiny_lm: LMPipeline) -> None:
        """Each receiver is one EditOp on the ``final`` input whose ReadSource
        addresses the patched ``base`` invoke — the per-receiver named edge that
        replaced pyvene's one-group-per-receiver + sorted_keys ordering."""
        pos = _last_token(mock_tiny_lm)
        specs = [
            ReceiverSpec(kind="head_value_input", layer=1, head=h, token_position=pos)
            for h in (0, 1)
        ]
        sites = [build_receiver_site(mock_tiny_lm, rs) for rs in specs]
        plan, key = build_edge_plan(
            mock_tiny_lm, _dataset(), _sender(mock_tiny_lm), sites, specs[0]
        )
        assert key == "final"
        final_ops = [op for op in plan.ops if op.input == "final"]
        assert [op.edit.site.site for op in final_ops] == [
            HeadSite("value", 1, 0),
            HeadSite("value", 1, 1),
        ]
        for op in final_ops:
            (rs,) = op.edit.read_sources
            assert rs.input == "base"  # v* read under the pass-1 interventions
            assert rs.value.site == op.edit.site.site

    def test_two_pass_receiver_edge_is_honored(self, mock_tiny_lm: LMPipeline) -> None:
        """The collect∘inject edge (``base`` → ``final``) is always served: the
        plan never lowers to ONE trace, and ``final`` either runs in a later
        stage than ``base`` (v* crosses as a saved value) or fuses with it
        behind an in-trace barrier (the compiler's pass-minimal choice when
        ``base`` did not itself consume in-trace)."""
        from causalab.neural.plan import StagingRequired

        pos = _last_token(mock_tiny_lm)
        spec = ReceiverSpec(kind="mlp_input", layer=1, token_position=pos)
        site = build_receiver_site(mock_tiny_lm, spec)
        plan, _ = build_edge_plan(
            mock_tiny_lm, _dataset(), _sender(mock_tiny_lm), [site], spec
        )
        with pytest.raises(StagingRequired):
            run_plan(mock_tiny_lm.model, plan, lowering="single")
        program = lower_staged(mock_tiny_lm.model, plan)
        assert program.num_traces >= 2
        base_stage, final_stage = (
            _stage_of(program, "base"),
            _stage_of(program, "final"),
        )
        if final_stage == base_stage:
            assert any(e.src == "base" and e.dst == "final" for e in program.in_trace)
        else:
            assert final_stage > base_stage

    def test_inputs_carry_position_ids(self, mock_tiny_lm: LMPipeline) -> None:
        """Every plan input goes through ``ensure_position_ids`` — the left-pad
        numbering that keeps absolute-position models correct (numerically
        pinned by ``TestLeftPadGenerateParity``)."""
        plan, _ = build_edge_plan(mock_tiny_lm, _dataset(), _sender(mock_tiny_lm), [])
        for name, inputs in plan.inputs.items():
            assert "position_ids" in inputs, f"input {name!r} lacks position_ids"


class TestPositionWiring:
    pytestmark = pytest.mark.property

    def test_restorers_and_receiver_freeze_at_receiver_position(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """Sender reads/writes at its own position; restorers and the receiver
        freeze at the *receiver's* (the residual at a position is written only
        by components at that position)."""
        first = TokenPosition(lambda inp: [0], mock_tiny_lm, id="first_token")
        spec = ReceiverSpec(kind="mlp_input", layer=1, token_position=first)
        site = build_receiver_site(mock_tiny_lm, spec)
        ds = _dataset()
        plan, _ = build_edge_plan(mock_tiny_lm, ds, _sender(mock_tiny_lm), [site], spec)

        base_traces = [ex["input"] for ex in ds]
        mask = plan.inputs["base"]["attention_mask"]
        last_rows = resolve_positions(
            _last_token(mock_tiny_lm), base_traces, mask, is_original=True
        )
        first_rows = resolve_positions(first, base_traces, mask, is_original=True)

        sender_op = plan.ops[0]
        assert isinstance(sender_op, EditOp) and sender_op.input == "base"
        assert sender_op.edit.positions == last_rows
        for op in plan.ops[1:]:
            assert op.edit.positions == first_rows

    def test_output_receiver_freezes_at_sender_position(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        # No internal read point: restorers fall back to the sender's position.
        ds = _dataset()
        plan, _ = build_edge_plan(mock_tiny_lm, ds, _sender(mock_tiny_lm), [], OUTPUT)
        mask = plan.inputs["base"]["attention_mask"]
        last_rows = resolve_positions(
            _last_token(mock_tiny_lm),
            [ex["input"] for ex in ds],
            mask,
            is_original=True,
        )
        assert all(op.edit.positions == last_rows for op in plan.ops)


class TestWidthGuards:
    pytestmark = pytest.mark.property

    def test_sender_base_source_width_mismatch_raises(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        # Base side selects 2 tokens, source side 1 — the interchange cannot
        # write one-to-one.
        def two_on_base(x):
            return [0, 1] if "hello" in x["raw_input"] else [0]

        pos = TokenPosition(two_on_base, mock_tiny_lm, id="uneven")
        hd = (
            mock_tiny_lm.model.config.hidden_size
            // mock_tiny_lm.model.config.num_attention_heads
        )
        sender = SiteSpec(
            fsite=FeaturizedSite(HeadSite("attention_value", 0, 0)),
            positions=pos,
            key="AttentionHead.L0.H0.custom",
            width=hd,
        )
        ds = [
            {
                "input": _trace("hello world"),
                "counterfactual_inputs": [_trace("blue green")],
            }
        ]
        with pytest.raises(ValueError, match="width differs between base and source"):
            build_edge_plan(mock_tiny_lm, ds, sender, [], restorer_sites=[])

    def test_ragged_sender_positions_raise(self, mock_tiny_lm: LMPipeline) -> None:
        def ragged(x):
            return [0, 1] if len(x["raw_input"].split()) >= 3 else [0]

        pos = TokenPosition(ragged, mock_tiny_lm, id="ragged")
        hd = (
            mock_tiny_lm.model.config.hidden_size
            // mock_tiny_lm.model.config.num_attention_heads
        )
        sender = SiteSpec(
            fsite=FeaturizedSite(HeadSite("attention_value", 0, 0)),
            positions=pos,
            key="AttentionHead.L0.H0.custom",
            width=hd,
        )
        ds = [
            {
                "input": _trace("the quick brown fox"),
                "counterfactual_inputs": [_trace("a slow lazy dog")],
            },
            {"input": _trace("hi there"), "counterfactual_inputs": [_trace("yo yo")]},
        ]
        with pytest.raises(
            ValueError, match="sender selects a variable number of tokens"
        ):
            build_edge_plan(mock_tiny_lm, ds, sender, [], restorer_sites=[])


class TestEdgePlanNumerics:
    """Fixed-weight numerics of the *executed* plans on the tiny stub (CPU).

    A self-patch (counterfactual text == base text) must reproduce the clean
    forward's logits through the full intervention stack — the sender
    interchange, every restorer freeze, and (two-pass) the receiver's
    collect∘inject edge all write back values numerically equal to what they
    overwrite, so any position miswiring or value-dataflow error shows up as a
    logits difference. The cross-patch companion pins the opposite direction:
    a genuine counterfactual must move the logits, so a silently-inert edge
    (e.g. positions resolving to nothing) cannot pass the identity tests
    vacuously. Raw-hook parity on genuine counterfactuals is the oracle files'
    job; these are the builder's own input–output pins.
    """

    pytestmark = pytest.mark.numerical_unit

    @staticmethod
    def _self_patch_dataset() -> list[CounterfactualExample]:
        return [
            {"input": _trace(text), "counterfactual_inputs": [_trace(text)]}
            for text in ("hello world", "red sky")
        ]

    @staticmethod
    def _clean_logits(pipeline: LMPipeline, plan: Plan) -> torch.Tensor:
        """The unintervened forward over the plan's own ``base`` tensors."""
        clean = run_plan(
            pipeline.model,
            Plan(inputs={"base": plan.inputs["base"]}, save_logits=("base",)),
        )
        return clean.logits["base"]

    def test_output_receiver_self_patch_reproduces_clean_logits(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        plan, key = build_edge_plan(
            mock_tiny_lm, self._self_patch_dataset(), _sender(mock_tiny_lm), []
        )
        assert len(plan.ops) > 1  # sender + a non-empty restorer set
        patched = run_plan(mock_tiny_lm.model, plan).logits[key]
        torch.testing.assert_close(patched, self._clean_logits(mock_tiny_lm, plan))

    def test_two_pass_receiver_self_patch_reproduces_clean_logits(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """Exercises PASS 2's ``base`` → ``final`` v* edge numerically: under
        identity PASS-1 interventions the collected receiver value equals the
        clean one, so re-injecting it must leave the logits unchanged."""
        pos = _last_token(mock_tiny_lm)
        spec = ReceiverSpec(kind="mlp_input", layer=1, token_position=pos)
        site = build_receiver_site(mock_tiny_lm, spec)
        assert site is not None  # internal receivers always resolve
        plan, key = build_edge_plan(
            mock_tiny_lm,
            self._self_patch_dataset(),
            _sender(mock_tiny_lm),
            [site],
            spec,
        )
        assert key == "final"
        patched = run_plan(mock_tiny_lm.model, plan).logits[key]
        torch.testing.assert_close(patched, self._clean_logits(mock_tiny_lm, plan))

    def test_cross_patch_moves_last_token_logits(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        plan, key = build_edge_plan(
            mock_tiny_lm, _dataset(), _sender(mock_tiny_lm), [], restorer_sites=[]
        )
        patched = run_plan(mock_tiny_lm.model, plan).logits[key]
        clean = self._clean_logits(mock_tiny_lm, plan)
        delta = (patched[:, -1, :] - clean[:, -1, :]).abs().max()
        assert float(delta) > 0.0, "the sender interchange left the logits inert"
