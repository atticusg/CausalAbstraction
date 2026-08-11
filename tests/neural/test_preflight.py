"""Tests for :mod:`causalab.neural.preflight` — the CAP5 ``scan()`` gate (#458).

Tiers (``causalab/neural`` owes ``unit`` + ``property``, docs/TESTS.md):

* ``unit`` — the model-free gates and the report contract: headroom and
  static-position failures verdict without any model access (asserted by
  passing ``model=None``, the test_plan refusal-ordering pattern), the
  probe-level "scan unsupported" classification on a stub whose ``scan``
  raises, and ``raise_if_failed`` semantics.
* ``property`` — the real gate on tiny-random Llama (CPU), pinning the
  issue's contract: a bad position spec **fails at preflight with the legible
  error and the real run agrees** (raises the same exception class); a good
  plan is ``clean`` and the real run executes it (with and without the
  ``run_plan(preflight=True)`` wiring, same values); a plan whose transform
  fake tensors cannot express is ``unsupported`` — distinguished from both
  ``clean`` and ``failed`` — while the *real* run of that same plan succeeds.

Models come from the fresh (uncached) factories in ``tests/_helpers/tiny.py``
— never the session-cached ``tiny_random_model`` singleton (leftover hooks
break later nnsight traces; see the factory docstrings). Multi-input plans
need frame-aligned inputs, so paired prompts are tokenized together.
"""

from __future__ import annotations

import dataclasses
import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from nnterp import StandardizedTransformer

from causalab.neural.edit import Edit, ReadSource
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.persistent import install_edits, uninstall_edits
from causalab.neural.plan import (
    CollectOp,
    EditOp,
    GenerateSpec,
    GradientRequest,
    Plan,
    run_plan,
)
from causalab.neural.preflight import (
    PreflightError,
    PreflightReport,
    _classify_not_implemented,
    check_scan_support,
    preflight_plan,
)
from causalab.neural.site import Site

from tests._helpers.tiny import fresh_tiny_random_llama

_BASE_TEXT = "the quick brown fox jumps"
_SOURCE_TEXT = "a slow green turtle sleeps deeply"


def _resid(layer: int) -> FeaturizedSite:
    return FeaturizedSite(Site("block_output", layer))


def _pretokenized(batch: int = 2, length: int = 5) -> dict[str, torch.Tensor]:
    """A model-free pre-tokenized input: the static gates only read shapes."""
    return {
        "input_ids": torch.ones(batch, length, dtype=torch.long),
        "attention_mask": torch.ones(batch, length, dtype=torch.long),
    }


class _NoScanModel:
    """A model whose ``scan()`` cannot run — the probe-level unsupported case."""

    def scan(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("this forward has no meta kernels")


# --------------------------------------------------------------------------- #
#  unit — model-free gates + the report contract                               #
# --------------------------------------------------------------------------- #
class TestPreflightUnit:
    pytestmark = pytest.mark.unit

    def test_headroom_fails_model_free(self) -> None:
        # Gradient plans get no scan verdict (the fake-mode dry run never
        # executes the backward), so the gate fails them up front — even
        # though run_plan executes single-input gradient plans (CAP3/EU2).
        plan = Plan(
            inputs={"base": _pretokenized()},
            ops=(CollectOp("base", _resid(0), key="h"),),
            gradients=GradientRequest(loss=lambda c: 0.0, wrt=("h",)),
        )
        report = preflight_plan(None, plan)
        assert report.status == "failed"
        assert report.stage == "headroom"
        assert "gradients" in report.error

    def test_generation_plan_is_unsupported_model_free(self) -> None:
        # A generation plan (CAP2, #468) gets NO verdict — scan cannot express
        # the KV-cached decode loop, so clean/failed would both be bogus. The
        # gate is model-free (asserted by model=None) and fires before any
        # position check (stepped positions resolve in the one-token step
        # frame, not the input frame the static check reads).
        plan = Plan(
            inputs={"base": _pretokenized()},
            ops=(CollectOp("base", _resid(0), key="h", positions=[-1], step=1),),
            generate=GenerateSpec(max_new_tokens=2),
        )
        report = preflight_plan(None, plan)
        assert report.status == "unsupported"
        assert report.stage == "generate"
        assert "not preflightable" in report.error
        report.raise_if_failed()  # no verdict, not a failure

    def test_flat_position_out_of_bounds_fails_model_free(self) -> None:
        plan = Plan(
            inputs={"base": _pretokenized(length=5)},
            ops=(CollectOp("base", _resid(0), key="h", positions=[7]),),
        )
        report = preflight_plan(None, plan)
        assert report.status == "failed"
        assert report.stage == "positions"
        assert "out of bounds" in report.error
        assert "padded length 5" in report.error

    def test_negative_positions_within_frame_pass_the_static_gate(self) -> None:
        # [-1] is the canonical last-token spec; the static gate must not
        # reject it. The plan then proceeds to the scan probe, which on this
        # scan-less stub reports unsupported — proving the positions gate
        # passed.
        plan = Plan(
            inputs={"base": _pretokenized(length=5)},
            ops=(CollectOp("base", _resid(0), key="h", positions=[-1]),),
        )
        report = preflight_plan(_NoScanModel(), plan)
        assert report.stage == "scan-probe"

    def test_negative_position_beyond_frame_fails(self) -> None:
        plan = Plan(
            inputs={"base": _pretokenized(length=5)},
            ops=(CollectOp("base", _resid(0), key="h", positions=[-6]),),
        )
        report = preflight_plan(None, plan)
        assert report.status == "failed"
        assert report.stage == "positions"

    def test_per_row_out_of_bounds_names_the_example(self) -> None:
        plan = Plan(
            inputs={"base": _pretokenized(batch=2, length=5)},
            ops=(CollectOp("base", _resid(0), key="h", positions=[[1], [9]]),),
        )
        report = preflight_plan(None, plan)
        assert report.status == "failed"
        assert report.stage == "positions"
        assert "example 1" in report.error

    def test_per_row_count_must_match_batch(self) -> None:
        plan = Plan(
            inputs={"base": _pretokenized(batch=2, length=5)},
            ops=(CollectOp("base", _resid(0), key="h", positions=[[1], [1], [1]]),),
        )
        report = preflight_plan(None, plan)
        assert report.status == "failed"
        assert report.stage == "positions"
        assert "3 per-row position rows" in report.error
        assert "batch of 2" in report.error

    def test_read_source_positions_checked_on_their_own_input(self) -> None:
        # The read-source addresses the *source* input (length 4): position 4
        # is out of bounds there even though the base frame (length 8) allows it.
        edit = Edit(
            _resid(0),
            g=lambda f, f_src: f_src,
            read_sources=(ReadSource(_resid(0), positions=[4], input="source"),),
            positions=[-1],
        )
        plan = Plan(
            inputs={
                "base": _pretokenized(length=8),
                "source": _pretokenized(length=4),
            },
            ops=(EditOp("base", edit),),
        )
        report = preflight_plan(None, plan)
        assert report.status == "failed"
        assert report.stage == "positions"
        assert "read_sources[0]" in report.error
        assert "'source'" in report.error

    def test_scanless_model_is_unsupported_not_failed(self) -> None:
        plan = Plan(
            inputs={"base": _pretokenized()},
            ops=(CollectOp("base", _resid(0), key="h", positions=[-1]),),
        )
        report = preflight_plan(_NoScanModel(), plan)
        assert report.status == "unsupported"
        assert report.stage == "scan-probe"
        assert "no meta kernels" in report.error
        assert not report.ok
        report.raise_if_failed()  # unsupported is a missing verdict, not a failure

    def test_check_scan_support_reports_the_error(self) -> None:
        assert "no meta kernels" in check_scan_support(_NoScanModel())

    def test_raise_if_failed_contract(self) -> None:
        PreflightReport("clean").raise_if_failed()
        PreflightReport("unsupported", stage="scan-probe", error="x").raise_if_failed()
        failed = PreflightReport("failed", stage="positions", error="pos 7 oob")
        assert not failed.ok
        with pytest.raises(PreflightError, match="'positions' gate: pos 7 oob"):
            failed.raise_if_failed()

    def test_clean_report_is_ok(self) -> None:
        assert PreflightReport("clean").ok


class TestNotImplementedClassificationUnit:
    """The ``failed``/``unsupported`` split for ``NotImplementedError`` — the
    review-flagged fragility (#469): torch has re-worded its missing-meta-
    kernel error across releases, and a wording drift past every marker would
    flip a genuinely *unsupported* op to ``failed``, hard-blocking a run that
    would have succeeded. The first test is the loud canary: it raises the
    error from the **installed** torch, so a torch bump that drifts fails
    here instead of silently flipping verdicts in the field."""

    pytestmark = pytest.mark.unit

    def test_installed_torch_meta_kernel_wording_still_recognized(self) -> None:
        # bincount has no meta kernel (data-dependent output length) — a
        # stable way to raise the real dispatcher error, no model needed.
        with pytest.raises(NotImplementedError) as excinfo:
            torch.bincount(torch.zeros(4, dtype=torch.long, device="meta"))
        assert _classify_not_implemented(excinfo.value) == "unsupported"

    def test_legacy_torch_dispatch_wording_recognized(self) -> None:
        # torch ≤2.3 wording — kept recognized so a downgrade/backport can't
        # flip verdicts either.
        exc = NotImplementedError(
            "Could not run 'aten::histc' with arguments from the 'Meta' "
            "backend. This could be because the operator doesn't exist for "
            "this backend."
        )
        assert _classify_not_implemented(exc) == "unsupported"

    def test_causalab_refusal_classified_failed(self) -> None:
        # A real honest-boundary refusal raised from causalab code — the
        # raising frame, not the message, is what classifies it.
        class WeirdMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.strange_proj = nn.Linear(2, 2)

        model = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=WeirdMLP())])
        )
        with pytest.raises(NotImplementedError) as excinfo:
            Site("mlp_activation", 0).mlp_activation_kind(model)
        assert _classify_not_implemented(excinfo.value) == "failed"

    def test_ambiguous_not_implemented_defaults_to_unsupported(self) -> None:
        # Neither a torch marker nor a causalab raising frame (raised from
        # test code): the safe direction is "no verdict", never a hard block.
        with pytest.raises(NotImplementedError) as excinfo:
            raise NotImplementedError("some future wording torch invents")
        assert _classify_not_implemented(excinfo.value) == "unsupported"


# --------------------------------------------------------------------------- #
#  property — the real gate on tiny-random (CPU); run agrees with the verdict  #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PreflightCase:
    st: StandardizedTransformer
    tok: Any

    def pair(self, t1: str, t2: str) -> tuple[dict, dict]:
        """Two single-row batches in ONE padded frame (tokenized together)."""
        enc = self.tok([t1, t2], padding=True, return_tensors="pt")

        def row(i: int) -> dict:
            return {
                "input_ids": enc["input_ids"][i : i + 1],
                "attention_mask": enc["attention_mask"][i : i + 1],
            }

        return row(0), row(1)


@pytest.fixture(scope="module")
def case() -> PreflightCase:
    raw, tok = fresh_tiny_random_llama()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    st = StandardizedTransformer(raw, tokenizer=tok, check_renaming=True)
    st.dispatch()
    return PreflightCase(st=st, tok=tok)


def _interchange_plan(
    case: PreflightCase,
    *,
    src_positions: Any = (-1,),
    dst_positions: Any = (-1,),
) -> Plan:
    base, source = case.pair(_BASE_TEXT, _SOURCE_TEXT)
    swap = Edit(
        _resid(1),
        g=lambda f, f_src: f_src,
        read_sources=(
            ReadSource(_resid(1), positions=list(src_positions), input="source"),
        ),
        positions=list(dst_positions),
    )
    return Plan(
        inputs={"source": source, "base": base},
        ops=(EditOp("base", swap),),
        save_logits=("base",),
    )


class TestPreflightScanProperty:
    pytestmark = pytest.mark.property

    def test_model_scan_support_probe(self, case: PreflightCase) -> None:
        assert check_scan_support(case.st) is None

    def test_clean_interchange_and_run_agrees(self, case: PreflightCase) -> None:
        plan = _interchange_plan(case)
        report = preflight_plan(case.st, plan)
        assert report == PreflightReport("clean")
        # The run agrees with the clean verdict — and the preflight-gated run
        # produces the same values as the ungated one (the scans are inert).
        plain = run_plan(case.st, plan)
        gated = run_plan(case.st, plan, preflight=True)
        torch.testing.assert_close(gated.logits["base"], plain.logits["base"])

    def test_layer_out_of_range_fails_and_run_agrees(self, case: PreflightCase) -> None:
        plan = Plan(
            inputs={"base": case.pair(_BASE_TEXT, _SOURCE_TEXT)[0]},
            ops=(CollectOp("base", _resid(99), key="h", positions=[-1]),),
        )
        report = preflight_plan(case.st, plan)
        assert report.status == "failed"
        assert report.stage == "scan-plan"
        assert "layer 99 out of range" in report.error
        with pytest.raises(IndexError, match="layer 99 out of range"):
            run_plan(case.st, plan)

    def test_position_out_of_bounds_fails_and_run_agrees(
        self, case: PreflightCase
    ) -> None:
        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        length = int(base["input_ids"].shape[-1])
        plan = Plan(
            inputs={"base": base},
            ops=(CollectOp("base", _resid(1), key="h", positions=[length + 3]),),
        )
        report = preflight_plan(case.st, plan)
        assert report.status == "failed"
        assert report.stage == "positions"
        assert "out of bounds" in report.error
        # Fake tensors would never see this (index bounds are data-dependent);
        # the real run hits it as a raw IndexError inside the trace.
        with pytest.raises(IndexError):
            run_plan(case.st, plan)

    def test_width_mismatch_fails_and_run_agrees(self, case: PreflightCase) -> None:
        # Two source positions feeding a one-position write — the classic
        # multi-token-variable-on-variable-length-pairs mismatch, expressed
        # per-row so both sides are explicit.
        plan = _interchange_plan(case, src_positions=[[0, 1]], dst_positions=[[-1]])
        report = preflight_plan(case.st, plan)
        assert report.status == "failed"
        assert report.stage == "scan-plan"
        assert "does not broadcast" in report.error
        assert "Widths must pair up" in report.error
        with pytest.raises(ValueError, match="does not broadcast"):
            run_plan(case.st, plan)

    def test_run_plan_preflight_gate_raises_before_any_forward(
        self, case: PreflightCase
    ) -> None:
        plan = Plan(
            inputs={"base": case.pair(_BASE_TEXT, _SOURCE_TEXT)[0]},
            ops=(CollectOp("base", _resid(99), key="h"),),
        )
        with pytest.raises(PreflightError, match="scan-plan"):
            run_plan(case.st, plan, preflight=True)

    def test_fake_incompatible_transform_is_unsupported_but_runs(
        self, case: PreflightCase
    ) -> None:
        # ``float(...)`` guards on a data-dependent value — fake tensors refuse
        # it, real tensors don't. The verdict must be "scan can't validate
        # this" (unsupported), NOT "the plan is wrong" (failed): the real run
        # of the very same plan succeeds.
        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        edit = Edit(
            _resid(1),
            g=lambda f: f * (0.0 * float(f.abs().sum().detach()) + 1.0),
            positions=[-1],
        )
        plan = Plan(
            inputs={"base": base},
            ops=(EditOp("base", edit),),
            save_logits=("base",),
        )
        report = preflight_plan(case.st, plan)
        assert report.status == "unsupported"
        assert report.stage == "scan-plan"
        result = run_plan(case.st, plan)
        assert result.logits["base"].shape[0] == 1
        # And the opt-in gate proceeds (warn, don't block) on unsupported.
        gated = run_plan(case.st, plan, preflight=True)
        torch.testing.assert_close(gated.logits["base"], result.logits["base"])

    def test_scan_leaves_real_traces_unaffected(self, case: PreflightCase) -> None:
        # A preflight scan must not perturb subsequent real runs (no leftover
        # hooks / fake-mode state) — the pyvene-cleanup lesson.
        plan = _interchange_plan(case)
        before = run_plan(case.st, plan)
        assert preflight_plan(case.st, plan).ok
        after = run_plan(case.st, plan)
        torch.testing.assert_close(after.logits["base"], before.logits["base"])

    def test_in_scan_causalab_refusal_is_failed_and_run_agrees(
        self, case: PreflightCase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An honest-boundary NotImplementedError raised *inside* the scan
        # (wrapped by nnsight) must classify as failed — the raising frame is
        # recovered from the wrapper's embedded traceback — because the real
        # run raises the very same refusal. Forced-error path: empty the
        # mlp_activation tap registry so the tiny Llama's MLP is "unmapped".
        monkeypatch.setattr("causalab.neural.site._MLP_ACTIVATION_TAPS", ())
        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        plan = Plan(
            inputs={"base": base},
            ops=(
                CollectOp(
                    "base",
                    FeaturizedSite(Site("mlp_activation", 1)),
                    key="h",
                    positions=[-1],
                ),
            ),
        )
        report = preflight_plan(case.st, plan)
        assert report.status == "failed"
        assert report.stage == "scan-plan"
        assert "mlp_activation" in report.error
        with pytest.raises(NotImplementedError, match="mlp_activation"):
            run_plan(case.st, plan)

    def test_unexpected_error_types_propagate_not_a_verdict(
        self, case: PreflightCase
    ) -> None:
        # A typo'd transform closure raises AttributeError — a bug, not a
        # plan property. The gate must NOT launder it into "your plan failed";
        # it propagates as-is (#469 review).
        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        edit = Edit(_resid(1), g=lambda f: f.sahpe, positions=[-1])  # typo
        plan = Plan(
            inputs={"base": base},
            ops=(EditOp("base", edit),),
            save_logits=("base",),
        )
        with pytest.raises(AttributeError, match="sahpe"):
            preflight_plan(case.st, plan)

    def test_edited_backbone_is_unsupported_and_never_scanned(
        self, case: PreflightCase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Persistent edits (#466) compose with real traces but NOT with an
        # aborted scan: an in-scan validation failure corrupts the installed
        # mediators' interleaver state and poisons every later real trace
        # (measured — "RuntimeError: release unlocked lock"). The gate must
        # therefore report unsupported BEFORE any scan opens on an edited
        # backbone; the patched probe fails the test if one is attempted.
        monkeypatch.setattr(
            "causalab.neural.preflight.check_scan_support",
            lambda m: pytest.fail("no scan may be attempted on an edited backbone"),
        )
        plan = _interchange_plan(case)
        steer = Edit(FeaturizedSite(Site("block_output", 0)), g=lambda f: f + 5.0)
        install_edits(case.st, steer)
        try:
            report = preflight_plan(case.st, plan)
            assert report.status == "unsupported"
            assert report.stage == "persistent-edits"
            assert "install_edits" in report.error
            report.raise_if_failed()  # no verdict, not a failure
        finally:
            uninstall_edits(case.st, force=True)
        # Refusing to scan left the model healthy: after uninstall the gate
        # is clean again and the real run works.
        monkeypatch.undo()
        assert preflight_plan(case.st, plan).ok
        run_plan(case.st, plan)

    def test_generation_plan_gated_run_warns_and_generates(
        self, case: PreflightCase, caplog: pytest.LogCaptureFixture
    ) -> None:
        # run_plan(preflight=True) on a generation plan must not crash and
        # must not block: the gate reports unsupported (no verdict), warns,
        # and hands the plan to the generation lowering.
        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        plan = Plan(
            inputs={"base": base},
            ops=(CollectOp("base", _resid(1), key="h", positions=[-1], step=1),),
            generate=GenerateSpec(max_new_tokens=2),
        )
        with caplog.at_level(logging.WARNING, logger="causalab.neural.plan"):
            result = run_plan(case.st, plan, preflight=True)
        assert result.sequences["base"].shape == (1, 2)
        assert "h" in result.collects
        assert any(
            "preflight unavailable" in record.getMessage() for record in caplog.records
        )
