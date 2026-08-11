"""Tests for :mod:`causalab.neural.staged` — the PL2 multi-trace / staged
compiler (#404).

Tiers (mirroring ``tests/neural/test_plan.py``; ``causalab/neural`` owes
``unit`` + ``property``, and a ``golden`` GPU pin is the established pattern
for the real coherent backbone):

* ``unit`` — the scheduling contract, no forward pass. Whole-component sites
  rank model-independently, so :func:`lower_staged` runs with ``model=None``:
  pass-minimality pins (the canonical interchange stays ONE fused trace; a
  chain fuses its first hop and stages the second; a rendezvous conflict
  dissolves its group), the clean-pass alias, cross-input backward ranks and
  mixed frames staging into two traces, cross-model bindings staging by
  model identity (``Plan.models``, PL4), cyclic flow refused with the
  break-the-cycle recipe, headroom refusals firing model-free, and the
  terminal generate stage (EU3, #484): ``generate_key`` never inside
  ``stages``, reads into the generate trace force-staged into collect
  stages (``"generate-with-variable-intervention"`` — before
  ``frames_align`` is consulted, so the canonical frame-aligned interchange
  shape still stages), the clean-pass reroute staging as a prefill-frame
  collect stage, and the ``lowering="single"`` strictness arm.
* ``property`` — on tiny Llama **and** GPT-2 (CPU): the staged lowering is
  *semantics-preserving* — forced-staged results match the single-trace
  lowering exactly on the canonical plan, and every shape the single trace
  refuses (self-graft, cross-input backward rank, chain, mixed frames, the
  dissolved rendezvous conflict) matches a hand-rolled sequential-trace
  reference; ``run_plan(lowering="auto")`` returns the staged result instead
  of raising. Cross-model plans (PL4) match a two-model capture-then-patch
  oracle against a perturbed-weights source.
* ``numerical_unit`` — the per-trace ``tracer.stop()`` early-exit (CAP6,
  #459) on a 4-layer tiny llama: every staged trace saving no logits stops
  after its own deepest tap (raw-hook fire-counters), cross-stage produce
  values land before the stop, results bitwise-match the full-forward
  control.
* ``golden`` — on the real Qwen3-4B backbone (GPU) against the raw-hook
  oracle: the self-graft (clean-pass read → earlier-layer write, two staged
  traces) and the cross-model patch (two model objects, bf16 values crossing
  traces on a real device).

The references here are sequential nnsight traces rather than raw hooks where
the point under test is *staging equivalence* (the per-site read/write
primitives underneath are already raw-hook-pinned by test_site /
test_featurized_site / test_edit / test_plan); the golden tier keeps one
raw-hook oracle on the coherent backbone.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from nnterp import StandardizedTransformer

from causalab.neural.edit import Edit, ReadSource
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.plan import (
    CollectOp,
    EditOp,
    GenerateSpec,
    GradientRequest,
    Plan,
    StagingRequired,
    run_plan,
)
from causalab.neural.site import Site
from causalab.neural.staged import lower_staged, run_plan_staged

from tests._helpers.tiny import fresh_tiny_random_gpt2, fresh_tiny_random_llama
from tests.neural.activations.hook_oracle import (
    capture_component,
    component_module,
    layer_fire_counts,
    next_token_logits,
)

_BASE_TEXT = "the quick brown fox jumps"
_SOURCE_TEXT = "a slow green turtle sleeps deeply"


def _resid(layer: int) -> FeaturizedSite:
    return FeaturizedSite(Site("block_output", layer))


def _interchange(
    dst_input: str,
    src_input: str | None,
    dst_layer: int,
    src_layer: int,
    positions: list[int] | None = None,
    src_positions: list[int] | None = None,
) -> EditOp:
    """``dst_input``'s site at ``dst_layer`` ← the read at ``src_layer``
    (under ``src_input``, or the same input when ``None``)."""
    return EditOp(
        dst_input,
        Edit(
            _resid(dst_layer),
            g=lambda f, f_src: f_src,
            read_sources=(
                ReadSource(
                    _resid(src_layer),
                    positions=src_positions or positions,
                    input=src_input,
                ),
            ),
            positions=positions,
        ),
    )


def _fake_batch(length: int, batch: int = 1) -> dict[str, torch.Tensor]:
    """A frame of the given padded length — scheduling only reads shapes."""
    return {
        "input_ids": torch.ones(batch, length, dtype=torch.long),
        "attention_mask": torch.ones(batch, length, dtype=torch.long),
    }


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class StagedCase:
    st: StandardizedTransformer
    oracle: Any
    tok: Any

    def pair(self, t1: str, t2: str) -> tuple[dict, dict]:
        enc = self.tok([t1, t2], padding=True, return_tensors="pt")

        def row(i: int) -> dict:
            return {
                "input_ids": enc["input_ids"][i : i + 1],
                "attention_mask": enc["attention_mask"][i : i + 1],
            }

        return row(0), row(1)

    def solo(self, text: str) -> dict:
        enc = self.tok(text, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def _case(raw: Any, tok: Any) -> StagedCase:
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    st = StandardizedTransformer(raw, tokenizer=tok, check_renaming=True)
    st.dispatch()
    return StagedCase(st=st, oracle=SimpleNamespace(hf_model=raw), tok=tok)


@pytest.fixture(scope="module")
def llama_case() -> StagedCase:
    return _case(*fresh_tiny_random_llama())


@pytest.fixture(scope="module")
def gpt2_case() -> StagedCase:
    return _case(*fresh_tiny_random_gpt2())


def _perturbed(raw: Any) -> Any:
    """Fixed-seed weight noise: a genuinely *different* model of the same
    architecture (and tokenizer), so a cross-model patch is provably the
    source's value — the ``test_cross_model_hook_oracle`` recipe."""
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in raw.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g))
    return raw


@pytest.fixture(scope="module")
def llama_source_case() -> StagedCase:
    raw, tok = fresh_tiny_random_llama()
    return _case(_perturbed(raw), tok)


@pytest.fixture(scope="module")
def gpt2_source_case() -> StagedCase:
    raw, tok = fresh_tiny_random_gpt2()
    return _case(_perturbed(raw), tok)


# --------------------------------------------------------------------------- #
#  unit — scheduling, no forward pass (model=None: Sites rank model-free)      #
# --------------------------------------------------------------------------- #
class TestScheduleUnit:
    pytestmark = pytest.mark.unit

    def test_canonical_interchange_stays_one_fused_trace(self) -> None:
        """Pass-minimality: what PL1 runs in one trace, staging must not
        split — auto never pays a second pass without a semantic reason."""
        a, b = _fake_batch(7), _fake_batch(7)
        plan = Plan(
            inputs={"source": a, "base": b},
            ops=(_interchange("base", "source", 0, 0, positions=[6]),),
            save_logits=("base",),
        )
        program = lower_staged(None, plan)  # type: ignore[arg-type]
        assert program.stages == ((("source", "base"),),)
        assert program.num_traces == 1

    def test_same_input_backward_dep_adds_a_clean_pass(self) -> None:
        plan = Plan(
            inputs={"base": _fake_batch(7)},
            ops=(_interchange("base", None, 0, 1, positions=[6]),),
        )
        program = lower_staged(None, plan)  # type: ignore[arg-type]
        assert program.stages == (((("base", "clean"),),), (("base",),))

    def test_cross_input_backward_rank_stages_two_traces(self) -> None:
        plan = Plan(
            inputs={"source": _fake_batch(7), "base": _fake_batch(7)},
            ops=(_interchange("base", "source", 0, 1, positions=[6]),),
        )
        program = lower_staged(None, plan)  # type: ignore[arg-type]
        assert program.stages == ((("source",),), (("base",),))

    def test_mixed_frames_stage_instead_of_fusing(self) -> None:
        plan = Plan(
            inputs={"source": _fake_batch(11), "base": _fake_batch(7)},
            ops=(
                _interchange("base", "source", 0, 0, positions=[6], src_positions=[10]),
            ),
        )
        program = lower_staged(None, plan)  # type: ignore[arg-type]
        assert program.num_traces == 2

    def test_chain_fuses_first_hop_and_stages_the_second(self) -> None:
        """A → B rides the barrier (B is a pure consumer there); B → C must
        cross a stage boundary (B cannot be consumer and producer in one
        trace — the measured one-phase-per-trace constraint)."""
        frames = {k: _fake_batch(7) for k in ("A", "B", "C")}
        plan = Plan(
            inputs=frames,
            ops=(
                _interchange("B", "A", 0, 0, positions=[6]),
                _interchange("C", "B", 1, 1, positions=[6]),
            ),
            save_logits=("C",),
        )
        program = lower_staged(None, plan)  # type: ignore[arg-type]
        assert program.stages == ((("A", "B"),), (("C",),))

    def test_rendezvous_conflict_dissolves_the_group(self) -> None:
        """Two per-edge-forward producers whose one barrier cannot serve both
        (latest signal after earliest wait) fall back to staged edges."""
        frames = {k: _fake_batch(7) for k in ("p1", "p2", "base")}
        plan = Plan(
            inputs=frames,
            ops=(
                _interchange("base", "p1", 1, 1, positions=[6]),
                _interchange("base", "p2", 0, 0, positions=[6]),
            ),
        )
        program = lower_staged(None, plan)  # type: ignore[arg-type]
        assert program.stages == ((("p1",), ("p2",)), (("base",),))
        assert not program.in_trace

    def test_cross_model_edge_stages_despite_aligned_frames(self) -> None:
        """The canonical interchange shape — frame-aligned, forward in rank —
        still stages when the source input is bound to another model
        (Plan.models, PL4): two models never share one fused forward."""
        plan = Plan(
            inputs={"source": _fake_batch(7), "base": _fake_batch(7)},
            ops=(_interchange("base", "source", 0, 0, positions=[6]),),
            save_logits=("base",),
            models={"source": object()},
        )
        program = lower_staged(None, plan)  # type: ignore[arg-type]
        assert program.stages == ((("source",),), (("base",),))
        assert not program.in_trace

    def test_same_bound_model_inputs_still_fuse(self) -> None:
        """Model sameness is object identity, not boundness: binding both
        endpoints to the SAME model keeps the canonical fused trace."""
        m = object()
        plan = Plan(
            inputs={"source": _fake_batch(7), "base": _fake_batch(7)},
            ops=(_interchange("base", "source", 0, 0, positions=[6]),),
            save_logits=("base",),
            models={"source": m, "base": m},
        )
        program = lower_staged(None, plan)  # type: ignore[arg-type]
        assert program.stages == ((("source", "base"),),)
        assert program.num_traces == 1

    def test_generation_plan_schedules_terminal_generate_key(self) -> None:
        """EU3 (#484): a generation plan's ops-addressed input is the ONE
        terminal ``model.generate`` invoke — ``generate_key``, never inside
        ``stages``. With no cross-input (or clean-pass) reads there are no
        collect stages: exactly one trace."""
        plan = Plan(
            inputs={"base": _fake_batch(7)},
            ops=(CollectOp("base", _resid(0), key="k", step=1),),
            generate=GenerateSpec(max_new_tokens=2),
        )
        program = lower_staged(None, plan)  # type: ignore[arg-type]
        assert program.generate_key == "base"
        assert program.stages == ()
        assert program.num_traces == 1

    def test_generation_cross_input_read_stages_despite_aligned_frames(self) -> None:
        """The canonical interchange shape — frame-aligned, forward in rank,
        same model: everything the fusability rules would fuse — still
        stages when the consumer is the generate invoke, because forcing is
        checked BEFORE ``frames_align`` (a generate trace accepts only
        constants). The source runs as an earlier collect stage; the edge
        records ``"generate-with-variable-intervention"``."""
        plan = Plan(
            inputs={"source": _fake_batch(7), "base": _fake_batch(7)},
            ops=(_interchange("base", "source", 0, 0, positions=[6]),),
            generate=GenerateSpec(max_new_tokens=2),
        )
        program = lower_staged(None, plan)  # type: ignore[arg-type]
        assert program.generate_key == "base"
        assert program.stages == ((("source",),),)
        assert not program.in_trace
        assert set(program.staged_why.values()) == {
            "generate-with-variable-intervention"
        }
        assert program.num_traces == 2

    def test_generation_same_input_backward_read_stages_clean_prefill_pass(
        self,
    ) -> None:
        """A same-input read after the written site inside a generation plan
        reroutes to the ``(input, "clean")`` alias — a plain prefill-frame
        pass staged as a collect stage before the generate trace (the
        per-step StagingRequired→ValueError wrapper died with EU3)."""
        plan = Plan(
            inputs={"base": _fake_batch(7)},
            ops=(_interchange("base", None, 0, 1, positions=[6]),),
            generate=GenerateSpec(max_new_tokens=2),
        )
        program = lower_staged(None, plan)  # type: ignore[arg-type]
        assert program.generate_key == "base"
        assert program.stages == (((("base", "clean"),),),)
        assert set(program.staged_why.values()) == {
            "generate-with-variable-intervention"
        }
        assert program.num_traces == 2

    def test_generation_forcing_strictness_arm(self) -> None:
        """``lowering="single"`` on a generation plan with a cross-input
        read: the schedule needs a collect stage plus the generate trace, so
        strictness raises from the ``"generate-with-variable-intervention"``
        schedule fact — model-free."""
        plan = Plan(
            inputs={"source": _fake_batch(7), "base": _fake_batch(7)},
            ops=(_interchange("base", "source", 0, 0, positions=[6]),),
            generate=GenerateSpec(max_new_tokens=2),
        )
        with pytest.raises(
            StagingRequired, match="generate trace accepts only constants"
        ):
            run_plan(None, plan, lowering="single")  # type: ignore[arg-type]

    def test_cyclic_cross_input_flow_refused_with_recipe(self) -> None:
        plan = Plan(
            inputs={"x": _fake_batch(7), "y": _fake_batch(7)},
            ops=(
                _interchange("y", "x", 0, 0, positions=[6]),
                _interchange("x", "y", 0, 0, positions=[6]),
            ),
        )
        with pytest.raises(ValueError, match="cyclic cross-input flow"):
            lower_staged(None, plan)  # type: ignore[arg-type]

    def test_headroom_refusals_fire_model_free(self) -> None:
        # The narrowed generation refusal (EU3, #484): *ops* must address
        # ONE input — the generated one; the scheduler raises before any
        # model access (reads of other inputs are legal via read_sources).
        multi_generate = Plan(
            inputs={"a": _fake_batch(7), "b": _fake_batch(7)},
            ops=(
                CollectOp("a", _resid(0), key="ka", step=1),
                CollectOp("b", _resid(0), key="kb", step=1),
            ),
            generate=GenerateSpec(max_new_tokens=2),
        )
        with pytest.raises(NotImplementedError, match="ONE input"):
            run_plan_staged(None, multi_generate)  # type: ignore[arg-type]
        grads = Plan(
            inputs={"base": _fake_batch(7)},
            ops=(CollectOp("base", _resid(0), key="h"),),
            gradients=GradientRequest(loss=lambda c: 0.0, wrt=("h",)),
        )
        with pytest.raises(NotImplementedError, match="gradients"):
            run_plan_staged(None, grads)  # type: ignore[arg-type]

    def test_unknown_lowering_rejected(self) -> None:
        plan = Plan(
            inputs={"base": _fake_batch(7)},
            ops=(CollectOp("base", _resid(0), key="h"),),
        )
        with pytest.raises(ValueError, match="unknown lowering"):
            run_plan(None, plan, lowering="eager")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
#  property — staging preserves semantics                                      #
# --------------------------------------------------------------------------- #
class TestStagedSemantics:
    pytestmark = pytest.mark.property

    @pytest.fixture(params=["llama", "gpt2"])
    def case(self, request, llama_case: StagedCase, gpt2_case: StagedCase):
        return {"llama": llama_case, "gpt2": gpt2_case}[request.param]

    def test_forced_staging_matches_single_trace_on_canonical_plan(
        self, case: StagedCase
    ) -> None:
        """The two lowerings are interchangeable where both apply — same
        collects, same logits, exactly."""
        base, source = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        last = base["input_ids"].shape[1] - 1
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                _interchange("base", "source", 0, 0, positions=[last]),
                CollectOp("base", _resid(1), key="mid"),
            ),
            save_logits=("base", "source"),
        )
        single = run_plan(case.st, plan, lowering="single")
        staged = run_plan(case.st, plan, lowering="staged")
        torch.testing.assert_close(staged.collects["mid"], single.collects["mid"])
        torch.testing.assert_close(staged.logits["base"], single.logits["base"])
        torch.testing.assert_close(staged.logits["source"], single.logits["source"])

    def test_self_graft_reads_the_clean_pass(self, case: StagedCase) -> None:
        """read layer 1 → write layer 0 on ONE input: the read must be the
        input's clean layer-1 activation (an extra pass), matching a
        hand-rolled capture-then-patch."""
        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        last = base["input_ids"].shape[1] - 1
        plan = Plan(
            inputs={"base": base},
            ops=(_interchange("base", None, 0, 1, positions=[last]),),
            save_logits=("base",),
        )
        result = run_plan(case.st, plan)  # auto → staged

        clean_l1 = case_capture(case, 1, base)[:, [last]]
        patched = next_token_logits(
            case.oracle, base, layer=0, positions=[last], patch_values=clean_l1
        )
        clean = next_token_logits(case.oracle, base)
        assert not torch.allclose(patched, clean, atol=1e-4), "inert graft"
        torch.testing.assert_close(
            result.logits["base"][:, -1, :], patched, atol=1e-5, rtol=1e-4
        )

    def test_cross_input_backward_rank_matches_oracle(self, case: StagedCase) -> None:
        base, source = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        last = base["input_ids"].shape[1] - 1
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(_interchange("base", "source", 0, 1, positions=[last]),),
            save_logits=("base",),
        )
        result = run_plan(case.st, plan)  # auto → staged

        src_l1 = case_capture(case, 1, source)[:, [last]]
        patched = next_token_logits(
            case.oracle, base, layer=0, positions=[last], patch_values=src_l1
        )
        torch.testing.assert_close(
            result.logits["base"][:, -1, :], patched, atol=1e-5, rtol=1e-4
        )

    def test_chain_matches_sequential_traces(self, case: StagedCase) -> None:
        """A → B → C: B runs under A's patch (fused first hop), C consumes
        B's *patched* layer-1 state from across the stage boundary."""
        enc = case.tok(
            [_BASE_TEXT, _SOURCE_TEXT, "seven red balloons drift far away"],
            padding=True,
            return_tensors="pt",
        )

        def row(i: int) -> dict:
            return {
                "input_ids": enc["input_ids"][i : i + 1],
                "attention_mask": enc["attention_mask"][i : i + 1],
            }

        a, b, c = row(0), row(1), row(2)
        last = a["input_ids"].shape[1] - 1
        plan = Plan(
            inputs={"A": a, "B": b, "C": c},
            ops=(
                _interchange("B", "A", 0, 0, positions=[last]),
                _interchange("C", "B", 1, 1, positions=[last]),
            ),
            save_logits=("C",),
        )
        result = run_plan(case.st, plan)  # auto → staged
        st = case.st
        with st.trace(a):
            va = st.layers_output[0][:, [last], :].cpu().save()
        with st.trace(b):
            st.layers_output[0][:, [last], :] = va.to(st.layers_output[0].dtype)
            vb = st.layers_output[1][:, [last], :].cpu().save()
        with st.trace(c):
            st.layers_output[1][:, [last], :] = vb.to(st.layers_output[1].dtype)
            ref = st.logits.cpu().save()
        torch.testing.assert_close(result.logits["C"], ref)

    def test_mixed_frames_stage_and_match_oracle(self, case: StagedCase) -> None:
        """Separately tokenized inputs (different padded lengths) run as
        separate traces — each input's positions stay in its own frame."""
        base = case.solo(_BASE_TEXT)
        source = case.solo(
            "a very much longer sentence that certainly pads differently "
            "than the base does"
        )
        last_b = base["input_ids"].shape[1] - 1
        last_s = source["input_ids"].shape[1] - 1
        assert last_b != last_s
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                _interchange(
                    "base",
                    "source",
                    0,
                    0,
                    positions=[last_b],
                    src_positions=[last_s],
                ),
            ),
            save_logits=("base",),
        )
        result = run_plan(case.st, plan)  # auto → staged

        src_vals = case_capture(case, 0, source)[:, [last_s]]
        patched = next_token_logits(
            case.oracle, base, layer=0, positions=[last_b], patch_values=src_vals
        )
        torch.testing.assert_close(
            result.logits["base"][:, -1, :], patched, atol=1e-5, rtol=1e-4
        )

    def test_dissolved_rendezvous_matches_two_patch_oracle(
        self, case: StagedCase
    ) -> None:
        enc = case.tok(
            [_BASE_TEXT, _SOURCE_TEXT, "seven red balloons drift far away"],
            padding=True,
            return_tensors="pt",
        )

        def row(i: int) -> dict:
            return {
                "input_ids": enc["input_ids"][i : i + 1],
                "attention_mask": enc["attention_mask"][i : i + 1],
            }

        base, p1, p2 = row(0), row(1), row(2)
        last = base["input_ids"].shape[1] - 1
        plan = Plan(
            inputs={"p1": p1, "p2": p2, "base": base},
            ops=(
                _interchange("base", "p1", 1, 1, positions=[last]),
                _interchange("base", "p2", 0, 0, positions=[last]),
            ),
            save_logits=("base",),
        )
        assert lower_staged(case.st, plan).num_traces == 3
        result = run_plan(case.st, plan)  # auto → staged
        st = case.st
        with st.trace(p1):
            v1 = st.layers_output[1][:, [last], :].cpu().save()
        with st.trace(p2):
            v2 = st.layers_output[0][:, [last], :].cpu().save()
        with st.trace(base):
            st.layers_output[0][:, [last], :] = v2.to(st.layers_output[0].dtype)
            st.layers_output[1][:, [last], :] = v1.to(st.layers_output[1].dtype)
            ref = st.logits.cpu().save()
        torch.testing.assert_close(result.logits["base"], ref)

    def test_raw_prompt_inputs_stage_per_trace(self, case: StagedCase) -> None:
        """Untokenized multi-input plans (no fusable frames) run one trace
        each and match the single-input collects."""
        plan = Plan(
            inputs={"a": _BASE_TEXT, "b": _SOURCE_TEXT},
            ops=(
                CollectOp("a", _resid(0), key="ha"),
                CollectOp("b", _resid(1), key="hb"),
            ),
        )
        result = run_plan(case.st, plan)  # auto → staged
        expected_a = _resid(0).collect(case.st, case.solo(_BASE_TEXT))
        expected_b = _resid(1).collect(case.st, case.solo(_SOURCE_TEXT))
        torch.testing.assert_close(result.collects["ha"], expected_a)
        torch.testing.assert_close(result.collects["hb"], expected_b)


def case_capture(case: StagedCase, layer: int, inputs: Any) -> torch.Tensor:
    module, kind = component_module(case.oracle, layer, "block_output")
    return capture_component(case.oracle, module, kind, inputs)


# --------------------------------------------------------------------------- #
#  numerical_unit — tracer.stop() early-exit per staged trace (CAP6, #459)     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def deep_llama_case() -> StagedCase:
    """A 4-layer fresh tiny llama — the default stub has only 2 layers, too
    few to observe layers past the deepest tap not running."""

    def deepen(cfg: Any) -> None:
        cfg.num_hidden_layers = 4

    return _case(*fresh_tiny_random_llama(mutate_config=deepen))


class TestEarlyStopNumerical:
    """The CAP6 (#459) contract on the staged lowering, pinned against raw
    ``register_forward_hook`` fire-counters on a 4-layer tiny llama (CPU):
    every staged trace that saves no logits stops its own forward after its
    deepest tap, and the results are bitwise-identical to the no-stop path
    (the same plan with ``save_logits``, whose full forwards are the
    control)."""

    pytestmark = pytest.mark.numerical_unit

    def test_mixed_frame_collect_only_stops_each_trace(
        self, deep_llama_case: StagedCase
    ) -> None:
        """Mixed padded frames stage into one trace per input; each stops
        after its OWN deepest tap: a (collect at layer 1) fires layers 0-1,
        b (collect at layer 0) fires layer 0 only."""
        case = deep_llama_case
        a, b = case.solo(_BASE_TEXT), case.solo(_SOURCE_TEXT)
        assert a["input_ids"].shape[1] != b["input_ids"].shape[1]
        ops = (
            CollectOp("a", _resid(1), key="a1"),
            CollectOp("b", _resid(0), key="b0"),
        )
        with layer_fire_counts(case.oracle) as counts:
            stopped = run_plan_staged(case.st, Plan(inputs={"a": a, "b": b}, ops=ops))
        assert counts == [2, 1, 0, 0], counts
        with layer_fire_counts(case.oracle) as counts:
            control = run_plan_staged(
                case.st,
                Plan(inputs={"a": a, "b": b}, ops=ops, save_logits=("a", "b")),
            )
        assert counts == [2, 2, 2, 2], counts
        for key in ("a1", "b0"):
            assert torch.equal(stopped.collects[key], control.collects[key])

    def test_cross_stage_produce_saved_before_stop(
        self, deep_llama_case: StagedCase
    ) -> None:
        """The intervene-backwards interchange (source read at layer 1 feeding a
        base write at layer 0) stages into two traces; the source trace's
        produce ``.save()`` lands before its stop, so the consuming trace
        still receives the value — pinned bitwise on a collect under the
        patch. Both traces stop after layer 1 (their deepest taps)."""
        case = deep_llama_case
        base, source = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        last = base["input_ids"].shape[1] - 1
        ops = (
            _interchange("base", "source", 0, 1, positions=[last]),
            CollectOp("base", _resid(1), key="mid"),
        )
        with layer_fire_counts(case.oracle) as counts:
            stopped = run_plan_staged(
                case.st, Plan(inputs={"source": source, "base": base}, ops=ops)
            )
        assert counts == [2, 2, 0, 0], counts
        control = run_plan_staged(
            case.st,
            Plan(
                inputs={"source": source, "base": base},
                ops=ops,
                save_logits=("base",),
            ),
        )
        assert torch.equal(stopped.collects["mid"], control.collects["mid"])
        clean_mid = case_capture(case, 1, base)
        assert not torch.allclose(stopped.collects["mid"], clean_mid, atol=1e-4), (
            "inert patch"
        )


# --------------------------------------------------------------------------- #
#  property — cross-model patching (PL4): capture-source → inject-target      #
# --------------------------------------------------------------------------- #
class TestCrossModelSemantics:
    """``Plan.models`` semantics on tiny models (CPU): the value written into
    the target's base run is the *source model's* activation — the contract
    ``test_cross_model_hook_oracle`` pins for the pyvene wrapper, here at the
    plan layer. The source is a perturbed-weights copy sharing architecture
    and tokenizer, so a wrong-model capture is numerically visible."""

    pytestmark = pytest.mark.property

    @pytest.fixture(params=["llama", "gpt2"])
    def cross(
        self,
        request,
        llama_case: StagedCase,
        gpt2_case: StagedCase,
        llama_source_case: StagedCase,
        gpt2_source_case: StagedCase,
    ) -> tuple[StagedCase, StagedCase]:
        return {
            "llama": (llama_case, llama_source_case),
            "gpt2": (gpt2_case, gpt2_source_case),
        }[request.param]

    def _cross_model_plan(
        self, target: StagedCase, source: StagedCase
    ) -> tuple[Plan, dict, dict, int]:
        # One tokenizer family, so tokenizing together keeps frames aligned —
        # proving the staging decision below is the model split, not frames.
        base, src_batch = target.pair(_BASE_TEXT, _SOURCE_TEXT)
        last = base["input_ids"].shape[1] - 1
        plan = Plan(
            inputs={"source": src_batch, "base": base},
            ops=(
                _interchange("base", "source", 0, 0, positions=[last]),
                CollectOp("source", _resid(1), key="src_mid"),
            ),
            save_logits=("base",),
            models={"source": source.st},
        )
        return plan, base, src_batch, last

    def test_patch_carries_the_source_models_activation(
        self, cross: tuple[StagedCase, StagedCase]
    ) -> None:
        """Cross-model interchange matches hand-rolled capture-on-source →
        patch-target raw-hook references, and collects on the bound input
        read the source model's forward."""
        target, source = cross
        plan, base, src_batch, last = self._cross_model_plan(target, source)
        assert lower_staged(target.st, plan).num_traces == 2
        result = run_plan(target.st, plan)  # auto → staged

        src_vals = case_capture(source, 0, src_batch)[:, [last]]
        patched = next_token_logits(
            target.oracle, base, layer=0, positions=[last], patch_values=src_vals
        )
        torch.testing.assert_close(
            result.logits["base"][:, -1, :], patched, atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            result.collects["src_mid"], case_capture(source, 1, src_batch)
        )

    def test_differs_from_same_model_patch(
        self, cross: tuple[StagedCase, StagedCase]
    ) -> None:
        """Patching the target's own activation at the same site must give a
        different answer — the injected value really is the source's."""
        target, source = cross
        plan, base, src_batch, last = self._cross_model_plan(target, source)
        result = run_plan(target.st, plan)

        tgt_vals = case_capture(target, 0, src_batch)[:, [last]]
        same_model = next_token_logits(
            target.oracle, base, layer=0, positions=[last], patch_values=tgt_vals
        )
        assert not torch.allclose(
            result.logits["base"][:, -1, :], same_model, atol=1e-4
        )

    def test_fully_bound_single_input_runs_on_its_model(
        self, cross: tuple[StagedCase, StagedCase]
    ) -> None:
        """A plan whose only input binds to a non-default model runs on that
        model — through the single-trace lowering (no staging needed)."""
        target, source = cross
        x = target.solo(_BASE_TEXT)
        plan = Plan(
            inputs={"x": x},
            ops=(CollectOp("x", _resid(0), key="h"),),
            save_logits=("x",),
            models={"x": source.st},
        )
        result = run_plan(target.st, plan, lowering="single")
        torch.testing.assert_close(
            result.collects["h"], _resid(0).collect(source.st, x)
        )
        with source.st.trace(x):
            expected = source.st.logits.cpu().save()
        torch.testing.assert_close(result.logits["x"], expected)


# --------------------------------------------------------------------------- #
#  golden — staged traces on the real Qwen3-4B backbone                        #
# --------------------------------------------------------------------------- #
class TestChatCoherentGolden:
    """The self-graft (clean later-layer read → earlier-layer write, two
    staged traces) on the coherent GPU backbone, matched against the
    capture-then-patch raw-hook oracle."""

    pytestmark = pytest.mark.golden

    def test_self_graft_matches_oracle_on_coherent_model(self) -> None:
        st = StandardizedTransformer(
            "Qwen/Qwen3-4B-Instruct-2507", dispatch=True, device_map="auto"
        )
        raw = st._model
        oracle = SimpleNamespace(hf_model=raw)
        device = next(raw.parameters()).device
        enc = st.tokenizer(_BASE_TEXT, return_tensors="pt")
        base = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
        last = int(base["input_ids"].shape[1] - 1)
        write_layer = int(st.num_layers) // 2
        read_layer = write_layer + 4
        plan = Plan(
            inputs={"base": base},
            ops=(
                _interchange("base", None, write_layer, read_layer, positions=[last]),
            ),
            save_logits=("base",),
        )
        assert lower_staged(st, plan).num_traces == 2
        result = run_plan(st, plan)  # auto → staged

        module, kind = component_module(oracle, read_layer, "block_output")
        clean_read = capture_component(oracle, module, kind, base)[:, [last]]
        patched = next_token_logits(
            oracle, base, layer=write_layer, positions=[last], patch_values=clean_read
        )
        clean = next_token_logits(oracle, base)
        assert not torch.allclose(patched, clean.float(), atol=1e-2), "inert graft"
        torch.testing.assert_close(
            result.logits["base"][:, -1, :].float(),
            patched.cpu().float(),
            atol=1e-3,
            rtol=1e-3,
        )


class TestChatCoherentCrossModelGolden:
    """Cross-model patching (PL4) on the coherent GPU backbone: two model
    objects — the target and a perturbed-weights second load — with the
    source's activation injected into the target's base run, matched against
    the two-model raw-hook oracle. Pins what the CPU property tier cannot:
    the saved produce value crossing traces on a real device in bfloat16."""

    pytestmark = pytest.mark.golden

    def test_cross_model_patch_matches_oracle_on_coherent_model(self) -> None:
        name = "Qwen/Qwen3-4B-Instruct-2507"
        tgt = StandardizedTransformer(name, dispatch=True, device_map="auto")
        src = StandardizedTransformer(name, dispatch=True, device_map="auto")
        # Perturb the second load in place: same architecture + tokenizer,
        # genuinely different weights (the cross-model oracle recipe). Small
        # scale — RMSNorm keeps activations bounded, bf16 stays finite.
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for p in src._model.parameters():
                noise = 0.005 * torch.randn(p.shape, generator=g)
                p.add_(noise.to(device=p.device, dtype=p.dtype))

        tgt_oracle = SimpleNamespace(hf_model=tgt._model)
        src_oracle = SimpleNamespace(hf_model=src._model)
        device = next(tgt._model.parameters()).device

        def load(text: str) -> dict:
            enc = tgt.tokenizer(text, return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].to(device),
                "attention_mask": enc["attention_mask"].to(device),
            }

        base, source = load(_BASE_TEXT), load(_SOURCE_TEXT)
        last_b = int(base["input_ids"].shape[1] - 1)
        last_s = int(source["input_ids"].shape[1] - 1)
        layer = int(tgt.num_layers) // 4
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                _interchange(
                    "base",
                    "source",
                    layer,
                    layer,
                    positions=[last_b],
                    src_positions=[last_s],
                ),
            ),
            save_logits=("base",),
            models={"source": src},
        )
        assert lower_staged(tgt, plan).num_traces == 2
        result = run_plan(tgt, plan)  # auto → staged

        module, kind = component_module(src_oracle, layer, "block_output")
        src_vals = capture_component(src_oracle, module, kind, source)[:, [last_s]]
        patched = next_token_logits(
            tgt_oracle, base, layer=layer, positions=[last_b], patch_values=src_vals
        )
        # Sanity: the perturbed source's activation differs from the target's
        # own at the same site — the patch below is provably cross-model.
        module_t, kind_t = component_module(tgt_oracle, layer, "block_output")
        tgt_vals = capture_component(tgt_oracle, module_t, kind_t, source)[:, [last_s]]
        assert not torch.allclose(src_vals.float(), tgt_vals.float(), atol=1e-3), (
            "inert perturbation"
        )
        torch.testing.assert_close(
            result.logits["base"][:, -1, :].float(),
            patched.cpu().float(),
            atol=1e-3,
            rtol=1e-3,
        )
