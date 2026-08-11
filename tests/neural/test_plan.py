"""Tests for :mod:`causalab.neural.plan` — the PL1 Plan IR + single-trace
compiler (#403).

Tiers (mirroring ``tests/neural/test_edit.py``; ``causalab/neural`` owes
``unit`` + ``property``, and a ``golden`` GPU pin is the established pattern
for the real coherent backbone):

* ``unit`` — the IR contract, no forward pass: Plan/op/GradientRequest
  validation, and the refusal-ordering contract (headroom refusals and the
  input-level staging check fire before the model is touched at all —
  asserted by passing ``model=None``).
* ``property`` — on tiny Llama **and** GPT-2 (CPU), against the same
  raw-``register_forward_hook`` oracle ST1/ST3/ED1 are pinned to: collect-only
  plans match ``FeaturizedSite.collect``; single-input logits match the
  oracle; the canonical cross-invoke interchange (identity and rotated
  feature space) matches capture-then-patch; a collect *under* an
  intervention (the collect∘intervene fusion) matches capture-under-patch;
  a two-source blend crosses one barrier; same-site collect-after-edit reads
  the written value; the model-aware multi-trace schedules raise
  :class:`StagingRequired` under ``lowering="single"`` strictness (EU2,
  #483 — schedule facts, each retired refusal's key phrase preserved).
* ``numerical_unit`` — the ``tracer.stop()`` early-exit (CAP6, #459) on a
  4-layer tiny llama: a plan that saves no logits stops after its deepest
  tap (raw-hook fire-counters prove later layers never ran) and collects
  bitwise-match the full-forward control.
* ``golden`` — the canonical interchange AND a decode-step steered-generation
  chain (CAP2, #455) on the real Qwen3-4B backbone (GPU), against the same
  oracle style, one model load each.

The generation-plan tests (CAP2) pin the ``tracer.iter`` lowering to a
*forward-pass-counting* raw-hook oracle: pass 0 of a module is the prefill,
pass k the k-th KV-cached decode step, so an oracle hook that fires on its
k-th call is the ground truth for ``step=k``.

Models come from the fresh (uncached) factories in ``tests/_helpers/tiny.py``
— never the session-cached ``tiny_random_model`` singleton, whose leftover
pyvene forward hooks break a later nnsight trace (see the factory docstrings).
Multi-input plans need frame-aligned inputs, so paired prompts are tokenized
together (left-padded) — the same alignment the compiler's mixed-frame error
message prescribes.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from nnterp import StandardizedTransformer

from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.edit import Edit, ReadSource
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.plan import (
    CollectOp,
    EditOp,
    GenerateSpec,
    GradientRequest,
    Plan,
    PlanResult,
    StagingRequired,
    run_plan,
)
from causalab.neural.pipeline import ensure_position_ids
from causalab.neural.site import Site

from tests._helpers.tiny import fresh_tiny_random_gpt2, fresh_tiny_random_llama
from tests.neural.activations.hook_oracle import (
    capture_component,
    component_edited_logits,
    component_module,
    hidden_of,
    layer_fire_counts,
    next_token_logits,
)

_BASE_TEXT = "the quick brown fox jumps"
_SOURCE_TEXT = "a slow green turtle sleeps deeply"


def _subspace(width: int, k: int, seed: int = 0) -> SubspaceFeaturizer:
    """A deterministic frozen rotation ``width → k`` (seeded orthogonal init)."""
    torch.manual_seed(seed)
    return SubspaceFeaturizer(shape=(width, k), trainable=False)


def _resid(layer: int) -> FeaturizedSite:
    return FeaturizedSite(Site("block_output", layer))


# --------------------------------------------------------------------------- #
#  Fixtures — fresh (uncached) StandardizedTransformers + a raw-model shim     #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PlanCase:
    st: StandardizedTransformer  # run_plan taps this
    oracle: Any  # SimpleNamespace(hf_model=raw) — for the hook_oracle helpers
    tok: Any

    def pair(self, t1: str, t2: str) -> tuple[dict, dict]:
        """Two single-row batches in ONE padded frame (tokenized together,
        left-padded) — what a multi-input plan requires."""
        enc = self.tok([t1, t2], padding=True, return_tensors="pt")

        def row(i: int) -> dict:
            return {
                "input_ids": enc["input_ids"][i : i + 1],
                "attention_mask": enc["attention_mask"][i : i + 1],
            }

        return row(0), row(1)

    def batch(self, *texts: str) -> dict:
        """One padded multi-row batch — what a generation plan runs."""
        enc = self.tok(list(texts), padding=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    def capture(self, component: str, layer: int, inputs: Any) -> torch.Tensor:
        """The raw-hook ground truth for ``(component, layer)`` — full
        ``(b, seq, d)``."""
        module, kind = component_module(self.oracle, layer, component)
        return capture_component(self.oracle, module, kind, inputs)

    def hidden(self) -> int:
        return int(self.oracle.hf_model.config.hidden_size)


def _case(raw: Any, tok: Any) -> PlanCase:
    tok.padding_side = "left"  # the pipeline convention; makes [-1] the last token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    st = StandardizedTransformer(raw, tokenizer=tok, check_renaming=True)
    st.dispatch()
    return PlanCase(st=st, oracle=SimpleNamespace(hf_model=raw), tok=tok)


@pytest.fixture(scope="module")
def llama_case() -> PlanCase:
    return _case(*fresh_tiny_random_llama())


@pytest.fixture(scope="module")
def gpt2_case() -> PlanCase:
    return _case(*fresh_tiny_random_gpt2())


# --------------------------------------------------------------------------- #
#  unit — the IR contract, no forward pass                                     #
# --------------------------------------------------------------------------- #
class TestIRContractUnit:
    pytestmark = pytest.mark.unit

    def test_op_with_unknown_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown input 'nope'"):
            Plan(inputs={"base": {}}, ops=(CollectOp("nope", _resid(0), key="k"),))

    def test_cross_ref_to_unknown_input_rejected(self) -> None:
        edit = Edit(
            _resid(0),
            g=lambda f, x: x,
            read_sources=(ReadSource(_resid(0), input="ghost"),),
        )
        with pytest.raises(ValueError, match="unknown input 'ghost'"):
            Plan(inputs={"base": {}}, ops=(EditOp("base", edit),))

    def test_duplicate_collect_keys_rejected(self) -> None:
        ops = (
            CollectOp("base", _resid(0), key="h"),
            CollectOp("base", _resid(1), key="h"),
        )
        with pytest.raises(ValueError, match="duplicate collect keys"):
            Plan(inputs={"base": {}}, ops=ops)

    def test_empty_plan_rejected(self) -> None:
        with pytest.raises(ValueError, match="does nothing"):
            Plan(inputs={"base": {}})
        with pytest.raises(ValueError, match="at least one input"):
            Plan(inputs={}, save_logits=())

    def test_bare_string_save_logits_rejected(self) -> None:
        with pytest.raises(ValueError, match="not the bare string"):
            Plan(inputs={"base": {}}, save_logits="base")

    def test_save_logits_unknown_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown input 'nope'"):
            Plan(inputs={"base": {}}, save_logits=("nope",))

    def test_collect_key_must_be_nonempty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CollectOp("base", _resid(0), key="")

    def test_editop_requires_writing_edit(self) -> None:
        with pytest.raises(ValueError, match="writing Edit"):
            EditOp("base", Edit(_resid(0)))  # g=None — a collect-shaped Edit

    def test_negative_step_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CollectOp("base", _resid(0), key="k", step=-1)

    def test_gradient_request_needs_wrt(self) -> None:
        with pytest.raises(ValueError, match="at least one collect key"):
            GradientRequest(loss=lambda c: 0.0, wrt=())

    def test_gradients_wrt_unknown_collect_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown collect keys"):
            Plan(
                inputs={"base": {}},
                ops=(CollectOp("base", _resid(0), key="h"),),
                gradients=GradientRequest(loss=lambda c: 0.0, wrt=("ghost",)),
            )

    def test_models_binding_unknown_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="models binds unknown inputs"):
            Plan(
                inputs={"base": {}},
                ops=(CollectOp("base", _resid(0), key="h"),),
                models={"ghost": object()},
            )

    def test_models_binding_none_rejected(self) -> None:
        with pytest.raises(ValueError, match="to None"):
            Plan(
                inputs={"base": {}},
                ops=(CollectOp("base", _resid(0), key="h"),),
                models={"base": None},
            )

    # --------------------- generation plans (CAP2, #455) ---------------------- #
    def test_step_without_generate_spec_rejected(self) -> None:
        with pytest.raises(ValueError, match="generation step"):
            Plan(
                inputs={"base": {}},
                ops=(CollectOp("base", _resid(0), key="k", step=2),),
            )

    def test_step_beyond_max_new_tokens_rejected(self) -> None:
        """The bounded-iterator guarantee: an out-of-range step is silently
        skipped by nnsight and abandons the rest of the trace — refused at
        construction instead."""
        with pytest.raises(ValueError, match="out of range"):
            Plan(
                inputs={"base": {}},
                ops=(CollectOp("base", _resid(0), key="k", step=2),),
                generate=GenerateSpec(max_new_tokens=2),
            )

    def test_generate_save_logits_rejected(self) -> None:
        with pytest.raises(ValueError, match="plain-forward contract"):
            Plan(
                inputs={"base": {}},
                ops=(CollectOp("base", _resid(0), key="k"),),
                save_logits=("base",),
                generate=GenerateSpec(max_new_tokens=2),
            )

    def test_generate_spec_max_new_tokens_validated(self) -> None:
        with pytest.raises(ValueError, match="max_new_tokens"):
            GenerateSpec(max_new_tokens=0)

    def test_generate_spec_reserved_kwargs_rejected(self) -> None:
        with pytest.raises(ValueError, match="owned by the generation lowering"):
            GenerateSpec(max_new_tokens=2, kwargs={"use_cache": False})

    # ------------------------- refusal ordering ------------------------------ #
    # Headroom refusals, the generation-plan checks, and the input-level
    # staging check are model-free — model=None proves they fire before any
    # model access.
    def test_generation_cross_input_read_schedules_instead_of_refusing(self) -> None:
        """EU3 (#484): the cross-input-generation refusal is GONE — the
        scheduler force-stages the source read into an earlier collect stage
        instead. Validation therefore proceeds past the retired refusal to
        the surviving generate input-shape check (the ``{}`` sentinel inputs
        are not pre-tokenized), still model-free and before any forward."""
        edit = Edit(
            _resid(0),
            g=lambda f, x: x,
            read_sources=(ReadSource(_resid(0), input="source"),),
        )
        plan = Plan(
            inputs={"source": {}, "base": {}},
            ops=(EditOp("base", edit, step=1),),
            generate=GenerateSpec(max_new_tokens=3),
        )
        with pytest.raises(ValueError, match="pre-tokenized"):
            run_plan(None, plan)  # type: ignore[arg-type]

    def test_generation_multi_input_refused_before_model(self) -> None:
        plan = Plan(
            inputs={"a": {}, "b": {}},
            ops=(
                CollectOp("a", _resid(0), key="ha", step=1),
                CollectOp("b", _resid(0), key="hb", step=1),
            ),
            generate=GenerateSpec(max_new_tokens=3),
        )
        with pytest.raises(NotImplementedError, match="ONE input"):
            run_plan(None, plan)  # type: ignore[arg-type]

    def test_generation_raw_prompt_input_refused_before_model(self) -> None:
        plan = Plan(
            inputs={"base": "a raw prompt"},
            ops=(CollectOp("base", _resid(0), key="k", step=1),),
            generate=GenerateSpec(max_new_tokens=3),
        )
        with pytest.raises(ValueError, match="pre-tokenized"):
            run_plan(None, plan)  # type: ignore[arg-type]

    def test_generation_multistep_position_ids_refused_before_model(self) -> None:
        """The plain-forward position_ids contract is prefill-only: multi-step
        generate numbers its own steps (measured drift on GPT-2 otherwise)."""
        inputs = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "position_ids": torch.arange(4).unsqueeze(0),
        }
        plan = Plan(
            inputs={"base": inputs},
            ops=(CollectOp("base", _resid(0), key="k", step=1),),
            generate=GenerateSpec(max_new_tokens=3),
        )
        with pytest.raises(ValueError, match="position_ids"):
            run_plan(None, plan)  # type: ignore[arg-type]

    def test_generation_min_new_tokens_override_below_last_step_refused(self) -> None:
        """A kwargs override below the last addressed step + 1 would let
        early EOS starve an addressed iteration — the exact silent-skip
        failure the construction-time step bound refuses — so the effective
        min_new_tokens is validated model-free, before any forward pass."""
        inputs = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
        plan = Plan(
            inputs={"base": inputs},
            ops=(CollectOp("base", _resid(0), key="k", step=4),),
            generate=GenerateSpec(max_new_tokens=5, kwargs={"min_new_tokens": 1}),
        )
        with pytest.raises(ValueError, match="min_new_tokens"):
            run_plan(None, plan)  # type: ignore[arg-type]
        # at or above the floor the same plan passes validation (it then
        # fails later on the sentinel model, proving the gate is the check)
        ok = Plan(
            inputs={"base": inputs},
            ops=(CollectOp("base", _resid(0), key="k", step=4),),
            generate=GenerateSpec(max_new_tokens=5, kwargs={"min_new_tokens": 5}),
        )
        with pytest.raises(AttributeError):
            run_plan(None, ok)  # type: ignore[arg-type]

    def test_generation_stepped_backward_read_refused_before_model(self) -> None:
        """A same-input read AFTER the written site on a ``step>0`` op is
        refused (the retired generation lowering's refusal, restored —
        review #492 F2): the op's positions resolve in that step's one-token
        decode frame, so the clean-prefill reroute would silently
        reinterpret them in the full prompt frame (``positions=[-1]`` would
        read the last PROMPT token). Model-free — whole-component sites
        rank without a model. The step-less/step=0 reroute stays and is
        oracle-pinned (``test_same_input_backward_read_reads_clean_prefill_
        pass``)."""
        inputs = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
        plan = Plan(
            inputs={"base": inputs},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        _resid(0),
                        g=lambda f, f_src: f_src,
                        read_sources=(ReadSource(_resid(1), positions=[-1]),),
                        positions=[-1],
                    ),
                    step=1,
                ),
            ),
            generate=GenerateSpec(max_new_tokens=3),
        )
        with pytest.raises(ValueError, match="at or before the written site"):
            run_plan(None, plan)  # type: ignore[arg-type]

    def test_generation_staged_alias_routes_to_generation_lowering(self) -> None:
        """``lowering="staged"`` is a deprecated alias of auto (EU2, #483):
        a generation plan under it routes to the generation lowering like
        any other — the old refusal died with the separate staged executor.
        The sentinel model proves the routing (validation passes, then the
        generation lowering touches the model)."""
        inputs = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
        plan = Plan(
            inputs={"base": inputs},
            ops=(CollectOp("base", _resid(0), key="k", step=1),),
            generate=GenerateSpec(max_new_tokens=3),
        )
        with pytest.raises(AttributeError):
            run_plan(None, plan, lowering="staged")  # type: ignore[arg-type]

    def test_gradient_plan_under_no_grad_refused_before_model(self) -> None:
        """Gradients execute now (CAP3, #456) — but never silently under
        ``torch.no_grad()``, and the guard is model-free."""
        plan = Plan(
            inputs={"base": {}},
            ops=(CollectOp("base", _resid(0), key="h"),),
            gradients=GradientRequest(loss=lambda c: 0.0, wrt=("h",)),
        )
        with torch.no_grad():
            with pytest.raises(RuntimeError, match="grad mode is off"):
                run_plan(None, plan)  # type: ignore[arg-type]

    def test_gradient_featurized_wrt_refused_before_model(self) -> None:
        """A ``wrt`` collect through a non-trivial featurizer is refused
        model-free — the compiler only delivers raw-activation gradients."""
        rotated = FeaturizedSite(Site("block_output", 0), _subspace(16, 4))
        plan = Plan(
            inputs={"base": {}},
            ops=(CollectOp("base", rotated, key="h"),),
            gradients=GradientRequest(loss=lambda c: 0.0, wrt=("h",)),
        )
        with pytest.raises(NotImplementedError, match="feature-space gradients"):
            run_plan(None, plan)  # type: ignore[arg-type]

    def test_gradient_collect_key_clashing_with_save_logits_rejected(self) -> None:
        """The loss receives collects and logits in one mapping, so their
        names must not collide — validated at construction."""
        with pytest.raises(ValueError, match="collide with save_logits"):
            Plan(
                inputs={"base": {}},
                ops=(CollectOp("base", _resid(0), key="base"),),
                save_logits=("base",),
                gradients=GradientRequest(loss=lambda c: 0.0, wrt=("base",)),
            )

    def test_gradient_generation_plan_rejected_at_construction(self) -> None:
        """CAP3 × CAP2: a Plan cannot carry both a GradientRequest and a
        GenerateSpec — backward through ``tracer.iter`` decode steps is
        unmeasured territory, refused legibly before any lowering runs."""
        with pytest.raises(ValueError, match="GradientRequest and a"):
            Plan(
                inputs={"base": {}},
                ops=(CollectOp("base", _resid(0), key="h"),),
                gradients=GradientRequest(loss=lambda c: 0.0, wrt=("h",)),
                generate=GenerateSpec(max_new_tokens=2),
            )

    def test_chained_cross_input_flow_staged_before_model(self) -> None:
        """An input that both consumes and produces cross-input reads never
        fits one producers → consumers phase — strict mode reports the
        chained flow from the schedule facts, model-free (whole-component
        sites rank without a model, so model=None proves no model access)."""
        plan = Plan(
            inputs={"a": {}, "b": {}, "c": {}},
            ops=(
                EditOp(
                    "b",
                    Edit(
                        _resid(0),
                        g=lambda f, x: x,
                        read_sources=(ReadSource(_resid(0), input="a"),),
                    ),
                ),
                EditOp(
                    "c",
                    Edit(
                        _resid(1),
                        g=lambda f, x: x,
                        read_sources=(ReadSource(_resid(1), input="b"),),
                    ),
                ),
            ),
        )
        with pytest.raises(StagingRequired, match="chained cross-input flow"):
            run_plan(None, plan, lowering="single")  # type: ignore[arg-type]

    def test_cross_model_inputs_staged_before_model(self) -> None:
        """Inputs bound to different models (Plan.models, PL4) cannot share
        one fused forward — they schedule as one trace group per model, and
        strict mode refuses without ever tracing (sentinel models prove no
        model access)."""
        plan = Plan(
            inputs={"source": {}, "base": {}},
            ops=(
                CollectOp("source", _resid(0), key="src"),
                CollectOp("base", _resid(0), key="dst"),
            ),
            models={"source": object()},
        )
        with pytest.raises(StagingRequired, match="bound to a different model"):
            run_plan(None, plan, lowering="single")  # type: ignore[arg-type]

    def test_staging_required_is_a_value_error(self) -> None:
        """PL2 dispatches on StagingRequired; callers that predate it catch
        ValueError — keep both working."""
        assert issubclass(StagingRequired, ValueError)


# --------------------------------------------------------------------------- #
#  property — the compiler matches the raw-hook oracle                         #
# --------------------------------------------------------------------------- #
class TestPlanMatchesOracle:
    pytestmark = pytest.mark.property

    @pytest.fixture(params=["llama", "gpt2"])
    def case(self, request, llama_case: PlanCase, gpt2_case: PlanCase) -> PlanCase:
        return {"llama": llama_case, "gpt2": gpt2_case}[request.param]

    # ------------------------------ collect-only ------------------------------ #
    def test_collect_only_plan_matches_featurized_site_collect(
        self, case: PlanCase
    ) -> None:
        """A multi-site collect plan returns exactly what per-site one-shot
        reads return (which are themselves oracle-pinned in test_site /
        test_featurized_site) — declaration order, not forward order."""
        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        sites = [_resid(1), _resid(0), FeaturizedSite(Site("mlp_output", 0))]
        plan = Plan(
            inputs={"base": base},
            ops=tuple(
                CollectOp("base", fsite, key=f"s{i}") for i, fsite in enumerate(sites)
            ),
        )
        result = run_plan(case.st, plan)
        for i, fsite in enumerate(sites):
            expected = fsite.collect(case.st, base)
            torch.testing.assert_close(result.collects[f"s{i}"], expected)

    def test_single_input_logits_match_oracle(self, case: PlanCase) -> None:
        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        plan = Plan(inputs={"base": base}, save_logits=("base",))
        result = run_plan(case.st, plan)
        clean = next_token_logits(case.oracle, base)
        torch.testing.assert_close(
            result.logits["base"][:, -1, :], clean, atol=1e-5, rtol=1e-4
        )

    # -------------------- the canonical cross-invoke interchange -------------- #
    def test_canonical_interchange_matches_oracle(self, case: PlanCase) -> None:
        """Source + base invokes + one barrier in ONE trace — the recipe every
        interchange analysis lowers to — against hand-rolled capture-then-patch
        on raw hooks."""
        base, source = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        last = base["input_ids"].shape[1] - 1
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        _resid(0),
                        g=lambda f, f_src: f_src,
                        read_sources=(
                            ReadSource(_resid(0), positions=[last], input="source"),
                        ),
                        positions=[last],
                    ),
                ),
            ),
            save_logits=("base", "source"),
        )
        result = run_plan(case.st, plan)

        src_vals = case.capture("block_output", 0, source)[:, [last]]
        patched = next_token_logits(
            case.oracle, base, layer=0, positions=[last], patch_values=src_vals
        )
        clean = next_token_logits(case.oracle, base)
        assert not torch.allclose(patched, clean, atol=1e-4), "inert patch"
        torch.testing.assert_close(
            result.logits["base"][:, -1, :], patched, atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            result.logits["source"][:, -1, :],
            next_token_logits(case.oracle, source),
            atol=1e-5,
            rtol=1e-4,
        )

    def test_featurized_interchange_matches_oracle(self, case: PlanCase) -> None:
        """The same recipe through a k-dim rotated subspace: only the selected
        feature directions move; base contributes the orthogonal complement."""
        base, source = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        last = base["input_ids"].shape[1] - 1
        feat = _subspace(case.hidden(), 4)
        src_fsite = FeaturizedSite(Site("block_output", 0), feat)
        dst_fsite = FeaturizedSite(Site("block_output", 0), feat)
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        dst_fsite,
                        g=lambda f, f_src: f_src,
                        read_sources=(
                            ReadSource(src_fsite, positions=[last], input="source"),
                        ),
                        positions=[last],
                    ),
                ),
            ),
            save_logits=("base",),
        )
        result = run_plan(case.st, plan)

        f_src, _ = feat.featurize(case.capture("block_output", 0, source)[:, [last]])
        module, kind = component_module(case.oracle, 0, "block_output")

        def edit(h: torch.Tensor) -> None:
            f, err = feat.featurize(h[:, [last]])
            h[:, [last]] = feat.inverse_featurize(f_src, err).to(h.dtype)

        manual = component_edited_logits(case.oracle, base, module, kind, edit)
        torch.testing.assert_close(
            result.logits["base"][:, -1, :], manual, atol=1e-5, rtol=1e-4
        )

    # ------------------------ collect ∘ intervene fusion ---------------------- #
    def test_collect_under_intervention_matches_oracle(self, case: PlanCase) -> None:
        """Patch at layer 0, read layer 1 in the SAME pass — the fused
        collect∘intervene pyvene needed the mixed model for. Ground truth:
        raw patch hook + raw capture hook in one forward."""
        base, source = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        last = base["input_ids"].shape[1] - 1
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        _resid(0),
                        g=lambda f, f_src: f_src,
                        read_sources=(
                            ReadSource(_resid(0), positions=[last], input="source"),
                        ),
                        positions=[last],
                    ),
                ),
                CollectOp("base", _resid(1), key="mid"),
            ),
        )
        result = run_plan(case.st, plan)

        src_vals = case.capture("block_output", 0, source)[:, [last]]
        patch_module, _ = component_module(case.oracle, 0, "block_output")
        read_module, _ = component_module(case.oracle, 1, "block_output")
        grabbed: dict[str, torch.Tensor] = {}

        def patch_hook(_m, _i, out):
            hidden = hidden_of(out).clone()
            hidden[:, [last]] = src_vals.to(hidden.dtype)
            return (hidden, *out[1:]) if isinstance(out, tuple) else hidden

        def read_hook(_m, _i, out):
            grabbed["mid"] = hidden_of(out).detach().clone()

        handles = [
            patch_module.register_forward_hook(patch_hook),
            read_module.register_forward_hook(read_hook),
        ]
        try:
            with torch.no_grad():
                case.oracle.hf_model(
                    input_ids=base["input_ids"],
                    attention_mask=base["attention_mask"],
                )
        finally:
            for h in handles:
                h.remove()

        clean_mid = case.capture("block_output", 1, base)
        assert not torch.allclose(grabbed["mid"], clean_mid, atol=1e-4), "inert patch"
        torch.testing.assert_close(
            result.collects["mid"], grabbed["mid"], atol=1e-5, rtol=1e-4
        )

    # ------------------------------ multi-source ------------------------------ #
    def test_two_source_blend_matches_oracle(self, case: PlanCase) -> None:
        """Two producing invokes feed one consuming edit across a single
        3-party barrier."""
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

        base, s1, s2 = row(0), row(1), row(2)
        last = base["input_ids"].shape[1] - 1
        plan = Plan(
            inputs={"s1": s1, "s2": s2, "base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        _resid(0),
                        g=lambda f, a, b: 0.5 * a + 0.5 * b,
                        read_sources=(
                            ReadSource(_resid(0), positions=[last], input="s1"),
                            ReadSource(_resid(0), positions=[last], input="s2"),
                        ),
                        positions=[last],
                    ),
                ),
            ),
            save_logits=("base",),
        )
        result = run_plan(case.st, plan)

        blend = (
            0.5 * case.capture("block_output", 0, s1)[:, [last]]
            + 0.5 * case.capture("block_output", 0, s2)[:, [last]]
        )
        patched = next_token_logits(
            case.oracle, base, layer=0, positions=[last], patch_values=blend
        )
        torch.testing.assert_close(
            result.logits["base"][:, -1, :], patched, atol=1e-5, rtol=1e-4
        )

    # ------------------------- same-site declaration order -------------------- #
    def test_collect_after_edit_at_same_site_sees_the_write(
        self, case: PlanCase
    ) -> None:
        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        last = base["input_ids"].shape[1] - 1
        plan = Plan(
            inputs={"base": base},
            ops=(
                CollectOp("base", _resid(0), key="before", positions=[last]),
                EditOp(
                    "base",
                    Edit(_resid(0), g=lambda f: torch.zeros_like(f), positions=[last]),
                ),
                CollectOp("base", _resid(0), key="after", positions=[last]),
            ),
        )
        result = run_plan(case.st, plan)
        assert not torch.allclose(
            result.collects["before"], torch.zeros_like(result.collects["before"])
        )
        torch.testing.assert_close(
            result.collects["after"], torch.zeros_like(result.collects["after"])
        )

    # -------------------- strictness (lowering="single") ---------------------- #
    # StagingRequired is a schedule FACT under the one scheduler (EU2, #483):
    # lowering="single" asserts the degenerate one-trace schedule and raises
    # iff num_traces > 1, its message assembled from the per-edge staged_why
    # reasons — each retired single-trace refusal's key phrase preserved.
    def test_same_input_backward_dep_strictness(self, case: PlanCase) -> None:
        """A same-input read after the written site schedules a clean-pass
        alias (two traces) — strict mode refuses with the "two passes"
        phrase; auto executes the same schedule (TestStagedSemantics)."""
        from causalab.neural.staged import lower_staged

        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        plan = Plan(
            inputs={"base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        _resid(0),
                        g=lambda f, x: x,
                        read_sources=(ReadSource(_resid(1)),),
                    ),
                ),
            ),
        )
        assert lower_staged(case.st, plan).num_traces == 2
        with pytest.raises(StagingRequired, match="two passes"):
            run_plan(case.st, plan, lowering="single")

    def test_cross_input_backward_rank_strictness(self, case: PlanCase) -> None:
        """A later-layer read on the source feeding an earlier-layer write on
        the base would surface as nnsight's after-the-fact
        MissedProviderError in one fused trace — it schedules as two traces,
        which strict mode reports up front."""
        from causalab.neural.staged import lower_staged

        base, source = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        _resid(0),
                        g=lambda f, x: x,
                        read_sources=(ReadSource(_resid(1), input="source"),),
                    ),
                ),
            ),
        )
        assert lower_staged(case.st, plan).num_traces == 2
        with pytest.raises(StagingRequired, match="backward in time"):
            run_plan(case.st, plan, lowering="single")

    def test_mixed_frames_strictness(self, case: PlanCase) -> None:
        """Inputs tokenized separately land in different padded frames —
        nnsight would left-pad the fused batch and shift the shorter input's
        positions, so the edge stages (one trace per frame) and strict mode
        refuses."""
        base = case.tok(_BASE_TEXT, return_tensors="pt")
        source = case.tok(
            "a very much longer sentence that certainly pads differently "
            "than the base does",
            return_tensors="pt",
        )
        base = {k: base[k] for k in ("input_ids", "attention_mask")}
        source = {k: source[k] for k in ("input_ids", "attention_mask")}
        assert base["input_ids"].shape[1] != source["input_ids"].shape[1]
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        _resid(0),
                        g=lambda f, x: x,
                        read_sources=(
                            ReadSource(_resid(0), positions=[0], input="source"),
                        ),
                    ),
                ),
            ),
        )
        with pytest.raises(StagingRequired, match="padded lengths"):
            run_plan(case.st, plan, lowering="single")

    def test_multi_input_raw_prompt_strictness(self, case: PlanCase) -> None:
        """Raw-prompt inputs have no static frame to fuse on — they schedule
        one trace each (connected-components grouping), and strict mode's
        message names the fix (pre-tokenized inputs)."""
        plan = Plan(
            inputs={"a": _BASE_TEXT, "b": _SOURCE_TEXT},
            ops=(
                CollectOp("a", _resid(0), key="ha"),
                CollectOp("b", _resid(0), key="hb"),
            ),
        )
        with pytest.raises(StagingRequired, match="pre-tokenized"):
            run_plan(case.st, plan, lowering="single")


# --------------------------------------------------------------------------- #
#  property — gradient plans match a grad-enabled raw-hook oracle (CAP3 #456)  #
# --------------------------------------------------------------------------- #
class TestGradientPlansMatchOracle:
    """GradientRequest execution against a grad-enabled raw-hook oracle: the
    oracle's hooks turn the tapped block output into a grad leaf
    (``requires_grad_`` on a frozen graph, ``retain_grad`` on a live one) —
    the exact contract the compiler implements — then backward the same
    scalar and compare ``.grad`` element-for-element."""

    pytestmark = pytest.mark.property

    @pytest.fixture(params=["llama", "gpt2"])
    def case(self, request, llama_case: PlanCase, gpt2_case: PlanCase) -> PlanCase:
        return {"llama": llama_case, "gpt2": gpt2_case}[request.param]

    @staticmethod
    def _leafy_forward(
        case: PlanCase, inputs: Any, layers: list[int]
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Raw-hook forward with each layer's block output made a grad leaf;
        returns ``(per-layer activations, logits)`` — the caller backwards
        its own scalar and reads ``.grad`` off the activations."""
        stash: dict[int, torch.Tensor] = {}
        handles = []
        for layer in layers:
            module, kind = component_module(case.oracle, layer, "block_output")
            assert kind == "out"

            def hook(_m, _i, out, layer: int = layer):
                hidden = hidden_of(out)
                if hidden.requires_grad:
                    hidden.retain_grad()
                else:
                    hidden.requires_grad_(True)
                stash[layer] = hidden
                return out

            handles.append(module.register_forward_hook(hook))
        try:
            out = case.oracle.hf_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        finally:
            for handle in handles:
                handle.remove()
        return stash, out.logits

    def test_logit_loss_gradients_match_hook_oracle(self, case: PlanCase) -> None:
        """Full-tensor and positions-gathered gradients of a final-position
        logit difference — the attribution-patching shape."""
        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)

        def scalar(logits: torch.Tensor) -> torch.Tensor:
            return (logits[:, -1, 7] - logits[:, -1, 3]).sum()

        plan = Plan(
            inputs={"base": base},
            ops=(
                CollectOp("base", _resid(0), key="h0"),
                CollectOp("base", _resid(1), key="h1", positions=[0, 2]),
            ),
            save_logits=("base",),
            gradients=GradientRequest(
                loss=lambda values: scalar(values["base"]), wrt=("h0", "h1")
            ),
        )
        result = run_plan(case.st, plan)

        stash, logits = self._leafy_forward(case, base, [0, 1])
        scalar(logits).backward()
        assert stash[0].grad is not None and stash[1].grad is not None
        torch.testing.assert_close(result.gradients["h0"], stash[0].grad)
        torch.testing.assert_close(result.gradients["h1"], stash[1].grad[:, [0, 2]])
        # the gradient is shaped like the collect it names, and every
        # returned tensor is CPU + detached (the PlanResult convention)
        assert result.gradients["h1"].shape == result.collects["h1"].shape
        assert not result.logits["base"].requires_grad
        assert not result.collects["h0"].requires_grad
        assert result.logits["base"].device.type == "cpu"

    def test_collect_only_loss_backward(self, case: PlanCase) -> None:
        """A loss over a deeper collect — no logits saved, so the trace
        early-stops after the deepest tap — still delivers the shallower
        site's gradient through the partial graph."""
        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        plan = Plan(
            inputs={"base": base},
            ops=(
                CollectOp("base", _resid(0), key="h0"),
                CollectOp("base", _resid(1), key="h1"),
            ),
            gradients=GradientRequest(
                loss=lambda values: values["h1"].float().pow(2).sum(), wrt=("h0",)
            ),
        )
        result = run_plan(case.st, plan)

        stash, _logits = self._leafy_forward(case, base, [0, 1])
        stash[1].float().pow(2).sum().backward()
        assert stash[0].grad is not None
        torch.testing.assert_close(result.gradients["h0"], stash[0].grad)
        torch.testing.assert_close(result.collects["h1"], stash[1].detach())

    def test_multi_invoke_gradient_plan_refused(self, case: PlanCase) -> None:
        """Measured boundary: an invoke's row-scoped reads branch off the
        fused multi-invoke forward, so gradient plans are single-input."""
        base, source = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                CollectOp("source", _resid(0), key="hs"),
                CollectOp("base", _resid(0), key="hb"),
            ),
            save_logits=("base",),
            gradients=GradientRequest(
                loss=lambda values: values["base"].sum(), wrt=("hb",)
            ),
        )
        with pytest.raises(NotImplementedError, match="single-input plans only"):
            run_plan(case.st, plan)


# --------------------------------------------------------------------------- #
#  property — the generation lowering (CAP2, #455) vs a step-counting oracle   #
# --------------------------------------------------------------------------- #
def _oracle_step_generate(
    case: PlanCase,
    inputs: dict,
    edits: list[tuple[int, int, Any]],
    max_new_tokens: int,
    capture: tuple[int, int] | None = None,
    **gen_kwargs: Any,
) -> tuple[Any, torch.Tensor | None]:
    """HF ``generate`` with forward-pass-counting raw hooks — the decode-step
    ground truth. ``edits``: ``(layer, pass_idx, fn)`` triples; at that
    layer's ``pass_idx``-th forward (0 = prefill, k = the k-th KV-cached
    decode step) ``fn`` mutates the full hidden frame in place. ``capture``:
    ``(layer, pass_idx)`` whose frame to also grab. Defaults mirror the
    generation lowering's.
    """
    per_layer: dict[int, dict[int, Any]] = {}
    for layer, pass_idx, fn in edits:
        per_layer.setdefault(layer, {})[pass_idx] = fn
    if capture is not None:
        per_layer.setdefault(capture[0], {})
    counters = {layer: 0 for layer in per_layer}
    grabbed: dict[str, torch.Tensor] = {}
    handles = []
    for layer, at in per_layer.items():
        module, _ = component_module(case.oracle, layer, "block_output")

        def hook(_m, _i, out, layer=layer, at=at):
            h = hidden_of(out)
            n = counters[layer]
            if n in at:
                at[n](h)
            if capture == (layer, n):
                grabbed["value"] = h.detach().clone()
            counters[layer] += 1
            return (h, *out[1:]) if isinstance(out, tuple) else h

        handles.append(module.register_forward_hook(hook))
    defaults: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=case.tok.pad_token_id,
    )
    defaults.update(gen_kwargs)
    try:
        with torch.no_grad():
            out = case.oracle.hf_model.generate(**inputs, **defaults)
    finally:
        for handle in handles:
            handle.remove()
    return out, grabbed.get("value")


def _steer_edit(layer: int, vec: torch.Tensor, scale: float, positions) -> Edit:
    """A steering Edit: the vector rides a constant ReadSource (aux), so the
    test also exercises the read-source coercion path inside iter."""
    return Edit(
        _resid(layer),
        g=lambda f, v: f + scale * v,
        read_sources=(ReadSource(vec),),
        positions=positions,
    )


class TestGeneratePlanProperty:
    pytestmark = pytest.mark.property

    _N = 5  # max_new_tokens
    _LAYER = 1

    @pytest.fixture(params=["llama", "gpt2"])
    def case(self, request, llama_case: PlanCase, gpt2_case: PlanCase) -> PlanCase:
        return {"llama": llama_case, "gpt2": gpt2_case}[request.param]

    def _steer_fn(self, vec: torch.Tensor, scale: float):
        def fn(h: torch.Tensor) -> None:
            h[:, -1, :] = h[:, -1, :] + (scale * vec).to(h.dtype)

        return fn

    def test_decode_step_steer_matches_oracle_and_flips_tokens(
        self, case: PlanCase
    ) -> None:
        """A steer addressed to decode step 2 fires at exactly that forward
        pass — the capability pyvene structurally forbade: steps 0-1 stay
        byte-identical to clean, the generated tokens change from step 2 on.
        """
        inputs = case.batch(_BASE_TEXT, _SOURCE_TEXT)
        vec = torch.linspace(-3.0, 3.0, case.hidden())
        scale = 50.0
        plan = Plan(
            inputs={"base": inputs},
            ops=(EditOp("base", _steer_edit(self._LAYER, vec, scale, [-1]), step=2),),
            generate=GenerateSpec(max_new_tokens=self._N, output_scores=True),
        )
        with torch.no_grad():
            result = run_plan(case.st, plan)

        want, _ = _oracle_step_generate(
            case,
            inputs,
            [(self._LAYER, 2, self._steer_fn(vec, scale))],
            self._N,
            min_new_tokens=3,  # the lowering's default: last step + 1
        )
        prompt_len = inputs["input_ids"].shape[1]
        assert torch.equal(result.sequences["base"], want.sequences[:, prompt_len:])
        for a, b in zip(result.scores["base"], want.scores):
            torch.testing.assert_close(a.float(), b.float(), atol=1e-5, rtol=1e-4)

        clean, _ = _oracle_step_generate(case, inputs, [], self._N, min_new_tokens=3)
        clean_seq = clean.sequences[:, prompt_len:]
        # steps before the edit are untouched; the edited step's tokens flip
        assert torch.equal(result.sequences["base"][:, :2], clean_seq[:, :2])
        for step in (0, 1):
            torch.testing.assert_close(
                result.scores["base"][step].float(), clean.scores[step].float()
            )
        assert not torch.equal(result.sequences["base"][:, 2:], clean_seq[:, 2:])

    def test_edit_at_every_step_matches_oracle(self, case: PlanCase) -> None:
        """One EditOp per step 0..N-1 — steering at EVERY generated token
        (step 0 = the prefill) — matches N forward-counting oracle hooks."""
        inputs = case.batch(_BASE_TEXT, _SOURCE_TEXT)
        vec = torch.linspace(1.0, -1.0, case.hidden())
        scale = 8.0
        plan = Plan(
            inputs={"base": inputs},
            ops=tuple(
                EditOp("base", _steer_edit(self._LAYER, vec, scale, [-1]), step=s)
                for s in range(self._N)
            ),
            generate=GenerateSpec(max_new_tokens=self._N, output_scores=True),
        )
        with torch.no_grad():
            result = run_plan(case.st, plan)

        want, _ = _oracle_step_generate(
            case,
            inputs,
            [(self._LAYER, s, self._steer_fn(vec, scale)) for s in range(self._N)],
            self._N,
            min_new_tokens=self._N,
        )
        prompt_len = inputs["input_ids"].shape[1]
        assert torch.equal(result.sequences["base"], want.sequences[:, prompt_len:])
        for a, b in zip(result.scores["base"], want.scores):
            torch.testing.assert_close(a.float(), b.float(), atol=1e-5, rtol=1e-4)

    def test_prefill_and_stepped_ops_coexist(self, case: PlanCase) -> None:
        """A step-less op (prefill semantics — pyvene's intervene_on_prompt)
        and a decode-step op share one generate trace."""
        inputs = case.batch(_BASE_TEXT, _SOURCE_TEXT)
        pre_vec = torch.linspace(2.0, -2.0, case.hidden())
        vec = torch.linspace(-3.0, 3.0, case.hidden())
        plan = Plan(
            inputs={"base": inputs},
            ops=(
                EditOp("base", _steer_edit(0, pre_vec, 1.0, None)),  # prefill
                EditOp("base", _steer_edit(self._LAYER, vec, 10.0, [-1]), step=2),
            ),
            generate=GenerateSpec(max_new_tokens=self._N, output_scores=True),
        )
        with torch.no_grad():
            result = run_plan(case.st, plan)

        def prefill_fn(h: torch.Tensor) -> None:
            h += pre_vec.to(h.dtype)

        want, _ = _oracle_step_generate(
            case,
            inputs,
            [(0, 0, prefill_fn), (self._LAYER, 2, self._steer_fn(vec, 10.0))],
            self._N,
            min_new_tokens=3,
        )
        prompt_len = inputs["input_ids"].shape[1]
        assert torch.equal(result.sequences["base"], want.sequences[:, prompt_len:])
        for a, b in zip(result.scores["base"], want.scores):
            torch.testing.assert_close(a.float(), b.float(), atol=1e-5, rtol=1e-4)

    def test_decode_step_collect_matches_oracle(self, case: PlanCase) -> None:
        """A collect at decode step k returns that pass's one-token frame."""
        inputs = case.batch(_BASE_TEXT, _SOURCE_TEXT)
        plan = Plan(
            inputs={"base": inputs},
            ops=(CollectOp("base", _resid(self._LAYER), key="h3", step=3),),
            generate=GenerateSpec(max_new_tokens=self._N),
        )
        with torch.no_grad():
            result = run_plan(case.st, plan)

        _, grabbed = _oracle_step_generate(
            case,
            inputs,
            [],
            self._N,
            capture=(self._LAYER, 3),
            min_new_tokens=4,
        )
        assert result.collects["h3"].shape == (2, 1, case.hidden())
        torch.testing.assert_close(
            result.collects["h3"].float(),
            grabbed.float(),
            atol=1e-5,
            rtol=1e-4,
        )
        assert result.sequences["base"].shape == (2, self._N)
        assert result.scores == {}  # output_scores defaults off

    def test_cross_input_read_feeds_decode_step_as_constant(
        self, case: PlanCase
    ) -> None:
        """EU3 (#484): a decode-step edit may read ANOTHER plan input — the
        scheduler captures the source read in an earlier collect stage (a
        generate trace accepts only constants) and the consume tap reads the
        saved value at its step, through the same aux/coerce path a constant
        ReadSource takes. Oracle: raw-hook capture of the source activation,
        written at the base's decode pass 2. Rows are swapped between the
        two batches so a wrong-input capture is numerically visible."""
        base = case.batch(_BASE_TEXT, _SOURCE_TEXT)
        source = case.batch(_SOURCE_TEXT, _BASE_TEXT)
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        _resid(self._LAYER),
                        g=lambda f, f_src: f_src,
                        read_sources=(
                            ReadSource(
                                _resid(self._LAYER), positions=[-1], input="source"
                            ),
                        ),
                        positions=[-1],
                    ),
                    step=2,
                ),
            ),
            generate=GenerateSpec(max_new_tokens=self._N, output_scores=True),
        )
        with torch.no_grad():
            result = run_plan(case.st, plan)

        src_vals = case.capture("block_output", self._LAYER, source)[:, [-1], :]

        def write_fn(h: torch.Tensor) -> None:
            h[:, [-1], :] = src_vals.to(h.dtype)

        want, _ = _oracle_step_generate(
            case,
            base,
            [(self._LAYER, 2, write_fn)],
            self._N,
            min_new_tokens=3,  # the lowering's default: last step + 1
        )
        prompt_len = base["input_ids"].shape[1]
        assert torch.equal(result.sequences["base"], want.sequences[:, prompt_len:])
        for a, b in zip(result.scores["base"], want.scores):
            torch.testing.assert_close(a.float(), b.float(), atol=1e-5, rtol=1e-4)
        # steps before the edit are untouched; the edited step's scores move
        clean, _ = _oracle_step_generate(case, base, [], self._N, min_new_tokens=3)
        for step in (0, 1):
            torch.testing.assert_close(
                result.scores["base"][step].float(), clean.scores[step].float()
            )
        assert not torch.allclose(
            result.scores["base"][2].float(), clean.scores[2].float(), atol=1e-5
        ), "inert cross-input constant"

    def test_same_input_forward_read_inside_decode_step(self, case: PlanCase) -> None:
        """The one in-generate-trace behavioral change of EU3 (#484): read
        taps now ``.save()`` unconditionally (the retired generation path
        built its per-step taps with ``save=False``), including a same-input
        FORWARD read that runs INSIDE a ``tracer.iter`` decode iteration —
        the read stays in the generate trace (no collect stage, no clean
        pass), feeding its stepped edit through the slot in the same
        one-token frame. Oracle: two forward-pass-counting hooks at pass 2 —
        layer 0 stashes its frame, layer 1 applies the same g."""
        inputs = case.batch(_BASE_TEXT, _SOURCE_TEXT)
        plan = Plan(
            inputs={"base": inputs},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        _resid(1),
                        g=lambda f, f_src: f + 2.0 * f_src,
                        read_sources=(ReadSource(_resid(0), positions=[-1]),),
                        positions=[-1],
                    ),
                    step=2,
                ),
            ),
            generate=GenerateSpec(max_new_tokens=self._N, output_scores=True),
        )
        # the read is same-input and forward (layer 0 -> layer 1): no edge,
        # no collect stage — ONE terminal generate trace
        from causalab.neural.staged import lower_staged

        program = lower_staged(case.st, plan)
        assert program.generate_key == "base"
        assert program.stages == ()
        with torch.no_grad():
            result = run_plan(case.st, plan)

        stash: dict[str, torch.Tensor] = {}

        def read_fn(h: torch.Tensor) -> None:
            stash["v"] = h[:, -1, :].detach().clone()

        def write_fn(h: torch.Tensor) -> None:
            h[:, -1, :] = h[:, -1, :] + 2.0 * stash["v"].to(h.dtype)

        want, _ = _oracle_step_generate(
            case,
            inputs,
            [(0, 2, read_fn), (1, 2, write_fn)],
            self._N,
            min_new_tokens=3,  # the lowering's default: last step + 1
        )
        prompt_len = inputs["input_ids"].shape[1]
        assert torch.equal(result.sequences["base"], want.sequences[:, prompt_len:])
        for a, b in zip(result.scores["base"], want.scores):
            torch.testing.assert_close(a.float(), b.float(), atol=1e-5, rtol=1e-4)
        clean, _ = _oracle_step_generate(case, inputs, [], self._N, min_new_tokens=3)
        for step in (0, 1):
            torch.testing.assert_close(
                result.scores["base"][step].float(), clean.scores[step].float()
            )
        assert not torch.allclose(
            result.scores["base"][2].float(), clean.scores[2].float(), atol=1e-5
        ), "inert in-step read"

    def test_stage_less_generation_plan_passes_single_strictness(
        self, case: PlanCase
    ) -> None:
        """The acceptance side of ``lowering="single"`` × generation (EU3,
        #484 — the refusing side is pinned in TestScheduleUnit): a
        generation plan with no cross-input or clean-pass reads schedules as
        exactly ONE trace (the terminal generate trace), so strictness runs
        it — byte-identical to auto — rather than refusing. Guards a future
        ``num_traces`` miscount from refusing a legal plan."""
        inputs = case.batch(_BASE_TEXT, _SOURCE_TEXT)
        vec = torch.linspace(-1.0, 1.0, case.hidden())
        plan = Plan(
            inputs={"base": inputs},
            ops=(EditOp("base", _steer_edit(self._LAYER, vec, 5.0, [-1]), step=1),),
            generate=GenerateSpec(max_new_tokens=self._N, output_scores=True),
        )
        with torch.no_grad():
            strict = run_plan(case.st, plan, lowering="single")
            auto = run_plan(case.st, plan)
        assert torch.equal(strict.sequences["base"], auto.sequences["base"])
        for a, b in zip(strict.scores["base"], auto.scores["base"]):
            assert torch.equal(a, b)

    def test_same_input_backward_read_reads_clean_prefill_pass(
        self, case: PlanCase
    ) -> None:
        """EU3 (#484): a same-input read AFTER the written site in a
        generation plan reroutes to the ``(input, "clean")`` collect stage —
        a plain pass over the same tensors — so the value written at the
        prefill is the input's CLEAN layer-1 activation in the PREFILL
        frame (the GenerateSpec contract; the per-step
        StagingRequired→ValueError wrapper died). Oracle: capture clean
        layer 1 during a REAL HF-generate prefill (pad-aware positions —
        the clean pass mirrors generate's own prefill numbering, review
        #492 F1), write it at layer 0 of the prefill pass, HF-generate."""
        base = case.batch(_BASE_TEXT, _SOURCE_TEXT)
        plan = Plan(
            inputs={"base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        _resid(0),
                        g=lambda f, f_src: f_src,
                        read_sources=(ReadSource(_resid(1), positions=[-1]),),
                        positions=[-1],
                    ),
                ),
            ),
            generate=GenerateSpec(max_new_tokens=self._N, output_scores=True),
        )
        with torch.no_grad():
            result = run_plan(case.st, plan)

        _, prefill_l1 = _oracle_step_generate(case, base, [], self._N, capture=(1, 0))
        assert prefill_l1 is not None
        clean_l1 = prefill_l1[:, [-1], :]

        def write_fn(h: torch.Tensor) -> None:
            h[:, [-1], :] = clean_l1.to(h.dtype)

        want, _ = _oracle_step_generate(case, base, [(0, 0, write_fn)], self._N)
        prompt_len = base["input_ids"].shape[1]
        assert torch.equal(result.sequences["base"], want.sequences[:, prompt_len:])
        for a, b in zip(result.scores["base"], want.scores):
            torch.testing.assert_close(a.float(), b.float(), atol=1e-5, rtol=1e-4)
        clean, _ = _oracle_step_generate(case, base, [], self._N)
        assert not torch.allclose(
            result.scores["base"][0].float(), clean.scores[0].float(), atol=1e-5
        ), "inert self-graft"

    def test_clean_prefill_pass_is_pad_aware_on_absolute_positions(
        self, gpt2_case: PlanCase
    ) -> None:
        """The hidden ``(input, "clean")`` pass of a generation plan numbers
        positions pad-aware from the attention mask — matching HF
        ``generate``'s own prefill — rather than the pad-blind ``arange`` a
        bare plain forward defaults to (review #492 F1). On a left-padded
        batch of an absolute-position model (GPT-2) the two disagree on the
        padded row, so a pad-blind clean pass would graft a value that is
        NOT the model's actual clean prefill activation. Oracle: the
        layer-1 frame captured during a REAL HF-generate prefill; the
        teeth-guard asserts the pad-blind plain forward genuinely differs
        there."""
        case = gpt2_case
        base = case.batch(_BASE_TEXT, _SOURCE_TEXT)
        padded_rows = base["attention_mask"][:, 0] == 0
        assert padded_rows.any(), "the batch must contain a left-padded row"
        plan = Plan(
            inputs={"base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        _resid(0),
                        g=lambda f, f_src: f_src,
                        read_sources=(ReadSource(_resid(1), positions=[-1]),),
                        positions=[-1],
                    ),
                ),
            ),
            generate=GenerateSpec(max_new_tokens=self._N, output_scores=True),
        )
        with torch.no_grad():
            result = run_plan(case.st, plan)

        # Pad-aware ground truth: layer 1's frame at the real generate
        # prefill (HF numbers it from the attention mask itself).
        _, prefill_l1 = _oracle_step_generate(case, base, [], self._N, capture=(1, 0))
        assert prefill_l1 is not None
        clean_l1 = prefill_l1[:, [-1], :]
        # Teeth: the pad-blind plain forward (default arange positions —
        # what the clean pass ran before the fix) disagrees with the real
        # prefill on the padded row.
        pad_blind_l1 = case.capture("block_output", 1, base)[:, [-1], :]
        assert not torch.allclose(
            pad_blind_l1[padded_rows], clean_l1[padded_rows], atol=1e-5
        ), "pad-blind and pad-aware prefills agree — the guard lost its teeth"

        def write_fn(h: torch.Tensor) -> None:
            h[:, [-1], :] = clean_l1.to(h.dtype)

        want, _ = _oracle_step_generate(case, base, [(0, 0, write_fn)], self._N)
        prompt_len = base["input_ids"].shape[1]
        assert torch.equal(result.sequences["base"], want.sequences[:, prompt_len:])
        for a, b in zip(result.scores["base"], want.scores):
            torch.testing.assert_close(a.float(), b.float(), atol=1e-5, rtol=1e-4)

    def test_single_step_position_ids_allowed_and_matches_oracle(
        self, case: PlanCase
    ) -> None:
        """The ALLOWED side of the position_ids contract: for the
        prefill-only case (max_new_tokens == 1) prompt-shaped position_ids
        pass through the generate trace, match an oracle given the same
        explicit key, and are behavior-preserving vs the bare run — HF
        ``generate`` numbers the prefill left-pad-aware itself, so the
        correct explicit key must reproduce it exactly. The negative control
        on GPT-2 (absolute positions): a pad-blind ``arange`` key changes
        the padded rows' scores, proving the key genuinely reaches the
        forward (the multi-step refusal is about a real effect, not a
        dropped kwarg)."""
        base = case.batch(_BASE_TEXT, _SOURCE_TEXT)
        inputs = ensure_position_ids(base)
        assert "position_ids" in inputs
        vec = torch.linspace(-1.0, 1.0, case.hidden())
        plan = Plan(
            inputs={"base": inputs},
            ops=(EditOp("base", _steer_edit(self._LAYER, vec, 5.0, [-1])),),
            generate=GenerateSpec(max_new_tokens=1, output_scores=True),
        )
        with torch.no_grad():
            result = run_plan(case.st, plan)

        edits = [(self._LAYER, 0, self._steer_fn(vec, 5.0))]
        want, _ = _oracle_step_generate(case, inputs, edits, 1)
        prompt_len = inputs["input_ids"].shape[1]
        assert result.sequences["base"].shape == (2, 1)
        assert torch.equal(result.sequences["base"], want.sequences[:, prompt_len:])
        torch.testing.assert_close(
            result.scores["base"][0].float(),
            want.scores[0].float(),
            atol=1e-5,
            rtol=1e-4,
        )
        # behavior-preserving: the explicit key equals HF's own numbering
        bare, _ = _oracle_step_generate(case, base, edits, 1)
        torch.testing.assert_close(
            want.scores[0].float(), bare.scores[0].float(), atol=1e-5, rtol=1e-4
        )
        if case.oracle.hf_model.config.model_type == "gpt2":
            seq_len = base["input_ids"].shape[1]
            pad_blind = {
                **base,
                "position_ids": torch.arange(seq_len).expand(2, -1),
            }
            wrong, _ = _oracle_step_generate(case, pad_blind, edits, 1)
            assert not torch.allclose(
                wrong.scores[0].float(), bare.scores[0].float(), atol=1e-4
            ), "pad-blind position_ids had no effect — did the key get dropped?"

    def test_abandoned_trace_body_raises_legibly(self, llama_case: PlanCase) -> None:
        """Defense-in-depth behind the min_new_tokens gate: a kwargs stopping
        criterion that ends generation before an addressed step abandons the
        trace body (nnsight skips the unfulfilled iteration silently, so the
        generator-output save never runs) — backstopped by a legible
        RuntimeError instead of a bare IndexError."""
        from transformers import MaxLengthCriteria, StoppingCriteriaList

        case = llama_case
        inputs = case.batch(_BASE_TEXT, _SOURCE_TEXT)
        prompt_len = inputs["input_ids"].shape[1]
        vec = torch.linspace(-1.0, 1.0, case.hidden())
        plan = Plan(
            inputs={"base": inputs},
            ops=(EditOp("base", _steer_edit(self._LAYER, vec, 5.0, [-1]), step=3),),
            generate=GenerateSpec(
                max_new_tokens=self._N,
                kwargs={
                    # stops after 2 generated tokens — before step 3 —
                    # sidestepping min_new_tokens (which only suppresses EOS)
                    "stopping_criteria": StoppingCriteriaList(
                        [MaxLengthCriteria(max_length=prompt_len + 2)]
                    )
                },
            ),
        )
        with pytest.raises(RuntimeError, match="abandoned"):
            with torch.no_grad():
                run_plan(case.st, plan)


# --------------------------------------------------------------------------- #
#  numerical_unit — tracer.stop() early-exit (CAP6, #459)                       #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def deep_llama_case() -> PlanCase:
    """A 4-layer fresh tiny llama — the default stub has only 2 layers, too
    few to observe layers past the deepest tap not running."""

    def deepen(cfg: Any) -> None:
        cfg.num_hidden_layers = 4

    return _case(*fresh_tiny_random_llama(mutate_config=deepen))


class TestEarlyStopNumerical:
    """The CAP6 (#459) contract, pinned against raw ``register_forward_hook``
    fire-counters on a 4-layer tiny llama (CPU): a plan that saves no logits
    stops its forward after the deepest tap — layers past it NEVER run — and
    the collected activations are bitwise-identical to the no-stop path (the
    same plan with ``save_logits``, whose full forward is the control)."""

    pytestmark = pytest.mark.numerical_unit

    def test_single_input_collect_only_stops_after_deepest_tap(
        self, deep_llama_case: PlanCase
    ) -> None:
        case = deep_llama_case
        base, _ = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        ops = (
            CollectOp("base", _resid(0), key="b0"),
            CollectOp("base", _resid(1), key="b1"),
        )
        with layer_fire_counts(case.oracle) as counts:
            stopped = run_plan(case.st, Plan(inputs={"base": base}, ops=ops))
        assert counts == [1, 1, 0, 0], counts
        with layer_fire_counts(case.oracle) as counts:
            control = run_plan(
                case.st, Plan(inputs={"base": base}, ops=ops, save_logits=("base",))
            )
        assert counts == [1, 1, 1, 1], counts
        for key in ("b0", "b1"):
            assert torch.equal(stopped.collects[key], control.collects[key])

    def test_no_edge_multi_input_runs_per_input_traces_each_stopping(
        self, deep_llama_case: PlanCase
    ) -> None:
        """A no-edge multi-input plan schedules one trace per input (EU2
        #483 — grouping is connected-components; value-identical to the
        retired fused forward), and each trace stops after its OWN deepest
        tap: source (collect at layer 0) fires layer 0 only, base (collects
        at layers 0-1) fires layers 0-1."""
        case = deep_llama_case
        base, source = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        ops = (
            CollectOp("source", _resid(0), key="s0"),
            CollectOp("base", _resid(1), key="b1"),
            CollectOp("base", _resid(0), key="b0"),
        )
        with layer_fire_counts(case.oracle) as counts:
            stopped = run_plan(
                case.st, Plan(inputs={"source": source, "base": base}, ops=ops)
            )
        assert counts == [2, 1, 0, 0], counts
        with layer_fire_counts(case.oracle) as counts:
            control = run_plan(
                case.st,
                Plan(
                    inputs={"source": source, "base": base},
                    ops=ops,
                    save_logits=("base", "source"),
                ),
            )
        assert counts == [2, 2, 2, 2], counts
        for key in ("s0", "b1", "b0"):
            assert torch.equal(stopped.collects[key], control.collects[key])

    def test_fused_multi_invoke_stop_rides_trace_wide_deepest_tap(
        self, deep_llama_case: PlanCase
    ) -> None:
        """The FUSED multi-invoke stop carrier (the pre-EU2 pin, restored in
        edge-connected form): in ONE fused trace with asymmetric per-invoke
        depths — source's taps (own collect + produce read) end at layer 0,
        base's collect sits at layer 1 — the stop rides the invoke touching
        the TRACE-WIDE deepest hook (base), after every other invoke's taps:
        one fused forward fires layers 0-1 exactly once ([1, 1, 0, 0], not a
        per-input [2, ...] layout), and source's shallower collect still
        lands before the stop (bitwise vs. the full-forward control)."""
        case = deep_llama_case
        base, source = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        last = base["input_ids"].shape[1] - 1
        ops = (
            CollectOp("source", _resid(0), key="s0"),
            EditOp(
                "base",
                Edit(
                    _resid(0),
                    g=lambda f, f_src: f_src,
                    read_sources=(
                        ReadSource(_resid(0), positions=[last], input="source"),
                    ),
                    positions=[last],
                ),
            ),
            CollectOp("base", _resid(1), key="b1"),
        )
        with layer_fire_counts(case.oracle) as counts:
            stopped = run_plan(
                case.st, Plan(inputs={"source": source, "base": base}, ops=ops)
            )
        assert counts == [1, 1, 0, 0], counts
        with layer_fire_counts(case.oracle) as counts:
            control = run_plan(
                case.st,
                Plan(
                    inputs={"source": source, "base": base},
                    ops=ops,
                    save_logits=("base",),
                ),
            )
        assert counts == [1, 1, 1, 1], counts
        for key in ("s0", "b1"):
            assert torch.equal(stopped.collects[key], control.collects[key])

    def test_interchange_collect_under_patch_stops_early(
        self, deep_llama_case: PlanCase
    ) -> None:
        """The stop composes with the cross-invoke barrier: source produce at
        layer 0, base consume at layer 0, collect under the patch at layer 1
        — the stop fires after the collect, and the collected value still
        reflects the interchange (bitwise vs. the full-forward control)."""
        case = deep_llama_case
        base, source = case.pair(_BASE_TEXT, _SOURCE_TEXT)
        last = base["input_ids"].shape[1] - 1
        ops = (
            EditOp(
                "base",
                Edit(
                    _resid(0),
                    g=lambda f, f_src: f_src,
                    read_sources=(
                        ReadSource(_resid(0), positions=[last], input="source"),
                    ),
                    positions=[last],
                ),
            ),
            CollectOp("base", _resid(1), key="mid"),
        )
        with layer_fire_counts(case.oracle) as counts:
            stopped = run_plan(
                case.st, Plan(inputs={"source": source, "base": base}, ops=ops)
            )
        assert counts == [1, 1, 0, 0], counts
        control = run_plan(
            case.st,
            Plan(
                inputs={"source": source, "base": base},
                ops=ops,
                save_logits=("base",),
            ),
        )
        assert torch.equal(stopped.collects["mid"], control.collects["mid"])
        clean_mid = case.capture("block_output", 1, base)
        assert not torch.allclose(stopped.collects["mid"], clean_mid, atol=1e-4), (
            "inert patch"
        )


# --------------------------------------------------------------------------- #
#  golden — the canonical interchange on the real Qwen3-4B backbone            #
# --------------------------------------------------------------------------- #
class TestChatCoherentGolden:
    """The canonical cross-invoke interchange on the coherent GPU backbone
    (Qwen3-4B), matched against the same capture-then-patch raw-hook oracle."""

    pytestmark = pytest.mark.golden

    def test_canonical_interchange_matches_oracle_on_coherent_model(self) -> None:
        st = StandardizedTransformer(
            "Qwen/Qwen3-4B-Instruct-2507", dispatch=True, device_map="auto"
        )
        raw = st._model
        oracle = SimpleNamespace(hf_model=raw)
        tok = st.tokenizer
        tok.padding_side = "left"
        device = next(raw.parameters()).device
        enc = tok([_BASE_TEXT, _SOURCE_TEXT], padding=True, return_tensors="pt")

        def row(i: int) -> dict:
            return {
                "input_ids": enc["input_ids"][i : i + 1].to(device),
                "attention_mask": enc["attention_mask"][i : i + 1].to(device),
            }

        base, source = row(0), row(1)
        last = int(base["input_ids"].shape[1] - 1)
        layer = int(st.num_layers) // 2
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        _resid(layer),
                        g=lambda f, f_src: f_src,
                        read_sources=(
                            ReadSource(_resid(layer), positions=[last], input="source"),
                        ),
                        positions=[last],
                    ),
                ),
                CollectOp("base", _resid(layer + 1), key="mid", positions=[last]),
            ),
            save_logits=("base",),
        )
        result = run_plan(st, plan)
        assert result.collects["mid"].shape[1] == 1

        module, kind = component_module(oracle, layer, "block_output")
        src_vals = capture_component(oracle, module, kind, source)[:, [last]]
        patched = next_token_logits(
            oracle, base, layer=layer, positions=[last], patch_values=src_vals
        )
        clean = next_token_logits(oracle, base)
        assert not torch.allclose(patched, clean.float(), atol=1e-2), "inert patch"
        torch.testing.assert_close(
            result.logits["base"][:, -1, :].float(),
            patched.cpu().float(),
            atol=1e-3,
            rtol=1e-3,
        )


# --------------------------------------------------------------------------- #
#  golden — a decode-step steered-generation chain on Qwen3-4B (CAP2, #455)    #
# --------------------------------------------------------------------------- #
class TestGenerateGoldenChatCoherent:
    """A representative steered-generation chain on the coherent GPU backbone:
    one steering vector applied at EVERY generation step (prefill + each
    KV-cached decode pass) at a mid layer, matched against the same
    forward-pass-counting raw-hook oracle the property tier pins — and
    asserted non-vacuous (the steer changes the generated tokens)."""

    pytestmark = pytest.mark.golden

    def test_decode_step_steer_chain_matches_oracle_on_coherent_model(self) -> None:
        n_steps = 8
        st = StandardizedTransformer(
            "Qwen/Qwen3-4B-Instruct-2507", dispatch=True, device_map="auto"
        )
        raw = st._model
        case = PlanCase(st=st, oracle=SimpleNamespace(hf_model=raw), tok=st.tokenizer)
        case.tok.padding_side = "left"
        device = next(raw.parameters()).device
        enc = case.tok([_BASE_TEXT, _SOURCE_TEXT], padding=True, return_tensors="pt")
        inputs = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
        layer = int(st.num_layers) // 2
        vec = torch.linspace(-2.0, 2.0, case.hidden())
        scale = 10.0

        plan = Plan(
            inputs={"base": inputs},
            ops=tuple(
                EditOp("base", _steer_edit(layer, vec, scale, [-1]), step=s)
                for s in range(n_steps)
            ),
            generate=GenerateSpec(max_new_tokens=n_steps, output_scores=True),
        )
        with torch.no_grad():
            result = run_plan(st, plan)

        def steer_fn(h: torch.Tensor) -> None:
            h[:, -1, :] = h[:, -1, :] + (scale * vec).to(h.device, h.dtype)

        want, _ = _oracle_step_generate(
            case,
            inputs,
            [(layer, s, steer_fn) for s in range(n_steps)],
            n_steps,
            min_new_tokens=n_steps,
        )
        clean, _ = _oracle_step_generate(
            case, inputs, [], n_steps, min_new_tokens=n_steps
        )
        prompt_len = inputs["input_ids"].shape[1]
        assert not torch.equal(
            want.sequences[:, prompt_len:].cpu(), clean.sequences[:, prompt_len:].cpu()
        ), "inert steer — raise the scale"
        assert torch.equal(
            result.sequences["base"], want.sequences[:, prompt_len:].cpu()
        )
        for step, (a, b) in enumerate(zip(result.scores["base"], want.scores)):
            torch.testing.assert_close(
                a.float(),
                b.float().cpu(),
                atol=1e-2,
                rtol=1e-2,
                msg=lambda m, step=step: f"score step {step}: {m}",
            )


# --------------------------------------------------------------------------- #
#  unit — PlanResult surface                                                   #
# --------------------------------------------------------------------------- #
class TestPlanResultUnit:
    pytestmark = pytest.mark.unit

    def test_plan_result_fields(self) -> None:
        r = PlanResult(collects={"a": torch.zeros(1)}, logits={})
        assert set(r.collects) == {"a"}
        assert r.logits == {}
        assert r.sequences == {} and r.scores == {}
