"""Tests for :mod:`causalab.neural.persistent` — CAP7 ``model.edit()``
persistent interventions (#460).

Tiers (``causalab/neural`` owes ``unit`` + ``property``):

* ``unit`` — the lifecycle refusals, no forward pass and no real model (a
  ``_StubModel`` stand-in proves they fire before any backbone access):
  read-only / cross-input / later-site / frame-bound edits are refused at
  install; the verifying read raises on registry↔backbone drift; the empty
  lifecycle calls behave (no-edit install refused, pristine-model reads and
  uninstalls are no-ops).
* ``property`` — on the fresh tiny Llama (CPU), the issue's contract: an
  edited model differs from base, the edit persists across multiple forwards
  and batch shapes, uninstall restores **bitwise-identical** outputs; plus
  the compose-or-refuse pins — persistent edit × traced Plan composition
  (same-site ordering, fused multi-invoke, and the CAP6 early-stop
  suppression fire-counted through every stop-emission site: collect_ordered,
  the single-input lowering, the generalized ``_emit_invokes`` stop, and the
  staged compiler — #459 × #460), traced generation composes, plain-HF
  ``LMPipeline.generate`` refuses, and the drift errors on out-of-band
  ``clear_edits()`` / raw ``model.edit()``.

Models come from the fresh (uncached) factories in ``tests/_helpers/tiny.py``
— never the session-cached ``tiny_random_model`` singleton (leftover hooks on
a shared instance break later nnsight traces; see the factory docstrings).
An autouse fixture force-uninstalls after every property test so a failing
test never leaks mediators into its neighbors.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from nnterp import StandardizedTransformer

from causalab.neural.edit import Edit, ReadSource
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.modes import replace, steer
from causalab.neural.persistent import (
    PersistentEditError,
    install_edits,
    installed_edits,
    persistent_edits,
    uninstall_edits,
)
from causalab.neural.plan import CollectOp, EditOp, Plan, run_plan
from causalab.neural.site import Site, backbone_has_edits

from tests._helpers.tiny import fresh_tiny_random_llama
from tests.neural.activations.hook_oracle import layer_fire_counts

_BATCH_A = ["the quick brown fox", "a slow green turtle sleeps"]
_BATCH_B = ["hello world", "one two three four five", "sixty"]
_FACTOR = 8.0


class _StubModel:
    """A model-free stand-in for the unit-tier refusal tests. A plain class,
    not a ``types.SimpleNamespace``: the persistent registry keeps *weak*
    keys, and ``SimpleNamespace`` is neither weak-referenceable nor (via its
    value ``__eq__``) hashable."""

    def __init__(self, **attrs: Any) -> None:
        self.__dict__.update(attrs)


def _resid(layer: int) -> FeaturizedSite:
    return FeaturizedSite(Site("block_output", layer))


# --------------------------------------------------------------------------- #
#  Fixtures — a fresh (uncached) StandardizedTransformer                       #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class EditCase:
    st: StandardizedTransformer
    tok: Any
    # The raw HF module: hidden size (wrapping a pre-loaded module leaves
    # st.config None) and the hook-oracle handle for layer_fire_counts.
    raw: Any

    @property
    def hidden(self) -> int:
        return int(self.raw.config.hidden_size)

    @property
    def oracle(self) -> _StubModel:
        return _StubModel(hf_model=self.raw)

    def encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        enc = self.tok(texts, padding=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def logits(self, texts: list[str]) -> torch.Tensor:
        with self.st.trace(self.encode(texts)):
            out = self.st.logits.cpu().save()
        return out

    def vector(self) -> torch.Tensor:
        return torch.ones(self.hidden)


@pytest.fixture(scope="module")
def case() -> EditCase:
    raw, tok = fresh_tiny_random_llama()
    tok.padding_side = "left"  # the pipeline convention; makes [-1] the last token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    st = StandardizedTransformer(raw, tokenizer=tok, check_renaming=True)
    st.dispatch()
    return EditCase(st=st, tok=tok, raw=raw)


@pytest.fixture(autouse=True)
def _no_leaked_edits(request: pytest.FixtureRequest):
    """Force-uninstall after every test in this file so a failure never leaks
    mediators (or registry entries) into its neighbors."""
    yield
    if "case" in request.fixturenames:
        uninstall_edits(request.getfixturevalue("case").st, force=True)


# --------------------------------------------------------------------------- #
#  unit — lifecycle refusals, no model                                         #
# --------------------------------------------------------------------------- #
class TestLifecycleRefusalsUnit:
    pytestmark = pytest.mark.unit

    def test_install_needs_at_least_one_edit(self):
        with pytest.raises(ValueError, match="at least one Edit"):
            install_edits(_StubModel())

    def test_non_edit_value_refused(self):
        with pytest.raises(TypeError, match="Edit values"):
            install_edits(_StubModel(), "not an edit")  # type: ignore[arg-type]

    def test_read_only_edit_refused(self):
        with pytest.raises(ValueError, match="read-only"):
            install_edits(_StubModel(), Edit(_resid(0)))

    def test_cross_input_read_source_refused(self):
        edit = Edit(
            _resid(1),
            g=lambda f, s: s,
            read_sources=(ReadSource(_resid(1), input="source"),),
        )
        with pytest.raises(ValueError, match="another plan input"):
            install_edits(_StubModel(), edit)

    def test_later_site_read_source_refused(self):
        edit = Edit(
            _resid(0),
            g=lambda f, s: s,
            read_sources=(ReadSource(_resid(1)),),
        )
        with pytest.raises(ValueError, match="fires after"):
            install_edits(_StubModel(), edit)

    @pytest.mark.parametrize(
        "positions",
        [
            [0],  # absolute index: counts from a frame-specific pad boundary
            [-2, 3],  # mixed — one absolute index poisons the row
            [[1], [2]],  # per-row: born from one batch's resolution
            torch.tensor([[1], [2]]),  # 2-D tensor per-row form
            torch.tensor([0, -1]),  # 1-D tensor with a non-negative entry
        ],
    )
    def test_frame_bound_positions_refused(self, positions):
        edit = steer(Site("block_output", 0), torch.ones(4), positions=positions)
        with pytest.raises(ValueError, match="frame-independent"):
            install_edits(_StubModel(), edit)

    def test_frame_bound_read_source_positions_refused(self):
        edit = Edit(
            _resid(1),
            g=lambda f, s: s,
            read_sources=(ReadSource(_resid(0), positions=[0]),),
            positions=[-1],
        )
        with pytest.raises(ValueError, match=r"read_sources\[0\].positions"):
            install_edits(_StubModel(), edit)

    def test_out_of_band_mediator_raises_on_read(self):
        # A backbone mediator this module never recorded (raw model.edit()).
        stranger = _StubModel(_default_mediators=[object()])
        with pytest.raises(PersistentEditError, match="raw model.edit"):
            installed_edits(stranger)

    def test_pristine_model_is_a_no_op(self):
        pristine = _StubModel()  # no .clear_edits — must never be called
        assert installed_edits(pristine) == ()
        assert uninstall_edits(pristine) == ()
        assert not backbone_has_edits(pristine)


# --------------------------------------------------------------------------- #
#  property — the issue contract + composition pins (tiny Llama, CPU)          #
# --------------------------------------------------------------------------- #
class TestPersistentEditLifecycle:
    pytestmark = pytest.mark.property

    def test_edit_differs_persists_and_uninstall_restores_bitwise(self, case: EditCase):
        base_a = case.logits(_BATCH_A)
        base_b = case.logits(_BATCH_B)

        edit = steer(Site("block_output", 1), case.vector(), factor=_FACTOR)
        assert install_edits(case.st, edit) == (edit,)
        assert installed_edits(case.st) == (edit,)
        assert backbone_has_edits(case.st)

        edited_a = case.logits(_BATCH_A)
        assert not torch.equal(edited_a, base_a)
        # persists across multiple forwards …
        assert torch.equal(case.logits(_BATCH_A), edited_a)
        # … and across a different batch shape (3 rows, different lengths)
        assert not torch.equal(case.logits(_BATCH_B), base_b)

        assert uninstall_edits(case.st) == (edit,)
        assert installed_edits(case.st) == ()
        assert not backbone_has_edits(case.st)
        assert torch.equal(case.logits(_BATCH_A), base_a)  # bitwise restore
        assert torch.equal(case.logits(_BATCH_B), base_b)

    def test_installs_stack_in_order(self, case: EditCase):
        first = steer(Site("block_output", 0), case.vector(), factor=1.0)
        second = steer(Site("block_output", 1), case.vector(), factor=2.0)
        install_edits(case.st, first)
        assert install_edits(case.st, second) == (first, second)
        assert installed_edits(case.st) == (first, second)
        assert uninstall_edits(case.st) == (first, second)

    def test_loop_installs_capture_their_own_edit(self, case: EditCase):
        """Two installs from one loop frame must keep their own factors —
        nnsight re-executes the captured body against the captured frame, and
        install goes through a per-call helper precisely so a shared loop
        variable cannot alias both mediators to the last edit."""
        site = _resid(1)
        inputs = case.encode(_BATCH_A)
        clean = site.collect(case.st, inputs)
        for factor in (1.0, 100.0):
            install_edits(
                case.st, steer(Site("block_output", 1), case.vector(), factor=factor)
            )
        steered = site.collect(case.st, inputs)
        # 101·v if each mediator kept its own edit; 200·v on frame aliasing.
        assert torch.allclose(steered - clean, 101.0 * case.vector(), atol=1e-4)

    def test_context_manager_uninstalls_on_exception(self, case: EditCase):
        base = case.logits(_BATCH_A)
        edit = steer(Site("block_output", 1), case.vector(), factor=_FACTOR)
        with pytest.raises(RuntimeError, match="boom"):
            with persistent_edits(case.st, edit) as installed:
                assert installed == (edit,)
                assert not torch.equal(case.logits(_BATCH_A), base)
                raise RuntimeError("boom")
        assert installed_edits(case.st) == ()
        assert torch.equal(case.logits(_BATCH_A), base)

    def test_out_of_band_clear_raises_and_force_recovers(self, case: EditCase):
        edit = steer(Site("block_output", 1), case.vector(), factor=_FACTOR)
        install_edits(case.st, edit)
        case.st.clear_edits()  # out-of-band
        with pytest.raises(PersistentEditError, match="clear_edits"):
            installed_edits(case.st)
        with pytest.raises(PersistentEditError, match="clear_edits"):
            install_edits(case.st, edit)  # drift surfaces on install too
        assert uninstall_edits(case.st, force=True) == (edit,)
        assert install_edits(case.st, edit) == (edit,)  # clean reinstall works
        uninstall_edits(case.st)

    def test_out_of_band_raw_edit_raises(self, case: EditCase):
        edit = steer(Site("block_output", 1), case.vector(), factor=_FACTOR)
        install_edits(case.st, edit)
        with case.st.edit(inplace=True):  # raw nnsight edit, unrecorded
            pass
        with pytest.raises(PersistentEditError, match="raw model.edit"):
            installed_edits(case.st)
        uninstall_edits(case.st, force=True)


class TestComposeWithTracedPlans:
    pytestmark = pytest.mark.property

    def test_collect_at_edit_site_sees_the_edit(self, case: EditCase):
        """A traced collect observes the *edited* model: the persistent steer
        fires before per-trace taps at the same site."""
        site = _resid(1)
        inputs = case.encode(_BATCH_A)
        clean = site.collect(case.st, inputs)
        with persistent_edits(
            case.st, steer(Site("block_output", 1), case.vector(), factor=_FACTOR)
        ):
            steered = site.collect(case.st, inputs)
        assert torch.allclose(steered - clean, _FACTOR * case.vector(), atol=1e-4)

    def test_shallow_collect_under_deeper_edit(self, case: EditCase):
        """Regression: collect_ordered's early stop is suppressed on an edited
        model — stopping at layer 0 under a layer-1 edit strands the edit's
        mediator (measured MissedProviderError on nnsight 0.7)."""
        site = _resid(0)
        inputs = case.encode(_BATCH_A)
        clean = site.collect(case.st, inputs)
        with persistent_edits(
            case.st, steer(Site("block_output", 1), case.vector(), factor=_FACTOR)
        ):
            shallow = site.collect(case.st, inputs)
        # runs (no MissedProviderError) and the edit downstream cannot reach it
        assert torch.equal(shallow, clean)

    def test_run_plan_collect_only_under_deeper_edit(self, case: EditCase):
        """The same early-stop guard on run_plan's single-input lowering."""
        inputs = case.encode(_BATCH_A)
        plan = Plan(inputs={"x": inputs}, ops=(CollectOp("x", _resid(0), key="l0"),))
        clean = run_plan(case.st, plan).collects["l0"]
        with persistent_edits(
            case.st, steer(Site("block_output", 1), case.vector(), factor=_FACTOR)
        ):
            shallow = run_plan(case.st, plan).collects["l0"]
        assert torch.equal(shallow, clean)

    def test_plan_write_lands_after_persistent_edit(self, case: EditCase):
        """Same-site ordering: the persistent edit fires first, so a plan
        write at that site overrides it — a collect declared after the write
        reads the written value, not the steered one."""
        inputs = case.encode(_BATCH_A)
        zeros = torch.zeros(case.hidden)
        plan = Plan(
            inputs={"x": inputs},
            ops=(
                EditOp("x", replace(Site("block_output", 1), zeros)),
                CollectOp("x", _resid(1), key="l1"),
            ),
        )
        with persistent_edits(
            case.st, steer(Site("block_output", 1), case.vector(), factor=_FACTOR)
        ):
            written = run_plan(case.st, plan).collects["l1"]
        # detach: the fresh factory model's params aren't frozen, so the
        # collected activation still carries grad history on CPU.
        assert float(written.detach().abs().max()) == 0.0

    def test_fused_multi_invoke_plan_composes(self, case: EditCase):
        """The canonical cross-invoke interchange runs under a persistent
        edit, and a collect at the edit's site (downstream of the patch, same
        forward) shifts by exactly the steer — plan semantics are relative to
        the edited model."""
        enc = case.encode(_BATCH_A)
        row0 = {k: v[0:1] for k, v in enc.items()}
        row1 = {k: v[1:2] for k, v in enc.items()}
        swap = Edit(
            _resid(0),
            g=lambda f, f_src: f_src,
            read_sources=(ReadSource(_resid(0), positions=[-1], input="source"),),
            positions=[-1],
        )
        plan = Plan(
            inputs={"source": row0, "base": row1},
            ops=(
                EditOp("base", swap),
                CollectOp("base", _resid(1), key="l1", positions=[-1]),
            ),
            save_logits=("base",),
        )
        clean = run_plan(case.st, plan).collects["l1"]
        with persistent_edits(
            case.st, steer(Site("block_output", 1), case.vector(), factor=_FACTOR)
        ):
            steered = run_plan(case.st, plan).collects["l1"]
        assert torch.allclose(steered - clean, _FACTOR * case.vector(), atol=1e-4)

    def test_traced_generation_composes(self, case: EditCase):
        """A traced ``model.generate`` runs under the edit (applied to the
        prefill, persisting through the KV-cached decode)."""
        inputs = case.encode(_BATCH_A)
        with case.st.generate(inputs, max_new_tokens=2):
            base = case.st.generator.output.save()
        with persistent_edits(
            case.st, steer(Site("block_output", 1), case.vector(), factor=_FACTOR)
        ):
            with case.st.generate(inputs, max_new_tokens=2):
                edited = case.st.generator.output.save()
        assert not torch.equal(edited, base)


class TestEarlyStopSuppression:
    """CAP6 × CAP7: every stop-emission site withholds the early stop under
    persistent edits — a stop before a deeper edit's module event strands its
    mediator (the measured ``MissedProviderError``). ``plan._stop_carrier``
    is the single may-I-stop authority (single-input, fused multi-invoke,
    and both staged branches route through it); ``collect_ordered`` mirrors
    the guard inline. Pinned with raw-hook fire counters on the 2-layer tiny
    Llama: an edit at layer 1 below a layer-0 deepest tap turns a stopped
    ``[1, 0]`` forward into a full ``[1, 1]`` one, with the layer-0 collects
    untouched (the edit fires downstream of them)."""

    pytestmark = pytest.mark.property

    _EDIT_LAYER = 1  # below (later than) every tap in these plans

    def _edit(self, case: EditCase) -> Edit:
        return steer(
            Site("block_output", self._EDIT_LAYER), case.vector(), factor=_FACTOR
        )

    def test_fused_multi_invoke_no_logits_under_deeper_edit(self, case: EditCase):
        """The generalized ``_emit_invokes`` stop (#459): a multi-invoke
        plan that saves no logits, run under a deeper persistent edit,
        must not stop — and must not strand the edit's mediator. The
        cross-input read connects the inputs so they schedule as ONE fused
        trace (EU2 #483: grouping is connected-components — a no-edge pair
        would run one trace each); the collects are declared before the
        in-plan write, so they read the clean layer-0 values either way."""
        enc = case.encode(_BATCH_A)
        row0 = {k: v[0:1] for k, v in enc.items()}
        row1 = {k: v[1:2] for k, v in enc.items()}
        plan = Plan(
            inputs={"source": row0, "base": row1},
            ops=(
                CollectOp("source", _resid(0), key="s0"),
                CollectOp("base", _resid(0), key="b0"),
                EditOp(
                    "base",
                    Edit(
                        _resid(0),
                        g=lambda f, f_src: f_src,
                        read_sources=(
                            ReadSource(_resid(0), positions=[-1], input="source"),
                        ),
                        positions=[-1],
                    ),
                ),
            ),
        )
        with layer_fire_counts(case.oracle) as counts:
            clean = run_plan(case.st, plan)
        assert counts == [1, 0], counts  # the CAP6 stop, active without edits
        with persistent_edits(case.st, self._edit(case)):
            with layer_fire_counts(case.oracle) as counts:
                edited = run_plan(case.st, plan)  # no MissedProviderError
        assert counts == [1, 1], counts  # stop suppressed: full forward
        for key in ("s0", "b0"):
            # layer-0 collects sit upstream of the layer-1 edit — unchanged
            assert torch.equal(edited.collects[key], clean.collects[key])

    def test_staged_collect_only_under_deeper_edit(self, case: EditCase):
        """The staged compiler's single-invoke stop (#459): same contract
        through ``lowering="staged"``."""
        plan = Plan(
            inputs={"x": case.encode(_BATCH_A)},
            ops=(CollectOp("x", _resid(0), key="l0"),),
        )
        with layer_fire_counts(case.oracle) as counts:
            clean = run_plan(case.st, plan, lowering="staged")
        assert counts == [1, 0], counts
        with persistent_edits(case.st, self._edit(case)):
            with layer_fire_counts(case.oracle) as counts:
                edited = run_plan(case.st, plan, lowering="staged")
        assert counts == [1, 1], counts
        assert torch.equal(edited.collects["l0"], clean.collects["l0"])

    def test_collect_ordered_under_deeper_edit(self, case: EditCase):
        """The inline ``collect_ordered`` mirror of the guard, pinned with
        the same fire counters (value equality is pinned in
        ``TestComposeWithTracedPlans::test_shallow_collect_under_deeper_edit``)."""
        site = _resid(0)
        inputs = case.encode(_BATCH_A)
        with layer_fire_counts(case.oracle) as counts:
            clean = site.collect(case.st, inputs)
        assert counts == [1, 0], counts
        with persistent_edits(case.st, self._edit(case)):
            with layer_fire_counts(case.oracle) as counts:
                edited = site.collect(case.st, inputs)
        assert counts == [1, 1], counts
        assert torch.equal(edited, clean)


class TestRawHFGenerateRefusal:
    pytestmark = pytest.mark.property

    def test_pipeline_generate_refuses_under_persistent_edits(self):
        """LMPipeline.generate runs plain HF generation, which bypasses
        nnsight edits — the compose-or-refuse contract says refuse loudly
        rather than silently generate unsteered outputs."""
        from causalab.neural.pipeline import LMPipeline

        raw, _ = fresh_tiny_random_llama()
        pipeline = LMPipeline(raw, max_new_tokens=1, padding_side="left", device="cpu")
        prompts = [{"raw_input": "the quick brown fox"}]
        pipeline.generate(prompts)  # sanity: fine without edits

        hidden = int(pipeline.model.config.hidden_size)
        edit = steer(Site("block_output", 1), torch.ones(hidden), factor=_FACTOR)
        with persistent_edits(pipeline.model, edit):
            with pytest.raises(PersistentEditError, match="bypasses"):
                pipeline.generate(prompts)
        pipeline.generate(prompts)  # restored after uninstall
