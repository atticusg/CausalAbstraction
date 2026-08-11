"""SH1 (#410): per-mode parity on the real coherent backbone (GPU, nightly).

The new-stack counterpart of the pyvene-era
``tests/neural/activations/test_chat_coherent_hook_oracle.py``: every mode
constructor runs through Site/Edit on ``chat-coherent``
(``Qwen/Qwen3-4B-Instruct-2507`` — GQA 32/8 heads, qk-norm, **decoupled
``head_dim`` = 128 ≠ hidden/n_head = 80``) against the same raw-hook oracle,
in eager attention, float32, one model load for the whole module.

Six modes are asserted against the live oracle at empirically-set tolerances
(see the measurement note above ``_ATOL``). ``noise`` cannot byte-match a
CPU-drawn oracle — the in-trace draw uses a **CUDA** generator — so its
contract here is seeded reproducibility, the same property split the coverage
doc pins for the pyvene backbone. The per-head value interchange exercises the decoupled
``head_dim`` slice on the production model — the exact contract pyvene breaks
(#386), honored by the new stack's ``HeadView``.

#451 restores the behavior families the SH2 cutover descoped to tiny-random
CPU models, on the same single model load (the module fixture is an
``LMPipeline`` so the public wrappers run too; the ``ParityCase`` view taps
its nnterp/raw halves):

* **cross-model patching** — ``run_interchange_interventions(source_pipeline=…)``
  writes a *source-model* activation into the Qwen base run
  (``TestChatCoherentCrossModel``, oracle pattern from
  ``tests/neural/activations/test_cross_model_hook_oracle.py``);
* **two-pass path patching + the forward-order contract**
  (``TestChatCoherentTwoPass``, from
  ``tests/methods/path_patching/test_two_pass_hook_oracle.py``);
* **mixed causal-tracing** — seeded noise corrupt + replace restore in one
  pass (``TestChatCoherentCausalTracing``, from
  ``tests/methods/test_causal_tracing_hook_oracle.py``);
* **wrapper generate/top-k contract** — batch-nested sequences/scores and
  the ``output_scores=int`` top-k conversion on the engine's generate path
  (``TestChatCoherentGenerateContract``).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from causalab.methods.path_patching.outputs import plan_outputs
from causalab.methods.path_patching.plans import build_edge_plan
from causalab.methods.path_patching.targets import ReceiverSpec, build_receiver_site
from causalab.methods.steer.steer import run_steering_interventions
from causalab.neural.activations.interchange_mode import run_interchange_interventions
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.plan import run_plan
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec
from causalab.neural.token_positions import TokenPosition, get_last_token_index

from tests.neural.activations.hook_oracle import (
    capture_residual,
    capture_with_edits,
    cf_example,
    decoder_block,
    make_trace,
    next_token_logits,
)
from tests.neural.parity.cases import (
    MODES,
    ModeCase,
    ParityCase,
    realize_new_stack,
    realize_oracle,
)

pytestmark = pytest.mark.golden

# Measured on H100 (job 1001370): the stack and the oracle are byte-identical
# per mode (max |got - want| = 0.0) — same float32 eager kernels in the same
# order. The gate leaves epsilon headroom without ever nearing the smallest
# effect size (interpolate, ~9e-3).
_ATOL, _RTOL = 1e-4, 1e-5
_MIN_EFFECT = 1e-3  # non-vacuity: the intervention must move some logit this much
_MODES_VS_ORACLE = tuple(m for m in MODES if m != "noise")

# Wrapper-level equivalence gates reuse the CPU oracle homes' tolerances: the
# wrapper side runs a generate/Plan trace, the oracle a plain forward — same
# float32 eager kernels, but not necessarily the same op order end to end.
_W_ATOL, _W_RTOL = 1e-4, 1e-3

_BASE = "the quick brown fox jumps"
_SOURCE = "a slow lazy old dog sits"


@pytest.fixture(scope="module")
def qwen_pipeline() -> LMPipeline:
    """One coherent-model load for the whole module, as the public-wrapper
    surface. ``eager_attn=True`` + float32 keep the load equivalent to the
    direct ``StandardizedTransformer(..., attn_implementation="eager",
    dtype=torch.float32, device_map="auto")`` this fixture replaced — the
    parity policy (pins must be eager, see ``cases.py``); ``max_new_tokens=1``
    is the single-step scoring contract every ported oracle relies on."""
    pipe = LMPipeline(
        "Qwen/Qwen3-4B-Instruct-2507",
        max_new_tokens=1,
        padding_side="left",
        device_map="auto",
        dtype=torch.float32,
        eager_attn=True,
    )
    assert pipe.hf_model.config._attn_implementation == "eager"
    return pipe


@pytest.fixture(scope="module")
def qwen_case(qwen_pipeline: LMPipeline) -> ParityCase:
    """The same load as a ``ParityCase``: the nnterp half for the new stack,
    the raw HF half for the hook oracle."""
    return ParityCase(
        family="chat-coherent",
        st=qwen_pipeline.model,
        oracle=SimpleNamespace(hf_model=qwen_pipeline.hf_model),
        tok=qwen_pipeline.tokenizer,
    )


def _mid(pc: ParityCase) -> int:
    return int(pc.st.num_layers) // 2


class TestChatCoherentParity:
    @pytest.mark.parametrize("mode", _MODES_VS_ORACLE)
    def test_mode_matches_oracle(self, mode: str, qwen_case: ParityCase) -> None:
        mid = _mid(qwen_case)
        mc = ModeCase(family="chat-coherent", mode=mode, layer=mid, src_layer=mid - 1)
        want = realize_oracle(mc, qwen_case)
        got = realize_new_stack(mc, qwen_case)
        assert got.kind == want.kind
        if want.clean is not None:
            effect = (want.value.float() - want.clean.float()).abs().max().item()
            assert effect > _MIN_EFFECT, f"{mc.case_id}: inert intervention ({effect=})"
        torch.testing.assert_close(
            got.value.float().cpu(),
            want.value.float().cpu(),
            atol=_ATOL,
            rtol=_RTOL,
        )

    def test_head_value_interchange_honors_decoupled_head_dim(
        self, qwen_case: ParityCase
    ) -> None:
        """The #386 contract on the production model: per-head value slices are
        true-``head_dim`` (128) wide, KV-head addressed — pyvene mis-slices
        this shape; the new stack must not."""
        mc = ModeCase(
            family="chat-coherent",
            mode="interchange",
            path="head",
            head_kind="value",
            layer=_mid(qwen_case),
            featurizer="identity",
        )
        want = realize_oracle(mc, qwen_case)
        got = realize_new_stack(mc, qwen_case)
        effect = (want.value.float() - want.clean.float()).abs().max().item()
        assert effect > _MIN_EFFECT, f"inert head transplant ({effect=})"
        torch.testing.assert_close(
            got.value.float().cpu(),
            want.value.float().cpu(),
            atol=_ATOL,
            rtol=_RTOL,
        )

    def test_noise_is_seeded_on_cuda(self, qwen_case: ParityCase) -> None:
        """Fresh same-seed streams reproduce byte-identically on a CUDA
        generator; the corruption is non-trivial."""
        mid = _mid(qwen_case)
        mc = ModeCase(family="chat-coherent", mode="noise", layer=mid)
        first = realize_new_stack(mc, qwen_case)
        again = realize_new_stack(mc, qwen_case)
        effect = (first.value.float() - first.clean.float()).abs().max().item()
        assert effect > _MIN_EFFECT, f"inert noise corruption ({effect=})"
        torch.testing.assert_close(again.value, first.value, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------- #
#  #451: behaviors descoped to tiny-random CPU at the SH2 cutover, restored    #
#  on the coherent backbone. One case per behavior family; every helper is     #
#  the CPU oracle home's pattern ported onto the module's single model load.   #
# --------------------------------------------------------------------------- #
def _last_site(pipeline: LMPipeline, layer: int) -> SiteSpec:
    """Last-token residual :class:`SiteSpec` after ``layer``."""
    tp = TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline), pipeline, id="last"
    )
    return SiteSpec(
        fsite=FeaturizedSite(Site("block_output", layer)),
        positions=tp,
        key=f"residual.L{layer}.last",
        width=pipeline.model.config.hidden_size,
    )


class TestChatCoherentCrossModel:
    """Cross-model patching (``source_pipeline=``, the SH2 engine path) on the
    production target module tree — the CPU home is
    ``test_cross_model_hook_oracle.py``."""

    _LAYER = 1  # must exist in BOTH models: the 2-layer source and the target
    _POS = 1

    @pytest.fixture(scope="class")
    def source_pipeline(self, qwen_pipeline: LMPipeline) -> LMPipeline:
        """A genuinely different source model sharing the target's tokenizer
        and hidden width: a fresh seeded random-init module of the target's
        architecture, cut to 2 layers (~0.6B params — the residual write only
        needs matching ``hidden_size``). Its activations share nothing with
        the target's, so the target output following the injected value
        proves the value came from the *source* model."""
        cfg = copy.deepcopy(qwen_pipeline.hf_model.config)
        cfg.num_hidden_layers = 2
        torch.manual_seed(0)
        src_model = type(qwen_pipeline.hf_model)(cfg)
        src_model.eval()  # constructed models default to train mode
        return LMPipeline(
            src_model,
            max_new_tokens=1,
            padding_side="left",
            device="cuda",
            dtype=torch.float32,
        )

    def test_injects_source_model_activation(
        self, qwen_pipeline: LMPipeline, source_pipeline: LMPipeline
    ) -> None:
        """The value written into the Qwen base run is the *source model's*
        residual at the counterfactual, captured by hand with a raw hook on
        the source and patched into the target by hand."""
        tp = TokenPosition(lambda _x: [self._POS], qwen_pipeline, id="pos1")
        site = SiteSpec(
            fsite=FeaturizedSite(Site("block_output", self._LAYER)),
            positions=tp,
            key=f"residual.L{self._LAYER}.pos1",
            width=qwen_pipeline.model.config.hidden_size,
        )
        groups = [[site]]

        # Source model's residual at the counterfactual, captured by hand.
        cf_inputs = source_pipeline.load([make_trace(_SOURCE)])
        src_resid = capture_residual(source_pipeline, self._LAYER, cf_inputs)[
            :, [self._POS], :
        ]
        # Provenance: the source model's activation is nothing like the
        # target's own at the same site (random-init vs trained weights).
        tgt_resid = capture_residual(qwen_pipeline, self._LAYER, cf_inputs)[
            :, [self._POS], :
        ]
        assert not torch.allclose(src_resid, tgt_resid, atol=1e-2)

        base_inputs = qwen_pipeline.load([make_trace(_BASE)])
        clean = next_token_logits(qwen_pipeline, base_inputs)
        manual = next_token_logits(
            qwen_pipeline, base_inputs, self._LAYER, [self._POS], src_resid
        )
        effect = (manual - clean).abs().max().item()
        assert effect > _MIN_EFFECT, f"inert cross-model patch ({effect=})"

        result = run_interchange_interventions(
            qwen_pipeline,
            [cf_example(_BASE, _SOURCE)],
            groups,
            source_pipeline=source_pipeline,
            output_scores=True,
        )
        causalab = result.scores[0]  # first generated step, (1, vocab)
        torch.testing.assert_close(causalab, manual, atol=_W_ATOL, rtol=_W_RTOL)


class TestChatCoherentTwoPass:
    """Two-pass path patching (collect-under-interchange → inject) and the
    forward-order contract it rests on, on the production module tree — the
    CPU home is ``tests/methods/path_patching/test_two_pass_hook_oracle.py``."""

    _SENDER_LAYER = 18
    _RECEIVER_LAYER = 20  # internal: PASS 2's injection propagates 15 layers

    def test_forward_order_contract(self, qwen_pipeline: LMPipeline) -> None:
        """Hooks fire in forward order on the real CUDA module tree: an
        upstream edit is visible to a downstream capture within one forward,
        and a downstream edit is invisible upstream."""
        base = qwen_pipeline.load([make_trace(_BASE)])
        clean_dn = capture_residual(qwen_pipeline, self._RECEIVER_LAYER, base)
        clean_up = capture_residual(qwen_pipeline, self._SENDER_LAYER, base)

        def bump(h: torch.Tensor) -> None:
            h[:, -1, :] += 5.0

        seen_dn = capture_with_edits(
            qwen_pipeline,
            base,
            decoder_block(qwen_pipeline, self._RECEIVER_LAYER),
            "out",
            [(decoder_block(qwen_pipeline, self._SENDER_LAYER), "out", bump)],
        )
        assert not torch.allclose(seen_dn, clean_dn, atol=1e-4)

        seen_up = capture_with_edits(
            qwen_pipeline,
            base,
            decoder_block(qwen_pipeline, self._SENDER_LAYER),
            "out",
            [(decoder_block(qwen_pipeline, self._RECEIVER_LAYER), "out", bump)],
        )
        torch.testing.assert_close(seen_up, clean_up, atol=1e-5, rtol=1e-4)

    def test_two_pass_matches_collect_inject(self, qwen_pipeline: LMPipeline) -> None:
        """The Plan-lowered edge (``build_edge_plan`` with ``restorer_sites=[]``)
        vs a hand-rolled PASS 1 (collect the receiver under the sender
        interchange) + PASS 2 (inject ``v*`` on the clean base)."""
        pipe = qwen_pipeline
        sender = _last_site(pipe, self._SENDER_LAYER)

        # --- oracle ----------------------------------------------------- #
        base_inputs = pipe.load([make_trace(_BASE)])
        source_inputs = pipe.load([make_trace(_SOURCE)])
        src_sender = capture_residual(pipe, self._SENDER_LAYER, source_inputs)[:, -1, :]

        def write_sender(h: torch.Tensor) -> None:
            h[:, -1, :] = src_sender

        v_star = capture_with_edits(
            pipe,
            base_inputs,
            decoder_block(pipe, self._RECEIVER_LAYER),
            "out",
            [(decoder_block(pipe, self._SENDER_LAYER), "out", write_sender)],
        )[:, [-1], :]
        manual = next_token_logits(
            pipe, base_inputs, self._RECEIVER_LAYER, [-1], v_star
        )
        clean = next_token_logits(pipe, base_inputs)
        assert not torch.allclose(manual, clean, atol=1e-4)  # non-vacuous edge

        # --- causalab --------------------------------------------------- #
        spec = ReceiverSpec(
            kind="residual",
            layer=self._RECEIVER_LAYER,
            token_position=TokenPosition(
                lambda inp: get_last_token_index(inp, pipe), pipe, id="last"
            ),
        )
        receiver_site = build_receiver_site(pipe, spec)
        plan, key = build_edge_plan(
            pipe,
            [cf_example(_BASE, _SOURCE)],
            sender,
            [receiver_site],
            spec,
            restorer_sites=[],
        )
        with torch.no_grad():
            plan_result = run_plan(pipe.model, plan)
        causalab = plan_outputs(pipe, plan_result.logits[key]).scores[0]
        torch.testing.assert_close(causalab, manual, atol=_W_ATOL, rtol=_W_RTOL)


class TestChatCoherentCausalTracing:
    """The mixed noise+replace model (ROME-style causal tracing) in ONE pass
    on the coherent backbone — the CPU home is
    ``tests/methods/test_causal_tracing_hook_oracle.py``. The noise draw is a
    backbone internal, so the pin is the RNG-independent recovery property."""

    _ENTRY_LAYER = 0
    _NOISE_SCALE = 5.0

    def test_seeded_corrupt_and_restore_recovers(
        self, qwen_pipeline: LMPipeline
    ) -> None:
        pipe = qwen_pipeline
        hidden = pipe.model.config.hidden_size
        restore_layer = pipe.model.config.num_hidden_layers - 1
        entry = _last_site(pipe, self._ENTRY_LAYER)
        restore = _last_site(pipe, restore_layer)

        base_inputs = pipe.load([make_trace(_BASE)])
        clean = next_token_logits(pipe, base_inputs)
        clean_restore = capture_residual(pipe, restore_layer, base_inputs)[
            :, -1, :
        ].squeeze(0)

        def run_mixed(sites: list[SiteSpec], vectors: dict, types: dict):
            result = run_steering_interventions(
                pipe,
                [{"input": make_trace(_BASE), "counterfactual_inputs": []}],
                sites,
                vectors,
                mode="replace",
                type_by_key=types,
                noise_seed=0,
                output_scores=True,
            )
            return result.scores[0]  # first generated step, (1, vocab)

        # Corruption alone is non-trivial: the seeded noise actually fires.
        corrupted = run_mixed(
            [entry],
            {entry.key: torch.full((hidden,), self._NOISE_SCALE)},
            {entry.key: "noise"},
        )
        assert not torch.allclose(corrupted, clean, atol=1e-4)

        # Corruption + restoring the mediating site (final-layer last-token
        # residual) to its clean value recovers the clean output: both
        # interventions run in one forward, and ``replace`` overwrites
        # whatever the corruption produced (forward order).
        recovered = run_mixed(
            [entry, restore],
            {
                entry.key: torch.full((hidden,), self._NOISE_SCALE),  # noise scale
                restore.key: clean_restore,  # clean value to restore
            },
            {entry.key: "noise", restore.key: "replace"},
        )
        torch.testing.assert_close(recovered, clean, atol=_W_ATOL, rtol=_W_RTOL)


class TestChatCoherentGenerateContract:
    """Wrapper-level engine contracts of ``run_intervened_generation`` (via
    ``run_interchange_interventions``) on the coherent backbone: the flat
    ``GenerationResult`` surface (EU5b #487 — the wrapper's legacy
    ``to_raw_results()`` tail is gone; every pinned VALUE below is identical
    to the pre-EU5b run, only the access shape changed), the greedy argmax
    contract, and the ``output_scores=int`` top-k compression
    (``compress_scores_top_k``) — previously golden only transitively
    through runner pins."""

    _LAYER = 18
    _K = 5

    def test_generate_topk_scores_contract(self, qwen_pipeline: LMPipeline) -> None:
        pipe = qwen_pipeline
        vocab = pipe.model.config.vocab_size
        groups = [[_last_site(pipe, self._LAYER)]]
        dataset = [cf_example(_BASE, _SOURCE), cf_example(_SOURCE, _BASE)]

        full = run_interchange_interventions(
            pipe, dataset, groups, batch_size=2, output_scores=True
        )
        # Flat result: sequences are the generated tokens only
        # (n_examples, max_new_tokens); one score step per generated token.
        sequences = full.sequences
        assert sequences.shape == (2, 1)
        assert full.scores is not None
        assert len(full.scores) == 1  # max_new_tokens=1 → one scored step
        scores = full.scores[0]
        assert scores.shape == (2, vocab)
        # Greedy single-step contract: the generated token IS the argmax.
        assert torch.equal(sequences.squeeze(1).cpu(), scores.argmax(dim=-1).cpu())
        # And ``strings`` decodes those tokens.
        assert full.strings == pipe.dump(sequences, is_logits=False)

        topk = run_interchange_interventions(
            pipe, dataset, groups, batch_size=2, output_scores=self._K
        )
        assert topk.scores is None
        assert topk.scores_top_k is not None
        entry = topk.scores_top_k[0]
        want_vals, want_idx = torch.topk(scores, k=self._K, dim=1)
        assert torch.equal(entry["top_k_indices"].cpu(), want_idx.cpu())
        torch.testing.assert_close(
            entry["top_k_logits"].cpu(), want_vals.cpu(), atol=_W_ATOL, rtol=_W_RTOL
        )
        assert entry["top_k_tokens"] == [
            [pipe.tokenizer.decode([idx]) for idx in row] for row in want_idx.tolist()
        ]
        assert topk.strings == full.strings
