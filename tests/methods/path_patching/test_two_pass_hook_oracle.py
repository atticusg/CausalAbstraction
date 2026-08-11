"""Hook oracle for two-pass path patching + the forward-order contract (GH #380).

An internal-receiver path patch needs two passes:

* **PASS 1** runs one forward with the sender reading its source (interchange)
  and *collects* the receiver's activation ``v*`` — the value it reads when the
  direct sender→receiver path is perturbed. This is only correct if hooks fire in
  forward order: the sender (upstream) must be overwritten *before* the receiver
  (downstream) is read.
* **PASS 2** re-runs the clean base with the receiver replaced by ``v*``, isolating
  the sender→receiver edge effect.

This file pins both halves with hand-rolled hooks (no backbone imports):

* ``TestForwardOrderContract`` proves the firing-order assumption directly: an
  upstream edit is visible to a downstream capture in one forward, and a
  downstream edit is *not* visible to an upstream capture.
* ``TestTwoPassHookOracle`` reproduces the Plan-lowered two-pass edge (single
  internal receiver, no restorers — via the ``build_edge_plan`` seam) by
  collecting ``v*`` under a sender edit and injecting it on the clean base.
* ``TestRestorerFreezeOracle`` adds the path-patching-specific semantic: a
  restorer frozen to its **clean** value between sender and receiver, against
  the public runner.

See ``docs/PYVENE_HOOK_COVERAGE.md``.
"""

from __future__ import annotations

import pytest
import torch

from causalab.methods.path_patching.outputs import plan_outputs
from causalab.methods.path_patching.plans import build_edge_plan
from causalab.methods.path_patching.run import run_path_patching
from causalab.methods.path_patching.targets import ReceiverSpec, build_receiver_site
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec
from causalab.neural.plan import run_plan
from causalab.neural.token_positions import TokenPosition

from tests.neural.activations.hook_oracle import (
    _attn_module,
    capture_component,
    capture_residual,
    capture_with_edits,
    cf_example,
    decoder_block,
    make_trace,
    next_token_logits,
)
from causalab.neural.token_positions import get_last_token_index
from tests._helpers.tiny import TINY_RANDOM_GPT2_MODEL_NAME, TINY_RANDOM_MODEL_NAME


@pytest.fixture(scope="module")
def _llama_pipe() -> LMPipeline:
    # Fresh name-based load (NOT the session-cached tiny_random_model() object):
    # pyvene tests elsewhere run on that shared object and pyvene's cleanup
    # clears ALL forward hooks, which would kill this module's nnterp wrapper.
    return LMPipeline(TINY_RANDOM_MODEL_NAME, max_new_tokens=1, padding_side="left")


@pytest.fixture(scope="module")
def _gpt2_pipe() -> LMPipeline:
    return LMPipeline(
        TINY_RANDOM_GPT2_MODEL_NAME, max_new_tokens=1, padding_side="left"
    )


@pytest.fixture(params=["llama", "gpt2"])
def oracle_pipeline(request, _llama_pipe, _gpt2_pipe) -> LMPipeline:
    """Tiny-random Llama and GPT-2 pipelines (``max_new_tokens=1``, own fresh
    model objects — see ``_llama_pipe``); the hook-oracle resolvers dispatch on
    the module tree, so the two-pass logic is asserted on both families."""
    return _llama_pipe if request.param == "llama" else _gpt2_pipe


_SENDER_LAYER = 0
_RECEIVER_LAYER = 1
_POS = 1
_BASE = "the quick brown fox jumps"
_SOURCE = "a slow lazy old dog sits"


def _resid_unit(pipeline: LMPipeline, layer: int) -> SiteSpec:
    # Sender and receiver sit on the *last* token so the edge reaches the
    # next-token logits: on the 2-layer Llama the receiver is the final layer
    # (the last token's residual feeds the unembed directly); on the 5-layer GPT-2
    # the receiver's last-token residual still reaches the logits through the
    # later layers' self-attention at the last position.
    tp = TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline), pipeline, id="last"
    )
    return SiteSpec(
        fsite=FeaturizedSite(Site("block_output", layer)),
        positions=tp,
        key=f"residual.L{layer}.last",
        width=pipeline.model.config.hidden_size,
    )


class TestForwardOrderContract:
    """PyTorch fires module hooks in forward (execution) order — the property
    PASS 1 of two-pass path patching and causal tracing both rely on."""

    pytestmark = pytest.mark.property

    def test_upstream_edit_visible_downstream(
        self, oracle_pipeline: LMPipeline
    ) -> None:
        """An edit at layer 0's output changes what a capture at layer 1's output
        sees, within a single forward."""
        base = oracle_pipeline.load([make_trace(_BASE)])
        clean = capture_residual(oracle_pipeline, _RECEIVER_LAYER, base)

        def bump(h: torch.Tensor) -> None:
            h[:, _POS, :] += 5.0

        edited = capture_with_edits(
            oracle_pipeline,
            base,
            decoder_block(oracle_pipeline, _RECEIVER_LAYER),
            "out",
            [(decoder_block(oracle_pipeline, _SENDER_LAYER), "out", bump)],
        )
        assert not torch.allclose(edited, clean, atol=1e-4)

    def test_downstream_edit_not_visible_upstream(
        self, oracle_pipeline: LMPipeline
    ) -> None:
        """An edit at layer 1's output cannot change a capture at layer 0's output
        — the upstream activation is already computed when the later hook fires."""
        base = oracle_pipeline.load([make_trace(_BASE)])
        clean = capture_residual(oracle_pipeline, _SENDER_LAYER, base)

        def bump(h: torch.Tensor) -> None:
            h[:, _POS, :] += 5.0

        captured = capture_with_edits(
            oracle_pipeline,
            base,
            decoder_block(oracle_pipeline, _SENDER_LAYER),
            "out",
            [(decoder_block(oracle_pipeline, _RECEIVER_LAYER), "out", bump)],
        )
        torch.testing.assert_close(captured, clean, atol=1e-5, rtol=1e-4)


class TestTwoPassHookOracle:
    """The Plan-lowered two-pass edge (one internal receiver, no restorers —
    built through the ``build_edge_plan`` seam with ``restorer_sites=[]``) vs. a
    hand-rolled collect-then-inject ground truth."""

    pytestmark = pytest.mark.property

    def test_two_pass_matches_collect_inject(self, oracle_pipeline: LMPipeline) -> None:
        sender = _resid_unit(oracle_pipeline, _SENDER_LAYER)

        # --- oracle ----------------------------------------------------- #
        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        source_inputs = oracle_pipeline.load([make_trace(_SOURCE)])
        # The source's sender activation at its last token (full interchange).
        src_sender = capture_residual(oracle_pipeline, _SENDER_LAYER, source_inputs)[
            :, -1, :
        ]

        def write_sender(h: torch.Tensor) -> None:
            h[:, -1, :] = src_sender

        # PASS 1: collect the receiver value while the sender reads the source —
        # forward order makes the receiver see the perturbed sender.
        v_star = capture_with_edits(
            oracle_pipeline,
            base_inputs,
            decoder_block(oracle_pipeline, _RECEIVER_LAYER),
            "out",
            [(decoder_block(oracle_pipeline, _SENDER_LAYER), "out", write_sender)],
        )[:, [-1], :]
        # PASS 2: inject v* into the receiver on an otherwise-clean base run.
        manual = next_token_logits(
            oracle_pipeline, base_inputs, _RECEIVER_LAYER, [-1], v_star
        )
        clean = next_token_logits(oracle_pipeline, base_inputs)

        # --- causalab --------------------------------------------------- #
        spec = ReceiverSpec(
            kind="residual",
            layer=_RECEIVER_LAYER,
            token_position=TokenPosition(
                lambda inp: get_last_token_index(inp, oracle_pipeline),
                oracle_pipeline,
                id="last",
            ),
        )
        receiver_site = build_receiver_site(oracle_pipeline, spec)
        plan, key = build_edge_plan(
            oracle_pipeline,
            [cf_example(_BASE, _SOURCE)],
            sender,
            [receiver_site],
            spec,
            restorer_sites=[],
        )
        with torch.no_grad():
            plan_result = run_plan(oracle_pipeline.model, plan)
        causalab = plan_outputs(oracle_pipeline, plan_result.logits[key]).scores[0]

        # Non-trivial: the sender→receiver edge actually moves the output.
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)


class TestRestorerFreezeOracle:
    """One restorer, hand-rolled: PASS 1 must read the receiver with the
    restorer frozen to its CLEAN base value (not its recomputed value under the
    sender patch) — the freeze that makes the measured effect the *direct* edge.
    Sender = residual after layer 0, receiver = residual after layer 1,
    restorer = layer-1 attention output (``restore=("attention",)`` builds
    exactly that set on both families)."""

    pytestmark = pytest.mark.property

    def test_freeze_matches_hand_rolled_and_matters(
        self, oracle_pipeline: LMPipeline
    ) -> None:
        pipe = oracle_pipeline
        sender = _resid_unit(pipe, _SENDER_LAYER)
        base_inputs = pipe.load([make_trace(_BASE)])
        source_inputs = pipe.load([make_trace(_SOURCE)])
        src_sender = capture_residual(pipe, _SENDER_LAYER, source_inputs)[:, -1, :]
        clean_attn = capture_component(
            pipe, _attn_module(pipe, _RECEIVER_LAYER), "out", base_inputs
        )[:, -1, :]

        def write_sender(h: torch.Tensor) -> None:
            h[:, -1, :] = src_sender

        def freeze_attn(h: torch.Tensor) -> None:
            h[:, -1, :] = clean_attn

        # PASS 1 with the freeze vs without: the freeze must change v*.
        v_star = capture_with_edits(
            pipe,
            base_inputs,
            decoder_block(pipe, _RECEIVER_LAYER),
            "out",
            [
                (decoder_block(pipe, _SENDER_LAYER), "out", write_sender),
                (_attn_module(pipe, _RECEIVER_LAYER), "out", freeze_attn),
            ],
        )[:, [-1], :]
        v_star_unfrozen = capture_with_edits(
            pipe,
            base_inputs,
            decoder_block(pipe, _RECEIVER_LAYER),
            "out",
            [(decoder_block(pipe, _SENDER_LAYER), "out", write_sender)],
        )[:, [-1], :]
        assert not torch.allclose(v_star, v_star_unfrozen, atol=1e-5)

        manual = next_token_logits(pipe, base_inputs, _RECEIVER_LAYER, [-1], v_star)
        clean = next_token_logits(pipe, base_inputs)
        assert not torch.allclose(manual, clean, atol=1e-4)  # non-vacuous edge

        spec = ReceiverSpec(
            kind="residual",
            layer=_RECEIVER_LAYER,
            token_position=TokenPosition(
                lambda inp: get_last_token_index(inp, pipe), pipe, id="last"
            ),
        )
        result = run_path_patching(
            pipe,
            [cf_example(_BASE, _SOURCE)],
            sender,
            spec,
            restore=("attention",),
            batch_size=1,
        )
        causalab = result.scores[0]  # first generated step, (1, vocab)
        torch.testing.assert_close(causalab, manual, atol=1e-4, rtol=1e-3)
