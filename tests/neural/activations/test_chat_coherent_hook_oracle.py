"""Comprehensive golden-tier hook oracle on the real coherent backbone (GH #380).

The CPU oracle suites run the full intervention matrix on tiny-random Llama /
GPT-2 / GQA stubs. This file re-runs **every behavior** on the production
``chat-coherent`` model (``Qwen/Qwen3-4B-Instruct-2507``) so the contracts are
verified against the real Qwen3 module tree — **grouped-query attention (32 heads
/ 8 KV heads) with a *decoupled* ``head_dim`` (128 ≠ 2560/32 = 80)** — and the
real chat-template tokenization path, end to end. The hook oracle consumes the
same chat-template-wrapped ``input_ids`` the wrapper does, so the template is
handled uniformly; what this adds over the CPU suites is the real architecture +
real tokenizer at scale.

This is intentionally a *faithful* port of the CPU oracles (same logic, GPU
float32 tolerances) — it is meant to **surface** any place where causalab's
addressing or plumbing breaks on the decoupled-``head_dim`` GQA backbone, not to
paper over it. The per-head value/query receivers are the sharpest case: their
width must follow ``config.head_dim`` (128 here) rather than
``hidden // n_head`` (80), and k/v must be addressed in KV-head space.

GPU-only (``@pytest.mark.golden``, the sole GPU tier — see ``docs/TESTS.md``);
deselected on the CPU ``test`` job, run on the nightly GPU runner. Loaded in
float32 for a clean equivalence signal (these are equivalence contracts, not the
bf16 value pins ``test_goldens.py`` owns). See ``docs/HOOK_ORACLES.md``.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.LM_units import (
    MLP,
    AttentionHead,
    AttentionHeadQuery,
    AttentionHeadValue,
    AttentionOutput,
    ResidualStream,
)
from causalab.neural.featurizer import Featurizer
from causalab.neural.interventions import build_intervention
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition, get_last_token_index
from causalab.neural.units import InterchangeTarget
from causalab.neural.activations.collect import collect_features
from causalab.neural.activations.interchange_mode import (
    batched_interchange_intervention,
    run_interchange_interventions,
    run_two_pass_path_patching,
)
from causalab.neural.activations.interpolate import batched_interpolation_intervention
from causalab.methods.steer.steer import make_zero_features, run_steering_interventions

from tests.neural.activations.hook_oracle import (
    capture_component,
    capture_head_value,
    capture_residual,
    capture_with_edits,
    cf_example,
    component_edited_logits,
    component_module,
    decoder_block,
    head_dim,
    make_trace,
    next_token_logits,
    random_rotation,
    rotate_featurizer,
)

pytestmark = pytest.mark.golden

# GPU float32 tolerances: two independent forward passes (causalab's vs the hook
# oracle's) differ by kernel non-determinism, so logit comparisons are looser than
# the CPU suites' 1e-4.
_ATOL_LOGIT, _RTOL_LOGIT = 2e-2, 1e-2
_ATOL_ACT, _RTOL_ACT = 2e-3, 1e-3
_MID = 4  # a mid layer: interventions here still reach the last-token logits
_BASE = "Q: What color is a clear daytime sky? A:"
_SOURCE = "Q: What color is fresh grass? A:"

# Real finding surfaced by this harness (GH #380): pyvene 0.1.8's per-head
# accessors (head_value_output / head_query_output / head_attention_value_output)
# assumed head_dim == hidden / n_head. Qwen3-4B *decouples* head_dim (128) from
# hidden / n_head (2560/32 = 80), so the old backbone returned 80-wide slices
# instead of the true 128 — a genuinely wrong activation (#386). causalab refused per-head ops on
# such models rather than report one, and the tests below were xfail(strict=True)
# pending the backbone migration.
#
# The nnsight engine reads per-head width from `config.head_dim`, so they pass:
# these are now live assertions that a decoupled-head_dim model is addressed
# correctly, on a real model rather than a synthetic config.


@pytest.fixture(scope="module")
def chat_coherent_pipeline() -> LMPipeline:
    """The coherent golden backbone (``Qwen/Qwen3-4B-Instruct-2507``), chat
    template on, float32 on GPU (``device='auto'``). Loaded once per module."""
    return LMPipeline(
        "Qwen/Qwen3-4B-Instruct-2507",
        max_new_tokens=1,
        use_chat_template=True,
        chat_answer_directive=(
            "Reply with only the final answer word and nothing else. "
            "Do not restate the question."
        ),
        device="auto",
        dtype="float32",
    )


# --------------------------------------------------------------------------- #
#  Local builders / runners (faithful ports of the CPU oracle helpers)         #
# --------------------------------------------------------------------------- #
def _last_tp(pipe: LMPipeline) -> TokenPosition:
    return TokenPosition(lambda inp: get_last_token_index(inp, pipe), pipe, id="last")


def _resid_target(pipe: LMPipeline, layer: int, featurizer=None) -> InterchangeTarget:
    unit = ResidualStream(
        layer=layer,
        token_indices=_last_tp(pipe),
        target_output=True,
        shape=(pipe.model.config.hidden_size,),
        featurizer=featurizer,
    )
    return InterchangeTarget([[unit]])


def _interchange_logits(pipe: LMPipeline, target: InterchangeTarget) -> torch.Tensor:
    out = batched_interchange_intervention(
        pipe, [cf_example(_BASE, _SOURCE)], target, output_scores=True
    )
    return out["scores"][0].float()


def _steer_logits(
    pipe: LMPipeline, target: InterchangeTarget, vec: torch.Tensor, *, mode: str
) -> torch.Tensor:
    uid = target.flatten()[0].id
    result = run_steering_interventions(
        pipe,
        [{"input": make_trace(_BASE), "counterfactual_inputs": []}],
        target,
        {uid: vec},
        mode=mode,
        output_scores=True,
    )
    return result["scores"][0][0].float()


def _last_logits(pipe: LMPipeline, text: str) -> torch.Tensor:
    return next_token_logits(pipe, pipe.load([make_trace(text)])).float()


# --------------------------------------------------------------------------- #
#  Collect                                                                     #
# --------------------------------------------------------------------------- #
class TestCollectGolden:
    def test_collect_residual_matches_hook(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        unit = ResidualStream(
            layer=_MID,
            token_indices=_last_tp(pipe),
            target_output=True,
            shape=(pipe.model.config.hidden_size,),
        )
        ds = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
        collected = collect_features(ds, pipe, [unit], batch_size=1)[unit.id].float()
        expected = (
            capture_residual(pipe, _MID, pipe.load([make_trace(_BASE)]))[:, -1, :]
            .cpu()
            .float()
        )
        torch.testing.assert_close(collected, expected, atol=_ATOL_ACT, rtol=_RTOL_ACT)

    def test_collect_through_featurizer(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        hidden = pipe.model.config.hidden_size
        R = random_rotation(hidden, seed=5)
        unit = ResidualStream(
            layer=_MID,
            token_indices=_last_tp(pipe),
            target_output=True,
            shape=(hidden,),
            featurizer=rotate_featurizer(R),
        )
        ds = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
        collected = collect_features(ds, pipe, [unit], batch_size=1)[unit.id].float()
        resid = (
            capture_residual(pipe, _MID, pipe.load([make_trace(_BASE)]))[:, -1, :]
            .cpu()
            .float()
        )
        torch.testing.assert_close(collected, resid @ R, atol=_ATOL_ACT, rtol=_RTOL_ACT)


# --------------------------------------------------------------------------- #
#  Interchange (residual, components, top-k)                                   #
# --------------------------------------------------------------------------- #
class TestInterchangeGolden:
    def test_full_residual_swap(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        target = _resid_target(pipe, _MID)
        patch = capture_residual(pipe, _MID, pipe.load([make_trace(_SOURCE)]))[
            :, [-1], :
        ]
        manual = next_token_logits(
            pipe, pipe.load([make_trace(_BASE)]), _MID, [-1], patch
        ).float()
        clean = _last_logits(pipe, _BASE)
        causalab = _interchange_logits(pipe, target)
        assert not torch.allclose(causalab, clean, atol=_ATOL_LOGIT)
        torch.testing.assert_close(causalab, manual, atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT)

    @pytest.mark.parametrize(
        "component", ["mlp_output", "mlp_input", "mlp_activation", "attention_output"]
    )
    def test_component_swap(self, chat_coherent_pipeline, component: str) -> None:
        pipe = chat_coherent_pipeline
        tp = _last_tp(pipe)
        unit = (
            AttentionOutput(layer=_MID, token_indices=tp)
            if component == "attention_output"
            else MLP(layer=_MID, token_indices=tp, location=component)
        )
        target = InterchangeTarget([[unit]])
        module, kind = component_module(pipe, _MID, component)
        src = capture_component(pipe, module, kind, pipe.load([make_trace(_SOURCE)]))[
            :, -1, :
        ]

        def edit(h: torch.Tensor) -> None:
            h[:, -1, :] = src

        manual = component_edited_logits(
            pipe, pipe.load([make_trace(_BASE)]), module, kind, edit
        ).float()
        clean = _last_logits(pipe, _BASE)
        causalab = _interchange_logits(pipe, target)
        # Equivalence to the hook oracle is the contract. Non-triviality is only
        # asserted where a single-token swap at this layer reliably moves the
        # last-token logits; `mlp_activation` (the intermediate, after RMSNorm +
        # 30 more layers) can be a small move on near-identical prompts, so its
        # logit delta isn't a reliable signal — equivalence still pins it.
        torch.testing.assert_close(causalab, manual, atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT)
        if component != "mlp_activation":
            assert not torch.allclose(causalab, clean, atol=_ATOL_LOGIT)

    def test_topk_scores_match_full_vocab(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        k = 5
        target = _resid_target(pipe, _MID)
        patch = capture_residual(pipe, _MID, pipe.load([make_trace(_SOURCE)]))[
            :, [-1], :
        ]
        manual = next_token_logits(
            pipe, pipe.load([make_trace(_BASE)]), _MID, [-1], patch
        ).float()
        expected = torch.topk(manual, k, dim=-1)
        result = run_interchange_interventions(
            pipe, [cf_example(_BASE, _SOURCE)], target, output_scores=k
        )
        topk = result["scores"][0][0]
        torch.testing.assert_close(
            topk["top_k_indices"], expected.indices, atol=0, rtol=0
        )


# --------------------------------------------------------------------------- #
#  Feature-space replace / steer / error preservation                         #
# --------------------------------------------------------------------------- #
class _SplitFeat(torch.nn.Module):
    def __init__(self, R: torch.Tensor, k: int) -> None:
        super().__init__()
        self.R = torch.nn.Parameter(R, requires_grad=False)
        self.k = k

    def forward(self, x: torch.Tensor):
        z = x @ self.R
        return z[..., : self.k], z[..., self.k :]


class _SplitInv(torch.nn.Module):
    def __init__(self, R: torch.Tensor) -> None:
        super().__init__()
        self.R = torch.nn.Parameter(R, requires_grad=False)

    def forward(self, features: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        return torch.cat([features, error], dim=-1) @ self.R.T


class TestFeatureSpaceGolden:
    def test_replace_in_rotated_basis(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        hidden = pipe.model.config.hidden_size
        R = random_rotation(hidden, seed=2)
        replacement = torch.linspace(-1.0, 1.0, hidden)
        target = _resid_target(pipe, _MID, rotate_featurizer(R))
        patch = (replacement @ R.T).reshape(1, 1, hidden)
        manual = next_token_logits(
            pipe, pipe.load([make_trace(_BASE)]), _MID, [-1], patch
        ).float()
        causalab = _steer_logits(pipe, target, replacement, mode="replace")
        torch.testing.assert_close(causalab, manual, atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT)

    def test_additive_steer_in_rotated_basis(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        hidden = pipe.model.config.hidden_size
        R = random_rotation(hidden, seed=3)
        vector = torch.linspace(0.2, 0.8, hidden)
        target = _resid_target(pipe, _MID, rotate_featurizer(R))
        fb = capture_residual(pipe, _MID, pipe.load([make_trace(_BASE)]))[:, -1, :]
        dev = fb.device  # oracle math on the model's device (GPU)
        patch = (fb + (vector.to(dev) @ R.to(dev).T)).unsqueeze(1)
        manual = next_token_logits(
            pipe, pipe.load([make_trace(_BASE)]), _MID, [-1], patch
        ).float()
        causalab = _steer_logits(pipe, target, vector, mode="add")
        torch.testing.assert_close(causalab, manual, atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT)

    def test_replace_preserves_orthogonal_error(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        hidden = pipe.model.config.hidden_size
        k = hidden // 2
        R = random_rotation(hidden, seed=4)
        featurizer = Featurizer(
            featurizer=_SplitFeat(R, k), inverse_featurizer=_SplitInv(R), n_features=k
        )
        target = _resid_target(pipe, _MID, featurizer)
        replacement = torch.linspace(-2.0, 2.0, k)
        base = capture_residual(pipe, _MID, pipe.load([make_trace(_BASE)]))[:, -1, :]
        dev = base.device  # oracle math on the model's device (GPU)
        rot = R.to(dev)
        base_z = base @ rot
        patch_z = base_z.clone()
        patch_z[:, :k] = replacement.to(dev)
        patch = (patch_z @ rot.T).unsqueeze(1)
        manual = next_token_logits(
            pipe, pipe.load([make_trace(_BASE)]), _MID, [-1], patch
        ).float()
        causalab = _steer_logits(pipe, target, replacement, mode="replace")
        torch.testing.assert_close(causalab, manual, atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT)


# --------------------------------------------------------------------------- #
#  Noise (backbone-agnostic properties; GPU is non-deterministic so the        #
#  byte-identical reproducibility check is dropped vs. CPU)                    #
# --------------------------------------------------------------------------- #
class TestNoiseGolden:
    def _run(self, pipe, target, *, scale: float, seed: int) -> torch.Tensor:
        uid = target.flatten()[0].id
        vec = {uid: torch.full_like(make_zero_features(target)[uid], float(scale))}
        r = run_steering_interventions(
            pipe,
            [{"input": make_trace(_BASE), "counterfactual_inputs": []}],
            target,
            vec,
            mode="replace",
            type_by_unit={uid: "noise"},
            noise_seed=seed,
            output_scores=True,
        )
        return r["scores"][0][0].float()

    def test_zero_scale_is_no_op(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        target = _resid_target(pipe, _MID)
        torch.testing.assert_close(
            self._run(pipe, target, scale=0.0, seed=0),
            _last_logits(pipe, _BASE),
            atol=_ATOL_LOGIT,
            rtol=_RTOL_LOGIT,
        )

    def test_positive_scale_is_non_trivial(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        target = _resid_target(pipe, _MID)
        noised = self._run(pipe, target, scale=3.0, seed=0)
        assert not torch.allclose(noised, _last_logits(pipe, _BASE), atol=_ATOL_LOGIT)

    def test_different_seed_differs(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        target = _resid_target(pipe, _MID)
        a = self._run(pipe, target, scale=3.0, seed=7)
        b = self._run(pipe, target, scale=3.0, seed=8)
        assert not torch.allclose(a, b, atol=_ATOL_LOGIT)


# --------------------------------------------------------------------------- #
#  Mask                                                                        #
# --------------------------------------------------------------------------- #
def _gate(mask_logits: torch.Tensor, temp: float, *, train: bool) -> torch.Tensor:
    if train:
        return torch.sigmoid(mask_logits / temp)
    return (torch.sigmoid(mask_logits) > 0.5).float()


def _run_mask(pipe, target, mask_logits, temp, *, train) -> torch.Tensor:
    """A mask means nothing until it is configured, so the caller builds and
    configures the interventions and hands them to the runner."""
    interventions = [
        build_intervention(unit.featurizer, "mask") for unit in target.flatten()
    ]
    for intervention in interventions:
        with torch.no_grad():
            intervention.mask.copy_(mask_logits.to(intervention.mask.dtype))
        intervention.set_temperature(temp)
        intervention.train(train)
    out = batched_interchange_intervention(
        pipe,
        [cf_example(_BASE, _SOURCE)],
        target,
        output_scores=True,
        mode="mask",
        interventions=interventions,
    )
    return out["scores"][0].float()


class TestMaskGolden:
    def _split_logits(self, pipe, mask_logits, temp, *, train, featurizer):
        target = _resid_target(pipe, _MID, featurizer)
        fb = capture_residual(pipe, _MID, pipe.load([make_trace(_BASE)]))[:, -1, :]
        fs = capture_residual(pipe, _MID, pipe.load([make_trace(_SOURCE)]))[:, -1, :]
        # gate is computed CPU-side from the mask; move it onto the model device
        # so the oracle blend with the GPU-captured fb/fs stays single-device.
        gate = _gate(mask_logits, temp, train=train).to(fb.device)
        # identity featurizer ⇒ raw blend; rotation handled by caller via R
        return target, fb, fs, gate

    def test_eval_hard_gate(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        hidden = pipe.model.config.hidden_size
        mask = torch.cat(
            [
                torch.full((hidden // 2,), 10.0),
                torch.full((hidden - hidden // 2,), -10.0),
            ]
        )
        target, fb, fs, gate = self._split_logits(
            pipe, mask, 1.0, train=False, featurizer=Featurizer(n_features=hidden)
        )
        patch = ((1 - gate) * fb + gate * fs).unsqueeze(1)
        manual = next_token_logits(
            pipe, pipe.load([make_trace(_BASE)]), _MID, [-1], patch
        ).float()
        causalab = _run_mask(pipe, target, mask, 1.0, train=False)
        assert not torch.allclose(causalab, _last_logits(pipe, _BASE), atol=_ATOL_LOGIT)
        torch.testing.assert_close(causalab, manual, atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT)

    def test_train_soft_gate(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        hidden = pipe.model.config.hidden_size
        mask = torch.cat(
            [torch.full((hidden // 2,), 1.0), torch.full((hidden - hidden // 2,), -1.0)]
        )
        target, fb, fs, gate = self._split_logits(
            pipe, mask, 2.0, train=True, featurizer=Featurizer(n_features=hidden)
        )
        patch = ((1 - gate) * fb + gate * fs).unsqueeze(1)
        manual = next_token_logits(
            pipe, pipe.load([make_trace(_BASE)]), _MID, [-1], patch
        ).float()
        causalab = _run_mask(pipe, target, mask, 2.0, train=True)
        torch.testing.assert_close(causalab, manual, atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT)

    def test_gate_applied_in_feature_space(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        hidden = pipe.model.config.hidden_size
        R = random_rotation(hidden, seed=1)
        mask = torch.cat(
            [
                torch.full((hidden // 2,), 10.0),
                torch.full((hidden - hidden // 2,), -10.0),
            ]
        )
        target = _resid_target(pipe, _MID, rotate_featurizer(R))
        fb = capture_residual(pipe, _MID, pipe.load([make_trace(_BASE)]))[:, -1, :]
        fs = capture_residual(pipe, _MID, pipe.load([make_trace(_SOURCE)]))[:, -1, :]
        dev = fb.device  # oracle math on the model's device (GPU)
        rot = R.to(dev)
        gate = _gate(mask, 1.0, train=False).to(dev)
        patch = (((1 - gate) * (fb @ rot) + gate * (fs @ rot)) @ rot.T).unsqueeze(1)
        manual = next_token_logits(
            pipe, pipe.load([make_trace(_BASE)]), _MID, [-1], patch
        ).float()
        causalab = _run_mask(pipe, target, mask, 1.0, train=False)
        torch.testing.assert_close(causalab, manual, atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT)

    def test_sparsity_loss_is_l1_of_soft_gate(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        hidden = pipe.model.config.hidden_size
        temp = 0.5
        mask = torch.linspace(-3.0, 3.0, hidden)
        target = _resid_target(pipe, _MID, Featurizer(n_features=hidden))
        intervention = build_intervention(target.flatten()[0].featurizer, "mask")
        with torch.no_grad():
            intervention.mask.copy_(mask.to(intervention.mask.dtype))
        intervention.set_temperature(temp)
        intervention.train()
        loss = intervention.get_sparsity_loss().detach().cpu().float()

        expected = torch.norm(torch.sigmoid(mask / temp), p=1)
        torch.testing.assert_close(loss, expected, atol=1e-3, rtol=1e-4)


# --------------------------------------------------------------------------- #
#  Interpolation                                                               #
# --------------------------------------------------------------------------- #
def _interp_logits(pipe, target, fn, params) -> torch.Tensor:
    out = batched_interpolation_intervention(
        pipe,
        [cf_example(_BASE, _SOURCE)],
        target,
        fn=fn,
        params=params,
        output_scores=True,
    )
    return out["scores"][0].float()


def _linear(f_base, f_src, alpha):
    return (1 - alpha) * f_base + alpha * f_src


class TestInterpolationGolden:
    def test_alpha_zero_is_identity(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        causalab = _interp_logits(
            pipe, _resid_target(pipe, _MID), _linear, {"alpha": 0.0}
        )
        torch.testing.assert_close(
            causalab, _last_logits(pipe, _BASE), atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT
        )

    def test_alpha_one_is_full_interchange(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        patch = capture_residual(pipe, _MID, pipe.load([make_trace(_SOURCE)]))[
            :, [-1], :
        ]
        manual = next_token_logits(
            pipe, pipe.load([make_trace(_BASE)]), _MID, [-1], patch
        ).float()
        causalab = _interp_logits(
            pipe, _resid_target(pipe, _MID), _linear, {"alpha": 1.0}
        )
        torch.testing.assert_close(causalab, manual, atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT)

    def test_midpoint_is_convex_blend(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        a = 0.5
        fb = capture_residual(pipe, _MID, pipe.load([make_trace(_BASE)]))[:, [-1], :]
        fs = capture_residual(pipe, _MID, pipe.load([make_trace(_SOURCE)]))[:, [-1], :]
        manual = next_token_logits(
            pipe, pipe.load([make_trace(_BASE)]), _MID, [-1], (1 - a) * fb + a * fs
        ).float()
        causalab = _interp_logits(
            pipe, _resid_target(pipe, _MID), _linear, {"alpha": a}
        )
        assert not torch.allclose(causalab, _last_logits(pipe, _BASE), atol=_ATOL_LOGIT)
        torch.testing.assert_close(causalab, manual, atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT)


# --------------------------------------------------------------------------- #
#  Cross-model (source == target reproduces interchange — no 2× memory)        #
# --------------------------------------------------------------------------- #
class TestCrossModelGolden:
    def test_source_representations_reproduce_interchange(
        self, chat_coherent_pipeline
    ) -> None:
        pipe = chat_coherent_pipeline
        target = _resid_target(pipe, _MID)
        patch = capture_residual(pipe, _MID, pipe.load([make_trace(_SOURCE)]))[
            :, [-1], :
        ]
        manual = next_token_logits(
            pipe, pipe.load([make_trace(_BASE)]), _MID, [-1], patch
        ).float()
        result = run_interchange_interventions(
            pipe,
            [cf_example(_BASE, _SOURCE)],
            target,
            source_pipeline=pipe,
            output_scores=True,
        )
        causalab = result["scores"][0][0].float()
        torch.testing.assert_close(causalab, manual, atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT)


# --------------------------------------------------------------------------- #
#  Path patching: forward-order contract + two-pass                            #
# --------------------------------------------------------------------------- #
class TestPathPatchingGolden:
    def test_upstream_edit_visible_downstream(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        base = pipe.load([make_trace(_BASE)])
        clean = capture_residual(pipe, _MID + 1, base)

        def bump(h: torch.Tensor) -> None:
            h[:, -1, :] += 5.0

        edited = capture_with_edits(
            pipe,
            base,
            decoder_block(pipe, _MID + 1),
            "out",
            [(decoder_block(pipe, _MID), "out", bump)],
        )
        assert not torch.allclose(edited, clean, atol=1e-3)

    def test_two_pass_matches_collect_inject(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        sender = ResidualStream(
            layer=_MID,
            token_indices=_last_tp(pipe),
            target_output=True,
            shape=(pipe.model.config.hidden_size,),
        )
        receiver = ResidualStream(
            layer=_MID + 1,
            token_indices=_last_tp(pipe),
            target_output=True,
            shape=(pipe.model.config.hidden_size,),
        )
        base_inputs = pipe.load([make_trace(_BASE)])
        src_sender = capture_residual(pipe, _MID, pipe.load([make_trace(_SOURCE)]))[
            :, -1, :
        ]

        def write_sender(h: torch.Tensor) -> None:
            h[:, -1, :] = src_sender

        v_star = capture_with_edits(
            pipe,
            base_inputs,
            decoder_block(pipe, _MID + 1),
            "out",
            [(decoder_block(pipe, _MID), "out", write_sender)],
        )[:, [-1], :]
        manual = next_token_logits(pipe, base_inputs, _MID + 1, [-1], v_star).float()
        clean = _last_logits(pipe, _BASE)
        result = run_two_pass_path_patching(
            pipe,
            [cf_example(_BASE, _SOURCE)],
            InterchangeTarget([[sender]]),
            receiver,
            output_scores=True,
        )
        causalab = result["scores"][0][0].float()
        assert not torch.allclose(manual, clean, atol=_ATOL_LOGIT)
        torch.testing.assert_close(causalab, manual, atol=_ATOL_LOGIT, rtol=_RTOL_LOGIT)


# --------------------------------------------------------------------------- #
#  Causal tracing: mixed noise + replace (restore the final layer → clean)     #
# --------------------------------------------------------------------------- #
class TestCausalTracingGolden:
    def _unit(self, pipe, layer):
        return ResidualStream(
            layer=layer,
            token_indices=_last_tp(pipe),
            target_output=True,
            shape=(pipe.model.config.hidden_size,),
        )

    def test_corruption_alone_is_non_trivial(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        hidden = pipe.model.config.hidden_size
        entry = self._unit(pipe, _MID)
        target = InterchangeTarget([[entry]])
        r = run_steering_interventions(
            pipe,
            [{"input": make_trace(_BASE), "counterfactual_inputs": []}],
            target,
            {entry.id: torch.full((hidden,), 5.0)},
            mode="replace",
            type_by_unit={entry.id: "noise"},
            noise_seed=0,
            output_scores=True,
        )
        assert not torch.allclose(
            r["scores"][0][0].float(), _last_logits(pipe, _BASE), atol=_ATOL_LOGIT
        )

    def test_restore_final_layer_recovers_clean(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        hidden = pipe.model.config.hidden_size
        last = pipe.model.config.num_hidden_layers - 1
        entry = self._unit(pipe, _MID)
        restore = self._unit(pipe, last)
        target = InterchangeTarget([[entry], [restore]])
        clean_restore = capture_residual(pipe, last, pipe.load([make_trace(_BASE)]))[
            :, -1, :
        ].squeeze(0)
        r = run_steering_interventions(
            pipe,
            [{"input": make_trace(_BASE), "counterfactual_inputs": []}],
            target,
            {entry.id: torch.full((hidden,), 5.0), restore.id: clean_restore},
            mode="replace",
            type_by_unit={entry.id: "noise", restore.id: "replace"},
            noise_seed=0,
            output_scores=True,
        )
        torch.testing.assert_close(
            r["scores"][0][0].float(),
            _last_logits(pipe, _BASE),
            atol=_ATOL_LOGIT,
            rtol=_RTOL_LOGIT,
        )


# --------------------------------------------------------------------------- #
#  Per-head value / query receivers — the decoupled-head_dim GQA risk          #
# --------------------------------------------------------------------------- #
class TestHeadReceiversGolden:
    def test_head_value_matches_vproj_kv_slice(self, chat_coherent_pipeline) -> None:
        """Per-head value receiver vs. the v_proj KV slice. Qwen3-4B has a
        *decoupled* head_dim (128 ≠ hidden/n_head=80) — the configuration the
        pyvene backbone could not address at all (#386), so this surfaces a real
        discrepancy if the engine gets the per-head width wrong."""
        pipe = chat_coherent_pipeline
        d = head_dim(pipe)
        v_proj = pipe.model.model.layers[_MID].self_attn.v_proj
        v_out = capture_component(pipe, v_proj, "out", pipe.load([make_trace(_BASE)]))
        unit = AttentionHeadValue(layer=_MID, head=0, token_indices=_last_tp(pipe))
        ds = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
        collected = collect_features(ds, pipe, [unit], batch_size=1)[unit.id].float()
        expected = v_out[:, -1, 0:d].cpu().float()
        torch.testing.assert_close(collected, expected, atol=_ATOL_ACT, rtol=_RTOL_ACT)

    def test_head_query_matches_qproj_slice(self, chat_coherent_pipeline) -> None:
        pipe = chat_coherent_pipeline
        d = head_dim(pipe)
        q_proj = pipe.model.model.layers[_MID].self_attn.q_proj
        q_out = capture_component(pipe, q_proj, "out", pipe.load([make_trace(_BASE)]))
        unit = AttentionHeadQuery(layer=_MID, head=0, token_indices=_last_tp(pipe))
        ds = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
        collected = collect_features(ds, pipe, [unit], batch_size=1)[unit.id].float()
        expected = q_out[:, -1, 0:d].cpu().float()
        torch.testing.assert_close(collected, expected, atol=_ATOL_ACT, rtol=_RTOL_ACT)

    def test_head_attention_value_output_sender(self, chat_coherent_pipeline) -> None:
        """The path-patching *sender* (head_attention_value_output) = the head's
        slice of o_proj's input."""
        pipe = chat_coherent_pipeline
        unit = AttentionHead(
            layer=_MID, head=0, token_indices=_last_tp(pipe), target_output=True
        )
        ds = [{"input": make_trace(_BASE), "counterfactual_inputs": []}]
        collected = collect_features(ds, pipe, [unit], batch_size=1)[unit.id].float()
        expected = (
            capture_head_value(pipe, _MID, 0, pipe.load([make_trace(_BASE)]))[:, -1, :]
            .cpu()
            .float()
        )
        torch.testing.assert_close(collected, expected, atol=_ATOL_ACT, rtol=_RTOL_ACT)
