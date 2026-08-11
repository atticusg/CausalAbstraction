"""Tests for :mod:`causalab.neural.trainable` — ED3 (#402), absorbing the F6
(#422) grad-flow spike's CPU-verifiable core.

Tiers (mirroring ``tests/neural/test_modes.py``; ``causalab/neural`` owes
``unit`` + ``property``, and a ``golden`` GPU pin is the established pattern):

* ``unit`` — the pure pieces: hard-threshold feature readout (tied/untied),
  the CE label slice on hand-built logits, parameter discovery/dedup. (The
  outer loop and its temperature schedule live in
  ``causalab/methods/edit_training.py`` — tested in
  ``tests/methods/test_edit_training.py``.)
* ``property`` — **the grad contract**, pinned against a *grad-enabled*
  raw-hook oracle (the shipped ``hook_oracle`` helpers run under ``no_grad``,
  so this file carries its own grad variant of the same hook pattern): the
  same featurizer/gate modules, the same loss, must produce the same
  ``param.grad`` through the new stack as through raw
  ``register_forward_hook`` — through the tuple-rewrap write (``block_output``
  on tiny Llama **and** GPT-2), a plain write (``mlp_output``), both featurize
  paths (base + raw source featurized live), a ``MaskGate``, batch scale with
  left-padding, and a ``HeadSite`` projection write (grads flow). Plus the
  saved-logits-backward pin and the label-concat loss slice against a plain
  HF forward. (Mini DBM/DAS training runs live with the loop in
  ``tests/methods/test_edit_training.py``.)
* ``golden`` — the F6 (#422) **multi-device
  sharded validation** (requires ≥2 CUDA devices, so the single-GPU nightly
  skips it — run on demand on a 2-GPU allocation): tiny Llama sharded with an
  explicit two-device ``hf_device_map``, pinning placement per site device,
  sharded-vs-single-device grad parity (including a cross-device raw source),
  and one optimizer stepping params scattered across devices.

Models come from the fresh (uncached) factories in ``tests/_helpers/tiny.py``
(see their docstrings — the session-cached singletons carry pyvene hooks that
break nnsight traces).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch
from nnterp import StandardizedTransformer

from causalab.methods.edit_training import TrainBatch, train_edits
from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.edit import Edit
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.head_view import HeadSite
from causalab.neural.modes import MaskGate, interchange
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import Site
from causalab.neural.trainable import (
    concat_label_inputs,
    das_edit,
    dbm_edit,
    edit_parameters,
    label_ce_loss,
    place_edit_parameters,
    selected_feature_ids,
    traced_label_loss,
)

from tests._helpers.tiny import fresh_tiny_random_gpt2, fresh_tiny_random_llama
from tests.neural.activations.hook_oracle import (
    capture_component,
    component_module,
    hidden_of,
)

_TEXT = "the quick brown fox jumps"


def _subspace(
    width: int, k: int, *, trainable: bool, seed: int = 0
) -> SubspaceFeaturizer:
    torch.manual_seed(seed)
    return SubspaceFeaturizer(shape=(width, k), trainable=trainable)


# --------------------------------------------------------------------------- #
#  Fixtures — fresh StandardizedTransformers + a grad-enabled hook oracle      #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class GradCase:
    st: StandardizedTransformer
    oracle: Any  # SimpleNamespace(hf_model=raw) — for the hook_oracle helpers
    tok: Any

    def inputs(self, texts: str | list[str] = _TEXT) -> dict[str, torch.Tensor]:
        texts = [texts] if isinstance(texts, str) else texts
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        enc = self.tok(texts, return_tensors="pt", padding=True)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def capture(self, component: str, layer: int, inputs: Any) -> torch.Tensor:
        module, kind = component_module(self.oracle, layer, component)
        return capture_component(self.oracle, module, kind, inputs)

    def hidden(self) -> int:
        return int(self.oracle.hf_model.config.hidden_size)

    # -- grad-enabled oracle ---------------------------------------------------- #
    def oracle_grads(
        self,
        site: Site,
        edit_fn: Callable[[torch.Tensor], None],
        positions: list[int],
        inputs: Any,
        params: list[torch.nn.Parameter],
        target_id: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Ground-truth ``(loss, param grads)``: a raw forward hook applies
        ``edit_fn`` (in-place, grad-enabled — the shipped oracle helpers run
        under ``no_grad``, so this is the grad variant of the same pattern),
        the last-token CE against ``target_id`` is backpropagated, and each
        param's grad is cloned out."""
        module, kind = component_module(self.oracle, site.layer, site.component)

        if kind == "out":

            def out_hook(_m: Any, _i: Any, out: Any) -> Any:
                hidden = hidden_of(out).clone()
                edit_fn(hidden)
                return (hidden, *out[1:]) if isinstance(out, tuple) else hidden

            handle = module.register_forward_hook(out_hook)
        else:

            def pre_hook(_m: Any, args: tuple) -> tuple:
                x = args[0].clone()
                edit_fn(x)
                return (x, *args[1:])

            handle = module.register_forward_pre_hook(pre_hook)
        for p in params:
            p.grad = None
        try:
            logits = self.oracle.hf_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, -1, :],
                torch.full((logits.shape[0],), target_id, dtype=torch.long),
            )
            loss.backward()
        finally:
            handle.remove()
        grads = [
            (p.grad.detach().clone() if p.grad is not None else None) for p in params
        ]
        for p in params:
            p.grad = None
        return loss.detach(), grads

    def stack_grads(
        self,
        edit: Edit,
        inputs: Any,
        params: list[torch.nn.Parameter],
        target_id: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The same loss through the new stack: apply ``edit`` in a trace, save
        the logits (on-device, graph intact — the pinned saved-logits-backward
        contract), CE outside the trace, backward, clone the grads out."""
        for p in params:
            p.grad = None
        with self.st.trace(inputs):
            edit.apply(self.st)
            logits = self.st.logits.save()
        loss = torch.nn.functional.cross_entropy(
            logits[:, -1, :],
            torch.full((logits.shape[0],), target_id, dtype=torch.long),
        )
        loss.backward()
        grads = [
            (p.grad.detach().clone() if p.grad is not None else None) for p in params
        ]
        for p in params:
            p.grad = None
        return loss.detach(), grads


def _case(raw: Any, tok: Any) -> GradCase:
    freeze_model_parameters_raw(raw)
    st = StandardizedTransformer(raw, tokenizer=tok, check_renaming=True)
    st.dispatch()
    return GradCase(st=st, oracle=SimpleNamespace(hf_model=raw), tok=tok)


def freeze_model_parameters_raw(raw: Any) -> None:
    """Freeze before wrapping — both the oracle path and the trace path then
    run with the trained-edit regime's frozen base (the at-load contract)."""
    for p in raw.parameters():
        p.requires_grad_(False)


@pytest.fixture(scope="module")
def llama_case() -> GradCase:
    return _case(*fresh_tiny_random_llama())


@pytest.fixture(scope="module")
def gpt2_case() -> GradCase:
    return _case(*fresh_tiny_random_gpt2())


def _trainable_params(feat: SubspaceFeaturizer) -> list[torch.nn.Parameter]:
    return [p for p in feat.featurizer.parameters() if p.requires_grad]


def _assert_grads_close(got: list[torch.Tensor], expected: list[torch.Tensor]) -> None:
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g is not None and e is not None
        assert torch.isfinite(g).all()
        assert g.abs().max() > 0  # the path actually carries gradient
        torch.testing.assert_close(g, e, atol=1e-6, rtol=1e-4)


# --------------------------------------------------------------------------- #
#  unit — pure pieces                                                          #
# --------------------------------------------------------------------------- #
class TestReadoutUnit:
    pytestmark = pytest.mark.unit

    def test_selected_feature_ids_per_feature(self) -> None:
        gate = MaskGate(4)
        with torch.no_grad():
            gate.mask.copy_(torch.tensor([-2.0, 3.0, -1.0, 0.5]))
        assert selected_feature_ids(gate) == [1, 3]

    def test_selected_feature_ids_tied(self) -> None:
        gate = MaskGate(tie=True)
        with torch.no_grad():
            gate.mask.copy_(torch.tensor([2.0]))
        assert selected_feature_ids(gate) is None  # tied-on = all features
        with torch.no_grad():
            gate.mask.copy_(torch.tensor([-2.0]))
        assert selected_feature_ids(gate) == []


class TestLossSliceUnit:
    pytestmark = pytest.mark.unit

    def test_label_ce_loss_slices_the_label_span(self) -> None:
        """Logits that put all mass on the right label tokens at positions
        ``-L-1:-1`` yield ~zero loss; garbage elsewhere is never read."""
        vocab, big = 6, 30.0
        logits = torch.zeros(1, 5, vocab)
        labels = torch.tensor([[2, 4]])
        logits[0, -3, 2] = big  # predicts label token 0
        logits[0, -2, 4] = big  # predicts label token 1
        logits[0, -1, :] = torch.randn(vocab)  # past the span — ignored
        assert label_ce_loss(logits, labels, pad_token_id=0).item() < 1e-4

    def test_label_ce_loss_ignores_pad(self) -> None:
        vocab, pad = 6, 0
        logits = torch.zeros(1, 5, vocab)
        labels = torch.tensor([[2, pad]])
        logits[0, -3, 2] = 30.0
        # position -2 predicts a pad label — must not contribute
        loss = label_ce_loss(logits, labels, pad_token_id=pad)
        assert loss.item() < 1e-4


class TestParameterDiscoveryUnit:
    pytestmark = pytest.mark.unit

    def test_edit_parameters_dedupes_shared_modules(self) -> None:
        feat = _subspace(16, 3, trainable=False)
        gate = MaskGate(3)
        fsite = FeaturizedSite(Site("block_output", 1), feat)
        e1 = dbm_edit(fsite, torch.zeros(1, 1, 16), gate)
        e2 = dbm_edit(fsite, torch.ones(1, 1, 16), gate)  # same gate, new batch
        params = edit_parameters([e1, e2])
        assert params == [gate.mask]  # frozen rotation excluded, gate once

    def test_edit_parameters_includes_trainable_featurizer(self) -> None:
        feat = _subspace(16, 3, trainable=True)
        edit = das_edit(
            FeaturizedSite(Site("block_output", 1), feat), torch.zeros(1, 1, 16)
        )
        params = edit_parameters([edit])
        assert params and all(p.requires_grad for p in params)


# --------------------------------------------------------------------------- #
#  property — the grad contract vs the grad-enabled hook oracle                #
# --------------------------------------------------------------------------- #
class TestGradContractMatchesOracle:
    pytestmark = pytest.mark.property

    @pytest.fixture(params=["llama", "gpt2"])
    def case(self, request, llama_case: GradCase, gpt2_case: GradCase) -> GradCase:
        return {"llama": llama_case, "gpt2": gpt2_case}[request.param]

    def test_das_rotation_grads_match_oracle(self, case: GradCase) -> None:
        """DAS shape at ``block_output`` (the tuple-rewrap write): the trainable
        rotation featurizes both the base read and the raw source, in-trace;
        its grads must equal the raw-hook implementation's exactly."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        feat = _subspace(case.hidden(), 3, trainable=True)
        raw_src = case.capture("block_output", 0, inputs)[:, [last]]
        site = Site("block_output", 1)
        edit = das_edit(FeaturizedSite(site, feat), raw_src, positions=[last])
        params = _trainable_params(feat)
        assert params

        def oracle_edit(h: torch.Tensor) -> None:
            f, err = feat.featurize(h[:, [last]])
            f_src, _ = feat.featurize(raw_src)
            h[:, [last]] = feat.inverse_featurize(f_src, err).to(h.dtype)

        loss_o, grads_o = case.oracle_grads(
            site, oracle_edit, [last], inputs, params, target_id=7
        )
        loss_s, grads_s = case.stack_grads(edit, inputs, params, target_id=7)
        torch.testing.assert_close(loss_s, loss_o, atol=1e-5, rtol=1e-4)
        _assert_grads_close(grads_s, grads_o)

    def test_dbm_mask_grads_match_oracle(self, case: GradCase) -> None:
        """DBM shape at ``mlp_output`` (a plain, non-tuple write): the gate's
        mask grads through the soft blend must equal the raw-hook path's."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        feat = _subspace(case.hidden(), 4, trainable=False)
        gate = MaskGate(4).train()
        gate.set_temperature(0.7)
        raw_src = case.capture("mlp_output", 0, inputs)[:, [last]]
        site = Site("mlp_output", 1)
        edit = dbm_edit(FeaturizedSite(site, feat), raw_src, gate, positions=[last])
        params = [gate.mask]

        def oracle_edit(h: torch.Tensor) -> None:
            f, err = feat.featurize(h[:, [last]])
            f_src, _ = feat.featurize(raw_src)
            h[:, [last]] = feat.inverse_featurize(gate(f, f_src), err).to(h.dtype)

        loss_o, grads_o = case.oracle_grads(
            site, oracle_edit, [last], inputs, params, target_id=7
        )
        loss_s, grads_s = case.stack_grads(edit, inputs, params, target_id=7)
        torch.testing.assert_close(loss_s, loss_o, atol=1e-5, rtol=1e-4)
        _assert_grads_close(grads_s, grads_o)

    def test_batch_scale_grads_match_oracle(self, case: GradCase) -> None:
        """The same DAS parity on a left-padded batch of 3 — the batch-scale
        row of the F6 matrix."""
        inputs = case.inputs(
            ["the quick brown fox", "a cat", "every valley shall be exalted"]
        )
        last = int(inputs["input_ids"].shape[1] - 1)
        feat = _subspace(case.hidden(), 2, trainable=True)
        raw_src = case.capture("block_output", 0, inputs)[:, [last]]
        site = Site("block_output", 1)
        edit = das_edit(FeaturizedSite(site, feat), raw_src, positions=[last])
        params = _trainable_params(feat)

        def oracle_edit(h: torch.Tensor) -> None:
            f, err = feat.featurize(h[:, [last]])
            f_src, _ = feat.featurize(raw_src)
            h[:, [last]] = feat.inverse_featurize(f_src, err).to(h.dtype)

        loss_o, grads_o = case.oracle_grads(
            site, oracle_edit, [last], inputs, params, target_id=7
        )
        loss_s, grads_s = case.stack_grads(edit, inputs, params, target_id=7)
        torch.testing.assert_close(loss_s, loss_o, atol=1e-5, rtol=1e-4)
        _assert_grads_close(grads_s, grads_o)

    def test_headsite_write_carries_gradient(self, llama_case: GradCase) -> None:
        """The ST4×ED3 combination: a trainable featurizer on a per-head
        projection write (``attention_value`` = o_proj input slice) receives
        finite, nonzero gradient through the head write path."""
        case = llama_case
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        head_dim = int(
            getattr(case.oracle.hf_model.config, "head_dim", None)
            or case.hidden() // case.oracle.hf_model.config.num_attention_heads
        )
        feat = _subspace(head_dim, 2, trainable=True)
        hsite = HeadSite("attention_value", layer=1, head=0)
        edit = das_edit(
            FeaturizedSite(hsite, feat),  # type: ignore[arg-type]  # SiteLike protocol
            torch.zeros(1, 1, head_dim),
            positions=[last],
        )
        params = _trainable_params(feat)
        _, grads = case.stack_grads(edit, inputs, params, target_id=7)
        for g in grads:
            assert g is not None and torch.isfinite(g).all()
        assert any(g.abs().max() > 0 for g in grads)


class TestLossSliceProperty:
    pytestmark = pytest.mark.property

    @pytest.fixture(scope="class")
    def pipe(self) -> LMPipeline:
        raw, _tok = fresh_tiny_random_llama()
        return LMPipeline(raw, max_new_tokens=3, position_ids=True)

    def test_concat_label_inputs_matches_plain_forward_ce(
        self, pipe: LMPipeline
    ) -> None:
        """The loss slice equals a hand-computed CE on a plain HF forward of
        the same joint batch — the label-concat + slice port is faithful."""
        base = pipe.load([_trace(t) for t in ["the quick brown", "a cat sat"]])
        joint, label_ids = concat_label_inputs(pipe, dict(base), [" fox", " down"])
        assert joint["input_ids"].shape[-1] == base["input_ids"].shape[-1] + int(
            pipe.max_new_tokens
        )
        assert "position_ids" in joint

        loss, pred_ids = traced_label_loss(
            pipe.model,
            joint,
            label_ids,
            edits=[],
            pad_token_id=pipe.tokenizer.pad_token_id,
        )
        with torch.no_grad():
            logits = pipe.hf_model(
                input_ids=joint["input_ids"],
                attention_mask=joint["attention_mask"],
                position_ids=joint["position_ids"],
            ).logits
        expected = label_ce_loss(logits, label_ids, pipe.tokenizer.pad_token_id)
        torch.testing.assert_close(loss.detach(), expected, atol=1e-5, rtol=1e-4)
        assert pred_ids.shape == label_ids.shape

    def test_forward_trims_logits_to_label_span(self, pipe: LMPipeline) -> None:
        """#449 finding 4: under ``force_last_token_logits`` the traced logits
        carry only the trailing positions — value-equal to the full forward's
        trailing slice, so the training loop stops saving full-sequence
        logits with graph. Not bit-exact: the trimmed lm_head GEMM has a
        different shape, and BLAS kernel dispatch varies with shape and
        machine (bit-identical on the cluster CPUs, ~1-ulp off on hosted CI
        runners) — so compare at float32 tolerance. A wrong-slice bug would
        miss by O(1), far beyond it."""
        from causalab.neural.trainable import force_last_token_logits

        base = pipe.load([_trace("the quick brown")])
        inputs = {k: base[k] for k in ("input_ids", "attention_mask")}
        with pipe.model.trace(inputs):
            full = pipe.model.logits.save()
        with force_last_token_logits(pipe.hf_model, 3):
            with pipe.model.trace(inputs):
                trimmed = pipe.model.logits.save()
        assert trimmed.shape[1] == 3
        assert full.shape[1] == inputs["input_ids"].shape[1]
        torch.testing.assert_close(trimmed, full[:, -3:])

    def test_force_last_token_logits_no_op_without_support(self) -> None:
        """A forward without ``logits_to_keep`` is left untouched — the
        honest fallback for architectures that predate the kwarg."""
        from causalab.neural.trainable import force_last_token_logits

        class Legacy:
            def forward(self, input_ids=None):  # no logits_to_keep
                return "unpatched"

        legacy = Legacy()
        with force_last_token_logits(legacy, 2):
            # No instance-level wrapper was installed — the class forward runs.
            assert "forward" not in legacy.__dict__
            assert legacy.forward() == "unpatched"

    def test_freshly_loaded_pipeline_is_frozen(self) -> None:
        """The at-load freeze contract: a name-loaded LMPipeline's base params
        never require grad (pre-loaded instances stay the caller's business)."""
        pipe = LMPipeline(
            "hf-internal-testing/tiny-random-LlamaForCausalLM", max_new_tokens=2
        )
        assert all(not p.requires_grad for p in pipe.hf_model.parameters())


def _trace(text: str) -> Any:
    from causalab.causal.trace import CausalTrace, Mechanism

    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


# --------------------------------------------------------------------------- #
#  golden — multi-device sharded validation (#422, the F6 row a single GPU     #
#  cannot pin)                                                                 #
# --------------------------------------------------------------------------- #
_TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"

# Explicit two-device split of the 2-layer tiny Llama — full sharding, unlike
# ``device_map="auto"``, which packs a tiny model onto one device (and the
# single-GPU nightly onto its only one). Untied embeddings, so head and
# embedding place freely.
_SHARD_MAP = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.rotary_emb": 0,
    "model.layers.1": 1,
    "model.norm": 1,
    "lm_head": 1,
}


class TestMultiDeviceShardedGolden:
    """The grad contract's sharded row: on an ``hf_device_map`` model whose
    layers live on different GPUs, :func:`place_edit_parameters` follows each
    site's layer device, the math matches the single-device run exactly (incl.
    a raw source collected on one device and consumed at a site on another),
    and one optimizer steps parameters scattered across devices.

    The nightly golden runner has one GPU, so these skip there — run them on
    demand on a ≥2-GPU allocation (``pytest -m golden -k MultiDeviceSharded``).
    """

    pytestmark = [
        pytest.mark.golden,
        pytest.mark.skipif(
            torch.cuda.device_count() < 2,
            reason="multi-device sharded validation needs >=2 CUDA devices",
        ),
    ]

    @pytest.fixture(scope="class")
    def sharded(self) -> LMPipeline:
        pipe = LMPipeline(
            _TINY_LLAMA, max_new_tokens=1, position_ids=True, device_map=_SHARD_MAP
        )
        dev0, dev1 = self._shard_devices(pipe)
        assert dev0 != dev1, "precondition: the explicit map really sharded"
        return pipe

    @pytest.fixture(scope="class")
    def single(self) -> LMPipeline:
        return LMPipeline(
            _TINY_LLAMA, max_new_tokens=1, position_ids=True, device="cuda:0"
        )

    @staticmethod
    def _shard_devices(pipe: LMPipeline) -> tuple[torch.device, torch.device]:
        model = pipe.model
        return (
            next(model.model.layers[0].parameters()).device,
            next(model.model.layers[1].parameters()).device,
        )

    @staticmethod
    def _joint_batch(
        pipe: LMPipeline, texts: list[str]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[str]]:
        """Label-concatenated inputs with the self-consistent objective (each
        label = the model's own greedy next token — see ``TestTrainingRuns``)."""
        base = pipe.load([_trace(t) for t in texts])
        with torch.no_grad():
            clean = pipe.hf_model(
                input_ids=base["input_ids"],
                attention_mask=base["attention_mask"],
            ).logits[:, -1, :]
        labels = [pipe.tokenizer.decode(i) for i in clean.argmax(dim=-1).tolist()]
        joint, label_ids = concat_label_inputs(pipe, dict(base), labels)
        return joint, label_ids, labels

    def test_placement_follows_each_sites_layer_device(
        self, sharded: LMPipeline
    ) -> None:
        """Each edit's modules land on *its* site's layer device; a site-backed
        read-source's featurizer lands on the *source* site's device (where its
        read fires), not the destination's."""
        model = sharded.model
        dev0, dev1 = self._shard_devices(sharded)
        d = int(sharded.hf_model.config.hidden_size)
        feat0 = _subspace(d, 2, trainable=True)
        feat_gate = _subspace(d, 3, trainable=False)
        gate = MaskGate(3)
        e_das = das_edit(
            FeaturizedSite(Site("block_output", 0), feat0), torch.zeros(1, 1, d)
        )
        e_dbm = dbm_edit(
            FeaturizedSite(Site("mlp_output", 1), feat_gate),
            torch.zeros(1, 1, d),
            gate,
        )
        feat_dst = _subspace(d, 2, trainable=False)
        feat_src = _subspace(d, 2, trainable=False)
        e_cross = interchange(
            FeaturizedSite(Site("block_output", 1), feat_dst),
            FeaturizedSite(Site("block_output", 0), feat_src),
        )
        place_edit_parameters(model, [e_das, e_dbm, e_cross])
        assert next(feat0.featurizer.parameters()).device == dev0
        assert next(feat_gate.featurizer.parameters()).device == dev1
        assert gate.mask.device == dev1
        assert next(feat_dst.featurizer.parameters()).device == dev1
        assert next(feat_src.featurizer.parameters()).device == dev0

    def test_sharded_grads_match_single_device(
        self, sharded: LMPipeline, single: LMPipeline
    ) -> None:
        """Sharding is invisible to the math: the same checkpoint, the same
        same-seed rotation, and the same batch produce the same loss and the
        same ``param.grad`` whether the model spans two GPUs or sits on one.
        The raw source is collected at layer 0 (first shard) and consumed at a
        layer-1 site (second shard) — the cross-device movement path."""
        texts = ["the quick brown fox", "a cat sat on"]
        results = {}
        for name, pipe in (("sharded", sharded), ("single", single)):
            d = int(pipe.hf_model.config.hidden_size)
            feat = _subspace(d, 2, trainable=True, seed=7)
            joint, label_ids, labels = self._joint_batch(pipe, texts)
            raw_src = Site("block_output", 0).collect(
                pipe.model, {k: joint[k] for k in ("input_ids", "attention_mask")}
            )[:, -2:-1]
            edit = das_edit(
                FeaturizedSite(Site("block_output", 1), feat),
                raw_src,
                positions=[-2],
            )
            place_edit_parameters(pipe.model, [edit])
            params = _trainable_params(feat)
            assert params
            loss, _ = traced_label_loss(
                pipe.model, joint, label_ids, [edit], pipe.tokenizer.pad_token_id
            )
            loss.backward()
            results[name] = (
                labels,
                loss.detach().cpu(),
                [p.grad.detach().cpu().clone() for p in params],
            )
        labels_s, loss_s, grads_s = results["sharded"]
        labels_r, loss_r, grads_r = results["single"]
        assert labels_s == labels_r  # same checkpoint ⇒ same greedy labels
        torch.testing.assert_close(loss_s, loss_r, atol=1e-5, rtol=1e-4)
        _assert_grads_close(grads_s, grads_r)

    def test_one_optimizer_steps_params_scattered_across_devices(
        self, sharded: LMPipeline
    ) -> None:
        """One AdamW over a dev0-site rotation and a dev1-site gate: grads
        arrive on each param's own device, the step moves both in place, the
        optimizer state lives with each param — and a ``train_edits`` run over
        the same edits keeps the placement."""
        model = sharded.model
        dev0, dev1 = self._shard_devices(sharded)
        d = int(sharded.hf_model.config.hidden_size)
        feat = _subspace(d, 2, trainable=True)
        feat_gate = _subspace(d, 4, trainable=False)
        gate = MaskGate(4).train()
        gate.set_temperature(0.7)
        joint, label_ids, _ = self._joint_batch(
            sharded, ["every valley shall", "few things are"]
        )
        raw1 = Site("block_output", 0).collect(
            model, {k: joint[k] for k in ("input_ids", "attention_mask")}
        )[:, -2:-1]
        e0 = das_edit(
            FeaturizedSite(Site("block_output", 0), feat),
            torch.zeros(1, 1, d),
            positions=[-2],
        )
        e1 = dbm_edit(
            FeaturizedSite(Site("mlp_output", 1), feat_gate),
            raw1,
            gate,
            positions=[-2],
        )
        place_edit_parameters(model, [e0, e1])
        params = edit_parameters([e0, e1])
        assert {p.device for p in params} == {dev0, dev1}  # genuinely scattered

        optimizer = torch.optim.AdamW(params, lr=0.05, weight_decay=0)
        loss, _ = traced_label_loss(
            model, joint, label_ids, [e0, e1], sharded.tokenizer.pad_token_id
        )
        optimizer.zero_grad()
        loss.backward()
        for p in params:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert p.grad.abs().max() > 0
            assert p.grad.device == p.device
        before = [p.detach().clone() for p in params]
        optimizer.step()
        for p, b in zip(params, before):
            assert p.device == b.device  # the step never migrates a param
            assert not torch.allclose(p.detach(), b)
            assert optimizer.state[p]["exp_avg"].device == p.device

        history = train_edits(
            model,
            [TrainBatch(inputs=joint, label_ids=label_ids, edits=(e0, e1))],
            pad_token_id=sharded.tokenizer.pad_token_id,
            epochs=2,
            lr=0.01,
            gates=[gate],
            temperature=(1.0, 0.1),
            annealing_fraction=0.5,
            regularization_coefficient=0.0,
        )
        assert all(torch.isfinite(torch.tensor(h["loss"])) for h in history)
        assert next(feat.featurizer.parameters()).device == dev0
        assert gate.mask.device == dev1
