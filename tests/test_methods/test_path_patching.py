"""Tests for causalab.methods.path_patching.

Three tiers:
  * pure logic (PathSpec, padding shifts, coverage drift) — no model, no
    network;
  * tiny-random end-to-end (construction guards, closure, cascade
    equivalence, twin agreement, negative controls) on 4-layer random
    models of each supported family, CPU — needs only the gpt2 tokenizer;
  * hygiene (no raw torch hooks anywhere in the method).

The tiny-random tier exercises structural identities (additivity, closure,
engine-vs-twin equality), which hold for random weights; validation on real
models is the experiment's per-model matrix, not CI's job.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from causalab.methods.path_patching import (
    GuardError,
    PatchEngine,
    PathSpec,
    UnsupportedArchitectureError,
    build_patch_cache,
    coverage_table,
    reference_patched_logits,
    resolve_descriptor,
)
from causalab.neural.padding import shift_indices_for_padding

METHOD_DIR = (
    Path(__file__).resolve().parents[2] / "causalab" / "methods" / "path_patching"
)
COVERAGE_ARTIFACT = (
    Path(__file__).resolve().parent / "artifacts" / "path_patching_coverage.json"
)


# ---------------------------------------------------------------------------
# pure logic
# ---------------------------------------------------------------------------


class TestPathSpec:
    def test_cascade_first_expansion(self):
        s = PathSpec.cascade([8, 10, 11], direct_to_logits=False)
        assert s.receivers == (8, 10, 11)
        assert s.sender_to == frozenset({8})
        assert s.receiver_to_receiver == frozenset({(8, 10), (8, 11), (10, 11)})
        assert not s.sender_to_logits
        assert s.receivers_to_logits == frozenset({8, 10, 11})

    def test_cascade_all_expansion(self):
        s = PathSpec.cascade([3, 5], multipath="all")
        assert s.sender_to == frozenset({3, 5})
        assert s.sender_to_logits

    def test_without_edge(self):
        s = PathSpec.cascade([1, 2, 3]).without_edge(1, 3)
        assert (1, 3) not in s.receiver_to_receiver
        assert (1, 2) in s.receiver_to_receiver

    def test_rejects_downstream_to_upstream_edge(self):
        with pytest.raises(ValueError):
            PathSpec(receivers=(1, 2), receiver_to_receiver=frozenset({(2, 1)}))

    def test_rejects_unknown_receiver(self):
        with pytest.raises(ValueError):
            PathSpec(receivers=(1,), sender_to=frozenset({4}))

    def test_rejects_unsorted_receivers(self):
        with pytest.raises(ValueError):
            PathSpec(receivers=(3, 1))


class TestPaddingShift:
    def test_left_padding_shifts_by_pad_count(self):
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        assert shift_indices_for_padding([[0, 2], [0, 2]], mask) == [[2, 4], [0, 2]]

    def test_right_padding_no_shift(self):
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        assert shift_indices_for_padding([[1]], mask) == [[1]]

    def test_negative_index_is_true_end_relative(self):
        # the documented TokenPosition(lambda x: [-1]) API: -1 must resolve
        # to the last REAL token under either padding side
        left = torch.tensor([[0, 0, 1, 1, 1]])
        right = torch.tensor([[1, 1, 1, 0, 0]])
        assert shift_indices_for_padding([[-1]], left) == [[4]]
        assert shift_indices_for_padding([[-1]], right) == [[2]]

    def test_out_of_range_raises(self):
        mask = torch.tensor([[0, 1, 1]])
        with pytest.raises(IndexError):
            shift_indices_for_padding([[2]], mask)  # true length is 2


def test_coverage_table_matches_committed_artifact():
    """Drift detection: a pyvene pin bump that adds/moves units must surface
    as a diff of the committed coverage artifact, not silently."""
    current = coverage_table()
    committed = json.loads(COVERAGE_ARTIFACT.read_text())
    assert current == committed, (
        "pyvene coverage changed (pin bump?). If intentional, regenerate: "
        'uv run python -c "import json; from causalab.methods.path_patching '
        'import coverage_table; print(json.dumps(coverage_table(), indent=2))" '
        f"> {COVERAGE_ARTIFACT}"
    )


def test_no_raw_torch_hooks_in_method():
    """causalab contract: everything routes through pyvene, no raw hooks."""
    pattern = re.compile(r"register_forward(_pre)?_hook|register_full_backward")
    offenders = [
        f"{p.name}:{i}"
        for p in METHOD_DIR.glob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, f"raw torch hooks in path_patching: {offenders}"


# ---------------------------------------------------------------------------
# tiny-random end-to-end (CPU)
# ---------------------------------------------------------------------------

TINY_VOCAB = 50304  # >= gpt2 tokenizer vocab


def _tiny_model(family: str):
    torch.manual_seed(0)
    if family == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel

        return GPT2LMHeadModel(
            GPT2Config(
                n_layer=4, n_head=4, n_embd=32, vocab_size=TINY_VOCAB, n_positions=128
            )
        )
    if family == "gpt_neox":
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

        return GPTNeoXForCausalLM(
            GPTNeoXConfig(
                num_hidden_layers=4,
                num_attention_heads=4,
                hidden_size=32,
                intermediate_size=64,
                vocab_size=TINY_VOCAB,
                use_parallel_residual=True,
                max_position_embeddings=128,
            )
        )
    if family == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        return LlamaForCausalLM(
            LlamaConfig(
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                hidden_size=32,
                intermediate_size=64,
                vocab_size=TINY_VOCAB,
                max_position_embeddings=128,
            )
        )
    if family == "gemma2":
        from transformers import Gemma2Config, Gemma2ForCausalLM

        return Gemma2ForCausalLM(
            Gemma2Config(
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                hidden_size=32,
                head_dim=8,
                intermediate_size=64,
                vocab_size=TINY_VOCAB,
                final_logit_softcapping=30.0,
                attn_logit_softcapping=50.0,
                max_position_embeddings=128,
                sliding_window=64,
            )
        )
    raise ValueError(family)


@pytest.fixture(scope="module")
def gpt2_tokenizer_pipeline_factory():
    """LMPipeline shells sharing one gpt2 tokenizer; model swapped per test."""
    from causalab.neural.pipeline import LMPipeline

    def make(model):
        pipeline = LMPipeline(
            "gpt2", max_new_tokens=1, load_weights=False, position_ids=True
        )
        model.eval()
        pipeline.model = model
        return pipeline

    return make


TEXTS_CLEAN = [
    "the dog was standing near the door and everyone looked at",
    "yesterday, the king was standing near the door and everyone looked at",
    "a long time ago, the car was standing near the gate and everyone looked at",
    "the bird was sitting near the window and everyone looked at",
]
TEXTS_CF = [
    "the cat was standing near the door and everyone looked at",
    "yesterday, the queen was standing near the door and everyone looked at",
    "a long time ago, the boat was standing near the gate and everyone looked at",
    "the fish was sitting near the window and everyone looked at",
]


def _build_engine(family, factory, **kwargs):
    pipeline = factory(_tiny_model(family))
    desc = resolve_descriptor(pipeline.model, **kwargs)
    clean = build_patch_cache(pipeline, desc, TEXTS_CLEAN, {"end": -1}, batch_size=4)
    cf = build_patch_cache(pipeline, desc, TEXTS_CF, {"end": -1}, batch_size=4)
    return pipeline, desc, clean, cf


@pytest.mark.parametrize("family", ["gpt2", "gpt_neox", "llama", "gemma2"])
class TestTinyRandomEndToEnd:
    def test_guards_pass_and_provenance_reported(
        self, family, gpt2_tokenizer_pipeline_factory
    ):
        _, desc, clean, cf = _build_engine(family, gpt2_tokenizer_pipeline_factory)
        engine = PatchEngine(desc, clean, cf)
        assert engine.guard_report["G3_patch_nothing_max_logit_err"] <= 2e-3
        assert engine.guard_report["G4_patch_everything_max_logit_err"] <= 2e-3
        mechanisms = {p["mechanism"] for p in engine.provenance}
        assert mechanisms <= {"pyvene-named", "pyvene-path", "direct-module-call"}

    def test_cascade_shorthand_bit_identical(
        self, family, gpt2_tokenizer_pipeline_factory
    ):
        _, desc, clean, cf = _build_engine(family, gpt2_tokenizer_pipeline_factory)
        engine = PatchEngine(desc, clean, cf)
        recv = [2, 3]
        short = PathSpec.cascade(recv, direct_to_logits=False)
        explicit = PathSpec(
            receivers=(2, 3),
            sender_to=frozenset({2}),
            receiver_to_receiver=frozenset({(2, 3)}),
            sender_to_logits=False,
            receivers_to_logits=frozenset({2, 3}),
        )
        sender = ("mlp", 0)
        assert torch.equal(
            engine.patched_logits(sender, short),
            engine.patched_logits(sender, explicit),
        )

    def test_twin_agreement(self, family, gpt2_tokenizer_pipeline_factory):
        pipeline, desc, clean, cf = _build_engine(
            family, gpt2_tokenizer_pipeline_factory
        )
        engine = PatchEngine(desc, clean, cf)
        cases = [
            (("head", 1, 2), PathSpec.cascade()),
            (("mlp", 0), PathSpec.cascade([2, 3], direct_to_logits=False)),
            (
                ("head", 0, 1),
                PathSpec.cascade([2, 3], direct_to_logits=False).without_edge(2, 3),
            ),
        ]
        for sender, spec in cases:
            analytic = engine.patched_logits(sender, spec)
            ref = reference_patched_logits(
                pipeline, desc, TEXTS_CLEAN, clean, cf, sender, spec, engine=engine
            )
            assert (analytic - ref).abs().max().item() <= 2e-3, (
                sender,
                spec.describe(),
            )


def test_misdeclared_block_order_fails_branch_wiring_guard(
    gpt2_tokenizer_pipeline_factory,
):
    """A parallel-residual model declared sequential must be refused by G2."""
    _, desc, clean, cf = _build_engine(
        "gpt_neox", gpt2_tokenizer_pipeline_factory, block_order="sequential"
    )
    with pytest.raises(GuardError, match="G2 branch wiring"):
        PatchEngine(desc, clean, cf)


def test_post_ln_trunk_fails_additivity_guard(gpt2_tokenizer_pipeline_factory):
    """A block that norms after its residual add must be refused by G1."""
    import types

    pipeline = gpt2_tokenizer_pipeline_factory(_tiny_model("gpt2"))
    block = pipeline.model.transformer.h[2]
    block._original_forward = block.forward

    def post_ln_forward(self, hidden_states, *args, **kwargs):
        out = self._original_forward(hidden_states, *args, **kwargs)
        return (
            (self.ln_1(out[0]),) + out[1:] if isinstance(out, tuple) else self.ln_1(out)
        )

    block.forward = types.MethodType(post_ln_forward, block)
    desc = resolve_descriptor(pipeline.model)
    clean = build_patch_cache(pipeline, desc, TEXTS_CLEAN, {"end": -1})
    cf = build_patch_cache(pipeline, desc, TEXTS_CF, {"end": -1})
    with pytest.raises(GuardError, match="G1 additivity"):
        PatchEngine(desc, clean, cf)


def test_unmapped_family_is_refused():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained("openai-gpt")
    model = AutoModelForCausalLM.from_config(cfg)
    with pytest.raises(UnsupportedArchitectureError):
        resolve_descriptor(model)


# ---------------------------------------------------------------------------
# K/V-side patching (tiny-random, CPU)
# ---------------------------------------------------------------------------

from causalab.methods.path_patching import (  # noqa: E402
    KVEdge,
    KVHead,
    KVPatchEngine,
    SlidingWindowError,
    build_attn_detail_cache,
)

KV_FAMILIES = ["gpt2", "llama", "gemma2"]  # gpt_neox: refused (no pyvene units)
PATCH_POS = 3  # unpadded index of the K/V-patched position in the fixtures


def _build_kv_engine(family, factory, model=None):
    pipeline = factory(model if model is not None else _tiny_model(family))
    # config-built tiny models default to sdpa; the K/V surface requires the
    # eager contract (its refusal has its own test below)
    pipeline.model.config._attn_implementation = "eager"
    desc = resolve_descriptor(pipeline.model)
    positions = {"end": -1, "mid": PATCH_POS}
    clean = build_patch_cache(pipeline, desc, TEXTS_CLEAN, positions, batch_size=4)
    cf = build_patch_cache(pipeline, desc, TEXTS_CF, positions, batch_size=4)
    engine_end = PatchEngine(desc, clean, cf, position="end")
    engine_mid = PatchEngine(desc, clean, cf, position="mid", run_guards=False)
    det_clean = build_attn_detail_cache(pipeline, desc, TEXTS_CLEAN, positions)
    det_cf = build_attn_detail_cache(pipeline, desc, TEXTS_CF, positions)
    kv = KVPatchEngine(engine_end, engine_mid, det_clean, det_cf, position_patch="mid")
    return pipeline, desc, clean, cf, engine_end, kv


class TestKVHeadMapping:
    def test_query_heads_of_kv_partition(self):
        model = _tiny_model("llama")  # 4 query heads, 2 kv heads
        desc = resolve_descriptor(model)
        assert desc.kv_group_size == 2
        assert desc.query_heads_of_kv(0) == [0, 1]
        assert desc.query_heads_of_kv(1) == [2, 3]
        assert KVHead.for_query_head(desc, 1, 3) == KVHead(1, 1)

    def test_mha_is_group_of_one(self):
        desc = resolve_descriptor(_tiny_model("gpt2"))
        assert desc.kv_group_size == 1
        assert desc.query_heads_of_kv(5) == [5]


@pytest.mark.parametrize("family", KV_FAMILIES)
class TestKVEndToEnd:
    def test_k1_guard_and_zero_delta_closure(
        self, family, gpt2_tokenizer_pipeline_factory
    ):
        _, desc, clean, cf, engine_end, kv = _build_kv_engine(
            family, gpt2_tokenizer_pipeline_factory
        )
        assert kv.guard_report["K1_z_reconstruction_clean"] <= 1e-4
        # zero residual delta -> exactly zero trunk delta (any dtype)
        edges = [KVEdge(KVHead(1, 0), patch_k=True, patch_v=True)]
        zero = torch.zeros(clean.n_examples, desc.d_model)
        d = kv.kv_trunk_delta(edges, zero)
        assert torch.equal(d, torch.zeros_like(d))
        logits = engine_end.patched_logits(d, PathSpec.cascade())
        err = (logits - clean.logits["end"]).abs().max().item()
        assert err <= 2e-3

    def test_substitution_twin_agreement(self, family, gpt2_tokenizer_pipeline_factory):
        pipeline, desc, clean, cf, engine_end, kv = _build_kv_engine(
            family, gpt2_tokenizer_pipeline_factory
        )
        det_cf = kv.detail["cf"]
        bidx = torch.arange(det_cf.n_examples)
        p_cf = torch.tensor(det_cf.positions["mid"])
        cases = [
            KVEdge(KVHead(1, 0), patch_v=True),
            KVEdge(KVHead(2, desc.n_kv_heads - 1), patch_k=True, patch_v=False),
            KVEdge(KVHead(2, 0), patch_k=True, patch_v=True),
        ]
        for edge in cases:
            analytic = engine_end.patched_logits(
                kv.kv_trunk_delta([edge]), PathSpec.cascade()
            )
            L, j = edge.kv.layer, edge.kv.kv_index
            payload = {
                "k": det_cf.k_all[bidx, L, j, p_cf] if edge.patch_k else None,
                "v": det_cf.v_all[bidx, L, j, p_cf] if edge.patch_v else None,
                "position_index": PATCH_POS,
            }
            ref = reference_patched_logits(
                pipeline,
                desc,
                TEXTS_CLEAN,
                clean,
                cf,
                ("kv", L, j, payload),
                PathSpec.cascade(),
                engine=engine_end,
            )
            err = (analytic - ref).abs().max().item()
            assert err <= 2e-3, (family, edge, err)

    def test_rotation_negative_control(self, family, gpt2_tokenizer_pipeline_factory):
        """A deliberately unrotated key delta must produce a different
        trunk delta than the correct rotation on rotary families (and the
        identical one on GPT-2, where rotation is identity). The comparison
        is at the trunk-delta level: on tiny random models the downstream
        logit effect of a single key patch can sit below any absolute
        logit tolerance, which would make a logits-level control vacuous."""
        _, desc, clean, cf, engine_end, kv = _build_kv_engine(
            family, gpt2_tokenizer_pipeline_factory
        )
        edge = KVEdge(KVHead(2, 0), patch_k=True, patch_v=False)
        d_correct = kv.kv_trunk_delta([edge])
        d_wrong = kv.kv_trunk_delta([edge], rotate_key_delta=False)
        scale = d_correct.abs().max().clamp_min(1e-12)
        rel = ((d_correct - d_wrong).abs().max() / scale).item()
        assert d_correct.abs().max() > 0, "key patch had no effect at all"
        if desc.attention_style == "fused-qkv-absolute":
            assert rel <= 1e-5, "rotation is identity on GPT-2; paths must agree"
        else:
            assert rel > 1e-2, (
                "unrotated key delta matched the rotated one; the rotation "
                "term has no teeth in this configuration"
            )

    def test_eager_contract_refused(self, family, gpt2_tokenizer_pipeline_factory):
        pipeline = gpt2_tokenizer_pipeline_factory(_tiny_model(family))
        pipeline.model.config._attn_implementation = "sdpa"
        desc = resolve_descriptor(pipeline.model)
        with pytest.raises(RuntimeError, match="attn_implementation"):
            build_attn_detail_cache(
                pipeline, desc, TEXTS_CLEAN, {"end": -1, "mid": PATCH_POS}
            )


def test_kv_refused_on_gpt_neox(gpt2_tokenizer_pipeline_factory):
    """gpt_neox has no pyvene q/k/v units (fused per-head-interleaved QKV):
    the K/V surface must refuse it by name, never fall back to hooks."""
    pipeline = gpt2_tokenizer_pipeline_factory(_tiny_model("gpt_neox"))
    desc = resolve_descriptor(pipeline.model)
    with pytest.raises(UnsupportedArchitectureError, match="key_output"):
        build_attn_detail_cache(
            pipeline, desc, TEXTS_CLEAN, {"end": -1, "mid": PATCH_POS}
        )


def test_kv_fanout_is_confined_to_the_group(gpt2_tokenizer_pipeline_factory):
    """Patching one KV head's value changes the reconstructed z of exactly
    its query-head group (verified structurally on cached activations)."""
    _, desc, clean, cf, engine_end, kv = _build_kv_engine(
        "llama", gpt2_tokenizer_pipeline_factory
    )
    det = kv.detail["clean"]
    layer, j = 1, 0
    v_patched = det.v_all[:, layer].clone()
    bidx = torch.arange(det.n_examples)
    p = torch.tensor(det.positions["mid"])
    v_patched[bidx, j, p] = v_patched[bidx, j, p] + 1.0
    z_base = kv._z_from_qkv(
        layer,
        det.q["end"][:, layer],
        det.k_all[:, layer],
        det.v_all[:, layer],
        "clean",
    )
    z_new = kv._z_from_qkv(
        layer, det.q["end"][:, layer], det.k_all[:, layer], v_patched, "clean"
    )
    dz = (z_new - z_base).abs().amax(dim=(0, 2))  # per query head
    group = set(desc.query_heads_of_kv(j))
    for h in range(desc.n_heads):
        if h in group:
            assert dz[h] > 0, f"group head {h} unaffected"
        else:
            assert dz[h] == 0, f"non-group head {h} affected"


def test_kv_sliding_window_refusal(gpt2_tokenizer_pipeline_factory):
    """Gemma-2 sliding layers must refuse a patch position outside the
    window and accept one inside; full-attention layers always accept."""
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(0)
    model = Gemma2ForCausalLM(
        Gemma2Config(
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=32,
            head_dim=8,
            intermediate_size=64,
            vocab_size=TINY_VOCAB,
            final_logit_softcapping=30.0,
            attn_logit_softcapping=50.0,
            max_position_embeddings=128,
            sliding_window=4,
        )
    )
    _, desc, clean, cf, engine_end, kv = _build_kv_engine(
        "gemma2", gpt2_tokenizer_pipeline_factory, model=model
    )
    sliding = [li for li in range(4) if desc.attn_sliding_window(li) is not None]
    full = [li for li in range(4) if desc.attn_sliding_window(li) is None]
    assert sliding and full, "tiny config must mix sliding and full layers"
    # the fixtures end >= 10 tokens after PATCH_POS=3, window is 4 -> refuse
    with pytest.raises(SlidingWindowError, match=f"layer {sliding[0]}"):
        kv.kv_trunk_delta([KVEdge(KVHead(sliding[0], 0))])
    # a full-attention layer accepts the same edge
    d = kv.kv_trunk_delta([KVEdge(KVHead(full[0], 0))])
    assert torch.isfinite(d).all()


# ---------------------------------------------------------------------------
# multi-position collect: the shipped keep_last_dim path is position-correct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["gpt2", "llama"])
class TestMultiPositionCollect:
    """Regression guard for the keep_last_dim fix (verified empirically against
    pinned pyvene 896a7dd4): without it, pyvene's flatten/slice cycle keeps only
    the first d values of a multi-position gather under causalab's single-unit
    wrapper, so a two-position collect would return position 1's vector split
    across both position slots. These tests pin the *fixed* path: collects run
    through ``keep_last_dim_on_collects`` must match a plain forward hook at
    every position."""

    @staticmethod
    def _wrapper_collect(model, batch, positions):
        from types import SimpleNamespace

        from causalab.methods.path_patching.cache import keep_last_dim_on_collects
        from causalab.neural.activations import (
            delete_intervenable_model,
            prepare_intervenable_model,
        )
        from causalab.neural.units import AtomicModelUnit, ComponentIndexer

        unit = AtomicModelUnit(
            0,
            "block_output",
            ComponentIndexer(lambda _x: [], id="explicit"),
            id="collect_block0",
        )
        im = prepare_intervenable_model(
            SimpleNamespace(model=model), [unit], intervention_type="collect"
        )
        keep_last_dim_on_collects(im)
        try:
            indices = [positions]
            with torch.no_grad():
                result = im(
                    batch,
                    unit_locations={"sources->base": (indices, indices)},
                    output_original_output=True,
                )
            return result[0][1][0].detach().clone()
        finally:
            delete_intervenable_model(im)

    @staticmethod
    def _ground_truth(model, batch, positions):
        block = (
            model.transformer.h[0]
            if hasattr(model, "transformer")
            else model.model.layers[0]
        )
        captured = {}

        def hook(mod, args, out):
            captured["h"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

        handle = block.register_forward_hook(hook)
        with torch.no_grad():
            model(**batch)
        handle.remove()
        idx = torch.tensor(positions)
        return torch.stack([captured["h"][i, idx[i]] for i in range(idx.shape[0])])

    @staticmethod
    def _batch():
        torch.manual_seed(7)
        return {
            "input_ids": torch.randint(1, 257, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }

    def test_two_position_collect_matches_forward_hook(self, family):
        model = _tiny_model(family)
        model.eval()
        batch = self._batch()
        positions = [[2, 5], [3, 6]]
        gt = self._ground_truth(model, batch, positions)
        collected = self._wrapper_collect(model, batch, positions)
        assert tuple(collected.shape) == tuple(gt.shape)
        assert torch.equal(collected, gt)

    def test_single_position_collect_matches_forward_hook(self, family):
        model = _tiny_model(family)
        model.eval()
        batch = self._batch()
        positions = [[2], [3]]
        gt = self._ground_truth(model, batch, positions)
        collected = self._wrapper_collect(model, batch, positions)
        assert torch.equal(collected.reshape(gt.shape), gt)
