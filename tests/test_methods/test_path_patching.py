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
