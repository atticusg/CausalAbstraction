"""Fixtures for the hook-oracle suites (GH #380).

The oracle tests are equivalence/invariance contracts that must hold across model
families, not just one architecture. ``oracle_pipeline`` parametrises every suite
over three tiny-random CPU pipelines that span the structural axes causalab
supports:

* ``llama`` — Llama-family with separate Q/K/V/O projections (RoPE);
* ``gpt2`` — GPT-2 with a *fused* ``c_attn`` QKV and absolute position embeddings
  (a genuinely different module tree and tokenizer);
* ``gqa`` — a grouped-query Llama (``n_kv == n_heads/2``), the structural class
  the coherent ``chat-coherent`` (Qwen3) backbone also belongs to.

``tiny_pipeline`` / ``tiny_gpt2_pipeline`` come from ``tests/neural/conftest.py``.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.pipeline import LMPipeline
from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME


@pytest.fixture(scope="session")
def gqa_pipeline() -> LMPipeline:
    """A grouped-query tiny-random Llama (``n_kv == n_heads/2``), CPU."""
    from transformers import AutoConfig, LlamaForCausalLM

    config = AutoConfig.from_pretrained(TINY_RANDOM_MODEL_NAME)
    assert config.num_attention_heads % 2 == 0
    config.num_key_value_heads = config.num_attention_heads // 2
    # Seed the random init. Without this the weights depend on whatever RNG
    # state exists when this session-scoped fixture is first built — i.e. on
    # which tests ran before it — and any oracle asserting "the intervention
    # changed the logits by more than atol" becomes intermittently vacuous.
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)  # random init — CPU equivalence only
    model.eval()
    return LMPipeline(model_or_name=model, max_new_tokens=1, padding_side="left")


@pytest.fixture(params=["llama", "gpt2", "gqa"])
def oracle_pipeline(
    request,
    tiny_pipeline: LMPipeline,
    tiny_gpt2_pipeline: LMPipeline,
    gqa_pipeline: LMPipeline,
) -> LMPipeline:
    """A tiny-random pipeline, parametrised across model families. Hook-oracle
    tests take this instead of a single ``tiny_pipeline`` so each assertion runs
    on Llama, GPT-2, and grouped-query Llama."""
    return {
        "llama": tiny_pipeline,
        "gpt2": tiny_gpt2_pipeline,
        "gqa": gqa_pipeline,
    }[request.param]
