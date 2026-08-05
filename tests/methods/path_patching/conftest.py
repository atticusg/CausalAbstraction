"""Fixtures shared by the path-patching method tests.

``gqa_tiny_lm`` is a grouped-query-attention variant of the tiny-random Llama
(``num_key_value_heads < num_attention_heads``), needed because the default tiny
stub is multi-head (n_kv == n_head) and so cannot exercise the query→KV-group
value-receiver mapping. Random weights, CPU — smoke-only, like ``mock_tiny_lm``.
"""

from __future__ import annotations

import pytest

from causalab.neural.pipeline import LMPipeline
from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME, tiny_random_gpt2_model


@pytest.fixture(scope="module")
def tiny_gpt2_lm() -> LMPipeline:
    """Left-padded :class:`LMPipeline` over the random-weight GPT-2 stub — an
    **absolute-position** (learned ``wpe``) model.

    The default ``mock_tiny_lm`` is RoPE (relative positions), immune to a uniform
    left-pad shift, so it cannot surface position_ids-under-padding bugs. GPT-2's
    ``wpe`` makes a left-padded row's activations sensitive to ``position_ids``, so
    this fixture is what pins the path-patching left-pad fix numerically (bs=N vs
    bs=1 parity). ``max_new_tokens=1`` matches the path-patching single-step scoring.
    Random weights, CPU — only for the position contract, not behaviour.
    """
    return LMPipeline(
        model_or_name=tiny_random_gpt2_model(), max_new_tokens=1, padding_side="left"
    )


@pytest.fixture(scope="module")
def gqa_tiny_lm() -> LMPipeline:
    from transformers import AutoConfig, LlamaForCausalLM

    config = AutoConfig.from_pretrained(TINY_RANDOM_MODEL_NAME)
    # Make it grouped-query: fewer KV heads than query heads (group size 2). The
    # tokenizer is loaded from config.name_or_path (set by from_pretrained), so the
    # model object can be handed straight to LMPipeline.
    assert config.num_attention_heads % 2 == 0
    config.num_key_value_heads = config.num_attention_heads // 2
    model = LlamaForCausalLM(config)  # random init — smoke-only
    model.eval()
    return LMPipeline(model_or_name=model, max_new_tokens=3)


@pytest.fixture(scope="module")
def gqa_decoupled_head_dim_lm() -> LMPipeline:
    """A grouped-query tiny Llama whose ``head_dim`` is *decoupled* from
    ``hidden_size // num_attention_heads``.

    The default tiny stub (and ``gqa_tiny_lm``) keep ``head_dim == hidden // n_head``,
    so the value-receiver's ``shape=(head_dim,)`` slice and the per-KV-head
    ``head_value_output`` split are only ever exercised in the coupled regime. Modern
    decoder-only models (e.g. Qwen3) set ``head_dim`` independently, so this fixture
    pins that the head-value path uses ``config.head_dim`` — not ``hidden // n_head`` —
    end to end. Random weights, CPU — smoke-only, like ``gqa_tiny_lm``.
    """
    from transformers import AutoConfig, LlamaForCausalLM

    config = AutoConfig.from_pretrained(TINY_RANDOM_MODEL_NAME)
    assert config.num_attention_heads % 2 == 0
    config.num_key_value_heads = config.num_attention_heads // 2
    # Decouple: head_dim != hidden // n_head (default would be 16 // 4 == 4).
    config.head_dim = (config.hidden_size // config.num_attention_heads) + 2
    assert config.head_dim != config.hidden_size // config.num_attention_heads
    model = LlamaForCausalLM(config)  # random init — smoke-only
    model.eval()
    return LMPipeline(model_or_name=model, max_new_tokens=3)
