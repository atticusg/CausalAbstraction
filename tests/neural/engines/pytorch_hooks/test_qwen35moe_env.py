"""The environment PR0 buys: the Qwen3.5-MoE architecture must actually load.

Everything in the hookpoint-vocabulary work targets ``qwen3_5_moe`` (the text
tower of Qwen3.6-35B-A3B), which only exists from transformers 5.16. These tests
are the gate on that bump: they assert the architecture is importable, that the
engine's own loader reaches the text tower rather than the composite
vision-language model, and that the layer stack really is hybrid — because the
per-layer split is the assumption every later PR in the stack builds on.

They deliberately assert *structure*, not activations: numerical behaviour is
pinned by the parity goldens, which this bump leaves untouched.
"""

from __future__ import annotations

import pytest

from .conftest import TINY_QWEN35_MOE

#: structural assertions on a tiny CPU model — no numerics, no GPU
pytestmark = pytest.mark.smoke


def test_transformers_ships_the_architecture():
    """``qwen3_5_moe`` is absent before transformers 5.16 — fail loudly if pinned back."""
    pytest.importorskip(
        "transformers.models.qwen3_5_moe",
        reason="transformers < 5.16 has no qwen3_5_moe; the lock must not slip back",
    )
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

    assert hasattr(modeling_qwen3_5_moe, "Qwen3_5MoeForCausalLM")


def test_the_config_offers_a_text_tower_beside_the_vlm():
    """The reason ``AutoModelForCausalLM`` is the right entry point.

    ``qwen3_5_moe`` is registered with *both* auto classes. That is what lets the
    engine load a plain causal LM and ignore the vision tower entirely — and it
    is why nnsight's ``LanguageModel``, which refuses anything registered for
    image-text-to-text, cannot load this checkpoint without help.
    """
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    )

    assert "qwen3_5_moe" in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    assert "qwen3_5_moe" in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def test_loader_reaches_the_text_tower_not_the_vlm(qwen35moe_bundle):
    """``load_model`` must give us the causal LM, with no vision tower attached."""
    model = qwen35moe_bundle.model
    assert type(model).__name__ == "Qwen3_5MoeForCausalLM"
    assert not hasattr(model, "visual"), "the vision tower must not be loaded"
    assert hasattr(model, "lm_head")
    assert hasattr(model.model, "layers")


def test_the_layer_stack_is_hybrid(qwen35moe_bundle):
    """The assumption the whole vocabulary plan rests on: block type is per-layer.

    A DeltaNet layer carries ``linear_attn`` and no ``self_attn``; a full-attention
    layer carries the reverse. Any site resolver that reads ``block.self_attn``
    unconditionally is wrong on this model, and this test is what says so.
    """
    model = qwen35moe_bundle.model
    layer_types = getattr(model.config, "layer_types", None)
    assert layer_types is not None, "config must expose layer_types"
    assert set(layer_types) == {"linear_attention", "full_attention"}, layer_types

    layers = model.model.layers
    assert len(layers) == len(layer_types)
    for idx, (block, block_type) in enumerate(zip(layers, layer_types)):
        if block_type == "linear_attention":
            assert hasattr(block, "linear_attn"), idx
            assert not hasattr(block, "self_attn"), idx
        else:
            assert hasattr(block, "self_attn"), idx
            assert not hasattr(block, "linear_attn"), idx


def test_every_layer_has_a_sparse_moe_block(qwen35moe_bundle):
    """Unlike the token mixer, the channel mixer is uniform across the stack."""
    for idx, block in enumerate(qwen35moe_bundle.model.model.layers):
        mlp = block.mlp
        assert type(mlp).__name__ == "Qwen3_5MoeSparseMoeBlock", idx
        # the four sub-taps the round-1 MoE components resolve against
        for attr in ("gate", "experts", "shared_expert", "shared_expert_gate"):
            assert hasattr(mlp, attr), (idx, attr)


def test_model_info_unwraps_the_composite_config(qwen35moe_bundle):
    """The composite config nests text fields; ``ModelInfo`` must see through it."""
    info = qwen35moe_bundle.info
    assert info.hidden_size > 0
    assert info.num_experts is not None and info.num_experts > 1, (
        "router_logits width comes from num_experts; a composite config must not "
        "hide it"
    )


def test_fixture_key_is_the_documented_one():
    """Guards against the fixture silently drifting to another checkpoint."""
    assert TINY_QWEN35_MOE == "tiny-random/qwen3.5-moe"
