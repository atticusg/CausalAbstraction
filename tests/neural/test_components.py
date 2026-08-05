"""Component resolution: does ``resolve_site`` point at the right activation?

Every component type must resolve to the activation its name promises, on every
module-tree family causalab supports, or every downstream method (collect /
interchange / steer / ablate / path-patch) silently moves.

Ground truth is raw ``register_forward_hook`` / ``register_forward_pre_hook`` on
the module named, with the same reshaping applied by hand — no engine, no
nnsight. So these tests are not satisfied by both paths being wrong in the same
way, and they outlived the backbone they were written under.

Two things the pyvene backbone got wrong are asserted here as *correct*:

* per-head widths follow ``config.head_dim`` (#386), and
* k/v heads are addressed in KV space under grouped-query attention,

so the ``gqa`` fixture — decoupled ``head_dim`` + GQA — is a model on which
causalab used to raise ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from causalab.neural.components import (
    UnsupportedComponent,
    component_width,
    head_layout,
    resolve_site,
)
from causalab.neural.pipeline import LMPipeline
from tests._helpers.tiny import (
    TINY_RANDOM_MODEL_NAME,
    tiny_random_gpt2_model,
    tiny_random_model,
)

pytestmark = pytest.mark.unit

PROMPT = "the quick brown fox jumps"
LAYER = 0

# Every component type both families expose, in the causalab vocabulary.
COMPONENTS = [
    "block_input",
    "block_output",
    "mlp_input",
    "mlp_output",
    "mlp_activation",
    "attention_input",
    "attention_output",
    "attention_value_output",
    "head_attention_value_output",
    "query_output",
    "key_output",
    "value_output",
    "head_query_output",
    "head_key_output",
    "head_value_output",
]


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _gqa_model() -> Any:
    """A Llama whose ``head_dim`` is decoupled from ``hidden // n_head``, with GQA.

    ``head_dim=8`` while ``hidden_size // num_attention_heads == 4``, and
    ``num_key_value_heads=2 < num_attention_heads=4``. The pyvene backbone realized every
    per-head component at the 4-wide ratio, so on this model it returns
    wrong-width vectors — which is why causalab refused it outright (#386).
    """
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config).eval()


@pytest.fixture(scope="module")
def pipelines() -> dict[str, LMPipeline]:
    return {
        "llama": LMPipeline(tiny_random_model(), max_new_tokens=1, padding_side="left"),
        "gpt2": LMPipeline(
            tiny_random_gpt2_model(), max_new_tokens=1, padding_side="left"
        ),
        # Built in-process, so it has no `name_or_path` to resolve a tokenizer
        # from; it borrows the tiny-Llama one (same vocab size).
        "gqa": LMPipeline(
            _gqa_model(),
            max_new_tokens=1,
            padding_side="left",
            tokenizer=AutoTokenizer.from_pretrained(TINY_RANDOM_MODEL_NAME),
        ),
    }


# --------------------------------------------------------------------------- #
#  Oracle — the named module, hooked by hand                                   #
# --------------------------------------------------------------------------- #
def _is_gpt2(model: Any) -> bool:
    return hasattr(model, "transformer") and hasattr(model.transformer, "h")


def _oracle_module_and_kind(model: Any, component: str) -> tuple[Any, str]:
    """``(module, 'in'|'out')`` — the anchor point for ``component``."""
    if _is_gpt2(model):
        block = model.transformer.h[LAYER]
        table = {
            "block_input": (block, "in"),
            "block_output": (block, "out"),
            "mlp_input": (block.mlp, "in"),
            "mlp_output": (block.mlp, "out"),
            "mlp_activation": (block.mlp.act, "out"),
            "attention_input": (block.attn, "in"),
            "attention_output": (block.attn.resid_dropout, "out"),
            "attention_value_output": (block.attn.c_proj, "in"),
            "head_attention_value_output": (block.attn.c_proj, "in"),
        }
        if component in table:
            return table[component]
        return (block.attn.c_attn, "out")  # every q/k/v variant
    block = model.model.layers[LAYER]
    table = {
        "block_input": (block, "in"),
        "block_output": (block, "out"),
        "mlp_input": (block.mlp, "in"),
        "mlp_output": (block.mlp, "out"),
        "mlp_activation": (block.mlp.act_fn, "out"),
        "attention_input": (block.self_attn, "in"),
        "attention_output": (block.self_attn, "out"),
        "attention_value_output": (block.self_attn.o_proj, "in"),
        "head_attention_value_output": (block.self_attn.o_proj, "in"),
        "query_output": (block.self_attn.q_proj, "out"),
        "key_output": (block.self_attn.k_proj, "out"),
        "value_output": (block.self_attn.v_proj, "out"),
        "head_query_output": (block.self_attn.q_proj, "out"),
        "head_key_output": (block.self_attn.k_proj, "out"),
        "head_value_output": (block.self_attn.v_proj, "out"),
    }
    return table[component]


def _first_tensor(value: Any) -> torch.Tensor:
    """The tensor a hooked value carries (attention blocks return tuples)."""
    return value[0] if isinstance(value, tuple) else value


def _oracle_activation(pipeline: LMPipeline, component: str, enc: dict) -> torch.Tensor:
    """The component's activation, captured with a raw hook and reshaped by hand."""
    model = pipeline.model
    module, kind = _oracle_module_and_kind(model, component)
    grabbed: dict[str, torch.Tensor] = {}

    def pre_hook(_m: Any, args: Any, kwargs: Any) -> None:
        # "The module's input" is its first positional arg, or — for a module
        # called with keyword-only args, which `LlamaAttention` is in
        # transformers 4.57 — its first keyword arg. Same rule nnsight's
        # `.input` follows.
        value = args[0] if args else next(iter(kwargs.values()))
        grabbed["v"] = _first_tensor(value).detach().clone()

    def post_hook(_m: Any, _args: Any, out: Any) -> None:
        grabbed["v"] = _first_tensor(out).detach().clone()

    handle = (
        module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        if kind == "in"
        else module.register_forward_hook(post_hook)
    )
    try:
        with torch.no_grad():
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    finally:
        handle.remove()
    value = grabbed["v"]

    n_q, n_kv, head_dim = head_layout(model.config)
    # GPT-2 fuses q|k|v into one c_attn output; take the right third. Matched by
    # exact name — `attention_value_output` is the o_proj input, not a QKV third.
    thirds = {
        "query_output": 0,
        "head_query_output": 0,
        "key_output": 1,
        "head_key_output": 1,
        "value_output": 2,
        "head_value_output": 2,
    }
    if _is_gpt2(model) and component in thirds:
        widths = (n_q * head_dim, n_kv * head_dim, n_kv * head_dim)
        third = thirds[component]
        start = sum(widths[:third])
        value = value[..., start : start + widths[third]]
    # Per-head components carry a head axis at the model's real head width.
    if component.startswith("head_"):
        n_heads = n_kv if component in ("head_key_output", "head_value_output") else n_q
        value = value.unflatten(-1, (n_heads, head_dim))
    return value


# --------------------------------------------------------------------------- #
#  Tests                                                                       #
# --------------------------------------------------------------------------- #
class TestResolveSiteReads:
    """``Site.read()`` must return the activation the component name promises."""

    @pytest.mark.parametrize("family", ["llama", "gpt2", "gqa"])
    @pytest.mark.parametrize("component", COMPONENTS)
    def test_read_matches_hook_oracle(
        self, pipelines: dict[str, LMPipeline], family: str, component: str
    ) -> None:
        pipeline = pipelines[family]
        enc = pipeline.load([_trace(PROMPT)])
        expected = _oracle_activation(pipeline, component, enc)

        with pipeline.nnsight.trace(PROMPT):
            site = resolve_site(pipeline.nnsight, component, LAYER)
            got = site.read().save()

        assert tuple(got.shape) == tuple(expected.shape), (
            f"{family}/{component}: shape {tuple(got.shape)} != oracle "
            f"{tuple(expected.shape)}"
        )
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("family", ["llama", "gpt2", "gqa"])
    def test_head_components_use_config_head_dim(
        self, pipelines: dict[str, LMPipeline], family: str
    ) -> None:
        """The head axis is ``config.head_dim`` wide, not ``hidden // n_head``.

        On the ``gqa`` fixture those two differ (8 vs 4); the pyvene backbone used the ratio,
        which is the silent wrong-width read behind #386.
        """
        pipeline = pipelines[family]
        config = pipeline.model.config
        n_q, n_kv, head_dim = head_layout(config)

        with pipeline.nnsight.trace(PROMPT):
            q = resolve_site(pipeline.nnsight, "head_query_output", LAYER).read().save()
            v = resolve_site(pipeline.nnsight, "head_value_output", LAYER).read().save()

        assert q.shape[-2:] == (n_q, head_dim)
        assert v.shape[-2:] == (n_kv, head_dim)
        assert component_width(config, "head_query_output") == head_dim


class TestResolveSiteWrites:
    """``Site.write()`` must land exactly where ``read()`` looked."""

    @pytest.mark.parametrize("family", ["llama", "gpt2", "gqa"])
    @pytest.mark.parametrize(
        "component",
        [
            "block_output",
            "mlp_output",
            "attention_value_output",
            "head_attention_value_output",
            "head_value_output",
        ],
    )
    def test_write_round_trips_through_read(
        self, pipelines: dict[str, LMPipeline], family: str, component: str
    ) -> None:
        """Writing a value then reading the same site back yields that value."""
        pipeline = pipelines[family]
        with pipeline.nnsight.trace(PROMPT):
            site = resolve_site(pipeline.nnsight, component, LAYER)
            original = site.read()
            replacement = torch.full_like(original, 0.5)
            site.write(replacement)
            after = site.read().save()
        torch.testing.assert_close(after, torch.full_like(after, 0.5))

    @pytest.mark.parametrize("family", ["llama", "gpt2", "gqa"])
    def test_write_one_head_leaves_others_untouched(
        self, pipelines: dict[str, LMPipeline], family: str
    ) -> None:
        """Zeroing head H changes H's slice and nothing else — the property that
        makes per-head ablation mean what it claims."""
        pipeline = pipelines[family]
        n_q, _n_kv, _head_dim = head_layout(pipeline.model.config)
        head = n_q - 1

        with pipeline.nnsight.trace(PROMPT):
            site = resolve_site(pipeline.nnsight, "head_attention_value_output", LAYER)
            before = site.read().clone().save()
            edited = site.read().clone()
            edited[:, :, head, :] = 0
            site.write(edited)
            after = site.read().clone().save()

        assert not torch.allclose(before[:, :, head, :], after[:, :, head, :]), (
            "the ablated head did not change — the write did not land"
        )
        assert torch.count_nonzero(after[:, :, head, :]) == 0
        other = [h for h in range(n_q) if h != head]
        torch.testing.assert_close(after[:, :, other, :], before[:, :, other, :])

    @pytest.mark.parametrize("family", ["llama", "gpt2", "gqa"])
    def test_head_write_changes_logits(
        self, pipelines: dict[str, LMPipeline], family: str
    ) -> None:
        """A per-head write must actually reach the model's output.

        A write that round-trips through ``read`` but never reaches the forward
        would pass the test above and still be useless.
        """
        pipeline = pipelines[family]
        n_q, _n_kv, _hd = head_layout(pipeline.model.config)

        with pipeline.nnsight.trace(PROMPT):
            clean = pipeline.nnsight.output.logits.clone().save()

        with pipeline.nnsight.trace(PROMPT):
            site = resolve_site(pipeline.nnsight, "head_attention_value_output", LAYER)
            edited = site.read().clone()
            edited[:, :, n_q - 1, :] = 0
            site.write(edited)
            ablated = pipeline.nnsight.output.logits.clone().save()

        assert float((clean - ablated).abs().max().detach()) > 1e-6


class TestUnsupported:
    def test_head_attention_value_input_is_rejected_with_a_reason(
        self, pipelines: dict[str, LMPipeline]
    ) -> None:
        """`AttentionHead(target_output=False)` names a read point that does not
        exist. Fail with an explanation — and name the component that *does*
        work — rather than resolving to something plausible-but-wrong."""
        with pytest.raises(UnsupportedComponent, match="head_attention_value_output"):
            resolve_site(pipelines["llama"].nnsight, "head_attention_value_input", 0)

    def test_unknown_component_lists_the_known_ones(
        self, pipelines: dict[str, LMPipeline]
    ) -> None:
        with pytest.raises(UnsupportedComponent, match="Unknown component type"):
            resolve_site(pipelines["llama"].nnsight, "not_a_component", 0)


def _trace(text: str):
    from causalab.causal.trace import CausalTrace, Mechanism

    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )
