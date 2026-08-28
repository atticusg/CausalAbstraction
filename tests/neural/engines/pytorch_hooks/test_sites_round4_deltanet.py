"""Round-4 Gated DeltaNet interior — tier 1: the mixer's module boundaries.

📐 Three of the DeltaNet diagram's boxes are ordinary ``nn.Module`` sides on
``tiny-random/qwen3.5-moe`` (transformers 5.16): ``in_proj_qkv``'s output is
the fused ``[q | k | v]`` projection (widths 128/128/256 — **unequal**, so no
head axis), ``in_proj_z``'s output is the output gate (8 v-heads × 32), and
``out_proj``'s **input** is the post-norm, post-gate mixer value — the exact
analogue of ``attention_premix``, which is why the name. The ``conv1d`` module
never fires (the forward calls the ``causal_conv1d_fn`` module global instead),
which is why the conv output and the kernel boundary are *function* taps
(round 4.2), not module taps.

The stream refusals mirror round 1's ``_FULL_ATTENTION_ONLY`` in the other
direction: a full-attention layer computes no delta-rule state, and a family
with no linear stream anywhere (llama, GPT-2) hits the same refusal at every
layer.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.errors import ProtocolError
from causalab.protocol.registry import component_shape
from causalab.protocol.schema import SiteSpec

from ._drive import base_data_section, executor_for
from .conftest import TINY_GPT2, TINY_LLAMA

pytestmark = pytest.mark.smoke

DELTANET_LAYER = 0
FULL_ATTENTION_LAYER = 3
TEXT = "the quick brown fox jumps"
CF_TEXT = "a slow green turtle sleeps"

TIER_ONE = ("delta_qkv", "delta_gate", "delta_premix")

#: 📐 fixture numbers: 8 v-heads, 4 k-heads (2× GVA tiling), head dims 32 —
#: key_dim 128, value_dim 256, conv_dim 512.
TIER_ONE_WIDTH = {"delta_qkv": 512, "delta_gate": 256, "delta_premix": 256}

#: which module side each component taps, for the resolve test
TIER_ONE_TAP = {
    "delta_qkv": ("in_proj_qkv", "out"),
    "delta_gate": ("in_proj_z", "out"),
    "delta_premix": ("out_proj", "in"),
}


def _read_doc(
    component: str, layer: int = DELTANET_LAYER, head: int | None = None
) -> dict:
    tap: dict = {"component": component, "layer": layer}
    if head is not None:
        tap["head"] = head
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "sites": {"tap": tap},
        "reads": {
            "r": {"site": "tap", "pos": "all", "model": "original", "input": "base"}
        },
        "save": [
            {
                "value": "r",
                "model": "original",
                "input": "base",
                "file_path": "a.safetensors",
            }
        ],
    }


def _write_doc(component: str, do: dict, *, layer: int = DELTANET_LAYER) -> dict:
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "sites": {
            "tap": {"component": component, "layer": layer},
            "lm_head": {"component": "lm_head"},
        },
        "reads": {
            "v_cf": {
                "site": "tap",
                "pos": "all",
                "model": "original",
                "input": "counterfactual",
            },
            "clean": {
                "site": "lm_head",
                "pos": {"index": -1},
                "model": "original",
                "input": "base",
            },
            "after": {
                "site": "lm_head",
                "pos": {"index": -1},
                "model": "patched",
                "input": "base",
            },
        },
        "writes": {"patch": {"site": "tap", "pos": "all", "do": do}},
        "intervened_models": {"patched": {"input": "base", "writes": ["patch"]}},
        "save": [
            {
                "value": "after",
                "model": "patched",
                "input": "base",
                "file_path": "p.safetensors",
            },
            {
                "value": "clean",
                "model": "original",
                "input": "base",
                "file_path": "c.safetensors",
            },
        ],
    }


def _moved(bundle: ModelBundle, doc: dict, **kw) -> float:
    executor = executor_for(
        doc, bundle, base_texts=[TEXT], counterfactual_texts=[CF_TEXT], **kw
    )
    after, clean = executor.read_value("after"), executor.read_value("clean")
    return float((after - clean).abs().max())


# --------------------------------------------------------------------------- #
# the taps resolve, at the module sides the forward really uses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", TIER_ONE)
def test_the_tap_is_the_declared_module_side(qwen35moe_bundle, component: str):
    site = resolve_site(
        qwen35moe_bundle, SiteSpec(component=component, layer=DELTANET_LAYER)
    )
    child, side = TIER_ONE_TAP[component]
    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    assert site.module is getattr(mixer, child)
    assert site.kind == side


@pytest.mark.parametrize("component", TIER_ONE)
def test_the_read_has_the_measured_width(qwen35moe_bundle, component: str):
    value = executor_for(
        _read_doc(component), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    assert tuple(value.shape) == (1, 5, TIER_ONE_WIDTH[component])


def test_the_conv1d_module_never_fires(qwen35moe_bundle):
    """📐 The premise of round 4.2's function taps, pinned where round 4.1 can
    see it: the forward calls the ``causal_conv1d_fn`` module global, so a hook
    on the ``conv1d`` module reads nothing — a module tap there would be the
    silent-empty-read failure."""
    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    fired: list[bool] = []
    handle = mixer.conv1d.register_forward_hook(lambda *_: fired.append(True))
    encoded = qwen35moe_bundle.tokenizer(TEXT, return_tensors="pt")
    try:
        with torch.no_grad():
            qwen35moe_bundle.model(**encoded)
    finally:
        handle.remove()
    assert fired == []


# --------------------------------------------------------------------------- #
# identity pins (tier 1)
# --------------------------------------------------------------------------- #


def test_the_gate_is_the_z_projection_of_the_norm_exactly(qwen35moe_bundle):
    """§4's tier-1 pin: ``delta_gate == in_proj_z(attention_input_norm)`` at
    exactly 0.0 — the tap is where it claims to be, on the tensor the mixer
    actually consumes."""
    doc = _read_doc("delta_gate")
    doc["sites"]["norm"] = {
        "component": "attention_input_norm",
        "layer": DELTANET_LAYER,
    }
    doc["reads"]["n"] = {
        "site": "norm",
        "pos": "all",
        "model": "original",
        "input": "base",
    }
    doc["save"].append(
        {
            "value": "n",
            "model": "original",
            "input": "base",
            "file_path": "n.safetensors",
        }
    )
    executor = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT])
    gate, norm_out = executor.read_value("r"), executor.read_value("n")
    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    with torch.no_grad():
        reference = mixer.in_proj_z(norm_out)
    torch.testing.assert_close(gate, reference, atol=0.0, rtol=0.0)


def test_the_qkv_projection_is_the_same_identity_one_module_over(qwen35moe_bundle):
    doc = _read_doc("delta_qkv")
    doc["sites"]["norm"] = {
        "component": "attention_input_norm",
        "layer": DELTANET_LAYER,
    }
    doc["reads"]["n"] = {
        "site": "norm",
        "pos": "all",
        "model": "original",
        "input": "base",
    }
    doc["save"].append(
        {
            "value": "n",
            "model": "original",
            "input": "base",
            "file_path": "n.safetensors",
        }
    )
    executor = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT])
    qkv, norm_out = executor.read_value("r"), executor.read_value("n")
    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    with torch.no_grad():
        reference = mixer.in_proj_qkv(norm_out)
    torch.testing.assert_close(qkv, reference, atol=0.0, rtol=0.0)


def test_the_premix_projects_to_the_mixer_output_exactly(qwen35moe_bundle):
    """The premix is ``out_proj``'s input, so ``out_proj(delta_premix)`` must
    be the mixer's output — the analogue of the ``attention_premix`` identity."""
    doc = _read_doc("delta_premix")
    doc["sites"]["out"] = {"component": "attention_output", "layer": DELTANET_LAYER}
    doc["reads"]["o"] = {
        "site": "out",
        "pos": "all",
        "model": "original",
        "input": "base",
    }
    doc["save"].append(
        {
            "value": "o",
            "model": "original",
            "input": "base",
            "file_path": "o.safetensors",
        }
    )
    executor = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT])
    premix, mixer_out = executor.read_value("r"), executor.read_value("o")
    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    with torch.no_grad():
        reference = mixer.out_proj(premix)
    torch.testing.assert_close(mixer_out, reference, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", TIER_ONE)
def test_a_swap_through_the_tap_moves_the_logits(qwen35moe_bundle, component: str):
    assert _moved(qwen35moe_bundle, _write_doc(component, {"swap": "v_cf"})) > 1e-4


@pytest.mark.parametrize("component", TIER_ONE)
def test_swapping_a_tap_with_its_own_value_moves_nothing(
    qwen35moe_bundle, component: str
):
    doc = _write_doc(component, {"swap": "v_cf"})
    doc["reads"]["v_cf"]["input"] = "base"
    assert _moved(qwen35moe_bundle, doc) == 0.0, component


# --------------------------------------------------------------------------- #
# head bounds
# --------------------------------------------------------------------------- #


def test_the_gate_and_premix_are_value_head_space(qwen35moe_bundle):
    info = qwen35moe_bundle.info
    assert info.linear_num_value_heads == 8
    assert component_shape(info, "delta_gate").head_space == 8
    assert component_shape(info, "delta_premix").head_space == 8
    assert component_shape(info, "delta_qkv").head_space is None


def test_a_head_selects_one_v_head_of_the_gate(qwen35moe_bundle):
    value = executor_for(
        _read_doc("delta_gate", head=3), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    assert tuple(value.shape) == (1, 5, 32)


def test_head_on_the_fused_qkv_is_refused_because_its_widths_are_unequal(
    qwen35moe_bundle,
):
    """The fused ``[q | k | v]`` splits are 128/128/256 — no equal per-head
    packing exists, so ``head`` cannot mean a slice of it. The refusal's note
    names the per-head faces (the round-4.2 kernel-boundary components)."""
    with pytest.raises(ProtocolError, match="no head axis") as excinfo:
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component="delta_qkv", layer=DELTANET_LAYER, head=0),
        )
    assert "delta_query" in str(excinfo.value)


def test_an_out_of_range_v_head_is_refused(qwen35moe_bundle):
    with pytest.raises(ProtocolError, match="which has 8 heads"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component="delta_gate", layer=DELTANET_LAYER, head=8),
        )


# --------------------------------------------------------------------------- #
# stream and family refusals — the `_LINEAR_ATTENTION_ONLY` mirror
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", TIER_ONE)
def test_a_full_attention_layer_refuses_with_the_architectural_reason(
    qwen35moe_bundle, component: str
):
    with pytest.raises(ProtocolError, match="delta-rule state"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component=component, layer=FULL_ATTENTION_LAYER),
        )


@pytest.mark.parametrize("key", [TINY_LLAMA, TINY_GPT2])
@pytest.mark.parametrize("component", TIER_ONE)
def test_a_family_with_no_linear_stream_refuses_at_every_layer(
    key: str, component: str
):
    """llama and gpt2 carry full attention everywhere, so the same per-layer
    refusal is the architectural one: the tower listing in the message shows a
    stream this family never has."""
    bundle = load_model(key)
    with pytest.raises(ProtocolError, match="delta-rule state"):
        resolve_site(bundle, SiteSpec(component=component, layer=1))


def test_a_declared_stream_still_wins_over_the_component_check(qwen35moe_bundle):
    """`stream: full_attention` on a delta component at a linear layer refuses
    on the *stream* mismatch first — the declaration is checked before the
    component's needs, so the message names the per-layer fact."""
    with pytest.raises(ProtocolError, match="per-layer fact"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(
                component="delta_gate",
                layer=DELTANET_LAYER,
                stream="full_attention",
            ),
        )


# --------------------------------------------------------------------------- #
# generated frames
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", TIER_ONE)
def test_a_continuation_read_accumulates_one_row_per_step(
    qwen35moe_bundle, component: str
):
    """All three are query-position-shaped (one row per token), so decode steps
    stack — nothing here is indexed by the growing prefix."""
    doc = _read_doc(component)
    doc["positions"] = {"window": {"generated": {"max_new_tokens": 3}, "all": True}}
    doc["reads"]["r"]["pos"] = "window"
    value = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")
    assert tuple(value.shape) == (1, 3, TIER_ONE_WIDTH[component])
