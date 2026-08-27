"""Round-1 block components: the four new taps, and per-layer stream dispatch.

PR2 of the hookpoint-vocabulary stack. Four components join the vocabulary —
``input_ids``, ``attention_input_norm``, ``block_mid``, ``mlp_input_norm`` — and
``attention_output`` learns that the mixer is a *per-layer* fact.

The shape assertions here are measurements against a real ``qwen3_5_moe``
checkpoint, so a mismatch is a finding rather than a stale expectation. The
identity assertions are stronger than shapes: they pin that each tap names the
tensor its *name* claims, by reproducing the decoder layer's residual algebra
from the taps alone.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.pytorch_hooks.layout import (
    LayoutError,
    from_contract,
    tap_tensor,
    to_contract,
)
from causalab.neural.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.pytorch_hooks.sites import resolve_site
from causalab.protocol.errors import ProtocolError, ValidationError
from causalab.protocol.plan import COMPONENT_RANK
from causalab.protocol.registry import component_width
from causalab.protocol.schema import COMPONENTS, LAYERLESS_COMPONENTS, SiteSpec

from .conftest import TINY_LLAMA, TINY_QWEN35_MOE

pytestmark = pytest.mark.smoke

#: The four new components, plus the one whose resolution changed.
NEW_COMPONENTS = ("input_ids", "attention_input_norm", "block_mid", "mlp_input_norm")
PR2_COMPONENTS = NEW_COMPONENTS + ("attention_output",)

TEXT = "the quick brown fox jumps"


def _spec(component: str, layer: int | None = None, **kw: object) -> SiteSpec:
    if component in LAYERLESS_COMPONENTS:
        return SiteSpec(component=component, **kw)
    return SiteSpec(component=component, layer=0 if layer is None else layer, **kw)


def _capture(
    bundle: ModelBundle, specs: dict[str, SiteSpec]
) -> dict[str, torch.Tensor]:
    """Read many sites in one forward pass, each in contract shape."""
    encoded = bundle.tokenizer(TEXT, return_tensors="pt")
    batch = encoded["input_ids"].shape[0]
    out: dict[str, torch.Tensor] = {}
    handles = []
    for name, spec in specs.items():
        site = resolve_site(bundle, spec)

        def hook(_mod, inp, output=None, *, _name=name, _site=site):
            payload = inp if output is None else output
            if output is None and isinstance(payload, tuple) and len(payload) == 1:
                payload = payload[0]
            tensor = tap_tensor(payload, _site.tuple_index)
            out[_name] = (
                to_contract(tensor, _site.layout, batch_size=batch).detach().clone()
            )

        handles.append(
            site.module.register_forward_pre_hook(hook)
            if site.kind == "in"
            else site.module.register_forward_hook(hook)
        )
    try:
        with torch.no_grad():
            bundle.model(**encoded)
    finally:
        for handle in handles:
            handle.remove()
    return out


# --------------------------------------------------------------------------- #
# the vocabulary itself
# --------------------------------------------------------------------------- #


def test_the_four_components_joined_the_closed_vocabulary():
    for component in NEW_COMPONENTS:
        assert component in COMPONENTS, f"{component} missing from Component"


def test_input_ids_is_layerless_and_the_others_are_not():
    """``input_ids`` is the model's *input* (§5.4) — there is no layer at which
    to read it, and naming one is a parse error."""
    assert "input_ids" in LAYERLESS_COMPONENTS
    for component in ("attention_input_norm", "block_mid", "mlp_input_norm"):
        assert component not in LAYERLESS_COMPONENTS


def test_the_new_ranks_order_the_block_as_the_forward_pass_runs():
    """Group elision (§4) picks a group's deepest tap, so the ranks must follow
    the decoder layer's actual order — the same order the identities below
    reproduce."""
    order = [
        "input_ids",
        "embeddings",
        "block_input",
        "attention_input_norm",
        "attention_output",
        "block_mid",
        "mlp_input_norm",
        "mlp_input",
        "mlp_output",
        "block_output",
    ]
    ranks = [COMPONENT_RANK[c] for c in order]
    assert ranks == sorted(ranks), dict(zip(order, ranks))


# --------------------------------------------------------------------------- #
# widths: three are residual-shaped, one is not a feature space at all
# --------------------------------------------------------------------------- #


def test_the_three_norm_taps_are_residual_width(qwen35moe_bundle):
    hidden = qwen35moe_bundle.info.hidden_size
    for component in ("attention_input_norm", "block_mid", "mlp_input_norm"):
        assert component_width(qwen35moe_bundle.info, component) == hidden


def test_input_ids_refuses_a_width_because_it_is_not_a_feature_space(qwen35moe_bundle):
    """§5.4: integer ids on a position axis. No featurizer can attach, and the
    refusal must say *why* rather than read as a missing table entry."""
    with pytest.raises(ValidationError) as excinfo:
        component_width(qwen35moe_bundle.info, "input_ids")
    assert "not a feature space" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# §5.2 — the mixer is a per-layer fact
# --------------------------------------------------------------------------- #


def test_the_fixture_tower_really_is_hybrid(qwen35moe_bundle):
    """📐 The premise every assertion below rests on."""
    assert qwen35moe_bundle.streams == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )


def test_attention_output_follows_the_stream_per_layer(qwen35moe_bundle):
    """The §5.2 fix. ``block.self_attn`` for every non-GPT-2 model raised
    AttributeError on 3 of these 4 layers — the tap now asks the layer."""
    for layer, stream in enumerate(qwen35moe_bundle.streams):
        site = resolve_site(qwen35moe_bundle, _spec("attention_output", layer))
        block = qwen35moe_bundle.blocks[layer]
        if stream == "full_attention":
            assert site.module is block.self_attn
        else:
            assert site.module is block.linear_attn
            # and the regression this replaces: there is no self_attn to reach
            assert not hasattr(block, "self_attn")


def test_a_site_naming_the_wrong_stream_refuses(qwen35moe_bundle):
    """``stream`` has parsed since schema.py gained it and nothing read it
    (§5.2). It reads it now, and a contradiction is an error."""
    with pytest.raises(ProtocolError) as excinfo:
        resolve_site(
            qwen35moe_bundle,
            _spec("attention_output", 0, stream="full_attention"),
        )
    assert "layer 0" in str(excinfo.value)

    with pytest.raises(ProtocolError):
        resolve_site(
            qwen35moe_bundle,
            _spec("attention_output", 3, stream="linear_attention"),
        )


def test_a_site_naming_the_right_stream_resolves(qwen35moe_bundle):
    for layer, stream in enumerate(qwen35moe_bundle.streams):
        site = resolve_site(
            qwen35moe_bundle, _spec("attention_output", layer, stream=stream)
        )
        assert site.module is qwen35moe_bundle.mixer_at(layer)


# --------------------------------------------------------------------------- #
# §5.3 — refuse for the permanent reason before the temporary one
# --------------------------------------------------------------------------- #


def test_attention_probs_at_a_deltanet_layer_refuses_on_the_architecture(
    qwen35moe_bundle,
):
    """A Gated DeltaNet block computes no attention matrix, so this is not a
    missing feature — it stays false after PR4 lands ``attention_probs``."""
    with pytest.raises(ProtocolError) as excinfo:
        resolve_site(qwen35moe_bundle, _spec("attention_probs", 0))
    message = str(excinfo.value)
    assert "full-attention mixer" in message
    assert "linear_attention" in message


def test_attention_probs_at_a_full_attention_layer_is_merely_unimplemented(
    qwen35moe_bundle,
):
    """The other half of the ordering: at layer 3 the tensor exists, and the
    refusal is a roadmap statement (PR4), not an architectural one."""
    with pytest.raises(NotImplementedError):
        resolve_site(qwen35moe_bundle, _spec("attention_probs", 3))


# --------------------------------------------------------------------------- #
# the taps name what they claim: the decoder layer's residual algebra
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("layer", [0, 3], ids=["deltanet", "full_attention"])
def test_the_taps_reproduce_the_residual_algebra(qwen35moe_bundle, layer):
    """The sanity check that shapes cannot give.

    A Qwen3.5-MoE decoder layer runs::

        h = input_layernorm(resid_pre)      -> attention_input_norm
        h = mixer(h)                        -> attention_output
        resid_mid = resid_pre + h           -> block_mid
        h = post_attention_layernorm(resid_mid) -> mlp_input_norm
        h = mlp(h)                          -> mlp_output
        resid_post = resid_mid + h          -> block_output

    So the three new taps are pinned by identities against taps that already
    existed. If ``block_mid`` were accidentally the layernorm's *output* rather
    than its input (the one-character mistake available here), the second
    identity fails.
    """
    block = qwen35moe_bundle.blocks[layer]
    got = _capture(
        qwen35moe_bundle,
        {
            name: _spec(name, layer)
            for name in (
                "block_input",
                "attention_input_norm",
                "attention_output",
                "block_mid",
                "mlp_input_norm",
                "mlp_output",
                "block_output",
            )
        },
    )

    torch.testing.assert_close(
        got["attention_input_norm"], block.input_layernorm(got["block_input"])
    )
    torch.testing.assert_close(
        got["block_mid"], got["block_input"] + got["attention_output"]
    )
    torch.testing.assert_close(
        got["mlp_input_norm"], block.post_attention_layernorm(got["block_mid"])
    )
    torch.testing.assert_close(
        got["block_output"], got["block_mid"] + got["mlp_output"]
    )


@pytest.mark.parametrize("layer", [0, 3], ids=["deltanet", "full_attention"])
def test_every_new_block_tap_is_residual_shaped(qwen35moe_bundle, layer):
    """📐 §3 panel 2: every box in the DecoderLayer panel is ``(1, 6, 8)`` on
    this fixture — 6 positions of hidden size 8."""
    got = _capture(
        qwen35moe_bundle,
        {
            n: _spec(n, layer)
            for n in ("attention_input_norm", "block_mid", "mlp_input_norm")
        },
    )
    hidden = qwen35moe_bundle.info.hidden_size
    for name, tensor in got.items():
        assert tensor.dim() == 3, name
        assert tensor.shape[0] == 1 and tensor.shape[-1] == hidden, (name, tensor.shape)


def test_the_three_norm_taps_are_three_different_tensors(qwen35moe_bundle):
    """Two of them share a module and differ only by side, so a copy-paste slip
    would make them alias — the failure a shape assertion cannot see."""
    got = _capture(
        qwen35moe_bundle,
        {
            n: _spec(n, 0)
            for n in ("attention_input_norm", "block_mid", "mlp_input_norm")
        },
    )
    values = list(got.values())
    for i, first in enumerate(values):
        for second in values[i + 1 :]:
            assert not torch.equal(first, second)


# --------------------------------------------------------------------------- #
# input_ids
# --------------------------------------------------------------------------- #


def test_input_ids_reads_the_ids_the_tokenizer_produced(qwen35moe_bundle):
    encoded = qwen35moe_bundle.tokenizer(TEXT, return_tensors="pt")
    got = _capture(qwen35moe_bundle, {"input_ids": _spec("input_ids")})["input_ids"]
    # contract shape: the degenerate feature axis of width 1
    assert got.shape == (*encoded["input_ids"].shape, 1)
    assert not got.dtype.is_floating_point, "token ids are integers"
    torch.testing.assert_close(got.squeeze(-1), encoded["input_ids"])


def test_input_ids_taps_the_embedding_input_on_both_families():
    """The ids cross exactly one module boundary, and it differs by family."""
    for key, attr in (
        (TINY_QWEN35_MOE, "embed_tokens"),
        (TINY_LLAMA, "embed_tokens"),
    ):
        bundle = load_model(key)
        site = resolve_site(bundle, _spec("input_ids"))
        assert site.module is getattr(bundle.model.model, attr)
        assert site.kind == "in"
        assert site.layout == "bs"


# --------------------------------------------------------------------------- #
# the "bs" layout
# --------------------------------------------------------------------------- #


def test_bs_layout_round_trips_and_returns_a_view():
    """§6.2 review point 2: ``to_contract`` must return a view, so an in-place
    edit reaches native storage. ``unsqueeze`` does; ``reshape`` might not."""
    native = torch.arange(12).reshape(3, 4)
    contract = to_contract(native, "bs", batch_size=3)
    assert contract.shape == (3, 4, 1)
    assert contract._base is native or contract.data_ptr() == native.data_ptr()
    torch.testing.assert_close(from_contract(contract, "bs", batch_size=3), native)


@pytest.mark.parametrize(
    "bad", [torch.zeros(3), torch.zeros(2, 3, 4)], ids=["1d", "3d"]
)
def test_bs_layout_refuses_a_shape_it_cannot_mean(bad):
    """A declared layout that contradicts the tensor raises rather than
    reinterpreting — the wrong-tap-with-plausible-numbers failure."""
    with pytest.raises(LayoutError):
        to_contract(bad, "bs", batch_size=2)


# --------------------------------------------------------------------------- #
# family parity: the gate says all five read on tiny-llama too
# --------------------------------------------------------------------------- #


def test_all_five_components_resolve_on_a_non_hybrid_family(llama_bundle):
    assert set(llama_bundle.streams) == {"full_attention"}
    for component in PR2_COMPONENTS:
        site = resolve_site(llama_bundle, _spec(component))
        assert site.module is not None


def test_all_five_components_read_on_a_non_hybrid_family(llama_bundle):
    got = _capture(llama_bundle, {n: _spec(n) for n in PR2_COMPONENTS})
    assert set(got) == set(PR2_COMPONENTS)
    hidden = llama_bundle.info.hidden_size
    for name, tensor in got.items():
        width = 1 if name == "input_ids" else hidden
        assert tensor.shape[-1] == width, (name, tensor.shape)


def test_the_llama_residual_algebra_holds_too(llama_bundle):
    """The identities are architecture-independent — same algebra, other family."""
    block = llama_bundle.blocks[0]
    got = _capture(
        llama_bundle,
        {
            n: _spec(n, 0)
            for n in (
                "block_input",
                "attention_input_norm",
                "attention_output",
                "block_mid",
                "mlp_input_norm",
                "mlp_output",
                "block_output",
            )
        },
    )
    torch.testing.assert_close(
        got["attention_input_norm"], block.input_layernorm(got["block_input"])
    )
    torch.testing.assert_close(
        got["block_mid"], got["block_input"] + got["attention_output"]
    )
    torch.testing.assert_close(
        got["mlp_input_norm"], block.post_attention_layernorm(got["block_mid"])
    )
