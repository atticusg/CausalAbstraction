"""Round-2 attention *function* interior: q, k, the scores, and z.

These four are not module boundaries. ``transformers`` computes them inside one
``attention_interface(...)`` call, so a ``register_forward_hook`` on the mixer
fires after they have been consumed — which is why writing the attention pattern
was already a special case in round 1.

The scores are the interesting one, and the reason this round is smaller than
the plan expected. #53 reached the pattern by calling the real eager function and
then **redoing** the two lines after its softmax — correct, but a transcription
of library internals that has to resolve a per-family ``eager_attention_forward``
to perform. A ``TorchFunctionMode`` scoped to the real call reaches the scores
with nothing transcribed at all:

📐 measured on all three CI fixtures, transformers 5.16 —

    llama  calls 1  scores (1,4,8,8)   identity_maxdiff 0.0  softmax(s)==p 0.0
    gpt2   calls 1  scores (1,4,14,14) identity_maxdiff 0.0  softmax(s)==p 0.0
    qwen   calls 1  scores (1,8,5,5)   identity_maxdiff 0.0  softmax(s)==p 0.0

and knocking one head off one token moved the qwen logits by 0.3114.

The capability that follows is the point of the round: ``attention_probs``
accepts only ``swap``, because a delta leaves rows that no longer sum to 1 and
nothing renormalizes them. One step earlier, the model's own softmax does.
Attention knockout, head boosting and every other arithmetic mechanism are legal
on the scores and produce a valid pattern by construction.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.attention_interface import (
    InterfaceTap,
    attention_interface_taps,
)
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.errors import ProtocolError
from causalab.protocol.registry import component_shape
from causalab.protocol.schema import SiteSpec

from ._drive import base_data_section, bundle_loader, executor_for
from .conftest import TINY_GPT2, TINY_LLAMA

pytestmark = pytest.mark.smoke

FULL_ATTENTION_LAYER = 3
DELTANET_LAYER = 0
LLAMA_LAYER = 1

TEXT = "the quick brown fox jumps"
CF_TEXT = "a slow green turtle sleeps"

FUNCTION_INTERIOR = (
    "attention_query",
    "attention_key",
    "attention_scores",
    "attention_z",
)

#: 📐 measured on `tiny-random/qwen3.5-moe` at 5 tokens, layer 3:
#: ``component -> (native shape at the interface, contract shape)``.
#: ``attention_scores`` has no contract form, so it is read whole.
QWEN_INTERFACE: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "attention_query": ((1, 8, 5, 32), (1, 5, 256)),
    "attention_key": ((1, 4, 5, 32), (1, 5, 128)),
    "attention_scores": ((1, 8, 5, 5), (1, 8, 5, 5)),
    "attention_z": ((1, 5, 8, 32), (1, 5, 256)),
}


def _read_doc(component: str, layer: int = FULL_ATTENTION_LAYER) -> dict:
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "sites": {"tap": {"component": component, "layer": layer}},
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


def _write_doc(component: str, do: dict, *, layer: int = FULL_ATTENTION_LAYER) -> dict:
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
# the taps resolve, and read the shapes the interface really has
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", FUNCTION_INTERIOR)
def test_the_tap_is_an_interface_slot_not_a_module_side(
    qwen35moe_bundle, component: str
):
    """There is no module boundary here, and the site says so rather than
    pointing at a module whose hook would fire too late."""
    site = resolve_site(
        qwen35moe_bundle, SiteSpec(component=component, layer=FULL_ATTENTION_LAYER)
    )
    assert site.kind == "interface"
    assert site.interface_slot is not None
    # the mixer is still carried: it is what says *which* call to tap
    assert site.module is qwen35moe_bundle.mixer_at(FULL_ATTENTION_LAYER)


@pytest.mark.parametrize("component", FUNCTION_INTERIOR)
def test_the_read_has_the_shape_the_interface_produces(
    qwen35moe_bundle, component: str
):
    _, contract = QWEN_INTERFACE[component]
    executor = executor_for(_read_doc(component), qwen35moe_bundle, base_texts=[TEXT])
    assert tuple(executor.read_value("r").shape) == contract


def test_the_key_is_read_before_repeat_kv(qwen35moe_bundle):
    """📐 The interface's second argument is ``(b, H_kv, s, d)`` — 4 KV heads on
    this fixture, not 8 query heads. Reading it after ``repeat_kv`` would give
    the same numbers duplicated, which is a different tensor wearing the same
    name."""
    info = qwen35moe_bundle.info
    assert info.num_kv_heads * 2 == info.num_heads
    executor = executor_for(
        _read_doc("attention_key"), qwen35moe_bundle, base_texts=[TEXT]
    )
    width = executor.read_value("attention_key" and "r").shape[-1]
    assert width == info.num_kv_heads * info.head_dim
    assert width != info.num_heads * info.head_dim


def test_the_scores_softmax_to_the_pattern(qwen35moe_bundle):
    """The identity that says the scores tap is where it claims to be: 📐
    ``softmax(attention_scores) == attention_probs``, maxdiff 0.0.

    It is also the check that the ``TorchFunctionMode`` intercepted the right
    call — a mode that had tapped some *other* softmax would produce a tensor
    of the right shape and the wrong values.
    """
    scores = executor_for(
        _read_doc("attention_scores"), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    probs = executor_for(
        _read_doc("attention_probs"), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    torch.testing.assert_close(
        torch.softmax(scores.float(), dim=-1), probs.float(), atol=0.0, rtol=0.0
    )


def test_reading_the_interior_does_not_change_the_model(qwen35moe_bundle):
    """Observe-only must be bit-identical. The mode is process-global while
    entered and the wrapper replaces the attention function, so "reading changed
    the answer" is a live failure mode rather than a hypothetical one."""
    encoded = qwen35moe_bundle.tokenizer(TEXT, return_tensors="pt")
    with torch.no_grad():
        clean = qwen35moe_bundle.model(**encoded).logits.clone()
    for component in FUNCTION_INTERIOR:
        executor = executor_for(
            _read_doc(component), qwen35moe_bundle, base_texts=[TEXT]
        )
        executor.read_value("r")
        with torch.no_grad():
            after = qwen35moe_bundle.model(**encoded).logits.clone()
        assert torch.equal(after, clean), component


# --------------------------------------------------------------------------- #
# writes — and the capability the scores exist for
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", FUNCTION_INTERIOR)
def test_a_swap_through_the_interface_moves_the_logits(
    qwen35moe_bundle, component: str
):
    assert _moved(qwen35moe_bundle, _write_doc(component, {"swap": "v_cf"})) > 1e-4


@pytest.mark.parametrize("component", FUNCTION_INTERIOR)
def test_swapping_a_tap_with_its_own_value_moves_nothing(
    qwen35moe_bundle, component: str
):
    """The non-vacuity half: an edit that substitutes the same tensor must be
    exactly the identity, or the write is landing somewhere it should not."""
    doc = _write_doc(component, {"swap": "v_cf"})
    doc["reads"]["v_cf"]["input"] = "base"
    assert _moved(qwen35moe_bundle, doc) == 0.0, component


def test_attention_knockout_is_expressible_on_the_scores(qwen35moe_bundle):
    """The headline capability, end to end.

    📐 Head 0 blocked from attending to token 0, as a full-shape ``-1e4`` mask
    added to the scores. The model's own softmax runs *after* the edit, so the
    rows it produces still sum to 1 by construction — nothing here has to know
    that, which is the point.
    """
    mask = torch.zeros(1, 8, 5, 5)
    mask[:, 0, :, 0] = -1e4
    doc = _write_doc("attention_scores", {"add_scaled": {"op": "knock", "alpha": 1.0}})
    doc["params"] = {"knock": {"file_path": "k.safetensors"}}
    del doc["reads"]["v_cf"]
    assert (
        _moved(
            qwen35moe_bundle,
            doc,
            load_tensors=bundle_loader({"k.safetensors": {"value": mask}}),
        )
        > 1e-3
    )


def test_a_uniform_shift_of_the_scores_is_a_no_op_because_softmax_is_shift_invariant(
    qwen35moe_bundle,
):
    """Worth pinning rather than discovering: adding the *same* constant to
    every score changes no probability at all, because softmax is invariant to a
    shift along the axis it normalizes. A knockout therefore has to be targeted,
    which is why the recipe above uses a full-shape mask rather than a scalar.
    """
    doc = _write_doc("attention_scores", {"add_scaled": {"op": -10000.0, "alpha": 1.0}})
    del doc["reads"]["v_cf"]
    assert _moved(qwen35moe_bundle, doc) < 1e-3


@pytest.mark.parametrize(
    "do",
    [
        {"add_scaled": {"op": -1.0, "alpha": 1.0}},
        {"clamp": {"lo": -1.0, "hi": 1.0}},
        {"lerp": {"op": 0.0, "alpha": 0.5}},
    ],
    ids=["delta", "clamp", "lerp"],
)
def test_every_arithmetic_mechanism_is_legal_on_the_scores(qwen35moe_bundle, do: dict):
    """The difference between the scores and the pattern, stated as a test.

    Both have the same axes. Only one is followed by a softmax, and that is the
    whole reason one accepts arithmetic and the other does not.
    """
    doc = _write_doc("attention_scores", do)
    del doc["reads"]["v_cf"]
    executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("after")


@pytest.mark.parametrize(
    "do",
    [
        {"add_scaled": {"op": -1.0, "alpha": 1.0}},
        {"clamp": {"lo": -1.0, "hi": 1.0}},
    ],
    ids=["delta", "clamp"],
)
def test_the_same_mechanism_is_refused_on_the_pattern(qwen35moe_bundle, do: dict):
    doc = _write_doc("attention_probs", do)
    del doc["reads"]["v_cf"]
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("after")
    message = str(excinfo.value)
    assert "sum to 1" in message
    # and the refusal names the component that *does* accept it
    assert "attention_scores" in message


def test_gaussian_is_refused_where_there_is_no_feature_axis(qwen35moe_bundle):
    """📐 The noise is drawn as ``(batch, position, feature)`` and its ``axis``
    names how the feature axis is sharded. On a tap whose last axis is key
    positions the draw does not even fit (measured: "shape '[1, 8, 5, 5]' is
    invalid for input of size 40"), so it is refused rather than reshaped into
    something that would run."""
    doc = _write_doc(
        "attention_scores",
        {"gaussian": {"seed": 0, "scale": 1.0, "axis": "tp_duplicated"}},
    )
    del doc["reads"]["v_cf"]
    with pytest.raises(ProtocolError, match="no feature axis at all"):
        executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("after")


# --------------------------------------------------------------------------- #
# the mode's guards
# --------------------------------------------------------------------------- #


def test_the_mode_taps_functional_softmax_and_nothing_else():
    """Strictness, unit-tested without a model.

    ``torch.softmax`` and ``Tensor.softmax`` reach the same maths by a different
    entry point. Matching them too would make the call count meaningless — and
    counting is what stands between this and tapping the wrong softmax on a
    family that has two.
    """
    from causalab.neural.engines.pytorch_hooks.attention_interface import _SoftmaxTap

    x = torch.randn(2, 3)
    mode = _SoftmaxTap(lambda t: t)
    with mode:
        torch.softmax(x, dim=-1)
        x.softmax(dim=-1)
    assert mode.calls == 0
    with mode:
        torch.nn.functional.softmax(x, dim=-1)
    assert mode.calls == 1


def test_a_family_that_does_not_softmax_exactly_once_is_refused(qwen35moe_bundle):
    """📐 All three CI fixtures call it once. A family that soft-caps, or runs a
    second sliding-window pass, would have *a* softmax tapped and which one
    would depend on source order — the silent-wrong-tensor failure this whole
    effort exists to prevent."""
    from causalab.neural.engines.pytorch_hooks.attention_interface import (
        _check_one_softmax,
    )

    module = qwen35moe_bundle.mixer_at(FULL_ATTENTION_LAYER)
    _check_one_softmax(module, 1)  # the measured case: fine
    for calls in (0, 2):
        with pytest.raises(ProtocolError, match="not once"):
            _check_one_softmax(module, calls)


def test_the_registry_key_is_restored_on_exit(qwen35moe_bundle):
    """The entry is process-global while installed, so leaking it would leave
    every later forward in the process running through our wrapper."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    before = "eager" in ALL_ATTENTION_FUNCTIONS
    module = qwen35moe_bundle.mixer_at(FULL_ATTENTION_LAYER)
    with attention_interface_taps(
        {id(module): (InterfaceTap(slot="query", read=lambda _t: None),)}
    ):
        assert "eager" in ALL_ATTENTION_FUNCTIONS
    assert ("eager" in ALL_ATTENTION_FUNCTIONS) is before


def test_a_tap_at_one_layer_leaves_another_layers_output_alone():
    """The scoping test that can actually fail.

    ⚠️ ``causalab-round1-review-handoff`` §7 flags the round-1 version as
    vacuous: it scoped against a *DeltaNet* layer, which never consults
    ``ALL_ATTENTION_FUNCTIONS`` at all, so it passed even for a wrapper that
    ignored its tap map entirely. 📐 tiny-llama has **two** full-attention
    layers, so this taps layer 1 and asserts layer 0's output is untouched —
    which is exactly what a process-global mode makes urgent.
    """
    bundle = load_model(TINY_LLAMA)
    encoded = bundle.tokenizer(TEXT, return_tensors="pt")
    layer0 = bundle.model.model.layers[0]
    seen: dict[str, torch.Tensor] = {}
    handle = layer0.register_forward_hook(
        lambda _m, _i, out: seen.__setitem__(
            "t", (out[0] if isinstance(out, tuple) else out).detach().clone()
        )
    )
    try:
        with torch.no_grad():
            bundle.model(**encoded)
        clean = seen["t"].clone()

        def wreck(scores: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(scores)

        with attention_interface_taps(
            {id(bundle.mixer_at(1)): (InterfaceTap(slot="scores", edit=wreck),)}
        ):
            with torch.no_grad():
                wrecked_logits = bundle.model(**encoded).logits.clone()
        after = seen["t"]
    finally:
        handle.remove()

    # layer 0 is upstream of the tap and must be untouched...
    assert torch.equal(after, clean)
    # ...and the tap must have done something, or the assertion above is vacuous
    with torch.no_grad():
        base_logits = bundle.model(**encoded).logits
    assert float((wrecked_logits - base_logits).abs().max()) > 1e-4


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", FUNCTION_INTERIOR)
def test_a_deltanet_layer_refuses_with_the_architectural_reason(
    qwen35moe_bundle, component: str
):
    with pytest.raises(ProtocolError, match="full-attention mixer"):
        resolve_site(
            qwen35moe_bundle, SiteSpec(component=component, layer=DELTANET_LAYER)
        )


@pytest.mark.parametrize(
    "component", ["attention_key", "attention_scores", "attention_probs"]
)
def test_a_continuation_read_of_a_key_indexed_tap_is_refused(component: str):
    """📐 A decode step attends over the whole KV cache, so a tensor indexed by
    the positions being attended *to* is ``prompt + step`` long at step
    ``step`` while the query axis stays 1. The steps do not stack.

    Both reasons are read off the declared axes rather than a component list:
    two position axes (the pattern, the scores), or one position axis that runs
    over the keys (``attention_key``).
    """
    bundle = load_model(TINY_LLAMA)
    doc = _read_doc(component, layer=LLAMA_LAYER)
    doc["positions"] = {"window": {"generated": {"max_new_tokens": 3}, "all": True}}
    doc["reads"]["r"]["pos"] = "window"
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(doc, bundle, base_texts=[TEXT]).read_value("r")
    assert "generated frame" in str(excinfo.value)


@pytest.mark.parametrize("component", ["attention_query", "attention_z"])
def test_a_continuation_read_of_a_query_indexed_tap_works(component: str):
    """The other side of the same rule: these two are query-axis-shaped, so one
    row per step is exactly what a continuation frame means."""
    bundle = load_model(TINY_LLAMA)
    doc = _read_doc(component, layer=LLAMA_LAYER)
    doc["positions"] = {"window": {"generated": {"max_new_tokens": 3}, "all": True}}
    doc["reads"]["r"]["pos"] = "window"
    value = executor_for(doc, bundle, base_texts=[TEXT]).read_value("r")
    info = bundle.info
    assert tuple(value.shape) == (1, 3, info.num_heads * info.head_dim)


def test_gpt2_reads_the_function_interior_too():
    """Unlike round 2.2's module-boundary components, these four do **not**
    depend on separate q/k/v projections: the interface receives q, k and v as
    arguments however the mixer produced them. 📐 gpt2's eager also calls
    softmax exactly once, so the scores tap works there as well."""
    bundle = load_model(TINY_GPT2)
    for component in FUNCTION_INTERIOR:
        value = executor_for(
            _read_doc(component, layer=LLAMA_LAYER), bundle, base_texts=[TEXT]
        ).read_value("r")
        assert value.numel() > 0, component


# --------------------------------------------------------------------------- #
# head bounds
# --------------------------------------------------------------------------- #


def test_the_key_is_kv_space_and_the_query_is_not(qwen35moe_bundle):
    info = qwen35moe_bundle.info
    assert component_shape(info, "attention_query").head_space == info.num_heads
    assert component_shape(info, "attention_key").head_space == info.num_kv_heads
    assert component_shape(info, "attention_z").head_space == info.num_heads


def test_a_query_space_head_on_the_key_is_refused(qwen35moe_bundle):
    with pytest.raises(ProtocolError, match="which has 4 heads"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component="attention_key", layer=FULL_ATTENTION_LAYER, head=5),
        )


def test_a_read_of_a_written_slot_matches_the_module_hook_path(qwen35moe_bundle):
    """Reading and writing one site in the same forward must mean the same thing
    whether the site is a module boundary or an interface slot.

    📐 On both paths the read sees the **written** value (difference 0.0),
    because the executor registers edits before reads and hooks fire in
    registration order. Pinned across the two mechanisms rather than within one:
    if they ever diverged, the same document would mean different things
    depending only on which components it happened to name — and nothing else in
    the suite compares them.
    """

    def written_minus_read(component: str) -> float:
        doc = {
            "version": "1",
            "model": {"key": "test", "revision": "main"},
            "data": base_data_section(with_counterfactual=True),
            "sites": {"tap": {"component": component, "layer": FULL_ATTENTION_LAYER}},
            "reads": {
                "src": {
                    "site": "tap",
                    "pos": "all",
                    "model": "original",
                    "input": "counterfactual",
                },
                "obs": {
                    "site": "tap",
                    "pos": "all",
                    "model": "patched",
                    "input": "base",
                },
            },
            "writes": {"p": {"site": "tap", "pos": "all", "do": {"swap": "src"}}},
            "intervened_models": {"patched": {"input": "base", "writes": ["p"]}},
            "save": [
                {
                    "value": "obs",
                    "model": "patched",
                    "input": "base",
                    "file_path": "o.safetensors",
                }
            ],
        }
        executor = executor_for(
            doc,
            qwen35moe_bundle,
            base_texts=[TEXT],
            counterfactual_texts=[CF_TEXT],
        )
        src, obs = executor.read_value("src"), executor.read_value("obs")
        return float((obs - src).abs().max())

    # a module boundary, and two interface slots
    assert written_minus_read("attention_premix") == 0.0
    assert written_minus_read("attention_query") == 0.0
    assert written_minus_read("attention_z") == 0.0
