"""``attention_probs``: read + write, and the one component `layout` won't describe.

PR4 of the hookpoint-vocabulary stack, and the last of round 1. The three checks
at the top are nnterp's, ported to this backend
(``nnterp/rename_utils.py`` ``check_source``): the pattern must have shape
``(batch, heads, seq, seq)``, its rows must sum to 1, and **writing it must
change the logits**. The third is the one that matters, because it is the one a
plausible-looking implementation fails.

Round-1 scope is the whole pattern. Addressing one query row, featurizing, or
slicing ``dims`` all need the typed feature-shape descriptor — the feature axis
here *is* a position axis — so each is refused and named as follow-up F1 rather
than approximated.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.pytorch_hooks.attention_probs import eager_attention_writes
from causalab.neural.pytorch_hooks.backend import PytorchHooksBackend
from causalab.neural.pytorch_hooks.sites import resolve_site
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import SiteSpec

from ._drive import base_data_section, executor_for

pytestmark = pytest.mark.smoke

#: 📐 the fixture's only full-attention layer; 0-2 are Gated DeltaNet.
FULL_ATTENTION_LAYER = 3
DELTANET_LAYER = 0

#: Same token count, so an attention-pattern interchange is shape-compatible.
BASE_TEXT = "the quick brown fox jumps"
CF_TEXT = "a slow green turtle sleeps"


def _read_doc(
    *,
    layer: int = FULL_ATTENTION_LAYER,
    pos: object = "all",
    featurizer: bool = False,
    dims: list[int] | None = None,
) -> dict:
    read: dict = {"site": "tap", "pos": pos, "model": "original", "input": "base"}
    doc: dict = {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "sites": {"tap": {"component": "attention_probs", "layer": layer}},
        "reads": {"r": read},
        "save": [
            {
                "value": "r",
                "model": "original",
                "input": "base",
                "file_path": "a.safetensors",
            }
        ],
    }
    if featurizer:
        doc["featurizers"] = {"f": {"kind": "subspace", "k": 1}}
        read["featurizer"] = "f"
    if dims is not None:
        read["dims"] = dims
    return doc


def _swap_doc(*, layer: int = FULL_ATTENTION_LAYER) -> dict:
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "sites": {
            "tap": {"component": "attention_probs", "layer": layer},
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
        "writes": {"patch": {"site": "tap", "pos": "all", "do": {"swap": "v_cf"}}},
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


@pytest.fixture(scope="module")
def pattern(qwen35moe_bundle) -> torch.Tensor:
    return executor_for(
        _read_doc(), qwen35moe_bundle, base_texts=[BASE_TEXT]
    ).read_value("r")


# --------------------------------------------------------------------------- #
# nnterp's three-part check
# --------------------------------------------------------------------------- #


def test_the_pattern_has_the_attention_shape(qwen35moe_bundle, pattern):
    """nnterp check 1: (batch, heads, query, key)."""
    batch, heads, query, key = pattern.shape
    assert batch == 1
    assert heads == qwen35moe_bundle.info.num_heads
    assert query == key, "both axes are positions — that is the whole difficulty"


def test_the_pattern_rows_sum_to_one(pattern):
    """nnterp check 2: it is a distribution over key positions."""
    sums = pattern.sum(-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


def test_writing_the_pattern_changes_the_logits(qwen35moe_bundle):
    """nnterp check 3, and the reason this PR is not a one-line tap.

    A ``register_forward_hook`` on the mixer CAN rewrite element 1 of its output
    tuple — and it would change nothing, because ``attn_output`` was computed
    from the pattern inside the attention function before the hook fires. That
    is the same silent-no-op shape as writing ``router_logits`` (PR3). The write
    goes through the eager attention function instead, and this test is what
    says the difference mattered.
    """
    executor = executor_for(
        _swap_doc(),
        qwen35moe_bundle,
        base_texts=[BASE_TEXT],
        counterfactual_texts=[CF_TEXT],
    )
    clean = executor.read_value("clean")
    after = executor.read_value("after")
    assert float((after - clean).abs().max()) > 0.0, (
        "swapping the whole attention pattern left the logits unchanged — the "
        "write is not reaching the value multiply"
    )


# --------------------------------------------------------------------------- #
# the property that makes duplicating the post-softmax math safe
# --------------------------------------------------------------------------- #


def test_an_identity_edit_is_bit_identical(qwen35moe_bundle):
    """The wrapper redoes the two lines that follow the softmax
    (``repeat_kv`` + ``matmul(...).transpose(1, 2).contiguous()``). Duplicating
    library internals is how a backend rots silently, so it is pinned: with an
    edit that changes nothing, the recomputed logits must equal the unpatched
    logits **exactly**. 📐 Measured max difference 0.0.

    If a future transformers changes what happens after the softmax, this fails
    instead of the numbers quietly drifting.
    """
    encoded = qwen35moe_bundle.tokenizer(BASE_TEXT, return_tensors="pt")
    attn = qwen35moe_bundle.mixer_at(FULL_ATTENTION_LAYER)

    with torch.no_grad():
        clean = qwen35moe_bundle.model(**encoded).logits.clone()

    seen: dict[str, object] = {}

    def identity(probs: torch.Tensor) -> None:
        seen["shape"] = tuple(probs.shape)  # mutate nothing

    with eager_attention_writes({id(attn): identity}):
        with torch.no_grad():
            recomputed = qwen35moe_bundle.model(**encoded).logits.clone()

    assert seen["shape"], "the wrapper never ran — the registry was not consulted"
    assert torch.equal(recomputed, clean), (
        "recomputing the attention output from an unmodified pattern did not "
        "reproduce the model's own result — the post-softmax math has drifted"
    )


def test_the_wrapper_only_touches_the_modules_it_was_given(qwen35moe_bundle):
    """Scoping: the registry entry is global while installed, so a mixer absent
    from the edit map must be left alone."""
    encoded = qwen35moe_bundle.tokenizer(BASE_TEXT, return_tensors="pt")
    with torch.no_grad():
        clean = qwen35moe_bundle.model(**encoded).logits.clone()

    def wreck(probs: torch.Tensor) -> None:
        probs.zero_()

    # a module that never runs an eager attention forward (DeltaNet layer)
    other = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    with eager_attention_writes({id(other): wreck}):
        with torch.no_grad():
            same = qwen35moe_bundle.model(**encoded).logits.clone()
    assert torch.equal(same, clean)


# --------------------------------------------------------------------------- #
# registry hygiene
# --------------------------------------------------------------------------- #


def test_the_registry_is_restored_afterwards(qwen35moe_bundle):
    """Leaving the key installed would keep the wrapper in force for the rest of
    the process — quietly, and for every later model."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    assert "eager" not in ALL_ATTENTION_FUNCTIONS
    attn = qwen35moe_bundle.mixer_at(FULL_ATTENTION_LAYER)
    with eager_attention_writes({id(attn): lambda p: None}):
        assert "eager" in ALL_ATTENTION_FUNCTIONS
    assert "eager" not in ALL_ATTENTION_FUNCTIONS


def test_the_registry_is_restored_after_an_exception(qwen35moe_bundle):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    attn = qwen35moe_bundle.mixer_at(FULL_ATTENTION_LAYER)
    with pytest.raises(RuntimeError):
        with eager_attention_writes({id(attn): lambda p: None}):
            raise RuntimeError("boom")
    assert "eager" not in ALL_ATTENTION_FUNCTIONS


def test_no_edits_installs_nothing(qwen35moe_bundle):
    """A run with no pattern writes must not pay for, or risk, the patch."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    with eager_attention_writes({}):
        assert "eager" not in ALL_ATTENTION_FUNCTIONS


def test_the_backend_now_declares_the_capability():
    """§8 routing refused these documents before; it must accept them now."""
    assert "writable_attention_probs" in PytorchHooksBackend.capabilities


# --------------------------------------------------------------------------- #
# refusals: what round 1 will not approximate
# --------------------------------------------------------------------------- #


def test_the_tap_is_native_layout_not_the_contract(qwen35moe_bundle):
    """`layout="native"` claims nothing about the axes. Defaulting to `"bsd"`
    would convert identically and *assert* something false."""
    site = resolve_site(
        qwen35moe_bundle,
        SiteSpec(component="attention_probs", layer=FULL_ATTENTION_LAYER),
    )
    assert site.layout == "native"
    assert site.tuple_index == 1  # (attn_output, attn_weights)


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"pos": {"index": 1}}, "addresses positions"),
        ({"pos": {"span": [0, 2]}}, "addresses positions"),
        ({"featurizer": True}, "featurizes"),
        ({"dims": [0, 1]}, "slices 'dims'"),
    ],
    ids=["pos_index", "pos_span", "featurizer", "dims"],
)
def test_the_forms_that_need_f1_are_refused(qwen35moe_bundle, kwargs, needle):
    """Each of these would silently read the wrong axis: ``_gather`` would index
    heads with positions, and ``dims`` would select key positions as features.
    Refused, and every message names follow-up F1."""
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(
            _read_doc(**kwargs), qwen35moe_bundle, base_texts=[BASE_TEXT]
        ).read_value("r")
    message = str(excinfo.value)
    assert needle in message
    assert "F1" in message


def test_a_deltanet_layer_refuses_on_the_architecture(qwen35moe_bundle):
    """PR2's stream check still owns this, and it must keep owning it: at a
    Gated DeltaNet layer there is no attention matrix, which stays true now that
    the component is implemented."""
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(
            _read_doc(layer=DELTANET_LAYER),
            qwen35moe_bundle,
            base_texts=[BASE_TEXT],
        ).read_value("r")
    assert "full-attention mixer" in str(excinfo.value)


def test_an_interchange_across_different_lengths_refuses(qwen35moe_bundle):
    """A whole-pattern swap needs both inputs to have the same position count —
    the pattern is (seq, seq), so a mismatch is not broadcastable. Refuse rather
    than let a broadcast produce a plausible tensor."""
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(
            _swap_doc(),
            qwen35moe_bundle,
            base_texts=[BASE_TEXT],
            counterfactual_texts=["a slow green turtle sleeps very deeply indeed"],
        ).read_value("after")
    assert "same" in str(excinfo.value)
