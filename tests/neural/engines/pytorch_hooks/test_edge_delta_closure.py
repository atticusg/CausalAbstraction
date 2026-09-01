"""Per-edge delta routing, closed against the write it must reduce to.

Corpus 15's idiom: an *edge* — one attention head's contribution to one
residual receiver — is routed to its counterfactual value by a pair of
additive writes at the receiver, ``+1`` on the counterfactual read of
``attention_result`` and ``-1`` on the base read. Any number of such pairs
compose in one intervened model (§2.8: unbounded additive writes at one
address), which is what lets a whole edge set run as one joint forward.

The identity that makes it exact is structural, not an assumption about the
trunk being linear all the way to the logits:

    attention_output(L) = Σ_h attention_result(L, h) + o_proj_bias

so routing *every* head of layer L into the mixer output itself must land it
on ``Σ_h result_cf + bias`` — which is ``attention_output`` on the
counterfactual, exactly what swapping the whole mixer output produces.

The o-projection bias belongs to no head, and it drops out for a reason worth
stating precisely: it is the *same constant* in the base and counterfactual
values, so it cancels in the difference **however the derivation attributes
it** — the closure is insensitive to bias attribution rather than a test of it.
What pins the attribution itself is
``test_sites_round2_result.py::test_the_bias_belongs_to_no_head``. What the
nonzero-bias test below adds is that a real bias, rather than the fixtures'
absent (llama) or all-zero (gpt2) one, leaks no term into the closure.

The receiver is ``attention_output`` and not ``block_mid``: a ``block_mid``
write is currently only half-applied (it reaches the MLP but not the residual
skip) — goodfire-ai/causalab#79, pinned by
``test_a_block_mid_write_does_not_reach_the_residual_skip`` below.

Everything downstream of the receiver is a real forward in both documents,
which is the whole point: the nonlinearity is computed, not assumed.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Sequence

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import (
    BASE_TEXT,
    COUNTERFACTUAL_TEXT,
    TINY_GPT2,
)

pytestmark = pytest.mark.unit

#: Wrapper-level tolerance, as for the other cross-document write cases: the
#: two forwards differ only in accumulation order at the receiver.
TOL = dict(atol=1e-5, rtol=1e-4)

LAYER = 1


def _logits(doc: dict[str, Any], bundle: ModelBundle) -> torch.Tensor:
    executor = executor_for(
        doc,
        bundle,
        base_texts=[BASE_TEXT],
        counterfactual_texts=[COUNTERFACTUAL_TEXT],
    )
    return executor.read_value("logits")


def _logits_read(model: str) -> dict[str, Any]:
    return {
        "site": "lm_head",
        "pos": {"index": -1},
        "model": model,
        "input": "base",
    }


def _save_logits(model: str) -> list[dict[str, Any]]:
    return [
        {
            "value": "logits",
            "model": model,
            "input": "base",
            "file_path": "logits.safetensors",
        }
    ]


def _edge_delta_doc(
    heads: Sequence[int], *, layer: int = LAYER, self_cancel: bool = False
) -> dict[str, Any]:
    """Route each head in ``heads`` at ``layer`` into that layer's own
    residual receiver, by an additive ``+counterfactual`` / ``-base`` pair.

    ``self_cancel`` routes each head to its *own base* value instead — a pair
    that sums to zero, which is the patch-nothing end of the closure.
    """
    sites: dict[str, Any] = {
        "recv": {"component": "attention_output", "layer": layer},
        "lm_head": {"component": "lm_head"},
    }
    reads: dict[str, Any] = {}
    writes: dict[str, Any] = {}
    for head in heads:
        sites[f"h{head}"] = {
            "component": "attention_result",
            "layer": layer,
            "head": head,
        }
        for role, tag in (("base", "b"), ("counterfactual", "c")):
            reads[f"{tag}{head}"] = {
                "site": f"h{head}",
                "pos": {"index": -1},
                "model": "original",
                "input": role,
            }
        on_operand = f"b{head}" if self_cancel else f"c{head}"
        writes[f"on{head}"] = {
            "site": "recv",
            "pos": {"index": -1},
            "do": {"add_scaled": {"op": on_operand, "alpha": 1.0}},
        }
        writes[f"off{head}"] = {
            "site": "recv",
            "pos": {"index": -1},
            "do": {"add_scaled": {"op": f"b{head}", "alpha": -1.0}},
        }
    if self_cancel:
        # the counterfactual reads would be dead declarations (rule 11)
        for head in heads:
            reads.pop(f"c{head}")
    reads["logits"] = _logits_read("routed")
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=not self_cancel),
        "sites": sites,
        "reads": reads,
        "writes": writes,
        "intervened_models": {
            "routed": {"input": "base", "writes": sorted(writes)},
        },
        "save": _save_logits("routed"),
    }


def _mixer_swap_doc(*, layer: int = LAYER) -> dict[str, Any]:
    """The write the all-heads edge set must reduce to: the whole mixer output
    at ``layer`` swapped to its counterfactual value."""
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "sites": {
            "mix": {"component": "attention_output", "layer": layer},
            "lm_head": {"component": "lm_head"},
        },
        "reads": {
            "v_cf": {
                "site": "mix",
                "pos": {"index": -1},
                "model": "original",
                "input": "counterfactual",
            },
            "logits": _logits_read("routed"),
        },
        "writes": {
            "swap": {"site": "mix", "pos": {"index": -1}, "do": {"swap": "v_cf"}}
        },
        "intervened_models": {"routed": {"input": "base", "writes": ["swap"]}},
        "save": _save_logits("routed"),
    }


def _clean_doc() -> dict[str, Any]:
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "sites": {"lm_head": {"component": "lm_head"}},
        "reads": {"logits": _logits_read("original")},
        "save": _save_logits("original"),
    }


def test_routing_every_head_equals_swapping_the_mixer_output(bundle: ModelBundle):
    """Patch-everything closure, on both families in the fixture."""
    every_head = range(bundle.info.num_heads)
    torch.testing.assert_close(
        _logits(_edge_delta_doc(every_head), bundle),
        _logits(_mixer_swap_doc(), bundle),
        **TOL,
    )


def test_routing_no_head_reproduces_the_clean_run(bundle: ModelBundle):
    """Patch-nothing closure, through the machinery rather than around it:
    ``+1·base -1·base`` is a live pair of additive writes that sums to zero."""
    torch.testing.assert_close(
        _logits(
            _edge_delta_doc(range(bundle.info.num_heads), self_cancel=True), bundle
        ),
        _logits(_clean_doc(), bundle),
        **TOL,
    )


def test_one_edge_is_not_the_whole_mixer(bundle: ModelBundle):
    """The guard that keeps the closure honest: if a single head's delta
    already equalled the full swap, the two closures above would be passing
    for a reason that has nothing to do with per-edge routing."""
    if bundle.info.num_heads < 2:
        pytest.skip("a one-head model cannot distinguish an edge from the mixer")
    one = _logits(_edge_delta_doc([0]), bundle)
    whole = _logits(_mixer_swap_doc(), bundle)
    assert not torch.allclose(one, whole, **TOL), (
        "one head's contribution reproduced the whole mixer swap — the "
        "per-head derivation is not attributing anything"
    )


# --------------------------------------------------------------------------- #
# why the receiver above is not `block_mid`
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a block_mid write is half-applied: it reaches post_attention_layernorm "
        "(so the MLP sees it) but not the residual skip, which HF's decoder "
        "layer captured from the same tensor before the norm ran. §2.4 states "
        "block_output = block_mid + mlp_output, and under a write that identity "
        "silently fails. goodfire-ai/causalab#79 — remove this marker with "
        "the fix."
    ),
)
def test_a_block_mid_write_does_not_reach_the_residual_skip(
    llama_bundle: ModelBundle,
):
    """§2.4's second block identity, under a write at ``block_mid``.

    Read in the *same* intervened model, ``block_output - mlp_output`` must be
    the ``block_mid`` that model actually has. It is instead the pre-write
    value, so the write moved the MLP's input and left the skip alone.

    Note the sibling identity ``block_output == block_input +
    attention_output + mlp_output`` is *blind* to this: a block_mid write
    touches neither of the first two terms, so it holds either way. Only the
    two-term form catches it.
    """
    pos = {"index": -1}
    doc = {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "sites": {
            "mid": {"component": "block_mid", "layer": LAYER},
            "mlp": {"component": "mlp_output", "layer": LAYER},
            "out": {"component": "block_output", "layer": LAYER},
        },
        "reads": {
            "v_cf": {
                "site": "mid",
                "pos": pos,
                "model": "original",
                "input": "counterfactual",
            },
            "r_mid": {"site": "mid", "pos": pos, "model": "m", "input": "base"},
            "r_mlp": {"site": "mlp", "pos": pos, "model": "m", "input": "base"},
            "r_out": {"site": "out", "pos": pos, "model": "m", "input": "base"},
        },
        "writes": {"swap": {"site": "mid", "pos": pos, "do": {"swap": "v_cf"}}},
        "intervened_models": {"m": {"input": "base", "writes": ["swap"]}},
        "save": [
            {
                "value": name,
                "model": "m",
                "input": "base",
                "file_path": f"{name}.safetensors",
            }
            for name in ("r_mid", "r_mlp", "r_out")
        ],
    }
    executor = executor_for(
        doc,
        llama_bundle,
        base_texts=[BASE_TEXT],
        counterfactual_texts=[COUNTERFACTUAL_TEXT],
    )
    mid, mlp, out = (executor.read_value(n) for n in ("r_mid", "r_mlp", "r_out"))
    # the write did land where it was addressed — this is not a lost write
    torch.testing.assert_close(mid, executor.read_value("v_cf"), **TOL)
    torch.testing.assert_close(out - mlp, mid, **TOL)


@contextlib.contextmanager
def _nonzero_o_projection_bias(bundle: ModelBundle, layer: int) -> Iterator[None]:
    """Give gpt2's attention output projection a bias that is actually there.

    The fixture ships zeros, which cannot tell "the bias cancels" apart from
    "the bias is dropped". Restored on the way out — the bundle is
    session-cached, so leaving it perturbed would leak into every later test.
    """
    proj = bundle.model.transformer.h[layer].attn.c_proj
    assert proj.bias is not None, "gpt2's c_proj is expected to carry a bias"
    saved = proj.bias.detach().clone()
    with torch.no_grad():
        proj.bias.copy_(torch.linspace(-0.7, 0.7, proj.bias.numel()))
    try:
        yield
    finally:
        with torch.no_grad():
            proj.bias.copy_(saved)


def test_the_closure_holds_with_a_nonzero_o_projection_bias():
    """The one term the per-head derivation cannot attribute, made real.

    Both fixtures dodge it — llama's o-projection has no bias, tiny-random
    gpt2's is all zeros — so the closure above never meets one. It should not
    care: the bias is the same constant on both inputs and cancels in the
    difference. This asserts that it does not care, which is a weaker and more
    honest claim than "this pins the bias".
    """
    bundle = load_model(TINY_GPT2)
    with _nonzero_o_projection_bias(bundle, LAYER):
        assert bundle.model.transformer.h[LAYER].attn.c_proj.bias.abs().max() > 0
        torch.testing.assert_close(
            _logits(_edge_delta_doc(range(bundle.info.num_heads)), bundle),
            _logits(_mixer_swap_doc(), bundle),
            **TOL,
        )
    # and the perturbation really was undone
    assert bundle.model.transformer.h[LAYER].attn.c_proj.bias.abs().max() == 0
