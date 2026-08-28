"""The Qwen3.6-35B-A3B hookpoint sweep: one table, two tiers.

The smoke tier runs it on ``tiny-random/qwen3.5-moe`` (the A3B architecture in
miniature — a hybrid Gated-DeltaNet/full-attention tower with a sparse MoE plus
shared expert in every layer) and the golden tier on the real checkpoint. Both
read the partition and the document builders from here, so "which hookpoints
exist" is answered once and a component cannot be exercised at one tier and
quietly skipped at the other.

The partition is the engines' own declarations, restated as data:

* **shared** — both engines serve it, so the agreement claim is the ordinary
  one: same document, same numbers;
* **hooks-only** — the ``delta_*`` kernel interior, which the reference engine
  reaches by swapping the modeling file's kernel globals; nnsight has no
  equivalent mechanism and does not declare the vocabulary;
* **nnsight-only** — the ``deltanet_*`` interior and ``expert_permutation``,
  ``.source`` lines inside a fused forward that no hook can reach.

📐 The two single-engine sets are not two blind spots: measured on the fixture,
they name the *same physical tensors* through the two different mechanisms
(:data:`DELTA_FAMILY_PAIRS`), which is the cross-engine agreement available for
30 of the target's 40 layers.

``mlp_activation`` is in neither list: the vocabulary carries it, but the A3B's
MLP is a sparse-MoE block with no ``act_fn`` child, so it does not exist on this
architecture at all (:data:`ABSENT_ON_A3B`). Its analogues are
``expert_activation`` and ``shared_expert_activation``.
"""

from __future__ import annotations

from typing import Any

import torch

from causalab.protocol.schema import COMPONENTS, LAYERLESS_COMPONENTS

__all__ = [
    "ABSENT_ON_A3B",
    "ATOL",
    "DELTA_FAMILY_PAIRS",
    "HOOKS_ONLY",
    "NNSIGHT_ONLY",
    "SHARED_ANY_STREAM",
    "SHARED_FULL_ONLY",
    "SHARED_LAYERLESS",
    "READ_ONLY",
    "SWAP_ONLY_WRITES",
    "WHOLE_TENSOR_ONLY",
    "assert_same",
    "default_pos",
    "interchange_doc",
    "make_executor",
    "read_doc",
    "stream_layers",
    "write_cases",
]

#: Reads agree to this tolerance when both engines run fp32 eager on the same
#: device: the same kernels in a different order of capture, so anything larger
#: is an executor bug rather than float noise.
ATOL = 1e-5

# --------------------------------------------------------------------------- #
# the partition (mirrors the engines' `components` declarations)
# --------------------------------------------------------------------------- #

#: Layer-less components both engines serve.
SHARED_LAYERLESS: tuple[str, ...] = (
    "input_ids",
    "embeddings",
    "ln_final",
    "lm_head",
)

#: Both engines, and the component exists in **either** block type — the
#: residual-stream boundaries and the whole MoE surface, which every layer of
#: the A3B carries.
SHARED_ANY_STREAM: tuple[str, ...] = (
    "block_input",
    "attention_input_norm",
    "attention_output",
    "block_mid",
    "mlp_input_norm",
    "mlp_input",
    "mlp_output",
    "router_logits",
    "router_scores",
    "expert_idx",
    "expert_gate_proj",
    "expert_up_proj",
    "expert_activation",
    "expert_output",
    "routed_output",
    "shared_expert_gate_proj",
    "shared_expert_up_proj",
    "shared_expert_activation",
    "shared_expert_output",
    "shared_expert_gate",
    "block_output",
)

#: Both engines, but only at a full-attention layer — 10 of the target's 40.
SHARED_FULL_ONLY: tuple[str, ...] = (
    "attention_query_pre_rope",
    "attention_key_pre_rope",
    "attention_value_states",
    "attention_gate",
    "attention_query",
    "attention_key",
    "attention_scores",
    "attention_z",
    "attention_result",
    "attention_premix",
    "attention_probs",
)

#: The reference engine's Gated DeltaNet interior — linear-attention layers only.
HOOKS_ONLY: tuple[str, ...] = tuple(c for c in COMPONENTS if c.startswith("delta_"))

#: The nnsight engine's fused-forward interiors.
NNSIGHT_ONLY: tuple[str, ...] = tuple(
    c for c in COMPONENTS if c.startswith("deltanet_")
) + ("expert_permutation",)

#: In the vocabulary, absent from this architecture — see the module docstring.
ABSENT_ON_A3B: tuple[str, ...] = ("mlp_activation",)

#: Components no write may target (`sites.READ_ONLY_COMPONENTS`), restated here
#: so the sweep's write half skips them by table rather than by exception.
READ_ONLY: frozenset[str] = frozenset(
    {
        "input_ids",
        "attention_result",
        "router_logits",
        "delta_kv_mem",
        "delta_state_update",
        "expert_permutation",
    }
)

#: Components a write may only **replace** — the integer routing table and the
#: normalized attention pattern. The sweep writes `swap` everywhere, so these
#: need no special case; the set is here because the docs table cites it.
SWAP_ONLY_WRITES: frozenset[str] = frozenset({"expert_idx", "attention_probs"})

#: Components that can only be addressed whole. 📐 The attention matrix has
#: **two** position axes (query and key), so an integer position is ambiguous
#: between them and the executor refuses it by shape — no component name
#: appears in that refusal, which is what makes it a rule rather than a case.
#: The sweep honours the rule rather than skipping the components.
WHOLE_TENSOR_ONLY: frozenset[str] = frozenset({"attention_scores", "attention_probs"})


def default_pos(component: str) -> object:
    """The position spec the sweep addresses ``component`` with."""
    return "all" if component in WHOLE_TENSOR_ONLY else -1


#: 📐 Measured on ``tiny-random/qwen3.5-moe`` (2026-08-28): the reference
#: engine's ``delta_*`` kernel taps and the nnsight engine's ``deltanet_*``
#: ``.source`` addresses name the same tensors. ``relation`` says how to line
#: the two up:
#:
#: * ``"identical"`` — same shape, max abs diff 0.0;
#: * ``"gva_tile"`` — ``delta_*`` is post ``repeat_interleave`` over the head
#:   axis (value-head space); ``deltanet_*`` is pre (key-head space). Exact
#:   after tiling.
#: * ``"chunk_boundary"`` — ``delta_state`` is per **step**, ``deltanet_state``
#:   per 64-token **chunk**; the chunk's state is the step-state at the chunk's
#:   last position (agreed to 3.4e-8 on the fixture).
DELTA_FAMILY_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("delta_qkv", "deltanet_qkv", "identical"),
    ("delta_conv", "deltanet_qkv_conv", "identical"),
    ("delta_gate", "deltanet_gate", "identical"),
    ("delta_value", "deltanet_value", "identical"),
    ("delta_beta", "deltanet_beta", "identical"),
    ("delta_decay", "deltanet_decay", "identical"),
    ("delta_kernel_output", "deltanet_core_out", "identical"),
    ("delta_premix", "deltanet_gated_out", "identical"),
    ("delta_query", "deltanet_query", "gva_tile"),
    ("delta_key", "deltanet_key", "gva_tile"),
    ("delta_state", "deltanet_state", "chunk_boundary"),
)

#: The kernel's chunk length, which ``chunk_boundary`` alignment needs. 📐 Read
#: off the kernel's own loop in the N7 verification, not off config.
DELTA_CHUNK = 64


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #


def _data(with_cf: bool) -> dict[str, Any]:
    data: dict[str, Any] = {"base": {"dataset": "inline", "field": "input"}}
    if with_cf:
        data["counterfactual"] = {
            "dataset": "inline",
            "field": "counterfactual_inputs[0]",
        }
    return data


def read_doc(
    component: str, layer: int | None, *, pos: object = -1, head: int | None = None
) -> dict[str, Any]:
    """Read one site on the base input and save it."""
    site: dict[str, Any] = {"component": component}
    if layer is not None:
        site["layer"] = layer
    if head is not None:
        site["head"] = head
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=False),
        "sites": {"tap": site},
        "reads": {
            "r": {"site": "tap", "pos": pos, "model": "original", "input": "base"}
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


def interchange_doc(
    component: str, layer: int | None, *, pos: object = -1
) -> dict[str, Any]:
    """Read the site on the counterfactual, swap it into the base forward, read
    the patched logits — the intervention whose downstream effect must agree."""
    site: dict[str, Any] = {"component": component}
    if layer is not None:
        site["layer"] = layer
    return {
        "version": "1",
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=True),
        "sites": {"tap": site, "head": {"component": "lm_head"}},
        "reads": {
            "v_cf": {
                "site": "tap",
                "pos": pos,
                "model": "original",
                "input": "counterfactual",
            },
            "logits": {
                "site": "head",
                "pos": -1,
                "model": "patched",
                "input": "base",
            },
        },
        "writes": {"patch": {"site": "tap", "pos": pos, "do": {"swap": "v_cf"}}},
        "intervened_models": {"patched": {"input": "base", "writes": ["patch"]}},
        "save": [
            {
                "value": "logits",
                "model": "patched",
                "input": "base",
                "file_path": "l.safetensors",
            }
        ],
    }


def make_executor(executor_cls, doc_raw, bundle, *, rows, with_cf: bool):
    """The same document driven through either engine's executor."""
    from causalab.protocol.schema import parse_document
    from causalab.protocol.validate import validate_document

    from tests.protocol._docs import in_order

    doc = parse_document(in_order(doc_raw))
    validate_document(doc, engine_is_local=True)
    role_rows = {"base": rows}
    role_fields = {"base": "input"}
    if with_cf:
        role_rows["counterfactual"] = rows
        role_fields["counterfactual"] = "counterfactual_inputs[0]"
    return executor_cls(
        doc,
        bundle,
        role_rows=role_rows,
        role_fields=role_fields,
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #


def assert_same(a: torch.Tensor, b: torch.Tensor, what: str, *, atol: float = ATOL):
    """Shape-then-value agreement; integers must match exactly.

    Integer components (``input_ids``, ``expert_idx``, ``expert_permutation``)
    carry labels, not measurements: a tolerance on them would let a routing
    table that sends a token to a different expert pass.
    """
    assert a.shape == b.shape, f"{what}: {tuple(a.shape)} != {tuple(b.shape)}"
    if not a.dtype.is_floating_point:
        assert torch.equal(a, b), f"{what}: integer values differ"
        return
    diff = (a.double() - b.double()).abs().max().item()
    assert torch.allclose(a, b, atol=atol, rtol=0), (
        f"{what}: max abs diff {diff:.3e} exceeds {atol}"
    )


def stream_layers(bundle) -> tuple[int, int]:
    """``(first linear-attention layer, first full-attention layer)`` of the
    loaded tower — read off the model rather than hardcoded, because the
    fixture's hybrid schedule and the real A3B's are different orders of the
    same two block types."""
    n = len(bundle.model.model.layers)
    streams = [bundle.stream_at(i) for i in range(n)]
    assert "linear_attention" in streams, f"no Gated DeltaNet layer in {streams}"
    assert "full_attention" in streams, f"no full-attention layer in {streams}"
    return streams.index("linear_attention"), streams.index("full_attention")


def write_cases(components: tuple[str, ...]) -> tuple[str, ...]:
    """The sweep's write half: everything a write may target."""
    return tuple(c for c in components if c not in READ_ONLY)


def coverage_partition() -> dict[str, tuple[str, ...]]:
    """Every component in the vocabulary, claimed by exactly one bucket — what
    the completeness guard checks the sweep against."""
    return {
        "shared_layerless": SHARED_LAYERLESS,
        "shared_any_stream": SHARED_ANY_STREAM,
        "shared_full_only": SHARED_FULL_ONLY,
        "hooks_only": HOOKS_ONLY,
        "nnsight_only": NNSIGHT_ONLY,
        "absent_on_a3b": ABSENT_ON_A3B,
    }


def unclaimed_components() -> tuple[str, ...]:
    claimed: set[str] = set()
    for group in coverage_partition().values():
        claimed |= set(group)
    return tuple(c for c in COMPONENTS if c not in claimed)


def double_claimed_components() -> tuple[str, ...]:
    seen: set[str] = set()
    twice: set[str] = set()
    for group in coverage_partition().values():
        for component in group:
            if component in seen:
                twice.add(component)
            seen.add(component)
    return tuple(sorted(twice))


def layerless(component: str) -> bool:
    return component in LAYERLESS_COMPONENTS
