"""SiteResolver: the spec's component vocabulary → concrete module taps.

Each site record resolves to ``(module, io side, feature-axis slice)``.
The map is engine-shared (plan §2.3): both engines tap the same modules —
pytorch_hooks with ``register_forward_hook`` / ``register_forward_pre_hook``,
the nnsight engine by handing the same tree access to envoys whose
``.input`` / ``.output`` it reads and assigns in-trace. Writes replace the
same tensor either way. The table mirrors the hook-oracle reference
(``tests/neural/activations/hook_oracle.py``) for the two supported
families:

* **Llama-tree** (Llama/Qwen/Mistral/Gemma): ``model.layers[L]``,
  separate ``self_attn.{q,k,v,o}_proj``, SwiGLU MLP;
* **GPT-2-tree**: ``transformer.h[L]``, fused ``attn.c_attn``,
  ``attn.c_proj``.

Two semantics deliberately preserved from the oracle:

* ``mlp_activation`` names *different tensors per family* — Llama taps
  ``act_fn``'s output (``act(gate_proj(x))``, NOT the down-projection's
  input), GPT-2 taps ``c_proj``'s input (which IS the down-projection's
  input). Inherited 1:1 from the pyvene era and pinned by the oracle.
* the mixer's **interior** is four module boundaries, not four chunk ops
  (``attention_query_pre_rope``, ``attention_key_pre_rope``,
  ``attention_value_states``, ``attention_gate``) — see
  :func:`_attention_interior_site`, which is where the family differences live;
* ``attention_premix`` with a ``head`` is the ``[H*d, (H+1)*d]`` column
  slice of the o-projection's **input** — query-head space (``head_dim``
  honours a decoupled ``config.head_dim``). 📐 On a **gated** attention
  family (Qwen3.5/3.6's ``self_attn``, where the mixer multiplies by a
  learned gate before projecting out) that input is therefore
  **post-gate**: it is ``gate * z``, not the attention output ``z``. The
  tap is unchanged and correct — this note exists because "value" reads
  like the pre-gate tensor, and the two differ by an elementwise factor
  that a subspace fit will happily absorb without complaining.

Unsupported components refuse with the registry-extension message style —
``attention_probs`` needs an attention-internal tap this engine does not
implement (its capability is absent, so routing already refuses documents
that write to it), and MoE components await an MoE family in the tree table.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from causalab.protocol.errors import ProtocolError
from causalab.protocol.registry import component_shape, head_space_refusal
from causalab.protocol.schema import SiteSpec
from causalab.protocol.shapes import FeatureShape


__all__ = [
    "ATTENTION_FUNCTION_SLOTS",
    "DELTA_KERNEL_SLOTS",
    "NORMALIZED_TAPS",
    "READ_ONLY_COMPONENTS",
    "SWAP_ONLY_COMPONENTS",
    "ResolvedSite",
    "resolve_site",
]

#: Components a write must not target, mapped to what to do instead.
#:
#: The two entries are refused for *different* reasons, and the difference is
#: worth keeping straight:
#:
#: * ``router_logits`` is a **silent no-op** — the block discards it, so the run
#:   succeeds, the numbers do not move at all, and the conclusion is wrong. That
#:   is the outcome most worth spending code to prevent.
#: * ``input_ids`` is the opposite: 📐 a write there *does* land. The tap is the
#:   embedding's pre-hook input, and mutating it in place changes the ids the
#:   model looks up (measured: it also writes through to the caller's own
#:   tensor). It is refused because token ids are not an activation — editing
#:   the model's input is a change to the *dataset*, which belongs in the row's
#:   text where it is visible in the document.
#:
#: 📐 ``router_logits`` earns its place by measurement, not by principle.
#: ``Qwen3_5MoeSparseMoeBlock.forward`` reads the router as
#: ``_, routing_weights, selected_experts = self.gate(...)`` — element 0 is
#: assigned to ``_`` and never used again, so overwriting it after the gate has
#: returned changes a tensor nothing downstream reads. Measured: patching it and
#: re-running moves the logits by exactly 0.0, while the same patch to
#: ``router_scores`` or ``expert_idx`` moves them.
READ_ONLY_COMPONENTS: dict[str, str] = {
    "input_ids": (
        "the model's token input is not an activation; change the row's text "
        "instead, or write 'embeddings' to edit the vector the ids look up"
    ),
    "attention_result": (
        "it is derived, not computed: the model never forms the per-head "
        "contribution at all — it forms their sum, by projecting the whole "
        "'attention_premix' at once — so there is no tensor here for a write "
        "to change. Write 'attention_premix' instead, with the same 'head'; "
        "'attention_result' is a linear function of it, so a write there moves "
        "this by exactly the projection of what you wrote"
    ),
    "router_logits": (
        "the MoE block discards the router's logits (it destructures them into "
        "'_') and routes on the scores and indices it computed from them, so a "
        "write here cannot reach anything — write 'router_scores' to reweight "
        "the chosen experts, or 'expert_idx' to change which experts fire"
    ),
}

#: Components a write may only **replace**, never arithmetically adjust, mapped
#: to why. Same philosophy as ``stream`` and as #53's ``attention_probs`` rule:
#: refuse by name rather than compute something plausible and wrong.
#:
#: 📐 ``expert_idx`` measured: an ``add_scaled`` write over the int64 routing
#: table runs to completion today with no refusal anywhere. Nothing downstream
#: is checking that the ids it produced still name experts — they index whatever
#: they happen to land on, and on CUDA an out-of-range id is a device-side
#: assert far from the write that caused it.
SWAP_ONLY_COMPONENTS: dict[str, str] = {
    "expert_idx": (
        "the routing table carries integer expert ids, not features: a delta, a "
        "scale or a clamp over them yields ids chosen by arithmetic on labels, "
        "which route to arbitrary experts where they stay in range and fail at "
        "the gather where they do not. Swap in an index tensor read from "
        "elsewhere to change which experts fire, or write 'router_scores' to "
        "reweight the experts already chosen"
    ),
}


@dataclasses.dataclass(frozen=True)
class ResolvedSite:
    """One tapped location: the module, which side of it carries the
    activation, an optional feature-axis slice (per-head views), and how the
    module's own tensor shape relates to the executor's ``(batch, position,
    feature)`` contract.

    ``shape`` is never chosen per tap: :func:`resolve_site` reads it from
    :func:`~causalab.protocol.registry.component_shape`, so the description each
    engine converts by and the description the protocol layer validates against
    are the same object. ``tuple_index`` defaults to the historical rule —
    element 0 of a tuple payload. See :mod:`causalab.neural.shared.layout` for
    how the conversion is computed from the declared axes.
    """

    module: Any
    kind: str  # "in" | "out"
    #: The module's native tensor shape; converted to/from the executor's
    #: contract at the hook boundary rather than special-cased per component.
    shape: FeatureShape
    feature_slice: slice | None = None
    layer: int = 0
    component: str = "block_output"
    #: Which element of a tuple payload the tap means. None keeps the historical
    #: rule (element 0 of a tuple, else the payload itself); an explicit index
    #: is required for e.g. a router returning (logits, scores, indices).
    tuple_index: int | None = None
    #: Where inside the attention function this component lives, when it is not
    #: a module boundary at all — see
    #: :mod:`causalab.neural.pytorch_hooks.attention_interface`.
    #:
    #: Set together with ``kind == "interface"`` for the four function-interior
    #: components. ``attention_probs`` is the one site that sets it while
    #: keeping an ordinary ``kind``: the mixer *returns* the pattern, so reading
    #: it is a plain module tap, and only the write has to go through the
    #: function.
    interface_slot: str | None = None
    #: The head the site named, kept alongside ``feature_slice`` because a
    #: *derived* component slices in a space the raw tensor does not have.
    head: int | None = None
    #: Set when the component's value is **computed from** the tapped tensor
    #: rather than being it. Then ``shape`` describes what is captured and
    #: :func:`~causalab.protocol.registry.component_shape` describes the value —
    #: the one place in the backend where those two differ, and the field exists
    #: so that difference is declared rather than inferred.
    derivation: str | None = None

    @property
    def depth(self) -> tuple[int, int]:
        """(layer, intra-order) — matches the protocol planner's ranks."""
        from causalab.protocol.plan import COMPONENT_RANK, UNRANKED  # one table

        rank = COMPONENT_RANK.get(self.component, UNRANKED)
        if self.component in ("ln_final", "lm_head"):
            return (1_000_000, rank)
        return (self.layer, rank)


def _blocks(bundle: Any) -> Any:
    return (
        bundle.model.transformer.h
        if bundle.is_gpt2_family
        else bundle.model.model.layers
    )


def _attn(bundle: Any, layer: int) -> Any:
    """The mixer at ``layer`` — ``self_attn``, ``attn`` or ``linear_attn``.

    Was ``block.self_attn`` for every non-GPT-2 model, which AttributeErrors on
    a hybrid tower: 📐 on ``tiny-random/qwen3.5-moe`` three of four layers carry
    ``linear_attn`` (Gated DeltaNet) and only one carries ``self_attn``. The
    per-layer answer lives on the bundle (§5.2)."""
    return bundle.mixer_at(layer)


def _o_proj(bundle: Any, layer: int) -> Any:
    attn = _attn(bundle, layer)
    return attn.c_proj if bundle.is_gpt2_family else attn.o_proj


def _embedding(bundle: Any) -> Any:
    return (
        bundle.model.transformer.wte
        if bundle.is_gpt2_family
        else bundle.model.model.embed_tokens
    )


def _head_slice(bundle: Any, component: str, head: int | None) -> slice | None:
    """The feature-axis slice a ``head`` names — or a refusal.

    The bound comes from the component's own shape, not from
    ``info.num_heads``. 📐 That distinction is not cosmetic: under GQA the
    KV-space components are ``num_key_value_heads`` wide, and a query-space
    bound over them produces a slice that is *empty* rather than out of range.
    Python does not raise on that — the read saves a ``(b, n_pos, 0)`` tensor
    and the write mutates nothing — which is the silent no-op that
    ``READ_ONLY_COMPONENTS`` exists to prevent elsewhere.
    """
    if head is None:
        return None
    shape = component_shape(bundle.info, component)
    space = shape.head_space
    if space is None:
        raise ProtocolError("P4", head_space_refusal(component, head, shape))
    if not 0 <= head < space:
        raise ProtocolError(
            "P4",
            f"site names head {head} on component {component!r}, which has "
            f"{space} heads ({shape.describe()})",
        )
    width = shape.width
    assert width is not None  # a head axis implies a feature axis
    per_head = width // space
    return slice(head * per_head, (head + 1) * per_head)


#: The mixer's interior at module boundaries — round 2.2.
#:
#: 📐 The plan note assumed these needed function-level taps inside the mixer
#: forward. Measured on ``tiny-random/qwen3.5-moe``, three of the four are
#: ordinary ``nn.Module`` outputs: ``Qwen3_5MoeAttention`` runs ``q_norm`` and
#: ``k_norm`` **before** RoPE, so their outputs *are* the pre-RoPE projections,
#: and ``v_proj``'s output is ``v`` itself. Only the gate needs a descriptor
#: trick, and only because it shares a projection with ``q``.
_ATTENTION_INTERIOR: frozenset[str] = frozenset(
    {
        "attention_query_pre_rope",
        "attention_key_pre_rope",
        "attention_value_states",
        "attention_gate",
    }
)


def _q_projection_splits(bundle: Any, attn: Any, layer: int) -> int:
    """How many per-head sub-tensors the q-projection emits: 1, or 2 on a
    gated-attention family.

    📐 Measured rather than matched on a model type: ``q_proj.out_features`` is
    ``H·d`` on llama (16 for H 4, d 4) and ``H·2·d`` on qwen3.5-moe (512 for
    H 8, d 32), because the latter packs ``[q_h | gate_h]`` per head. A family
    that is neither is refused by name rather than chunked on a guess — the
    tensor would split into plausible halves either way.
    """
    info = bundle.info
    per_head_block = info.num_heads * info.head_dim
    out = int(getattr(attn.q_proj, "out_features", -1))
    if out == per_head_block:
        return 1
    if out == 2 * per_head_block:
        return 2
    raise ProtocolError(
        "P4",
        f"the q-projection at layer {layer} of {bundle.key!r} emits {out} "
        f"features, which is neither {per_head_block} (heads·head_dim) nor "
        f"{2 * per_head_block} (a gated family's [q | gate] per head). This "
        "backend cannot say which columns are the queries — extend the "
        "attention tap table in pytorch_hooks/sites.py for this family.",
    )


def _attention_interior_site(
    bundle: Any,
    attn: Any,
    component: str,
    layer: int,
    head: int | None,
    tap: Any,
) -> ResolvedSite:
    """Resolve one module-boundary tap inside the mixer.

    This is the per-family attention table follow-up F5 should absorb: the
    family differences are *here*, in one function, rather than spread as
    ``hasattr`` chains through ``resolve_site``.

    📐 The three families, measured:

    =========================  ==================  =============  ============
    ``self_attn`` children     qwen3.5-moe         tiny-llama     tiny-gpt2
    =========================  ==================  =============  ============
    projections                q,k,v,o_proj        q,k,v,o_proj   ``c_attn``
    ``q_norm``/``k_norm``      ✅ (b,s,H,d)        ❌             ❌
    ``q_proj`` width           H·2·d = 512         H·d = 16       — (fused qkv)
    =========================  ==================  =============  ============
    """
    if bundle.is_gpt2_family:
        # D4: GPT-2 fuses q, k and v into one `c_attn` projection, so every one
        # of these components is a chunk of it rather than a module boundary.
        # Splitting that is the declarative family table's work (follow-up F5)
        # and buys nothing for the Qwen3.6 target, so it is refused by name
        # rather than half-implemented.
        raise NotImplementedError(
            f"component {component!r} needs separate q/k/v projections, and "
            f"this mixer fuses them into one 'c_attn' (children="
            f"{sorted(name for name, _ in attn.named_children())}). Splitting a "
            "fused qkv projection is the per-family tap table (follow-up F5); "
            "'attention_premix' and 'attention_output' read on this family "
            "today."
        )
    feature_slice = _head_slice(bundle, component, head)
    if component == "attention_value_states":
        # 📐 v_proj's output, (b, s, H_kv·d) — already flat, no head axis to
        # keep. The tap is BEFORE `past_key_values.update`, so a write here is
        # the one interior write that reaches the cache.
        return tap(attn.v_proj, "out", feature_slice=feature_slice)
    if component == "attention_gate":
        if _q_projection_splits(bundle, attn, layer) != 2:
            raise ProtocolError(
                "P4",
                f"component 'attention_gate' at layer {layer} of "
                f"{bundle.key!r}: this mixer computes no output gate. The box "
                "exists only on the gated-attention family (Qwen3.5/3.6), "
                "whose q-projection emits [q | gate] per head and which "
                "multiplies the mixer's output by sigmoid(gate) before "
                "projecting out. On this family there is no such tensor to "
                "read or write.",
            )
        return tap(attn.q_proj, "out", feature_slice=feature_slice)
    norm_name, proj_name = (
        ("q_norm", "q_proj")
        if component == "attention_query_pre_rope"
        else ("k_norm", "k_proj")
    )
    norm = getattr(attn, norm_name, None)
    if norm is not None:
        # 📐 q_norm/k_norm run before apply_rotary_pos_emb, so their output IS
        # the pre-RoPE projection — and they emit (b, s, H, d), a kept head axis.
        return tap(norm, "out", feature_slice=feature_slice, keeps_head_axis=True)
    if _q_projection_splits(bundle, attn, layer) != 1:
        raise ProtocolError(
            "P4",
            f"component {component!r} at layer {layer} of {bundle.key!r}: this "
            f"mixer has no {norm_name!r}, so the projection's output would have "
            "to be the pre-RoPE tensor — but that projection is fused "
            "([q | gate] per head), so its output is not the queries alone. "
            "Addressing a split of a projection with no norm to tap after it "
            "is the per-family tap table (follow-up F5).",
        )
    return tap(getattr(attn, proj_name), "out", feature_slice=feature_slice)


#: The mixer's interior *inside the attention function* — round 2.3.
#:
#: 📐 These four are not module boundaries: ``transformers`` computes them within
#: one ``attention_interface(...)`` call, so ``query`` and ``key`` are its
#: arguments (post-RoPE, and for ``key`` before ``repeat_kv``), the scores are
#: the softmax's input inside it, and ``z`` is its return. See
#: :mod:`causalab.neural.pytorch_hooks.attention_interface`.
ATTENTION_FUNCTION_SLOTS: dict[str, str] = {
    "attention_query": "query",
    "attention_key": "key",
    "attention_scores": "scores",
    "attention_z": "z",
}


#: Taps a write may only **replace**, because the tensor is a normalized
#: distribution and nothing downstream restores that property.
#:
#: This is the distinction ``attention_scores`` exists to remove. Both tensors
#: have the same axes — ``(batch, head, query, key)`` — and a delta, a scale or a
#: clamp on either one is arithmetically fine; the difference is entirely in what
#: happens *next*. After the pattern: the value multiply, which assumes rows
#: summing to 1 and gets whatever the edit produced. After the scores: the
#: model's own softmax, which renormalizes by construction.
#:
#: So the pattern accepts only ``swap`` (an interchange, which substitutes one
#: valid distribution for another) and the scores accept every mechanism —
#: attention knockout is ``add_scaled`` with a large negative constant, head
#: boosting is a scale — and the refusal names the alternative rather than just
#: saying no.
NORMALIZED_TAPS: dict[str, str] = {
    "attention_probs": (
        "its rows are a probability distribution and the value multiply "
        "immediately downstream assumes they sum to 1 — nothing renormalizes "
        "them after an edit. Write 'attention_scores' instead: it is the same "
        "tensor one step earlier, upstream of the model's own softmax, so every "
        "mechanism is legal there and the rows still sum to 1 by construction"
    ),
}


#: Components that only exist on a full-attention mixer. A Gated DeltaNet layer
#: has no attention matrix at all — there is nothing to read and nothing to
#: write — so naming one at such a layer is an error about the *architecture*,
#: not a missing feature (§5.3).
#: 🐞 ``attention_premix`` and ``attention_result`` belong here too, and did not
#: before. Both are the o-projection's input, and 📐 a Gated DeltaNet layer has
#: no ``o_proj`` at all — its children are
#: ``[conv1d, in_proj_a, in_proj_b, in_proj_qkv, in_proj_z, norm, out_proj]`` —
#: so naming either at such a layer raised a bare
#: ``AttributeError: 'Qwen3_5MoeGatedDeltaNet' object has no attribute 'o_proj'``
#: out of the tap table instead of the architectural refusal that says why the
#: box does not exist there. ``attention_output`` is deliberately *not* here: a
#: DeltaNet layer does produce a mixer output, and it resolves.
_FULL_ATTENTION_ONLY: frozenset[str] = frozenset(
    {"attention_probs", "attention_premix", "attention_result"}
    | _ATTENTION_INTERIOR
    | set(ATTENTION_FUNCTION_SLOTS)
)

#: The mirror (round 4): components that only exist on a Gated DeltaNet mixer.
#: A full-attention layer computes no delta-rule state — its mixer has no
#: ``in_proj_qkv``/``in_proj_z``/``out_proj`` children at all — and a family
#: with no linear stream anywhere (llama, gpt2) hits the same refusal at every
#: layer, which is the architectural refusal by name.
#: The kernel boundary *inside* the DeltaNet forward — round 4.2. 📐 These are
#: not module boundaries: the forward calls two module-global functions
#: (``causal_conv1d_fn`` and the delta-rule kernel), so the taps swap those
#: globals for the dynamic extent of the tapped mixer's forward. See
#: :mod:`causalab.neural.engines.pytorch_hooks.delta_interface`.
DELTA_KERNEL_SLOTS: dict[str, str] = {
    "delta_conv": "conv",
    "delta_query": "query",
    "delta_key": "key",
    "delta_value": "value",
    "delta_beta": "beta",
    "delta_decay": "decay",
    "delta_kernel_output": "kernel_output",
}

_LINEAR_ATTENTION_ONLY: frozenset[str] = frozenset(
    {"delta_qkv", "delta_gate", "delta_premix"} | set(DELTA_KERNEL_SLOTS)
)


def _check_stream(bundle: Any, component: str, spec: SiteSpec, layer: int) -> None:
    """Refuse a site whose stream the layer does not carry, before hooking.

    Two ways to get this wrong, and both are caught here rather than as an
    AttributeError from inside a hook:

    * the site *declares* a ``stream`` the layer does not have — ``stream`` has
      parsed since ``schema.py`` gained it and nothing read it until now (§5.2);
    * the site names a full-attention-only component at a linear-attention
      layer, which no ``stream`` spelling can make true (§5.3).
    """
    actual = bundle.stream_at(layer)
    declared = spec.stream if isinstance(spec.stream, str) else None
    if declared is not None and declared != actual:
        raise ProtocolError(
            "P4",
            f"site names stream {declared!r} at layer {layer}, but that layer "
            f"carries {actual!r} — this is a hybrid tower ({', '.join(bundle.streams)}), "
            "so the stream is a per-layer fact, not a model-wide one",
        )
    if component in _FULL_ATTENTION_ONLY and actual != "full_attention":
        raise ProtocolError(
            "P4",
            f"component {component!r} needs a full-attention mixer, but layer "
            f"{layer} of {bundle.key!r} carries {actual!r} — a Gated DeltaNet "
            "block computes no attention matrix, so there is no such tensor at "
            f"this layer. This tower is ({', '.join(bundle.streams)}).",
        )
    if component in _LINEAR_ATTENTION_ONLY and actual != "linear_attention":
        raise ProtocolError(
            "P4",
            f"component {component!r} needs a Gated DeltaNet (linear-attention) "
            f"mixer, but layer {layer} of {bundle.key!r} carries {actual!r} — a "
            "gated-attention mixer computes no delta-rule state, so there is no "
            f"such tensor at this layer. This tower is "
            f"({', '.join(bundle.streams)}).",
        )


#: The MoE surface round 1 exposes. Every one of these is a plain module output
#: (or input) — see §2.1 of the plan note: the router is a module returning a
#: 3-tuple and the experts are a fused module, so none of this needs the ragged
#: value shape that the per-expert interior (``expert_output``) does.
_MOE_COMPONENTS: frozenset[str] = frozenset(
    {
        "router_logits",
        "router_scores",
        "expert_idx",
        "routed_output",
        "shared_expert_gate_proj",
        "shared_expert_up_proj",
        "shared_expert_activation",
        "shared_expert_output",
        "shared_expert_gate",
    }
)


def _moe_site(
    bundle: Any, mlp: Any, component: str, spec: SiteSpec, layer: int
) -> ResolvedSite:
    """Resolve one MoE tap.

    📐 Every tap here is ``flat_td``: ``Qwen3_5MoeSparseMoeBlock`` reshapes to
    ``(-1, hidden)`` before the router, so the whole interior is flattened over
    (batch, position) and only the block's own input and output are contract
    shaped. Measured on ``tiny-random/qwen3.5-moe`` at 1x6 tokens, hidden 8,
    128 experts, top-10::

        gate       out -> ((6,128) logits, (6,10) scores, (6,10) int64 indices)
        experts    out -> (6, 8)
        shared_expert.gate_proj / up_proj out -> (6, 32)
        shared_expert.down_proj       in  -> (6, 32)
        shared_expert                 out -> (6, 8)
        shared_expert_gate            out -> (6, 1)

    The router is the reason ``tuple_index`` exists: ``Qwen3_5MoeTopKRouter``
    returns three tensors and the historical "element 0 of a tuple" rule would
    have silently handed back the logits for all three.
    """
    if not hasattr(mlp, "gate") or not hasattr(mlp, "experts"):
        raise NotImplementedError(
            f"component {component!r} needs a sparse-MoE block at layer {layer}, "
            f"but this MLP (children="
            f"{sorted(name for name, _ in mlp.named_children())}) is not one — "
            "extend the tap table in pytorch_hooks/sites.py."
        )
    # The `expert` sub-axis selects one of `num_experts` experts, which none of
    # these tensors is indexed by: the router's axes are all-experts (logits) or
    # top-k (scores, indices), and the shared expert is not one of the routed
    # ones. Refusing beats silently ignoring it — the mistake `stream` made.
    if spec.expert is not None:
        raise ProtocolError(
            "P4",
            f"site names expert {spec.expert!r} on component {component!r}, "
            "which has no per-expert axis: the router's axes are all-experts "
            "or top-k, and the shared expert is not one of the routed experts. "
            "The per-expert interior is 'expert_output' (follow-up F2).",
        )

    shape = component_shape(bundle.info, component)

    def flat(module: Any, kind: str, tuple_index: int | None = None) -> ResolvedSite:
        return ResolvedSite(
            module=module,
            kind=kind,
            layer=layer,
            component=component,
            shape=shape,
            tuple_index=tuple_index,
        )

    if component == "router_logits":
        return flat(mlp.gate, "out", 0)
    if component == "router_scores":
        return flat(mlp.gate, "out", 1)
    if component == "expert_idx":
        return flat(mlp.gate, "out", 2)
    if component == "routed_output":
        # the fused experts module, already combined over the top-k
        return flat(mlp.experts, "out")
    shared = getattr(mlp, "shared_expert", None)
    if shared is None:
        raise NotImplementedError(
            f"component {component!r} needs a shared expert, which this MoE "
            f"block at layer {layer} does not have."
        )
    if component == "shared_expert_gate_proj":
        return flat(shared.gate_proj, "out")
    if component == "shared_expert_up_proj":
        return flat(shared.up_proj, "out")
    if component == "shared_expert_activation":
        # down_proj's INPUT: silu(gate_proj(x)) * up_proj(x), the one tensor the
        # shared expert never exposes as a module output of its own
        return flat(shared.down_proj, "in")
    if component == "shared_expert_output":
        return flat(shared, "out")
    if component == "shared_expert_gate":
        return flat(mlp.shared_expert_gate, "out")
    raise ProtocolError("P4", f"unhandled MoE component {component!r}")


def resolve_site(bundle: Any, spec: SiteSpec) -> ResolvedSite:
    """Resolve one site record to its tap. Refuses honestly on components
    this engine does not implement yet."""
    component = spec.component
    if not isinstance(component, str):
        raise ProtocolError("P2", f"unresolved site component {component!r}")
    layer = spec.layer if isinstance(spec.layer, int) else 0
    head = spec.head if isinstance(spec.head, int) else None

    def tap(
        module: Any,
        kind: str,
        *,
        feature_slice: slice | None = None,
        tuple_index: int | None = None,
        keeps_head_axis: bool = False,
        interface_slot: str | None = None,
        shape: FeatureShape | None = None,
        derivation: str | None = None,
    ) -> ResolvedSite:
        """One tap, with its shape read from the component table.

        The shape is resolved *here* rather than at each branch so that adding a
        component means adding a table entry and a module, never a third place
        that has an opinion about the tensor's axes.

        ``keeps_head_axis`` is the one thing a tap may say about its own shape,
        and it is a fact about the *module*, not the component: 📐
        ``Qwen3_5MoeAttention.q_norm`` emits ``(b, s, H, d)`` while llama's bare
        ``q_proj`` emits ``(b, s, H·d)``. Same component, same width, same head
        space — only the packing differs, and packing is the half of the
        descriptor the backend owns. Everything the protocol layer validates
        against is family-independent and stays in the one table.
        """
        # `shape` is overridden only by a *derived* component, whose tap
        # captures a different tensor than the one the component names.
        if shape is None:
            shape = component_shape(bundle.info, component)
        if keeps_head_axis:
            shape = dataclasses.replace(shape, flat_inner=False)
        return ResolvedSite(
            module=module,
            kind=kind,
            shape=shape,
            feature_slice=feature_slice,
            layer=layer,
            component=component,
            tuple_index=tuple_index,
            interface_slot=interface_slot,
            head=head,
            derivation=derivation,
        )

    if component == "input_ids":
        # the ids themselves, taken as the embedding's INPUT: that is the one
        # module boundary they cross, and a forward pre-hook already exists for
        # the "in" side. Read-only and layer-less (§5.4); the shape has no
        # feature axis at all, only one integer per position.
        return tap(_embedding(bundle), "in")
    if component == "embeddings":
        return tap(_embedding(bundle), "out")
    if component == "lm_head":
        return tap(bundle.model.lm_head, "out")
    if component == "ln_final":
        module = (
            bundle.model.transformer.ln_f
            if bundle.is_gpt2_family
            else bundle.model.model.norm
        )
        return tap(module, "out")

    # Order matters: the stream check runs FIRST so that a full-attention-only
    # component at a Gated DeltaNet layer refuses with the architectural reason
    # ("there is no attention matrix here") rather than the temporary one ("this
    # engine has not implemented it yet"). The first is permanent and true even
    # after PR4 lands attention_probs; the second is a roadmap statement.
    _check_stream(bundle, component, spec, layer)

    if component == "attention_probs":
        # element 1 of the mixer's (attn_output, attn_weights). Reading is an
        # ordinary tap; WRITING is not — see the attention_probs module, which
        # owns that half. Its shape has two position axes and so no contract
        # form, which is what makes the executor refuse to gather or featurize
        # it without any component name appearing in that refusal.
        return tap(_attn(bundle, layer), "out", tuple_index=1, interface_slot="probs")
    if component == "expert_output":
        raise NotImplementedError(
            f"the pytorch_hooks engine has no tap for {component!r} yet — "
            "expert_output names the per-expert loop interior, which needs the "
            "ragged value shape (follow-up F2). The rest of the MoE surface — "
            "the router, the combined routed output and the shared expert — "
            "resolves; see _moe_site (sites.py)."
        )

    block = _blocks(bundle)[layer]
    if component == "attention_input_norm":
        # input_layernorm's OUTPUT: what the mixer actually consumes
        return tap(block.input_layernorm, "out")
    if component == "block_mid":
        # post_attention_layernorm's INPUT is resid_mid — the residual stream
        # after the mixer has been added and before the MLP branch
        return tap(block.post_attention_layernorm, "in")
    if component == "mlp_input_norm":
        # ...and its OUTPUT is what the MoE block consumes
        return tap(block.post_attention_layernorm, "out")
    if component == "block_output":
        return tap(block, "out")
    if component == "block_input":
        return tap(block, "in")
    if component == "attention_output":
        return tap(_attn(bundle, layer), "out")
    if component == "attention_premix":
        return tap(
            _o_proj(bundle, layer),
            "in",
            feature_slice=_head_slice(bundle, component, head),
        )
    if component == "attention_result":
        # 📐 The model never computes this. Each head's contribution to the
        # residual stream is `premix[..., h·d:(h+1)·d] @ W_o[:, h·d:(h+1)·d].T`,
        # and the model forms only their sum — by projecting the whole premix at
        # once. So the tap is `attention_premix`'s, and the value is derived
        # from it after the position gather, which keeps the cost
        # `n_positions · H · hidden` rather than `seq · H · hidden`.
        #
        # No `feature_slice`: a `head` here selects in the *result's* space
        # (hidden-wide blocks), not the captured tensor's (head_dim-wide), and
        # naming one makes the derivation cheap rather than making it a slice.
        if head is not None:
            _head_slice(bundle, component, head)  # bound-check, discard
        return tap(
            _o_proj(bundle, layer),
            "in",
            shape=component_shape(bundle.info, "attention_premix"),
            derivation="attention_result",
        )
    if component in _LINEAR_ATTENTION_ONLY:
        # The DeltaNet mixer's module boundaries (round 4.1). 📐 All three are
        # ordinary nn.Module sides: in_proj_qkv/in_proj_z fire before the conv
        # and the kernel, out_proj's input is the post-norm post-gate mixer
        # value — the exact analogue of attention_premix, hence the name. The
        # conv1d MODULE never fires (measured: the forward calls the
        # causal_conv1d_fn global instead), which is why the conv output and
        # the kernel boundary are function taps (round 4.2), not module taps.
        mixer = _attn(bundle, layer)
        if component in DELTA_KERNEL_SLOTS:
            # no module boundary: the tensor is an argument or return of the
            # kernel-boundary globals. The mixer is carried as the module — it
            # is what identifies *which* forward's calls to tap.
            if component in ("delta_conv",):
                if head is not None:
                    _head_slice(bundle, component, head)  # refuses, by shape
                return tap(mixer, "delta", interface_slot=DELTA_KERNEL_SLOTS[component])
            return tap(
                mixer,
                "delta",
                feature_slice=_head_slice(bundle, component, head),
                interface_slot=DELTA_KERNEL_SLOTS[component],
            )
        if component == "delta_qkv":
            # no head axis: the fused [q | k | v] widths are unequal (the
            # shape's note says where the per-head faces live)
            if head is not None:
                _head_slice(bundle, component, head)  # refuses, by shape
            return tap(mixer.in_proj_qkv, "out")
        if component == "delta_gate":
            return tap(
                mixer.in_proj_z,
                "out",
                feature_slice=_head_slice(bundle, component, head),
            )
        return tap(
            mixer.out_proj,
            "in",
            feature_slice=_head_slice(bundle, component, head),
        )
    if component in _ATTENTION_INTERIOR:
        return _attention_interior_site(
            bundle, _attn(bundle, layer), component, layer, head, tap
        )
    if component in ATTENTION_FUNCTION_SLOTS:
        # No module boundary to hook: these four live inside one call. The
        # module is carried anyway, because it is what identifies *which*
        # mixer's call to tap.
        return tap(
            _attn(bundle, layer),
            "interface",
            feature_slice=_head_slice(bundle, component, head),
            interface_slot=ATTENTION_FUNCTION_SLOTS[component],
        )
    mlp = block.mlp
    if component in _MOE_COMPONENTS:
        return _moe_site(bundle, mlp, component, spec, layer)
    if component == "mlp_input":
        return tap(mlp, "in")
    if component == "mlp_output":
        return tap(mlp, "out")
    if component == "mlp_activation":
        if bundle.is_gpt2_family:
            return tap(mlp.c_proj, "in")
        if hasattr(mlp, "act_fn"):
            return tap(mlp.act_fn, "out")
        raise NotImplementedError(
            f"mlp_activation: this MLP (children="
            f"{sorted(name for name, _ in mlp.named_children())}) matches no "
            "known family — extend the tap table in pytorch_hooks/sites.py "
            "(and mirror it in the hook oracle)."
        )
    raise ProtocolError("P4", f"unknown component {component!r}")
