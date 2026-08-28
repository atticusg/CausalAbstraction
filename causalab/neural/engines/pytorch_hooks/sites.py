"""SiteResolver: the spec's component vocabulary → concrete module taps.

Each site record resolves to ``(module, io side, feature-axis slice)`` for
raw pytorch hooks: reads are ``register_forward_hook`` on the ``out`` side
or ``register_forward_pre_hook`` on the ``in`` side; writes replace the
same tensor. The table mirrors the hook-oracle reference
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
* ``attention_value`` with a ``head`` is the ``[H*d, (H+1)*d]`` column
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
from causalab.protocol.schema import SiteSpec

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.engines.pytorch_hooks.attention_probs import ATTENTION_PROBS_LAYOUT
from causalab.neural.engines.pytorch_hooks.layout import Layout

__all__ = [
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

    ``layout`` and ``tuple_index`` both default to the historical behaviour, so
    a tap that does not mention them is unchanged: contract-shaped, and element
    0 of a tuple payload. See :mod:`causalab.neural.engines.pytorch_hooks.layout` for
    what the non-default values mean and which architectures need them.
    """

    module: Any
    kind: str  # "in" | "out"
    feature_slice: slice | None = None
    layer: int = 0
    component: str = "block_output"
    #: The module's native tensor layout; converted to/from the contract at the
    #: hook boundary rather than special-cased per component.
    layout: Layout = "bsd"
    #: Which element of a tuple payload the tap means. None keeps the historical
    #: rule (element 0 of a tuple, else the payload itself); an explicit index
    #: is required for e.g. a router returning (logits, scores, indices).
    tuple_index: int | None = None

    @property
    def depth(self) -> tuple[int, int]:
        """(layer, intra-order) — matches the protocol planner's ranks."""
        from causalab.protocol.plan import COMPONENT_RANK  # one shared table

        rank = COMPONENT_RANK.get(self.component, 100)
        if self.component in ("ln_final", "lm_head"):
            return (1_000_000, rank)
        return (self.layer, rank)


def _blocks(bundle: ModelBundle) -> Any:
    return (
        bundle.model.transformer.h
        if bundle.is_gpt2_family
        else bundle.model.model.layers
    )


def _attn(bundle: ModelBundle, layer: int) -> Any:
    """The mixer at ``layer`` — ``self_attn``, ``attn`` or ``linear_attn``.

    Was ``block.self_attn`` for every non-GPT-2 model, which AttributeErrors on
    a hybrid tower: 📐 on ``tiny-random/qwen3.5-moe`` three of four layers carry
    ``linear_attn`` (Gated DeltaNet) and only one carries ``self_attn``. The
    per-layer answer lives on the bundle (§5.2)."""
    return bundle.mixer_at(layer)


def _o_proj(bundle: ModelBundle, layer: int) -> Any:
    attn = _attn(bundle, layer)
    return attn.c_proj if bundle.is_gpt2_family else attn.o_proj


def _head_dim(bundle: ModelBundle) -> int:
    return bundle.info.head_dim


#: Components that only exist on a full-attention mixer. A Gated DeltaNet layer
#: has no attention matrix at all — there is nothing to read and nothing to
#: write — so naming one at such a layer is an error about the *architecture*,
#: not a missing feature (§5.3).
_FULL_ATTENTION_ONLY: frozenset[str] = frozenset({"attention_probs"})


def _check_stream(
    bundle: ModelBundle, component: str, spec: SiteSpec, layer: int
) -> None:
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
    bundle: ModelBundle, mlp: Any, component: str, spec: SiteSpec, layer: int
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

    def flat(module: Any, kind: str, tuple_index: int | None = None) -> ResolvedSite:
        return ResolvedSite(
            module=module,
            kind=kind,
            layer=layer,
            component=component,
            layout="flat_td",
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


def resolve_site(bundle: ModelBundle, spec: SiteSpec) -> ResolvedSite:
    """Resolve one site record to its tap. Refuses honestly on components
    this engine does not implement yet."""
    component = spec.component
    if not isinstance(component, str):
        raise ProtocolError("P2", f"unresolved site component {component!r}")
    layer = spec.layer if isinstance(spec.layer, int) else 0
    head = spec.head if isinstance(spec.head, int) else None

    if component == "input_ids":
        # the ids themselves, taken as the embedding's INPUT: that is the one
        # module boundary they cross, and a forward pre-hook already exists for
        # the "in" side. Read-only and layer-less (§5.4); `bs` layout because
        # there is no feature axis, only one integer per position.
        module = (
            bundle.model.transformer.wte
            if bundle.is_gpt2_family
            else bundle.model.model.embed_tokens
        )
        return ResolvedSite(
            module=module, kind="in", layer=0, component=component, layout="bs"
        )
    if component == "embeddings":
        module = (
            bundle.model.transformer.wte
            if bundle.is_gpt2_family
            else bundle.model.model.embed_tokens
        )
        return ResolvedSite(module=module, kind="out", layer=0, component=component)
    if component == "lm_head":
        return ResolvedSite(
            module=bundle.model.lm_head, kind="out", layer=layer, component=component
        )
    if component == "ln_final":
        module = (
            bundle.model.transformer.ln_f
            if bundle.is_gpt2_family
            else bundle.model.model.norm
        )
        return ResolvedSite(module=module, kind="out", layer=layer, component=component)

    # Order matters: the stream check runs FIRST so that a full-attention-only
    # component at a Gated DeltaNet layer refuses with the architectural reason
    # ("there is no attention matrix here") rather than the temporary one ("this
    # engine has not implemented it yet"). The first is permanent and true even
    # after PR4 lands attention_probs; the second is a roadmap statement.
    _check_stream(bundle, component, spec, layer)

    if component == "attention_probs":
        # element 1 of the mixer's (attn_output, attn_weights). Reading is an
        # ordinary tap; WRITING is not — see the attention_probs module, which
        # owns that half. `native` because this shape is (batch, heads, query,
        # key) and honestly undescribed (follow-up F1), not because it is
        # contract-shaped.
        return ResolvedSite(
            module=_attn(bundle, layer),
            kind="out",
            layer=layer,
            component=component,
            layout=ATTENTION_PROBS_LAYOUT,
            tuple_index=1,
        )
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
        return ResolvedSite(
            module=block.input_layernorm, kind="out", layer=layer, component=component
        )
    if component == "block_mid":
        # post_attention_layernorm's INPUT is resid_mid — the residual stream
        # after the mixer has been added and before the MLP branch
        return ResolvedSite(
            module=block.post_attention_layernorm,
            kind="in",
            layer=layer,
            component=component,
        )
    if component == "mlp_input_norm":
        # ...and its OUTPUT is what the MoE block consumes
        return ResolvedSite(
            module=block.post_attention_layernorm,
            kind="out",
            layer=layer,
            component=component,
        )
    if component == "block_output":
        return ResolvedSite(module=block, kind="out", layer=layer, component=component)
    if component == "block_input":
        return ResolvedSite(module=block, kind="in", layer=layer, component=component)
    if component == "attention_output":
        return ResolvedSite(
            module=_attn(bundle, layer), kind="out", layer=layer, component=component
        )
    if component == "attention_value":
        feature_slice = None
        if head is not None:
            d = _head_dim(bundle)
            feature_slice = slice(head * d, (head + 1) * d)
        return ResolvedSite(
            module=_o_proj(bundle, layer),
            kind="in",
            feature_slice=feature_slice,
            layer=layer,
            component=component,
        )
    mlp = block.mlp
    if component in _MOE_COMPONENTS:
        return _moe_site(bundle, mlp, component, spec, layer)
    if component == "mlp_input":
        return ResolvedSite(module=mlp, kind="in", layer=layer, component=component)
    if component == "mlp_output":
        return ResolvedSite(module=mlp, kind="out", layer=layer, component=component)
    if component == "mlp_activation":
        if bundle.is_gpt2_family:
            return ResolvedSite(
                module=mlp.c_proj, kind="in", layer=layer, component=component
            )
        if hasattr(mlp, "act_fn"):
            return ResolvedSite(
                module=mlp.act_fn, kind="out", layer=layer, component=component
            )
        raise NotImplementedError(
            f"mlp_activation: this MLP (children="
            f"{sorted(name for name, _ in mlp.named_children())}) matches no "
            "known family — extend the tap table in pytorch_hooks/sites.py "
            "(and mirror it in the hook oracle)."
        )
    raise ProtocolError("P4", f"unknown component {component!r}")
