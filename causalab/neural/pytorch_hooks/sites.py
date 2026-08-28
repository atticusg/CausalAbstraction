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
``attention_probs`` needs an attention-internal tap this backend does not
implement (its capability is absent, so routing already refuses documents
that write to it), and MoE components await an MoE family in the tree table.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import SiteSpec

from causalab.neural.pytorch_hooks.loading import ModelBundle
from causalab.neural.pytorch_hooks.layout import Layout

__all__ = ["ResolvedSite", "resolve_site"]


@dataclasses.dataclass(frozen=True)
class ResolvedSite:
    """One tapped location: the module, which side of it carries the
    activation, an optional feature-axis slice (per-head views), and how the
    module's own tensor shape relates to the executor's ``(batch, position,
    feature)`` contract.

    ``layout`` and ``tuple_index`` both default to the historical behaviour, so
    a tap that does not mention them is unchanged: contract-shaped, and element
    0 of a tuple payload. See :mod:`causalab.neural.pytorch_hooks.layout` for
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


def resolve_site(bundle: ModelBundle, spec: SiteSpec) -> ResolvedSite:
    """Resolve one site record to its tap. Refuses honestly on components
    this backend does not implement yet."""
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
    # backend has not implemented it yet"). The first is permanent and true even
    # after PR4 lands attention_probs; the second is a roadmap statement.
    _check_stream(bundle, component, spec, layer)

    if component in ("attention_probs", "router_logits", "expert_output"):
        raise NotImplementedError(
            f"the pytorch_hooks backend has no tap for {component!r} yet — "
            "attention_probs needs an attention-internal surface (this backend "
            "declares no writable_attention_probs capability), and the MoE "
            "components await an MoE entry in the family table (sites.py)."
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
