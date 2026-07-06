"""Architecture descriptors for the path-patching engine.

A descriptor captures the small set of architecture facts pyvene does not
track: how a decoder block wires its branches into the residual stream
("trunk"), where each branch's contribution to the trunk is defined, and how
the final logits are produced. Module *paths* for activation capture delegate
to pyvene's per-family component mappings wherever pyvene knows them; the
descriptor only names modules pyvene has no vocabulary for (per-branch
post-norms, the final norm, the LM head).

Nothing in a descriptor is trusted: `guards.run_construction_guards` verifies
the declared wiring empirically (additivity of trunk contributions, per-layer
branch reconstruction) before the engine will patch anything.

Supported families: gpt2, gpt_neox (sequential or parallel residual), llama,
gemma2. New families need a `_FamilySpec` entry plus a passing guard run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

BlockOrder = Literal["sequential", "parallel"]
AttentionStyle = Literal["fused-qkv-absolute", "rotary", "rotary-gqa"]
NormKind = Literal["layernorm", "rmsnorm"]

__all__ = [
    "ArchitectureDescriptor",
    "AttentionStyle",
    "BlockOrder",
    "SUPPORTED_MODEL_TYPES",
    "resolve_descriptor",
]


@dataclass(frozen=True)
class _FamilySpec:
    """Static per-family module naming (relative attribute paths)."""

    layers_path: str  # container of decoder blocks, on the CausalLM model
    final_norm_path: str
    lm_head_path: str
    embed_dropout_path: str | None  # module whose output is the embed contribution
    attn_out_proj: str  # on a block: the attention out-projection
    mlp_path: str  # on a block: the MLP module
    mlp_down_proj: str  # on a block: the MLP output projection
    mlp_pre_norm: str  # on a block: norm applied to the MLP branch input
    attn_post_norm: str | None  # branch post-norm (Gemma-2 style), or None
    mlp_post_norm: str | None
    out_proj_is_conv1d: bool  # GPT-2 Conv1D (x @ W) vs nn.Linear (x @ W.T)
    attention_style_default: AttentionStyle
    norm_kind: NormKind


_FAMILIES: dict[str, _FamilySpec] = {
    "gpt2": _FamilySpec(
        layers_path="transformer.h",
        final_norm_path="transformer.ln_f",
        lm_head_path="lm_head",
        embed_dropout_path="transformer.drop",
        attn_out_proj="attn.c_proj",
        mlp_path="mlp",
        mlp_down_proj="mlp.c_proj",
        mlp_pre_norm="ln_2",
        attn_post_norm=None,
        mlp_post_norm=None,
        out_proj_is_conv1d=True,
        attention_style_default="fused-qkv-absolute",
        norm_kind="layernorm",
    ),
    "gpt_neox": _FamilySpec(
        layers_path="gpt_neox.layers",
        final_norm_path="gpt_neox.final_layer_norm",
        lm_head_path="embed_out",
        embed_dropout_path="gpt_neox.emb_dropout",
        attn_out_proj="attention.dense",
        mlp_path="mlp",
        mlp_down_proj="mlp.dense_4h_to_h",
        mlp_pre_norm="post_attention_layernorm",
        attn_post_norm=None,
        mlp_post_norm=None,
        out_proj_is_conv1d=False,
        attention_style_default="rotary",
        norm_kind="layernorm",
    ),
    "llama": _FamilySpec(
        layers_path="model.layers",
        final_norm_path="model.norm",
        lm_head_path="lm_head",
        embed_dropout_path=None,
        attn_out_proj="self_attn.o_proj",
        mlp_path="mlp",
        mlp_down_proj="mlp.down_proj",
        mlp_pre_norm="post_attention_layernorm",
        attn_post_norm=None,
        mlp_post_norm=None,
        out_proj_is_conv1d=False,
        attention_style_default="rotary",
        norm_kind="rmsnorm",
    ),
    "gemma2": _FamilySpec(
        layers_path="model.layers",
        final_norm_path="model.norm",
        lm_head_path="lm_head",
        embed_dropout_path=None,
        attn_out_proj="self_attn.o_proj",
        mlp_path="mlp",
        mlp_down_proj="mlp.down_proj",
        mlp_pre_norm="pre_feedforward_layernorm",
        attn_post_norm="post_attention_layernorm",
        mlp_post_norm="post_feedforward_layernorm",
        out_proj_is_conv1d=False,
        attention_style_default="rotary",
        norm_kind="rmsnorm",
    ),
}

SUPPORTED_MODEL_TYPES = tuple(sorted(_FAMILIES))


def _get_by_path(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


@dataclass
class ArchitectureDescriptor:
    """Resolved architecture facts + module handles for one loaded model.

    Built via :func:`resolve_descriptor`. All weight accessors return
    matrices in the ``x @ W`` convention regardless of the underlying
    parameterization (Conv1D vs Linear), so the engine's slicing code is
    family-agnostic.
    """

    model_type: str
    block_order: BlockOrder
    attention_style: AttentionStyle
    norm_kind: NormKind
    n_layers: int
    n_heads: int
    head_dim: int
    d_model: int
    d_ff: int
    final_logit_softcapping: float | None
    spec: _FamilySpec
    model: nn.Module  # the CausalLM model

    # ---------------- module handles ----------------
    def layer(self, layer_idx: int) -> nn.Module:
        return _get_by_path(self.model, self.spec.layers_path)[layer_idx]

    def final_norm(self) -> nn.Module:
        return _get_by_path(self.model, self.spec.final_norm_path)

    def lm_head(self) -> nn.Module:
        return _get_by_path(self.model, self.spec.lm_head_path)

    def mlp(self, layer_idx: int) -> nn.Module:
        return _get_by_path(self.layer(layer_idx), self.spec.mlp_path)

    def mlp_pre_norm(self, layer_idx: int) -> nn.Module:
        return _get_by_path(self.layer(layer_idx), self.spec.mlp_pre_norm)

    def attn_post_norm(self, layer_idx: int) -> nn.Module | None:
        if self.spec.attn_post_norm is None:
            return None
        return _get_by_path(self.layer(layer_idx), self.spec.attn_post_norm)

    def mlp_post_norm(self, layer_idx: int) -> nn.Module | None:
        if self.spec.mlp_post_norm is None:
            return None
        return _get_by_path(self.layer(layer_idx), self.spec.mlp_post_norm)

    # ---------------- weight accessors (x @ W convention) ----------------
    def attn_out_weight(self, layer_idx: int) -> torch.Tensor:
        """(n_heads * head_dim, d_model) so head h occupies rows
        ``h*head_dim : (h+1)*head_dim``."""
        proj = _get_by_path(self.layer(layer_idx), self.spec.attn_out_proj)
        w = proj.weight.detach()
        return w if self.spec.out_proj_is_conv1d else w.T

    def attn_out_bias(self, layer_idx: int) -> torch.Tensor | None:
        proj = _get_by_path(self.layer(layer_idx), self.spec.attn_out_proj)
        b = getattr(proj, "bias", None)
        return None if b is None else b.detach()

    def mlp_out_weight(self, layer_idx: int) -> torch.Tensor:
        """(d_ff, d_model): row n is neuron n's output direction."""
        proj = _get_by_path(self.layer(layer_idx), self.spec.mlp_down_proj)
        w = proj.weight.detach()
        return w if self.spec.out_proj_is_conv1d else w.T

    # ---------------- pyvene component names ----------------
    # Named components delegate to pyvene's per-family mapping. The one
    # capture point pyvene has no name for — the MLP output projection's
    # *input* (the per-neuron values of a gated MLP) — uses pyvene's dotted
    # module-path fallback, which is resolved by pyvene itself, not by hooks
    # of ours.
    def component_head_values(self) -> str:
        """Per-head attention value output (out-projection input)."""
        return "attention_value_output"

    def component_mlp_branch(self) -> str:
        """MLP module output (pre branch post-norm where one exists)."""
        return "mlp_output"

    def component_block_output(self) -> str:
        return "block_output"

    def component_block_input(self) -> str:
        return "block_input"

    def _layer_dotted(self, layer_idx: int) -> str:
        return f"{self.spec.layers_path}[{layer_idx}]"

    def component_neuron_values(self, layer_idx: int) -> str:
        """Input to the MLP output projection, as a dotted pyvene path."""
        return f"{self._layer_dotted(layer_idx)}.{self.spec.mlp_down_proj}.input"

    def component_mlp_pre_norm_input(self, layer_idx: int) -> str:
        """Input to the MLP branch's pre-norm (dotted pyvene path)."""
        return f"{self._layer_dotted(layer_idx)}.{self.spec.mlp_pre_norm}.input"

    def component_mlp_trunk_output(self, layer_idx: int) -> str:
        """The MLP branch's contribution to the trunk: the post-norm's output
        where one exists, else the MLP module's output (dotted path)."""
        tail = self.spec.mlp_post_norm or self.spec.mlp_path
        return f"{self._layer_dotted(layer_idx)}.{tail}.output"

    def component_final_norm_input(self) -> str:
        return f"{self.spec.final_norm_path}.input"

    # ---------------- semantics helpers ----------------
    def mlp_input_resid_is_block_input(self) -> bool:
        """True for parallel-residual blocks: the MLP reads the block input,
        so same-layer attention does NOT feed the same-layer MLP."""
        return self.block_order == "parallel"

    def head_feeds_mlp(self, head_layer: int, mlp_layer: int) -> bool:
        if self.block_order == "sequential":
            return head_layer <= mlp_layer
        return head_layer < mlp_layer

    def summary(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "block_order": self.block_order,
            "attention_style": self.attention_style,
            "norm_kind": self.norm_kind,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "final_logit_softcapping": self.final_logit_softcapping,
            "branch_post_norms": bool(self.spec.attn_post_norm),
        }


def resolve_descriptor(
    model: nn.Module,
    *,
    block_order: BlockOrder | None = None,
    attention_style: AttentionStyle | None = None,
) -> ArchitectureDescriptor:
    """Resolve a descriptor from a loaded HF CausalLM.

    Facts are read from the HF config where the config carries them
    (``use_parallel_residual``, GQA head counts, softcapping); the explicit
    keyword overrides exist for testing the guards (a deliberately
    mis-declared order must fail construction) and for configs that lie.
    """
    config = model.config
    model_type = config.model_type
    if model_type not in _FAMILIES:
        from .provenance import UnsupportedArchitectureError

        raise UnsupportedArchitectureError(
            f"path_patching has no architecture descriptor for model_type="
            f"{model_type!r}. Supported: {SUPPORTED_MODEL_TYPES}. Add a "
            f"_FamilySpec and validate it with guards.run_construction_guards."
        )
    spec = _FAMILIES[model_type]

    if block_order is None:
        if getattr(config, "use_parallel_residual", False):
            block_order = "parallel"
        else:
            block_order = "sequential"

    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    if attention_style is None:
        if spec.attention_style_default == "fused-qkv-absolute":
            attention_style = "fused-qkv-absolute"
        elif n_kv_heads < n_heads:
            attention_style = "rotary-gqa"
        else:
            attention_style = "rotary"

    d_model = config.hidden_size
    head_dim = getattr(config, "head_dim", None) or d_model // n_heads
    d_ff = getattr(config, "intermediate_size", None) or getattr(
        config, "n_inner", None
    )
    if d_ff is None:
        d_ff = 4 * d_model
    n_layers = config.num_hidden_layers

    return ArchitectureDescriptor(
        model_type=model_type,
        block_order=block_order,
        attention_style=attention_style,
        norm_kind=spec.norm_kind,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        d_model=d_model,
        d_ff=d_ff,
        final_logit_softcapping=getattr(config, "final_logit_softcapping", None),
        spec=spec,
        model=model,
    )
