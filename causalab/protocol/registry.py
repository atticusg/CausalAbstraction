"""Static model metadata — the widths canonicalization derives from.

Deriving featurizer widths and param shapes (spec §6) needs the model's
*static configuration* (hidden size, depth, head counts), never its weights.
This registry keeps that metadata deterministic and offline: entries for the
models the repo actually uses are declared here as data, tests register
their tiny-random models, and an HF config can be adapted explicitly with
:func:`model_info_from_hf_config` when a caller opts in — the protocol layer
itself never touches the network.

Widths per component (the ``(model, site) → d`` rule of §2.5):

=================  =======================================================
component          feature width
=================  =======================================================
``embeddings``, ``block_input``, ``block_output``, ``attention_output``,
``mlp_input``, ``mlp_output``, ``ln_final``, ``expert_output``,
``attention_input_norm``, ``block_mid``, ``mlp_input_norm``
                   ``hidden_size`` — the three norm taps are residual-stream
                   shaped, being an RMSNorm's input or output
``mlp_activation`` ``intermediate_size`` (the family caveat of what tensor
                   this names lives in the backend, not here)
``attention_value``
                   ``head_dim`` with a ``head`` sub-axis, else
                   ``num_kv_heads * head_dim`` (KV-head space under GQA)
``lm_head``        ``vocab_size``
``router_logits``  ``num_experts``
``router_scores``  ``num_experts_per_tok`` (top-k). ⚠️ Dimensionally well
                   defined, but the axis is a per-token **ranking**, not a
                   basis: column *k* is the *k*-th ranked expert, a different
                   expert for different tokens. A `subspace` fit across
                   positions is therefore not meaningful even though it is
                   accepted here. See the plan note's follow-up list.
``routed_output``, ``shared_expert_output``
                   ``hidden_size`` — both branches write the residual stream
``shared_expert_gate_proj``, ``shared_expert_up_proj``,
``shared_expert_activation``
                   ``shared_expert_intermediate_size``
``shared_expert_gate``
                   1 — one mixing scalar per token
``expert_idx``     no static width — a routing table of integer expert ids on a
                   top-k axis (§5.4): no featurizer, no gradient
``attention_probs``
                   no static width — sequence-shaped; featurizers are
                   refused on it
``input_ids``      no static width — token ids are not a feature space at all
                   (§5.4): no featurizer, no gradient, read-only
=================  =======================================================
"""

from __future__ import annotations

import dataclasses
from typing import Any

from causalab.protocol.errors import ValidationError

__all__ = [
    "ModelInfo",
    "component_width",
    "get_model_info",
    "model_info_from_hf_config",
    "register_model",
]


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """The static facts canonicalization needs about one model."""

    key: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    native_dtype: str = "fp32"
    num_experts: int | None = None
    #: top-k: how many of ``num_experts`` each token is routed to. The width of
    #: ``router_scores``, whose axis is that top-k list.
    num_experts_per_tok: int | None = None
    #: The shared expert's inner width. Deliberately separate from
    #: ``intermediate_size``: a MoE checkpoint can carry three different inner
    #: widths (dense ``intermediate_size``, ``moe_intermediate_size`` per routed
    #: expert, and this one), and reading the wrong one is silent.
    shared_expert_intermediate_size: int | None = None


_REGISTRY: dict[str, ModelInfo] = {}


def register_model(info: ModelInfo) -> None:
    """Register (or replace) one model's static metadata."""
    _REGISTRY[info.key] = info


def get_model_info(key: str) -> ModelInfo:
    """Look up a model key; a missing entry is a load error (the alternative
    — fetching a config from the network mid-canonicalization — would make
    digests depend on connectivity)."""
    info = _REGISTRY.get(key)
    if info is None:
        raise ValidationError(
            4,
            f"model {key!r} is not in the protocol model registry — register "
            "its static config (causalab.protocol.registry.register_model, or "
            "model_info_from_hf_config on a loaded HF config)",
            path="model.key",
        )
    return info


def model_info_from_hf_config(key: str, config: Any) -> ModelInfo:
    """Adapt a loaded HF config object (its text config, on multimodal
    wrappers) into a :class:`ModelInfo`. The caller owns where the config
    came from; this function only reads attributes."""
    text = getattr(config, "text_config", None) or config
    num_heads = int(getattr(text, "num_attention_heads"))
    hidden = int(getattr(text, "hidden_size"))
    head_dim = int(getattr(text, "head_dim", None) or hidden // num_heads)
    # transformers 5 renamed this to ``dtype``; ``torch_dtype`` still resolves but
    # warns on every access, and is scheduled for removal.
    dtype = str(
        getattr(text, "dtype", None) or getattr(text, "torch_dtype", None) or "float32"
    )
    return ModelInfo(
        key=key,
        hidden_size=hidden,
        num_layers=int(getattr(text, "num_hidden_layers")),
        num_heads=num_heads,
        num_kv_heads=int(getattr(text, "num_key_value_heads", None) or num_heads),
        head_dim=head_dim,
        intermediate_size=int(getattr(text, "intermediate_size", None) or 4 * hidden),
        vocab_size=int(getattr(text, "vocab_size")),
        native_dtype={"bfloat16": "bf16", "float16": "fp16"}.get(
            dtype.removeprefix("torch."), "fp32"
        ),
        # Two spellings in the wild, and neither is universal: mixtral and
        # qwen3_moe carry both, while qwen2_moe and qwen3_5_moe carry only
        # ``num_experts``. Reading ``num_local_experts`` alone silently left
        # num_experts=None on those, which makes component_width refuse
        # router_logits on a model that plainly has a router.
        num_experts=(
            getattr(text, "num_experts", None)
            or getattr(text, "num_local_experts", None)
        ),
        num_experts_per_tok=getattr(text, "num_experts_per_tok", None),
        # ⚠️ Three spellings, and on `tiny-random/qwen3.5-moe` all three are 32,
        # so the fixture CANNOT tell a wrong choice from a right one. Ordered
        # most-specific first and never silently defaulted to the dense
        # `intermediate_size`, because that is the one that would be wrong on a
        # real checkpoint while still producing a plausible number.
        shared_expert_intermediate_size=(
            getattr(text, "shared_expert_intermediate_size", None)
            or getattr(text, "moe_intermediate_size", None)
        ),
    )


def component_width(info: ModelInfo, component: str, *, head: int | None = None) -> int:
    """The feature width at one site (the table in the module docstring)."""
    if component in (
        "embeddings",
        "block_input",
        "block_output",
        "attention_output",
        "mlp_input",
        "mlp_output",
        "ln_final",
        "expert_output",
        # the three norm taps: an RMSNorm maps the residual stream to itself,
        # so both its sides are hidden_size-wide
        "attention_input_norm",
        "block_mid",
        "mlp_input_norm",
        # both MoE branches write into the residual stream, so both are
        # hidden-wide even though their interiors are not
        "routed_output",
        "shared_expert_output",
    ):
        return info.hidden_size
    if component == "mlp_activation":
        return info.intermediate_size
    if component == "attention_value":
        # the per-head o-projection input: query-head space (n_heads * head_dim
        # = the o_proj input width), NOT the GQA KV-head space of v_proj
        return info.head_dim if head is not None else info.num_heads * info.head_dim
    if component == "lm_head":
        return info.vocab_size
    if component == "router_logits":
        if info.num_experts is None:
            raise ValidationError(
                4, f"model {info.key!r} declares no experts; router_logits has no width"
            )
        return info.num_experts
    if component == "router_scores":
        if info.num_experts_per_tok is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no num_experts_per_tok; "
                "router_scores has no width",
            )
        return info.num_experts_per_tok
    if component in (
        "shared_expert_gate_proj",
        "shared_expert_up_proj",
        "shared_expert_activation",
    ):
        if info.shared_expert_intermediate_size is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no shared-expert inner width; "
                f"{component} has no width",
            )
        return info.shared_expert_intermediate_size
    if component == "shared_expert_gate":
        # one scalar per token: how much of the shared expert to mix in
        return 1
    if component == "expert_idx":
        raise ValidationError(
            4,
            "component 'expert_idx' is a routing table, not a feature space "
            "(§5.4): it carries integer expert ids on a top-k axis, so there is "
            "no width for a featurizer to match and no gradient to train "
            "through. Read or write it directly to inspect or edit routing.",
        )
    if component == "input_ids":
        raise ValidationError(
            4,
            "component 'input_ids' is the model's token input, not a feature "
            "space (§5.4): it carries integer ids on a position axis, so it has "
            "no width for a featurizer to match, no gradient to train through, "
            "and no meaningful subspace. Read it directly, or read 'embeddings' "
            "if you want the vector the ids look up.",
        )
    raise ValidationError(
        4,
        f"component {component!r} has no static feature width — featurizers "
        "cannot attach to it",
    )


# --------------------------------------------------------------------------- #
# built-in entries — the models the repo's configs and corpus name.
# Sources: HF config.json of each checkpoint (static metadata, no weights).
# --------------------------------------------------------------------------- #

register_model(
    ModelInfo(
        key="meta-llama/Llama-3.1-8B",
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=14336,
        vocab_size=128256,
        native_dtype="bf16",
    )
)
register_model(
    ModelInfo(
        key="meta-llama/Llama-3.1-8B-Instruct",
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=14336,
        vocab_size=128256,
        native_dtype="bf16",
    )
)
register_model(
    ModelInfo(
        key="gpt2",
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        intermediate_size=3072,
        vocab_size=50257,
        native_dtype="fp32",
    )
)
register_model(
    ModelInfo(
        key="gpt2-xl",
        hidden_size=1600,
        num_layers=48,
        num_heads=25,
        num_kv_heads=25,
        head_dim=64,
        intermediate_size=6400,
        vocab_size=50257,
        native_dtype="fp32",
    )
)
register_model(
    ModelInfo(
        key="Qwen/Qwen3-4B-Instruct-2507",
        hidden_size=2560,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=9728,
        vocab_size=151936,
        native_dtype="bf16",
    )
)
register_model(
    ModelInfo(
        key="google/gemma-2-2b-it",
        hidden_size=2304,
        num_layers=26,
        num_heads=8,
        num_kv_heads=4,
        head_dim=256,
        intermediate_size=9216,
        vocab_size=256000,
        native_dtype="bf16",
    )
)
