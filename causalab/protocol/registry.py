"""Static model metadata — the widths canonicalization derives from.

Deriving featurizer widths and param shapes (spec §6) needs the model's
*static configuration* (hidden size, depth, head counts), never its weights.
This registry keeps that metadata deterministic and offline: entries for the
models the repo actually uses are declared here as data, tests register
their tiny-random models, and an HF config can be adapted explicitly with
:func:`model_info_from_hf_config` when a caller opts in — the protocol layer
itself never touches the network.

Shapes per component (the ``(model, site) → d`` rule of §2.5, and more).
:func:`component_shape` answers with a
:class:`~causalab.protocol.shapes.FeatureShape` rather than an integer, because
four questions turn on the same fact and used to be answered in four places:
how wide the feature axis is, whether there is one at all, how many heads
``head`` may name, and how the module's native tensor relates to the executor's
``(batch, position, feature)`` contract.

===================================  =======================================
component                            shape
===================================  =======================================
``embeddings``, ``block_input``,     ``(batch, position, hidden)`` — the
``block_output``, ``attention_output``,   residual stream. The three norm taps
``mlp_input``, ``mlp_output``,       are here because an RMSNorm maps the
``ln_final``, ``attention_input_norm``,   residual stream to itself, so both
``block_mid``, ``mlp_input_norm``,   its sides are hidden-wide
``expert_output``
``mlp_activation``                   ``(batch, position, intermediate)`` (the
                                     family caveat of *which* tensor this
                                     names lives in the backend, not here)
``attention_value``                  ``(batch, position, heads·head_dim)``,
                                     head-major and already flattened — the
                                     o-projection's input, query-head space
``lm_head``                          ``(batch, position, vocab)``
``routed_output``,                   ``(batch·position, hidden)`` — hidden-wide,
``shared_expert_output``             but flattened like the rest of the MoE
                                     interior
``router_logits``                    ``(batch·position, num_experts)``
``router_scores``                    ``(batch·position, top_k)``, **ranking**:
                                     column *k* is the *k*-th ranked expert, a
                                     different expert for different tokens, so
                                     a basis fitted across positions is fitted
                                     across a shuffled basis. Basis-fitting
                                     featurizers are refused; per-column ones
                                     are not
``expert_idx``                       ``(batch·position, top_k)``, **integral**:
                                     a routing table of integer expert ids —
                                     no featurizer, no gradient
``shared_expert_gate_proj``,         ``(batch·position, shared_inner)``
``shared_expert_up_proj``,
``shared_expert_activation``
``shared_expert_gate``               ``(batch·position, 1)`` — one mixing
                                     scalar per token
``input_ids``                        ``(batch, position)``, **integral**: no
                                     feature axis at all, so not a feature
                                     space in any sense
``attention_probs``                  ``(batch, head, position[query],
                                     key_position[key])`` — **two position
                                     axes**, so no contract form. Every
                                     refusal the executor makes about it is
                                     derived from that
===================================  =======================================
"""

from __future__ import annotations

import dataclasses
from typing import Any

from causalab.protocol import shapes
from causalab.protocol.errors import ValidationError
from causalab.protocol.shapes import FeatureShape

__all__ = [
    "ModelInfo",
    "component_shape",
    "component_width",
    "get_model_info",
    "head_space_refusal",
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
        intermediate_size=_intermediate_size(text, hidden),
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


#: Components whose tensor is the residual stream: an RMSNorm maps it to itself,
#: and both MoE branches write into it, so every one of these is hidden-wide.
_HIDDEN_COMPONENTS: frozenset[str] = frozenset(
    {
        "embeddings",
        "block_input",
        "block_output",
        "attention_output",
        "mlp_input",
        "mlp_output",
        "ln_final",
        "expert_output",
        "attention_input_norm",
        "block_mid",
        "mlp_input_norm",
    }
)

#: Both MoE branches write into the residual stream, so both are hidden-wide —
#: but the block reshapes to ``(-1, hidden)`` before the router, so their
#: tensors are flattened over (batch, position) like the rest of its interior.
#: 🐞 The width table and the layout table used to say these two things in
#: different places and the descriptor now says them together; they had drifted,
#: and nothing checked, because no code compared a declared width against a real
#: tensor.
_FLAT_HIDDEN_COMPONENTS: frozenset[str] = frozenset(
    {"routed_output", "shared_expert_output"}
)

_SHARED_EXPERT_INNER: frozenset[str] = frozenset(
    {
        "shared_expert_gate_proj",
        "shared_expert_up_proj",
        "shared_expert_activation",
    }
)


def _intermediate_size(text: Any, hidden: int) -> int:
    """The MLP's inner width, resolved the way the modeling code resolves it.

    🐞 Reading ``intermediate_size`` unconditionally is wrong on the GPT-2
    family: ``GPT2Config`` spells the field ``n_inner`` and the block computes
    ``config.n_inner if config.n_inner is not None else 4 * hidden_size``
    (transformers ``models/gpt2/modeling_gpt2.py:250``), ignoring any
    ``intermediate_size`` in the config. 📐 ``hf-internal-testing/tiny-random-gpt2``
    carries a stray ``intermediate_size: 37`` next to ``n_inner: null`` and a
    128-wide MLP, so the adapter reported 37 for a tensor that is 128 wide — a
    featurizer on ``mlp_activation`` would have been sized against nothing. It
    went unnoticed because no code compared a declared width to a real tensor
    until :func:`component_shape` did.
    """
    if hasattr(text, "n_inner"):  # the GPT-2 family's spelling, authoritative
        n_inner = getattr(text, "n_inner")
        return int(n_inner) if n_inner is not None else 4 * hidden
    return int(getattr(text, "intermediate_size", None) or 4 * hidden)


def component_shape(info: ModelInfo, component: str) -> FeatureShape:
    """The axes of one component's tensor (the table in the module docstring).

    This is the single description everything else derives from: the feature
    width, whether a featurizer may attach, whether ``head`` means anything and
    how many heads it selects among, and the native↔contract conversion the
    backend performs. It replaced a set of parallel answers — a width function
    with hand-written refusal texts, a five-string layout vocabulary, and a head
    bound that read ``info.num_heads`` regardless of component — that could and
    did disagree with each other.
    """
    if component in _HIDDEN_COMPONENTS:
        return shapes.bsd(info.hidden_size)
    if component in _FLAT_HIDDEN_COMPONENTS:
        return shapes.flat_td(info.hidden_size)
    if component == "mlp_activation":
        return shapes.bsd(info.intermediate_size)
    if component == "attention_value":
        # the per-head o-projection input: query-head space (num_heads *
        # head_dim = the o_proj input width), NOT the GQA KV-head space of
        # v_proj. Head-major and already flattened — `(b, s, H*d)`.
        return shapes.bs_flat_heads(info.num_heads, info.head_dim)
    if component == "lm_head":
        return shapes.bsd(info.vocab_size)
    if component == "attention_probs":
        return shapes.attention_pattern(
            info.num_heads,
            note=(
                "Round 1 exposes the whole pattern, which is what an "
                'interchange on attention needs: read it at pos: "all", '
                "without a featurizer and without 'dims'."
            ),
        )
    if component == "input_ids":
        return shapes.bs(
            integral=True,
            note=(
                "Read it directly, or read 'embeddings' if you want the vector "
                "the ids look up."
            ),
        )
    if component == "router_logits":
        if info.num_experts is None:
            raise ValidationError(
                4, f"model {info.key!r} declares no experts; router_logits has no width"
            )
        return shapes.flat_td(info.num_experts)
    if component in ("router_scores", "expert_idx"):
        if info.num_experts_per_tok is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no num_experts_per_tok; "
                f"{component} has no width",
            )
        if component == "expert_idx":
            return shapes.flat_topk(
                info.num_experts_per_tok,
                integral=True,
                note=(
                    "It is the MoE routing table: integer expert ids on a "
                    "top-k axis. Read or write it directly to inspect or edit "
                    "routing."
                ),
            )
        # ⚠️ Dimensionally well defined, and a plain read of it is meaningful —
        # but column *k* is the *k*-th ranked expert, a different expert for
        # different tokens, so the axis is a ranking rather than a basis. That
        # is what `ranking` says, and what makes `subspace`/`pca`/`sae` refuse.
        return shapes.flat_topk(
            info.num_experts_per_tok,
            ranking=True,
            note=(
                "Its axis is a per-token ranking: column k is the k-th ranked "
                "expert, a different expert for different tokens, so a basis "
                "fitted across positions is fitted across a shuffled basis. "
                "Read it directly, or featurize 'router_logits', whose axis is "
                "the fixed all-experts one."
            ),
        )
    if component in _SHARED_EXPERT_INNER:
        if info.shared_expert_intermediate_size is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no shared-expert inner width; "
                f"{component} has no width",
            )
        return shapes.flat_td(info.shared_expert_intermediate_size)
    if component == "shared_expert_gate":
        # one scalar per token: how much of the shared expert to mix in
        return shapes.flat_td(1)
    raise ValidationError(
        4,
        f"component {component!r} has no declared feature shape — the protocol "
        "layer cannot size it, and featurizers cannot attach to it",
    )


def head_space_refusal(component: str, head: int, shape: FeatureShape) -> str:
    """Why ``head`` does not apply to ``component`` — the §2.2 refusal.

    Shared by the canonicalizer (which refuses at load) and
    :func:`component_width` (which refuses if anything reaches it another way),
    so the two cannot drift into disagreeing about what a head means.
    """
    return (
        f"component {component!r} has no head axis — its shape is "
        f"{shape.describe()} — so head {head} would be validated and then "
        "silently dropped. Name a component that has heads "
        "('attention_value'), or drop the 'head' field."
    )


def component_width(info: ModelInfo, component: str, *, head: int | None = None) -> int:
    """The feature width at one site.

    A thin reading of :func:`component_shape`: the product of the feature axes,
    or one head's slice of it. Kept as a function because three call sites want
    exactly this number and nothing else about the shape.
    """
    shape = component_shape(info, component)
    if not shape.is_feature_space:
        raise ValidationError(4, shape.refusal(f"component {component!r}"))
    width = shape.width
    assert width is not None  # is_feature_space implies a feature axis
    if head is None:
        return width
    space = shape.head_space
    if space is None:
        raise ValidationError(4, head_space_refusal(component, head, shape))
    return width // space


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
