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
``mlp_input``, ``mlp_output``, ``ln_final``, ``expert_output``
                   ``hidden_size``
``mlp_activation`` ``intermediate_size`` (the family caveat of what tensor
                   this names lives in the backend, not here)
``attention_value``
                   ``head_dim`` with a ``head`` sub-axis, else
                   ``num_kv_heads * head_dim`` (KV-head space under GQA)
``lm_head``        ``vocab_size``
``router_logits``  ``num_experts``
``attention_probs``
                   no static width — sequence-shaped; featurizers are
                   refused on it
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
    dtype = str(getattr(text, "torch_dtype", None) or "float32")
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
        num_experts=getattr(text, "num_local_experts", None),
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
