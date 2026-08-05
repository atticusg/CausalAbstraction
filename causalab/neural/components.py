"""
components.py
=============
Resolve a causalab *component type* to a readable/writable value on the nnsight
envoy tree.

This is the replacement for pyvene's ``<family>_type_to_module_mapping`` tables.
A component type (``block_output``, ``mlp_activation``,
``head_attention_value_output``, …) names a place in a transformer; this module
turns that name plus a layer index into a :class:`Site` whose ``read()`` returns
the activation as a tensor and whose ``write()`` puts one back — both valid only
inside an ``nnsight`` trace.

Two things it does that the pyvene table could not:

* **Per-head widths come from the model, not from arithmetic.** pyvene realized
  every per-head component at ``hidden_size // num_attention_heads``, which is
  wrong whenever ``config.head_dim`` is decoupled from that ratio (Qwen3) and
  silently returned a wrong-width vector. Here the head width is
  ``config.head_dim`` and the head *count* is the count for that projection —
  ``num_key_value_heads`` for k/v, ``num_attention_heads`` for q and for the
  attention output — so grouped-query attention is addressed in its own space
  rather than remapped by the caller.
* **Value shaping is a composable chain.** Tuple element, fused-QKV slice, and
  head split are separate transforms that compose, so GPT-2's
  ``head_query_output`` (slice a third of the fused ``c_attn`` output, then split
  heads) is declarative instead of special-cased, and every one of them knows how
  to write back through itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from torch import Tensor

__all__ = [
    "Site",
    "resolve_site",
    "component_width",
    "head_layout",
    "is_head_component",
    "device_for_layer",
    "UnsupportedComponent",
]


class UnsupportedComponent(NotImplementedError):
    """Raised for a component this model family does not expose."""


Key = Literal["input", "output"]


# --------------------------------------------------------------------------- #
#  Value transforms — how to get from the hooked value to the activation       #
# --------------------------------------------------------------------------- #
class Transform:
    """One step between a module's hooked value and the activation we intervene on.

    ``forward`` narrows (tuple element, channel slice, head split); ``back``
    widens a replacement for the narrowed part into a full replacement for the
    container it came from. Every transform must round-trip:
    ``back(v, forward(v)) == v``.
    """

    def forward(self, value: Any) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def back(self, value: Any, inner: Any) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class TupleElement(Transform):
    """The value is a tuple (an attention block's ``(out, weights)``); take one element."""

    index: int = 0

    def forward(self, value: Any) -> Any:
        return value[self.index] if isinstance(value, tuple) else value

    def back(self, value: Any, inner: Any) -> Any:
        if not isinstance(value, tuple):
            return inner
        return value[: self.index] + (inner,) + value[self.index + 1 :]


@dataclass(frozen=True)
class ChannelSlice(Transform):
    """A contiguous span of the last dimension — GPT-2's fused ``c_attn`` q/k/v thirds."""

    start: int
    stop: int

    def forward(self, value: Tensor) -> Tensor:
        return value[..., self.start : self.stop]

    def back(self, value: Tensor, inner: Tensor) -> Tensor:
        out = value.clone()
        out[..., self.start : self.stop] = inner.to(out.dtype)
        return out


@dataclass(frozen=True)
class HeadSplit(Transform):
    """Expose ``[..., n_heads * head_dim]`` as ``[..., n_heads, head_dim]``.

    ``head_dim`` is the model's real head width (``config.head_dim`` when it
    declares one), never ``hidden_size // num_attention_heads``.
    """

    n_heads: int
    head_dim: int

    def forward(self, value: Tensor) -> Tensor:
        return value.unflatten(-1, (self.n_heads, self.head_dim))

    def back(self, value: Tensor, inner: Tensor) -> Tensor:
        return inner.flatten(-2)


# --------------------------------------------------------------------------- #
#  Sites                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Site:
    """A hookable value on the envoy tree, plus how to shape it.

    ``read`` / ``write`` are only meaningful inside a trace: reading parks the
    worker until the model produces the value, writing replaces it. Callers must
    read a site before writing it, and must visit sites in forward order — the
    interleaver serves each location once.
    """

    envoy: Any
    key: Key
    transforms: tuple[Transform, ...] = field(default_factory=tuple)

    # -- the raw hooked value ------------------------------------------------
    def _get_raw(self) -> Any:
        return self.envoy.output if self.key == "output" else self.envoy.input

    def _set_raw(self, value: Any) -> None:
        if self.key == "output":
            self.envoy.output = value
        else:
            self.envoy.input = value

    # -- the shaped activation ------------------------------------------------
    def read(self) -> Tensor:
        """The activation at this site, shaped for intervention."""
        value = self._get_raw()
        for transform in self.transforms:
            value = transform.forward(value)
        return value

    def write(self, activation: Tensor) -> None:
        """Replace the activation at this site, rebuilding every enclosing shape."""
        raw = self._get_raw()
        # Re-derive each intermediate so `back` has the container it narrowed from.
        stack: list[Any] = [raw]
        for transform in self.transforms[:-1]:
            stack.append(transform.forward(stack[-1]))
        value = activation
        for transform, container in zip(reversed(self.transforms), reversed(stack)):
            value = transform.back(container, value)
        self._set_raw(value)


# --------------------------------------------------------------------------- #
#  Model-family layout                                                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Family:
    """Where one architecture family keeps the modules causalab intervenes on."""

    name: str
    layers: tuple[str, ...]  # path from the root envoy to the block ModuleList
    attn: str  # block attribute holding self-attention
    mlp: str  # block attribute holding the MLP
    act: str  # activation module inside the MLP
    o_proj: str  # attention output projection; its INPUT is per-head
    attn_out: tuple[str, ...]  # path (from the block) whose OUTPUT is the
    #                            attention sublayer output
    fused_qkv: str | None  # a single q/k/v projection, if fused (GPT-2 c_attn)
    q_proj: str | None
    k_proj: str | None
    v_proj: str | None
    block_output_is_tuple: bool


LLAMA = Family(
    name="llama",
    layers=("model", "layers"),
    attn="self_attn",
    mlp="mlp",
    act="act_fn",
    o_proj="o_proj",
    attn_out=("self_attn",),
    fused_qkv=None,
    q_proj="q_proj",
    k_proj="k_proj",
    v_proj="v_proj",
    block_output_is_tuple=False,
)

GPT2 = Family(
    name="gpt2",
    layers=("transformer", "h"),
    attn="attn",
    mlp="mlp",
    act="act",
    o_proj="c_proj",
    # GPT-2's attention_output is read at `resid_dropout`'s OUTPUT, not the
    # attention module's: the sublayer output as it enters the residual stream.
    attn_out=("attn", "resid_dropout"),
    fused_qkv="c_attn",
    q_proj=None,
    k_proj=None,
    v_proj=None,
    block_output_is_tuple=True,
)


def detect_family(model: Any) -> Family:
    """Pick the layout for ``model`` (a raw HF module or an envoy)."""
    module = getattr(model, "_module", model)
    if hasattr(module, "transformer") and hasattr(module.transformer, "h"):
        return GPT2
    if hasattr(module, "model") and hasattr(module.model, "layers"):
        return LLAMA
    raise UnsupportedComponent(
        f"No component layout is registered for {type(module).__name__}. "
        f"causalab knows the Llama-family (`model.layers`) and GPT-2-family "
        f"(`transformer.h`) module trees; add a `Family` in "
        f"causalab/neural/components.py for a new architecture."
    )


# --------------------------------------------------------------------------- #
#  Head geometry                                                               #
# --------------------------------------------------------------------------- #
def head_layout(config: Any) -> tuple[int, int, int]:
    """``(num_query_heads, num_kv_heads, head_dim)`` as the model realizes them.

    ``head_dim`` is read from the config when declared. Only when it is absent
    does this fall back to ``hidden_size // num_query_heads`` — pyvene used that
    ratio unconditionally, which is the #386 miscomputation on models like Qwen3
    where the two differ.
    """
    hidden = getattr(config, "hidden_size", None) or config.n_embd
    n_q = getattr(config, "num_attention_heads", None) or config.n_head
    n_kv = getattr(config, "num_key_value_heads", None) or n_q
    head_dim = getattr(config, "head_dim", None) or (hidden // n_q)
    return int(n_q), int(n_kv), int(head_dim)


# --------------------------------------------------------------------------- #
#  Component table                                                             #
# --------------------------------------------------------------------------- #
# Each entry: component_type -> (path-from-block, key, head-space or None)
# where head-space is "q" (per query head), "kv" (per kv head), or None.
# `path-from-block` of () means the decoder block itself.
_SIMPLE: dict[str, tuple[tuple[str, ...], Key]] = {
    "block_input": ((), "input"),
    "block_output": ((), "output"),
    "mlp_input": (("mlp",), "input"),
    "mlp_output": (("mlp",), "output"),
    "mlp_activation": (("mlp", "act"), "output"),
    "attention_input": (("attn",), "input"),
    "attention_value_output": (("attn", "o_proj"), "input"),
}

# Components whose value is one third of a fused QKV projection (GPT-2) or a
# dedicated projection (Llama). Value: (which third, which projection attr, head space)
_QKV: dict[str, tuple[int, str, str]] = {
    "query_output": (0, "q_proj", "q"),
    "key_output": (1, "k_proj", "kv"),
    "value_output": (2, "v_proj", "kv"),
    "head_query_output": (0, "q_proj", "q"),
    "head_key_output": (1, "k_proj", "kv"),
    "head_value_output": (2, "v_proj", "kv"),
}

_HEAD_COMPONENTS = frozenset(
    {
        "head_attention_value_output",
        "head_query_output",
        "head_key_output",
        "head_value_output",
    }
)


# Order in which a decoder block reaches each component during its forward.
# The interleaver serves each location once and in order, so a trace that touches
# several components must request them in this order or hit `OutOfOrderError`.
# Sites are ordered globally by ``(layer, _FORWARD_RANK[component_type])``.
_FORWARD_RANK: dict[str, int] = {
    "block_input": 0,
    "attention_input": 1,
    "query_output": 2,
    "key_output": 2,
    "value_output": 2,
    "head_query_output": 2,
    "head_key_output": 2,
    "head_value_output": 2,
    "attention_value_output": 3,
    "head_attention_value_output": 3,
    "attention_output": 4,
    "mlp_input": 5,
    "mlp_activation": 6,
    "mlp_output": 7,
    "block_output": 8,
}


def forward_order(layer: int, component_type: str) -> tuple[int, int]:
    """Sort key placing ``(layer, component_type)`` in forward-pass order."""
    try:
        return (layer, _FORWARD_RANK[component_type])
    except KeyError:
        raise UnsupportedComponent(
            f"No forward-order rank for component type {component_type!r}; add one "
            f"to `_FORWARD_RANK` in causalab/neural/components.py."
        ) from None


def _walk(root: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        root = getattr(root, part)
    return root


def resolve_site(nnsight_model: Any, component_type: str, layer: int) -> Site:
    """The :class:`Site` for ``component_type`` at ``layer``.

    ``nnsight_model`` is the :class:`nnsight.TransformersModel` envoy (i.e.
    ``pipeline.nnsight``). The returned site's ``read``/``write`` are valid only
    inside a trace on that model.
    """
    family = detect_family(nnsight_model)
    config = nnsight_model.config
    block = _walk(nnsight_model, family.layers)[layer]
    n_q, n_kv, head_dim = head_layout(config)

    def attr(name: str) -> str:
        """Translate a family-neutral name to this family's attribute."""
        return {
            "attn": family.attn,
            "mlp": family.mlp,
            "act": family.act,
            "o_proj": family.o_proj,
        }.get(name, name)

    if component_type in _SIMPLE:
        path, key = _SIMPLE[component_type]
        envoy = _walk(block, tuple(attr(p) for p in path))
        transforms: tuple[Transform, ...] = ()
        # A GPT-2 block returns a tuple; its first element is the residual stream.
        if component_type == "block_output" and family.block_output_is_tuple:
            transforms = (TupleElement(0),)
        return Site(envoy, key, transforms)

    if component_type == "attention_output":
        envoy = _walk(block, family.attn_out)
        # Llama's attention module returns (out, weights); resid_dropout returns
        # a bare tensor, so the tuple step only applies to the former.
        transforms = () if len(family.attn_out) > 1 else (TupleElement(0),)
        return Site(envoy, "output", transforms)

    if component_type in ("head_attention_value_output",):
        envoy = _walk(block, (family.attn, family.o_proj))
        # o_proj's input is the concatenated per-head attention output, laid out
        # as n_query_heads * head_dim.
        return Site(envoy, "input", (HeadSplit(n_q, head_dim),))

    if component_type in _QKV:
        third, proj_attr, space = _QKV[component_type]
        n_heads = n_q if space == "q" else n_kv
        qkv_transforms: list[Transform] = []
        if family.fused_qkv is not None:
            envoy = _walk(block, (family.attn, family.fused_qkv))
            # GPT-2 packs q|k|v into one output. Widths are per-projection so a
            # fused GQA model would slice correctly too; on GPT-2 itself
            # n_kv == n_q, so the three spans are equal thirds.
            widths = (n_q * head_dim, n_kv * head_dim, n_kv * head_dim)
            start = sum(widths[:third])
            qkv_transforms.append(ChannelSlice(start, start + widths[third]))
        else:
            envoy = _walk(block, (family.attn, getattr(family, proj_attr)))
        if component_type in _HEAD_COMPONENTS:
            qkv_transforms.append(HeadSplit(n_heads, head_dim))
        return Site(envoy, "output", tuple(qkv_transforms))

    if component_type == "head_attention_value_input":
        raise UnsupportedComponent(
            "`head_attention_value_input` names no real read point: the per-head "
            "attention output is only exposed on the *input* side of the output "
            "projection, which is `head_attention_value_output`. "
            "`AttentionHead(target_output=False)` therefore has no read point. "
            "Use `target_output=True`."
        )

    raise UnsupportedComponent(
        f"Unknown component type {component_type!r}. Known: "
        f"{sorted(set(_SIMPLE) | set(_QKV) | {'attention_output', 'head_attention_value_output'})}"
    )


def component_width(config: Any, component_type: str) -> int:
    """Feature width of ``component_type`` — the last dimension of its activation.

    Per-head components report ``config.head_dim`` — *not*
    ``hidden_size // num_attention_heads``, which the two need not agree on
    (see :func:`head_layout`).
    """
    hidden = int(getattr(config, "hidden_size", None) or config.n_embd)
    _n_q, _n_kv, head_dim = head_layout(config)
    if component_type in _HEAD_COMPONENTS:
        return head_dim
    if component_type == "mlp_activation":
        intermediate = getattr(config, "intermediate_size", None) or getattr(
            config, "n_inner", None
        )
        return int(intermediate or 4 * hidden)
    return hidden


def device_for_layer(pipeline: Any, layer: int) -> Any:
    """The device a given transformer layer lives on.

    For a model loaded with ``device_map="auto"`` different layers can sit on
    different GPUs, and a tensor that participates in an operation at that layer
    — a steering vector, a featurizer's weights — must be on the same one. For a
    single-device model this is just ``model.device``.

    The intervention engine no longer needs this: it combines values with
    activations the trace already produced, so they inherit the right device.
    It remains for code that *pre-builds* a tensor for a specific layer.
    """
    import torch

    model = pipeline.model
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        return model.device
    path = f"model.layers.{layer}"
    while path:
        if path in device_map:
            return torch.device(device_map[path])
        path = path.rsplit(".", 1)[0] if "." in path else ""
    return torch.device(next(iter(device_map.values())))


def is_head_component(component_type: str) -> bool:
    """Whether ``component_type``'s activation carries a head axis."""
    return component_type in _HEAD_COMPONENTS
