"""The per-layer hybrid stream table — one answer, shared by every engine.

A hybrid architecture varies its mixer *per layer*: 📐 on
``tiny-random/qwen3.5-moe`` the text tower is ``['linear_attention',
'linear_attention', 'linear_attention', 'full_attention']``. Which stream a
layer carries is read off the module that is really there (what a tap has to
attach to), never off a family flag or the config alone — and because every
per-layer tap downstream depends on the answer, it must never diverge between
engines. Hence one module.
"""

from __future__ import annotations

from typing import Any

from causalab.protocol.errors import ProtocolError

__all__ = [
    "FULL_ATTENTION_CHILDREN",
    "LINEAR_ATTENTION_CHILDREN",
    "mixer_at",
    "stream_at",
]

#: The mixer children that mean a layer runs full (softmax) attention, and the
#: ones that mean it runs a linear-attention kernel.
FULL_ATTENTION_CHILDREN: tuple[str, ...] = ("self_attn", "attn")
LINEAR_ATTENTION_CHILDREN: tuple[str, ...] = ("linear_attn",)


def stream_at(blocks: Any, layer: int, *, key: str) -> str:
    """Which mixer stream ``blocks[layer]`` actually carries.

    Returns one of ``"full_attention"`` (a ``self_attn``/``attn`` child) or
    ``"linear_attention"`` (a ``linear_attn`` child). ``key`` names the model
    in refusals.

    Raises:
        ProtocolError: the block has no recognised mixer child, or has
            children of *both* kinds. The second case is hypothetical — no
            family in the round-1 box map ships it — but probing in a fixed
            order would answer "full_attention" for it silently, and every
            per-layer tap downstream would then attach to the wrong module
            and still produce plausible numbers. A named refusal is the same
            trade this vocabulary makes everywhere else.
    """
    block = blocks[layer]
    full = [name for name in FULL_ATTENTION_CHILDREN if hasattr(block, name)]
    linear = [name for name in LINEAR_ATTENTION_CHILDREN if hasattr(block, name)]
    if full and linear:
        raise ProtocolError(
            "P4",
            f"layer {layer} of {key!r} carries both a full-attention "
            f"child ({', '.join(full)}) and a linear-attention child "
            f"({', '.join(linear)}) — the stream of a layer must be one or "
            "the other, so extend the stream table in "
            "neural/shared/streams.py to say which this family means",
        )
    if full:
        return "full_attention"
    if linear:
        return "linear_attention"
    raise ProtocolError(
        "P4",
        f"layer {layer} of {key!r} has no recognised mixer child "
        f"(children={sorted(name for name, _ in block.named_children())}) — "
        "extend the stream table in neural/shared/streams.py",
    )


def mixer_at(blocks: Any, layer: int, *, key: str) -> Any:
    """The attention/mixer module at ``layer``, whichever stream it is.

    Resolved *through* :func:`stream_at` rather than by its own probe, so the
    two can never disagree about a block: one answer, one place."""
    names = (
        FULL_ATTENTION_CHILDREN
        if stream_at(blocks, layer, key=key) == "full_attention"
        else LINEAR_ATTENTION_CHILDREN
    )
    block = blocks[layer]
    for name in names:
        child = getattr(block, name, None)
        if child is not None:
            return child
    raise AssertionError("unreachable")  # stream_at only answers if one exists
