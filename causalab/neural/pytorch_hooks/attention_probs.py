"""Reading and writing the attention pattern, which no forward hook can write.

``self_attn`` returns ``(attn_output, attn_weights)``, so **reading** the pattern
is an ordinary tap: element 1 of the mixer's output. 📐 Measured on
``tiny-random/qwen3.5-moe`` layer 3: ``(1, 8, 6, 6)`` = (batch, heads, query,
key), rows summing to 1.

**Writing** it is not, and the reason is worth stating precisely because it looks
like it should be. By the time the mixer returns, ``attn_output`` has already
been computed from the pattern *inside* the attention function::

    attn_output, attn_weights = attention_interface(self, q, k, v, mask, ...)
    ...
    return attn_output, attn_weights

So a ``register_forward_hook`` that rewrites element 1 changes a tensor nothing
downstream reads — the same silent-no-op shape as writing ``router_logits``, and
measured the same way: 0.0 change in the logits. The edit has to happen *between*
the softmax and the value multiply, which means going through the attention
function itself.

``transformers`` resolves that function per forward::

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

📐 ``"eager"`` is **not** registered by default (the registry holds
``sdpa``, the flash variants, ``flex_attention`` and the ``paged|*`` family), so
that call falls through to the module's own ``eager_attention_forward``.
Registering ``"eager"`` therefore *inserts* a wrapper rather than replacing one,
and removing the key restores the original behaviour exactly. The backend forces
eager attention at load time (``loading.py``), which is what makes this the only
implementation we need to wrap.

**What the wrapper duplicates, and why that is safe.** It calls the real eager
function, applies the edit, and then redoes only the two lines that follow the
softmax::

    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_output = torch.matmul(edited, value_states).transpose(1, 2).contiguous()

Duplicating library internals is normally how a backend rots silently. Here it is
pinned: with an **identity** edit the recomputed output must equal the
unpatched output *exactly*, which
:func:`tests...test_attention_probs.test_an_identity_edit_is_bit_identical`
asserts (📐 measured max difference 0.0). If a future transformers changes what
happens after the softmax — a different scaling, a dropout that is not a no-op in
eval, an output reshape — that test fails rather than the numbers quietly
drifting.

Round-1 scope: a write replaces the **whole** pattern, which is what an
interchange on attention does (and what nnterp's own check does:
``self[layer] = rnd``). Addressing a single query position, or a feature slice of
the key axis, needs the typed feature-shape descriptor — the feature axis here
*is* a position axis — and that is follow-up F1. Both are refused rather than
approximated.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterator, Mapping

import torch

__all__ = ["ATTENTION_PROBS_LAYOUT", "eager_attention_writes"]

#: The layout marker an ``attention_probs`` tap carries. It is deliberately NOT
#: a description of the tensor's axes — it is the *absence* of one, and the
#: conversion for it is the identity in both directions. Describing (batch,
#: heads, query, key) properly is follow-up F1; until then this marker says "this
#: tap's shape is native and undescribed" so that nothing downstream can mistake
#: it for the executor's contract.
ATTENTION_PROBS_LAYOUT = "native"


@contextlib.contextmanager
def eager_attention_writes(
    edits: Mapping[int, Callable[[torch.Tensor], None]],
) -> Iterator[None]:
    """Make in-place edits to the attention pattern actually reach the output.

    Args:
        edits: ``id(module) -> edit``, where ``edit`` mutates a
            ``(batch, heads, query, key)`` tensor **in place**. A mixer module
            absent from the mapping is untouched and pays only a dict lookup.

    The registry entry is global while installed, so this must wrap the forward
    it applies to and nothing wider. On exit the ``"eager"`` key is removed (or
    restored, if something else had registered one), which puts
    ``get_interface`` back on the module default.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if not edits:
        yield
        return

    def wrapper(
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Resolved per call rather than captured, so the wrapper always defers to
        # the implementation belonging to the model being run.
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            eager_attention_forward,
            repeat_kv,
        )

        out, weights = eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
        edit = edits.get(id(module))
        if edit is None:
            return out, weights
        edited = weights.clone()
        edit(edited)
        value_states = repeat_kv(value, module.num_key_value_groups)
        recomputed = torch.matmul(edited.to(value_states.dtype), value_states)
        return recomputed.transpose(1, 2).contiguous(), edited

    had_key = "eager" in ALL_ATTENTION_FUNCTIONS
    previous = ALL_ATTENTION_FUNCTIONS["eager"] if had_key else None
    ALL_ATTENTION_FUNCTIONS["eager"] = wrapper
    try:
        yield
    finally:
        if had_key:
            ALL_ATTENTION_FUNCTIONS["eager"] = previous
        else:
            _unregister(ALL_ATTENTION_FUNCTIONS, "eager")


def _unregister(registry: Any, name: str) -> None:
    """Remove a key from ``ALL_ATTENTION_FUNCTIONS``.

    ``AttentionInterface`` is dict-like but its deletion surface has moved
    between versions, so try the documented spelling first and fall back to the
    backing mapping. Leaving the key installed would silently keep the wrapper
    in force for the rest of the process, which is the one outcome worth being
    thorough about.
    """
    try:
        del registry[name]
        return
    except (KeyError, TypeError, AttributeError):
        pass
    backing = getattr(registry, "_local_mapping", None)
    if isinstance(backing, dict):
        backing.pop(name, None)
