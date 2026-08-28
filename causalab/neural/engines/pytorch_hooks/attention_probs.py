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

How the call is intercepted — and what is left in this module
------------------------------------------------------------

:mod:`.attention_interface` owns the interception (registering a wrapper under
``ALL_ATTENTION_FUNCTIONS["eager"]``, scoped to one forward). What stays here is
the part that is specific to the *pattern*: an edit to it has to be carried
forward by redoing the two lines the eager function runs after its softmax,
because by the time the function returns, ``attn_output`` has already been
computed from the pattern.

**What that duplicates, and why it is safe.** It redoes exactly::

    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_output = torch.matmul(edited, value_states).transpose(1, 2).contiguous()

Duplicating library internals is normally how a backend rots silently. Here it
is pinned: with an **identity** edit the recomputed output must equal the
unpatched output *exactly*, which
:func:`tests...test_attention_probs.test_an_identity_edit_is_bit_identical`
asserts (📐 measured max difference 0.0). If a future transformers changes what
happens after the softmax — a different scaling, a dropout that is not a no-op
in eval, an output reshape — that test fails rather than the numbers quietly
drifting.

⚠️ **This is the transcription round 2.5 exists to delete.** Round 2.3 taps the
softmax to reach ``attention_scores``; intercepting the same softmax's *output*
reaches the pattern with nothing duplicated and no per-family resolution at all.

Scope: a write replaces the **whole** pattern, which is what an interchange on
attention does (and what nnterp's own check does: ``self[layer] = rnd``).
Addressing a single query position, or a feature slice of the key axis, is
refused rather than approximated — and the refusal is *generated*: the tap
declares ``(batch, head, position[query], key_position[key])``, which has two
position axes and therefore no ``(batch, position, feature)`` contract form, and
the executor refuses every operation that needs one. See
:mod:`causalab.protocol.shapes`.

⚠️ Every mechanism but ``swap`` is refused here, because a delta or a scale
leaves rows that no longer sum to 1 and nothing downstream renormalizes them.
One step earlier that objection disappears entirely — write
``attention_scores``, upstream of the model's own softmax.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

import torch

from causalab.protocol.errors import ProtocolError

__all__ = ["module_eager_attention", "post_softmax_value_multiply"]


def _from_modeling(module: Any, name: str, why: str) -> Callable[..., Any]:
    """One symbol out of the modeling file the module's class was defined in.

    Resolved per module, so a wrapper defers to the implementation that belongs
    to the model being run — never another family's. transformers stamps these
    names into every modeling file that uses the attention-interface pattern,
    and a family where one is missing is refused by name rather than served
    another family's version, which would silently change what the model
    computes (gemma-2's eager soft-caps its logits, say).
    """
    modeling = importlib.import_module(type(module).__module__)
    found = getattr(modeling, name, None)
    if found is None:
        raise ProtocolError(
            "P4",
            f"{why} on {type(module).__name__}: its modeling module "
            f"{type(module).__module__!r} exports no {name!r}. Extend "
            "attention_probs.py for this family — borrowing another family's "
            "version would silently change what the model computes.",
        )
    return found


def module_eager_attention(module: Any) -> Callable[..., Any]:
    """The mixer's own ``eager_attention_forward``.

    ⚠️ Deliberately does **not** also require ``repeat_kv``. The two are needed
    by different things: every interface tap needs the eager function, and only
    a *pattern write* needs ``repeat_kv`` to redo the value multiply. 📐 GPT-2's
    modeling file exports the first and not the second (it has no GQA, so it has
    nothing to repeat), and asking for both made a plain read of the attention
    interior on gpt2 fail with a message about pattern writes.
    """
    return _from_modeling(
        module, "eager_attention_forward", "an attention-interface tap"
    )


def post_softmax_value_multiply(
    payload: tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carry an edited attention pattern forward, by redoing the two lines the
    eager function runs after its softmax::

        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_output = torch.matmul(probs, value_states).transpose(1, 2).contiguous()

    Duplicating library internals is normally how a backend rots silently. Here
    it is pinned: with an **identity** edit the recomputed output must equal the
    unpatched output exactly, which
    ``tests...test_attention_probs.test_an_identity_edit_is_bit_identical``
    asserts (📐 measured max difference 0.0). If a future transformers changes
    what happens after the softmax, that test fails rather than the numbers
    quietly drifting.

    ⚠️ This is the transcription round 2.5 exists to delete: intercepting the
    softmax's *output* (which :mod:`.attention_interface` already does to reach
    the scores) reaches the same place with nothing duplicated and no per-family
    resolution.
    """
    module, probs, value, _out, _weights = payload
    repeat_kv = _from_modeling(module, "repeat_kv", "an attention-pattern write")
    value_states = repeat_kv(value, module.num_key_value_groups)
    recomputed = torch.matmul(probs.to(value_states.dtype), value_states)
    return recomputed.transpose(1, 2).contiguous(), probs
