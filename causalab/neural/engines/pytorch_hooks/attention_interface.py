"""Taps *inside* the attention function, where no forward hook can reach.

Four of the mixer's tensors are not module boundaries. ``transformers`` computes
them inside one call::

    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask, ...)

so ``query`` and ``key`` are that call's *arguments* (post-RoPE, and for ``key``
before ``repeat_kv``), the scores are the softmax's input several lines further
in, and ``z`` is the call's return. A ``register_forward_hook`` on the mixer
fires after all of it, which is why writing the attention pattern was already a
special case: by then the tensor has been consumed.

📐 That is measured, not assumed. ``self_attn`` *returns*
``(attn_output, attn_weights)``, so a ``register_forward_hook`` that rewrites
element 1 looks like it should work — and changes a tensor nothing downstream
reads. The same silent-no-op shape as writing ``router_logits``, and measured
the same way: 0.0 change in the logits. **Reading** the pattern is still an
ordinary module tap, because the mixer hands it back; only the write has to come
through here.

How the call is intercepted
---------------------------

``transformers`` resolves the function per forward::

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward)

📐 ``"eager"`` is **not** registered by default, so that call falls through to
the module's own ``eager_attention_forward``. Registering ``"eager"`` therefore
*inserts* a wrapper rather than replacing one, and removing the key restores the
original behaviour exactly. The backend forces eager attention at load time
(``loading.py``), which is what makes this the only implementation to wrap.

The registry entry is process-global while installed, so this context manager
must wrap the single forward it applies to and nothing wider.

The scores, and why nothing is transcribed
------------------------------------------

The scores have no name in the eager function's signature — they are a local,
between the ``matmul`` and the ``softmax``. #53 reached the pattern by calling
the real function and then **redoing** the two lines after the softmax, which
works but duplicates library internals and has to resolve a per-family
``eager_attention_forward`` to do it.

The scores need no such thing. A :class:`torch.overrides.TorchFunctionMode`
entered around — and only around — the real call intercepts
``torch.nn.functional.softmax`` where it happens:

* **reading** its input is ``attention_scores``;
* **writing** its input is a write the model's own softmax then consumes, so the
  rows it produces still sum to 1 *by construction*. Nothing is reimplemented,
  so nothing can drift.

📐 Measured on all three CI fixtures (transformers 5.16): exactly **one**
``F.softmax`` call inside the tapped eager, an observe-only pass is
bit-identical (logits maxdiff 0.0), ``softmax(captured scores)`` equals the
returned ``attn_weights`` to 0.0, and knocking one head off one token moved the
qwen fixture's logits by 0.3114.

Two guards the design has to carry, both because the mode is a blunt instrument:

* it matches ``torch.nn.functional.softmax`` **only** — not ``torch.softmax``,
  not ``Tensor.softmax`` — and **counts** the calls. A family whose eager calls
  softmax twice (soft-capping, a sliding-window pass) is refused by name rather
  than tapped at whichever one came first;
* it is entered around the ``real(...)`` call and nothing wider, so it cannot
  see another module's arithmetic even though it is process-global while active.

Writing the pattern needs no recompute either
---------------------------------------------

#53 carried a pattern edit forward by calling the real eager function and then
**redoing** the two lines that follow its softmax::

    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_output = torch.matmul(edited, value_states).transpose(1, 2).contiguous()

That worked, and it was pinned by an identity-edit test — but it duplicated
library internals, and it had to resolve a second per-family symbol
(``repeat_kv``) to do it, which GPT-2 does not even export.

Intercepting the softmax's **output** removes all of it. The mode returns the
edited pattern *into* the eager function, which then does its own value multiply
with its own code. The identity-edit test that used to guard the transcription
is now trivially satisfied, which is exactly the point: there is nothing left to
drift.

Why every mechanism is legal on the scores
------------------------------------------

``attention_probs`` accepts only ``swap``: a delta or a scale leaves rows that no
longer sum to 1, and nothing downstream renormalizes them. One step earlier that
objection disappears. Attention knockout is ``delta: -1e4``; head boosting is a
scale; and the softmax cleans up after both. This is the write surface the
pattern could not be.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
from typing import Any, Callable, Iterator, Mapping

import torch
from torch.overrides import TorchFunctionMode

from causalab.protocol.errors import ProtocolError

__all__ = [
    "INTERFACE_SLOTS",
    "InterfaceTap",
    "attention_interface_taps",
    "module_eager_attention",
]

#: The points inside the attention function a component may name, in the order
#: the function reaches them. ``"probs"`` is the pattern's *write* slot — reading
#: it is an ordinary module tap, because the mixer returns it.
INTERFACE_SLOTS: tuple[str, ...] = ("query", "key", "scores", "probs", "z")

#: The slots served by intercepting the softmax rather than by touching an
#: argument or a return value.
_SOFTMAX_SLOTS: frozenset[str] = frozenset({"scores", "probs"})


@dataclasses.dataclass(frozen=True)
class InterfaceTap:
    """One read and/or edit at a named point inside the attention function.

    ``read`` is handed the tensor as the function sees it. ``edit`` is handed a
    **clone** and returns the tensor to use in its place, so an edit that
    mutates in place and an edit that returns a new tensor are both correct and
    neither can reach the model's own storage by accident.
    """

    slot: str
    read: Callable[[torch.Tensor], None] | None = None
    edit: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.slot not in INTERFACE_SLOTS:
            raise ValueError(
                f"unknown attention-interface slot {self.slot!r}; "
                f"expected one of {INTERFACE_SLOTS}"
            )


class _SoftmaxTap(TorchFunctionMode):
    """Intercept the one ``F.softmax`` inside a tapped attention function.

    Strict on purpose: ``torch.nn.functional.softmax`` and nothing else, and the
    call is counted so that a family which softmaxes twice is refused rather than
    silently tapped at the first one.
    """

    def __init__(
        self,
        on_input: Callable[[torch.Tensor], torch.Tensor] | None = None,
        on_output: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.on_input = on_input
        self.on_output = on_output
        self.calls = 0

    def __torch_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        kwargs = dict(kwargs or {})
        if func is not torch.nn.functional.softmax:
            return func(*args, **kwargs)
        self.calls += 1
        if self.on_input is not None and args:
            args = (self.on_input(args[0]), *args[1:])
        out = func(*args, **kwargs)
        return out if self.on_output is None else self.on_output(out)


def _apply(taps: "tuple[InterfaceTap, ...]", slot: str, value: torch.Tensor):
    """Run every tap declared for ``slot``, in order.

    Two orderings, and only one of them is this function's to choose:

    * **within one tap**, the read runs before the edit — a tap that does both
      observes the value the model computed, not its own replacement;
    * **across taps**, registration order decides, and the executor registers
      edits before reads. So a document that reads and writes the same slot in
      one forward sees the *written* value.

    That second one is not an accident of this file: it is what the module-hook
    path already does, because ``_installed`` is entered before ``_capturing``
    and hooks fire in registration order. 📐 Measured equal on both paths —
    ``attention_premix`` (a module boundary) and ``attention_query`` /
    ``attention_z`` (interface slots) all read back exactly the written value,
    difference 0.0. The two tap mechanisms have to agree here or the same
    document would mean different things depending on which components it named.
    """
    for tap in taps:
        if tap.slot != slot:
            continue
        if tap.read is not None:
            tap.read(value)
        if tap.edit is not None:
            value = tap.edit(value.clone())
    return value


def _has(taps: "tuple[InterfaceTap, ...]", slot: str) -> bool:
    return any(tap.slot == slot for tap in taps)


@contextlib.contextmanager
def attention_interface_taps(
    taps: Mapping[int, "tuple[InterfaceTap, ...]"],
) -> Iterator[None]:
    """Install reads and edits inside the eager attention function.

    Args:
        taps: ``id(mixer module) -> taps``. A mixer absent from the mapping is
            untouched and pays only a dict lookup, which is what keeps a tap at
            one layer from changing any other layer's arithmetic.
    The ``"eager"`` registry key is removed on exit (or restored, if something
    else had registered one), which puts ``get_interface`` back on the module
    default.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if not taps:
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
        # Resolved from the MODULE's own modeling file, per call: while the
        # registry entry is installed it intercepts every attention forward, so
        # borrowing one family's function would silently replace another's math
        # (gemma-2's eager soft-caps the logits, say).
        real = module_eager_attention(module)
        entries = taps.get(id(module), ())
        if not entries:
            return real(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **kwargs,
            )

        query = _apply(entries, "query", query)
        key = _apply(entries, "key", key)

        wants_softmax = any(_has(entries, slot) for slot in _SOFTMAX_SLOTS)
        if not wants_softmax:
            out, weights = real(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **kwargs,
            )
            return _apply(entries, "z", out), weights

        def on_scores(scores: torch.Tensor) -> torch.Tensor:
            return _apply(entries, "scores", scores)

        def on_probs(probs: torch.Tensor) -> torch.Tensor:
            # Returning the edited pattern is the whole write: the model's own
            # eager function receives it and does its own value multiply, so
            # nothing here has to know what that multiply is.
            return _apply(entries, "probs", probs)

        mode = _SoftmaxTap(
            on_scores if _has(entries, "scores") else None,
            on_probs if _has(entries, "probs") else None,
        )
        with mode:
            out, weights = real(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **kwargs,
            )
        _check_one_softmax(module, mode.calls)
        return _apply(entries, "z", out), weights

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


def module_eager_attention(module: Any) -> Callable[..., Any]:
    """The mixer's own ``eager_attention_forward``.

    Resolved from the modeling file the module's class was defined in, per call.
    While the registry entry is installed it intercepts *every* attention
    forward, so borrowing one family's function would silently replace another
    family's math — gemma-2's eager soft-caps its logits, for instance. A family
    whose modeling file exports no such function is refused by name rather than
    served somebody else's.

    ⚠️ Deliberately asks for **only** this symbol. Round 2.3 also needed
    ``repeat_kv`` here, to redo the value multiply after a pattern edit; 📐 GPT-2
    exports the first and not the second (no GQA, nothing to repeat), so asking
    for both made a plain read of the attention interior on gpt2 fail with a
    message about pattern writes. Round 2.5 removed the second requirement
    entirely along with the recompute that needed it.
    """
    modeling = importlib.import_module(type(module).__module__)
    found = getattr(modeling, "eager_attention_forward", None)
    if found is None:
        raise ProtocolError(
            "P4",
            f"an attention-interface tap on {type(module).__name__}: its "
            f"modeling module {type(module).__module__!r} exports no "
            "'eager_attention_forward'. Extend attention_interface.py for this "
            "family — borrowing another family's version would silently change "
            "what the model computes.",
        )
    return found


def _check_one_softmax(module: Any, calls: int) -> None:
    """Refuse a family whose eager attention does not softmax exactly once.

    📐 All three CI fixtures call it exactly once. A family that calls it twice —
    soft-capping, a second sliding-window pass — would have *a* softmax tapped,
    and which one would depend on source order. That is the silent-wrong-tensor
    failure the whole descriptor effort exists to prevent, so it is refused by
    name; zero calls means the function did not softmax at all, which means the
    tap read nothing.
    """
    if calls == 1:
        return
    raise ProtocolError(
        "P4",
        f"the eager attention of {type(module).__name__} called "
        f"torch.nn.functional.softmax {calls} times, not once. This backend taps "
        "the softmax to reach the attention scores, and with "
        f"{'no call' if calls == 0 else 'more than one'} it cannot say which "
        "tensor it read. Extend attention_interface.py for this family rather "
        "than tapping whichever call came first.",
    )


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
