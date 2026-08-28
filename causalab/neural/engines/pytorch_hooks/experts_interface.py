"""Taps *inside* the routed-experts dispatch, where no forward hook can reach.

The per-expert interior of a sparse-MoE block is not a set of module
boundaries. 📐 ``Qwen3_5MoeExperts`` stores its weights as 3-D parameters
(``gate_up_proj (num_experts, 2·d_e, hidden)``) and its only child is one
shared ``act_fn`` — there is no per-expert module for a hook to attach to, on
any path. ``transformers`` computes the whole interior inside one dispatched
call::

    Qwen3_5MoeExperts.forward
      -> ALL_EXPERTS_FUNCTIONS[config._experts_implementation]
      -> grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)

and the grouped function materializes every diagram box densely, in one call:
it sorts the ``S = tokens · top_k`` (token, slot) pairs by expert, runs
``_grouped_linear`` exactly twice (the fused ``[gate | up]`` projection, then
the down-projection), applies ``act_fn`` once between them, and un-sorts. So
the interior tensors already exist — this module's job is *addressing* them,
not extracting them.

How the call is intercepted
---------------------------

Mirrors :mod:`.attention_interface` one level up. ``ALL_EXPERTS_FUNCTIONS`` is
a ``GeneralInterface``: registering ``"grouped_mm"`` installs a **local**
override that shadows the library's global entry, and restoring the previous
value on exit puts dispatch back exactly where it was — this key exists in the
global mapping (unlike ``"eager"`` in the attention registry), so containment
is restore-not-delete. The entry is process-global while installed, so the
wrapper answers for *every* experts module and calls straight through for any
module not in its tap table.

Why the grouped path, and only the grouped path
-----------------------------------------------

📐 ``config._experts_implementation`` defaults to ``"grouped_mm"`` everywhere —
including CPU (``modeling_utils`` sets the default; the dispatch gate is a
class check, not a device check) — and a pass-through wrapper on it is
**bit-identical** (measured 0.0). Forcing ``experts_implementation="eager"``
(the per-expert loop) moves the fixture logits by 4.2e-7: numerically real,
so the identity-pin bar ("the no-op case is exactly equal") is met by tapping
the path that actually runs and *missed* by path-forcing. Other
implementations compute the same numbers in a different order and their
interiors live elsewhere, so the site resolver refuses them by name — the
dispatch pin — rather than reading a tensor whose provenance is wrong.

The sort, and why it is recomputed inside the dynamic extent
------------------------------------------------------------

The grouped function orders its rows with ``torch.sort(top_k_index.reshape(-1))``
— an **unstable** sort, so tie order within one expert's group is not promised
across implementations. The wrapper recomputes the same sort on the same
input, *inside* the one call it taps, and the suite pins the reconstruction
identity (un-sort + weight + sum reproduces the block's own output at exactly
0.0) so a kernel that ever breaks ties differently fails loudly instead of
attributing rows to the wrong tokens.

Two guards, same philosophy as the softmax count of round 2.3:

* ``_grouped_linear`` is counted and must fire **exactly twice** inside the
  tapped call — call 1's output is the fused ``[gate | up]`` projection, call
  2's is the down-projection (pre-routing-weight), and a family that calls it
  any other number of times is refused rather than mislabeled;
* the module must declare ``has_gate`` — a ``has_gate=False`` family also
  calls it twice, with *different meanings* (up, then down), and must say so
  rather than have its up-projection labeled ``[gate | up]``.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Callable, Iterator, Mapping

import torch

from causalab.protocol.errors import ProtocolError

__all__ = [
    "EXPERTS_SLOTS",
    "ExpertsTap",
    "experts_implementation_of",
    "experts_interface_taps",
]

#: The points inside the grouped experts function a component may name, in the
#: order the function reaches them. ``"gate_up"`` is the fused ``[gate | up]``
#: projection (its two halves are separate components, via the descriptor's
#: fused axis); ``"activation"`` is the shared ``act_fn``'s output (the
#: activated gate half, before the ``· up`` multiply — the same tensor
#: ``mlp_activation`` names on the llama family); ``"down"`` is the
#: down-projection's output **before** the routing weight is applied.
EXPERTS_SLOTS: tuple[str, ...] = ("gate_up", "activation", "down")


@dataclasses.dataclass(frozen=True)
class ExpertsTap:
    """One read and/or edit at a named point inside the experts function.

    Values cross this interface **token-major**: the wrapper un-sorts the
    expert-sorted rows with the inverse permutation it computed, hands
    ``(tokens, top_k · width)`` to the tap along with the routing table
    ``top_k_index (tokens, top_k)``, and re-sorts whatever an edit returns. So
    a tap never sees the expert-sorted order, and the permutation never leaves
    this module.

    ``read`` observes the value as the model computed it (before any edit from
    the same tap); ``edit`` is handed a **clone** and returns the tensor to use
    in its place.
    """

    slot: str
    read: Callable[[torch.Tensor, torch.Tensor], None] | None = None
    edit: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.slot not in EXPERTS_SLOTS:
            raise ValueError(
                f"unknown experts-interface slot {self.slot!r}; "
                f"expected one of {EXPERTS_SLOTS}"
            )


def experts_implementation_of(model: Any) -> str:
    """The experts implementation a loaded model dispatches on.

    Read from the config the modeling code itself reads
    (``config._experts_implementation``, set at load time; the text config on a
    multimodal wrapper). This is the fact every experts-interface tap is pinned
    against: the interior tensors this module addresses are the *grouped*
    function's locals, and a different implementation computes different
    intermediates in a different order even where the block's output agrees.
    """
    config = getattr(model.config, "text_config", None) or model.config
    return str(getattr(config, "_experts_implementation", "<undeclared>"))


def _apply(
    taps: tuple[ExpertsTap, ...], slot: str, value: torch.Tensor, idx: torch.Tensor
) -> torch.Tensor:
    """Run every tap declared for ``slot``, in order — the ordering contract of
    :func:`.attention_interface._apply`, verbatim: within one tap the read runs
    before the edit, across taps registration order decides, and the executor
    registers edits before reads so a same-forward read sees the written value.
    """
    for tap in taps:
        if tap.slot != slot:
            continue
        if tap.read is not None:
            tap.read(value, idx)
        if tap.edit is not None:
            value = tap.edit(value.clone(), idx)
    return value


def _has(taps: tuple[ExpertsTap, ...], slot: str) -> bool:
    return any(tap.slot == slot for tap in taps)


@contextlib.contextmanager
def experts_interface_taps(
    taps: Mapping[int, tuple[ExpertsTap, ...]],
) -> Iterator[None]:
    """Install reads and edits inside the grouped experts function.

    Args:
        taps: ``id(experts module) -> taps``. An experts module absent from the
            mapping is untouched and pays only a dict lookup — the scoping that
            keeps a tap at one layer from changing any other layer's
            arithmetic.

    The ``"grouped_mm"`` registry entry is **restored** on exit (the key exists
    in the library's global mapping, so containment is restore-not-delete —
    setting back the callable that dispatch resolved before entry preserves
    even a pre-existing local override).
    """
    import transformers.integrations.moe as moe

    if not taps:
        yield
        return

    real_impl = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]

    def wrapper(
        module: Any,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        entries = taps.get(id(module), ())
        if not entries:
            return real_impl(module, hidden_states, top_k_index, top_k_weights)
        return _tapped_forward(
            module, hidden_states, top_k_index, top_k_weights, entries, real_impl
        )

    moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] = wrapper
    try:
        yield
    finally:
        moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] = real_impl


def _tapped_forward(
    module: Any,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    entries: tuple[ExpertsTap, ...],
    real_impl: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """One tapped experts call: recompute the sort, patch the two grouped
    linears and hook the shared ``act_fn`` for the duration of the one real
    call, and convert every crossing tensor between the function's
    expert-sorted rows and the taps' token-major form."""
    import transformers.integrations.moe as moe

    if getattr(module, "has_gate", None) is not True:
        # a has_gate=False family also calls _grouped_linear twice, but call 1
        # is then the plain up-projection — labeling it [gate | up] would be
        # the silent-wrong-tensor failure the descriptors exist to prevent
        raise ProtocolError(
            "P4",
            f"an experts-interface tap on {type(module).__name__}: this experts "
            "module does not declare a gated projection (has_gate is "
            f"{getattr(module, 'has_gate', None)!r}), so the first grouped "
            "linear is not [gate | up] and the slot labels here would lie. "
            "Extend experts_interface.py for this family.",
        )

    tokens = hidden_states.shape[0]
    expert_ids = top_k_index.reshape(-1)
    # ⚠️ the same unstable sort the grouped function performs, recomputed on the
    # same input inside its dynamic extent; the reconstruction-identity test is
    # what pins the tie order (module docstring)
    _, perm = torch.sort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)

    def token_major(rows: torch.Tensor) -> torch.Tensor:
        """(S, width) expert-sorted → (tokens, top_k · width)."""
        return rows[inv_perm].reshape(tokens, -1)

    def expert_sorted(value: torch.Tensor, width: int) -> torch.Tensor:
        """The inverse: (tokens, top_k · width) → (S, width) expert-sorted."""
        return value.reshape(-1, width)[perm]

    def run(slot: str, rows: torch.Tensor) -> torch.Tensor:
        if not _has(entries, slot):
            return rows
        value = _apply(entries, slot, token_major(rows), top_k_index)
        return expert_sorted(value, rows.shape[-1])

    gl_calls = 0
    real_gl = moe._grouped_linear

    def grouped_linear(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal gl_calls
        out = real_gl(*args, **kwargs)
        gl_calls += 1
        if gl_calls == 1:
            return run("gate_up", out)
        if gl_calls == 2:
            return run("down", out)
        return out  # counted; refused below rather than mislabeled here

    act_calls = 0

    def act_hook(_m: Any, _i: Any, out: torch.Tensor) -> torch.Tensor:
        nonlocal act_calls
        act_calls += 1
        return run("activation", out)

    handle = module.act_fn.register_forward_hook(act_hook)
    moe._grouped_linear = grouped_linear
    try:
        result = real_impl(module, hidden_states, top_k_index, top_k_weights)
    finally:
        moe._grouped_linear = real_gl
        handle.remove()
    _check_call_counts(module, gl_calls, act_calls)
    return result


def _check_call_counts(module: Any, gl_calls: int, act_calls: int) -> None:
    """Refuse a family whose grouped forward does not have the measured shape.

    📐 On the grouped path ``_grouped_linear`` fires exactly twice per block
    (the fused up-projection, then the down-projection) and the shared
    ``act_fn`` exactly once. Any other count means a different factorization —
    the eager loop fires ``act_fn`` once per *hit expert* — and the slot labels
    above would be attached to the wrong tensors.
    """
    if gl_calls == 2 and act_calls == 1:
        return
    raise ProtocolError(
        "P4",
        f"the grouped experts forward of {type(module).__name__} called "
        f"_grouped_linear {gl_calls} times and act_fn {act_calls} times, not "
        "(2, 1). This backend labels call 1 '[gate | up]', call 2 'down' and "
        "the activation 'act_fn(gate)', and with any other shape it cannot say "
        "which tensor it read — extend experts_interface.py for this family "
        "rather than tapping whichever call came first.",
    )
