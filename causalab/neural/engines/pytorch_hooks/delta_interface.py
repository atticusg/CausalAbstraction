"""Taps at the DeltaNet mixer's kernel boundary, where no forward hook reaches.

Seven of the Gated DeltaNet diagram's boxes are neither module boundaries nor
locals of one interceptable library call — they are the arguments and returns
of two **module-global** call sites inside the mixer's forward
(``modeling_qwen3_5_moe.py``)::

    mixed_qkv = causal_conv1d_fn(mixed_qkv, conv1d.weight.squeeze(1), ...)
    q, k, v = split(...); ...
    core_attn_out, S = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, ...)
                     | torch_recurrent_gated_delta_rule(...)   # cached decode

📐 The ``conv1d`` *module* is never called (its hook never fires — pinned in
round 4.1's suite), so the conv output exists only as this function's return;
the kernel's arguments are the post-conv, post-tiling, **pre**-l2norm q/k/v
plus the per-head gates ``beta``/``g``; and its return[0] is the pre-norm,
pre-gate ``core_attn_out``. During decode the model natively runs the
*recurrent* kernel and ``causal_conv1d_update`` (📐 measured: chunked once per
linear layer at prefill, recurrent + conv-update at every cached step), so all
four globals are swapped together and decode steps are tapped identically to
prefill.

Containment — the R2.3 replace-restore shape, one level down
------------------------------------------------------------

The globals are resolved **from the tapped mixer's own modeling module**
(``importlib.import_module(type(mixer).__module__)`` — the
``module_eager_attention`` pattern), so another family's file is never
patched. The wrappers are ``**kwargs``-transparent and call through to the
*original globals captured at entry* — which keeps whatever hub/``fla``
dispatch the environment resolved, because the taps only touch arguments and
returns; only interiors care which body runs (round-4 plan §0). Identity is
bit-exact by construction: 📐 a pass-through wrapper on either surface measured
0.0.

The wrapper answers for every mixer of the family while installed, so *which*
call to tap is tracked by dynamic extent: pre/post hooks on each tapped mixer
mark it active, and an untapped mixer's calls fall straight through.

Two refusals the design has to carry:

* a **kernelized** model (``kernelize()``) replaces the mixer class's
  ``forward`` wholesale, and no module-global patch applies inside a hub
  kernel — detected as a ``forward`` defined outside the modeling file, and
  refused by name;
* a modeling file that does not export all four globals cannot be tapped —
  refused rather than served another family's kernels.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
from typing import Any, Callable, Iterator, Mapping

import torch

from causalab.protocol.errors import ProtocolError

__all__ = [
    "DELTA_SLOTS",
    "DeltaTap",
    "delta_kernel_taps",
]

#: The points at the kernel boundary a component may name, in the order the
#: forward reaches them. ``"conv"`` is the causal-conv function's return
#: (channels-first); the five argument slots and ``"kernel_output"`` belong to
#: whichever delta-rule kernel the forward dispatches (chunked at prefill,
#: recurrent at cached decode steps — tapped identically).
DELTA_SLOTS: tuple[str, ...] = (
    "conv",
    "query",
    "key",
    "value",
    "beta",
    "decay",
    "kv_mem",
    "state_update",
    "state",
    "kernel_output",
)

#: The per-step interior (round 4.3). At prefill these exist only inside the
#: recurrent formulation, which the chunked kernel never materializes — so a
#: read **steps the library's own recurrent kernel** in the chunked call's
#: shadow (nothing transcribed: every number is the library's), and a state
#: write substitutes that stepwise loop for the chunked call (path-forcing,
#: measured 5.4e-7 on the logits, pinned per layer). At decode the model runs
#: the recurrent kernel natively and all three are plain per-step captures.
_STATE_SLOTS: frozenset[str] = frozenset({"kv_mem", "state_update", "state"})

#: The four module globals swapped together, per modeling module.
_GLOBALS: tuple[str, ...] = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)


@dataclasses.dataclass(frozen=True)
class DeltaTap:
    """One read and/or edit at a named point at the kernel boundary.

    Same contract as :class:`.attention_interface.InterfaceTap`: ``read`` is
    handed the tensor as the function sees it, ``edit`` is handed a **clone**
    and returns the replacement.

    ``edit_state`` is the state write's own surface — ``(step, S_t) -> S_t`` —
    because a state edit must **feed forward**: step ``t``'s replacement is
    what step ``t+1`` decays and writes into, so the whole-tensor ``edit``
    contract cannot express it. Only the ``"state"`` slot may carry one, and
    carrying one is what switches the chunked call to the stepwise
    substitution.
    """

    slot: str
    read: Callable[[torch.Tensor], None] | None = None
    edit: Callable[[torch.Tensor], torch.Tensor] | None = None
    edit_state: Callable[[int, torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.slot not in DELTA_SLOTS:
            raise ValueError(
                f"unknown delta-kernel slot {self.slot!r}; expected one of "
                f"{DELTA_SLOTS}"
            )
        if self.edit_state is not None and self.slot != "state":
            raise ValueError(
                f"edit_state is the state write's surface; slot {self.slot!r} "
                "cannot carry one"
            )


def _apply(taps: tuple[DeltaTap, ...], slot: str, value: torch.Tensor) -> torch.Tensor:
    """The ordering contract of :func:`.attention_interface._apply`, verbatim."""
    for tap in taps:
        if tap.slot != slot:
            continue
        if tap.read is not None:
            tap.read(value)
        if tap.edit is not None:
            value = tap.edit(value.clone())
    return value


def _has(taps: tuple[DeltaTap, ...], slot: str) -> bool:
    return any(tap.slot == slot for tap in taps)


def _modeling_module(mixer: Any) -> Any:
    """The modeling file a tapped mixer's kernels live in — with the two
    refusals the module docstring names."""
    cls = type(mixer)
    forward_home = getattr(cls.forward, "__module__", None)
    if forward_home != cls.__module__:
        raise ProtocolError(
            "P4",
            f"a delta-kernel tap on {cls.__name__}: its forward comes from "
            f"{forward_home!r}, not its own modeling module {cls.__module__!r} "
            "— a kernelize()d (hub-kernel) mixer computes inside a fused kernel "
            "no module-global patch can reach. Load the model without "
            "kernelize(), or extend delta_interface.py for this kernel.",
        )
    modeling = importlib.import_module(cls.__module__)
    missing = [name for name in _GLOBALS if not hasattr(modeling, name)]
    if missing:
        raise ProtocolError(
            "P4",
            f"a delta-kernel tap on {cls.__name__}: its modeling module "
            f"{cls.__module__!r} exports no {', '.join(missing)}. Extend "
            "delta_interface.py for this family — borrowing another family's "
            "kernels would silently change what the model computes.",
        )
    return modeling


@contextlib.contextmanager
def delta_kernel_taps(taps: Mapping[Any, tuple[DeltaTap, ...]]) -> Iterator[None]:
    """Install reads and edits at the DeltaNet kernel boundary.

    Args:
        taps: ``mixer module -> taps`` (keyed by the module object itself —
            unlike the attention registry, the wrappers here never receive the
            module as an argument, so the mixers are also where the dynamic
            extent is tracked). A mixer absent from the mapping is untouched.

    All four globals are restored on exit, in every patched modeling module.
    """
    if not taps:
        yield
        return

    #: the mixer whose forward is currently executing, if it is tapped
    active: dict[str, Any] = {"mixer": None, "seq_len": None}

    per_module: dict[int, tuple[Any, dict[str, Any]]] = {}
    for mixer in taps:
        modeling = _modeling_module(mixer)
        if id(modeling) not in per_module:
            originals = {name: getattr(modeling, name) for name in _GLOBALS}
            per_module[id(modeling)] = (modeling, originals)

    def conv_wrapper(real: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            out = real(hidden_states, *args, **kwargs)
            current = _by_identity(taps, active["mixer"])
            if not current or not _has(current, "conv"):
                return out
            # ⚠️ Under a cache, `update_conv_state` may hand the conv a tensor
            # longer than the mixer's own sequence (prepended state), and the
            # forward keeps only the last seq_len columns. The tap addresses
            # exactly what the forward keeps, so the same slice is applied
            # here — a no-op when the lengths already agree.
            seq_len = active["seq_len"]
            if seq_len is not None and out.shape[-1] != seq_len:
                kept = _apply(current, "conv", out[..., -seq_len:])
                out = out.clone()
                out[..., -seq_len:] = kept
                return out
            return _apply(current, "conv", out)

        return wrapped

    def kernel_wrapper(
        real: Callable[..., Any], originals: dict[str, Any], modeling: Any
    ) -> Callable[..., Any]:
        def wrapped(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            g: torch.Tensor | None = None,
            beta: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> Any:
            current = _by_identity(taps, active["mixer"])
            if not current:
                return real(query, key, value, g=g, beta=beta, **kwargs)
            query = _apply(current, "query", query)
            key = _apply(current, "key", key)
            value = _apply(current, "value", value)
            if beta is not None:
                beta = _apply(current, "beta", beta)
            if g is not None:
                g = _apply(current, "decay", g)
            if any(tap.slot in _STATE_SLOTS for tap in current):
                out, state = _with_state_taps(
                    current,
                    real,
                    originals,
                    modeling,
                    query,
                    key,
                    value,
                    g,
                    beta,
                    kwargs,
                )
            else:
                out, state = real(query, key, value, g=g, beta=beta, **kwargs)
            return _apply(current, "kernel_output", out), state

        return wrapped

    with contextlib.ExitStack() as stack:
        for modeling, originals in per_module.values():
            setattr(
                modeling,
                "causal_conv1d_fn",
                conv_wrapper(originals["causal_conv1d_fn"]),
            )
            setattr(
                modeling,
                "causal_conv1d_update",
                conv_wrapper(originals["causal_conv1d_update"]),
            )
            setattr(
                modeling,
                "torch_chunk_gated_delta_rule",
                kernel_wrapper(
                    originals["torch_chunk_gated_delta_rule"], originals, modeling
                ),
            )
            setattr(
                modeling,
                "torch_recurrent_gated_delta_rule",
                kernel_wrapper(
                    originals["torch_recurrent_gated_delta_rule"], originals, modeling
                ),
            )
            stack.callback(_restore, modeling, originals)
        for mixer in taps:
            pre = mixer.register_forward_pre_hook(_enter(active, mixer))
            post = mixer.register_forward_hook(_leave(active))
            stack.callback(pre.remove)
            stack.callback(post.remove)
        yield


def _l2norm_of(modeling: Any) -> Callable[..., torch.Tensor]:
    """The modeling file's own ``l2norm`` — needed to form k̂ for the derived
    per-step faces, resolved per family like everything else here."""
    found = getattr(modeling, "l2norm", None)
    if found is None:
        raise ProtocolError(
            "P4",
            f"a per-step state tap needs the modeling module "
            f"{modeling.__name__!r} to export 'l2norm' (the normalization its "
            "own kernel applies to k), and it does not — extend "
            "delta_interface.py for this family.",
        )
    return found


def _state_faces(
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
    s_prev: torch.Tensor,
    l2norm: Callable[..., torch.Tensor],
    use_l2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two derived per-step faces, from adjacent states (round-4 plan
    §2.3): ``kv_mem_t = (S_{t-1}·exp(g_t) · k̂_t).sum(-2)`` and
    ``delta_t = (v_t − kv_mem_t)·β_t`` — the recurrent kernel's own lines
    (``modeling:369-374``), computed in float32 exactly as it computes them,
    and pinned by the reconstruction identity
    ``S_t == S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t`` against the kernel's returned
    states.

    Args are one step's slices: ``k_t/v_t (b, h, d)``, ``g_t/beta_t (b, h)``,
    ``s_prev (b, h, d_k, d_v)`` in float32.
    """
    k_hat = l2norm(k_t, dim=-1, eps=1e-6) if use_l2 else k_t
    decayed = s_prev * g_t.to(torch.float32).exp()[..., None, None]
    kv_mem = (decayed * k_hat.to(torch.float32).unsqueeze(-1)).sum(dim=-2)
    delta = (v_t.to(torch.float32) - kv_mem) * beta_t.to(torch.float32).unsqueeze(-1)
    return kv_mem, delta


def _stepwise(
    real_recurrent: Callable[..., Any],
    l2norm: Callable[..., torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    use_l2: bool,
    edit_state: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drive the library's own recurrent kernel one timestep at a time.

    No transcription: every state is the kernel's own return, threaded back in
    as the next step's ``initial_state``. ``edit_state`` (a state write) is
    applied to each step's state *before* it threads forward, which is the
    whole reason a write substitutes this loop for the chunked call.

    Returns ``(out, final_state, states, kv_mems, deltas)`` with the per-step
    tensors stacked on a steps axis: ``states (b, s, h, d_k, d_v)``,
    ``kv_mems/deltas (b, s, h, d_v)``.
    """
    batch, seq_len, heads, d_k = key.shape
    d_v = value.shape[-1]
    state = initial_state
    state_fp = (
        torch.zeros(batch, heads, d_k, d_v, dtype=torch.float32, device=value.device)
        if state is None
        else state.to(torch.float32)
    )
    outs: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    kv_mems: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    for t in range(seq_len):
        kv_mem, delta = _state_faces(
            key[:, t], value[:, t], g[:, t], beta[:, t], state_fp, l2norm, use_l2
        )
        out_t, new_state = real_recurrent(
            query[:, t : t + 1],
            key[:, t : t + 1],
            value[:, t : t + 1],
            g=g[:, t : t + 1],
            beta=beta[:, t : t + 1],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_l2,
        )
        if edit_state is not None:
            new_state = edit_state(t, new_state)
        outs.append(out_t)
        states.append(new_state)
        kv_mems.append(kv_mem)
        deltas.append(delta)
        state = new_state
        state_fp = new_state.to(torch.float32)
    return (
        torch.cat(outs, dim=1),
        state,
        torch.stack(states, dim=1),
        torch.stack(kv_mems, dim=1),
        torch.stack(deltas, dim=1),
    )


def _with_state_taps(
    current: tuple[DeltaTap, ...],
    real: Callable[..., Any],
    originals: dict[str, Any],
    modeling: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor | None,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Serve the per-step slots around one kernel call.

    Three cases, in the order they are checked:

    * a **single cached decode step** (the recurrent kernel, one token): the
      per-step interior is the stock path — the state is the call's own
      return, the faces derive from its ``initial_state`` and arguments,
      nothing extra runs;
    * a **state write** at prefill: the stepwise loop *substitutes* for the
      chunked call, so edits feed forward. That is path-forcing and carries
      the measured cost (📐 logits 5.4e-7, pinned as a per-layer test bound) —
      paid only when a write targets the state, only at this layer;
    * **reads** at prefill: the chunked kernel still runs untouched (the
      forward's numbers are bit-identical) and the loop runs in its shadow,
      costing O(seq) extra kernel calls at this layer only.
    """
    assert g is not None and beta is not None  # the modeling call always passes both
    l2norm = _l2norm_of(modeling)
    use_l2 = bool(kwargs.get("use_qk_l2norm_in_kernel", False))
    initial_state = kwargs.get("initial_state")
    real_recurrent = originals["torch_recurrent_gated_delta_rule"]
    wants_final = bool(kwargs.get("output_final_state", False))
    edits = [tap.edit_state for tap in current if tap.edit_state is not None]

    def edit_state(step: int, state: torch.Tensor) -> torch.Tensor:
        for edit in edits:
            state = edit(step, state)
        return state

    if real is real_recurrent and key.shape[1] == 1 and not edits:
        # a native decode step: state = the call's own return, faces derived
        out, new_state = real(query, key, value, g=g, beta=beta, **kwargs)
        state_fp = (
            torch.zeros(
                key.shape[0],
                key.shape[2],
                key.shape[-1],
                value.shape[-1],
                dtype=torch.float32,
                device=value.device,
            )
            if initial_state is None
            else initial_state.to(torch.float32)
        )
        kv_mem, delta = _state_faces(
            key[:, 0], value[:, 0], g[:, 0], beta[:, 0], state_fp, l2norm, use_l2
        )
        _apply(current, "kv_mem", kv_mem.unsqueeze(1))
        _apply(current, "state_update", delta.unsqueeze(1))
        if new_state is not None:
            _apply(current, "state", new_state.unsqueeze(1))
        return out, new_state

    if edits:
        # substitution: the loop IS the forward for this layer
        out, final_state, states, kv_mems, deltas = _stepwise(
            real_recurrent,
            l2norm,
            query,
            key,
            value,
            g,
            beta,
            initial_state,
            use_l2,
            edit_state=edit_state,
        )
        _apply(current, "kv_mem", kv_mems)
        _apply(current, "state_update", deltas)
        _apply(current, "state", states)
        return out, (final_state if wants_final else None)

    # reads only: the base forward is untouched — the chunked kernel still
    # runs and the logits are bit-identical; the loop runs in its shadow
    out, state = real(query, key, value, g=g, beta=beta, **kwargs)
    _, _, states, kv_mems, deltas = _stepwise(
        real_recurrent, l2norm, query, key, value, g, beta, initial_state, use_l2
    )
    _apply(current, "kv_mem", kv_mems)
    _apply(current, "state_update", deltas)
    _apply(current, "state", states)
    return out, state


def _by_identity(
    taps: Mapping[Any, tuple[DeltaTap, ...]], mixer: Any
) -> tuple[DeltaTap, ...]:
    """The taps for one mixer, by object identity (nn.Module hashes by
    identity, so a plain lookup is exactly this — kept as a function so the
    intent survives a mapping type that hashes differently)."""
    if mixer is None:
        return ()
    return taps.get(mixer, ())


def _enter(active: dict[str, Any], mixer: Any) -> Callable[..., None]:
    def hook(_m: Any, args: tuple[Any, ...]) -> None:
        active["mixer"] = mixer
        hidden = args[0] if args else None
        active["seq_len"] = (
            int(hidden.shape[1]) if isinstance(hidden, torch.Tensor) else None
        )

    return hook


def _leave(active: dict[str, Any]) -> Callable[..., None]:
    def hook(_m: Any, _args: Any, _out: Any) -> None:
        active["mixer"] = None
        active["seq_len"] = None

    return hook


def _restore(modeling: Any, originals: dict[str, Any]) -> None:
    for name, value in originals.items():
        setattr(modeling, name, value)
