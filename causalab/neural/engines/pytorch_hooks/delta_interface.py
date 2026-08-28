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
    "kernel_output",
)

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
    """

    slot: str
    read: Callable[[torch.Tensor], None] | None = None
    edit: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.slot not in DELTA_SLOTS:
            raise ValueError(
                f"unknown delta-kernel slot {self.slot!r}; expected one of "
                f"{DELTA_SLOTS}"
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

    def kernel_wrapper(real: Callable[..., Any]) -> Callable[..., Any]:
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
                kernel_wrapper(originals["torch_chunk_gated_delta_rule"]),
            )
            setattr(
                modeling,
                "torch_recurrent_gated_delta_rule",
                kernel_wrapper(originals["torch_recurrent_gated_delta_rule"]),
            )
            stack.callback(_restore, modeling, originals)
        for mixer in taps:
            pre = mixer.register_forward_pre_hook(_enter(active, mixer))
            post = mixer.register_forward_hook(_leave(active))
            stack.callback(pre.remove)
            stack.callback(post.remove)
        yield


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
