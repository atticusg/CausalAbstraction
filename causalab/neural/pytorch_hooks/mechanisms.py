"""The closed ``do`` mechanism set (spec §2.8), applied in feature space.

Per (site, overlapping pos, model): the absolute write (if any) applies
first, then additive deltas sum — the class order that makes write sets
order-free. ``dims`` scatter and the error-term contract live in the
executor (the mechanism sees the feature slice it writes).

``gaussian`` realizes the RNG contract the parity goldens pin: the draw is
``torch.Generator().manual_seed(seed)`` → ``randn((batch, n_pos, width))``,
made **outside** the model, once per write application.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping

import torch

from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import ADDITIVE_MECHANISMS, Do

__all__ = ["apply_absolute", "apply_delta", "is_additive"]

#: Resolve an operand to a tensor/scalar: the executor passes a lookup over
#: read values, params, and dotted featurizer slots.
OperandLookup = Callable[[Any], torch.Tensor | float]


def is_additive(do: Do) -> bool:
    return str(do.mechanism) in ADDITIVE_MECHANISMS


def _operand(lookup: OperandLookup, value: Any) -> torch.Tensor | float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return lookup(value)


def apply_absolute(do: Do, f: torch.Tensor, lookup: OperandLookup) -> torch.Tensor:
    """The absolute-class write ``f ← …`` for one mechanism; ``f`` is the
    pre-write feature slice (already dims-selected)."""
    mech = str(do.mechanism)
    payload = do.payload
    if mech == "swap":
        return _coerce(_operand(lookup, payload), f)
    if mech == "lerp":
        alpha = _scalar(_operand(lookup, payload["alpha"]))
        op = _coerce(_operand(lookup, payload["op"]), f)
        return (1.0 - alpha) * f + alpha * op
    if mech == "affine":
        a = _operand(lookup, payload["A"])
        b = _operand(lookup, payload["b"])
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ProtocolError("P2", "affine A/b must resolve to tensors")
        return f @ a.T.to(f.dtype) + b.to(f.dtype)
    if mech == "renormalize":
        raise AssertionError(
            "renormalize is applied by the executor after additive deltas "
            "(apply_renormalize) — under strict absolute-first order it would "
            "always be the identity; surfaced as a spec question"
        )
    if mech == "clamp":
        lo = _scalar(_operand(lookup, payload["lo"]))
        hi = _scalar(_operand(lookup, payload["hi"]))
        return f.clamp(min=lo, max=hi)
    if mech == "pytorch_fn":
        qualname = str(payload["qualname"])
        module_name, _, attr = qualname.rpartition(".")
        fn = getattr(importlib.import_module(module_name), attr)
        return fn(f)
    raise ProtocolError("P4", f"{mech!r} is not an absolute mechanism")


def apply_delta(
    do: Do, f_pre: torch.Tensor, lookup: OperandLookup, *, batch: int, n_pos: int
) -> torch.Tensor:
    """The additive-class delta for one mechanism (summed by the caller)."""
    mech = str(do.mechanism)
    payload = do.payload
    if mech == "add_scaled":
        alpha = _scalar(_operand(lookup, payload["alpha"]))
        op = _coerce(_operand(lookup, payload["op"]), f_pre)
        return alpha * op
    if mech == "gaussian":
        seed = int(payload["seed"])
        scale = float(payload["scale"])
        generator = torch.Generator().manual_seed(seed)
        draw = torch.randn(
            (batch, n_pos, f_pre.shape[-1]), generator=generator, dtype=torch.float32
        )
        return scale * draw.to(dtype=f_pre.dtype, device=f_pre.device).reshape(
            f_pre.shape
        )
    raise ProtocolError("P4", f"{mech!r} is not an additive mechanism")


def apply_renormalize(f: torch.Tensor, f_pre: torch.Tensor) -> torch.Tensor:
    """``f ← f·‖f₀‖/‖f‖`` with ``f₀`` the pre-write feature value. Runs after
    the additive deltas (the only order under which it is not the identity);
    it still counts as the address's one absolute write for rule 8."""
    target = f_pre.norm(dim=-1, keepdim=True)
    return f * (target / f.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def _scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _coerce(value: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    """Move an operand onto the written slice's device/dtype and check the
    one-way broadcast (right-aligned; the classic width mismatch — a counterfactual
    read contributing a different number of positions than the write
    addresses — must fail legibly here, not as a scatter assert)."""
    if not isinstance(value, torch.Tensor):
        return torch.full_like(like, float(value))
    value = value.to(device=like.device, dtype=like.dtype)
    ok = value.dim() <= like.dim() and all(
        v == 1 or v == t for v, t in zip(reversed(value.shape), reversed(like.shape))
    )
    if not ok:
        raise ProtocolError(
            "P2",
            f"operand of shape {tuple(value.shape)} does not broadcast to the "
            f"{tuple(like.shape)} slice it writes — position widths must "
            "pair up per example, or the operand must be a broadcastable vector",
        )
    return value


def operand_names(payload: Any) -> tuple[str, ...]:
    """Names referenced by a mechanism payload (mirrors the validator)."""
    if isinstance(payload, str):
        return (payload,)
    if isinstance(payload, Mapping):
        return tuple(v for v in payload.values() if isinstance(v, str))
    return ()
