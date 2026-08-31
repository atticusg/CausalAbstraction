"""The reference engine: intervention protocols on native pytorch hooks.

Implements the spec §8 services on the two supported architecture families.
Capabilities: ``grad`` (the train loop, train.py), ``paired_forward``
(cross-input operand flow via the lazy group executor), ``full_logits``
(lm_head is an ordinary tap), ``generate`` (the greedy decode in
executor.py, which interventions reach only through the prefill), and
``pytorch_fn_local`` (this engine is local), and
``writable_attention_probs`` — a write to the attention pattern reaches the
output through the eager attention function rather than a forward hook,
because the hook fires after the pattern has already been consumed (see
``attention_probs.py``).

The engine-neutral half of ``execute`` — metric lowering and the output
tables — lives in :mod:`causalab.neural.shared.execution`; this module keeps
what is genuinely this engine's: its loader, its executor, its train loop.
"""

from __future__ import annotations

import functools
from typing import Any, Mapping

from causalab.neural.engines.pytorch_hooks.executor import Interning, PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.execution import execute_request
from causalab.neural.shared.services import load_tensors, resolve_roles
from causalab.protocol.canonical import canonical_model
from causalab.protocol.engine import Engine, ExecutionRequest, RunResult
from causalab.protocol.schema import COMPONENTS, Document

__all__ = ["PytorchHooksEngine"]


class PytorchHooksEngine(Engine):
    name = "pytorch_hooks"
    capabilities = frozenset(
        {
            "grad",
            "paired_forward",
            "full_logits",
            "generate",
            "pytorch_fn_local",
            "quantized_weights",
            "writable_attention_probs",
        }
    )
    # The reference engine serves the module-boundary and attention-interface
    # vocabulary, writes included — and the routed-expert interior, which
    # round 3 reaches by wrapping the grouped experts dispatch (there is no
    # per-expert module, but the dispatch entry is this engine's to replace).
    # Read-only/swap-only components and stream constraints are *protocol
    # policy* (the sites.py refusal tables, shared across engines), not
    # capability gaps: declaring router_logits unwritable here would turn "a
    # write here reaches nothing, write router_scores instead" into "try
    # another engine", which is the wrong answer for every engine. What stays
    # another engine's vocabulary: `expert_permutation` (the serving kernel's
    # own bookkeeping, a `.source` line with no dispatch-slot face) and the
    # Gated DeltaNet interior (N7) — tensors inside a fused forward where no
    # hook can reach. Absent from these sets, so routing names the nnsight
    # engine for free, and a document arriving here unrouted refuses by name
    # in the executor.
    _INTERIOR_COMPONENTS = frozenset({"expert_permutation"}) | frozenset(
        c for c in COMPONENTS if c.startswith("deltanet_")
    )
    components = frozenset(COMPONENTS) - _INTERIOR_COMPONENTS
    writable_components = frozenset(COMPONENTS) - _INTERIOR_COMPONENTS
    is_local = True

    def __init__(self, *, device: str = "cpu") -> None:
        # placement is execution (the engine's call, §8); precision is not —
        # dtype and quantization come from each point's own `model` section
        self.device = device

    # ------------------------------------------------------------------ #

    def execute(self, request: ExecutionRequest) -> RunResult:
        from causalab.neural.engines.pytorch_hooks.train import run_training

        return execute_request(
            request,
            engine_name=self.name,
            executor_factory=lambda doc, req, coords, interning: self._executor(
                doc, req, coords=coords, interning=interning
            ),
            train_runner=run_training,
            # this engine's executor consults the shared ForwardCache, so it
            # claims §3's cross-point interning and can report what it paid
            intern_forwards=True,
        )

    # ------------------------------------------------------------------ #

    def _executor(
        self,
        doc: Document,
        request: ExecutionRequest,
        *,
        grad_enabled: bool = False,
        coords: Mapping[str, Any] | None = None,
        interning: Interning | None = None,
    ) -> PointExecutor:
        realization = canonical_model(doc.raw["model"])
        bundle = load_model(
            str(doc.model.key),
            str(doc.model.revision),
            dtype=str(realization["dtype"]),
            device=self.device,
            quantization=_quantization_key(realization),
        )
        role_rows, role_fields = resolve_roles(doc, request)
        return PointExecutor(
            doc,
            bundle,
            role_rows=role_rows,
            role_fields=role_fields,
            load_tensors=functools.partial(load_tensors, request),
            grad_enabled=grad_enabled,
            coords=coords,
            interning=interning,
        )


def _quantization_key(
    realization: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...] | None:
    """The materialized ``quantization`` block as a hashable, order-free key —
    :func:`load_model` caches on it, so it must hash and compare by value."""
    quantization = realization.get("quantization")
    if quantization is None:
        return None
    return tuple(sorted(quantization.items()))
