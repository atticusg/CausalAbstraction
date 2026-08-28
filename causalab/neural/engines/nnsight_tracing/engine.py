"""The nnsight engine's entry point.

Same shape as the reference engine's: capability and component declarations
for routing, a loader, an executor factory, and the shared execution
orchestration. What it does **not** declare says as much as what it does:

* ``grad`` — training through traces is real design work (plan §2.5, D6);
  ``train`` documents route to the reference engine.
* ``quantized_weights`` — unverified through nnsight's loader; refused until
  someone needs it and proves it.

The components only this engine serves — the fused-forward interiors of
N6/N7, and the decode-side DeltaNet state (N8) — route here by name: the
reference engine simply does not declare them.
"""

from __future__ import annotations

import functools
from typing import Any, Mapping

from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.nnsight_tracing.loading import load_model
from causalab.neural.shared.execution import execute_request
from causalab.neural.shared.services import load_tensors, resolve_roles
from causalab.protocol.canonical import canonical_model
from causalab.protocol.engine import Engine, ExecutionRequest, RunResult
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import COMPONENTS, Document

__all__ = ["NnsightEngine"]


class NnsightEngine(Engine):
    name = "nnsight"
    capabilities = frozenset(
        {
            "paired_forward",
            "full_logits",
            "pytorch_fn_local",
            # the pattern write lands on the softmax's output *inside* the
            # eager function ('attn_weights_2' in the N5 address table), where
            # the value multiply consumes it — a write to the mixer's returned
            # attn_weights would reach nothing (#53's finding)
            "writable_attention_probs",
            # continuation reads through one model.generate trace, decode
            # steps walked with tracer.iter (N8); writes stay in the prefill,
            # as everywhere
            "generate",
        }
    )
    # The whole current vocabulary, like the reference engine (N5): the
    # module boundaries land on envoys, the attention interior through the
    # `.source` address table, and 'attention_result' — derived by re-invoking
    # the o-projection — works because an envoy outside a trace calls its
    # underlying module (measured in the N5 probes). Read-only/swap-only
    # components and stream constraints are *protocol policy* (the shared
    # sites.py refusal tables), not capability gaps — the same argument the
    # reference engine's declaration makes. The per-expert MoE interior (N6)
    # is the first vocabulary only this engine serves — the reference engine
    # leaves it undeclared, so routing lands it here by name; the DeltaNet
    # interior joins the schema with N7 and enters the same way.
    components = frozenset(COMPONENTS)
    writable_components = frozenset(COMPONENTS)
    is_local = True

    def __init__(self, *, device: str = "cpu") -> None:
        # placement is execution (the engine's call, §8); precision is not —
        # dtype comes from each point's own `model` section
        self.device = device

    # ------------------------------------------------------------------ #

    def execute(self, request: ExecutionRequest) -> RunResult:
        return execute_request(
            request,
            engine_name=self.name,
            executor_factory=lambda doc, req, coords: self._executor(
                doc, req, coords=coords
            ),
            train_runner=None,
        )

    # ------------------------------------------------------------------ #

    def _executor(
        self,
        doc: Document,
        request: ExecutionRequest,
        *,
        coords: Mapping[str, Any] | None = None,
    ) -> TracePointExecutor:
        realization = canonical_model(doc.raw["model"])
        if realization.get("quantization") is not None:
            raise ProtocolError(
                "P4",
                "this document declares weight quantization, which the "
                "nnsight engine has not verified through its loader — its "
                "'quantized_weights' capability is absent, so routing should "
                "not have sent it here; the reference engine serves it",
            )
        bundle = load_model(
            str(doc.model.key),
            str(doc.model.revision),
            dtype=str(realization["dtype"]),
            device=self.device,
        )
        role_rows, role_fields = resolve_roles(doc, request)
        return TracePointExecutor(
            doc,
            bundle,
            role_rows=role_rows,
            role_fields=role_fields,
            load_tensors=functools.partial(load_tensors, request),
            coords=coords,
        )
