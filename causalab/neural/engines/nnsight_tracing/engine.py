"""The nnsight engine's entry point.

Same shape as the reference engine's: capability and component declarations
for routing, a loader, an executor factory, and the shared execution
orchestration. What it does **not** declare says as much as what it does:

* ``grad`` — training through traces is real design work (plan §2.5, D6);
  ``train`` documents route to the reference engine.
* ``generate`` — step-anchored trace reads are phase N8.
* ``writable_attention_probs`` / the ``attention_probs`` component — the
  attention-interior taps (eager interface source) are phase N5.
* ``quantized_weights`` — unverified through nnsight's loader; refused until
  someone needs it and proves it.

Components it will serve that no other engine can — the DeltaNet state, the
expert interiors (plan §3, N6–N7) — enter ``schema.py`` when their phases
land, and routing carries documents here by name.
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
        }
    )
    # The module-boundary vocabulary (N4). The attention interior joins with
    # N5; the interiors this engine exists for (DeltaNet state, expert
    # interiors) join the schema with their phases.
    #
    # ⚠️ Round 2's nine attention components are excluded here, for two
    # different reasons and neither of them "not implemented yet":
    #
    # * the four *function-interior* taps and 'attention_probs' live inside one
    #   `attention_interface(...)` call and are reached by registering an eager
    #   wrapper — a pytorch_hooks mechanism with no nnsight equivalent, which is
    #   exactly what N5 is for;
    # * 'attention_result' is *derived*: computing it re-invokes the
    #   o-projection, and this engine's `site.module` is an nnsight envoy rather
    #   than a callable module.
    #
    # The four module-boundary taps (v, the pre-RoPE projections, the gate)
    # would very likely work here unchanged — they are ordinary envoy reads —
    # but nothing exercises them on this engine, and declaring support this
    # engine has never been tested for is the claim worth not making.
    _ROUND_TWO_ATTENTION = frozenset(
        {
            "attention_probs",
            "attention_query",
            "attention_key",
            "attention_scores",
            "attention_z",
            "attention_result",
            "attention_value_states",
            "attention_query_pre_rope",
            "attention_key_pre_rope",
            "attention_gate",
        }
    )
    # Round 3's routed-expert interior is excluded for the first of those two
    # reasons: every one of its components is reached by replacing the
    # `ALL_EXPERTS_FUNCTIONS["grouped_mm"]` dispatch entry and patching
    # `_grouped_linear` inside it — a pytorch_hooks mechanism the tracing
    # engine cannot express (there is no per-expert module for an envoy to
    # bind to; the experts module's only child is one shared act_fn).
    _ROUND_THREE_MOE_INTERIOR = frozenset(
        {
            "expert_gate_proj",
            "expert_up_proj",
            "expert_activation",
            "expert_output",
        }
    )
    _UNDECLARED = _ROUND_TWO_ATTENTION | _ROUND_THREE_MOE_INTERIOR
    components = frozenset(COMPONENTS) - _UNDECLARED
    writable_components = frozenset(COMPONENTS) - _UNDECLARED
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
