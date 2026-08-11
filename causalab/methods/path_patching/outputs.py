"""Plan-saved logits → the :class:`~causalab.neural.pipeline.GenerationResult`
generate-output shape.

The scoring stack (:func:`causalab.methods.metric.score_intervention_outputs`
and every :class:`~causalab.methods.metric.InterchangeMetric`) consumes the
flat :class:`~causalab.neural.pipeline.GenerationResult` a generation run
produces: ``sequences`` ``(n, new_tokens)``, per-step ``scores`` ``(n,
vocab)``, and decoded ``strings`` (EU5b, #487). A
:class:`~causalab.neural.plan.PlanResult` carries full-sequence prefill logits
instead; at ``max_new_tokens == 1`` — the path-patching scoring contract — the
two are the same numbers: greedy generation's single step *is* the prefill, so
``scores[0]`` equals the last-position logits and the generated token is their
argmax. This module is that adapter, private to the path-patching runner.

Multi-token generation under interventions is a genuine capability boundary of
the prefill-only Plan IR (decode-step edits are CAP1 headroom, #425):
:func:`check_single_step` refuses it loudly rather than silently scoring
step 0 of what the caller believes is a longer generation.
"""

from __future__ import annotations

import torch

from causalab.neural.pipeline import GenerationResult, LMPipeline

__all__ = ["check_single_step", "plan_outputs"]


def check_single_step(pipeline: LMPipeline) -> None:
    """Refuse a multi-token pipeline before any forward pass."""
    if pipeline.max_new_tokens != 1:
        raise NotImplementedError(
            f"path patching scores single-step outputs, but the pipeline has "
            f"max_new_tokens={pipeline.max_new_tokens}. Decode-step edits "
            f"exist at the plan layer (generation plans, CAP2 #455), but the "
            f"path-patching metrics score the first generated token only — "
            f"load the pipeline with max_new_tokens=1."
        )


def plan_outputs(
    pipeline: LMPipeline,
    logits: torch.Tensor,
    *,
    output_scores: bool | int = True,
) -> GenerationResult:
    """One batch's Plan-saved logits as a :class:`GenerationResult`.

    ``logits`` is the saved full-sequence tensor ``(B, seq, vocab)`` (CPU, the
    Plan convention). ``scores`` keeps full-vocab tensors here regardless of an
    integer ``output_scores`` — the runner applies the shared
    :func:`~causalab.neural.pipeline.compress_scores_top_k` over the flat
    result at the end, exactly like the generate path does.
    """
    last = logits[:, -1, :]
    sequences = last.argmax(dim=-1, keepdim=True)
    decoded = pipeline.dump(sequences, is_logits=False)
    return GenerationResult(
        sequences=sequences,
        strings=[decoded] if isinstance(decoded, str) else decoded,
        scores=[last] if output_scores else None,
    )
