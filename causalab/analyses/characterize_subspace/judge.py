"""Two-call judge for the subspace-characterization analysis.

Composes the domain-neutral ``causalab.methods.llm_judge`` primitives
(``call_llm``, ``extract_json_response``, ``assert_no_forbidden_substrings``)
with this analysis's prompts and schemas to enforce judge independence on
three layers:

1. **Type-level** — ``derive_hypothesis`` does not accept ``Significance``.
2. **Schema-level** — ``Step1Summary`` carries no significance fields.
3. **Runtime** — before the derive request is sent, the rendered prompt
   is scanned against the caller-supplied list of forbidden substrings.
   Collision raises :class:`~causalab.methods.llm_judge.ForbiddenSubstringError`.

The reconcile path is the only place that mixes both sides of the handoff.
"""

from __future__ import annotations

import logging
from typing import Any

from causalab.analyses.characterize_subspace.prompts import (
    DERIVE_FRAMINGS,
    reconcile as build_reconcile_prompt,
)
from causalab.analyses.characterize_subspace.schemas import (
    JudgeHypothesis,
    ReconciliationIteration,
    ReconciliationResult,
    Significance,
    Step1Summary,
    Verdict,
    WebtextEvidence,
)
from causalab.methods.llm_judge import (
    Provider,
    assert_no_forbidden_substrings,
    call_llm,
    extract_json_response,
)

logger = logging.getLogger(__name__)


def derive_hypothesis(
    *,
    evidence: WebtextEvidence,
    step1_summary: Step1Summary,
    framing: str,
    model: str,
    provider: Provider = "openrouter",
    max_tokens: int = 4096,
    forbidden_substrings: list[str] = [],
) -> JudgeHypothesis:
    """Derive a semantic-axis hypothesis from broad webtext evidence.

    The caller passes ``forbidden_substrings`` (typically the result of
    ``Significance.non_empty_values()``) so the runtime guard can verify
    the rendered prompt is free of leaked significance content.
    ``Significance`` is intentionally NOT in this signature — that is the
    type-system half of the independence invariant.

    Raises:
        ForbiddenSubstringError: a ``forbidden_substrings`` entry was found
            in the prompt — orchestrator bug, do not retry blindly.
        KeyError: ``framing`` is not a registered derive framing.
        ValueError: the LLM response is unparseable.
    """
    if framing not in DERIVE_FRAMINGS:
        raise KeyError(
            f"Unknown derive framing: {framing!r}. Available: {sorted(DERIVE_FRAMINGS)}"
        )
    builder = DERIVE_FRAMINGS[framing]
    messages = builder(evidence, step1_summary)

    rendered = "\n\n".join(m["content"] for m in messages)
    assert_no_forbidden_substrings(rendered, forbidden_substrings)

    raw = call_llm(messages, model=model, max_tokens=max_tokens, provider=provider)
    parsed = extract_json_response(raw)

    hypothesis_text = str(parsed.get("hypothesis", "")).strip()
    if not hypothesis_text:
        raise ValueError(
            f"LLM response missing 'hypothesis' field. Raw (truncated): {raw[:400]!r}"
        )
    confidence = float(parsed.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    supporting = parsed.get("supporting_spans", []) or []
    supporting_spans = [str(s) for s in supporting if isinstance(s, str)]

    return JudgeHypothesis(
        hypothesis_text=hypothesis_text,
        confidence=confidence,
        supporting_spans=supporting_spans,
        framing=framing,
        model=model,
        raw_response=raw,
    )


def _normalise_verdict(value: Any) -> Verdict:
    if isinstance(value, str) and value.strip().lower() in {
        "confirmed",
        "refined",
        "disagreed",
        "unresolved",
    }:
        return value.strip().lower()  # type: ignore[return-value]
    return "unresolved"


def reconcile_hypotheses(
    *,
    judge: JudgeHypothesis,
    provided: Significance,
    framings: list[str],
    model: str,
    provider: Provider = "openrouter",
    max_tokens: int = 4096,
    max_iterations: int = 3,
) -> ReconciliationResult:
    """Compare a derived hypothesis to the provided significance description.

    Iterates through ``framings`` up to ``max_iterations`` total tries. The
    first ``confirmed`` short-circuits the loop. Otherwise the final verdict
    is the last iteration's verdict.
    """
    if not framings:
        raise ValueError("reconcile_hypotheses requires at least one framing.")

    iterations: list[ReconciliationIteration] = []
    chosen_framing = framings[0]
    final_verdict: Verdict = "unresolved"
    final_rationale = ""

    for i in range(max_iterations):
        framing = framings[i % len(framings)]
        messages = build_reconcile_prompt(judge, provided, framing)
        raw = call_llm(messages, model=model, max_tokens=max_tokens, provider=provider)
        parsed = extract_json_response(raw)
        verdict = _normalise_verdict(parsed.get("verdict"))
        rationale = str(parsed.get("rationale", "")).strip()
        iterations.append(
            ReconciliationIteration(
                framing=framing,
                verdict=verdict,
                rationale=rationale,
                raw_response=raw,
            )
        )
        chosen_framing = framing
        final_verdict = verdict
        final_rationale = rationale
        if verdict == "confirmed":
            break

    return ReconciliationResult(
        verdict=final_verdict,
        judge_hypothesis=judge.hypothesis_text,
        provided_hypothesis=provided.hypothesis_text,
        rationale=final_rationale,
        iterations=iterations,
        final_framing=chosen_framing,
        model=model,
    )
