"""Prompt builders for the subspace-characterization judge.

Three derive framings (``default``, ``broad``, ``contrastive``) feed
``derive_hypothesis``. One ``reconcile`` builder feeds
``reconcile_hypotheses``. Derive builders see only ``WebtextEvidence`` and
``Step1Summary``; the reconcile builder sees ``JudgeHypothesis`` and
``Significance``. The independence invariant lives in the *signatures*: no
derive builder accepts a ``Significance`` argument.
"""

from __future__ import annotations

from typing import Any

from causalab.analyses.characterize_subspace.schemas import (
    JudgeHypothesis,
    Significance,
    Span,
    Step1Summary,
    WebtextEvidence,
)


# Each chat message is `{"role": "system" | "user" | "assistant", "content": str}`.
# We keep this as a plain dict shape so this module doesn't import openai;
# the LLM client converts to the SDK's TypedDicts at send time.
Message = dict[str, Any]


_DERIVE_RESPONSE_INSTRUCTIONS = """\
Respond with a single JSON object on its own line. The schema is:

{
  "hypothesis": "<one-sentence claim about what concept makes tokens activate strongly in this subspace>",
  "confidence": <float in [0, 1]>,
  "supporting_spans": ["<short excerpt 1>", "<short excerpt 2>", ...]
}

Do not include any prose outside the JSON object.
"""


_RECONCILE_RESPONSE_INSTRUCTIONS = """\
Respond with a single JSON object on its own line. The schema is:

{
  "verdict": "confirmed" | "refined" | "disagreed" | "unresolved",
  "rationale": "<one paragraph explaining the verdict>"
}

Use "confirmed" when the two hypotheses describe the same semantic axis.
Use "refined" when the derived hypothesis is a sharper or broader version of
the provided one along the same axis. Use "disagreed" when they describe
different axes. Use "unresolved" only when the evidence is insufficient to
choose. Do not include any prose outside the JSON object.
"""


def _format_span(s: Span, *, max_chars: int = 240) -> str:
    # ``projection_value`` is the token's subspace-activation norm (>= 0): how
    # strongly it fires in the subspace, not a signed position on an axis.
    text = s.text.replace("\n", " ").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return f"  [‖act‖={s.projection_value:.4f}] {text}"


def _format_evidence_block(evidence: WebtextEvidence) -> str:
    lines: list[str] = []
    lines.append(f"WEBTEXT CORPUS: {evidence.corpus}")
    stats = evidence.stats
    lines.append(
        f"Activation-norm distribution: n={stats.n_samples}, "
        f"mean={stats.mean:+.4f}, std={stats.std:.4f}, "
        f"min={stats.min:+.4f}, max={stats.max:+.4f}"
    )
    lines.append("")
    lines.append("QUANTILE-BINNED SPANS (weakest activation → strongest activation):")
    for qb in evidence.quantile_bins:
        lo, hi = qb.projection_range
        lines.append(f"\n  quantile {qb.quantile:.2f} (‖act‖ in [{lo:.4f}, {hi:.4f}]):")
        for s in qb.spans:
            lines.append(_format_span(s))
    lines.append("")
    lines.append("STRONGEST-ACTIVATING SPANS (largest subspace norm):")
    for s in evidence.topk_spans:
        lines.append(_format_span(s))
    lines.append("")
    lines.append("WEAKEST-ACTIVATING SPANS (smallest norm — typically generic text):")
    for s in evidence.bottomk_spans:
        lines.append(_format_span(s))
    return "\n".join(lines)


def _format_step1_block(step1: Step1Summary) -> str:
    stats = step1.stats
    lines = [
        f"PHASE-1 DATASET: {step1.dataset_name}",
        (
            f"Activation-norm distribution: n={stats.n_samples}, "
            f"mean={stats.mean:+.4f}, std={stats.std:.4f}, "
            f"min={stats.min:+.4f}, max={stats.max:+.4f}"
        ),
        "",
        "EXAMPLE PHASE-1 SPANS:",
    ]
    for s in step1.example_spans:
        lines.append(_format_span(s))
    return "\n".join(lines)


def derive_default(
    evidence: WebtextEvidence,
    step1_summary: Step1Summary,
) -> list[Message]:
    """Standard auto-interp framing."""
    system = (
        "You are a mechanistic-interpretability research assistant. You will "
        "be shown text spans drawn from a broad webtext corpus and from a "
        "narrow phase-1 dataset, each annotated with ‖act‖ — the Euclidean "
        "norm of its strongest token's projection into a candidate subspace "
        "inside a language model's residual stream (a non-negative measure of "
        "how strongly that token fires in the subspace, with direction "
        "discarded). Your job is to identify the concept that makes tokens fire "
        "strongly: what do the strongest-activating spans share that the "
        "weakest (generic) ones lack? Be specific: 'sentiment' is too vague; "
        "'expressions of regret about past decisions' is the right level."
    )
    user = (
        _format_step1_block(step1_summary)
        + "\n\n"
        + _format_evidence_block(evidence)
        + "\n\n"
        + _DERIVE_RESPONSE_INSTRUCTIONS
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def derive_broad(
    evidence: WebtextEvidence,
    step1_summary: Step1Summary,
) -> list[Message]:
    """Framing that asks the judge to consider broader concepts first.

    Useful when the default framing tends to over-fit to surface features
    of the phase-1 dataset.
    """
    system = (
        "You are a mechanistic-interpretability research assistant. The "
        "subspace you are analysing was discovered using a narrow phase-1 "
        "dataset, but it may track a *broader* concept that the phase-1 "
        "data only samples a slice of. Start by considering high-level "
        "concept categories (topic, register, style, sentiment, syntactic "
        "structure, named entity type, factual / procedural / narrative "
        "mode, ...) that the webtext quantile bins as a whole are consistent "
        "with, then refine to the most specific description supported by "
        "all the evidence."
    )
    user = (
        _format_step1_block(step1_summary)
        + "\n\n"
        + _format_evidence_block(evidence)
        + "\n\n"
        + _DERIVE_RESPONSE_INSTRUCTIONS
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def derive_contrastive(
    evidence: WebtextEvidence,
    step1_summary: Step1Summary,
) -> list[Message]:
    """Contrastive framing: focus on strongest- vs weakest-activating spans.

    Suppresses the quantile bins in favour of an explicit strongest-vs-weakest
    contrast, which is sometimes easier for the model to characterise.
    """
    system = (
        "You are a mechanistic-interpretability research assistant. You will "
        "be shown two sets of text spans: the STRONGEST-activating spans "
        "(largest subspace norm) and the WEAKEST-activating spans (smallest "
        "norm, typically generic text). Describe the concept that makes the "
        "strongest spans fire in this subspace. The score is a non-negative "
        "magnitude, so direction is not represented — characterise what the "
        "strongest spans share that makes them activate. Note that two opposite "
        "concepts on a single axis can both activate strongly; if the strongest "
        "spans look like two distinct groups, say so."
    )
    user = (
        _format_step1_block(step1_summary)
        + "\n\n"
        + f"WEBTEXT CORPUS: {evidence.corpus}\n\n"
        + "STRONGEST-ACTIVATING SPANS:\n"
        + "\n".join(_format_span(s) for s in evidence.topk_spans)
        + "\n\nWEAKEST-ACTIVATING SPANS (typically generic text):\n"
        + "\n".join(_format_span(s) for s in evidence.bottomk_spans)
        + "\n\n"
        + _DERIVE_RESPONSE_INSTRUCTIONS
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


DERIVE_FRAMINGS: dict[str, Any] = {
    "default": derive_default,
    "broad": derive_broad,
    "contrastive": derive_contrastive,
}


def reconcile(
    judge: JudgeHypothesis,
    provided: Significance,
    framing: str,
) -> list[Message]:
    """Compare a derived hypothesis against the provided significance description.

    The full ``Significance`` is shown here — this is the only call where it
    is allowed to enter a prompt. The framing string is logged but does not
    change the prompt content; it is used to track which reconcile attempt
    in a retry loop produced which verdict.
    """
    system = (
        "You are a mechanistic-interpretability research assistant. You are "
        "comparing two hypotheses about what semantic axis a particular "
        "subspace inside a language model tracks. Decide whether they "
        "describe the same axis."
    )

    provided_block_parts: list[str] = []
    if provided.hypothesis_text:
        provided_block_parts.append(
            f"Provided hypothesis (text):\n  {provided.hypothesis_text}"
        )
    if provided.figure_path:
        provided_block_parts.append(
            f"Provided significance figure (path, content not shown):\n  {provided.figure_path}"
        )
    if provided.topology_description:
        provided_block_parts.append(
            f"Provided topology description:\n  {provided.topology_description}"
        )
    if not provided_block_parts:
        provided_block_parts.append("Provided hypothesis: (none supplied)")
    provided_block = "\n\n".join(provided_block_parts)

    derived_block = (
        f"Independently derived hypothesis:\n  {judge.hypothesis_text}\n\n"
        f"Derivation confidence: {judge.confidence:.2f}\n"
        f"Derivation framing: {judge.framing}"
    )
    if judge.supporting_spans:
        derived_block += "\n\nDerivation cited these spans as evidence:\n" + "\n".join(
            f"  - {s}" for s in judge.supporting_spans[:5]
        )

    user = (
        provided_block
        + "\n\n"
        + derived_block
        + "\n\n"
        + f"(Reconciliation framing: {framing})\n\n"
        + _RECONCILE_RESPONSE_INSTRUCTIONS
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
