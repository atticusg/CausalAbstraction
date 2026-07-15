"""Judge-independence tests for the subspace-characterization analysis.

The general LLM-judge primitives are covered in
``tests/test_llm_judge_primitives.py``; this file checks the analysis-level
invariants: ``derive_hypothesis`` never sees the provided significance
description, ``reconcile_hypotheses`` always does, and the reconcile loop
behaves as documented.

The LLM call is mocked so no network is hit; assertions are on the
rendered prompt strings the mock captures.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from causalab.analyses.characterize_subspace import (
    JudgeHypothesis,
    ProjectionStats,
    QuantileBin,
    Significance,
    Span,
    Step1Summary,
    WebtextEvidence,
    derive_hypothesis,
    reconcile_hypotheses,
)
from causalab.analyses.characterize_subspace import judge as judge_module
from causalab.methods.llm_judge import ForbiddenSubstringError

pytestmark = pytest.mark.unit


SENTINEL = "MAGIC_MARKER_47291"


def _make_evidence() -> WebtextEvidence:
    return WebtextEvidence(
        corpus="HuggingFaceFW/fineweb-edu",
        quantile_bins=[
            QuantileBin(
                quantile=0.1,
                projection_range=(-1.5, -0.5),
                spans=[Span(text="The river ran cold today.", projection_value=-1.2)],
            ),
            QuantileBin(
                quantile=0.9,
                projection_range=(0.5, 1.5),
                spans=[
                    Span(
                        text="The committee approved the motion.", projection_value=1.1
                    )
                ],
            ),
        ],
        topk_spans=[
            Span(text="A landmark ruling reshaped policy.", projection_value=2.3)
        ],
        bottomk_spans=[
            Span(text="Heavy rain pooled in the street.", projection_value=-2.1)
        ],
        stats=ProjectionStats(n_samples=10_000, mean=0.01, std=0.97, min=-2.4, max=2.5),
    )


def _make_step1_summary() -> Step1Summary:
    return Step1Summary(
        dataset_name="phase1_dataset_v3",
        stats=ProjectionStats(n_samples=512, mean=0.12, std=1.05, min=-2.1, max=2.4),
        example_spans=[
            Span(text="The bill cleared committee unanimously.", projection_value=1.7),
            Span(text="The water level rose past midnight.", projection_value=-1.4),
        ],
    )


def _stub_call_returning(payload: dict[str, Any]):
    """Return a stub for ``call_llm`` that captures every messages list."""
    captured: list[list[dict[str, Any]]] = []
    response_text = json.dumps(payload)

    def _stub(messages, *, model, max_tokens, provider="openrouter", max_retries=3):  # noqa: ARG001
        captured.append(messages)
        return response_text

    return _stub, captured


class TestDeriveIndependence:
    def test_sentinel_absent_from_derive_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub, captured = _stub_call_returning(
            {
                "hypothesis": "tracks formal procedural language",
                "confidence": 0.7,
                "supporting_spans": ["committee approved the motion"],
            }
        )
        monkeypatch.setattr(judge_module, "call_llm", stub)

        sig = Significance(hypothesis_text=f"the subspace tracks {SENTINEL} content")

        result = derive_hypothesis(
            evidence=_make_evidence(),
            step1_summary=_make_step1_summary(),
            framing="default",
            model="anthropic/claude-sonnet-4-5",
            forbidden_substrings=sig.non_empty_values(),
        )

        assert isinstance(result, JudgeHypothesis)
        assert result.hypothesis_text
        assert captured, "Stubbed call_llm was not invoked"
        for messages in captured:
            for msg in messages:
                assert SENTINEL not in msg["content"], (
                    "Sentinel from Significance.hypothesis_text leaked into the "
                    "derive-call prompt"
                )

    def test_runtime_guard_catches_leak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulate a buggy upstream that smuggles the sentinel into a span.

        The guard must fail loud — independent of where the leakage came from.
        """
        stub, _ = _stub_call_returning(
            {
                "hypothesis": "irrelevant — should not be reached",
                "confidence": 0.5,
                "supporting_spans": [],
            }
        )
        monkeypatch.setattr(judge_module, "call_llm", stub)

        leaky_summary = Step1Summary(
            dataset_name="phase1_dataset_v3",
            stats=ProjectionStats(n_samples=10, mean=0.0, std=1.0, min=-1.0, max=1.0),
            example_spans=[
                Span(
                    text=f"this span improperly contains {SENTINEL}.",
                    projection_value=0.3,
                ),
            ],
        )

        with pytest.raises(ForbiddenSubstringError):
            derive_hypothesis(
                evidence=_make_evidence(),
                step1_summary=leaky_summary,
                framing="default",
                model="anthropic/claude-sonnet-4-5",
                forbidden_substrings=[SENTINEL],
            )

    def test_empty_forbidden_substrings_allows_derive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Description-absent path: no guard needles, derive proceeds."""
        stub, _ = _stub_call_returning(
            {
                "hypothesis": "tracks formal procedural language",
                "confidence": 0.6,
                "supporting_spans": [],
            }
        )
        monkeypatch.setattr(judge_module, "call_llm", stub)

        result = derive_hypothesis(
            evidence=_make_evidence(),
            step1_summary=_make_step1_summary(),
            framing="default",
            model="anthropic/claude-sonnet-4-5",
            forbidden_substrings=[],
        )
        assert result.hypothesis_text


class TestMarkerCollision:
    def test_marked_window_spans_do_not_break_derive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spans now carry ``<<peak>>``-marked windows; derive must still run.

        The forbidden-substring guard must continue to fire on a real needle
        even when the (marker-bearing) span text is present in the prompt.
        """
        stub, captured = _stub_call_returning(
            {
                "hypothesis": "tracks formal procedural language",
                "confidence": 0.7,
                "supporting_spans": ["committee <<approved>> the motion"],
            }
        )
        monkeypatch.setattr(judge_module, "call_llm", stub)

        marked_evidence = WebtextEvidence(
            corpus="HuggingFaceFW/fineweb-edu",
            quantile_bins=[
                QuantileBin(
                    quantile=0.9,
                    projection_range=(0.5, 1.5),
                    spans=[
                        Span(
                            text="the committee <<approved>> the motion today",
                            projection_value=1.1,
                        )
                    ],
                )
            ],
            topk_spans=[
                Span(text="a landmark <<ruling>> reshaped policy", projection_value=2.3)
            ],
            bottomk_spans=[
                Span(text="heavy <<rain>> pooled in the street", projection_value=-2.1)
            ],
            stats=ProjectionStats(n_samples=10, mean=0.0, std=1.0, min=-2.1, max=2.3),
        )

        result = derive_hypothesis(
            evidence=marked_evidence,
            step1_summary=_make_step1_summary(),
            framing="default",
            model="anthropic/claude-sonnet-4-5",
            forbidden_substrings=[],
        )
        assert result.hypothesis_text
        joined = "\n".join(msg["content"] for messages in captured for msg in messages)
        assert "<<approved>>" in joined, "Marked window should reach the derive prompt"

        # The guard still fires on a real needle even with markers around.
        with pytest.raises(ForbiddenSubstringError):
            derive_hypothesis(
                evidence=marked_evidence,
                step1_summary=_make_step1_summary(),
                framing="default",
                model="anthropic/claude-sonnet-4-5",
                forbidden_substrings=["committee <<approved>> the motion"],
            )


class TestReconcileSeesSignificance:
    def test_sentinel_present_in_reconcile_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub, captured = _stub_call_returning(
            {
                "verdict": "confirmed",
                "rationale": "Both hypotheses describe the same axis.",
            }
        )
        monkeypatch.setattr(judge_module, "call_llm", stub)

        judge = JudgeHypothesis(
            hypothesis_text="tracks formal procedural language",
            confidence=0.8,
            supporting_spans=["committee approved the motion"],
            framing="default",
            model="anthropic/claude-sonnet-4-5",
            raw_response="{}",
        )
        sig = Significance(hypothesis_text=f"the subspace tracks {SENTINEL} content")

        result = reconcile_hypotheses(
            judge=judge,
            provided=sig,
            framings=["default"],
            model="anthropic/claude-sonnet-4-5",
            max_iterations=1,
        )

        assert result.verdict == "confirmed"
        assert captured, "Stubbed call_llm was not invoked"
        joined = "\n".join(msg["content"] for messages in captured for msg in messages)
        assert SENTINEL in joined, (
            "Significance.hypothesis_text should appear in the reconcile prompt"
        )

    def test_loop_short_circuits_on_first_confirmed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        n_calls = 0

        def _stub(messages, *, model, max_tokens, provider="openrouter", max_retries=3):  # noqa: ARG001
            nonlocal n_calls
            n_calls += 1
            return json.dumps({"verdict": "confirmed", "rationale": "ok"})

        monkeypatch.setattr(judge_module, "call_llm", _stub)

        judge = JudgeHypothesis(
            hypothesis_text="tracks X",
            confidence=0.5,
            supporting_spans=[],
            framing="default",
            model="m",
            raw_response="{}",
        )
        result = reconcile_hypotheses(
            judge=judge,
            provided=Significance(hypothesis_text="tracks X"),
            framings=["default", "broad", "contrastive"],
            model="m",
            max_iterations=3,
        )
        assert result.verdict == "confirmed"
        assert n_calls == 1, "Loop should short-circuit on the first 'confirmed'"
        assert len(result.iterations) == 1

    def test_loop_runs_to_completion_on_persistent_disagreement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _stub(messages, *, model, max_tokens, provider="openrouter", max_retries=3):  # noqa: ARG001
            return json.dumps({"verdict": "disagreed", "rationale": "different axes"})

        monkeypatch.setattr(judge_module, "call_llm", _stub)

        judge = JudgeHypothesis(
            hypothesis_text="tracks X",
            confidence=0.5,
            supporting_spans=[],
            framing="default",
            model="m",
            raw_response="{}",
        )
        result = reconcile_hypotheses(
            judge=judge,
            provided=Significance(hypothesis_text="tracks Y"),
            framings=["default", "broad", "contrastive"],
            model="m",
            max_iterations=3,
        )
        assert result.verdict == "disagreed"
        assert len(result.iterations) == 3
        framings_used = [it.framing for it in result.iterations]
        assert framings_used == ["default", "broad", "contrastive"]
