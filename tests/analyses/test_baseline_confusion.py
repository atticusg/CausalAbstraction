"""Answer-space confusion grouping in ``baseline`` (issue #259).

``baseline`` used to group the confusion / ``ground_truth_dim0`` rows by the
localization *target* (``intervention_variable``). When the model's answer
(``raw_output``) depends on a variable *other* than the target — e.g.
``magnitude_comparison``, where the box letter flips under "min/less" polarity
— a single target row mixes both answers and averages to ~chance even at high
accuracy. These tests pin the detector that recognises that contamination
(resolving each answer through the score-token groups, like the scoring path)
and show that regrouping by the answer restores a diagonal confusion, all on
synthetic logits (no model / GPU).
"""

from __future__ import annotations

import pytest
import torch

from causalab.analyses.baseline.main import (
    _answer_class_index,
    _answer_rows_differ_from_target,
)
from causalab.methods.metric import compute_reference_distributions

pytestmark = pytest.mark.unit

# Token-id group per score label "A"/"B" — what ``get_output_token_ids`` builds
# from the variable's ``output_tokens`` declaration via the resolver.
SCORE_TOKEN_GROUPS = [[0], [1]]


class _StubTokenizer:
    """Maps the A/B answer forms to single token ids; everything else encodes to
    a 2-token sequence so it is filtered out as multi-token (mirrors the stub in
    tests/methods/test_metric.py). Lets the tests pin answer→class resolution
    without loading a real model."""

    _TABLE = {
        "A": [0],
        " A": [0],
        "a": [0],
        " a": [0],
        "B": [1],
        " B": [1],
        "b": [1],
        " b": [1],
    }

    def encode(self, text, add_special_tokens=False):
        return list(self._TABLE.get(text, [98, 99]))


TOKENIZER = _StubTokenizer()


class _FakeTask:
    """Minimal stand-in for a ``magnitude_comparison``-style task: the target
    variable is the latent ``magnitude_order``, distinct from the emitted
    answer letter under mixed polarity."""

    intervention_variable = "magnitude_order"
    intervention_values = ["A", "B"]

    def intervention_value_index(self, ex: dict) -> int:
        return self.intervention_values.index(ex["input"]["magnitude_order"])


def _ex(magnitude_order: str, answer: str) -> dict:
    return {
        "input": {
            "magnitude_order": magnitude_order,
            "raw_output": answer,
            "raw_input": "q",
        }
    }


# magnitude_order=A holds both a max-polarity (answer A) and a min-polarity
# (answer B) example, and likewise for B — so target rows mix answers.
MIXED = [_ex("A", "A"), _ex("A", "B"), _ex("B", "B"), _ex("B", "A")]
# Answer always equals the target (e.g. a single-polarity task).
COHERENT = [_ex("A", "A"), _ex("A", "A"), _ex("B", "B"), _ex("B", "B")]


def _logits_for(answer_idx: int, vocab: int = 3) -> list[torch.Tensor]:
    """One-step logits that put nearly all mass on ``answer_idx``'s token."""
    t = torch.full((vocab,), -10.0)
    t[answer_idx] = 10.0
    return [t]


def test_answer_class_index_resolves_via_token_groups() -> None:
    assert _answer_class_index(_ex("A", "A"), SCORE_TOKEN_GROUPS, TOKENIZER) == 0
    assert _answer_class_index(_ex("A", "B"), SCORE_TOKEN_GROUPS, TOKENIZER) == 1
    # multi-step tasks emit a per-step list; the first step is compared.
    assert (
        _answer_class_index(
            {"input": {"raw_output": ["B"]}}, SCORE_TOKEN_GROUPS, TOKENIZER
        )
        == 1
    )
    # an answer that tokenizes to nothing in the score space is unresolvable.
    assert (
        _answer_class_index(
            {"input": {"raw_output": "Z"}}, SCORE_TOKEN_GROUPS, TOKENIZER
        )
        is None
    )


def test_detects_mixed_polarity_contamination() -> None:
    task = _FakeTask()
    contaminated, answer_classes = _answer_rows_differ_from_target(
        MIXED, task, SCORE_TOKEN_GROUPS, TOKENIZER
    )
    assert contaminated is True
    assert answer_classes == [0, 1, 1, 0]


def test_coherent_task_is_not_contaminated() -> None:
    task = _FakeTask()
    contaminated, answer_classes = _answer_rows_differ_from_target(
        COHERENT, task, SCORE_TOKEN_GROUPS, TOKENIZER
    )
    assert contaminated is False
    assert answer_classes == [0, 0, 1, 1]


def test_unresolvable_answer_keeps_target_grouping() -> None:
    task = _FakeTask()
    dataset = [_ex("A", "A"), {"input": {"magnitude_order": "B", "raw_output": "Z"}}]
    contaminated, answer_classes = _answer_rows_differ_from_target(
        dataset, task, SCORE_TOKEN_GROUPS, TOKENIZER
    )
    # Can't relabel rows when an answer falls outside the score space.
    assert contaminated is False
    assert answer_classes is None


def test_answer_space_confusion_is_diagonal() -> None:
    """The crux of #259: grouping by the answer yields a diagonal confusion,
    whereas grouping by the target washes both rows to ~chance."""
    task = _FakeTask()
    score_token_ids = [0, 1]  # token id per score label "A"/"B"
    # Coherent model: each example puts its mass on the *answer* token.
    answer_idx = {"A": 0, "B": 1}
    output_logits = [_logits_for(answer_idx[ex["input"]["raw_output"]]) for ex in MIXED]

    _, answer_classes = _answer_rows_differ_from_target(
        MIXED, task, SCORE_TOKEN_GROUPS, TOKENIZER
    )
    assert answer_classes is not None
    answer_class_by_ex = {id(ex): ac for ex, ac in zip(MIXED, answer_classes)}

    answer_dists = compute_reference_distributions(
        dataset=MIXED,
        score_token_ids=score_token_ids,
        n_classes=2,
        example_to_class=lambda ex: answer_class_by_ex[id(ex)],
        output_logits=output_logits,
        score_token_index=0,
        full_vocab_softmax=True,
    )
    # Strongly diagonal: row A → token A, row B → token B.
    assert answer_dists[0, 0] > 0.9 and answer_dists[0, 1] < 0.1
    assert answer_dists[1, 1] > 0.9 and answer_dists[1, 0] < 0.1

    target_dists = compute_reference_distributions(
        dataset=MIXED,
        score_token_ids=score_token_ids,
        n_classes=len(task.intervention_values),
        example_to_class=task.intervention_value_index,
        output_logits=output_logits,
        score_token_index=0,
        full_vocab_softmax=True,
    )
    # The bug: target rows mix polarities and look like chance (~50/50).
    assert abs(target_dists[0, 0] - 0.5) < 0.1
    assert abs(target_dists[1, 0] - 0.5) < 0.1
