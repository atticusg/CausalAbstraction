"""Metric lowering (spec §2.10) on hand-built logits — exact formulas."""

from __future__ import annotations

import math

import pytest
import torch

from causalab.neural.pytorch_hooks.metrics import (
    column_first_token_id,
    column_token_id,
    compute_metric,
)
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import MetricSpec

from tests.neural.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.numerical_unit


@pytest.fixture(scope="module")
def tokenizer():
    from causalab.neural.pytorch_hooks.loading import load_model

    return load_model(TINY_LLAMA).tokenizer


def _logits(tokenizer, favored: str, disfavored: str) -> torch.Tensor:
    logits = torch.zeros(1, 1, 32000)
    logits[0, 0, column_token_id(tokenizer, favored)] = 4.0
    logits[0, 0, column_token_id(tokenizer, disfavored)] = 1.0
    return logits


def test_space_prefixed_first_resolution(tokenizer):
    # sentencepiece: " Monday" is two pieces, "Monday" is the ▁Monday piece
    assert column_token_id(tokenizer, " Monday") == column_token_id(tokenizer, "Monday")
    with pytest.raises(ProtocolError):
        column_token_id(tokenizer, " Tuesday")  # 3 pieces either way — refuse


def test_logit_diff(tokenizer):
    metric = MetricSpec(kind="logit_diff", of="logits", fields={"a": "x", "b": "y"})
    values = compute_metric(
        metric,
        _logits(tokenizer, " Monday", " Friday"),
        [{"x": " Monday", "y": " Friday"}],
        tokenizer,
    )
    assert values == [pytest.approx(3.0)]


def test_cross_entropy(tokenizer):
    metric = MetricSpec(kind="cross_entropy", of="logits", fields={"target": "label"})
    logits = _logits(tokenizer, " Monday", " Friday")
    values = compute_metric(metric, logits, [{"label": " Monday"}], tokenizer)
    want = -torch.log_softmax(logits[0, 0].float(), dim=-1)[
        column_token_id(tokenizer, " Monday")
    ]
    assert values == [pytest.approx(float(want))]


def test_match_is_an_argmax_indicator(tokenizer):
    metric = MetricSpec(kind="match", of="logits", fields={"expected": "ans"})
    logits = _logits(tokenizer, " Monday", " Friday")
    assert compute_metric(metric, logits, [{"ans": " Monday"}], tokenizer) == [1.0]
    assert compute_metric(metric, logits, [{"ans": " Friday"}], tokenizer) == [0.0]


def test_kl_of_identical_distributions_is_zero(tokenizer):
    metric = MetricSpec(kind="kl", of="p", fields={"target": "q"})
    logits = _logits(tokenizer, " Monday", " Friday")
    values = compute_metric(
        metric, logits, [{}], tokenizer, target_value=logits.clone()
    )
    assert values == [pytest.approx(0.0, abs=1e-6)]


def test_top_k_orders_by_probability(tokenizer):
    metric = MetricSpec(kind="top_k", of="logits", fields={"k": 2})
    values = compute_metric(
        metric, _logits(tokenizer, " Monday", " Friday"), [{}], tokenizer
    )
    (entry,) = values
    assert entry["tokens"][0].strip() == "Monday"
    assert entry["probs"][0] > entry["probs"][1]


def test_class_probs_sums_group_members(tokenizer):
    metric = MetricSpec(
        kind="class_probs",
        of="logits",
        fields={"groups": {"days": [" Monday", " Friday"], "other": [" Sunday"]}},
    )
    logits = _logits(tokenizer, " Monday", " Friday")
    (entry,) = compute_metric(metric, logits, [{}], tokenizer)
    probs = torch.softmax(logits[0, 0].float(), dim=-1)
    want = float(
        probs[column_token_id(tokenizer, " Monday")]
        + probs[column_token_id(tokenizer, " Friday")]
    )
    assert entry["days"] == pytest.approx(want)
    assert math.isclose(
        entry["other"],
        float(probs[column_token_id(tokenizer, " Sunday")]),
        rel_tol=1e-6,
    )


# --------------------------------------------------------------------------- #
#  match: answer-form groups and first-token grading (§2.10)                   #
# --------------------------------------------------------------------------- #


def test_match_accepts_any_form_in_a_group(tokenizer):
    """A list-valued expected column is a group of equivalent surface forms:
    the argmax matching any member scores 1.0 (the synonym channel — 'US' /
    'USA' / 'United States' — with the group serialized by the task)."""
    metric = MetricSpec(kind="match", of="logits", fields={"expected": "forms"})
    logits = _logits(tokenizer, " Monday", " Friday")
    assert compute_metric(
        metric, logits, [{"forms": [" Sunday", " Monday"]}], tokenizer
    ) == [1.0]
    assert compute_metric(
        metric, logits, [{"forms": [" Sunday", " Friday"]}], tokenizer
    ) == [0.0]


def test_match_scalar_column_is_a_group_of_one(tokenizer):
    """The pre-existing spelling keeps its meaning — a scalar column is a
    one-member group, so no existing document changes behaviour."""
    grouped = MetricSpec(kind="match", of="logits", fields={"expected": "ans"})
    logits = _logits(tokenizer, " Monday", " Friday")
    assert compute_metric(grouped, logits, [{"ans": " Monday"}], tokenizer) == [1.0]


def test_match_empty_group_refuses(tokenizer):
    metric = MetricSpec(kind="match", of="logits", fields={"expected": "forms"})
    with pytest.raises(ProtocolError):
        compute_metric(
            metric, _logits(tokenizer, " Monday", " Friday"), [{"forms": []}], tokenizer
        )


def test_first_token_mode_credits_a_multi_token_answer(tokenizer):
    """``exact`` refuses " Thursday" (3 sentencepiece pieces either spelling);
    ``first_token`` credits its first *content* piece — what the model emits
    in context, and what the retired string-prefix grading meant."""
    exact = MetricSpec(kind="match", of="logits", fields={"expected": "ans"})
    first = MetricSpec(
        kind="match", of="logits", fields={"expected": "ans", "mode": "first_token"}
    )
    thursday_first = tokenizer.encode("Thursday", add_special_tokens=False)[0]
    logits = torch.zeros(1, 1, 32000)
    logits[0, 0, thursday_first] = 4.0

    with pytest.raises(ProtocolError):
        compute_metric(exact, logits, [{"ans": " Thursday"}], tokenizer)
    assert compute_metric(first, logits, [{"ans": " Thursday"}], tokenizer) == [1.0]


def test_first_token_skips_the_lone_space_piece(tokenizer):
    """The sentencepiece trap: " Thursday" encodes as ▁ + Th + urs + day, so
    crediting the *first* id would credit the bare-space piece — which every
    space-prefixed answer shares, making every answer match."""
    space_piece = tokenizer.encode(" Thursday", add_special_tokens=False)[0]
    assert tokenizer.decode([space_piece]).strip() == ""
    assert column_first_token_id(tokenizer, " Thursday") != space_piece
    assert (
        column_first_token_id(tokenizer, " Thursday")
        == tokenizer.encode("Thursday", add_special_tokens=False)[0]
    )


def test_first_token_agrees_with_exact_on_single_token_answers(tokenizer):
    """``first_token`` is a strict generalization: where ``exact`` resolves, it
    resolves to the same id."""
    for value in (" Monday", " Friday", " Sunday", "Monday"):
        assert column_first_token_id(tokenizer, value) == column_token_id(
            tokenizer, value
        )
