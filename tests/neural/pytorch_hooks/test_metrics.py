"""Metric lowering (spec §2.10) on hand-built logits — exact formulas."""

from __future__ import annotations

import math

import pytest
import torch

from causalab.neural.pytorch_hooks.metrics import (
    column_token_id,
    column_token_ids,
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
# §2.10 token_form — the answer-tokenization knob
#
# The bug this guards: `auto` returns the FIRST single-token candidate and tries
# the space-prefixed form first, so a punctuation answer resolves to a row the
# model never emits. Under gpt2 "?" is token 30 and " ?" is token 5633 — both
# single tokens — so a `match` metric on a punctuation answer read a flat 0.000
# at all 48 layers of a real gpt2-xl scan with no error raised anywhere.
#
# These use the real gpt2 tokenizer, not tiny-random-gpt2: the tiny stub's
# 1000-token vocabulary has no " ?" row, so it cannot express the ambiguity.
# The IOI suite already loads real gpt2 (tests/tasks/IOI/conftest.py).
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


def test_gpt2_punctuation_is_the_ambiguous_case(gpt2_tokenizer):
    """The premise, pinned: both forms are one token and they differ."""
    assert gpt2_tokenizer.encode("?", add_special_tokens=False) == [30]
    assert gpt2_tokenizer.encode(" ?", add_special_tokens=False) == [5633]


def test_auto_still_takes_the_space_prefixed_form(gpt2_tokenizer):
    """Backward compatibility: `auto` is exactly the historical resolver, so
    every document written before ``token_form`` existed is unchanged — this
    is also the CORRECT answer for the common case (an answer after a space)."""
    assert column_token_id(gpt2_tokenizer, "?") == 5633
    for word in (" Monday", "Monday", "Mary", " Mary"):
        spaced = gpt2_tokenizer.encode(" " + word.strip(), add_special_tokens=False)
        assert len(spaced) == 1
        assert column_token_id(gpt2_tokenizer, word) == spaced[0]


def test_bare_form_resolves_the_token_the_model_emits(gpt2_tokenizer):
    """The fix: a document that says `bare` gets the bare row."""
    assert column_token_id(gpt2_tokenizer, "?", token_form="bare") == 30
    assert column_token_id(gpt2_tokenizer, ".", token_form="bare") == 13
    # a leading space in the authored value is stripped, not honored
    assert column_token_id(gpt2_tokenizer, " ?", token_form="bare") == 30


def test_space_prefixed_form_is_pinnable(gpt2_tokenizer):
    assert column_token_id(gpt2_tokenizer, "?", token_form="space_prefixed") == 5633
    assert (
        column_token_id(gpt2_tokenizer, "Monday", token_form="space_prefixed")
        == (gpt2_tokenizer.encode(" Monday", add_special_tokens=False)[0])
    )


def test_a_pinned_form_refuses_rather_than_falling_back(tokenizer):
    """Pinning a form means it: sentencepiece makes " Monday" two pieces, and
    `space_prefixed` must refuse instead of quietly using the bare piece."""
    with pytest.raises(ProtocolError):
        column_token_id(tokenizer, "Monday", token_form="space_prefixed")


def test_match_scores_a_punctuation_answer_only_under_bare(gpt2_tokenizer):
    """The end-to-end regression, at the metric level: the model emits "?"
    (token 30); `auto` scores token 5633 and reads 0.0, `bare` reads 1.0."""
    logits = torch.zeros(1, 1, gpt2_tokenizer.vocab_size)
    logits[0, 0, 30] = 4.0  # what the model actually emits

    metric = MetricSpec(kind="match", of="logits", fields={"expected": "ans"})
    assert compute_metric(metric, logits, [{"ans": "?"}], gpt2_tokenizer) == [0.0]

    fixed = MetricSpec(
        kind="match", of="logits", fields={"expected": "ans"}, token_form="bare"
    )
    assert compute_metric(fixed, logits, [{"ans": "?"}], gpt2_tokenizer) == [1.0]


def test_auto_warns_once_per_column_when_the_forms_disagree(gpt2_tokenizer):
    """`auto` stays the default, but it no longer guesses in silence."""
    with pytest.warns(UserWarning, match="ambiguous under this tokenizer"):
        column_token_ids(gpt2_tokenizer, ["?", "?", ".", "!"])


def test_auto_is_silent_when_there_is_nothing_to_disambiguate(tokenizer, recwarn):
    """Sentencepiece " Monday" is two pieces, so only the bare form resolves —
    no choice was made and no warning is owed."""
    column_token_ids(tokenizer, [" Monday", "Monday"])
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


def test_a_pinned_form_never_warns(gpt2_tokenizer, recwarn):
    column_token_ids(gpt2_tokenizer, ["?", "."], token_form="bare")
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []
