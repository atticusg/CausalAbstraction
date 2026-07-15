"""``top_logits.json`` example-record contract in ``baseline`` (issue #269, I8).

The per-example record used to store the ground-truth label under a field named
``raw_output``, which read as if it were the model's output — the actual
prediction is the top-1 decoded token. These tests pin the renamed/clarified
contract: the label is ``expected_output``, the model's top-1 token is
``prediction`` (== ``top_tokens[0]``), and ``correct`` compares the two. All
synthetic — no model / GPU.
"""

from __future__ import annotations

import pytest

from causalab.analyses.baseline.main import _top_logits_example

pytestmark = pytest.mark.unit

# A plausible top_tokens list as built in main(): the top-1 token is the
# model's prediction.
TOP_TOKENS = [
    {"token": "B", "token_id": 1, "prob": 0.7},
    {"token": "A", "token_id": 0, "prob": 0.3},
]


def test_field_names_separate_label_from_prediction():
    """The label is ``expected_output``; the model's top-1 token is ``prediction``.

    Guards against the old misleading ``raw_output`` field (== the label) being
    mistaken for the model's output.
    """
    rec = _top_logits_example("A", "B", TOP_TOKENS)
    assert set(rec) == {"expected_output", "prediction", "top_tokens", "correct"}
    assert "raw_output" not in rec
    assert rec["expected_output"] == "A"
    assert rec["prediction"] == "B"
    # ``prediction`` mirrors the top-1 decoded token.
    assert rec["prediction"] == rec["top_tokens"][0]["token"]


def test_correct_compares_prediction_to_label():
    assert _top_logits_example("A", "A", TOP_TOKENS)["correct"] is True
    assert _top_logits_example("A", "B", TOP_TOKENS)["correct"] is False


def test_correct_is_whitespace_insensitive():
    assert _top_logits_example("A", " A", TOP_TOKENS)["correct"] is True


def test_list_label_compares_against_first_step():
    """Multi-step generation labels are lists; the model emits one token here, so
    ``correct`` compares against the first expected step while ``expected_output``
    preserves the full list."""
    rec = _top_logits_example(["A", "B", "C"], "A", TOP_TOKENS)
    assert rec["expected_output"] == ["A", "B", "C"]
    assert rec["correct"] is True
    assert _top_logits_example(["B", "A"], "A", TOP_TOKENS)["correct"] is False
