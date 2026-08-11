"""Tests for causalab.methods.first_tokens.get_first_tokens.

The trailing-space cases (#169 lineage) on a byte-level-BPE tokenizer (the
tiny-random GPT-2, ``add_prefix_space=False``): the bare and leading-space forms
of a word/digit map to *different* vocab ids, and :func:`get_first_tokens`
returns both. Also pins the ``"🍐word"`` fallback used when a tokenizer was not
built with ``add_prefix_space=False`` (bare form silently gets a leading space),
and asserts faithful borrowing from nnterp.
"""

from __future__ import annotations

import warnings

import pytest

from causalab.methods.first_tokens import get_first_tokens


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    from transformers import AutoTokenizer
    from tests._helpers.tiny import TINY_RANDOM_GPT2_MODEL_NAME

    try:
        return AutoTokenizer.from_pretrained(TINY_RANDOM_GPT2_MODEL_NAME)
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"Could not load tiny-random-gpt2 tokenizer: {exc}")


class TestGetFirstTokensNumerical:
    pytestmark = pytest.mark.numerical_unit

    def test_digit_trailing_space_case(self, gpt2_tokenizer):
        """'4' and ' 4' are different ids; both returned (the #169 gotcha)."""
        tok = gpt2_tokenizer
        ids = get_first_tokens("4", tok)
        assert ids == [20, 493]
        # Contract: first id is the bare form's, second the leading-space form's.
        assert ids[0] == tok.encode("4", add_special_tokens=False)[0]
        assert ids[1] == tok.encode(" 4", add_special_tokens=False)[0]

    def test_word_first_tokens(self, gpt2_tokenizer):
        assert get_first_tokens("blue", gpt2_tokenizer) == [66, 841]

    def test_list_input_dedup_and_order(self, gpt2_tokenizer):
        # Duplicates collapse; order is first-seen across words.
        assert get_first_tokens(["4", "4", "blue"], gpt2_tokenizer) == [
            20,
            493,
            66,
            841,
        ]

    def test_hacky_implementation_recovers_bare_token(self, gpt2_tokenizer):
        # The pear trick recovers exactly the bare first token.
        assert get_first_tokens("4", gpt2_tokenizer, use_hacky_implementation=True) == [
            20
        ]

    def test_matches_nnterp_reference(self, gpt2_tokenizer):
        """Borrowed logic stays faithful to nnterp.prompt_utils.get_first_tokens."""
        from nnterp.prompt_utils import get_first_tokens as nnterp_get_first_tokens

        for word in ["4", "7", "blue", "Paris", "Monday"]:
            assert get_first_tokens(word, gpt2_tokenizer) == nnterp_get_first_tokens(
                word, gpt2_tokenizer
            )

    def test_fallback_when_add_prefix_space_true(self):
        """A tokenizer built with add_prefix_space=True triggers the pear fallback.

        The bare form then silently carries a leading space (colliding with the
        space form), so the fast path can't distinguish them; the ``"🍐word"``
        fallback still recovers the true bare token.
        """
        from transformers import AutoTokenizer
        from tests._helpers.tiny import TINY_RANDOM_GPT2_MODEL_NAME

        try:
            tok = AutoTokenizer.from_pretrained(
                TINY_RANDOM_GPT2_MODEL_NAME, add_prefix_space=True
            )
        except Exception as exc:  # pragma: no cover - offline
            pytest.skip(f"Could not load tokenizer: {exc}")

        # Precondition: bare and space forms are indistinguishable here.
        assert (
            tok("4", add_special_tokens=False).input_ids[0]
            == tok(" 4", add_special_tokens=False).input_ids[0]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert get_first_tokens("4", tok) == [20]
            assert get_first_tokens("blue", tok) == [66]


class TestGetFirstTokensUnit:
    pytestmark = pytest.mark.unit

    def test_str_and_list_equivalent_for_single_word(self, gpt2_tokenizer):
        assert get_first_tokens("blue", gpt2_tokenizer) == get_first_tokens(
            ["blue"], gpt2_tokenizer
        )

    def test_returns_plain_int_ids(self, gpt2_tokenizer):
        ids = get_first_tokens("Paris", gpt2_tokenizer)
        assert ids and all(isinstance(i, int) for i in ids)
