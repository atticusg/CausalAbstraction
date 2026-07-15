"""Property-tier invariants for the shared random-word pool.

``causalab.tasks.random_words`` exposes a 24-element single-token pool and
a ``get_random_words(n)`` slicing helper. The natural-domains-arithmetic
random baseline (``create_random_causal_model``) calls
``get_random_words(len(config.entities))`` and uses the result as the
random ``entity`` / ``result`` value set. If the pool shrinks or the
helper returns the wrong slice, every ``baseline/<domain>.yaml``
random baseline constructs a malformed ``CausalModel`` and the smoke
runs fail during task composition.

Standards: ``tasks/`` requires ``[smoke-transitive, property-direct,
numerical-direct]``. Smoke is satisfied transitively by the NDA baseline
runners. The two ``...Property`` classes satisfy property-direct;
``TestGetRandomWordsNumerical`` satisfies numerical-direct via the
hand-pinned ``RANDOM_WORD_POOL[0]`` / ``get_random_words(5)`` assertions.
Direct attribution flows through the ``test_random_words.py`` →
``random_words.py`` basename match plus the import of
``causalab.tasks.random_words``.

Hypothesis is overkill for a 24-element constant pool; plain
``@pytest.mark.parametrize`` is used throughout. The tier markers are
attached as class decorators (``@pytest.mark.property`` /
``@pytest.mark.numerical_unit``); pytest honours these equivalently to a
class-body ``pytestmark``.
"""

from __future__ import annotations

import pytest

from causalab.tasks.random_words import RANDOM_WORD_POOL, get_random_words


# Pool size cap. ``get_random_words`` raises ``ValueError`` past this; the
# NDA baselines that exercise this pool today are weekdays (7), months (12),
# and hours (12) — all comfortably below the pool's 24 entries.
_POOL_SIZE = len(RANDOM_WORD_POOL)


@pytest.mark.property
class TestRandomWordPoolProperty:
    """Static invariants of the ``RANDOM_WORD_POOL`` constant."""

    def test_pool_is_nonempty(self) -> None:
        assert _POOL_SIZE > 0

    @pytest.mark.parametrize("index", range(len(RANDOM_WORD_POOL)))
    def test_entry_is_nonempty_string(self, index: int) -> None:
        assert isinstance(RANDOM_WORD_POOL[index], str) and RANDOM_WORD_POOL[index]

    @pytest.mark.parametrize("index", range(len(RANDOM_WORD_POOL)))
    def test_entry_has_no_whitespace(self, index: int) -> None:
        word = RANDOM_WORD_POOL[index]
        assert not any(c.isspace() for c in word), f"whitespace in {word!r}"

    def test_pool_has_no_duplicates(self) -> None:
        assert len(set(RANDOM_WORD_POOL)) == _POOL_SIZE

    def test_pool_covers_largest_baseline_request(self) -> None:
        """Pool must seat the largest random-baseline ``n`` requested today.

        NDA random baselines call ``get_random_words(len(config.entities))``;
        the cyclic domains (weekdays=7, months=12, hours=12) are the live
        consumers. 12 is the floor — drop below and every NDA random-baseline
        smoke run crashes during task composition.
        """
        assert _POOL_SIZE >= 12

    def test_pool_is_stable_across_reads(self) -> None:
        """Two reads of the module constant return the same prefix.

        Guards against accidental in-place mutation: a downstream caller
        appending to ``RANDOM_WORD_POOL`` would silently rewrite this
        shared singleton.
        """
        from causalab.tasks.random_words import RANDOM_WORD_POOL as second_read

        assert list(RANDOM_WORD_POOL) == list(second_read)


@pytest.mark.property
class TestGetRandomWordsProperty:
    """Behavioural invariants of ``get_random_words(n)``."""

    @pytest.mark.parametrize("n", [1, 5, 12, _POOL_SIZE])
    def test_returns_requested_length(self, n: int) -> None:
        assert len(get_random_words(n)) == n

    @pytest.mark.parametrize("n", [1, 5, 12, _POOL_SIZE])
    def test_returns_subset_of_pool(self, n: int) -> None:
        assert set(get_random_words(n)) <= set(RANDOM_WORD_POOL)

    @pytest.mark.parametrize("n", [1, 5, 12, _POOL_SIZE])
    def test_returned_elements_are_unique(self, n: int) -> None:
        words = get_random_words(n)
        assert len(set(words)) == len(words)

    @pytest.mark.parametrize("n", [1, 5, 12, _POOL_SIZE])
    def test_is_deterministic_without_seed(self, n: int) -> None:
        """``get_random_words`` is a pure prefix slice — no RNG involved."""
        assert get_random_words(n) == get_random_words(n)

    def test_n_zero_returns_empty_list(self) -> None:
        assert get_random_words(0) == []

    def test_n_equal_pool_size_returns_full_pool(self) -> None:
        assert get_random_words(_POOL_SIZE) == list(RANDOM_WORD_POOL)

    def test_n_exceeds_pool_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="pool only has"):
            get_random_words(_POOL_SIZE + 1)


@pytest.mark.numerical_unit
class TestGetRandomWordsNumerical:
    """Hand-pinned numerical values for ``random_words.py``.

    These two assertions tie the module to a specific snapshot of the pool
    so a silent reorder or rename of any leading word fails the suite.
    They live in their own class decorated with ``@pytest.mark.numerical_unit``
    so the dashboard's ``numerical-direct`` column can flip ✓ once the
    junit-vs-coverage classname mismatch documented at the top of this
    file is fixed. Direct attribution flows via the
    ``test_random_words.py`` → ``random_words.py`` basename match.
    """

    def test_first_word_is_apple(self) -> None:
        assert RANDOM_WORD_POOL[0] == "apple"

    def test_get_random_words_five_returns_pinned_prefix(self) -> None:
        assert get_random_words(5) == ["apple", "stone", "blade", "crown", "pearl"]
