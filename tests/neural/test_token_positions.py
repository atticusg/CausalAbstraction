"""Tests for ``causalab.neural.token_positions``.

This module resolves *which token positions in a prompt* an interchange /
locate / attention-pattern intervention should read from or write to. The
:class:`TokenPosition` class (a :class:`ComponentIndexer` subclass), the
declarative spec system (:class:`Template`,
:func:`build_token_position_factories`), and the combinators
(:func:`paired_token_position`, :func:`combined_token_position`) feed every
``AtomicModelUnit.index_component`` call routed through
``causalab.methods.interchange.single_pair``, ``causalab.analyses.locate.main``,
``causalab.methods.attention_pattern_analysis``,
``causalab.neural.activations.targets``, and the per-task wrappers in
``causalab/tasks/*/token_positions.py``. If indices come back wrong, every
downstream intervention reads/writes the wrong residual-stream rows and the
IIA / locate / attention-pattern artifacts are silently corrupted.

All LM-touching tests use the tiny-real Llama stub via the module-scoped
``tiny_pipeline`` fixture defined in ``tests/neural/conftest.py`` — mocking
the tokenizer here would silently freeze the offset_mapping contract that
the variable / scoped / relative factories all depend on.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import (
    PromptTemplateMismatchError,
    Template,
    TokenPosition,
    _assert_raw_input_matches,
    _first_difference,
    build_token_position_factories,
    build_token_positions,
    combined_token_position,
    get_all_tokens,
    get_last_token_index,
    get_list_of_each_token,
    get_substring_token_ids,
    get_tokens_in_char_range,
    paired_token_position,
)
from tests._helpers.tiny import tiny_random_model


# --------------------------------------------------------------------------- #
#  Shared helpers                                                             #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clear_template_cache() -> None:
    """Reset the class-level ``Template._position_cache`` before each test.

    The cache is keyed on ``(tokenizer.name_or_path, use_chat_template,
    full_text_str)`` and shared across all ``Template`` instances. Without a
    per-test clear, property tests that re-tokenize the same text silently
    observe cache hits rather than fresh tokenization, which lets
    offset-semantics drift through unnoticed.
    """
    Template._position_cache.clear()


def _trace(text: str) -> CausalTrace:
    """Build a minimal :class:`CausalTrace` carrying a single ``raw_input``.

    Many indexers in this module call ``pipeline.load([trace])`` which
    expects a list of CausalTrace-like objects, each subscriptable by
    variable name. Using the real ``CausalTrace`` keeps the contract
    aligned with the production interchange/locate paths.
    """
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _input_sample(text: str, **vars: Any) -> CausalTrace:
    """Build a ``CausalTrace`` carrying ``raw_input`` plus template vars.

    Template-driven factories index into ``input_sample[var_name]`` to read
    the value for each template variable, so they need a trace whose
    ``inputs`` include both ``raw_input`` and every templated variable.
    """
    mechanisms: dict[str, Mechanism] = {
        "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"]),
    }
    for name in vars:
        mechanisms[name] = Mechanism(parents=[], compute=lambda t, n=name: t[n])

    inputs: dict[str, Any] = {"raw_input": text, **vars}
    return CausalTrace(mechanisms=mechanisms, inputs=inputs)


# --------------------------------------------------------------------------- #
#  TokenPosition                                                              #
# --------------------------------------------------------------------------- #
class TestTokenPositionUnit:
    """:class:`TokenPosition` wraps an indexer + pipeline; constructor stores
    ``pipeline`` and :meth:`highlight_selected_token` re-renders the prompt with
    ``**bold**`` markers at the indexed positions."""

    pytestmark = pytest.mark.unit

    def test_stores_pipeline_and_id(self, tiny_pipeline) -> None:
        tp = TokenPosition(lambda x: [0], tiny_pipeline, id="my_id")
        assert tp.pipeline is tiny_pipeline
        assert tp.id == "my_id"

    def test_flag_aware_indexer_dispatch(self, tiny_pipeline) -> None:
        """An indexer that accepts ``is_original`` receives the flag value."""

        def smart(input, is_original=True):
            return [0] if is_original else [1]

        tp = TokenPosition(smart, tiny_pipeline, id="smart")
        assert tp.index({"any": "thing"}, is_original=True) == [0]
        assert tp.index({"any": "thing"}, is_original=False) == [1]

    def test_legacy_indexer_ignores_flag(self, tiny_pipeline) -> None:
        """An old-style indexer without ``is_original`` still gets called."""

        def old(input):
            return [3]

        tp = TokenPosition(old, tiny_pipeline, id="old")
        assert tp.index({"x": 1}, is_original=True) == [3]
        assert tp.index({"x": 1}, is_original=False) == [3]
        assert tp.index({"x": 1}) == [3]

    def test_highlight_selected_token_wraps_indexed_position(
        self, tiny_pipeline
    ) -> None:
        """``highlight_selected_token`` wraps each selected token in
        ``**...**`` and leaves the rest of the prompt verbatim (modulo
        whitespace re-introduction by SentencePiece detokenisation)."""
        trace = _trace("Hello world")
        # Tokenize so we know how many positions exist.
        ids = tiny_pipeline.load([trace])["input_ids"][0]
        # Pick a non-pad position to highlight.
        pad_id = tiny_pipeline.tokenizer.pad_token_id
        target = next(i for i, t in enumerate(ids) if t.item() != pad_id)
        tp = TokenPosition(lambda inp, pos=target: [pos], tiny_pipeline, id="single")
        rendered = tp.highlight_selected_token(trace)
        assert "**" in rendered
        # The bolded token round-trips through tokenizer.decode.
        expected_chunk = tiny_pipeline.tokenizer.decode(ids[target])
        assert f"**{expected_chunk}**" in rendered

    def test_highlight_selected_token_skips_pad(self, tiny_pipeline) -> None:
        """Pad tokens are dropped from the rendered string entirely (no bold,
        no plain reproduction)."""
        # Force a short prompt and rely on left-padding behaviour. The
        # pipeline calls the tokenizer with no padding for a single input,
        # so there are no pad ids here — instead verify the contract by
        # asking ``highlight_selected_token`` to highlight every index
        # (selected) and confirming the result is just the (non-pad)
        # detokenisation of the prompt.
        trace = _trace("Hello")
        ids = tiny_pipeline.load([trace])["input_ids"][0]
        n = len(ids)
        tp = TokenPosition(lambda inp, n=n: list(range(n)), tiny_pipeline, id="all")
        rendered = tp.highlight_selected_token(trace)
        # Every non-pad token is wrapped.
        pad_id = tiny_pipeline.tokenizer.pad_token_id
        non_pad_count = sum(1 for t in ids if t.item() != pad_id)
        assert rendered.count("**") == 2 * non_pad_count


class TestTokenPositionProperty:
    """Invariants of :class:`TokenPosition`: highlighting only mutates the
    positions selected by the indexer, and flag-aware dispatch is consistent
    across repeated calls (subsumes the ``test_is_original_flag.py`` cases
    that targeted ``TokenPosition`` specifically)."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("is_original_value", [True, False])
    def test_flag_value_round_trips(
        self, tiny_pipeline, is_original_value: bool
    ) -> None:
        """For any ``is_original`` value, a flag-aware indexer's branch is
        consistently invoked (no fallback to the wrong branch)."""

        def branchy(input, is_original=True):
            return [0] if is_original else [99]

        tp = TokenPosition(branchy, tiny_pipeline, id="branchy")
        for _ in range(5):
            result = tp.index({}, is_original=is_original_value)
            expected = [0] if is_original_value else [99]
            assert result == expected

    def test_highlight_is_identity_when_no_positions_selected(
        self, tiny_pipeline
    ) -> None:
        """Highlighting an empty set of positions yields a plain
        detokenisation with no ``**`` markers."""
        trace = _trace("Hello world")
        tp = TokenPosition(lambda inp: [], tiny_pipeline, id="empty")
        rendered = tp.highlight_selected_token(trace)
        assert "**" not in rendered


# --------------------------------------------------------------------------- #
#  get_last_token_index                                                       #
# --------------------------------------------------------------------------- #
class TestGetLastTokenIndexUnit:
    """Returns a single-element list ``[len(ids) - 1]``; used by
    ``causalab/tasks/*/token_positions.py`` wrappers to point an
    intervention at the final residual-stream slot."""

    pytestmark = pytest.mark.unit

    def test_returns_one_element_list_with_last_index(self, tiny_pipeline) -> None:
        trace = _trace("Hello world")
        ids = tiny_pipeline.load([trace])["input_ids"][0]
        result = get_last_token_index(trace, tiny_pipeline)
        assert result == [len(ids) - 1]


# --------------------------------------------------------------------------- #
#  get_all_tokens                                                             #
# --------------------------------------------------------------------------- #
class TestGetAllTokensUnit:
    """Returns a :class:`TokenPosition` whose indexer enumerates all (or all
    non-pad) sequence positions; used wherever an intervention needs to
    target the entire prompt (e.g. cumulative attention-pattern scans)."""

    pytestmark = pytest.mark.unit

    def test_returns_token_position_with_canonical_id(self, tiny_pipeline) -> None:
        trace = _trace("Hello world")
        tp = get_all_tokens(trace, tiny_pipeline)
        assert isinstance(tp, TokenPosition)
        assert tp.id == "all_tokens"

    def test_default_skips_pad_tokens(self, tiny_pipeline) -> None:
        """With ``padding=False`` (default), pad ids are filtered out."""
        trace = _trace("Hello world")
        tp = get_all_tokens(trace, tiny_pipeline)
        ids = tiny_pipeline.load([trace])["input_ids"][0]
        pad_id = tiny_pipeline.tokenizer.pad_token_id
        expected = [i for i, t in enumerate(ids) if t.item() != pad_id]
        assert tp.index(trace) == expected

    def test_padding_true_includes_all_positions(self, tiny_pipeline) -> None:
        """With ``padding=True`` every position is returned, pad ids
        included."""
        trace = _trace("Hello world")
        tp = get_all_tokens(trace, tiny_pipeline, padding=True)
        ids = tiny_pipeline.load([trace])["input_ids"][0]
        assert tp.index(trace) == list(range(len(ids)))


# --------------------------------------------------------------------------- #
#  get_list_of_each_token                                                     #
# --------------------------------------------------------------------------- #
class TestGetListOfEachTokenUnit:
    """Returns one :class:`TokenPosition` per non-pad token, each indexer
    returning a singleton ``[i]`` captured at definition time. Used by the
    locate analysis to sweep over individual positions."""

    pytestmark = pytest.mark.unit

    def test_accepts_string_input(self, tiny_pipeline) -> None:
        positions = get_list_of_each_token("Hello world", tiny_pipeline)
        assert len(positions) > 0
        assert all(isinstance(p, TokenPosition) for p in positions)

    def test_accepts_causaltrace_input(self, tiny_pipeline) -> None:
        trace = _trace("Hello world")
        positions = get_list_of_each_token(trace, tiny_pipeline)
        assert len(positions) > 0
        assert all(isinstance(p, TokenPosition) for p in positions)

    def test_one_position_per_non_pad_token(self, tiny_pipeline) -> None:
        text = "Hello world"
        positions = get_list_of_each_token(text, tiny_pipeline)
        trace = _trace(text)
        ids = tiny_pipeline.load([trace])["input_ids"][0]
        pad_id = tiny_pipeline.tokenizer.pad_token_id
        non_pad = sum(1 for t in ids if t.item() != pad_id)
        assert len(positions) == non_pad

    def test_each_indexer_returns_its_captured_singleton(self, tiny_pipeline) -> None:
        """Regression: indexer closures capture ``i`` via default-arg
        binding, not late binding — if a future refactor reintroduces the
        late-binding bug, every TokenPosition would resolve to the last
        index. Asserts each one returns *its own* singleton."""
        text = "Hello world"
        positions = get_list_of_each_token(text, tiny_pipeline)
        trace = _trace(text)
        ids = tiny_pipeline.load([trace])["input_ids"][0]
        pad_id = tiny_pipeline.tokenizer.pad_token_id
        non_pad_indices = [i for i, t in enumerate(ids) if t.item() != pad_id]
        for tp, expected_idx in zip(positions, non_pad_indices):
            # Indexer ignores its argument and returns the captured index.
            assert tp.index(trace) == [expected_idx]

    def test_token_position_ids_are_prefixed(self, tiny_pipeline) -> None:
        """Each returned :class:`TokenPosition` carries a ``tok_<i>_...``
        identifier so downstream artifacts can attribute a result back to a
        specific position."""
        positions = get_list_of_each_token("Hello world", tiny_pipeline)
        for tp in positions:
            assert tp.id.startswith("tok_")


# --------------------------------------------------------------------------- #
#  get_tokens_in_char_range                                                   #
# --------------------------------------------------------------------------- #
class TestGetTokensInCharRangeUnit:
    """Maps a character-level half-open range ``[start_char, end_char)`` onto
    the token indices whose offset spans intersect that range. The Template
    system relies on this to translate variable-substitution char ranges
    into the residual-stream rows that interventions should hit."""

    pytestmark = pytest.mark.unit

    @staticmethod
    def _offsets(*spans: tuple[int, int]) -> torch.Tensor:
        return torch.tensor(spans)

    def test_empty_range_when_end_le_start(self) -> None:
        offsets = self._offsets((0, 3), (3, 6))
        assert get_tokens_in_char_range(offsets, 3, 3) == []
        assert get_tokens_in_char_range(offsets, 5, 3) == []

    def test_pad_tokens_skipped(self) -> None:
        """A token whose offset is exactly ``(0, 0)`` is treated as
        padding and dropped, even if the range overlaps zero."""
        offsets = self._offsets((0, 0), (0, 3), (3, 6))
        assert get_tokens_in_char_range(offsets, 0, 6) == [1, 2]

    def test_half_open_semantics_at_token_boundary(self) -> None:
        """A token whose start equals ``end_char`` is excluded (half-open)."""
        offsets = self._offsets((0, 3), (3, 6), (6, 9))
        # [0, 3) overlaps only token 0; token 1 starts at end_char=3 → excluded.
        assert get_tokens_in_char_range(offsets, 0, 3) == [0]

    def test_partial_overlap_at_start(self) -> None:
        offsets = self._offsets((0, 3), (3, 6), (6, 9))
        # Range [2, 5) overlaps tokens 0 and 1.
        assert get_tokens_in_char_range(offsets, 2, 5) == [0, 1]

    def test_partial_overlap_at_end(self) -> None:
        offsets = self._offsets((0, 3), (3, 6), (6, 9))
        # Range [4, 8) overlaps tokens 1 and 2.
        assert get_tokens_in_char_range(offsets, 4, 8) == [1, 2]

    def test_full_range_returns_all_non_pad(self) -> None:
        offsets = self._offsets((0, 3), (3, 6), (6, 9))
        assert get_tokens_in_char_range(offsets, 0, 9) == [0, 1, 2]


class TestGetTokensInCharRangeProperty:
    """Invariants: returned indices are monotonically increasing and
    contiguous over the non-pad slice they cover, and translating both
    endpoints by the same shift translates the offsets too."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize(
        "spans, start, end",
        [
            ([(0, 3), (3, 6), (6, 9)], 0, 9),
            ([(0, 3), (3, 6), (6, 9)], 2, 5),
            ([(0, 3), (3, 6), (6, 9)], 4, 8),
            ([(0, 0), (0, 4), (4, 8), (8, 12)], 0, 12),
        ],
    )
    def test_indices_are_monotonic_and_contiguous(
        self,
        spans: list[tuple[int, int]],
        start: int,
        end: int,
    ) -> None:
        offsets = torch.tensor(spans)
        result = get_tokens_in_char_range(offsets, start, end)
        if not result:
            return
        assert result == sorted(result)
        assert result == list(range(result[0], result[-1] + 1))

    def test_shift_invariance(self) -> None:
        """Shifting both the range and the offsets by ``k`` yields the same
        set of indices — the function is translation-equivariant."""
        spans = [(0, 3), (3, 6), (6, 9)]
        offsets = torch.tensor(spans)
        start, end = 2, 7
        base = get_tokens_in_char_range(offsets, start, end)

        k = 100
        shifted_offsets = torch.tensor([(s + k, e + k) for s, e in spans])
        shifted = get_tokens_in_char_range(shifted_offsets, start + k, end + k)
        assert shifted == base


# --------------------------------------------------------------------------- #
#  get_substring_token_ids                                                    #
# --------------------------------------------------------------------------- #
#                                                                             #
#  Exercised on a real (tiny) tokenizer. Replaces the mock-tokenizer test    #
#  suite that previously lived in tests/neural/test_LM_units.py. docs/TESTS.md  #
#  is explicit: numerical code between two pieces we wrote ourselves must    #
#  run against a real tokenizer or the suite freezes into a lie when         #
#  offset semantics drift.                                                    #
# --------------------------------------------------------------------------- #
class TestGetSubstringTokenIdsUnit:
    """Behavioural cases for ``get_substring_token_ids`` on the tiny Llama
    tokenizer. The inputs cover single-token, multi-token, partial-overlap
    at one or both ends, punctuation, and newline boundaries — the variants
    that mock tokenizers historically got wrong."""

    pytestmark = pytest.mark.unit

    @pytest.mark.parametrize(
        "text, substring",
        [
            ("The quick brown fox", "quick"),
            ("The quick brown fox", "quick brown"),
            # "fox" straddles two tokens ("fo" + "x") under tiny Llama BPE.
            ("The quick brown fox", "fox"),
            ("Hello, world!", ", world!"),
            ("The cat sat", "c"),
            ("Hello\nWorld", "\nWorld"),
            # Substring spans three consecutive tokens (abc|def|g|hi).
            ("abcdefghi", "cdefg"),
            # Substring sits entirely inside a multi-character token.
            ("unbelievable", "liev"),
            ("The cat sat", "The cat sat"),
        ],
    )
    def test_selection_decodes_to_cover_substring(
        self, tiny_pipeline, text: str, substring: str
    ) -> None:
        ids = get_substring_token_ids(text, substring, tiny_pipeline)
        assert ids, "expected at least one token index"

        # ``get_substring_token_ids`` defaults to ``add_special_tokens=False``;
        # mirror that here so indices line up.
        token_ids = tiny_pipeline.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        decoded = tiny_pipeline.tokenizer.decode([token_ids[i].item() for i in ids])
        # SentencePiece may reshape leading whitespace; compare stripped.
        assert substring.strip().lower() in decoded.strip().lower()

    def test_empty_substring_raises(self, tiny_pipeline) -> None:
        with pytest.raises(ValueError, match="Substring cannot be empty"):
            get_substring_token_ids("hello", "", tiny_pipeline)

    def test_empty_text_raises(self, tiny_pipeline) -> None:
        with pytest.raises(ValueError, match="Text cannot be empty"):
            get_substring_token_ids("", "test", tiny_pipeline)

    def test_substring_not_found_raises(self, tiny_pipeline) -> None:
        with pytest.raises(ValueError, match="not found in text"):
            get_substring_token_ids("hello world", "xyz", tiny_pipeline)

    def test_substring_longer_than_text_raises(self, tiny_pipeline) -> None:
        with pytest.raises(ValueError, match="not found in text"):
            get_substring_token_ids("hi", "hi there", tiny_pipeline)

    def test_default_occurrence_is_first(self, tiny_pipeline) -> None:
        """Default ``occurrence=0`` picks the first match."""
        first = get_substring_token_ids("cat and cat", "cat", tiny_pipeline)
        assert first == [0]

    def test_explicit_occurrence_index(self, tiny_pipeline) -> None:
        """``occurrence=1`` selects the second match (later token range)."""
        second = get_substring_token_ids(
            "cat and cat", "cat", tiny_pipeline, occurrence=1
        )
        assert second != [0]
        assert min(second) > 0

    def test_negative_occurrence_aliases_last(self, tiny_pipeline) -> None:
        """With two matches, ``occurrence=-1`` agrees with ``occurrence=1``."""
        last = get_substring_token_ids(
            "cat and cat", "cat", tiny_pipeline, occurrence=-1
        )
        second = get_substring_token_ids(
            "cat and cat", "cat", tiny_pipeline, occurrence=1
        )
        assert last == second

    def test_occurrence_out_of_range_raises(self, tiny_pipeline) -> None:
        with pytest.raises(ValueError, match="out of range"):
            get_substring_token_ids("cat and cat", "cat", tiny_pipeline, occurrence=5)

    def test_strict_mode_raises_on_ambiguity(self, tiny_pipeline) -> None:
        with pytest.raises(ValueError, match="occurrences"):
            get_substring_token_ids("cat and cat", "cat", tiny_pipeline, strict=True)

    def test_strict_mode_passes_when_unique(self, tiny_pipeline) -> None:
        """``strict=True`` is a no-op when the substring is unique."""
        ids = get_substring_token_ids(
            "The quick brown fox", "quick", tiny_pipeline, strict=True
        )
        assert ids


class TestGetSubstringTokenIdsProperty:
    """Invariants of ``get_substring_token_ids``: any returned list must be
    a contiguous, monotonically increasing slice of in-range token indices,
    deterministic across repeated calls, and equal to the full token range
    when the substring is the whole text."""

    pytestmark = pytest.mark.property

    INPUTS: list[tuple[str, str]] = [
        ("The quick brown fox", "quick"),
        ("The quick brown fox", "quick brown"),
        ("The quick brown fox", "fox"),
        ("Hello, world!", ", world!"),
        ("abcdefghi", "cdefg"),
        ("unbelievable", "liev"),
        ("Hello\nWorld", "\nWorld"),
    ]

    @pytest.mark.parametrize("text, substring", INPUTS)
    def test_indices_are_monotonic_and_contiguous(
        self, tiny_pipeline, text: str, substring: str
    ) -> None:
        ids = get_substring_token_ids(text, substring, tiny_pipeline)
        assert ids == sorted(ids)
        assert ids == list(range(ids[0], ids[-1] + 1))

    @pytest.mark.parametrize("text, substring", INPUTS)
    def test_indices_are_in_range(
        self, tiny_pipeline, text: str, substring: str
    ) -> None:
        ids = get_substring_token_ids(text, substring, tiny_pipeline)
        n_tokens = len(
            tiny_pipeline.tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
        )
        assert 0 <= ids[0]
        assert ids[-1] < n_tokens

    @pytest.mark.parametrize("text, substring", INPUTS)
    def test_idempotent(self, tiny_pipeline, text: str, substring: str) -> None:
        a = get_substring_token_ids(text, substring, tiny_pipeline)
        b = get_substring_token_ids(text, substring, tiny_pipeline)
        assert a == b

    @pytest.mark.parametrize(
        "text",
        ["The quick brown fox", "Hello, world!", "abcdefghi"],
    )
    def test_full_text_returns_all_tokens(self, tiny_pipeline, text: str) -> None:
        ids = get_substring_token_ids(text, text, tiny_pipeline)
        n_tokens = len(
            tiny_pipeline.tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
        )
        assert ids == list(range(n_tokens))


# --------------------------------------------------------------------------- #
#  Template                                                                   #
# --------------------------------------------------------------------------- #
class TestTemplateUnit:
    """Parses ``"... {var} ..."`` template strings into literal/variable
    parts, fills them with values, and maps each variable to the token
    indices the value occupies in the tokenized output. Drives every
    ``"type": "variable"`` and ``"scope"`` / ``"relative_to"`` spec in the
    declarative system."""

    pytestmark = pytest.mark.unit

    def test_get_variable_names_returns_declared_variables(self) -> None:
        t = Template("The sum of {x} and {y} is ")
        names = t.get_variable_names()
        assert set(names) == {"x", "y"}

    def test_get_variable_names_dedupes_duplicates(self) -> None:
        """``get_variable_names`` is ``list(set(...))`` — order is
        non-deterministic but membership must be exact and dedup-d."""
        t = Template("{x} and {x} again")
        names = t.get_variable_names()
        assert set(names) == {"x"}
        assert len(names) == 1

    def test_get_variable_names_empty_template(self) -> None:
        t = Template("no variables here")
        assert t.get_variable_names() == []

    def test_fill_produces_concatenation(self) -> None:
        t = Template("The sum of {x} and {y} is {z}")
        out = t.fill({"x": 5, "y": 7, "z": 12})
        assert out == "The sum of 5 and 7 is 12"

    def test_fill_raises_on_missing_variable(self) -> None:
        t = Template("Hello {name}")
        with pytest.raises(ValueError, match="Missing value"):
            t.fill({})

    def test_get_variable_positions_assigns_tokens_to_vars(self, tiny_pipeline) -> None:
        t = Template("The sum of {x} and {y} is ")
        positions = t.get_variable_positions({"x": "5", "y": "7"}, tiny_pipeline)
        assert set(positions) == {"x", "y"}
        assert all(isinstance(v, list) for v in positions.values())
        assert all(positions[v] for v in positions)

    def test_get_variable_positions_no_variables(self, tiny_pipeline) -> None:
        """A template with no variables returns an empty position dict."""
        t = Template("just some literal text")
        positions = t.get_variable_positions({}, tiny_pipeline)
        assert positions == {}


class TestTemplateProperty:
    """Invariants: cache hits return identical objects; the decoded
    substrings at the returned token indices cover the substituted value;
    multiple occurrences of the same variable produce multiple disjoint
    index runs."""

    pytestmark = pytest.mark.property

    def test_decoded_indices_cover_value(self, tiny_pipeline) -> None:
        """The decoded slice at ``positions[var]`` contains the substituted
        value (modulo SentencePiece whitespace normalisation).

        Uses ``pipeline.load`` (not the bare tokenizer) so the index basis
        matches what :meth:`Template.get_variable_positions` computes —
        ``pipeline.load`` adds the BOS token, shifting offsets by +1.
        """
        t = Template("The sum of {x} and {y} is ")
        values = {"x": "5", "y": "7"}
        positions = t.get_variable_positions(values, tiny_pipeline)

        full_text = t.fill(values)
        trace = _trace(full_text)
        token_ids = tiny_pipeline.load([trace])["input_ids"][0]

        for var_name, indices in positions.items():
            assert indices, f"variable {var_name!r} got no token indices"
            decoded = tiny_pipeline.tokenizer.decode(
                [token_ids[i].item() for i in indices]
            )
            assert values[var_name].strip() in decoded.strip()

    def test_cache_hit_returns_identical_object(self, tiny_pipeline) -> None:
        """Second call with same ``(tokenizer, full_text)`` returns the same
        dict object cached by ``Template._position_cache``."""
        t = Template("Hello {name}")
        values = {"name": "world"}
        first = t.get_variable_positions(values, tiny_pipeline)
        second = t.get_variable_positions(values, tiny_pipeline)
        # Same dict object (cache hit) — not just equal.
        assert first is second

    def test_multiple_occurrences_produce_multiple_runs(self, tiny_pipeline) -> None:
        """A variable referenced twice in the template produces a token
        list covering both substitution sites."""
        t = Template("{x} plus {x}")
        positions = t.get_variable_positions({"x": "5"}, tiny_pipeline)
        x_indices = positions["x"]
        # Both occurrences contribute tokens — total span > one substitution.
        single_t = Template("{x}")
        single_positions = single_t.get_variable_positions({"x": "5"}, tiny_pipeline)
        # Two occurrences should produce strictly more indices than one.
        assert len(x_indices) > len(single_positions["x"])


class TestPositionCacheKeyUnit:
    """Regression pins for issue #429: ``Template._position_cache`` keys on the
    tokenizer's *stable identity* (``name_or_path``), not its object address
    (``id()``), and is bounded via LRU eviction.

    Keying on ``id()`` was unsound: Python recycles addresses after GC, so a
    long-lived process that frees and rebuilds pipelines could get a cache hit
    from a *different* tokenizer that happened to land at the same address —
    silently returning another tokenizer's token indices. The dict was also
    unbounded, growing for the whole process lifetime across sweeps.

    ``name_or_path`` is the cache's notion of tokenizer identity and is expected
    to be unique per tokenizer; a missing/empty value is not a usable key and is
    rejected loudly rather than silently aliasing distinct tokenizers.
    """

    pytestmark = pytest.mark.unit

    def test_distinct_tokenizer_identities_resolve_independently(
        self, tiny_pipeline, monkeypatch
    ) -> None:
        """Two tokenizer *identities* (different ``name_or_path``) get separate
        cache entries for the same text — neither serves the other's indices.

        The tokenizer *object* is held constant (only ``name_or_path`` changes),
        so under the old ``id()`` key both calls would collide on one entry;
        under the identity key the second lookup misses and resolves
        independently. Positions are equal here (same underlying tokenizer) but
        come from distinct cached dict objects.
        """
        t = Template("The sum of {x} and {y} is ")
        values = {"x": "5", "y": "7"}
        flag = tiny_pipeline.use_chat_template
        text = t.fill(values)

        monkeypatch.setattr(tiny_pipeline.tokenizer, "name_or_path", "identity/A")
        first = t.get_variable_positions(values, tiny_pipeline)
        monkeypatch.setattr(tiny_pipeline.tokenizer, "name_or_path", "identity/B")
        second = t.get_variable_positions(values, tiny_pipeline)

        # Separate cache entries (would alias to a single entry under id()).
        assert first is not second
        # ... but resolved consistently: same tokenizer behaviour, same indices.
        assert first == second
        # Both identities are cached under their own key.
        assert ("identity/A", flag, text) in Template._position_cache
        assert ("identity/B", flag, text) in Template._position_cache

    def test_same_identity_hits_across_distinct_objects(self, tiny_pipeline) -> None:
        """A cache hit is served across *distinct* pipeline/tokenizer objects
        that share a ``name_or_path`` — proving the key is identity, not
        ``id()``.

        The first (real) call populates the cache. The second goes through a
        lightweight stand-in whose tokenizer only exposes the same
        ``name_or_path`` and ``use_chat_template``: on a cache hit
        ``get_variable_positions`` returns before tokenizing, so the stand-in's
        (absent) tokenizer is never exercised. Under the old ``id()`` key this
        would miss (different object) and fall through to tokenization.
        """
        t = Template("Hello {name}")
        values = {"name": "world"}
        first = t.get_variable_positions(values, tiny_pipeline)

        stand_in = SimpleNamespace(
            tokenizer=SimpleNamespace(
                name_or_path=tiny_pipeline.tokenizer.name_or_path
            ),
            use_chat_template=tiny_pipeline.use_chat_template,
        )
        second = t.get_variable_positions(values, stand_in)
        # Distinct object, same identity → the first call's cached dict is served.
        assert second is first

    def test_cache_is_bounded_by_lru_eviction(self, tiny_pipeline, monkeypatch) -> None:
        """The cache never exceeds ``_POSITION_CACHE_MAXSIZE``; the oldest
        (least-recently-used) entries are evicted first."""
        monkeypatch.setattr(Template, "_POSITION_CACHE_MAXSIZE", 3)
        t = Template("value is {v}")
        name = tiny_pipeline.tokenizer.name_or_path
        flag = tiny_pipeline.use_chat_template

        # Insert more distinct texts than the cap allows.
        for i in range(6):
            t.get_variable_positions({"v": str(i)}, tiny_pipeline)

        assert len(Template._position_cache) == 3
        # The first three inserts were evicted; the most-recent three survive.
        for i in range(3):
            assert (name, flag, t.fill({"v": str(i)})) not in Template._position_cache
        for i in range(3, 6):
            assert (name, flag, t.fill({"v": str(i)})) in Template._position_cache

    @pytest.mark.parametrize("bad", [None, ""])
    def test_missing_name_or_path_fails_loudly(self, tiny_pipeline, bad) -> None:
        """A tokenizer whose ``name_or_path`` is missing/empty is not a usable
        cache identity. Rather than silently collapsing distinct tokenizers onto
        one key (``(None, ...)`` / ``("", ...)``) and serving the wrong indices,
        ``get_variable_positions`` raises an explicit error naming the bad value.
        """
        t = Template("Hello {name}")
        values = {"name": "world"}
        stand_in = SimpleNamespace(
            tokenizer=SimpleNamespace(name_or_path=bad),
            use_chat_template=tiny_pipeline.use_chat_template,
        )
        with pytest.raises(ValueError, match="name_or_path"):
            t.get_variable_positions(values, stand_in)
        # Nothing was cached under the bad key.
        assert not any(key[0] == bad for key in Template._position_cache)


# --------------------------------------------------------------------------- #
#  build_token_position_factories                                             #
# --------------------------------------------------------------------------- #
class TestBuildTokenPositionFactoriesUnit:
    """Compiles a declarative ``{name: spec}`` dict into a parallel ``{name:
    factory}`` dict where each factory accepts a pipeline and returns a
    :class:`TokenPosition`. Every static spec type (``index``, ``variable``,
    scoped, relative_to) plus the dynamic-callable path is exercised here."""

    pytestmark = pytest.mark.unit

    def test_returns_factory_per_name(self) -> None:
        template = "The sum of {x} and {y} is "
        specs: dict[str, Any] = {
            "last": {"type": "index", "position": -1},
            "first": {"type": "index", "position": 0},
            "x": {"type": "variable", "name": "x"},
        }
        factories = build_token_position_factories(specs, template)
        assert set(factories) == {"last", "first", "x"}
        assert all(callable(f) for f in factories.values())

    def test_absolute_first_token(self, tiny_pipeline) -> None:
        template = "The sum of {x} and {y} is "
        specs = {"first": {"type": "index", "position": 0}}
        factories = build_token_position_factories(specs, template)
        tp = factories["first"](tiny_pipeline)
        result = tp.index(_input_sample("The sum of 5 and 7 is ", x="5", y="7"))
        assert result == [0]

    def test_absolute_last_token(self, tiny_pipeline) -> None:
        template = "The sum of {x} and {y} is "
        specs = {"last": {"type": "index", "position": -1}}
        factories = build_token_position_factories(specs, template)
        tp = factories["last"](tiny_pipeline)
        sample = _input_sample("The sum of 5 and 7 is ", x="5", y="7")
        ids = tiny_pipeline.load([sample])["input_ids"][0]
        assert tp.index(sample) == [len(ids) - 1]

    def test_variable_factory_returns_value_tokens(self, tiny_pipeline) -> None:
        template = "The sum of {x} and {y} is "
        specs = {"x": {"type": "variable", "name": "x"}}
        factories = build_token_position_factories(specs, template)
        tp = factories["x"](tiny_pipeline)
        sample = _input_sample("The sum of 5 and 7 is ", x="5", y="7")
        result = tp.index(sample)
        assert isinstance(result, list)
        assert len(result) >= 1
        # Decoding the returned indices yields a span that contains "5".
        ids = tiny_pipeline.load([sample])["input_ids"][0]
        decoded = tiny_pipeline.tokenizer.decode([ids[i].item() for i in result])
        assert "5" in decoded

    def test_unknown_type_raises(self) -> None:
        template = "x={x}"
        specs = {"bad": {"type": "definitely_not_a_type"}}
        with pytest.raises(ValueError, match="Unknown token position type"):
            build_token_position_factories(specs, template)

    def test_variable_missing_name_raises(self) -> None:
        template = "x={x}"
        specs = {"bad": {"type": "variable"}}  # no 'name'
        with pytest.raises(ValueError, match="requires 'name' field"):
            build_token_position_factories(specs, template)

    def test_variable_not_in_template_raises(self) -> None:
        template = "x={x}"
        specs = {"z": {"type": "variable", "name": "z"}}
        with pytest.raises(ValueError, match="not found in template"):
            build_token_position_factories(specs, template)

    def test_scoped_missing_variable_raises(self) -> None:
        template = "x={x}"
        specs = {
            "bad": {
                "type": "index",
                "position": 0,
                "scope": {},  # missing 'variable'
            }
        }
        with pytest.raises(ValueError, match="scope must specify 'variable'"):
            build_token_position_factories(specs, template)

    def test_relative_to_missing_variable_raises(self) -> None:
        template = "x={x}"
        specs = {
            "bad": {
                "type": "index",
                "position": 1,
                "relative_to": {},  # missing 'variable'
            }
        }
        with pytest.raises(ValueError, match="relative_to must specify 'variable'"):
            build_token_position_factories(specs, template)

    def test_absolute_position_out_of_range_raises(self, tiny_pipeline) -> None:
        template = "x={x}"
        specs = {"oob": {"type": "index", "position": 9999}}
        factories = build_token_position_factories(specs, template)
        tp = factories["oob"](tiny_pipeline)
        with pytest.raises(ValueError, match="out of range"):
            tp.index(_input_sample("x=5", x="5"))

    def test_scoped_index_within_variable(self, tiny_pipeline) -> None:
        """``scope`` selects the n-th token of a variable's value."""
        template = "Answer: {answer}"
        specs = {
            "first_of_answer": {
                "type": "index",
                "position": 0,
                "scope": {"variable": "answer"},
            }
        }
        factories = build_token_position_factories(specs, template)
        tp = factories["first_of_answer"](tiny_pipeline)
        sample = _input_sample("Answer: alpha beta", answer="alpha beta")
        var_factories = build_token_position_factories(
            {"answer": {"type": "variable", "name": "answer"}}, template
        )
        all_answer_tokens = var_factories["answer"](tiny_pipeline).index(sample)
        assert tp.index(sample) == [all_answer_tokens[0]]

    def test_relative_to_after_variable(self, tiny_pipeline) -> None:
        """Positive offset measures from the *last* variable token."""
        template = "Answer: {answer}."
        specs = {
            "after": {
                "type": "index",
                "position": 1,
                "relative_to": {"variable": "answer"},
            }
        }
        factories = build_token_position_factories(specs, template)
        tp = factories["after"](tiny_pipeline)
        sample = _input_sample("Answer: yes.", answer="yes")
        var_factories = build_token_position_factories(
            {"answer": {"type": "variable", "name": "answer"}}, template
        )
        ref = var_factories["answer"](tiny_pipeline).index(sample)
        assert tp.index(sample) == [ref[-1] + 1]

    def test_dynamic_spec_receives_input_sample(self, tiny_pipeline) -> None:
        """A callable spec is invoked with the ``input_sample`` and the
        returned sub-spec is then resolved as if it had been declared
        statically."""
        template = "Pick: {a} or {b}"

        def spec_func(setting):
            return {
                "type": "variable",
                "name": "a" if setting["choose"] == "a" else "b",
            }

        factories = build_token_position_factories({"target": spec_func}, template)
        tp = factories["target"](tiny_pipeline)

        sample_a = _input_sample("Pick: yes or no", a="yes", b="no", choose="a")
        sample_b = _input_sample("Pick: yes or no", a="yes", b="no", choose="b")

        # The two branches resolve to disjoint index runs.
        idx_a = tp.index(sample_a)
        idx_b = tp.index(sample_b)
        assert idx_a != idx_b
        assert set(idx_a).isdisjoint(set(idx_b))


class TestBuildTokenPositionFactoriesProperty:
    """Structural invariants of the factory dict: parallel to the input
    spec dict in keys and length; every value is callable and produces a
    :class:`TokenPosition` with a matching ``id``."""

    pytestmark = pytest.mark.property

    def test_factories_dict_mirrors_specs_dict(self) -> None:
        template = "x={x} y={y}"
        specs: dict[str, Any] = {
            "last": {"type": "index", "position": -1},
            "first": {"type": "index", "position": 0},
            "x": {"type": "variable", "name": "x"},
            "y": {"type": "variable", "name": "y"},
        }
        factories = build_token_position_factories(specs, template)
        assert list(factories) == list(specs)
        assert len(factories) == len(specs)

    def test_each_factory_produces_token_position_with_matching_id(
        self, tiny_pipeline
    ) -> None:
        template = "x={x}"
        specs: dict[str, Any] = {
            "last": {"type": "index", "position": -1},
            "first": {"type": "index", "position": 0},
            "x": {"type": "variable", "name": "x"},
        }
        factories = build_token_position_factories(specs, template)
        for name, factory in factories.items():
            tp = factory(tiny_pipeline)
            assert isinstance(tp, TokenPosition)
            assert tp.id == name

    @pytest.mark.parametrize("offset", [1, 2, 3])
    def test_relative_offset_is_linear_after_variable(
        self, tiny_pipeline, offset: int
    ) -> None:
        """Positive ``relative_to.position`` offsets shift the returned
        index by exactly that offset from the last variable token."""
        # Use a template with enough trailing tokens that offsets up to 3
        # stay in-range under the tiny Llama tokenizer.
        template = "Answer: {answer} . . . . . end"
        text = "Answer: yes . . . . . end"
        specs = {
            "rel": {
                "type": "index",
                "position": offset,
                "relative_to": {"variable": "answer"},
            }
        }
        factories = build_token_position_factories(specs, template)
        tp = factories["rel"](tiny_pipeline)
        sample = _input_sample(text, answer="yes")
        var_factories = build_token_position_factories(
            {"answer": {"type": "variable", "name": "answer"}}, template
        )
        ref = var_factories["answer"](tiny_pipeline).index(sample)
        assert tp.index(sample) == [ref[-1] + offset]


# --------------------------------------------------------------------------- #
#  build_token_positions                                                      #
# --------------------------------------------------------------------------- #
class TestBuildTokenPositionsUnit:
    """Builds declarative specs and materializes them into :class:`TokenPosition`
    instances bound to a pipeline — the single call a task's
    ``create_token_positions`` should use. Guards issue #179: returning the
    un-materialized factories from :func:`build_token_position_factories` instead
    crashes ``locate`` because consumers (``build_targets_for_grid``,
    ``build_residual_stream_targets``) read ``.id`` off each value and a bare factory
    function has none."""

    pytestmark = pytest.mark.unit

    def test_returns_token_position_instances(self, tiny_pipeline) -> None:
        template = "The sum of {x} and {y} is "
        specs: dict[str, Any] = {
            "last": {"type": "index", "position": -1},
            "x": {"type": "variable", "name": "x"},
        }
        positions = build_token_positions(specs, template, tiny_pipeline)
        assert set(positions) == {"last", "x"}
        for name, tp in positions.items():
            # Materialized instance (carries ``.id``), NOT the bare factory function
            # a consumer would choke on — the #179 regression.
            assert isinstance(tp, TokenPosition)
            assert tp.id == name
            assert tp.pipeline is tiny_pipeline

    def test_materialized_positions_resolve_indices(self, tiny_pipeline) -> None:
        """The returned instances are usable directly: indexing resolves tokens."""
        template = "The sum of {x} and {y} is "
        specs = {"last": {"type": "index", "position": -1}}
        positions = build_token_positions(specs, template, tiny_pipeline)
        sample = _input_sample("The sum of 5 and 7 is ", x="5", y="7")
        ids = tiny_pipeline.load([sample])["input_ids"][0]
        assert positions["last"].index(sample) == [len(ids) - 1]

    def test_equivalent_to_factory_then_call(self, tiny_pipeline) -> None:
        """Equivalent to ``build_token_position_factories`` followed by calling each
        factory with the pipeline — the duplicated pattern this helper replaces."""
        template = "x={x}"
        specs: dict[str, Any] = {
            "first": {"type": "index", "position": 0},
            "x": {"type": "variable", "name": "x"},
        }
        positions = build_token_positions(specs, template, tiny_pipeline)
        factories = build_token_position_factories(specs, template)
        sample = _input_sample("x=5", x="5")
        for name in specs:
            assert positions[name].index(sample) == factories[name](
                tiny_pipeline
            ).index(sample)


# --------------------------------------------------------------------------- #
#  paired_token_position                                                      #
# --------------------------------------------------------------------------- #
class TestPairedTokenPositionUnit:
    """Wraps an original/counterfactual :class:`TokenPosition` pair into a
    single :class:`TokenPosition` whose indexer branches on the
    ``is_original`` flag. Used by the arrow-syntax ``var1<-var2`` parser in
    ``causalab.methods.interchange`` to drive different positions on the
    original-input vs counterfactual-input forward passes."""

    pytestmark = pytest.mark.unit

    def test_is_original_true_dispatches_to_original(self, tiny_pipeline) -> None:
        orig = TokenPosition(lambda x: [5], tiny_pipeline, id="orig")
        cf = TokenPosition(lambda x: [8], tiny_pipeline, id="cf")
        paired = paired_token_position(orig, cf, id="p")
        result = paired.index({}, is_original=True)
        assert result == [5]

    def test_is_original_false_dispatches_to_counterfactual(
        self, tiny_pipeline
    ) -> None:
        orig = TokenPosition(lambda x: [5], tiny_pipeline, id="orig")
        cf = TokenPosition(lambda x: [8], tiny_pipeline, id="cf")
        paired = paired_token_position(orig, cf, id="p")
        assert paired.index({}, is_original=False) == [8]

    def test_id_propagates(self, tiny_pipeline) -> None:
        orig = TokenPosition(lambda x: [0], tiny_pipeline, id="orig")
        cf = TokenPosition(lambda x: [0], tiny_pipeline, id="cf")
        paired = paired_token_position(orig, cf, id="last<-answer")
        assert paired.id == "last<-answer"

    def test_pipeline_from_original_preserved(self, tiny_pipeline) -> None:
        orig = TokenPosition(lambda x: [0], tiny_pipeline, id="orig")
        cf = TokenPosition(lambda x: [0], tiny_pipeline, id="cf")
        paired = paired_token_position(orig, cf)
        assert paired.pipeline is tiny_pipeline


# --------------------------------------------------------------------------- #
#  combined_token_position                                                    #
# --------------------------------------------------------------------------- #
class TestCombinedTokenPositionUnit:
    """Concatenates the indices returned by several :class:`TokenPosition`
    objects into a single :class:`TokenPosition`. Used by analyses that
    need to intervene on a *set* of positions identified by independent
    factories (e.g. ``answer`` ∪ ``last`` in a locate sweep)."""

    pytestmark = pytest.mark.unit

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            combined_token_position([])

    def test_concatenates_in_input_order(self, tiny_pipeline) -> None:
        a = TokenPosition(lambda x: [0, 1], tiny_pipeline, id="a")
        b = TokenPosition(lambda x: [5], tiny_pipeline, id="b")
        combined = combined_token_position([a, b], id="ab")
        assert combined.index({}) == [0, 1, 5]

    def test_forwards_is_original_to_children(self, tiny_pipeline) -> None:
        """Each child's indexer receives the same ``is_original`` value
        passed to the combined indexer."""

        def branch_a(input, is_original=True):
            return [0] if is_original else [10]

        def branch_b(input, is_original=True):
            return [1] if is_original else [11]

        a = TokenPosition(branch_a, tiny_pipeline, id="a")
        b = TokenPosition(branch_b, tiny_pipeline, id="b")
        combined = combined_token_position([a, b], id="ab")

        assert combined.index({}, is_original=True) == [0, 1]
        assert combined.index({}, is_original=False) == [10, 11]


class TestCombinedTokenPositionProperty:
    """Invariants: combined length is the sum of child lengths; permuting
    the input list permutes the output multiset."""

    pytestmark = pytest.mark.property

    def test_length_additivity(self, tiny_pipeline) -> None:
        a = TokenPosition(lambda x: [0, 1, 2], tiny_pipeline, id="a")
        b = TokenPosition(lambda x: [5, 6], tiny_pipeline, id="b")
        c = TokenPosition(lambda x: [9], tiny_pipeline, id="c")
        combined = combined_token_position([a, b, c])
        result = combined.index({})
        assert len(result) == 3 + 2 + 1

    def test_commutativity_up_to_order(self, tiny_pipeline) -> None:
        """Combining ``[a, b]`` and ``[b, a]`` yields permutations of the
        same multiset of indices."""
        a = TokenPosition(lambda x: [0, 1], tiny_pipeline, id="a")
        b = TokenPosition(lambda x: [5], tiny_pipeline, id="b")
        ab = combined_token_position([a, b]).index({})
        ba = combined_token_position([b, a]).index({})
        assert sorted(ab) == sorted(ba)
        assert set(ab) == set(ba)


# --------------------------------------------------------------------------- #
#  is_original flag plumbing — migrated from tests/neural/test_is_original_flag.py
# --------------------------------------------------------------------------- #
class TestTokenPositionIsOriginalFlagUnit:
    """Migrated from ``tests/neural/test_is_original_flag.py``: the
    ``TokenPosition``-specific cases pinning the per-call ``is_original``
    dispatch contract. The ``ComponentIndexer`` / ``AtomicModelUnit`` cases
    stay in their original file. (The vestigial ``is_original`` *constructor*
    flag was removed in #430 — routing is per call only.)"""

    pytestmark = pytest.mark.unit

    def test_old_style_token_position_still_works(self) -> None:
        """``TokenPosition`` with a legacy (no-``is_original``) indexer
        still returns the indexer's output unchanged."""
        mock_pipeline = Mock()

        def old_indexer(input):
            return [0]

        tp = TokenPosition(old_indexer, mock_pipeline, id="old")
        assert tp.index({"text": "test"}) == [0]

    def test_different_positions_for_original_and_counterfactual(self) -> None:
        """Realistic arrow-syntax scenario: the same indexer picks
        different positions depending on ``is_original``."""
        mock_pipeline = Mock()

        def variable_position_indexer(input, is_original=True):
            return [5] if is_original else [8]

        tp = TokenPosition(variable_position_indexer, mock_pipeline, id="variable_pos")
        assert tp.index({"text": "original"}, is_original=True) == [5]
        assert tp.index({"text": "counterfactual"}, is_original=False) == [8]


# --------------------------------------------------------------------------- #
#  Chat-template token-position realignment (Layer 2)                         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_chat_pipeline() -> LMPipeline:
    """A second tiny-Llama pipeline with ``use_chat_template=True``.

    The tiny-random Llama stub ships a real chat template
    (``<s>[INST] ... [/INST]``), so wrapping a prompt prepends a multi-token
    prefix and shifts every content token. Reuses the cached
    :func:`tiny_random_model` instance — only the (HF-cached) tokenizer load is
    paid. Distinct tokenizer object from the session-scoped ``tiny_pipeline``,
    which is fine: these tests compare *positions / decoded values*, and the one
    test that pins cache-key isolation toggles ``use_chat_template`` on a single
    pipeline so the tokenizer id is held constant.
    """
    return LMPipeline(
        tiny_random_model(),
        max_new_tokens=1,
        padding_side="left",
        use_chat_template=True,
    )


class TestChatTemplatePositionRealignmentUnit:
    """Under a chat template, ``LMPipeline.load`` tokenizes the *wrapped* prompt,
    so every char-range → token-index path must rebase by the chat prefix or
    interventions read the wrong residual-stream rows. These pin the Layer-2
    realignment: variable / substring / absolute-index paths, the single-BOS
    fix, role-marker robustness, cache isolation, and the unpreservable guard.
    """

    pytestmark = pytest.mark.unit

    def test_variable_parity_same_token_shifted_index(
        self, tiny_pipeline, tiny_chat_pipeline
    ) -> None:
        """The variable resolves to the same decoded content with and without
        the chat template, but at a shifted integer index (a real rebase, not a
        no-op). The exact shift is not the prefix token count: the prefix/content
        seam (leading space) and the bare path's BOS both perturb BPE
        boundaries, so we pin the robust invariant — same content, indices move.
        """
        t = Template("The sum of {x} and {y} is ")
        values = {"x": "5", "y": "7"}
        bare = t.get_variable_positions(values, tiny_pipeline)
        chat = t.get_variable_positions(values, tiny_chat_pipeline)

        assert bare["x"] != chat["x"], "expected a real index shift under chat mode"

        full = t.fill(values)
        bare_ids = tiny_pipeline.load([_trace(full)])["input_ids"][0]
        chat_ids = tiny_chat_pipeline.load([_trace(full)])["input_ids"][0]
        bare_dec = tiny_pipeline.tokenizer.decode([bare_ids[i] for i in bare["x"]])
        chat_dec = tiny_chat_pipeline.tokenizer.decode([chat_ids[i] for i in chat["x"]])
        assert bare_dec.strip() == "5"
        assert chat_dec.strip() == "5"

    def test_substring_parity_same_token_after_prefix(
        self, tiny_pipeline, tiny_chat_pipeline
    ) -> None:
        """``get_substring_token_ids`` resolves the same content token under both
        modes; the chat indices land after the chat prefix and differ from bare.
        """
        text = "The sum of 5 and 7 is 12"
        bare = get_substring_token_ids(text, "7", tiny_pipeline)
        chat = get_substring_token_ids(text, "7", tiny_chat_pipeline)

        assert bare != chat
        # Content tokens live after the wrapped prefix.
        assert min(chat) >= tiny_chat_pipeline._chat_prefix_token_count()

        # Bare path uses add_special_tokens=False; mirror it to line up indices.
        bare_ids = tiny_pipeline.load([_trace(text)], add_special_tokens=False)[
            "input_ids"
        ][0]
        chat_ids = tiny_chat_pipeline.load([_trace(text)])["input_ids"][0]
        assert (
            tiny_pipeline.tokenizer.decode([bare_ids[i] for i in bare]).strip() == "7"
        )
        assert (
            tiny_chat_pipeline.tokenizer.decode([chat_ids[i] for i in chat]).strip()
            == "7"
        )

    def test_chat_wrapped_prompt_has_single_bos(self, tiny_chat_pipeline) -> None:
        """Regression for the double-BOS bug: the wrapped string already embeds
        the BOS, so ``load`` tokenizes it with ``add_special_tokens=False`` and
        exactly one BOS survives."""
        ids = tiny_chat_pipeline.load([_trace("hello world")])["input_ids"][0]
        bos_id = tiny_chat_pipeline.tokenizer.bos_token_id
        assert int((ids == bos_id).sum().item()) == 1

    def test_absolute_index_rebased_under_chat(
        self, tiny_pipeline, tiny_chat_pipeline
    ) -> None:
        """Negative absolute indices are unchanged (``-1`` is the generation
        slot); non-negative indices skip the chat prefix so ``position=0`` is the
        first *content* token (== prefix token count), not BOS."""
        specs = {
            "first": {"type": "index", "position": 0},
            "last": {"type": "index", "position": -1},
        }
        factories = build_token_position_factories(specs, "x={x}")
        sample = _input_sample("x=5", x="5")

        bare_ids = tiny_pipeline.load([sample])["input_ids"][0]
        chat_ids = tiny_chat_pipeline.load([sample])["input_ids"][0]

        # position=-1 → last token, both modes.
        assert factories["last"](tiny_pipeline).index(sample) == [len(bare_ids) - 1]
        assert factories["last"](tiny_chat_pipeline).index(sample) == [
            len(chat_ids) - 1
        ]

        # position=0 → first token (bare) / first content token (chat).
        assert factories["first"](tiny_pipeline).index(sample) == [0]
        assert factories["first"](tiny_chat_pipeline).index(sample) == [
            tiny_chat_pipeline._chat_prefix_token_count()
        ]

    def test_role_marker_collision_resolves_to_content(
        self, tiny_chat_pipeline
    ) -> None:
        """Content equal to a role marker (``"INST"`` vs the ``[INST]`` marker)
        must resolve to the *content* occurrence. The sentinel-based offset (not
        ``wrapped.find(content)``) makes this unambiguous."""
        t = Template("{m}")
        pos = t.get_variable_positions({"m": "INST"}, tiny_chat_pipeline)

        enc = tiny_chat_pipeline.load([_trace("INST")], return_offsets_mapping=True)
        ids = enc["input_ids"][0]
        offsets = enc["offset_mapping"][0]
        content_off = enc["content_char_offset"]

        decoded = tiny_chat_pipeline.tokenizer.decode([ids[i] for i in pos["m"]])
        assert decoded.strip() == "INST"
        # Every resolved token overlaps the content span [content_off, +len),
        # never the earlier ``[INST]`` role marker (whose tokens all end before
        # the content begins). A content token may *start* one char early when it
        # absorbs the boundary space, so we check span overlap, not start >= off.
        for i in pos["m"]:
            tok_start, tok_end = int(offsets[i][0]), int(offsets[i][1])
            assert tok_end > content_off
            assert tok_start < content_off + len("INST")

    def test_position_cache_isolated_by_chat_mode(self, tiny_chat_pipeline) -> None:
        """``Template._position_cache`` keys on ``use_chat_template``: toggling
        only that flag on one pipeline (tokenizer id held constant) must NOT
        return a stale hit from the other mode."""
        t = Template("{m}")
        chat_pos = t.get_variable_positions({"m": "5"}, tiny_chat_pipeline)
        tiny_chat_pipeline.use_chat_template = False
        try:
            bare_pos = t.get_variable_positions({"m": "5"}, tiny_chat_pipeline)
        finally:
            tiny_chat_pipeline.use_chat_template = True
        assert chat_pos["m"] != bare_pos["m"]

    def test_template_stripping_raises_descriptive_error(
        self, tiny_chat_pipeline
    ) -> None:
        """The tiny Llama template strips content whitespace, so a leading-space
        value can't be located verbatim — the guard converts that silent
        corruption into a descriptive ``ValueError`` instead of wrong indices."""
        t = Template("{x}")
        with pytest.raises(ValueError, match="Chat template altered"):
            t.get_variable_positions({"x": "  spaced"}, tiny_chat_pipeline)


class TestRawInputGuard:
    """Position resolution tokenizes the *template filled with the example's
    variables*, but the model actually runs the trace's ``raw_input`` (see
    :meth:`LMPipeline._load`). For causalab-sampled data those strings are equal
    by construction; an externally-built example can make them diverge, which
    would silently read every intervention at the wrong token. The guard turns
    that into a loud :class:`PromptTemplateMismatchError` at the source.
    """

    pytestmark = pytest.mark.property

    @staticmethod
    def _var_trace(raw_input: str, x: str) -> CausalTrace:
        """A trace carrying both ``raw_input`` and the template variable ``x``."""
        return CausalTrace(
            mechanisms={
                "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"]),
                "x": Mechanism(parents=[], compute=lambda t: t["x"]),
            },
            inputs={"raw_input": raw_input, "x": x},
        )

    # -- unit (no model) ---------------------------------------------------- #

    def test_assert_passes_when_equal_or_absent(self) -> None:
        # Equal: no raise.
        _assert_raw_input_matches({"raw_input": "A cat B"}, "A cat B")
        # Absent: template fill is authoritative, check is skipped.
        _assert_raw_input_matches({"x": "cat"}, "A cat B")
        # Non-str raw_input: skipped rather than crashing.
        _assert_raw_input_matches({"raw_input": 123}, "A cat B")

    def test_assert_raises_on_divergence(self) -> None:
        with pytest.raises(PromptTemplateMismatchError, match="character 7"):
            _assert_raw_input_matches({"raw_input": "A cat B EXTRA"}, "A cat B")

    def test_first_difference(self) -> None:
        assert _first_difference("A cat B", "A cat B") == 7  # prefix-equal -> min len
        assert _first_difference("A cat B", "A dog B") == 2

    # -- integration (tiny pipeline, real tokenizer) ------------------------ #

    def test_consistent_trace_resolves(self, tiny_pipeline: LMPipeline) -> None:
        positions = build_token_positions(
            {"x": {"type": "variable", "name": "x"}}, "A {x} B", tiny_pipeline
        )
        idx = positions["x"].index(self._var_trace("A cat B", "cat"))
        assert idx and all(isinstance(i, int) for i in idx)

    def test_divergent_raw_input_raises(self, tiny_pipeline: LMPipeline) -> None:
        positions = build_token_positions(
            {"x": {"type": "variable", "name": "x"}}, "A {x} B", tiny_pipeline
        )
        with pytest.raises(PromptTemplateMismatchError):
            positions["x"].index(self._var_trace("A cat B and extra", "cat"))
