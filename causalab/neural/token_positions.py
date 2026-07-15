"""
Token Position Utilities

This module provides tools for working with token positions in language models:

1. **Core utilities** (TokenPosition, get_substring_token_ids, etc.)
2. **Declarative specification system** for complex position patterns

Declarative system supports:

1. Fixed positions (first, last, nth token)
2. Variable positions (where a template variable appears)
3. Indexed positions (nth token within a variable)
4. Relative positions (tokens before/after a variable)
5. Dynamic positions (function that returns a spec based on causal model setting)

Usage:
    token_positions = {
        "last": {"type": "index", "position": -1},
        "x": {"type": "variable", "name": "x"},
        "second_token_of_x": {"type": "index", "position": 1, "scope": {"variable": "x"}},
        "token_after_x": {"type": "index", "position": +1, "relative_to": {"variable": "x"}},
        # Dynamic spec based on causal model variables
        "correct_answer": lambda setting: {
            "type": "variable",
            "name": "option_Z" if setting["answer_letter"] == 'Z' else "option_X"
        }
    }

    # Build token position factories
    factories = build_token_position_factories(token_positions, template)

    # Use in Task
    task = Task(..., token_positions=factories)
"""

import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Union, cast

import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.units import ComponentIndexer
from causalab.neural.pipeline import LMPipeline


class PromptTemplateMismatchError(ValueError):
    """The example's ``raw_input`` disagrees with the template-filled prompt.

    Raised by token-position resolution when the string it tokenizes to locate a
    variable (this task's template filled with the example's variable values)
    differs from the trace's own ``raw_input`` — the string the model actually
    runs (see :meth:`LMPipeline._load`). The two are equal by construction for
    any example produced by causalab's own sampler under this template, so a
    mismatch means the dataset was built elsewhere (a foreign tokenizer/renderer,
    a different template, or a hand-built example dict). Continuing would compute
    token indices against a different string than the run prompt, silently
    reading interventions at the wrong positions; we fail here instead.
    """


# --------------------------------------------------------------------------- #
#  Token Position Utilities                                                   #
# --------------------------------------------------------------------------- #


def _load_for_indexing(
    pipeline: LMPipeline, traces: List[CausalTrace], **load_kwargs: Any
) -> Dict[str, Any]:
    """Tokenize for token-position resolution, enforcing the unpadded-frame invariant.

    Indexers return indices in each example's *unpadded* frame; the padded-batch shift
    in ``units._apply_padding_shift`` depends on that. A non-None ``pipeline.max_length``
    pads each per-example load up to ``max_length``, pushing indices out of the unpadded
    frame and silently corrupting interventions. Refuse it loudly here.
    """
    if getattr(pipeline, "max_length", None) is not None:
        raise ValueError(
            "Token positions must be resolved with a pipeline whose max_length is "
            f"None, but got max_length={pipeline.max_length!r}. A fixed max_length pads "
            "each per-example tokenization, pushing token indices out of the unpadded "
            "frame that interventions assume. Construct the LMPipeline with "
            "max_length=None (the default) when it is used to build token positions; "
            "max_length still works for generation-only pipelines."
        )
    return pipeline.load(traces, **load_kwargs)


class TokenPosition(ComponentIndexer):
    """Dynamic indexer: returns position(s) of interest for a prompt.

    Attributes
    ----------
    pipeline :
        The :class:`neural.pipeline.LMPipeline` supplying the tokenizer.

    Notes
    -----
    Whether a position resolves differently for original vs. counterfactual
    inputs is decided per call, via the ``is_original`` keyword threaded through
    :meth:`ComponentIndexer.index` to indexers that accept it (see
    :func:`paired_token_position`). There is deliberately no ``is_original``
    *constructor* flag: it routed nothing and only duplicated that name (#430).
    """

    def __init__(self, indexer, pipeline: LMPipeline, **kwargs):
        super().__init__(indexer, **kwargs)
        self.pipeline = pipeline

    def highlight_selected_token(self, input: CausalTrace) -> str:
        """Return *prompt* with selected token(s) wrapped in ``**bold**``.

        The method tokenizes *prompt*, calls self.index to obtain the
        positions, then re-assembles a detokenised string with the
        selected token(s) wrapped in ``**bold**``.  The rest of the
        prompt is unchanged.

        Note that whitespace handling may be approximate for tokenizers
        that encode leading spaces as special glyphs (e.g. ``Ġ``).
        """
        ids = _load_for_indexing(self.pipeline, [input])["input_ids"][0]
        highlight = self.index(input)

        pad_token_id = self.pipeline.tokenizer.pad_token_id

        return "".join(
            f"**{self.pipeline.tokenizer.decode(t)}**"
            if i in highlight
            else self.pipeline.tokenizer.decode(t)
            for i, t in enumerate(ids)
            if t != pad_token_id
        )


# Convenience indexers
def get_last_token_index(input: CausalTrace, pipeline: LMPipeline) -> List[int]:
    """Return a one-element list containing the *last* token index."""
    ids = list(_load_for_indexing(pipeline, [input])["input_ids"][0])
    return [len(ids) - 1]


def get_all_tokens(
    input: CausalTrace, pipeline: LMPipeline, padding: bool = False
) -> TokenPosition:
    """Return a single TokenPosition object containing all (non-pad) token indices."""
    pad_token_id = pipeline.tokenizer.pad_token_id

    # Create indexer function that returns all non-pad token indices
    def all_tokens_indexer(inp: CausalTrace) -> List[int]:
        token_ids = _load_for_indexing(pipeline, [inp])["input_ids"][0]
        if padding:
            return [i for i in range(len(token_ids))]
        return [i for i in range(len(token_ids)) if token_ids[i] != pad_token_id]

    return TokenPosition(indexer=all_tokens_indexer, pipeline=pipeline, id="all_tokens")


def get_list_of_each_token(
    input: CausalTrace | str, pipeline: LMPipeline
) -> List[TokenPosition]:
    """Return a list of TokenPosition objects, each containing a single token index."""
    # Convert string to CausalTrace if needed
    if isinstance(input, str):
        trace = CausalTrace(
            mechanisms={
                "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
            },
            inputs={"raw_input": input},
        )
    else:
        trace = input
    ids = list(_load_for_indexing(pipeline, [trace])["input_ids"][0])
    pad_token_id = pipeline.tokenizer.pad_token_id

    token_positions = []
    for i in range(len(ids)):
        if ids[i] != pad_token_id:
            # Create indexer function for this specific position
            def single_token_indexer(inp, pos=i):
                return [pos]

            # Decode the token to create a meaningful label
            token_str = pipeline.tokenizer.decode([ids[i]])
            # Clean up the token string for display
            token_label = token_str.strip().replace("\n", "\\n")
            if len(token_label) > 10:
                token_label = token_label[:10] + "..."

            token_positions.append(
                TokenPosition(
                    indexer=single_token_indexer,
                    pipeline=pipeline,
                    id=f"tok_{i}_{token_label}",
                )
            )

    return token_positions


def get_tokens_in_char_range(
    offsets: torch.Tensor,  # shape [seq_len, 2]
    start_char: int,
    end_char: int,
) -> List[int]:
    """
    Find which tokens overlap with a character range.

    Given tokenizer offset_mapping and a character range [start_char, end_char),
    returns the list of token indices whose character spans overlap with the range.

    Parameters
    ----------
    offsets : tensor or list of tuples
        The offset_mapping from tokenizer output, where each entry is (start, end)
        character positions for that token. Padding tokens have offset (0, 0).
    start_char : int
        Start of the character range (inclusive)
    end_char : int
        End of the character range (exclusive)

    Returns
    -------
    List[int]
        Token indices that overlap with the character range, in order.

    Notes
    -----
    - Padding tokens (offset (0, 0)) are automatically skipped
    - A token overlaps if its character span has any intersection with [start_char, end_char)
    """
    if end_char <= start_char:
        return []

    starts = offsets[:, 0]
    ends = offsets[:, 1]

    nonpad = ~((starts == 0) & (ends == 0))
    overlap = (starts < end_char) & (ends > start_char)
    idx = torch.nonzero(nonpad & overlap, as_tuple=False).flatten()
    return idx.tolist()


def rebase_char_range(
    tokenized: Mapping[str, Any],
    start_char: int,
    end_char: int,
    expected: str,
    label: str,
) -> tuple[int, int]:
    """Shift a bare-text character range into chat-wrapped coordinates.

    Char ranges (variable substitutions, substrings) are computed against the
    *bare* task text, but under a chat template ``pipeline.load`` tokenizes the
    *wrapped* prompt — so its ``offset_mapping`` indexes the wrapped string. When
    chat wrapping is active ``load`` attaches ``content_char_offset`` (where the
    bare content begins in the wrapped prompt) and the ``wrapped_text``; we add
    that offset so the range lines up with the offsets, and assert the spanned
    text survived verbatim.

    The verbatim check converts a silent corruption into an actionable error:
    some chat templates ``.strip()`` the content (Llama does), so a value that
    *is* or *borders* whitespace loses characters and every intervention index
    downstream would be off by the lost characters. Returns the original range
    unchanged (a no-op) when chat templating is off.
    """
    offset = tokenized.get("content_char_offset", 0)
    if not offset:
        return start_char, end_char
    wrapped_list = tokenized.get("wrapped_text")
    if wrapped_list is not None:
        wrapped = wrapped_list[0]
        got = wrapped[start_char + offset : end_char + offset]
        if got != expected:
            raise ValueError(
                f"Chat template altered {label} (expected {expected!r}): it does "
                f"not appear verbatim at the expected position in the chat-wrapped "
                f"prompt (found {got!r}). Templates that strip whitespace corrupt "
                f"token positions for content that is or borders whitespace. Pad "
                f"the value away from the template boundary, or run this analysis "
                f"with chat_template disabled."
            )
    return start_char + offset, end_char + offset


def get_substring_token_ids(
    text: str,
    substring: str,
    pipeline: LMPipeline,
    add_special_tokens: bool = False,
    occurrence: int = 0,
    strict: bool = False,
) -> List[int]:
    """Return token position indices for tokens that overlap with a substring.

    Given a text and a substring that occurs within it, returns the list of
    token position indices corresponding to tokens that overlap with the substring.
    When the substring boundaries fall in the middle of a token, that token is
    included in the result.

    Parameters
    ----------
    text : str
        The full input text to tokenize.
    substring : str
        A substring that occurs within `text`. Must be present in the text.
    pipeline : LMPipeline
        The pipeline containing the tokenizer to use.
    add_special_tokens : bool, optional
        Whether to add special tokens (BOS/EOS) during tokenization. Default is False.
        No-op when the pipeline applies a chat template: the wrapped prompt already
        embeds the specials, so ``LMPipeline.load`` forces ``add_special_tokens=False``
        regardless of this argument (see the double-BOS note there).
    occurrence : int, optional
        Which occurrence of the substring to use (0-indexed). Supports negative indexing
        like Python lists (-1 for last, -2 for second-to-last, etc.). Default is 0 (first occurrence).
    strict : bool, optional
        If True, raises ValueError when multiple occurrences exist. Default is False.

    Returns
    -------
    List[int]
        A list of token position indices (0-indexed) for tokens overlapping the substring.

    Raises
    ------
    ValueError
        If substring is empty, text is empty, substring is not found, the specified
        occurrence doesn't exist, or (when strict=True) multiple occurrences exist.

    Examples
    --------
    >>> text = "The sum of 5 and 5 is 10"
    >>> substring = "5"
    >>> # Get first occurrence (default)
    >>> indices = get_substring_token_ids(text, substring, pipeline)
    >>> # Get second occurrence explicitly
    >>> indices = get_substring_token_ids(text, substring, pipeline, occurrence=1)
    >>> # Get last occurrence using negative indexing
    >>> indices = get_substring_token_ids(text, substring, pipeline, occurrence=-1)
    >>> # Fail if ambiguous
    >>> indices = get_substring_token_ids(text, substring, pipeline, strict=True)  # Raises!

    Notes
    -----
    - This function is inclusive: any token with any character overlap gets included.
    - Handles tokenizer-specific behaviors like leading space encoding (e.g., Ġ in GPT-2).
    - When multiple occurrences exist and strict=False, uses the first by default.
    """
    # Validation
    if not text:
        raise ValueError("Text cannot be empty")
    if not substring:
        raise ValueError("Substring cannot be empty")
    if substring not in text:
        raise ValueError(f"Substring '{substring}' not found in text")

    # Find all occurrences
    occurrences = []
    start = 0
    while True:
        pos = text.find(substring, start)
        if pos == -1:
            break
        occurrences.append(pos)
        start = pos + 1

    num_occurrences = len(occurrences)

    # Check for ambiguity in strict mode
    if strict and num_occurrences > 1:
        raise ValueError(
            f"Found {num_occurrences} occurrences of '{substring}' in the text. "
            f"Please either:\n"
            f"  1. Use more specific context to make substring unique\n"
            f"  2. Specify which occurrence with occurrence parameter (0 to {num_occurrences - 1} or -1 to -{num_occurrences})\n"
            f"  3. Set strict=False to use first occurrence (default behavior)"
        )

    # Handle negative indexing (Python-style)
    if occurrence < 0:
        occurrence = num_occurrences + occurrence

    # Validate occurrence parameter
    if occurrence < 0 or occurrence >= num_occurrences:
        raise ValueError(
            f"Occurrence index {occurrence if occurrence >= 0 else occurrence - num_occurrences} out of range. "
            f"Found {num_occurrences} occurrence(s) of '{substring}'. "
            f"Valid indices: 0 to {num_occurrences - 1} or -1 to -{num_occurrences}"
        )

    # Convert text to CausalTrace
    trace = CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )

    # Use pipeline.load() with offset_mapping to get character→token mapping
    # This ensures we use the exact same tokenization as interventions
    tokenized = _load_for_indexing(
        pipeline,
        [trace],
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    offsets = tokenized["offset_mapping"][0]  # Get first sequence from batch

    # Find which tokens overlap with the substring's character range. Under a
    # chat template the offsets index the wrapped prompt, so rebase the bare
    # char range by the chat prefix (and verify the template preserved it).
    substring_start = occurrences[occurrence]
    substring_end = substring_start + len(substring)
    substring_start, substring_end = rebase_char_range(
        tokenized, substring_start, substring_end, substring, f"substring {substring!r}"
    )

    return get_tokens_in_char_range(offsets, substring_start, substring_end)


# --------------------------------------------------------------------------- #
#  Template System                                                            #
# --------------------------------------------------------------------------- #


def _first_difference(a: str, b: str) -> int:
    """Char index of the first divergence (or ``min(len)`` if one is a prefix)."""
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))


def _assert_raw_input_matches(values: Any, full_text_str: str) -> None:
    """Fail if the example's ``raw_input`` differs from the template fill.

    ``values`` is the example being indexed (a :class:`CausalTrace` or a plain
    dict). When it exposes a ``raw_input``, that string is what the model runs,
    so it must equal ``full_text_str`` (this template filled with the example's
    variables) for the computed token indices to be meaningful. If ``raw_input``
    is absent or not yet computable, nothing was overridden and the template fill
    is authoritative, so the check is skipped.
    """
    try:
        raw_input = values["raw_input"]
    except (KeyError, ValueError, TypeError):
        return
    if not isinstance(raw_input, str) or raw_input == full_text_str:
        return
    i = _first_difference(raw_input, full_text_str)
    raise PromptTemplateMismatchError(
        "Token positions are resolved by tokenizing this task's template filled "
        "with the example's variables, but the example's own 'raw_input' (the "
        f"prompt the model runs) differs starting at character {i}, so the "
        "computed positions would not align with that prompt:\n"
        f"  raw_input (run)        : {raw_input!r}\n"
        f"  template-filled (posn) : {full_text_str!r}\n"
        "The dataset was likely built outside causalab (a different "
        "tokenizer/renderer, a different template, or a hand-built example dict). "
        "Rebuild each example through the task's causal model — e.g. "
        "causal_model.new_trace(input_variables) — so raw_input is rendered from "
        "the template, before running interventions."
    )


class Template:
    """
    A proper templating system that parses templates, fills them with values,
    and tracks where each variable appears in the tokenized output.

    Template format: "The value of {x} plus {y} equals "
    Variables are specified with {variable_name} syntax.
    """

    def __init__(self, template_str: str):
        """
        Parse a template string to identify variables and literal parts.

        Args:
            template_str: Template string with {variable} placeholders
        """
        self.template_str = template_str
        self.parts = []  # List of (type, content) where type is 'literal' or 'variable'
        self._parse()

    def _parse(self):
        """Parse template into alternating literals and variables."""
        # Split on {variable} patterns while keeping the variable names
        pattern = r"\{([^}]+)\}"
        last_end = 0

        for match in re.finditer(pattern, self.template_str):
            # Add literal text before this variable
            if match.start() > last_end:
                literal = self.template_str[last_end : match.start()]
                self.parts.append(("literal", literal))

            # Add the variable
            var_name = match.group(1)
            self.parts.append(("variable", var_name))

            last_end = match.end()

        # Add any trailing literal
        if last_end < len(self.template_str):
            literal = self.template_str[last_end:]
            self.parts.append(("literal", literal))

    def fill(self, values: Dict[str, Any]) -> str:
        """
        Fill the template with values.

        Args:
            values: Dictionary mapping variable names to their values

        Returns:
            The filled template string
        """
        result = []
        for part_type, content in self.parts:
            if part_type == "literal":
                result.append(content)
            else:  # variable
                if content not in values:
                    raise ValueError(
                        f"Missing value for template variable: {content}. You probably put a non-input variable in your template."
                    )
                result.append(str(values[content]))
        return "".join(result)

    # Class-level LRU cache for tokenization results.
    #
    # Key: ``(tokenizer.name_or_path, use_chat_template, full_text_str)``.
    #   * ``name_or_path`` is the tokenizer's *stable, unique identity*, not
    #     ``id()``. ``id()`` is recycled after GC, so a long-lived process that
    #     frees and rebuilds pipelines (sweeps, session reuse) could get a cache
    #     hit from a *different* tokenizer that landed at the same address —
    #     silently returning another tokenizer's token indices. Keying on a
    #     stable identity removes that aliasing. Distinct tokenizers are expected
    #     to carry distinct ``name_or_path`` values; a missing/empty
    #     ``name_or_path`` is not a usable identity and is rejected loudly at
    #     lookup time (see ``get_variable_positions``) rather than silently
    #     collapsing distinct tokenizers onto one cache key.
    #   * The chat-template flag is part of the key because the same bare text
    #     tokenizes to different token positions with vs. without the chat
    #     wrapper — without it the two modes would collide on a cache hit and
    #     silently return the wrong indices.
    #
    # Bounded via LRU eviction (``_POSITION_CACHE_MAXSIZE``): the least-recently
    # -used entry is dropped once the bound is exceeded, so the class-level dict
    # cannot grow for the whole process lifetime across sweeps.
    _POSITION_CACHE_MAXSIZE: int = 4096
    _position_cache: "OrderedDict[tuple[str, bool, str], Dict[str, List[int]]]" = (
        OrderedDict()
    )

    def get_variable_positions(
        self, values: Dict[str, Any], pipeline
    ) -> Dict[str, List[int]]:
        """
        Fill the template and track which tokens correspond to each variable.

        This is the key method: we tokenize the template piece by piece,
        tracking exactly where each variable's tokens appear.

        Results are cached by the filled text to avoid re-tokenization.

        Args:
            values: Dictionary mapping variable names to their values
            pipeline: The tokenization pipeline

        Returns:
            Dictionary mapping variable names to lists of token indices
        """
        # Build the full text while tracking character positions
        char_positions = {}  # var_name -> [(start_char, end_char, value_str), ...]
        current_pos = 0
        full_text = []

        for part_type, content in self.parts:
            if part_type == "literal":
                full_text.append(content)
                current_pos += len(content)
            else:  # variable
                if content not in values:
                    raise ValueError(f"Missing value for template variable: {content}")

                value_str = str(values[content])
                start_char = current_pos
                end_char = current_pos + len(value_str)

                # Track this variable's character positions (and the substituted
                # value, used to verify chat-template preservation when rebasing).
                # Support multiple occurrences - store as list of ranges.
                if content not in char_positions:
                    char_positions[content] = []
                char_positions[content].append((start_char, end_char, value_str))

                full_text.append(value_str)
                current_pos = end_char

        full_text_str = "".join(full_text)

        # Guardrail: ``full_text_str`` (this template filled with the example's
        # variables) is what we tokenize below to locate variables, but the model
        # actually runs the trace's ``raw_input``. They match by construction for
        # causalab-sampled data; a pre-existing/externally-built example can carry
        # a divergent ``raw_input`` that would make every index here land on the
        # wrong token with no error. Validate before the cache lookup so cache
        # hits are checked too.
        _assert_raw_input_matches(values, full_text_str)

        # Check cache first. Keyed on the tokenizer's *stable identity*
        # (``name_or_path``, not ``id()``) plus the chat-template flag, so bare
        # and chat-wrapped runs of the same text don't collide and a recycled
        # object address can't serve a different tokenizer's indices (see
        # _position_cache). ``name_or_path`` is the cache's notion of tokenizer
        # identity and is expected to be unique per tokenizer; a missing/empty
        # value (e.g. a programmatically-constructed tokenizer that never set it)
        # is not a usable key and would silently collapse distinct tokenizers
        # onto one entry, so fail loudly instead of caching under a bad key.
        name_or_path = pipeline.tokenizer.name_or_path
        if not name_or_path:
            raise ValueError(
                "Cannot resolve token positions: the tokenizer's "
                f"`name_or_path` is {name_or_path!r}. The position cache uses "
                "`name_or_path` as a unique tokenizer identity, and a "
                "missing/empty value would alias distinct tokenizers onto one "
                "cache entry and silently return the wrong token indices. Load "
                "the tokenizer via `from_pretrained` (which populates "
                "`name_or_path`) or set `tokenizer.name_or_path` to a unique, "
                "stable identifier before running interventions."
            )
        cache_key: tuple[str, bool, str] = (
            name_or_path,
            pipeline.use_chat_template,
            full_text_str,
        )
        cached = Template._position_cache.get(cache_key)
        if cached is not None:
            # Mark as most-recently-used for LRU eviction.
            Template._position_cache.move_to_end(cache_key)
            return cached

        # Convert text to CausalTrace
        trace = CausalTrace(
            mechanisms={
                "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
            },
            inputs={"raw_input": full_text_str},
        )

        # Use pipeline.load() with offset_mapping to get character→token mapping
        # This ensures we use the exact same tokenization as interventions
        tokenized = _load_for_indexing(pipeline, [trace], return_offsets_mapping=True)
        offsets = tokenized["offset_mapping"][0]  # Get first sequence from batch

        # Map character positions to token indices using offsets. Under a chat
        # template the offsets index the wrapped prompt, so rebase each bare char
        # range by the chat prefix (and verify the template preserved the value).
        variable_tokens = {}
        for var_name, char_ranges in char_positions.items():
            variable_tokens[var_name] = []

            for start_char, end_char, value_str in char_ranges:
                rebased_start, rebased_end = rebase_char_range(
                    tokenized, start_char, end_char, value_str, f"variable {var_name!r}"
                )
                # Find which tokens overlap with this character range
                tokens = get_tokens_in_char_range(offsets, rebased_start, rebased_end)
                variable_tokens[var_name].extend(tokens)

        # Cache the result, evicting the least-recently-used entries once the
        # bound is exceeded so the class-level dict stays bounded across sweeps.
        Template._position_cache[cache_key] = variable_tokens
        while len(Template._position_cache) > Template._POSITION_CACHE_MAXSIZE:
            Template._position_cache.popitem(last=False)

        return variable_tokens

    def get_variable_names(self) -> List[str]:
        """Return list of all variable names in the template."""
        return list(
            set(content for part_type, content in self.parts if part_type == "variable")
        )


def build_token_position_factories(
    specs: Mapping[str, Union[Dict[str, Any], Callable]], template: str
) -> Dict[str, Callable]:
    """
    Build token position factory functions from declarative specifications.

    Args:
        specs: Dictionary mapping position names to either:
               - Declarative specs (dict): {"type": "index", "position": -1}
               - Spec generator functions (callable): lambda setting: {"type": "variable", "name": "x"}
        template: The raw_input template string with {variable} placeholders
                  Example: "The sum of {x} and {y} is "

    Returns:
        Dictionary mapping position names to factory functions that take a pipeline
        and return TokenPosition objects
    """
    factories = {}

    for name, spec in specs.items():
        factories[name] = _build_factory(name, spec, template)

    return factories


def build_token_positions(
    specs: Mapping[str, Union[Dict[str, Any], Callable]],
    template: str,
    pipeline: LMPipeline,
) -> Dict[str, TokenPosition]:
    """Build declarative specs and materialize them into TokenPosition instances.

    Thin wrapper over :func:`build_token_position_factories` that immediately calls
    each factory with ``pipeline``. This is the shape every consumer needs: a task's
    ``create_token_positions`` returns ``Dict[str, TokenPosition]``, and the runner
    (``build_targets_for_grid``) / activations layer (``build_residual_stream_targets``)
    read ``.id`` off each value.

    Prefer this over calling :func:`build_token_position_factories` directly in a task
    wrapper. Returning the un-materialized factories crashes the ``locate`` step with
    ``AttributeError: 'function' object has no attribute 'id'`` (issue #179).

    Args:
        specs: Position name → declarative spec dict or dynamic spec generator, as
            accepted by :func:`build_token_position_factories`.
        template: The ``raw_input`` template string with ``{variable}`` placeholders.
        pipeline: The pipeline the positions resolve against.

    Returns:
        Dict mapping position name → :class:`TokenPosition` instance.
    """
    factories = build_token_position_factories(specs, template)
    return {name: factory(pipeline) for name, factory in factories.items()}


def _build_factory(
    name: str, spec: Union[Dict[str, Any], Callable], template: str
) -> Callable:
    """
    Build a single token position factory from a spec.

    Args:
        name: Name of the token position
        spec: Either a declarative spec dict or a function that takes a setting and returns a spec dict
        template: The raw_input template string

    Returns:
        Factory function that takes a pipeline and returns a TokenPosition
    """
    # Check if spec is a callable (dynamic spec generator)
    if callable(spec):
        return _build_dynamic_factory(name, spec, template)

    # Otherwise, it's a static declarative spec
    spec_type = spec.get("type")

    if spec_type == "index":
        return _build_index_factory(name, spec, template)
    elif spec_type == "variable":
        return _build_variable_factory(name, spec, template)
    else:
        raise ValueError(f"Unknown token position type: {spec_type}")


def _build_dynamic_factory(name: str, spec_func: Callable, template: str) -> Callable:
    """
    Build factory for dynamic spec generators.

    The spec_func receives the full causal model setting and returns a declarative spec dict.

    Args:
        name: Name of the token position
        spec_func: Function that takes input_sample and returns a spec dict
        template: The raw_input template string

    Returns:
        Factory function that creates a TokenPosition with dynamic behavior
    """

    def factory(pipeline):
        def indexer(input_sample):
            # Call the spec function to get the actual spec for this example
            actual_spec = spec_func(input_sample)

            # Build a factory from the returned spec
            temp_factory = _build_factory(f"{name}_dynamic", actual_spec, template)
            # Get the TokenPosition from that factory
            return temp_factory(pipeline).index(input_sample)

        return TokenPosition(indexer, pipeline, id=name)

    return factory


def _build_index_factory(name: str, spec: Dict[str, Any], template: str) -> Callable:
    """
    Build factory for index-based positions.

    Spec format:
        {"type": "index", "position": -1}  # Last token
        {"type": "index", "position": 0}   # First token
        {"type": "index", "position": 1, "scope": {"variable": "x"}}  # 2nd token of x
        {"type": "index", "position": +1, "relative_to": {"variable": "x"}}  # After x
    """
    position = spec.get("position")
    if position is None:
        raise ValueError(
            f"index-type token position spec {name!r} is missing required "
            f"'position' field."
        )
    scope = spec.get("scope")
    relative_to = spec.get("relative_to")

    if scope is not None:
        # Index within a variable's token sequence
        return _build_scoped_index_factory(name, position, scope, template)
    elif relative_to is not None:
        # Index relative to a variable
        return _build_relative_index_factory(name, position, relative_to, template)
    else:
        # Index in full sequence
        return _build_absolute_index_factory(name, position)


def _build_absolute_index_factory(name: str, position: int) -> Callable:
    """Build factory for absolute index positions (e.g., first, last token)."""

    def factory(pipeline):
        def indexer(input_sample: CausalTrace):
            ids = _load_for_indexing(pipeline, [input_sample])["input_ids"][0]
            total_tokens = len(ids)

            # Handle negative indices. Negative positions count from the end and
            # need no rebasing — `position=-1` is the generation slot regardless
            # of any chat prefix. Non-negative positions are content-relative, so
            # under a chat template we skip past the prefix tokens (BOS, role
            # markers, any system directive) — `position=0` means the first
            # *content* token, not BOS. `_chat_prefix_token_count()` is 0 when no
            # chat template is applied, so this is a no-op for bare prompts.
            if position < 0:
                actual_position = total_tokens + position
            else:
                actual_position = position + pipeline._chat_prefix_token_count()

            if actual_position < 0 or actual_position >= total_tokens:
                raise ValueError(
                    f"Position {position} out of range for sequence of length {total_tokens}"
                )

            return [actual_position]

        return TokenPosition(indexer, pipeline, id=name)

    return factory


def _build_variable_factory(name: str, spec: Dict[str, Any], template: str) -> Callable:
    """
    Build factory for variable-based positions.

    Spec format:
        {"type": "variable", "name": "x"}  # All tokens of variable x
    """
    var_name = spec.get("name")

    if not var_name:
        raise ValueError(
            f"Token position '{name}': variable type requires 'name' field"
        )

    # Parse the template to validate the variable exists
    template_obj = Template(template)
    if var_name not in template_obj.get_variable_names():
        raise ValueError(
            f"Token position '{name}': variable '{var_name}' not found in template: {template}"
        )

    def factory(pipeline):
        def indexer(input_sample):
            if var_name not in input_sample:
                raise ValueError(
                    f"Variable '{var_name}' not found in input sample: {list(input_sample.keys())}"
                )

            # Get all variable positions from the template
            variable_positions = template_obj.get_variable_positions(
                input_sample, pipeline
            )

            # Return the token indices for this variable
            if var_name not in variable_positions:
                raise ValueError(
                    f"Variable '{var_name}' was not found in tokenized output. "
                    f"This should not happen - template parsing may have failed."
                )

            return variable_positions[var_name]

        return TokenPosition(indexer, pipeline, id=name)

    return factory


def _build_scoped_index_factory(
    name: str, position: int, scope: Dict[str, Any], template: str
) -> Callable:
    """
    Build factory for index within a variable's tokens.

    Example: {"type": "index", "position": 1, "scope": {"variable": "x"}}
    Returns the 2nd token (index 1) of variable x's tokenization.
    """
    if "variable" not in scope:
        raise ValueError(f"Token position '{name}': scope must specify 'variable'")

    var_name = scope["variable"]

    # First build the variable factory to get all tokens
    var_spec = {"type": "variable", "name": var_name}
    var_factory = _build_variable_factory(f"{name}_base", var_spec, template)

    def factory(pipeline):
        # Get the base variable position
        var_token_pos = var_factory(pipeline)

        def indexer(input_sample):
            # Get all tokens for the variable
            var_tokens = var_token_pos.index(input_sample)

            # Index into those tokens
            if position < 0:
                actual_position = len(var_tokens) + position
            else:
                actual_position = position

            if actual_position < 0 or actual_position >= len(var_tokens):
                raise ValueError(
                    f"Position {position} out of range for variable '{var_name}' "
                    f"with {len(var_tokens)} tokens"
                )

            return [var_tokens[actual_position]]

        return TokenPosition(indexer, pipeline, id=name)

    return factory


def _build_relative_index_factory(
    name: str, offset: int, relative_to: Dict[str, Any], template: str
) -> Callable:
    """
    Build factory for positions relative to a variable.

    Example: {"type": "index", "position": +1, "relative_to": {"variable": "x"}}
    Returns the token immediately after variable x.
    """
    if "variable" not in relative_to:
        raise ValueError(
            f"Token position '{name}': relative_to must specify 'variable'"
        )

    var_name = relative_to["variable"]

    # Build the variable factory to find the reference point
    var_spec = {"type": "variable", "name": var_name}
    var_factory = _build_variable_factory(f"{name}_ref", var_spec, template)

    def factory(pipeline):
        var_token_pos = var_factory(pipeline)

        def indexer(input_sample):
            # Get the variable's tokens
            var_tokens = var_token_pos.index(input_sample)

            # Compute relative position
            if offset >= 0:
                # Offset from end of variable
                reference_pos = var_tokens[-1]
                target_pos = reference_pos + offset
            else:
                # Offset from start of variable (negative offset)
                reference_pos = var_tokens[0]
                target_pos = reference_pos + offset

            # Validate it's in bounds
            ids = _load_for_indexing(pipeline, [input_sample])["input_ids"][0]
            total_tokens = len(ids)

            if target_pos < 0 or target_pos >= total_tokens:
                raise ValueError(
                    f"Relative position {offset} from variable '{var_name}' "
                    f"results in index {target_pos}, out of range [0, {total_tokens})"
                )

            return [target_pos]

        return TokenPosition(indexer, pipeline, id=name)

    return factory


# --------------------------------------------------------------------------- #
#  Token Position Combinators                                                  #
# --------------------------------------------------------------------------- #


def paired_token_position(
    original_position: TokenPosition,
    counterfactual_position: TokenPosition,
    id: str = "paired",
) -> TokenPosition:
    """
    Create a TokenPosition that uses different positions for original vs counterfactual.

    This is useful for interchange interventions where you want to patch activations
    from one token position in the counterfactual to a different token position in
    the original input.

    Args:
        original_position: TokenPosition to use for original inputs (is_original=True)
        counterfactual_position: TokenPosition to use for counterfactual inputs (is_original=False)
        id: Identifier for the new TokenPosition

    Returns:
        A new TokenPosition that delegates based on is_original flag

    Example:
        >>> # Intervene at last_token in original, using activation from answer_var in counterfactual
        >>> last_token = TokenPosition(lambda x: [-1], pipeline, id="last")
        >>> answer_var = TokenPosition(lambda x: [5], pipeline, id="answer")
        >>> paired = paired_token_position(last_token, answer_var, id="last<-answer")
    """

    def indexer(
        input_sample: CausalTrace, is_original: bool = True
    ) -> List[int] | List[List[int]]:
        if is_original:
            return original_position.index(input_sample)
        else:
            return counterfactual_position.index(input_sample)

    # Use pipeline from original (they should be the same)
    return TokenPosition(indexer, original_position.pipeline, id=id)


def combined_token_position(
    token_positions: List[TokenPosition],
    id: str = "combined",
) -> TokenPosition:
    """
    Combine multiple TokenPositions into a single TokenPosition.

    Returns the concatenation of all token indices from each position.

    Args:
        token_positions: List of TokenPosition objects to combine
        id: Identifier for the new TokenPosition

    Returns:
        A new TokenPosition that returns all indices from all input positions

    Example:
        >>> pos1 = TokenPosition(lambda x: [0, 1], pipeline, id="first_two")
        >>> pos2 = TokenPosition(lambda x: [5], pipeline, id="middle")
        >>> combined = combined_token_position([pos1, pos2], id="first_two_and_middle")
        >>> # combined.index(...) returns [0, 1, 5]
    """
    if not token_positions:
        raise ValueError("token_positions list cannot be empty")

    def indexer(input_sample: CausalTrace, is_original: bool = True) -> List[int]:
        all_indices: List[int] = []
        for tp in token_positions:
            indices = cast(List[int], tp.index(input_sample, is_original=is_original))
            all_indices.extend(indices)
        return all_indices

    return TokenPosition(indexer, token_positions[0].pipeline, id=id)
