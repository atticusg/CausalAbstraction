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
from typing import Any, Callable, Dict, List, Union

from causalab.neural.model_units import ComponentIndexer
from causalab.neural.pipeline import LMPipeline


# --------------------------------------------------------------------------- #
#  Token Position Utilities                                                   #
# --------------------------------------------------------------------------- #


class TokenPosition(ComponentIndexer):
    """Dynamic indexer: returns position(s) of interest for a prompt.

    Attributes
    ----------
    pipeline :
        The :class:`neural.pipeline.LMPipeline` supplying the tokenizer.
    is_original : bool
        Whether this indexer is for original inputs (True) or counterfactual inputs (False).
        Default is True for backward compatibility.
    """

    def __init__(
        self, indexer, pipeline: LMPipeline, is_original: bool = True, **kwargs
    ):
        super().__init__(indexer, **kwargs)
        self.pipeline = pipeline
        self.is_original = is_original

    def highlight_selected_token(self, input: dict) -> str:
        """Return *prompt* with selected token(s) wrapped in ``**bold**``.

        The method tokenizes *prompt*, calls self.index to obtain the
        positions, then re-assembles a detokenised string with the
        selected token(s) wrapped in ``**bold**``.  The rest of the
        prompt is unchanged.

        Note that whitespace handling may be approximate for tokenizers
        that encode leading spaces as special glyphs (e.g. ``Ġ``).
        """
        ids = self.pipeline.load(input)["input_ids"][0]
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
def get_last_token_index(input: dict, pipeline: LMPipeline) -> List[int]:
    """Return a one-element list containing the *last* token index."""
    ids = list(pipeline.load(input)["input_ids"][0])
    return [len(ids) - 1]


def get_all_tokens(
    input: dict[str, Any], pipeline: LMPipeline, padding: bool = False
) -> TokenPosition:
    """Return a single TokenPosition object containing all (non-pad) token indices."""
    pad_token_id = pipeline.tokenizer.pad_token_id

    # Create indexer function that returns all non-pad token indices
    def all_tokens_indexer(inp):
        token_ids = pipeline.load(inp)["input_ids"][0]
        if padding:
            return [i for i in range(len(token_ids))]
        return [i for i in range(len(token_ids)) if token_ids[i] != pad_token_id]

    return TokenPosition(indexer=all_tokens_indexer, pipeline=pipeline, id="all_tokens")


def get_list_of_each_token(
    input: dict[str, Any], pipeline: LMPipeline
) -> List[TokenPosition]:
    """Return a list of TokenPosition objects, each containing a single token index."""
    ids = list(pipeline.load(input)["input_ids"][0])
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


def get_tokens_in_char_range(offsets, start_char: int, end_char: int) -> List[int]:
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
    tokens = []
    for token_idx, (token_start, token_end) in enumerate(offsets):
        # Skip padding tokens (they have offset (0, 0))
        if token_start == 0 and token_end == 0:
            continue

        # Check if this token overlaps with the character range
        if token_start < end_char and token_end > start_char:
            tokens.append(token_idx)

    return tokens


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

    # Use pipeline.load() with offset_mapping to get character→token mapping
    # This ensures we use the exact same tokenization as interventions
    input_dict = {"raw_input": text}
    tokenized = pipeline.load(
        input_dict, add_special_tokens=add_special_tokens, return_offsets_mapping=True
    )
    offsets = tokenized["offset_mapping"][0]  # Get first sequence from batch

    # Find which tokens overlap with the substring's character range
    substring_start = occurrences[occurrence]
    substring_end = substring_start + len(substring)

    return get_tokens_in_char_range(offsets, substring_start, substring_end)


# --------------------------------------------------------------------------- #
#  Template System                                                            #
# --------------------------------------------------------------------------- #


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

    def get_variable_positions(
        self, values: Dict[str, Any], pipeline
    ) -> Dict[str, List[int]]:
        """
        Fill the template and track which tokens correspond to each variable.

        This is the key method: we tokenize the template piece by piece,
        tracking exactly where each variable's tokens appear.

        Args:
            values: Dictionary mapping variable names to their values
            pipeline: The tokenization pipeline

        Returns:
            Dictionary mapping variable names to lists of token indices
        """
        # Build the full text while tracking character positions
        char_positions = {}  # var_name -> [(start_char, end_char), ...]
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

                # Track this variable's character positions
                # Support multiple occurrences - store as list of ranges
                if content not in char_positions:
                    char_positions[content] = []
                char_positions[content].append((start_char, end_char))

                full_text.append(value_str)
                current_pos = end_char

        full_text_str = "".join(full_text)

        # Use pipeline.load() with offset_mapping to get character→token mapping
        # This ensures we use the exact same tokenization as interventions
        input_dict = {"raw_input": full_text_str}
        tokenized = pipeline.load(input_dict, return_offsets_mapping=True)
        offsets = tokenized["offset_mapping"][0]  # Get first sequence from batch

        # Map character positions to token indices using offsets
        variable_tokens = {}
        for var_name, char_ranges in char_positions.items():
            variable_tokens[var_name] = []

            for start_char, end_char in char_ranges:
                # Find which tokens overlap with this character range
                tokens = get_tokens_in_char_range(offsets, start_char, end_char)
                variable_tokens[var_name].extend(tokens)

        return variable_tokens

    def get_variable_names(self) -> List[str]:
        """Return list of all variable names in the template."""
        return list(
            set(content for part_type, content in self.parts if part_type == "variable")
        )


def build_token_position_factories(
    specs: Dict[str, Union[Dict[str, Any], Callable]], template: str
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
            temp_token_pos = temp_factory(pipeline)

            # Call its indexer to get the actual token indices
            return temp_token_pos.index(input_sample)

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
        def indexer(input_sample):
            ids = pipeline.load(input_sample)["input_ids"][0]
            total_tokens = len(ids)

            # Handle negative indices
            if position < 0:
                actual_position = total_tokens + position
            else:
                actual_position = position

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
            ids = pipeline.load(input_sample)["input_ids"][0]
            total_tokens = len(ids)

            if target_pos < 0 or target_pos >= total_tokens:
                raise ValueError(
                    f"Relative position {offset} from variable '{var_name}' "
                    f"results in index {target_pos}, out of range [0, {total_tokens})"
                )

            return [target_pos]

        return TokenPosition(indexer, pipeline, id=name)

    return factory
