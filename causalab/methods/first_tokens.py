"""Canonical first-token id resolution, space-prefix-safe.

Turning an answer *word* into the vocab id the model actually places probability
on is a persistent sharp edge: at a word boundary a subword tokenizer emits the
**leading-space** form (``" 4"`` / ``" blue"``), a *different* id from the bare
form (``"4"`` / ``"blue"``) — the digit/answer trailing-space gotcha behind the
``output_tokens`` scoring lineage (#169). :func:`get_first_tokens` returns the
first token id of *both* the bare and the leading-space form of each word
(deduplicated, order-stable), so a caller tracking answer mass captures whichever
form the model emits.

The logic is **borrowed from** ``nnterp.prompt_utils.get_first_tokens`` (nnterp
1.3.0), adapted to take a plain tokenizer so it stays analysis-neutral. Its
robustness trick: some tokenizers can't be built with ``add_prefix_space=False``,
so the bare form silently gets a leading space and collides with the space form;
in that case it falls back to tokenizing ``"🍐" + word`` and dropping the pear's
tokens, which is guaranteed to recover the true first token of ``word`` or raise.

This is a resolution primitive only. causalab's ``output_tokens`` declaration and
the ``derive_checker`` string-match authority
(``causalab.causal.causal_model``) remain the scoring authority — this module
does not decide *which forms count as a match*, only *which vocab id* a form maps
to.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TokenizationError(Exception):
    """The ``"🍐word"`` fallback could not recover a word's first token id."""


def get_first_tokens(
    words: str | list[str],
    tokenizer,
    *,
    use_hacky_implementation: bool = False,
) -> list[int]:
    """First token id of ``"word"`` and ``" word"`` for each word (deduplicated).

    Borrowed from ``nnterp.prompt_utils.get_first_tokens``. For every word,
    collects the first token id of the bare form and of the leading-space form —
    the two ids a next-token grader must consider, since a word-boundary emission
    uses the space-prefixed id. The returned list is order-stable and
    de-duplicated across all words.

    Args:
        words: A word or list of words.
        tokenizer: A HuggingFace tokenizer. For byte-level-BPE tokenizers built
            with ``add_prefix_space=False`` (e.g. GPT-2 family) the bare and
            space forms resolve directly; otherwise the ``"🍐word"`` fallback is
            used automatically.
        use_hacky_implementation: Force the ``"🍐word"`` fallback (tokenize
            ``"🍐" + word`` and drop the pear's tokens). Guaranteed to recover
            the true first token of ``word`` or raise :class:`TokenizationError`.

    Returns:
        The de-duplicated, order-stable list of first token ids.
    """
    if isinstance(words, str):
        words = [words]

    final_tokens: list[int] = []
    for word in words:
        if use_hacky_implementation:
            pear = tokenizer("🍐", add_special_tokens=False).input_ids
            length = len(pear)
            tokens = tokenizer("🍐" + word, add_special_tokens=False).input_ids
            if tokens[:length] != pear:
                raise TokenizationError(
                    "The '🍐' prefix did not tokenize as a stable prefix of "
                    f"'🍐{word}'; cannot recover the first token of {word!r}."
                )
            if len(tokens) > length:
                final_tokens.append(tokens[length])
            continue

        # Fast path: assumes the tokenizer was built with add_prefix_space=False,
        # so the bare form carries no implicit leading space.
        token = tokenizer(word, add_special_tokens=False).input_ids[0]
        token_with_start_of_word = tokenizer(
            " " + word, add_special_tokens=False
        ).input_ids[0]
        if token == token_with_start_of_word:
            # The bare form silently got a leading space (add_prefix_space=True):
            # fall back to the pear trick for every word to recover true ids.
            try:
                recovered = get_first_tokens(
                    words, tokenizer, use_hacky_implementation=True
                )
                logger.warning(
                    "Tokenizer was not initialized with add_prefix_space=False; "
                    "used the '🍐word' fallback to resolve first tokens."
                )
                return recovered
            except TokenizationError:
                raise TokenizationError(
                    "Tokenizer was not initialized with add_prefix_space=False "
                    "and the '🍐word' fallback failed. Pass a tokenizer built "
                    "with add_prefix_space=False."
                )

        final_tokens.append(token)
        space_ids = tokenizer(" ", add_special_tokens=False).input_ids
        space_token = space_ids[0] if space_ids else None
        # Only record the space form when its first id is a genuine word token
        # (not a lone-space glyph that a bare " " would also produce).
        if token_with_start_of_word != space_token:
            final_tokens.append(token_with_start_of_word)

    # dict.fromkeys de-duplicates while preserving first-seen order.
    return list(dict.fromkeys(final_tokens))
