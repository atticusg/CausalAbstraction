"""Local conftest for the IOI property-tier tests.

The token-positions class is tokenizer-coupled: every invariant decodes
indices through a real :class:`~causalab.neural.pipeline.LMPipeline`.
Building the pipeline once per session (rather than per test) keeps the
hypothesis sweep sub-second even at ``max_examples=20``.

Why ``gpt2`` instead of the tiny Llama stub from
``tests/_helpers/tiny.py``? IOI's token-position logic locates names in
the prompt by substring-matching the tokenizer's output. The tiny stub's
tokenizer (Llama BPE with vocab_size=32000 and random init) is fine for
runner smoke tests but its tokenization of name strings differs subtly
enough from a real production tokenizer that the decoding round-trip
invariants become noisy. ``gpt2`` is the same tokenizer the IOI runner
baseline uses, so the invariants pinned here line up with production.
"""

from __future__ import annotations

import pytest

from causalab.neural.pipeline import LMPipeline, resolve_device


@pytest.fixture(scope="session")
def gpt2_pipeline() -> LMPipeline:
    """Session-scoped ``gpt2`` :class:`LMPipeline` for token-position tests.

    Loaded once per pytest session; subsequent fixture injections reuse
    the same instance. ``max_new_tokens=1`` keeps the pipeline cheap — the
    property tests never call ``generate``, just ``tokenizer.encode``/
    ``decode``. ``max_length`` is left at its ``None`` default: token-position
    resolution requires the unpadded tokenization frame, and a fixed
    ``max_length`` is rejected by ``_load_for_indexing``.
    """
    device = resolve_device()
    return LMPipeline("gpt2", max_new_tokens=1, device=device)
