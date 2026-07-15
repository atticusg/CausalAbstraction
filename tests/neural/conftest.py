"""Shared fixtures for ``tests/neural/``.

Provides a session-scoped :class:`LMPipeline` wrapping the tiny random Llama
stub from :mod:`tests._helpers.tiny`. Real tokenization on a real (tiny)
model is the canonical alternative to the mock tokenizers that docs/TESTS.md's
mocking-policy appendix calls out as failure-prone.
"""

from __future__ import annotations

import pytest

from causalab.neural.pipeline import LMPipeline
from tests._helpers.tiny import tiny_random_gpt2_model, tiny_random_model


@pytest.fixture(scope="session")
def tiny_pipeline() -> LMPipeline:
    """Session-scoped :class:`LMPipeline` backed by the tiny-random Llama stub.

    Reuses :func:`tests._helpers.tiny.tiny_random_model`'s cached model
    instance, so the pipeline construction cost is dominated by the
    ``AutoTokenizer.from_pretrained`` call (sub-second, also HF-cached).
    """
    return LMPipeline(tiny_random_model(), max_new_tokens=1, padding_side="left")


@pytest.fixture(scope="session")
def tiny_gpt2_pipeline() -> LMPipeline:
    """Session-scoped left-padded :class:`LMPipeline` over the random-weight GPT-2
    stub — an **absolute-position** (``wpe``) model.

    The counterpart to :func:`tiny_pipeline`: identical role, but GPT-2's learned
    position embeddings make collected activations sensitive to the left-pad
    ``position_ids`` convention, so this fixture can surface position_ids-under-
    padding bugs numerically where the RoPE ``tiny_pipeline`` (relative positions)
    cannot.
    """
    return LMPipeline(tiny_random_gpt2_model(), max_new_tokens=1, padding_side="left")
