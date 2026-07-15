"""Shared fixtures for ``tests/tasks/entity_binding/``.

Provides a session-scoped :class:`LMPipeline` wrapping the tiny random Llama
stub from :mod:`tests._helpers.tiny`. The entity_binding ``token_positions``
module reaches into a real tokenizer, so pure-symbolic sampling is not
enough — these tests need a real (but tiny) HF pipeline. The pilot pattern
(``tests/neural/conftest.py`` in the main-branch lineage) does the same;
re-implementing it locally keeps the import surface here scoped to the one
test file that needs it.
"""

from __future__ import annotations

import pytest

from causalab.neural.pipeline import LMPipeline
from tests._helpers.tiny import tiny_random_model


@pytest.fixture(scope="session")
def tiny_pipeline() -> LMPipeline:
    """Session-scoped :class:`LMPipeline` backed by the tiny Llama stub.

    Reuses :func:`tests._helpers.tiny.tiny_random_model`'s ``lru_cache``d model
    instance so this fixture's only cost is the
    ``AutoTokenizer.from_pretrained`` call (sub-second, HF-cached).
    """
    return LMPipeline(tiny_random_model(), max_new_tokens=1, padding_side="left")
